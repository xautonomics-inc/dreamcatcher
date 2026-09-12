"""Measured host inventory.

SPEC-016 requirement 7: the scheduler's world model must be *measured*, never
declared.  Every stale map on this fleet has bitten us (ring-autostart running
retired recipes, fleet-console holding a v6 ring map, agent-console's endpoint
registry going demo-stale).  So the inventory here is read from ``/sys`` and
from the vendor SMI tools at daemon start, stamped with the time it was taken,
and re-read on demand -- there is no hand-written table.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_SYS_CPU = Path("/sys/devices/system/cpu")


@dataclass(frozen=True, slots=True)
class CpuUnit:
    """One logical CPU, with the physical core, NUMA node and core CLASS.

    The ``core`` field is load-bearing.  Two SMT siblings share one physical
    core's execution resources, so handing sibling 0 to tenant A and sibling 1
    to tenant B is *not* a partition -- it is the collision we are trying to
    prevent, wearing a partition's clothes.  The solver allocates whole cores.

    ``kind`` is load-bearing on hybrid parts.  The Core Ultra 7 265K has 8
    P-cores (capacity 1012) and 12 E-cores (756), and its ``core_id`` values
    interleave the two arbitrarily -- 0, 12, 24, 32, 36, 39 -- so dealing cores
    in id order hands out a silently unequal mix.  "Four cores each" is not a
    fair partition when one tenant's four are 34% faster.
    """

    cpu: int
    core: int
    node: int
    kind: str = ""  # "P" | "E" | "" (uniform / unknown)


@dataclass(frozen=True, slots=True)
class GpuUnit:
    """One GPU compute domain, keyed by a stable identity, never an index.

    FLEET STANDARD (fleet-gpu-identity-pinning): never address a GPU by index.
    ``key`` is the vendor UUID where available, else the PCI bus id.  ``alias``
    carries the ``/dev/dri/by-gpu/`` symlink name on AMD/Intel hosts.
    """

    key: str
    vendor: str
    name: str
    alias: str | None
    vram_total_bytes: int
    vram_used_bytes: int


@dataclass(slots=True)
class HostInventory:
    """A point-in-time reading of what this host physically has."""

    host: str
    taken_at: float
    cpus: tuple[CpuUnit, ...]
    gpus: tuple[GpuUnit, ...]
    mem_total_bytes: int
    mem_available_bytes: int
    warnings: list[str] = field(default_factory=list)

    @property
    def age_s(self) -> float:
        return time.time() - self.taken_at

    def cores(self) -> dict[tuple[int, int], tuple[int, ...]]:
        """Map ``(node, core)`` -> the logical CPUs on that physical core."""
        out: dict[tuple[int, int], list[int]] = {}
        for u in self.cpus:
            out.setdefault((u.node, u.core), []).append(u.cpu)
        return {k: tuple(sorted(v)) for k, v in sorted(out.items())}

    def core_kinds(self) -> dict[tuple[int, int], str]:
        """Map ``(node, core)`` -> core class.  Empty string when uniform."""
        out: dict[tuple[int, int], str] = {}
        for u in self.cpus:
            out.setdefault((u.node, u.core), u.kind)
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "host": self.host,
            "taken_at": self.taken_at,
            "age_s": round(self.age_s, 3),
            "cpus": [
                {"cpu": u.cpu, "core": u.core, "node": u.node, "kind": u.kind} for u in self.cpus
            ],
            "n_logical": len(self.cpus),
            "n_cores": len(self.cores()),
            "cores_by_kind": _tally(self.core_kinds().values()),
            "gpus": [
                {
                    "key": g.key,
                    "vendor": g.vendor,
                    "name": g.name,
                    "alias": g.alias,
                    "vram_total_bytes": g.vram_total_bytes,
                    "vram_used_bytes": g.vram_used_bytes,
                }
                for g in self.gpus
            ],
            "mem_total_bytes": self.mem_total_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "warnings": self.warnings,
        }


def _read_int(path: Path, default: int = 0) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def _hybrid_map(sys_dev: Path = Path("/sys/devices")) -> dict[int, str]:
    """Read Intel hybrid core classes from ``cpu_core``/``cpu_atom``.

    Falls back to ``cpu_capacity`` (the scheduler's own view) when those are
    absent, so the classification comes from the kernel either way rather than
    from a table of model names.
    """
    out: dict[int, str] = {}
    for name, kind in (("cpu_core", "P"), ("cpu_atom", "E")):
        f = sys_dev / name / "cpus"
        try:
            spec = f.read_text().strip()
        except OSError:
            continue
        for chunk in spec.split(","):
            if not chunk:
                continue
            if "-" in chunk:
                lo, hi = chunk.split("-", 1)
                for c in range(int(lo), int(hi) + 1):
                    out[c] = kind
            else:
                out[int(chunk)] = kind
    return out


def _capacity_map(sys_cpu: Path) -> dict[int, str]:
    caps: dict[int, int] = {}
    for entry in sys_cpu.glob("cpu[0-9]*"):
        m = re.fullmatch(r"cpu(\d+)", entry.name)
        if m is None:
            continue
        c = _read_int(entry / "cpu_capacity", 0)
        if c:
            caps[int(m.group(1))] = c
    if len(set(caps.values())) < 2:
        return {}
    top = max(caps.values())
    return {cpu: ("P" if cap == top else "E") for cpu, cap in caps.items()}


def read_cpus(sys_cpu: Path = _SYS_CPU) -> tuple[CpuUnit, ...]:
    """Enumerate online logical CPUs with their core/node/class topology."""
    kinds = _hybrid_map()
    if not kinds:
        kinds = _capacity_map(sys_cpu)
    units: list[CpuUnit] = []
    for entry in sorted(sys_cpu.glob("cpu[0-9]*")):
        m = re.fullmatch(r"cpu(\d+)", entry.name)
        if m is None:
            continue
        cpu = int(m.group(1))
        online = entry / "online"
        if online.exists() and _read_int(online, 1) == 0:
            continue
        core = _read_int(entry / "topology" / "core_id", cpu)
        node = 0
        for nd in entry.glob("node[0-9]*"):
            nm = re.fullmatch(r"node(\d+)", nd.name)
            if nm is not None:
                node = int(nm.group(1))
                break
        units.append(CpuUnit(cpu=cpu, core=core, node=node, kind=kinds.get(cpu, "")))
    return tuple(sorted(units, key=lambda u: u.cpu))


def _mem() -> tuple[int, int]:
    total = avail = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    return total, avail


def _by_gpu_aliases() -> dict[str, str]:
    """``renderD*`` -> ``/dev/dri/by-gpu/`` symlink name, resolved with realpath.

    TRAP (qwen38-vulkan-solo-a770): passing a by-gpu symlink straight to a
    runtime that wants a real device path yields zero devices.  Always resolve.
    """
    out: dict[str, str] = {}
    d = Path("/dev/dri/by-gpu")
    if not d.is_dir():
        return out
    for link in d.iterdir():
        try:
            out[os.path.realpath(link)] = link.name
        except OSError:
            continue
    return out


def _nvidia_gpus(warnings: list[str]) -> list[GpuUnit]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        raw = subprocess.run(
            [
                exe,
                "--query-gpu=uuid,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        warnings.append(f"nvidia-smi failed: {exc}")
        return []
    gpus: list[GpuUnit] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        uuid, name, total_mib, used_mib = parts
        gpus.append(
            GpuUnit(
                key=uuid,
                vendor="nvidia",
                name=name,
                alias=None,
                vram_total_bytes=int(float(total_mib)) * 1024 * 1024,
                vram_used_bytes=int(float(used_mib)) * 1024 * 1024,
            )
        )
    return gpus


def _amd_gpus(warnings: list[str]) -> list[GpuUnit]:
    exe = shutil.which("rocm-smi")
    if exe is None:
        return []
    try:
        raw = subprocess.run(
            [exe, "--showuniqueid", "--showmeminfo", "vram", "--showproductname", "--json"],
            capture_output=True,
            text=True,
            timeout=25,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        warnings.append(f"rocm-smi failed: {exc}")
        return []
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        warnings.append(f"rocm-smi json parse failed: {exc}")
        return []
    aliases = _by_gpu_aliases()
    gpus: list[GpuUnit] = []
    for card, fields in sorted(doc.items()):
        if not isinstance(fields, dict):
            continue
        uniq = str(fields.get("Unique ID") or fields.get("GUID") or card)
        name = str(fields.get("Card Series") or fields.get("Card model") or "amd-gpu")
        total = int(fields.get("VRAM Total Memory (B)", 0) or 0)
        used = int(fields.get("VRAM Total Used Memory (B)", 0) or 0)
        idx_m = re.fullmatch(r"card(\d+)", card)
        alias = None
        if idx_m is not None:
            alias = aliases.get(f"/dev/dri/renderD{128 + int(idx_m.group(1))}")
        gpus.append(
            GpuUnit(
                key=f"amd:{uniq}" if uniq != card else f"amd:{card}",
                vendor="amd",
                name=name,
                alias=alias,
                vram_total_bytes=total,
                vram_used_bytes=used,
            )
        )
    return gpus


def read_inventory(host: str | None = None, *, probe_gpus: bool = True) -> HostInventory:
    """Take a fresh reading of this host.  Never cached, always stamped."""
    warnings: list[str] = []
    gpus: list[GpuUnit] = []
    if probe_gpus:
        gpus = _nvidia_gpus(warnings) + _amd_gpus(warnings)
    total, avail = _mem()
    return HostInventory(
        host=host or os.uname().nodename,
        taken_at=time.time(),
        cpus=read_cpus(),
        gpus=tuple(gpus),
        mem_total_bytes=total,
        mem_available_bytes=avail,
        warnings=warnings,
    )


def _tally(vals: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in vals:
        out[v or "uniform"] = out.get(v or "uniform", 0) + 1
    return out
