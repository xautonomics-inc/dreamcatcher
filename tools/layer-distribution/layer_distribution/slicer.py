"""GGUF model layer slicer.

Slices monolithic or multi-shard GGUF models into per-layer libraries
with explicit whole-file hash scope and per-tensor digests.
"""

from __future__ import annotations

import glob
import hashlib
import inspect
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFEndian, GGUFReader, GGUFValueType, GGUFWriter


def supports_tensor_endianess(writer_target: Any = GGUFWriter) -> bool:
    """Check if GGUFWriter.write_tensor_data accepts 'tensor_endianess'."""
    func = getattr(writer_target, "write_tensor_data", None)
    if func is None:
        func = getattr(GGUFWriter, "write_tensor_data", None)
    if func is None:
        return False
    try:
        sig = inspect.signature(func)
        return "tensor_endianess" in sig.parameters
    except (ValueError, TypeError):
        return False


def tensor_bytes(t: Any) -> memoryview:
    """Extract contiguous byte memoryview from a GGUF tensor object."""
    d = t.data
    if not isinstance(d, np.ndarray):
        d = np.asarray(d)
    if not d.flags["C_CONTIGUOUS"]:
        d = np.ascontiguousarray(d)
    return memoryview(d)


class RateLimiter:
    """Monotonic rate limiter in MiB/s."""

    def __init__(self, rate_mb_s: float | None = None) -> None:
        if rate_mb_s is not None and (rate_mb_s <= 0 or not math.isfinite(rate_mb_s)):
            raise ValueError(f"Rate must be positive and finite: {rate_mb_s}")
        self.rate_mb_s = rate_mb_s
        self.total_bytes = 0
        self.started = time.monotonic()

    def update(self, n_bytes: int) -> None:
        """Record written or read bytes and sleep if exceeding pacing target."""
        if self.rate_mb_s is None or self.rate_mb_s <= 0:
            return
        self.total_bytes += n_bytes
        elapsed = time.monotonic() - self.started
        target_time = self.total_bytes / (self.rate_mb_s * 1024 * 1024)
        delay = target_time - elapsed
        if delay > 0:
            time.sleep(delay)


def compute_file_hash(path: Path | str, rate_limiter: RateLimiter | None = None) -> tuple[int, str]:
    """Compute whole-file byte length and blake2b-128 hex digest with optional pacing."""
    h = hashlib.blake2b(digest_size=16)
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
            if rate_limiter:
                rate_limiter.update(len(chunk))
    return size, h.hexdigest()


def sanitize_source_glob(raw_glob: str, label: str | None = None) -> str:
    """Ensure manifest provenance does not leak absolute private host paths."""
    if label:
        return label
    p = Path(raw_glob)
    if p.is_absolute():
        return p.name
    return raw_glob


def can_reuse_file(
    path: Path,
    expected_tensors: list[tuple[str, Any]],
    no_hash: bool,
    rate_limiter: RateLimiter | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Check if an existing file matches expected tensors and can be reused."""
    if not path.is_file() or path.stat().st_size == 0:
        return False, None
    try:
        reader = GGUFReader(path, "r")
        r_tensors = {t.name: t for t in reader.tensors}
        if len(r_tensors) != len(expected_tensors):
            return False, None
        for nn, t in expected_tensors:
            if nn not in r_tensors:
                return False, None
            rt = r_tensors[nn]
            if list(rt.shape) != list(t.shape) or rt.tensor_type != t.tensor_type:
                return False, None

        trecs: list[dict[str, Any]] = []
        for nn, t in expected_tensors:
            rt = r_tensors[nn]
            rec: dict[str, Any] = {
                "name": nn,
                "source_name": t.name,
                "shape": [int(x) for x in t.shape],
                "type": t.tensor_type.name,
                "n_bytes": int(t.n_bytes),
            }
            if not no_hash:
                th = hashlib.blake2b(tensor_bytes(rt), digest_size=16).hexdigest()
                rec["hash"] = th
            trecs.append(rec)

        n_bytes_file = path.stat().st_size
        fhash: str | None = None
        if not no_hash:
            n_bytes_file, fhash = compute_file_hash(path, rate_limiter)

        return True, {
            "n_bytes_file": n_bytes_file,
            "hash": fhash,
            "tensors": trecs,
        }
    except Exception:
        return False, None


@dataclass(frozen=True)
class SliceStats:
    """Summary statistics for a slice operation."""

    files_written: int
    files_reused: int
    total_bytes_data: int
    total_bytes_files: int
    manifest_path: Path


def slice_model(
    model_glob: str,
    out_dir: str | Path,
    *,
    rate: float | None = None,
    force: bool = False,
    resume: bool = False,
    no_hash: bool = False,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Slice a GGUF model into a per-layer layer library with whole-file hash scope.

    Parameters
    ----------
    model_glob:
        Path or glob pattern matching source GGUF file(s).
    out_dir:
        Destination directory for sliced layer library.
    rate:
        Optional I/O rate limit in MiB/s.
    force:
        If True, overwrite existing files in out_dir without resuming.
    resume:
        If True, reuse complete valid existing files and continue partial slice.
    no_hash:
        If True, skip hash computation in manifest.
    source_label:
        Optional sanitized label for source glob in manifest.

    Returns
    -------
    dict[str, Any]:
        The serialized manifest dictionary.
    """
    shards = sorted(glob.glob(model_glob))
    if not shards:
        single_path = Path(model_glob)
        if single_path.is_file():
            shards = [str(single_path)]
        else:
            raise FileNotFoundError(f"No files match {model_glob}")

    out_path = Path(out_dir)
    if out_path.is_dir() and any(out_path.iterdir()) and not force and not resume:
        raise FileExistsError(f"error: {out_dir} exists and is not empty (use --force or --resume)")
    out_path.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter(rate)

    meta = GGUFReader(shards[0], "r")
    readers = [meta] + [GGUFReader(s, "r") for s in shards[1:]]

    has_tensor_endianess = supports_tensor_endianess()
    is_big_endian = getattr(meta, "endianess", None) == GGUFEndian.BIG
    if is_big_endian and not has_tensor_endianess:
        raise ValueError(
            f"Big-endian GGUF source ({shards[0]}) cannot be sliced: "
            "installed gguf-py GGUFWriter.write_tensor_data does not support tensor_endianess"
        )

    arch_field = meta.get_field("general.architecture")
    if not arch_field:
        raise ValueError(f"Missing general.architecture in GGUF metadata: {shards[0]}")
    arch = str(arch_field.contents())

    def srcget(k: str, default: Any = None) -> Any:
        f = meta.get_field(f"{arch}.{k}")
        return f.contents() if f else default

    src_blocks = srcget("block_count")
    if src_blocks is None:
        raise ValueError(f"Missing {arch}.block_count in GGUF metadata")
    src_blocks = int(src_blocks)

    src_nextn = int(srcget("nextn_predict_layers", 0) or 0)
    src_leading = int(srcget("leading_dense_block_count", 0) or 0)
    nextn_lo = src_blocks - src_nextn

    name_field = meta.get_field("general.name")
    model_name = str(name_field.contents()) if name_field else Path(shards[0]).name

    # Bucket tensors: ("layer", i) / ("nextn", i) / ("embd", None) /
    # ("output", None) / ("other", None)
    buckets: dict[tuple[str, int | None], list[tuple[str, Any]]] = {}
    for r in readers:
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.(.*)", t.name)
            if m:
                i = int(m.group(1))
                if i >= src_blocks:
                    raise ValueError(f"tensor {t.name} beyond block_count={src_blocks}")
                key: tuple[str, int | None] = ("nextn", i) if i >= nextn_lo else ("layer", i)
                new_name = f"blk.0.{m.group(2)}"
            elif t.name == "token_embd.weight":
                key, new_name = ("embd", None), t.name
            elif t.name in ("output_norm.weight", "output.weight"):
                key, new_name = ("output", None), t.name
            else:
                key, new_name = ("other", None), t.name
            buckets.setdefault(key, []).append((new_name, t))

    for i in range(src_blocks):
        kind = "nextn" if i >= nextn_lo else "layer"
        if (kind, i) not in buckets:
            raise ValueError(f"source has no tensors for blk.{i}")

    def write_file(
        fname: str,
        kind: str,
        abs_index: int | None,
        tensors: list[tuple[str, Any]],
        n_blk: int,
        n_dense: int,
        n_nextn: int,
    ) -> dict[str, Any]:
        target_path = out_path / fname
        if resume:
            reusable, res = can_reuse_file(target_path, tensors, no_hash, rate_limiter)
            if reusable and res is not None:
                entry: dict[str, Any] = {
                    "file": fname,
                    "kind": kind,
                    "abs_index": abs_index,
                    "n_tensors": len(tensors),
                    "n_bytes_data": int(sum(t.n_bytes for _, t in tensors)),
                    "n_bytes_file": res["n_bytes_file"],
                    "window_kv": {
                        "block_count": n_blk,
                        "leading_dense_block_count": n_dense,
                        "nextn_predict_layers": n_nextn,
                    },
                    "tensors": res["tensors"],
                }
                if not no_hash and res["hash"] is not None:
                    entry["hash"] = res["hash"]
                return entry

        if not force and not resume and target_path.exists():
            raise FileExistsError(f"Target file {target_path} exists (use --force or --resume)")

        tmp_path = out_path / f".{fname}.tmp.{os.getpid()}"
        try:
            overrides = {
                f"{arch}.block_count": (n_blk, GGUFValueType.UINT32),
                f"{arch}.leading_dense_block_count": (n_dense, GGUFValueType.UINT32),
                f"{arch}.nextn_predict_layers": (n_nextn, GGUFValueType.UINT32),
            }
            w = GGUFWriter(tmp_path, arch, endianess=meta.endianess)
            applied: set[str] = set()
            for f in meta.fields.values():
                if (
                    f.name == "general.architecture"
                    or f.name.startswith("GGUF.")
                    or f.name.startswith("split.")
                ):
                    continue
                vt = f.types[0]
                st = f.types[-1] if vt == GGUFValueType.ARRAY else None
                if f.name in overrides:
                    val, vt = overrides[f.name]
                    st = None
                    applied.add(f.name)
                else:
                    val = f.contents()
                if val is not None:
                    w.add_key_value(f.name, val, vt, sub_type=st)
            for k, (v, vt) in overrides.items():
                if k not in applied:
                    w.add_key_value(k, v, vt)
            for nn, t in tensors:
                w.add_tensor_info(nn, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_ti_data_to_file()

            trecs: list[dict[str, Any]] = []
            for nn, t in tensors:
                rec: dict[str, Any] = {
                    "name": nn,
                    "source_name": t.name,
                    "shape": [int(x) for x in t.shape],
                    "type": t.tensor_type.name,
                    "n_bytes": int(t.n_bytes),
                }
                if not no_hash:
                    rec["hash"] = hashlib.blake2b(tensor_bytes(t), digest_size=16).hexdigest()
                trecs.append(rec)
                if supports_tensor_endianess(w):
                    w.write_tensor_data(t.data, tensor_endianess=meta.endianess)
                elif is_big_endian:
                    raise ValueError(
                        f"Big-endian GGUF source ({shards[0]}) cannot be sliced: "
                        "installed gguf-py GGUFWriter.write_tensor_data does not support "
                        "tensor_endianess"
                    )
                else:
                    w.write_tensor_data(t.data)
                rate_limiter.update(int(t.n_bytes))
            w.close()

            # Compute whole-file hash on tmp_path
            n_bytes_file = tmp_path.stat().st_size
            fhash: str | None = None
            if not no_hash:
                n_bytes_file, fhash = compute_file_hash(tmp_path, rate_limiter)

            os.replace(tmp_path, target_path)

            file_entry: dict[str, Any] = {
                "file": fname,
                "kind": kind,
                "abs_index": abs_index,
                "n_tensors": len(tensors),
                "n_bytes_data": int(sum(t.n_bytes for _, t in tensors)),
                "n_bytes_file": n_bytes_file,
                "window_kv": {
                    "block_count": n_blk,
                    "leading_dense_block_count": n_dense,
                    "nextn_predict_layers": n_nextn,
                },
                "tensors": trecs,
            }
            if not no_hash and fhash is not None:
                file_entry["hash"] = fhash
            return file_entry
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    manifest_files: list[dict[str, Any]] = []
    for i in range(nextn_lo):
        manifest_files.append(
            write_file(
                f"blk-{i:05d}.gguf",
                "layer",
                i,
                buckets[("layer", i)],
                n_blk=1,
                n_dense=1 if i < src_leading else 0,
                n_nextn=0,
            )
        )
    for i in range(nextn_lo, src_blocks):
        manifest_files.append(
            write_file(
                f"parts-nextn-{i:05d}.gguf",
                "nextn",
                i,
                buckets[("nextn", i)],
                n_blk=1,
                n_dense=0,
                n_nextn=1,
            )
        )
    if ("embd", None) in buckets:
        manifest_files.append(
            write_file("parts-embd.gguf", "embd", None, buckets[("embd", None)], 0, 0, 0)
        )
    if ("output", None) in buckets:
        manifest_files.append(
            write_file("parts-output.gguf", "output", None, buckets[("output", None)], 0, 0, 0)
        )
    if ("other", None) in buckets:
        manifest_files.append(
            write_file("parts-other.gguf", "other", None, buckets[("other", None)], 0, 0, 0)
        )

    src_hash: str | None = None
    if not no_hash:
        acc = hashlib.blake2b(digest_size=16)
        pairs = sorted((t["source_name"], t["hash"]) for e in manifest_files for t in e["tensors"])
        for name, th in pairs:
            acc.update(name.encode())
            acc.update(bytes.fromhex(str(th)))
        src_hash = acc.hexdigest()

    manifest: dict[str, Any] = {
        "format": "gguf-layer-library/v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "layer_distribution.slice",
        "hash_algo": None if no_hash else "blake2b-128",
        "hash_scope": "whole-file" if not no_hash else None,
        "source": {
            "glob": sanitize_source_glob(model_glob, source_label),
            "shards": [{"name": Path(s).name, "n_bytes": Path(s).stat().st_size} for s in shards],
            "model_name": model_name,
            "arch": arch,
            "block_count": src_blocks,
            "leading_dense_block_count": src_leading,
            "nextn_predict_layers": src_nextn,
            "content_hash": src_hash,
        },
        "files": manifest_files,
    }

    manifest_tmp = out_path / f".manifest.json.tmp.{os.getpid()}"
    try:
        with open(manifest_tmp, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=1)
        os.replace(manifest_tmp, out_path / "manifest.json")
    finally:
        if manifest_tmp.exists():
            manifest_tmp.unlink(missing_ok=True)

    return manifest
