#!/usr/bin/env python3
"""gpu-rig -- tier-2 validation on one GPU, ROCm or CUDA (SPEC-016 section 8).

Two co-resident llama.cpp tenants on ONE card. Serving flags are identical on
both backends so the numbers compare line for line; only the isolation
mechanism differs -- ROCm pins a render node into a container, CUDA pins a
UUID through ``CUDA_VISIBLE_DEVICES``. Both are index-independent by
construction, which matters on a host where somebody else's cards can briefly
read 0 MiB during a restart.

Arms:
  ``solo``            one tenant. The ceiling.
  ``unco-fit``        two tenants that both fit, uncoordinated. The baseline.
  ``turn-fit``        as above, request-granularity turn leases.
  ``unco-pressure``   an incumbent serving; a newcomer arrives that cannot fit.
  ``lease-pressure``  as above, but the newcomer must pass admission first.

Stdlib only. One JSON document on stdout, everything else on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# gslot/ lives one level up, in tools/gslot/ — the directory you scp to the host
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gslot.client import Client, GslotError

HERE = os.path.dirname(os.path.abspath(__file__))
# ROCm runtime image that carries the llama-server build used by the rig.
# Set GSLOT_ROCM_IMG to your own registry path (a HIP/ROCm container with the
# built binaries), or --backend cuda to run host binaries directly.
ROCM_IMG = os.environ.get("GSLOT_ROCM_IMG", "<registry>/ling-hip:<tag>")


def log(m: str) -> None:
    print(f"[gpu-rig] {m}", file=sys.stderr, flush=True)


def sh(cmd: list[str], timeout: float = 120.0) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)


@dataclass
class Sample:
    tenant: str
    ok: bool
    wall_s: float
    predicted: int = 0
    prompt_n: int = 0
    predicted_per_s: float = 0.0
    lease_wait_ms: float = 0.0
    error: str = ""


@dataclass
class Tenant:
    name: str
    port: int
    ctx: int
    proc: subprocess.Popen[bytes] | None = None
    samples: list[Sample] = field(default_factory=list)
    started: bool = False


# -- backend abstraction ----------------------------------------------------


def vram_used(a: argparse.Namespace) -> int:
    """Bytes in use on THIS card."""
    if a.backend == "cuda":
        _rc, out = sh(
            ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
            timeout=30,
        )
        for ln in out.splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) == 2 and parts[0] == a.gpu_uuid:
                return int(parts[1]) * 1024 * 1024
        return -1
    _rc, out = sh(["rocm-smi", "--showmeminfo", "vram"], timeout=30)
    lines = [ln for ln in out.splitlines() if "Used Memory" in ln]
    if a.rocm_index < len(lines):
        return int(lines[a.rocm_index].split()[-1])
    return -1


def serve_flags(a: argparse.Namespace, t: Tenant) -> list[str]:
    return [
        "-m",
        a.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(t.port),
        "-c",
        str(t.ctx),
        "-ngl",
        "99",
        "-np",
        str(a.np),
        "-b",
        "2048",
        "-ub",
        "512",
        "-t",
        str(a.threads),
        "--alias",
        t.name,
    ]


def launch_argv(a: argparse.Namespace, t: Tenant) -> list[str]:
    if a.backend == "cuda":
        return [a.binary, *serve_flags(a, t)]
    rd = os.path.realpath(f"/dev/dri/by-gpu/{a.alias}")
    return [
        "docker",
        "run",
        "-d",
        "--name",
        t.name,
        "--network",
        "host",
        "--device",
        "/dev/kfd",
        "--device",
        rd,
        "--group-add",
        "video",
        "--security-opt",
        "seccomp=unconfined",
        "-e",
        "GGML_HIP_GRAPHS=0",
        "-e",
        "HSA_NO_SCRATCH_RECLAIM=1",
        "-v",
        os.environ.get("GSLOT_MODEL_DIR", "/models") + ":/models:ro",
        ROCM_IMG,
        *serve_flags(a, t),
    ]


def launch_env(a: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if a.backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = a.gpu_uuid
        env["LD_LIBRARY_PATH"] = os.path.dirname(a.binary)
    return env


def spawn(a: argparse.Namespace, t: Tenant) -> int:
    if a.backend == "cuda":
        fh = open(os.path.join(a.outdir, f"{t.name}.log"), "wb")  # noqa: SIM115
        t.proc = subprocess.Popen(
            launch_argv(a, t), stdout=fh, stderr=subprocess.STDOUT, env=launch_env(a)
        )
        return 0
    sh(["docker", "rm", "-f", t.name], timeout=60)
    rc, _out = sh(launch_argv(a, t), timeout=180)
    return rc


def alive(a: argparse.Namespace, t: Tenant) -> bool:
    if a.backend == "cuda":
        return t.proc is not None and t.proc.poll() is None
    _rc, out = sh(["docker", "ps", "-q", "-f", f"name={t.name}"], timeout=30)
    return bool(out.strip())


def tenant_log(a: argparse.Namespace, t: Tenant) -> str:
    if a.backend == "cuda":
        try:
            with open(os.path.join(a.outdir, f"{t.name}.log"), errors="replace") as fh:
                return fh.read()[-8000:]
        except OSError:
            return ""
    _rc, out = sh(["docker", "logs", "--tail", "60", t.name], timeout=60)
    return out


def stop_tenant(a: argparse.Namespace, t: Tenant) -> None:
    """Graceful only. A HIP process must never be SIGKILLed -- an MES timeout
    takes the ASIC down and the collateral is every other tenant on the card.
    CUDA has no equivalent, but the same discipline costs nothing."""
    if a.backend == "cuda":
        if t.proc is None:
            return
        try:
            t.proc.terminate()
            t.proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            log(f"{t.name} did not exit on TERM after 45 s; killing pid {t.proc.pid}")
            t.proc.kill()
            t.proc.wait(timeout=20)
        except OSError:
            pass
        return
    sh(["docker", "kill", "--signal=TERM", t.name], timeout=60)
    for _ in range(45):
        if not alive(a, t):
            break
        time.sleep(1.0)
    else:
        log(f"{t.name} still alive 45 s after TERM -- leaving it, NOT killing (HIP)")
    sh(["docker", "rm", "-f", t.name], timeout=60)


# -- driving ----------------------------------------------------------------


def wait_health(a: argparse.Namespace, t: Tenant, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{t.port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if not alive(a, t):
            return False
        time.sleep(2.0)
    return False


def completion(port: int, prompt: str, n_predict: int, timeout: float) -> dict[str, object]:
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "cache_prompt": False,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out if isinstance(out, dict) else {}


def verify(t: Tenant, prompt: str) -> bool:
    """Health says the socket answers. Only a real completion says the tenant
    actually works -- a rig that measured a half-started instance cost a whole
    arm once already."""
    try:
        res = completion(t.port, prompt[:64], 4, 300.0)
        return bool(res.get("content") or res.get("text") or res.get("choices"))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        log(f"{t.name} verification completion failed: {exc!r}")
        return False


def drive(
    t: Tenant, a: argparse.Namespace, prompt: str, barrier: threading.Barrier, use_turns: bool
) -> None:
    client: Client | None = None
    if use_turns:
        try:
            client = Client(a.socket, timeout=120.0)
            client.connect()
            client.register(f"{t.name}-drv", a.resource, "turn", weight=1.0)
        except (OSError, GslotError, ValueError) as exc:
            log(f"{t.name}: turn registration failed ({exc}); uncoordinated")
            client = None
    try:
        barrier.wait(timeout=180)
    except threading.BrokenBarrierError:
        return
    for _ in range(a.requests):
        lease, t0l = None, time.monotonic()
        if client is not None:
            lease = client.lease(est_ms=a.est_ms, timeout=600.0)
        wait_ms = (time.monotonic() - t0l) * 1000.0
        t0 = time.monotonic()
        try:
            res = completion(t.port, prompt, a.n_predict, a.req_timeout)
            tm = res.get("timings") or {}
            tim = tm if isinstance(tm, dict) else {}
            t.samples.append(
                Sample(
                    tenant=t.name,
                    ok=True,
                    wall_s=time.monotonic() - t0,
                    predicted=int(tim.get("predicted_n", 0) or 0),
                    prompt_n=int(tim.get("prompt_n", 0) or 0),
                    predicted_per_s=float(tim.get("predicted_per_second", 0.0) or 0.0),
                    lease_wait_ms=wait_ms,
                )
            )
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            t.samples.append(
                Sample(
                    tenant=t.name,
                    ok=False,
                    wall_s=time.monotonic() - t0,
                    lease_wait_ms=wait_ms,
                    error=repr(exc),
                )
            )
        finally:
            if client is not None:
                client.release(lease)
    if client is not None:
        client.close()


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(s[min(len(s) - 1, max(0, int(q * len(s) + 0.9999) - 1))], 4)


OOM_PATTERNS = (
    "failed to allocate",
    "out of memory",
    "hiperroroutofmemory",
    "cudaerrormemoryallocation",
    "cuda error",
    "hip error",
    "ggml_backend",
)


def main() -> int:
    ap = argparse.ArgumentParser(prog="gpu-rig")
    ap.add_argument(
        "--arm",
        required=True,
        choices=["solo", "unco-fit", "turn-fit", "unco-pressure", "lease-pressure"],
    )
    ap.add_argument("--backend", default="rocm", choices=["rocm", "cuda"])
    ap.add_argument("--binary", default="/path/to/llama-server")
    ap.add_argument("--gpu-uuid", default="")
    ap.add_argument("--alias", default="xt-14-render")
    ap.add_argument("--rocm-index", type=int, default=2)
    ap.add_argument("--resource", required=True)
    ap.add_argument("--socket", default="/run/gslotd.sock")
    ap.add_argument("--model", default="/models/gemma-4-12b-it-qat-q4_0.gguf")
    ap.add_argument("--ctx-fit", type=int, default=2048)
    ap.add_argument("--ctx-pressure", type=int, default=16384)
    ap.add_argument("--need-bytes", type=int, default=14306144256)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument(
        "--np",
        type=int,
        default=1,
        help="parallel slots. On CUDA, gemma's sliding window bounds KV by the window "
        "rather than the context, so slots -- not ctx -- are the footprint lever.",
    )
    ap.add_argument("--base-port", type=int, default=18710)
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--est-ms", type=float, default=20000.0)
    ap.add_argument("--req-timeout", type=float, default=900.0)
    ap.add_argument("--boot-timeout", type=float, default=420.0)
    ap.add_argument("--outdir", default="gslot-out")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.backend == "cuda" and not a.gpu_uuid:
        ap.error("--gpu-uuid is required for the cuda backend (never an index)")

    words = [
        "orbit",
        "lantern",
        "gravel",
        "muster",
        "vellum",
        "thicket",
        "quorum",
        "sable",
        "ridge",
        "hollow",
        "tangent",
        "murmur",
        "cinder",
        "plait",
        "wrangle",
        "fathom",
        "nettle",
        "brisk",
        "quarry",
        "lattice",
        "pommel",
        "drought",
        "vantage",
        "sortie",
        "kindle",
        "marrow",
    ]
    prompt = " ".join(words[i % len(words)] for i in range(256))

    doc: dict[str, object] = {
        "arm": a.arm,
        "backend": a.backend,
        "card": a.gpu_uuid or a.alias,
        "resource": a.resource,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "threads": a.threads,
        "vram_before": vram_used(a),
    }
    pressure = a.arm.endswith("pressure")
    ctx = a.ctx_pressure if pressure else a.ctx_fit
    n = 1 if a.arm == "solo" else 2
    tenants = [Tenant(name=f"gs{i}", port=a.base_port + i, ctx=ctx) for i in range(n)]
    faults: list[str] = []

    try:
        if not pressure:
            for t in tenants:
                rc = spawn(a, t)
                t.started = rc == 0
                log(f"start {t.name} ctx={t.ctx} rc={rc}")
            for t in tenants:
                if not wait_health(a, t, a.boot_timeout):
                    faults.append(f"{t.name}: never became healthy")
                elif not verify(t, prompt):
                    faults.append(f"{t.name}: failed verification completion")
                else:
                    log(f"{t.name} verified")
            doc["vram_loaded"] = vram_used(a)
            live = [t for t in tenants if not any(t.name in f for f in faults)]
            if not live:
                raise SystemExit("no live tenants; aborting arm")
            barrier = threading.Barrier(len(live) + 1)
            ths = [
                threading.Thread(
                    target=drive, args=(t, a, prompt, barrier, a.arm == "turn-fit"), daemon=True
                )
                for t in live
            ]
            for th in ths:
                th.start()
            barrier.wait(timeout=180)
            t0 = time.monotonic()
            for th in ths:
                th.join()
            doc["wall_s"] = round(time.monotonic() - t0, 3)
        else:
            inc, new = tenants[0], tenants[1]
            spawn(a, inc)
            if not wait_health(a, inc, a.boot_timeout) or not verify(inc, prompt):
                faults.append(f"{inc.name}: incumbent never came up")
                raise SystemExit("incumbent failed to start; aborting arm")
            inc.started = True
            doc["vram_incumbent"] = vram_used(a)
            log(f"incumbent verified, vram={doc['vram_incumbent']}")

            cli: Client | None = None
            if a.arm == "lease-pressure":
                cli = Client(a.socket, timeout=30.0)
                cli.connect()
                cli.register(
                    "gpu-incumbent", a.resource, "share", needs={"vram_bytes": float(a.need_bytes)}
                )
                doc["incumbent_registered_need"] = a.need_bytes

            stop_inc = threading.Event()
            inc_res: list[Sample] = []

            def keep_serving() -> None:
                # The incumbent must be DOING WORK while the newcomer allocates.
                # An idle incumbent proves nothing.
                while not stop_inc.is_set():
                    t0 = time.monotonic()
                    try:
                        r = completion(inc.port, prompt, a.n_predict, a.req_timeout)
                        tm = r.get("timings") or {}
                        tim = tm if isinstance(tm, dict) else {}
                        inc_res.append(
                            Sample(
                                tenant=inc.name,
                                ok=True,
                                wall_s=time.monotonic() - t0,
                                predicted=int(tim.get("predicted_n", 0) or 0),
                                predicted_per_s=float(tim.get("predicted_per_second", 0.0) or 0.0),
                            )
                        )
                    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
                        inc_res.append(
                            Sample(
                                tenant=inc.name,
                                ok=False,
                                wall_s=time.monotonic() - t0,
                                error=repr(exc),
                            )
                        )

            th = threading.Thread(target=keep_serving, daemon=True)
            th.start()
            time.sleep(5)

            if a.arm == "lease-pressure":
                cmd = [
                    sys.executable,
                    os.path.join(HERE, "gslot-run"),
                    "--socket",
                    a.socket,
                    "--tenant",
                    "gpu-newcomer",
                    "--resource",
                    a.resource,
                    "--mode",
                    "turn",
                    "--needs",
                    f"vram_bytes={a.need_bytes}",
                    "--require-admission",
                    "--admit-until-healthy",
                    f"http://127.0.0.1:{new.port}/health",
                    "--",
                    *launch_argv(a, new),
                ]
                p = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600, env=launch_env(a)
                )
                rc, out = p.returncode, p.stdout + p.stderr
            else:
                rc = spawn(a, new)
                out = ""
            doc["newcomer_rc"] = rc
            doc["newcomer_out"] = out[-1500:]
            log(f"newcomer launch rc={rc}")
            if rc == 0:
                new.started = True
                ok = wait_health(a, new, 180)
                doc["newcomer_healthy"] = ok
                if not ok:
                    faults.append(f"{new.name}: allocation failed / never healthy")
            else:
                doc["newcomer_healthy"] = False
            time.sleep(20)
            stop_inc.set()
            th.join(timeout=a.req_timeout + 30)
            inc.samples = inc_res
            doc["vram_after_newcomer"] = vram_used(a)
            if cli is not None:
                cli.close()
            for t in (inc, new):
                if t.started:
                    lg = tenant_log(a, t).lower()
                    for pat in OOM_PATTERNS:
                        if pat in lg:
                            faults.append(f"{t.name}: {pat}")
                            break
    finally:
        for t in tenants:
            if t.started or t.proc is not None:
                stop_tenant(a, t)
        time.sleep(3)
        doc["vram_after"] = vram_used(a)

    occ = None
    try:
        with Client(a.socket, timeout=10.0) as c:
            occ = c.occupancy()
    except (OSError, GslotError, ValueError):
        pass

    alls = [s for t in tenants for s in t.samples]
    good = [s for s in alls if s.ok]
    wall = float(doc.get("wall_s") or 0.0)
    doc.update(
        {
            "faults": faults,
            "fault_count": len(faults),
            "requests_ok": len(good),
            "requests_failed": len(alls) - len(good),
            "decode_tok_total": sum(s.predicted for s in good),
            "throughput_tok_per_s": (
                round(sum(s.prompt_n + s.predicted for s in good) / wall, 4) if wall else None
            ),
            "latency_s_p50": pct([s.wall_s for s in good], 0.50),
            "latency_s_p99": pct([s.wall_s for s in good], 0.99),
            "decode_rate_mean": (
                round(statistics.fmean([s.predicted_per_s for s in good]), 4) if good else 0.0
            ),
            "lease_wait_ms_p50": pct([s.lease_wait_ms for s in good], 0.50),
            "per_tenant": {
                t.name: {
                    "ok": sum(1 for s in t.samples if s.ok),
                    "failed": sum(1 for s in t.samples if not s.ok),
                    "decode_tok": sum(s.predicted for s in t.samples if s.ok),
                    "errors": [s.error for s in t.samples if not s.ok][:3],
                }
                for t in tenants
            },
            "arbiter_occupancy": occ,
            "samples": [
                {
                    "tenant": s.tenant,
                    "ok": s.ok,
                    "wall_s": round(s.wall_s, 3),
                    "predicted": s.predicted,
                    "rate": round(s.predicted_per_s, 3),
                    "error": s.error[:120],
                }
                for s in alls
            ],
        }
    )
    print(json.dumps(doc, indent=2))
    with open(os.path.join(a.outdir, f"arm-{a.backend}-{a.arm}.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
