#!/usr/bin/env python3
"""gslot-bench -- coordinated vs uncoordinated co-residency, measured.

Starts N co-resident llama.cpp CPU tenants on one shared core pool, drives them
concurrently with identical work, and reports aggregate throughput and per-tenant
tail latency.  Arms:

``solo``    one tenant, threads = all cores.  The ceiling.
``unco``    N tenants, each threads = all cores, unpinned.  The naive case --
            every launcher on this fleet sizes itself for the whole machine.
``half``    N tenants, each threads = cores/N, unpinned.  The obvious manual
            fix, included so the partition arm has to beat it and not just beat
            the strawman.
``part``    N tenants, threads = cores/N, arbiter-partitioned onto disjoint
            physical cores (tier 1).
``turn``    ``part`` plus request-level turn leases (tier 2), so the tenants
            also take the shared pool in coarse alternating runs.

Stdlib only.  Emits one JSON document on stdout; everything else goes to stderr.
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


def log(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr, flush=True)


@dataclass
class Sample:
    tenant: str
    ok: bool
    wall_s: float
    predicted: int = 0
    prompt_n: int = 0
    predicted_per_s: float = 0.0
    prompt_per_s: float = 0.0
    predicted_ms: float = 0.0
    prompt_ms: float = 0.0
    lease_wait_ms: float = 0.0
    error: str = ""


@dataclass
class Tenant:
    name: str
    port: int
    proc: subprocess.Popen[bytes] | None = None
    samples: list[Sample] = field(default_factory=list)


def wait_health(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1.0)
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
        parsed = json.loads(r.read())
    return parsed if isinstance(parsed, dict) else {}


def start_tenant(
    args: argparse.Namespace, name: str, port: int, threads: int, coordinated: bool
) -> subprocess.Popen[bytes]:
    server = [
        args.server,
        "-m",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-t",
        str(threads),
        "-c",
        str(args.ctx),
        "-np",
        "1",
        "--alias",
        name,
    ]
    if args.extra_server_args:
        server += args.extra_server_args.split()
    # Left open on purpose: it is the child's stdout for the whole arm.
    logf = open(os.path.join(args.outdir, f"{name}.log"), "wb")  # noqa: SIM115
    if coordinated:
        cmd = [
            sys.executable,
            os.path.join(HERE, "gslot-run"),
            "--socket",
            args.socket,
            "--tenant",
            name,
            "--resource",
            args.resource,
            "--min-cores",
            str(args.min_cores),
            "--",
            *server,
        ]
    else:
        cmd = server
    log(f"start {name} port={port} threads={threads} coordinated={coordinated}")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)


def stop_tenant(t: Tenant) -> None:
    if t.proc is None:
        return
    # SIGTERM by PID, never pkill -f: the pattern would match this very process
    # (pgrep/pkill self-match, fleet strike #5).  And these are CPU processes --
    # if this were ever a HIP tenant, SIGKILL would risk an ASIC reset.
    try:
        t.proc.terminate()
        t.proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log(f"{t.name} did not exit on TERM; killing pid {t.proc.pid}")
        t.proc.kill()
        t.proc.wait(timeout=15)
    except OSError:
        pass


def drive(
    tenant: Tenant,
    args: argparse.Namespace,
    prompt: str,
    barrier: threading.Barrier,
    use_turns: bool,
) -> None:
    client: Client | None = None
    if use_turns:
        try:
            client = Client(args.socket, timeout=120.0)
            client.connect()
            client.register(f"{tenant.name}-driver", args.resource, "turn", weight=1.0)
        except (OSError, GslotError, ValueError) as exc:
            log(f"{tenant.name}: turn registration failed ({exc}); running uncoordinated")
            client = None
    try:
        barrier.wait(timeout=120)
    except threading.BrokenBarrierError:
        return
    for _ in range(args.requests):
        lease = None
        t_lease = time.monotonic()
        if client is not None:
            lease = client.lease(est_ms=args.est_ms, timeout=180.0)
        wait_ms = (time.monotonic() - t_lease) * 1000.0
        t0 = time.monotonic()
        try:
            res = completion(tenant.port, prompt, args.n_predict, args.req_timeout)
            wall = time.monotonic() - t0
            tm = res.get("timings") or {}
            timings = tm if isinstance(tm, dict) else {}
            tenant.samples.append(
                Sample(
                    tenant=tenant.name,
                    ok=True,
                    wall_s=wall,
                    predicted=int(timings.get("predicted_n", 0) or 0),
                    prompt_n=int(timings.get("prompt_n", 0) or 0),
                    predicted_per_s=float(timings.get("predicted_per_second", 0.0) or 0.0),
                    prompt_per_s=float(timings.get("prompt_per_second", 0.0) or 0.0),
                    predicted_ms=float(timings.get("predicted_ms", 0.0) or 0.0),
                    prompt_ms=float(timings.get("prompt_ms", 0.0) or 0.0),
                    lease_wait_ms=wait_ms,
                )
            )
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            tenant.samples.append(
                Sample(
                    tenant=tenant.name,
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
    idx = min(len(s) - 1, max(0, int(q * len(s) + 0.9999) - 1))
    return round(s[idx], 4)


def main() -> int:
    ap = argparse.ArgumentParser(prog="gslot-bench")
    ap.add_argument("--arm", required=True, choices=["solo", "unco", "half", "part", "turn"])
    ap.add_argument("--server", required=True, help="path to llama-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--socket", default=os.environ.get("GSLOT_SOCKET", "/run/gslotd.sock"))
    ap.add_argument("--resource", default="cpu:host")
    ap.add_argument("--tenants", type=int, default=2)
    ap.add_argument("--cores", type=int, required=True, help="physical cores in the shared pool")
    ap.add_argument("--min-cores", type=int, default=1)
    ap.add_argument("--base-port", type=int, default=18500)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--est-ms", type=float, default=8000.0)
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--req-timeout", type=float, default=1800.0)
    ap.add_argument("--boot-timeout", type=float, default=900.0)
    ap.add_argument("--outdir", default="/tmp/gslot-bench")
    ap.add_argument("--extra-server-args", default="")
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--threads-each",
        type=int,
        default=0,
        help="override -t per tenant. The realistic naive value is nproc: no launcher "
        "on this fleet sizes itself to its share, they all size to the machine.",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read()
    else:
        # Non-repetitive prose: a markov-friendly prompt inflates prefill and
        # (with a drafter) acceptance, which is how a bench flatters itself.
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
        prompt = " ".join(words[i % len(words)] for i in range(args.prompt_tokens))

    n = 1 if args.arm == "solo" else args.tenants
    threads = args.cores if args.arm in ("solo", "unco") else max(1, args.cores // n)
    if args.threads_each > 0:
        threads = args.threads_each
    coordinated = args.arm in ("part", "turn")
    use_turns = args.arm == "turn"

    tenants = [Tenant(name=f"bench-{i}", port=args.base_port + i) for i in range(n)]
    t_arm0 = time.monotonic()
    try:
        for t in tenants:
            t.proc = start_tenant(args, t.name, t.port, threads, coordinated)
        for t in tenants:
            if not wait_health(t.port, args.boot_timeout):
                raise SystemExit(f"{t.name} never became healthy on :{t.port}")
        log("all tenants healthy; driving load")
        barrier = threading.Barrier(n + 1)
        threads_l = [
            threading.Thread(target=drive, args=(t, args, prompt, barrier, use_turns), daemon=True)
            for t in tenants
        ]
        for th in threads_l:
            th.start()
        barrier.wait(timeout=120)
        t0 = time.monotonic()
        for th in threads_l:
            th.join()
        wall = time.monotonic() - t0
    finally:
        for t in tenants:
            stop_tenant(t)

    occ = None
    if coordinated:
        try:
            with Client(args.socket, timeout=10.0) as c:
                occ = c.occupancy()
        except (OSError, GslotError, ValueError):
            occ = None

    all_samples = [s for t in tenants for s in t.samples]
    good = [s for s in all_samples if s.ok]
    total_pred = sum(s.predicted for s in good)
    doc = {
        "arm": args.arm,
        "label": args.label,
        "host": os.uname().nodename,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "tenants": n,
            "threads_each": threads,
            "cores_pool": args.cores,
            "requests_each": args.requests,
            "n_predict": args.n_predict,
            "ctx": args.ctx,
            "model": args.model,
            "server": args.server,
            "coordinated": coordinated,
            "turn_leases": use_turns,
        },
        "wall_s": round(wall, 3),
        "arm_wall_s": round(time.monotonic() - t_arm0, 3),
        "requests_ok": len(good),
        "requests_failed": len(all_samples) - len(good),
        "aggregate": {
            # The headline is WALL-CLOCK aggregate: every token the machine
            # actually produced, divided by the time the arm took.  Summing the
            # tenants' instantaneous rates flatters a contended arm, because a
            # tenant that spent most of the arm blocked still reports a healthy
            # rate for the slice it ran.  Wall-clock aggregate cannot lie that way.
            "decode_tok_total": total_pred,
            "prompt_tok_total": sum(s.prompt_n for s in good),
            "tok_total": sum(s.prompt_n + s.predicted for s in good),
            "throughput_tok_per_s": round(sum(s.prompt_n + s.predicted for s in good) / wall, 4)
            if wall
            else 0.0,
            "decode_tok_per_s_aggregate": round(total_pred / wall, 4) if wall else 0.0,
            "sum_of_per_request_decode_rate": round(sum(s.predicted_per_s for s in good), 4),
            "decode_busy_s": round(sum(s.predicted_ms for s in good) / 1000.0, 3),
            "prefill_busy_s": round(sum(s.prompt_ms for s in good) / 1000.0, 3),
            "latency_s_p50": pct([s.wall_s for s in good], 0.50),
            "latency_s_p95": pct([s.wall_s for s in good], 0.95),
            "latency_s_p99": pct([s.wall_s for s in good], 0.99),
            "latency_s_max": round(max((s.wall_s for s in good), default=0.0), 4),
            "decode_rate_mean": round(
                statistics.fmean([s.predicted_per_s for s in good]) if good else 0.0, 4
            ),
            "prompt_rate_mean": round(
                statistics.fmean([s.prompt_per_s for s in good]) if good else 0.0, 4
            ),
            "lease_wait_ms_p50": pct([s.lease_wait_ms for s in good], 0.50),
            "lease_wait_ms_p99": pct([s.lease_wait_ms for s in good], 0.99),
        },
        "per_tenant": {
            t.name: {
                "ok": sum(1 for s in t.samples if s.ok),
                "decode_tok": sum(s.predicted for s in t.samples if s.ok),
                "decode_rate_mean": round(
                    statistics.fmean([s.predicted_per_s for s in t.samples if s.ok])
                    if any(s.ok for s in t.samples)
                    else 0.0,
                    4,
                ),
                "latency_s_p50": pct([s.wall_s for s in t.samples if s.ok], 0.50),
                "latency_s_p99": pct([s.wall_s for s in t.samples if s.ok], 0.99),
                "errors": [s.error for s in t.samples if not s.ok][:3],
            }
            for t in tenants
        },
        "samples": [vars(s) for s in all_samples],
        "arbiter_occupancy": occ,
    }
    print(json.dumps(doc, indent=2))
    with open(os.path.join(args.outdir, f"arm-{args.arm}{args.label}.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
