#!/usr/bin/env python3
"""Co-tenant load driver for SPEC-016 Half B.

Hammers a llama-server, optionally taking a turn lease around each request so
the co-tenant participates in the same arbitration as the ring's stage. The
point of --n-predict here is not the workload, it is the LEASE HOLD TIME: a
64-token request holds the device ~1.4 s, a 4-token request ~80 ms. That is the
granularity-compatibility variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# gslot/ lives one level up, in tools/gslot/ — the directory you scp to the host
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gslot.client import Client, GslotError

PROMPT = " ".join(
    [
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
    ]
    * 16
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--lease", action="store_true")
    ap.add_argument("--socket", default="/run/gslotd.sock")
    ap.add_argument("--resource", required=True)
    ap.add_argument("--tenant", default="cotenant")
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--out", default="/tmp/cot.json")
    a = ap.parse_args()

    cli: Client | None = None
    if a.lease:
        try:
            cli = Client(a.socket, timeout=120.0)
            cli.connect()
            cli.register(a.tenant, a.resource, "turn", weight=a.weight)
        except (OSError, GslotError, ValueError) as exc:
            print(f"[cot] lease registration failed ({exc}); uncoordinated", file=sys.stderr)
            cli = None

    body = json.dumps(
        {
            "prompt": PROMPT,
            "n_predict": a.n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "cache_prompt": False,
            "stream": False,
        }
    ).encode()
    url = f"http://127.0.0.1:{a.port}/completion"
    done = 0
    toks = 0
    waits: list[float] = []
    t_end = time.monotonic() + a.duration
    while time.monotonic() < t_end:
        lease = None
        t0 = time.monotonic()
        if cli is not None:
            # est_ms tracks the real hold so the arbiter's virtual time is honest
            lease = cli.lease(est_ms=a.n_predict * 20.0, timeout=300.0)
        waits.append((time.monotonic() - t0) * 1000.0)
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            tm = d.get("timings") or {}
            toks += int((tm if isinstance(tm, dict) else {}).get("predicted_n", 0) or 0)
            done += 1
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            pass
        finally:
            if cli is not None:
                cli.release(lease)
    if cli is not None:
        cli.close()
    waits.sort()
    with open(a.out, "w") as fh:
        json.dump(
            {
                "requests": done,
                "tokens": toks,
                "leased": cli is not None,
                "n_predict": a.n_predict,
                "lease_wait_ms_p50": waits[len(waits) // 2] if waits else 0.0,
            },
            fh,
        )
    print(f"[cot] {done} requests, {toks} tokens, leased={cli is not None}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
