"""gslot -- the global slot compute arbiter (SPEC-016 phase 1+).

A host-local daemon that lets independent inference processes (stage-runner
instances, llama-server endpoints, expert-servers) share the same GPU and CPU
compute/bandwidth by scheduling *around* each other instead of colliding.

Design notes live in ``SPEC-016-global-slot-compute-scheduler.md`` next to this
package; the deploy recipe is ``runbook-gslot-expert-server.md``.

The package is deliberately **stdlib-only** so a single directory copy runs the
daemon on any GPU host (no uv, no venv, often no pip): ``scp -r`` plus
``python3 -m gslot`` is the whole install story, so nothing here may gain a
third-party import.  Ported into one home; the source project keeps a
pointer to this directory until the operator cuts it over.
"""

from gslot.protocol import (
    GRANT,
    HEARTBEAT,
    LEASE,
    REGISTER,
    RELEASE,
    Mode,
    ProtocolError,
    TenantState,
    decode_line,
    encode_line,
)

__all__ = [
    "GRANT",
    "HEARTBEAT",
    "LEASE",
    "REGISTER",
    "RELEASE",
    "Mode",
    "ProtocolError",
    "TenantState",
    "decode_line",
    "encode_line",
]
