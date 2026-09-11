"""Wire protocol for the gslot arbiter.

Framing: newline-delimited JSON over ``AF_UNIX`` SOCK_STREAM.  One JSON object
per line, no embedded newlines (``json.dumps`` guarantees this).  Chosen over a
binary frame format because every diagnostic on this fleet ends with somebody
piping a socket through ``jq`` at 3am, and because the per-wave RPC budget
(~100us on a 100-200ms wave) leaves three orders of magnitude of headroom.

Every message carries ``op``.  Requests carry ``id`` (client-chosen, echoed in
the reply) so a client may pipeline.  Server-initiated pushes carry ``id: 0``.
"""

from __future__ import annotations

import json
from typing import Any, Final, Literal

# --- ops -------------------------------------------------------------------

REGISTER: Final = "register"
UNREGISTER: Final = "unregister"
LEASE: Final = "lease"
RELEASE: Final = "release"
HEARTBEAT: Final = "hb"
GRANT: Final = "grant"  # server -> client push (partition assignment changed)
STATS: Final = "stats"
RESOURCES: Final = "resources"
TENANTS: Final = "tenants"
FEASIBILITY: Final = "feasibility"

Mode = Literal["partition", "turn", "share"]
"""How a tenant intends to consume a resource.

``partition``
    Give me a disjoint slice of the capacity for my lifetime (a core set).  The
    arbiter returns a concrete assignment; *the tenant applies it to itself*.
    Zero steady-state RPC.  This is the fix for the ggml intra-graph spin
    barrier, where two co-resident CPU tenants each spinning at the node
    barrier evict each other (measured 4x collapse, expert-disagg P1).

``turn``
    Grant me a time-bounded exclusive turn before each wave/batch.  The arbiter
    serialises turns with weighted fair queuing and a quantum, so co-tenants
    alternate in coarse blocks instead of thrashing at kernel granularity.

``share``
    I will run uncoordinated; count me against capacity but never gate me.
    Used to model non-participating tenants (e.g. the production GLM ring) so
    the arbiter does not over-commit a resource it does not fully own.
"""

TenantState = Literal["active", "saturated", "stalled", "dead"]
"""SPEC-016 requirement 9: saturated != stalled != dead.

Classified from a *measured* progress counter plus heartbeat freshness, never
from a health probe.  ``saturated`` means progress is advancing but slowly;
``stalled`` means the heartbeat is fresh and progress is frozen; ``dead`` means
the heartbeat itself went stale and the claims were reclaimed.
"""


class ProtocolError(Exception):
    """Malformed frame, unknown op, or a field with the wrong type."""


def encode_line(msg: dict[str, Any]) -> bytes:
    """Serialise one frame, newline-terminated."""
    return (json.dumps(msg, separators=(",", ":"), sort_keys=True) + "\n").encode()


def decode_line(line: bytes | str) -> dict[str, Any]:
    """Parse one frame.

    Raises:
        ProtocolError: if the line is not a JSON object.
    """
    try:
        obj = json.loads(line)
    except ValueError as exc:  # pragma: no cover - trivial passthrough
        raise ProtocolError(f"not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"frame is {type(obj).__name__}, want object")
    return obj


def require_str(msg: dict[str, Any], key: str) -> str:
    val = msg.get(key)
    if not isinstance(val, str) or not val:
        raise ProtocolError(f"{key!r} must be a non-empty string")
    return val


def require_int(msg: dict[str, Any], key: str, *, default: int | None = None) -> int:
    val = msg.get(key, default)
    if isinstance(val, bool) or not isinstance(val, int):
        raise ProtocolError(f"{key!r} must be an int")
    return val


def require_float(msg: dict[str, Any], key: str, *, default: float | None = None) -> float:
    val = msg.get(key, default)
    if isinstance(val, bool) or not isinstance(val, int | float):
        raise ProtocolError(f"{key!r} must be a number")
    return float(val)


def require_mode(msg: dict[str, Any], key: str = "mode") -> Mode:
    val = msg.get(key)
    if val == "partition":
        return "partition"
    if val == "turn":
        return "turn"
    if val == "share":
        return "share"
    raise ProtocolError(f"{key!r} must be one of partition|turn|share, got {val!r}")


def ok(req_id: int, **fields: Any) -> dict[str, Any]:
    return {"id": req_id, "ok": True, **fields}


def err(req_id: int, reason: str, **fields: Any) -> dict[str, Any]:
    return {"id": req_id, "ok": False, "error": reason, **fields}
