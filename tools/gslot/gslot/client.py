"""Synchronous client for the gslot arbiter.

Deliberately blocking and thread-based rather than async: the callers are
inference processes whose hot loop is a C++ pump thread or a shell launcher, and
neither has an event loop to join.  ``lease()`` is one round trip -- the daemon
holds the request open until the turn is granted, so there is no state machine
on this side.

Failure policy is **fail-open**, everywhere, without exception.  If the arbiter
is unreachable, slow, or wrong, every call degrades to "proceed uncoordinated".
Coordination is an optimisation and is treated as one: a scheduler that can
stop inference by dying is worse than no scheduler.  There is deliberately no
option to make this fail closed.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from collections.abc import Callable
from types import TracebackType
from typing import Any

log = logging.getLogger("gslot.client")

DEFAULT_SOCKET = os.environ.get("GSLOT_SOCKET", "/run/gslotd.sock")


class GslotError(Exception):
    pass


class Client:
    """One connection to the arbiter.  Not safe for concurrent use by threads.

    Use one Client per thread that needs leases; registration is per-connection
    because a dropped connection is how the arbiter learns a tenant died.
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET,
        *,
        timeout: float = 10.0,
        on_grant: Callable[[list[int], int], None] | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.timeout = timeout
        self.on_grant = on_grant
        self._sock: socket.socket | None = None
        self._buf = b""
        self._id = 0
        self.tenant: str | None = None
        self.cpus: list[int] = []
        self.epoch = 0

    # -- connection --------------------------------------------------------

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.socket_path)
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> Client:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- framing -----------------------------------------------------------

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _write(self, msg: dict[str, Any]) -> None:
        if self._sock is None:
            raise GslotError("not connected")
        self._sock.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode())

    def _read_frame(self) -> dict[str, Any]:
        if self._sock is None:
            raise GslotError("not connected")
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise GslotError("arbiter closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise GslotError("malformed frame")
        return obj

    def _rpc(self, msg: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Send one request; consume server pushes until the matching reply."""
        req_id = self._next_id()
        msg["id"] = req_id
        self._write(msg)
        if self._sock is not None:
            self._sock.settimeout(self.timeout if timeout is None else timeout)
        while True:
            frame = self._read_frame()
            if frame.get("op") == "grant" and frame.get("id") in (0, None):
                self._apply_grant(frame)
                continue
            if frame.get("id") == req_id:
                return frame
            # A late reply to an abandoned request; drop it.
            log.debug("dropping stale frame id=%s", frame.get("id"))

    def _apply_grant(self, frame: dict[str, Any]) -> None:
        cpus = frame.get("cpus")
        if isinstance(cpus, list):
            self.cpus = [int(c) for c in cpus]
        self.epoch = int(frame.get("epoch", 0) or 0)
        if self.on_grant is not None:
            self.on_grant(list(self.cpus), self.epoch)

    # -- ops ---------------------------------------------------------------

    def register(
        self,
        tenant: str,
        resource: str,
        mode: str,
        *,
        weight: float = 1.0,
        pid: int | None = None,
        min_units: int = 1,
        max_units: int | None = None,
        prefer_class: str | None = None,
        needs: dict[str, float] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "op": "register",
            "tenant": tenant,
            "resource": resource,
            "mode": mode,
            "weight": weight,
            "min_units": min_units,
        }
        if pid is not None:
            msg["pid"] = pid
        if max_units is not None:
            msg["max_units"] = max_units
        if prefer_class:
            msg["prefer_class"] = prefer_class
        if needs:
            msg["needs"] = needs
        if meta:
            msg["meta"] = meta
        reply = self._rpc(msg)
        if not reply.get("ok"):
            raise GslotError(f"register failed: {reply.get('error')} {reply.get('detail', '')}")
        self.tenant = tenant
        cpus = reply.get("cpus")
        if isinstance(cpus, list):
            self.cpus = [int(c) for c in cpus]
        self.epoch = int(reply.get("epoch", 0) or 0)
        return reply

    def heartbeat(self, progress: int | None = None) -> str:
        if self.tenant is None:
            raise GslotError("not registered")
        msg: dict[str, Any] = {"op": "hb", "tenant": self.tenant}
        if progress is not None:
            msg["progress"] = progress
        reply = self._rpc(msg)
        return str(reply.get("state", "unknown"))

    def lease(self, est_ms: float, *, timeout: float = 60.0) -> int | None:
        """Block until a turn is granted.  ``None`` means proceed uncoordinated."""
        if self.tenant is None:
            raise GslotError("not registered")
        try:
            reply = self._rpc(
                {"op": "lease", "tenant": self.tenant, "est_ms": est_ms}, timeout=timeout
            )
        except (OSError, GslotError, ValueError) as exc:
            log.warning("lease failed (%s); proceeding uncoordinated", exc)
            return None
        if not reply.get("ok"):
            return None
        return int(reply["lease"])

    def release(self, lease_id: int | None) -> None:
        if lease_id is None or self.tenant is None:
            return
        try:
            self._rpc({"op": "release", "lease": lease_id})
        except (OSError, GslotError, ValueError) as exc:
            log.warning("release failed (%s)", exc)

    def occupancy(self) -> dict[str, Any]:
        reply = self._rpc({"op": "stats"})
        occ = reply.get("occupancy")
        return occ if isinstance(occ, dict) else {}

    def feasibility(self, resource: str, needs: dict[str, float]) -> dict[str, Any]:
        reply = self._rpc({"op": "feasibility", "resource": resource, "needs": needs})
        rep = reply.get("report")
        return rep if isinstance(rep, dict) else {"feasible": False, "error": reply.get("error")}


def apply_affinity(cpus: list[int], pid: int = 0) -> bool:
    """Pin ``pid`` to ``cpus``.  Returns False if the set was empty or refused.

    This is the tenant applying the arbiter's grant *to itself*.  The arbiter
    never calls this on anyone else's pid -- SPEC-016 requirement 10 is upheld
    by the direction of the call, not by a permission check.
    """
    if not cpus:
        return False
    try:
        os.sched_setaffinity(pid, set(cpus))
    except (OSError, AttributeError) as exc:
        log.warning("sched_setaffinity(%s) refused: %s", cpus, exc)
        return False
    return True
