"""The gslotd daemon: AF_UNIX control plane + read-only HTTP observation plane.

Stdlib only -- asyncio, json, logging.  A GPU host here has no uv, no
venv and often no pip; ``scp -r tools/gslot`` plus ``python3 -m gslot``
has to be the whole install story, so the daemon may not import aiohttp or
structlog even though other Python in the fleet does.

Two planes, deliberately separated:

*control plane* (unix socket, write)
    Tenants register, request turns, heartbeat.  Reaching it requires filesystem
    access to the socket, which is the authorisation model.

*observation plane* (TCP, read-only)
    ``/occupancy``, ``/tenants``, ``/resources``, ``/inventory``, ``/healthz``.
    Nothing here mutates state, so a fleet-wide scraper can poll it without any
    possibility of actuation -- SPEC-016 requirement 10, enforced by having no
    code path rather than by a permission check.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gslot import protocol as P
from gslot.arbiter import (
    DEFAULT_QUANTUM_MS,
    Arbiter,
    Infeasible,
    Resource,
    Tenant,
    cpu_resource,
)
from gslot.topology import HostInventory, read_inventory

log = logging.getLogger("gslotd")

POLL_INTERVAL_S = 0.05


@dataclass(slots=True)
class _Conn:
    writer: asyncio.StreamWriter
    peer: str
    tenants: set[str]


class Server:
    def __init__(
        self,
        arbiter: Arbiter,
        inventory: HostInventory,
        *,
        socket_path: Path,
        http_addr: tuple[str, int] | None,
    ) -> None:
        self.arb = arbiter
        self.inventory = inventory
        self.socket_path = socket_path
        self.http_addr = http_addr
        self._conns: dict[int, _Conn] = {}
        self._tenant_conn: dict[str, int] = {}
        # lease_id -> (conn_key, request id) for turns still waiting for a grant
        self._pending: dict[int, tuple[int, int]] = {}
        self._stop = asyncio.Event()
        self.started_at = time.time()

    # -- plumbing ----------------------------------------------------------

    async def _send(self, key: int, msg: dict[str, Any]) -> None:
        conn = self._conns.get(key)
        if conn is None:
            return
        try:
            conn.writer.write(P.encode_line(msg))
            await conn.writer.drain()
        except (ConnectionError, RuntimeError):
            log.debug("send failed to %s; dropping", conn.peer)

    async def _handle_conn(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        key = id(writer)
        peer = str(writer.get_extra_info("peername") or f"unix:{key}")
        self._conns[key] = _Conn(writer=writer, peer=peer, tenants=set())
        log.info("conn open %s", peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = P.decode_line(line)
                except P.ProtocolError as exc:
                    await self._send(key, P.err(0, str(exc)))
                    continue
                reply = await self._dispatch(key, msg)
                if reply is not None:
                    await self._send(key, reply)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            await self._close_conn(key)

    async def _close_conn(self, key: int) -> None:
        conn = self._conns.pop(key, None)
        if conn is None:
            return
        # A dropped connection is a *measured* death signal -- stronger than a
        # heartbeat timeout, so we act on it immediately rather than leaving the
        # tenant's capacity stranded for hb_timeout_s.  This is the registry-decay
        # class of bug (SPEC-016 req 7) closed at its cheapest point.
        for tid in sorted(conn.tenants):
            self._tenant_conn.pop(tid, None)
            self.arb.unregister(tid)
            log.info("tenant %s unregistered (conn closed)", tid)
        for lid, (ck, _rid) in list(self._pending.items()):
            if ck == key:
                self._pending.pop(lid, None)
        with contextlib.suppress(ConnectionError, RuntimeError):
            conn.writer.close()
        await self._flush_grants()
        await self._push_partitions()

    # -- ops ---------------------------------------------------------------

    async def _dispatch(self, key: int, msg: dict[str, Any]) -> dict[str, Any] | None:
        req_id = int(msg.get("id", 0) or 0)
        op = msg.get("op")
        who = msg.get("tenant")
        if isinstance(who, str) and op != P.REGISTER:
            # Any frame is liveness evidence -- see Arbiter.touch().
            self.arb.touch(who)
        try:
            if op == P.REGISTER:
                return await self._op_register(key, req_id, msg)
            if op == P.UNREGISTER:
                tid = P.require_str(msg, "tenant")
                self._conns[key].tenants.discard(tid)
                self._tenant_conn.pop(tid, None)
                self.arb.unregister(tid)
                await self._push_partitions()
                return P.ok(req_id)
            if op == P.HEARTBEAT:
                tid = P.require_str(msg, "tenant")
                prog = msg.get("progress")
                state = self.arb.heartbeat(tid, int(prog) if isinstance(prog, int) else None)
                return P.ok(req_id, state=state, epoch=self.arb.epoch)
            if op == P.LEASE:
                return await self._op_lease(key, req_id, msg)
            if op == P.RELEASE:
                lid = P.require_int(msg, "lease")
                self.arb.release(lid)
                await self._flush_grants()
                return P.ok(req_id)
            if op == P.STATS:
                return P.ok(req_id, occupancy=self.arb.occupancy())
            if op == P.RESOURCES:
                return P.ok(req_id, resources=self._resources_doc())
            if op == P.TENANTS:
                return P.ok(req_id, occupancy=self.arb.occupancy())
            if op == P.FEASIBILITY:
                rid = P.require_str(msg, "resource")
                needs = msg.get("needs") or {}
                if not isinstance(needs, dict):
                    raise P.ProtocolError("'needs' must be an object")
                return P.ok(req_id, report=self.arb.feasibility(rid, needs))
            return P.err(req_id, f"unknown op {op!r}")
        except P.ProtocolError as exc:
            return P.err(req_id, str(exc))
        except Infeasible as exc:
            return P.err(req_id, exc.reason, detail=exc.detail)

    async def _op_register(self, key: int, req_id: int, msg: dict[str, Any]) -> dict[str, Any]:
        tid = P.require_str(msg, "tenant")
        rid = P.require_str(msg, "resource")
        mode = P.require_mode(msg)
        if tid in self.arb.tenants:
            return P.err(req_id, f"tenant {tid!r} already registered")
        needs = msg.get("needs") or {}
        if not isinstance(needs, dict):
            raise P.ProtocolError("'needs' must be an object")
        report = self.arb.feasibility(rid, needs) if needs else None
        if report is not None and not report.get("feasible", True):
            return P.err(req_id, "infeasible claim", report=report)
        pid_raw = msg.get("pid")
        tenant = Tenant(
            tid=tid,
            rid=rid,
            mode=mode,
            weight=P.require_float(msg, "weight", default=1.0),
            pid=int(pid_raw) if isinstance(pid_raw, int) else None,
            min_units=P.require_int(msg, "min_units", default=1),
            max_units=(
                int(msg["max_units"])
                if isinstance(msg.get("max_units"), int)
                and not isinstance(msg.get("max_units"), bool)
                else None
            ),
            prefer_class=(
                str(msg["prefer_class"]) if isinstance(msg.get("prefer_class"), str) else None
            ),
            needs={str(k): float(v) for k, v in needs.items()},
            meta=dict(msg.get("meta") or {}),
        )
        self.arb.register(tenant)
        self._conns[key].tenants.add(tid)
        self._tenant_conn[tid] = key
        log.info(
            "tenant %s registered on %s mode=%s weight=%.2f pid=%s",
            tid,
            rid,
            mode,
            tenant.weight,
            tenant.pid,
        )
        await self._push_partitions(exclude=tid)
        return P.ok(
            req_id,
            tenant=tid,
            resource=rid,
            mode=mode,
            epoch=tenant.epoch,
            cpus=list(tenant.assigned),
            feasibility=report,
            hb_timeout_s=self.arb.hb_timeout_s,
        )

    async def _op_lease(self, key: int, req_id: int, msg: dict[str, Any]) -> dict[str, Any] | None:
        tid = P.require_str(msg, "tenant")
        est_ms = P.require_float(msg, "est_ms", default=0.0)
        nowait = bool(msg.get("nowait"))
        lease, granted = self.arb.request_lease(tid, est_ms, nowait=nowait)
        if granted:
            return self._grant_msg(req_id, lease.lease_id)
        if nowait:
            return P.err(req_id, "busy", queued=False)
        # Hold the request open; the reply is the grant.  A blocking lease()
        # keeps the client contract to one round trip and no state machine.
        self._pending[lease.lease_id] = (key, req_id)
        return None

    def _grant_msg(self, req_id: int, lease_id: int) -> dict[str, Any]:
        lease = self.arb.lease_by_id(lease_id)
        wait_ms = 0.0
        ttl_ms = 0.0
        if lease is not None:
            wait_ms = (lease.granted_at - lease.requested_at) * 1000.0
            ttl_ms = (lease.expires_at - lease.granted_at) * 1000.0
        return P.ok(
            req_id,
            lease=lease_id,
            wait_ms=round(wait_ms, 3),
            ttl_ms=round(ttl_ms, 3),
        )

    async def _flush_grants(self) -> None:
        """Deliver replies for any lease that just became active."""
        for rid in self.arb.resources:
            for lease in self.arb.active_leases(rid):
                pend = self._pending.pop(lease.lease_id, None)
                if pend is None:
                    continue
                key, req_id = pend
                await self._send(key, self._grant_msg(req_id, lease.lease_id))

    async def _push_partitions(self, exclude: str | None = None) -> None:
        """Tell every partition tenant its (possibly new) core set.

        ``exclude`` skips the tenant that is about to receive the same
        assignment in its own reply -- a self-push before the reply would make
        every client parse a frame it is going to be told anyway.
        """
        for tid, t in sorted(self.arb.tenants.items()):
            if t.mode != "partition" or tid == exclude:
                continue
            key = self._tenant_conn.get(tid)
            if key is None:
                continue
            await self._send(
                key,
                {
                    "id": 0,
                    "op": P.GRANT,
                    "tenant": tid,
                    "resource": t.rid,
                    "epoch": t.epoch,
                    "cpus": list(t.assigned),
                },
            )

    # -- background --------------------------------------------------------

    async def _poller(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(POLL_INTERVAL_S)
            dead = self.arb.reap()
            for tid in dead:
                self._tenant_conn.pop(tid, None)
                log.warning("tenant %s reaped: heartbeat stale", tid)
            self.arb.poll()
            await self._flush_grants()
            if dead:
                await self._push_partitions()

    # -- observation plane -------------------------------------------------

    def _resources_doc(self) -> dict[str, Any]:
        return {
            rid: {
                "kind": r.kind,
                "units": [list(u) for u in r.units],
                "reserved_units": [list(u) for u in r.reserved_units],
                "allocatable_units": len(r.allocatable),
                "concurrency": r.concurrency,
                "quantum_ms": r.quantum_ms,
                "dims": r.dims,
                "meta": r.meta,
            }
            for rid, r in sorted(self.resources_items())
        }

    def resources_items(self) -> list[tuple[str, Resource]]:
        return sorted(self.arb.resources.items())

    async def _handle_http(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            while True:
                hdr = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if hdr in (b"\r\n", b"\n", b""):
                    break
            parts = request_line.decode("latin-1").split()
            method = parts[0] if parts else ""
            path = parts[1].split("?", 1)[0] if len(parts) > 1 else "/"
            if method != "GET":
                body = json.dumps({"error": "read-only plane"}).encode()
                status = "405 Method Not Allowed"
            elif path in ("/occupancy", "/", "/stats"):
                body, status = json.dumps(self.arb.occupancy()).encode(), "200 OK"
            elif path == "/resources":
                body, status = json.dumps(self._resources_doc()).encode(), "200 OK"
            elif path == "/tenants":
                body, status = (
                    json.dumps(self.arb.occupancy()["resources"]).encode(),
                    "200 OK",
                )
            elif path == "/inventory":
                body, status = json.dumps(self.inventory.to_dict()).encode(), "200 OK"
            elif path == "/healthz":
                body = json.dumps(
                    {
                        "ok": True,
                        "server_time": time.time(),
                        "uptime_s": round(time.time() - self.started_at, 3),
                        "tenants": len(self.arb.tenants),
                        "resources": len(self.arb.resources),
                        "inventory_age_s": round(self.inventory.age_s, 3),
                    }
                ).encode()
                status = "200 OK"
            else:
                body, status = json.dumps({"error": "not found"}).encode(), "404 Not Found"
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (TimeoutError, ConnectionError, UnicodeDecodeError):
            pass
        finally:
            with contextlib.suppress(ConnectionError, RuntimeError):
                writer.close()

    # -- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        unix = await asyncio.start_unix_server(self._handle_conn, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o660)
        servers = [unix]
        if self.http_addr is not None:
            http = await asyncio.start_server(
                self._handle_http, host=self.http_addr[0], port=self.http_addr[1]
            )
            servers.append(http)
            log.info("http observation plane on %s:%d", *self.http_addr)
        log.info("control plane on %s", self.socket_path)
        poller = asyncio.create_task(self._poller())
        try:
            await self._stop.wait()
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller
            for s in servers:
                s.close()
                await s.wait_closed()
            with contextlib.suppress(OSError):
                self.socket_path.unlink()

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_cpu_list(spec: str) -> list[int]:
    """Parse a Linux cpu-list (``0-3,8,12-15``)."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def build_server(args: argparse.Namespace) -> Server:
    inv = read_inventory(probe_gpus=not args.no_gpu_probe)
    arb = Arbiter(hb_timeout_s=args.hb_timeout, stall_s=args.stall_after)
    cores = inv.cores()
    reserved_cpus = set(parse_cpu_list(args.reserve_cpus)) if args.reserve_cpus else set()
    # A physical core is reserved if ANY of its logical CPUs is reserved: half a
    # core is not a usable grant, and handing out the other sibling is the SMT
    # collision this whole daemon exists to prevent.
    reserved_cores = [k for k, cpus in cores.items() if reserved_cpus.intersection(cpus)]
    arb.add_resource(
        cpu_resource(
            args.cpu_resource,
            cores,
            kinds=inv.core_kinds(),
            reserved_cores=reserved_cores,
            quantum_ms=args.quantum_ms,
            dims={"host_ram_bytes": float(inv.mem_available_bytes)},
        )
    )
    for g in inv.gpus:
        arb.add_resource(
            Resource(
                rid=f"gpu:{g.key}",
                kind="gpu",
                concurrency=args.gpu_concurrency,
                quantum_ms=args.quantum_ms,
                dims={"vram_bytes": float(g.vram_total_bytes - g.vram_used_bytes)},
                meta={"vendor": g.vendor, "name": g.name, "alias": g.alias},
            )
        )
    http = None
    if args.http:
        host, _, port = args.http.rpartition(":")
        http = (host or "127.0.0.1", int(port))
    return Server(arb, inv, socket_path=Path(args.socket), http_addr=http)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gslotd", description="global slot compute arbiter")
    ap.add_argument("--socket", default="/run/gslotd.sock")
    ap.add_argument("--http", default="127.0.0.1:8099", help="observation plane; '' to disable")
    ap.add_argument("--cpu-resource", default="cpu:host")
    ap.add_argument(
        "--reserve-cpus",
        default="",
        help="cpu-list withheld from tenants (non-participants: ring stages, OS)",
    )
    ap.add_argument("--quantum-ms", type=float, default=DEFAULT_QUANTUM_MS)
    ap.add_argument("--gpu-concurrency", type=int, default=1)
    ap.add_argument("--hb-timeout", type=float, default=30.0)
    ap.add_argument("--stall-after", type=float, default=10.0)
    ap.add_argument("--no-gpu-probe", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    srv = build_server(args)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, srv.stop)
        await srv.run()

    asyncio.run(_run())
    return 0
