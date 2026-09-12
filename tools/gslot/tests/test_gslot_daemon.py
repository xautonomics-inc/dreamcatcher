"""End-to-end tests for the gslot daemon over a real unix socket.

These exercise the wire protocol and the connection lifecycle -- the parts the
pure-logic tests in ``test_gslot_arbiter.py`` cannot reach.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from gslot.arbiter import Arbiter, Resource, Tenant, cpu_resource
from gslot.client import Client, GslotError
from gslot.daemon import Server, parse_cpu_list
from gslot.topology import HostInventory


def _inventory() -> HostInventory:
    return HostInventory(
        host="test", taken_at=0.0, cpus=(), gpus=(), mem_total_bytes=0, mem_available_bytes=0
    )


@pytest.fixture
async def server(tmp_path: Path) -> AsyncIterator[tuple[Server, Path]]:
    arb = Arbiter(hb_timeout_s=30.0)
    arb.add_resource(cpu_resource("cpu:host", {(0, i): (i, i + 8) for i in range(8)}))
    arb.add_resource(Resource(rid="gpu:test", kind="gpu", concurrency=1, quantum_ms=250.0))
    sock = tmp_path / "gslotd.sock"
    srv = Server(arb, _inventory(), socket_path=sock, http_addr=None)
    task = asyncio.create_task(srv.run())
    for _ in range(200):
        if sock.exists():
            break
        await asyncio.sleep(0.01)
    yield srv, sock
    srv.stop()
    await asyncio.wait_for(task, timeout=5.0)


def _raw(sock: Path, *msgs: dict[str, object]) -> list[dict[str, object]]:
    """Blocking round trip on a worker thread, so the loop keeps serving."""
    out: list[dict[str, object]] = []
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect(str(sock))
    buf = b""
    try:
        for m in msgs:
            s.sendall((json.dumps(m) + "\n").encode())
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    raise AssertionError("server closed")
                buf += chunk
            line, _, buf = buf.partition(b"\n")
            out.append(json.loads(line))
    finally:
        s.close()
    return out


async def _in_thread(fn, *a):  # type: ignore[no-untyped-def]
    return await asyncio.get_running_loop().run_in_executor(None, fn, *a)


async def test_register_returns_a_partition(server: tuple[Server, Path]) -> None:
    _, sock = server
    (reply,) = await _in_thread(
        lambda s: _raw(
            s,
            {"id": 1, "op": "register", "tenant": "A", "resource": "cpu:host", "mode": "partition"},
        ),
        sock,
    )
    assert reply["ok"] is True
    assert len(reply["cpus"]) == 16  # all 8 cores, both siblings


async def test_disconnect_reclaims_capacity(server: tuple[Server, Path]) -> None:
    """A dropped connection is a stronger death signal than a heartbeat timeout.

    Waiting hb_timeout_s to notice would strand the tenant's cores for 30
    seconds after the process is already gone.
    """
    srv, sock = server
    await _in_thread(
        lambda s: _raw(
            s,
            {"id": 1, "op": "register", "tenant": "A", "resource": "cpu:host", "mode": "partition"},
        ),
        sock,
    )
    for _ in range(200):
        if not srv.arb.tenants:
            break
        await asyncio.sleep(0.01)
    assert srv.arb.tenants == {}, "closing the socket must unregister the tenant"


async def test_unknown_op_is_an_error_not_a_crash(server: tuple[Server, Path]) -> None:
    _, sock = server
    (reply,) = await _in_thread(lambda s: _raw(s, {"id": 7, "op": "nonsense"}), sock)
    assert reply == {"id": 7, "ok": False, "error": "unknown op 'nonsense'"}


async def test_malformed_frame_is_rejected_without_dropping_the_connection(
    server: tuple[Server, Path],
) -> None:
    _, sock = server

    def go(p: Path) -> list[dict[str, object]]:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(str(p))
        s.sendall(b"not json at all\n")
        buf = b""
        while b"\n" not in buf:
            buf += s.recv(4096)
        first = json.loads(buf.split(b"\n")[0])
        s.sendall(json.dumps({"id": 2, "op": "resources"}).encode() + b"\n")
        buf = buf.split(b"\n", 1)[1]
        while b"\n" not in buf:
            buf += s.recv(65536)
        second = json.loads(buf.split(b"\n")[0])
        s.close()
        return [first, second]

    first, second = await _in_thread(go, sock)
    assert first["ok"] is False
    assert second["ok"] is True, "one bad frame must not kill the session"


async def test_nowait_lease_reports_busy(server: tuple[Server, Path]) -> None:
    _, sock = server

    def go(p: Path) -> dict[str, object]:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(str(p))
        buf = b""

        def rt(m: dict[str, object]) -> dict[str, object]:
            nonlocal buf
            s.sendall((json.dumps(m) + "\n").encode())
            while b"\n" not in buf:
                buf += s.recv(65536)
            line, _, buf2 = buf.partition(b"\n")
            buf = buf2
            return json.loads(line)

        rt({"id": 1, "op": "register", "tenant": "A", "resource": "gpu:test", "mode": "turn"})
        rt({"id": 2, "op": "register", "tenant": "B", "resource": "gpu:test", "mode": "turn"})
        rt({"id": 3, "op": "lease", "tenant": "A", "est_ms": 100})
        out = rt({"id": 4, "op": "lease", "tenant": "B", "est_ms": 100, "nowait": True})
        s.close()
        return out

    reply = await _in_thread(go, sock)
    assert reply["ok"] is False
    assert reply["error"] == "busy"


async def test_observation_plane_is_read_only(tmp_path: Path) -> None:
    """The HTTP plane rejects every method but GET -- by having no other path."""
    arb = Arbiter()
    arb.add_resource(Resource(rid="gpu:test", kind="gpu"))
    srv = Server(arb, _inventory(), socket_path=tmp_path / "s.sock", http_addr=("127.0.0.1", 0))
    reader = asyncio.StreamReader()
    written: list[bytes] = []

    class W:
        def write(self, b: bytes) -> None:
            written.append(b)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

    reader.feed_data(b"POST /occupancy HTTP/1.1\r\nHost: x\r\n\r\n")
    reader.feed_eof()
    await srv._handle_http(reader, W())  # type: ignore[arg-type]
    body = b"".join(written)
    assert b"405 Method Not Allowed" in body
    assert b"read-only plane" in body


async def test_healthz_serves_wall_clock(tmp_path: Path) -> None:
    """/healthz carries server_time so consumers can stamp facts with the
    arbiter's own clock instead of arrival time (fleet-console freshness)."""
    srv = Server(Arbiter(), _inventory(), socket_path=tmp_path / "s.sock",
                 http_addr=("127.0.0.1", 0))
    reader = asyncio.StreamReader()
    written: list[bytes] = []

    class W:
        def write(self, b: bytes) -> None:
            written.append(b)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

    reader.feed_data(b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
    reader.feed_eof()
    await srv._handle_http(reader, W())  # type: ignore[arg-type]
    doc = json.loads(b"".join(written).split(b"\r\n\r\n", 1)[1])
    assert doc["ok"] is True
    assert abs(doc["server_time"] - time.time()) < 5.0


def test_parse_cpu_list() -> None:
    assert parse_cpu_list("0-3,8,12-15") == [0, 1, 2, 3, 8, 12, 13, 14, 15]
    assert parse_cpu_list("") == []


def test_client_fails_open_when_the_arbiter_is_absent(tmp_path: Path) -> None:
    """No arbiter must mean 'proceed uncoordinated', never 'stop'."""
    c = Client(str(tmp_path / "nothing.sock"), timeout=0.5)
    with pytest.raises(OSError):
        c.connect()
    c.tenant = "A"
    assert c.lease(10.0) is None, "an unreachable arbiter grants permission by default"
    c.release(None)


def test_registered_tenants_survive_a_heartbeat(tmp_path: Path) -> None:
    arb = Arbiter()
    arb.add_resource(cpu_resource("cpu:host", {(0, 0): (0,), (0, 1): (1,)}))
    arb.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    assert arb.heartbeat("A", progress=1) == "active"
    assert arb.reap() == []


@pytest.fixture
def _threads() -> Iterator[None]:
    before = threading.active_count()
    yield
    assert threading.active_count() <= before + 4


def test_client_error_type_is_public() -> None:
    assert issubclass(GslotError, Exception)


async def test_infeasible_claim_is_refused_with_the_math(tmp_path: Path) -> None:
    """An explicit refusal is a DEFINITE answer, and must read as one.

    The client fails open on transport failure -- an arbiter that is down must
    never stop inference. It must NOT fail open on a refusal: "this does not
    fit" is exactly the answer that stops a starting process from OOMing the
    one already serving. `gslot-run --require-admission` turns this reply into
    exit 75 without launching the child.
    """
    arb = Arbiter()
    arb.add_resource(Resource(rid="gpu:card", kind="gpu", dims={"vram_bytes": 20e9}))
    sock = tmp_path / "s.sock"
    srv = Server(arb, _inventory(), socket_path=sock, http_addr=None)
    task = asyncio.create_task(srv.run())
    for _ in range(200):
        if sock.exists():
            break
        await asyncio.sleep(0.01)
    try:
        first, second = await _in_thread(
            lambda s: _raw(
                s,
                {
                    "id": 1,
                    "op": "register",
                    "tenant": "incumbent",
                    "resource": "gpu:card",
                    "mode": "share",
                    "needs": {"vram_bytes": 12e9},
                },
                {
                    "id": 2,
                    "op": "register",
                    "tenant": "newcomer",
                    "resource": "gpu:card",
                    "mode": "turn",
                    "needs": {"vram_bytes": 12e9},
                },
            ),
            sock,
        )
        assert first["ok"] is True
        assert second["ok"] is False
        assert second["error"] == "infeasible claim"
        dims = second["report"]["dims"]["vram_bytes"]
        assert dims["already_committed"] == 12e9
        assert dims["headroom_after"] == -4e9, "the refusal must show the arithmetic"
        assert second["report"]["proof_required"] == "survived-prefill"
    finally:
        srv.stop()
        await asyncio.wait_for(task, timeout=5.0)
