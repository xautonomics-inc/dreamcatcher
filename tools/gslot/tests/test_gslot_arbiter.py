"""Unit tests for the gslot arbiter core (SPEC-016).

The clock is injected everywhere, so none of these sleep.
"""

from __future__ import annotations

import pytest

from gslot.arbiter import (
    Arbiter,
    Infeasible,
    Resource,
    Tenant,
    apportion,
    assign_units,
    cpu_resource,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def smt_cores(n: int = 16) -> dict[tuple[int, int], tuple[int, ...]]:
    """An SMT2 host shape: 16 physical cores, cpu i pairs with cpu i+16."""
    return {(0, i): (i, i + n) for i in range(n)}


def make(clock: FakeClock | None = None, reserved: list[tuple[int, int]] | None = None) -> Arbiter:
    a = Arbiter(clock=clock or FakeClock(), hb_timeout_s=30.0, stall_s=10.0)
    a.add_resource(cpu_resource("cpu:host", smt_cores(), reserved_cores=reserved or []))
    return a


# -- tier 1: partitioning ---------------------------------------------------


def test_partition_is_disjoint_and_whole_core() -> None:
    a = make()
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    a.register(Tenant(tid="B", rid="cpu:host", mode="partition"))
    ca, cb = set(a.tenants["A"].assigned), set(a.tenants["B"].assigned)
    assert not ca & cb, "partitions must not overlap"
    # Every granted core must arrive with BOTH its SMT siblings.  Half a core
    # is not a partition; it is the collision wearing a partition's clothes.
    for cpu in ca:
        assert (cpu + 16 if cpu < 16 else cpu - 16) in ca
    for cpu in cb:
        assert (cpu + 16 if cpu < 16 else cpu - 16) in cb


def test_partition_respects_reserved_cores() -> None:
    a = make(reserved=[(0, 0), (0, 1), (0, 2), (0, 3)])
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    granted = set(a.tenants["A"].assigned)
    for cpu in (0, 1, 2, 3, 16, 17, 18, 19):
        assert cpu not in granted, "reserved capacity must never be handed out"
    assert len(granted) == 24


def test_partition_is_weighted() -> None:
    a = make()
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition", weight=1.0))
    a.register(Tenant(tid="B", rid="cpu:host", mode="partition", weight=3.0))
    assert len(a.tenants["A"].assigned) == 8  # 4 cores
    assert len(a.tenants["B"].assigned) == 24  # 12 cores


def test_partition_is_stable_across_membership_change() -> None:
    """A surviving tenant keeps the cores it already had where it can.

    Re-pinning a ggml thread pool costs a warm-up on every graph, so churn is
    not free.  Continuity is part of the contract, same instinct as alias
    continuity in SPEC-016 requirement 6.
    """
    a = make()
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    a.register(Tenant(tid="B", rid="cpu:host", mode="partition"))
    before = set(a.tenants["A"].assigned)
    a.register(Tenant(tid="C", rid="cpu:host", mode="partition"))
    after = set(a.tenants["A"].assigned)
    assert after <= before, "a shrinking tenant must only lose cores, never move"


def test_partition_reclaims_on_unregister() -> None:
    a = make()
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    a.register(Tenant(tid="B", rid="cpu:host", mode="partition"))
    assert len(a.tenants["A"].assigned) == 16
    a.unregister("B")
    assert len(a.tenants["A"].assigned) == 32, "freed capacity must go back to survivors"


def test_partition_infeasible_shows_the_math() -> None:
    a = make()
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition", min_units=10))
    with pytest.raises(Infeasible) as ei:
        a.register(Tenant(tid="B", rid="cpu:host", mode="partition", min_units=10))
    assert ei.value.detail["allocatable_units"] == 16
    assert ei.value.detail["sum_min_units"] == 20
    assert ei.value.detail["per_tenant_min"] == {"A": 10, "B": 10}


def test_apportion_clamps_to_max() -> None:
    ts = [
        Tenant(tid="A", rid="r", mode="partition", weight=1.0, max_units=2),
        Tenant(tid="B", rid="r", mode="partition", weight=1.0),
    ]
    got = apportion(16, ts)
    assert got["A"] == 2
    assert got["B"] == 14
    assert sum(got.values()) == 16


def test_apportion_is_deterministic() -> None:
    ts = [Tenant(tid=t, rid="r", mode="partition") for t in ("A", "B", "C")]
    assert apportion(10, ts) == apportion(10, ts)


def test_assign_units_prefers_previously_held() -> None:
    units = [(i,) for i in range(8)]
    got = assign_units(units, {"A": 3, "B": 3}, {"A": [(5,), (6,), (7,)]})
    assert got["A"] == ((5,), (6,), (7,))
    assert set(got["B"]).isdisjoint(got["A"])


# -- tier 2: turns ----------------------------------------------------------


def turn_arb(clock: FakeClock) -> Arbiter:
    a = Arbiter(clock=clock)
    a.add_resource(Resource(rid="gpu:x", kind="gpu", concurrency=1, quantum_ms=250.0))
    a.register(Tenant(tid="A", rid="gpu:x", mode="turn"))
    a.register(Tenant(tid="B", rid="gpu:x", mode="turn"))
    return a


def test_turn_is_exclusive() -> None:
    c = FakeClock()
    a = turn_arb(c)
    _, g1 = a.request_lease("A", 50.0)
    _, g2 = a.request_lease("B", 50.0)
    assert g1 is True
    assert g2 is False, "concurrency 1 means exactly one tenant computes at a time"
    assert len(a.active_leases("gpu:x")) == 1
    assert a.queue_depth("gpu:x") == 1


def test_turn_incumbent_keeps_the_quantum() -> None:
    """A pipelining tenant runs consecutive waves inside one quantum.

    CLIENT CONTRACT: request the next turn BEFORE releasing the current one.
    A blocking client that releases first and asks second is not the incumbent
    at the moment the arbiter dispatches -- it has left the queue -- so the
    waiting tenant wins and the quantum buys nothing.  Keeping the pipeline
    full is the caller's job, exactly as it is for the ring's wave pump.
    """
    c = FakeClock()
    a = turn_arb(c)
    la, _ = a.request_lease("A", 50.0)
    a.request_lease("B", 50.0)
    c.advance(0.05)  # 50ms: well inside the 250ms quantum
    _, again = a.request_lease("A", 50.0)  # pipelined: ask before releasing
    a.release(la.lease_id)
    assert again is False, "concurrency 1: the pipelined request waits its turn"
    granted = a.release(la.lease_id)
    assert granted == [], "already released"
    # A's pipelined request is at the head because its quantum is unspent.
    assert a.active_leases("gpu:x")[0].tid == "A"
    assert a.stats["gpu:x"].switches == 1


def test_turn_hands_over_when_the_quantum_is_spent() -> None:
    c = FakeClock()
    a = turn_arb(c)
    la, _ = a.request_lease("A", 50.0)
    a.request_lease("B", 50.0)
    c.advance(0.30)  # past the 250ms quantum
    granted = a.release(la.lease_id)
    assert granted, "B must get the resource once A's quantum is spent"
    assert a.active_leases("gpu:x")[0].tid == "B"


def test_nowait_miss_leaves_no_trace() -> None:
    """A polling client that misses must not queue and must not be charged.

    A left-behind waiter is granted later to a caller that has moved on, so the
    turn is held by nobody until its TTL expires.  A left-behind virtual-finish
    advance charges the tenant for compute it never got, so the tenant that
    polls most often starves itself.  Both were real bugs.
    """
    c = FakeClock()
    a = turn_arb(c)
    a.request_lease("A", 50.0)
    before = a.tenants["B"].vfinish
    _, granted = a.request_lease("B", 50.0, nowait=True)
    assert granted is False
    assert a.queue_depth("gpu:x") == 0, "nowait must not leave a phantom waiter"
    assert a.tenants["B"].vfinish == before, "a miss must not cost virtual time"


def test_polling_tenant_is_not_starved() -> None:
    """The starvation bug, locked down.

    Two pure pollers modelled exactly as the C client behaves: hold the lease
    for a whole quantum answering locally, then release and re-ask.  Without
    demand tracking the incumbent renewed forever and the loser took zero turns
    in a 4-second run -- measured, not hypothetical.
    """
    c = FakeClock()
    a = turn_arb(c)
    hold: dict[str, int | None] = {"A": None, "B": None}
    until: dict[str, float] = {"A": 0.0, "B": 0.0}
    wins = {"A": 0, "B": 0}
    for _ in range(800):
        c.advance(0.005)
        for tid in ("A", "B"):
            held = hold[tid]
            if held is not None:
                if c.t < until[tid]:
                    continue  # inside our quantum: no RPC, just compute
                a.release(held)
                hold[tid] = None
            lease, granted = a.request_lease(tid, 250.0, nowait=True)
            if granted:
                hold[tid] = lease.lease_id
                until[tid] = c.t + 0.25
                wins[tid] += 1
    assert wins["A"] > 0 and wins["B"] > 0, f"one tenant was starved: {wins}"
    ratio = min(wins.values()) / max(wins.values())
    assert ratio > 0.5, f"turns should be roughly even, got {wins}"


def test_overrun_reclaims_the_turn_but_not_the_registration() -> None:
    c = FakeClock()
    a = turn_arb(c)
    a.request_lease("A", 10.0)  # ttl = max(10*4, 2000) = 2000ms
    a.request_lease("B", 10.0)
    c.advance(3.0)
    a.poll()
    assert a.stats["gpu:x"].overruns == 1
    assert "A" in a.tenants, "a slow tenant is not a dead tenant -- never evict on overrun"
    assert a.active_leases("gpu:x")[0].tid == "B"


# -- liveness ---------------------------------------------------------------


def test_states_discriminate_saturated_stalled_dead() -> None:
    c = FakeClock()
    a = make(c)
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    t = a.tenants["A"]
    assert t.state(c.t, hb_timeout_s=30.0, stall_s=10.0) == "active"
    c.advance(11.0)
    assert t.state(c.t, hb_timeout_s=30.0, stall_s=10.0) == "stalled", (
        "fresh heartbeat + frozen progress is stalled, not dead"
    )
    a.heartbeat("A", progress=5)
    assert t.state(c.t, hb_timeout_s=30.0, stall_s=10.0) == "active"
    c.advance(31.0)
    assert t.state(c.t, hb_timeout_s=30.0, stall_s=10.0) == "dead"


def test_reap_frees_capacity() -> None:
    c = FakeClock()
    a = make(c)
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    a.register(Tenant(tid="B", rid="cpu:host", mode="partition"))
    c.advance(40.0)
    a.heartbeat("A", progress=1)
    assert a.reap() == ["B"]
    assert len(a.tenants["A"].assigned) == 32, "the world model must decay explicitly"


# -- feasibility ------------------------------------------------------------


def test_feasibility_multi_dimension_math() -> None:
    a = Arbiter()
    a.add_resource(
        Resource(
            rid="gpu:card",
            kind="gpu",
            dims={"vram_bytes": 16e9, "host_ram_bytes": 100e9},
        )
    )
    a.register(Tenant(tid="resident", rid="gpu:card", mode="share", needs={"vram_bytes": 13.45e9}))
    rep = a.feasibility("gpu:card", {"vram_bytes": 4e9})
    assert rep["feasible"] is False
    assert rep["dims"]["vram_bytes"]["already_committed"] == 13.45e9
    assert rep["proof_required"] == "survived-prefill", (
        "arithmetic screens; only a survived prefill proves fit"
    )


def test_feasibility_reports_undeclared_dimensions_rather_than_guessing() -> None:
    a = Arbiter()
    a.add_resource(Resource(rid="gpu:card", kind="gpu", dims={"vram_bytes": 16e9}))
    rep = a.feasibility("gpu:card", {"membw_gbps": 40.0})
    assert rep["dims"]["membw_gbps"]["capacity"] is None
    assert rep["dims"]["membw_gbps"]["note"] == "undeclared dimension"


# -- reporting --------------------------------------------------------------


def test_occupancy_publishes_queue_depth_with_latency() -> None:
    """A queued request looks exactly like a slow one unless both are published."""
    c = FakeClock()
    a = turn_arb(c)
    a.request_lease("A", 50.0)
    a.request_lease("B", 50.0)
    occ = a.occupancy()["resources"]["gpu:x"]
    assert occ["queue_depth"] == 1
    assert occ["active"] == 1
    assert occ["run_owner"] == "A"
    assert occ["wait_ms_p99"] is not None


def test_turn_weights_are_honoured() -> None:
    """A weight-3 tenant gets ~3x the turns of a weight-1 tenant.

    Both tenants keep an outstanding request at all times, so the arbiter is
    always choosing between them -- polling one at a time only ever measures
    who asked first.

    This scenario keeps both tenants continuously backlogged, which is the case
    where the virtual clock's units do not matter -- an earlier wall-clock
    version scored the same 299:100 here.  The test still earns its place: it is
    the only thing standing between "weight is parsed, stored and reported" and
    "weight changes who runs".
    """
    c = FakeClock()
    a = Arbiter(clock=c)
    a.add_resource(Resource(rid="gpu:x", kind="gpu", concurrency=1, quantum_ms=0.0))
    a.register(Tenant(tid="light", rid="gpu:x", mode="turn", weight=1.0))
    a.register(Tenant(tid="heavy", rid="gpu:x", mode="turn", weight=3.0))
    wins = {"light": 0, "heavy": 0}
    outstanding: dict[str, bool] = {"light": False, "heavy": False}
    for _ in range(400):
        c.advance(0.01)
        act = a.active_leases("gpu:x")
        if act:
            wins[act[0].tid] += 1
            outstanding[act[0].tid] = False
        # Re-arm BOTH before releasing: the dispatch that follows a release must
        # have a real choice, or the loop measures alternation rather than weight.
        for tid in ("light", "heavy"):
            if not outstanding[tid]:
                a.request_lease(tid, 100.0)
                outstanding[tid] = True
        if act:
            a.release(act[0].lease_id)
    ratio = wins["heavy"] / max(wins["light"], 1)
    assert 2.0 < ratio < 4.5, f"expected ~3:1 by weight, got {wins}"


def test_a_tenant_holding_a_lease_is_not_reaped() -> None:
    """Turn traffic is liveness evidence; a long turn must not read as death.

    Regression, caught in a live run: the request-granularity turn arm issued
    12 requests and the arbiter recorded only 3 grants.  Requests ran ~20 s and
    hb_timeout_s was 30 s, so both drivers were reaped mid-arm and every later
    lease came back "unknown tenant".  The client failed open, exactly as
    designed, so the arm completed and looked fine -- the only trace was the
    grant count.
    """
    c = FakeClock()
    a = Arbiter(clock=c, hb_timeout_s=30.0)
    a.add_resource(Resource(rid="gpu:x", kind="gpu", concurrency=1, quantum_ms=250.0))
    a.register(Tenant(tid="A", rid="gpu:x", mode="turn"))
    a.register(Tenant(tid="B", rid="gpu:x", mode="turn"))
    held, _ = a.request_lease("A", 8000.0)
    for _ in range(4):
        c.advance(20.0)  # a 20-second turn, with no explicit heartbeat
        a.touch("A")
        a.touch("B")
        assert a.reap() == [], "a tenant mid-turn is alive, not dead"
        a.release(held.lease_id)
        held, _ = a.request_lease("A", 8000.0)
    assert set(a.tenants) == {"A", "B"}


def test_a_long_turn_is_active_not_stalled() -> None:
    """Silence inside a tenant's own lease TTL is expected, not suspicious.

    Seen live: both drivers on a 20-second turn reported `stalled`, because the
    stall window was 10 s and the arbiter's only progress signal was the grant
    itself. A liveness ladder that cries wolf on the healthy case is worse than
    none. A genuine hang still surfaces -- the TTL expires, the turn is
    reclaimed as an overrun, and normal stall detection resumes.
    """
    c = FakeClock()
    a = Arbiter(clock=c, hb_timeout_s=300.0, stall_s=10.0)
    a.add_resource(Resource(rid="gpu:x", kind="gpu", concurrency=1, quantum_ms=250.0))
    a.register(Tenant(tid="A", rid="gpu:x", mode="turn"))
    a.request_lease("A", 8000.0)  # ttl = 32 s
    c.advance(20.0)
    a.touch("A")
    assert a.occupancy()["resources"]["gpu:x"]["tenants"]["A"]["state"] == "active"
    c.advance(20.0)  # past the 32 s TTL: the turn is reclaimed...
    a.poll()
    a.touch("A")
    assert a.stats["gpu:x"].overruns == 1
    # ...and with nothing held, the frozen progress counter shows through again.
    assert a.occupancy()["resources"]["gpu:x"]["tenants"]["A"]["state"] == "stalled"


def hybrid_cores() -> tuple[dict[tuple[int, int], tuple[int, ...]], dict[tuple[int, int], str]]:
    """A real hybrid part: Core Ultra 7 265K, 8 P + 12 E, no SMT.

    The ``core_id`` values deliberately interleave the classes the way the real
    part does (P at 0,8,16,24,...; E at 12,20,28,...), because dealing in id
    order is exactly what used to hand out an unequal mix.
    """
    cores: dict[tuple[int, int], tuple[int, ...]] = {}
    kinds: dict[tuple[int, int], str] = {}
    for cpu in range(8):  # P-cores, ids 0,8,16,...
        cores[(0, cpu * 8)] = (cpu,)
        kinds[(0, cpu * 8)] = "P"
    for i, cpu in enumerate(range(8, 20)):  # E-cores, ids 12,20,28,...
        cores[(0, 12 + i * 8)] = (cpu,)
        kinds[(0, 12 + i * 8)] = "E"
    return cores, kinds


def hybrid_arb() -> Arbiter:
    cores, kinds = hybrid_cores()
    a = Arbiter()
    a.add_resource(cpu_resource("cpu:host", cores, kinds=kinds))
    return a


def classes_of(a: Arbiter, tid: str) -> dict[str, int]:
    res = a.resources["cpu:host"]
    out: dict[str, int] = {}
    for u in a.tenants[tid].assigned_units:
        k = res.class_of(u)
        out[k] = out.get(k, 0) + 1
    return out


def test_hybrid_partition_is_proportional_by_default() -> None:
    """Equal weights must mean an equal MIX, not just an equal count.

    Without class awareness the solver deals in core_id order, which on this
    part interleaves P and E arbitrarily -- so two tenants get "ten cores each"
    and one set is up to 34% faster than the other.
    """
    a = hybrid_arb()
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    a.register(Tenant(tid="B", rid="cpu:host", mode="partition"))
    ca, cb = classes_of(a, "A"), classes_of(a, "B")
    assert ca == {"P": 4, "E": 6}
    assert cb == {"P": 4, "E": 6}
    assert not set(a.tenants["A"].assigned) & set(a.tenants["B"].assigned)


def test_hybrid_prefer_class_takes_the_fast_cores() -> None:
    a = hybrid_arb()
    a.register(Tenant(tid="hot", rid="cpu:host", mode="partition", weight=3.0, prefer_class="P"))
    a.register(Tenant(tid="cold", rid="cpu:host", mode="partition", weight=1.0))
    assert classes_of(a, "hot")["P"] == 8, "a P preference must exhaust P before taking E"
    assert classes_of(a, "cold").get("P", 0) == 0
    assert not set(a.tenants["hot"].assigned) & set(a.tenants["cold"].assigned)


def test_expert_server_recipe_reproduces_the_hand_tuned_layout() -> None:
    """The hybrid P/E recipe in docs/runbook-gslot-expert-server.md, asserted.

    The expert-server workstream's hand-written cpusets today are
    ``expert-server 0-15`` (8 P + 8 E) and ``client 16-19`` (4 E). A recipe the
    other lead can apply verbatim has to produce exactly that, or it is a
    suggestion rather than a recipe.
    """
    a = hybrid_arb()
    a.register(
        Tenant(
            tid="expert-server",
            rid="cpu:host",
            mode="partition",
            min_units=16,
            max_units=16,
            prefer_class="P",
        )
    )
    a.register(Tenant(tid="front", rid="cpu:host", mode="partition", min_units=4, max_units=4))
    assert classes_of(a, "expert-server") == {"P": 8, "E": 8}
    assert classes_of(a, "front") == {"E": 4}
    assert len(a.tenants["expert-server"].assigned) == 16
    assert len(a.tenants["front"].assigned) == 4


def test_uniform_host_is_unaffected_by_class_logic() -> None:
    a = make()
    a.register(Tenant(tid="A", rid="cpu:host", mode="partition"))
    a.register(Tenant(tid="B", rid="cpu:host", mode="partition"))
    assert len(a.tenants["A"].assigned) == 16
    assert len(a.tenants["B"].assigned) == 16
    occ = a.occupancy()["resources"]["cpu:host"]
    assert occ["allocatable_by_class"] == {"uniform": 16}
