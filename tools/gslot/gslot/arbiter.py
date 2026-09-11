"""The arbiter core: pure scheduling logic, no I/O, deterministic clock.

Two tiers, because the two real collisions on this fleet have different shapes.

**Tier 1 -- partition leases (zero steady-state cost).**
Two co-resident ggml CPU tenants each sized for the whole machine do not
time-share it; they *destroy* it.  While thread 0 of tenant A blocks, A's other
workers spin at the intra-graph node barrier and evict B's workers.  Measured on
amd during expert-disagg P1: 4.409 tok/s partitioned vs 1.2 tok/s naive, a 3.7x
collapse, with per-call latency going 0.7ms -> 8.7ms.  No amount of per-wave
signalling fixes that, because the damage happens *inside* one graph.  The fix
is a disjoint core set, computed once and applied by the tenant to itself.

**Tier 2 -- turn leases (per-wave RPC).**
Two tenants sharing a GPU interleave at kernel-dispatch granularity, which is
fair and slow: caches thrash, and every tenant's p99 inherits every other
tenant's queue.  The fix is a coarse quantum -- each tenant gets an exclusive
run of a few hundred milliseconds, so a wave completes without interruption.
Weighted fair queuing keeps that from becoming starvation.

Both tiers are opt-in.  A process that never registers is never gated; it is at
most declared as a ``share`` tenant so the arbiter does not hand out capacity it
does not own.  SPEC-016 requirement 10 (no silent actuation) is structural here:
the arbiter has no mechanism to touch an unregistered process.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gslot.protocol import Mode, TenantState

Clock = Callable[[], float]

DEFAULT_QUANTUM_MS = 250.0
DEFAULT_TTL_FACTOR = 4.0
DEFAULT_MIN_TTL_MS = 2_000.0
DEFAULT_HB_TIMEOUT_S = 30.0
DEFAULT_STALL_S = 10.0


class Infeasible(Exception):
    """A claim cannot be satisfied, with the multi-resource math attached.

    SPEC-016 requirements 1/2/4: VRAM alone is the wrong unit, host RAM is
    first-class, and a placement is only feasible when *every* dimension fits.
    We refuse with the arithmetic shown rather than silently over-committing --
    but note the arithmetic is a *screen*, not a proof.  The only accepted proof
    of feasibility on this fleet is a survived prefill.
    """

    def __init__(self, reason: str, detail: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Resource:
    """A named compute domain the arbiter schedules.

    ``units`` are the partitionable atoms.  For a CPU resource one unit is one
    *physical core* (a tuple of its logical CPUs), because splitting SMT
    siblings across tenants is a collision, not a partition.  For a GPU there
    are no partitionable units -- only turns.

    ``dims`` are advisory capacity dimensions (``vram_bytes``, ``host_ram_bytes``,
    ...) used by the feasibility screen.  They are checked, never enforced.
    """

    rid: str
    kind: str  # "cpu" | "gpu" | "generic"
    units: tuple[tuple[int, ...], ...] = ()
    unit_classes: tuple[str, ...] = ()  # aligned with `units`; "" == uniform
    reserved_units: tuple[tuple[int, ...], ...] = ()
    concurrency: int = 1
    quantum_ms: float = DEFAULT_QUANTUM_MS
    dims: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def allocatable(self) -> tuple[tuple[int, ...], ...]:
        reserved = set(self.reserved_units)
        return tuple(u for u in self.units if u not in reserved)

    def class_of(self, unit: tuple[int, ...]) -> str:
        """Core class of one unit ("P"/"E"/""), by position in ``units``."""
        try:
            idx = self.units.index(unit)
        except ValueError:
            return ""
        return self.unit_classes[idx] if idx < len(self.unit_classes) else ""

    def allocatable_by_class(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for u in self.allocatable:
            k = self.class_of(u) or "uniform"
            out[k] = out.get(k, 0) + 1
        return out


# --------------------------------------------------------------------------
# tenants
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Tenant:
    tid: str
    rid: str
    mode: Mode
    weight: float = 1.0
    pid: int | None = None
    min_units: int = 1
    max_units: int | None = None
    prefer_class: str | None = None  # "P"/"E": fill from this core class first
    needs: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # measured liveness (SPEC-016 req 7 + 9)
    registered_at: float = 0.0
    last_hb: float = 0.0
    progress: int = 0
    last_progress_at: float = 0.0

    # tier-1 state
    assigned: tuple[int, ...] = ()
    assigned_units: tuple[tuple[int, ...], ...] = ()
    epoch: int = 0

    # tier-2 accounting
    vfinish: float = 0.0
    demand_at: float = -1e9
    grants: int = 0
    overruns: int = 0
    busy_ms: float = 0.0
    wait_ms_total: float = 0.0
    waits: list[float] = field(default_factory=list)

    def state(
        self, now: float, *, hb_timeout_s: float, stall_s: float, holding: bool = False
    ) -> TenantState:
        """Classify from measured signals only.  Never from a health probe.

        A dead tenant is one whose *heartbeat* went stale -- that is the only
        signal that means "this process is gone".  A tenant whose heartbeat is
        fresh but whose progress counter is frozen is ``stalled``: it is alive
        and doing nothing useful, which is a different fault with a different
        remedy.  ``saturated`` is the healthy-but-slow case that gets mistaken
        for both (SPEC-013 section 4.0; cost us 90 minutes on 08-18 when
        ``/health`` returned 200 through a dead ring).
        """
        if now - self.last_hb > hb_timeout_s:
            return "dead"
        if holding:
            # Mid-turn.  The tenant asked for the resource and got it, and its
            # lease TTL has not run out -- that TTL is the tenant's own declared
            # bound on how long it will hold without reporting, so silence
            # inside it is expected rather than suspicious.  Calling a
            # 20-second turn "stalled" because the stall window is 10 seconds
            # is a false signal, and a liveness ladder that cries wolf is worse
            # than none.  A genuine hang still surfaces: the TTL expires, the
            # turn is reclaimed as an overrun, holding goes false, and the
            # normal stall logic resumes.
            return "active"
        if now - self.last_progress_at > stall_s:
            return "stalled"
        if self.waits and self.wait_ms_total > self.busy_ms:
            return "saturated"
        return "active"


@dataclass(slots=True)
class Lease:
    lease_id: int
    tid: str
    rid: str
    est_ms: float
    requested_at: float
    granted_at: float = 0.0
    expires_at: float = 0.0
    vstart: float = 0.0  # virtual-time stamp; NOT wall clock (see _dispatch)


@dataclass(slots=True)
class _Waiter:
    lease: Lease
    vstart: float
    seq: int


# --------------------------------------------------------------------------
# tier 1: the partition solver
# --------------------------------------------------------------------------


def apportion(
    n_units: int,
    tenants: Sequence[Tenant],
) -> dict[str, int]:
    """Weighted apportionment of ``n_units`` atoms, clamped to each [min,max].

    Largest-remainder (Hamilton) on the *whole* capacity, then a repair pass for
    tenants whose clamp pushed them off quota.  Apportioning only the surplus
    above the minimums looks equivalent and is not: with a floor of one core
    each, a 1:3 weight split of 16 cores comes out 5:11 instead of 4:12, because
    the two floor cores were handed out evenly before the weights were applied.

    Deterministic: ties break on tenant id, so identical inputs always give an
    identical partition.  An unstable partition churns ggml thread pools for
    nothing.

    Raises:
        Infeasible: if the minimums alone exceed capacity.
    """
    if not tenants:
        return {}
    total_min = sum(t.min_units for t in tenants)
    if total_min > n_units:
        raise Infeasible(
            "cpu partition infeasible: minimums exceed allocatable cores",
            {
                "allocatable_units": n_units,
                "sum_min_units": total_min,
                "per_tenant_min": {t.tid: t.min_units for t in tenants},
            },
        )

    def cap(t: Tenant) -> int:
        return min(t.max_units if t.max_units is not None else n_units, n_units)

    wsum = sum(t.weight for t in tenants) or float(len(tenants))
    quota = {t.tid: n_units * (t.weight / wsum) for t in tenants}
    alloc = {t.tid: max(t.min_units, min(cap(t), math.floor(quota[t.tid]))) for t in tenants}

    # Repair upward: hand out what is left to whoever is furthest below quota.
    while sum(alloc.values()) < n_units:
        cands = [t for t in tenants if alloc[t.tid] < cap(t)]
        if not cands:
            break
        pick = min(cands, key=lambda t: (-(quota[t.tid] - alloc[t.tid]), t.tid))
        alloc[pick.tid] += 1

    # Repair downward: clamping upward can overshoot capacity.
    while sum(alloc.values()) > n_units:
        cands = [t for t in tenants if alloc[t.tid] > t.min_units]
        if not cands:
            break
        pick = min(cands, key=lambda t: (-(alloc[t.tid] - quota[t.tid]), t.tid))
        alloc[pick.tid] -= 1

    return alloc


def assign_units(
    units: Sequence[tuple[int, ...]],
    counts: Mapping[str, int],
    previous: Mapping[str, Sequence[tuple[int, ...]]],
    classes: Mapping[tuple[int, ...], str] | None = None,
    prefer: Mapping[str, str | None] | None = None,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Turn per-tenant unit *counts* into concrete unit sets.

    Runs in two phases: decide how many units of each CLASS each tenant gets,
    then pick which concrete units satisfy that.

    **Preference** is honoured first.  A tenant may name a core class to fill
    from before anything else.  On a hybrid host "give the expert-server the
    P-cores" is a real and correct placement decision, and the scheduler should
    express it rather than sending the operator back to hand-written cpusets.

    **Proportionality** then splits each remaining class among the tenants in
    proportion to what they still need.  Without it, dealing in ``core_id``
    order on a hybrid part gives one tenant a silently faster set --
    the ids interleave P and E, so "ten cores each" is not a fair partition
    when one tenant's ten include all eight P-cores.

    **Stability** is applied *within* a class, not across the pool.  A tenant
    keeps as many of its previously-held units of each class as its new target
    for that class allows.  Applying stability first instead looks like the
    same rule and is not: a tenant that had run alone holds every unit, so
    "keep the first N you held" hands it back a P-heavy subset and quietly
    undoes the proportional split.  Re-pinning a ggml thread pool costs a
    warm-up on every graph, so churn is still avoided -- just never at the cost
    of a biased share.
    """
    kls: Mapping[tuple[int, ...], str] = classes or {}
    pref: Mapping[str, str | None] = prefer or {}

    def cls(u: tuple[int, ...]) -> str:
        return kls.get(u, "")

    avail: dict[str, list[tuple[int, ...]]] = {}
    for u in units:
        avail.setdefault(cls(u), []).append(u)

    need = {tid: counts[tid] for tid in sorted(counts)}
    target: dict[str, dict[str, int]] = {tid: {} for tid in need}
    left = {k: len(v) for k, v in avail.items()}

    # phase 1a: preferences
    for tid in sorted(need):
        want_cls = pref.get(tid)
        if not want_cls or want_cls not in left:
            continue
        n = min(need[tid], left[want_cls])
        if n:
            target[tid][want_cls] = target[tid].get(want_cls, 0) + n
            need[tid] -= n
            left[want_cls] -= n

    # phase 1b: proportional, scarcest class first so a class that runs out is
    # divided fairly before the abundant ones mop up the remainder.
    for k in sorted(left, key=lambda k: (left[k], k)):
        claimants = [t for t in sorted(need) if need[t] > 0]
        total_need = sum(need[t] for t in claimants)
        if not claimants or not total_need or left[k] <= 0:
            continue
        pool = left[k]
        quota = {t: pool * need[t] / total_need for t in claimants}
        give = {t: min(math.floor(quota[t]), need[t]) for t in claimants}
        while sum(give.values()) < min(pool, total_need):
            cands = [t for t in claimants if give[t] < need[t]]
            if not cands:
                break
            t = min(cands, key=lambda t: (-(quota[t] - give[t]), t))
            give[t] += 1
        for t, n in give.items():
            if not n:
                continue
            target[t][k] = target[t].get(k, 0) + n
            need[t] -= n
            left[k] -= n

    # phase 2: materialise, preferring units this tenant already held
    free: dict[str, list[tuple[int, ...]]] = {k: list(v) for k, v in avail.items()}
    out: dict[str, list[tuple[int, ...]]] = {tid: [] for tid in counts}
    for tid in sorted(counts):
        held = [u for u in previous.get(tid, ()) if u in free.get(cls(u), ())]
        for k, n in sorted(target[tid].items()):
            mine = [u for u in held if cls(u) == k][:n]
            for u in mine:
                free[k].remove(u)
            while len(mine) < n and free[k]:
                mine.append(free[k].pop(0))
            out[tid].extend(mine)
    return {tid: tuple(sorted(out[tid])) for tid in sorted(counts)}


# --------------------------------------------------------------------------
# the arbiter
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ResourceStats:
    grants: int = 0
    switches: int = 0
    overruns: int = 0
    yields: int = 0
    busy_ms: float = 0.0
    granted_at_first: float = 0.0
    waits: list[float] = field(default_factory=list)


class Arbiter:
    """Owns resources, tenants, partitions and turn queues for one host."""

    def __init__(
        self,
        *,
        clock: Clock = time.monotonic,
        hb_timeout_s: float = DEFAULT_HB_TIMEOUT_S,
        stall_s: float = DEFAULT_STALL_S,
        ttl_factor: float = DEFAULT_TTL_FACTOR,
        min_ttl_ms: float = DEFAULT_MIN_TTL_MS,
    ) -> None:
        self.clock = clock
        self.hb_timeout_s = hb_timeout_s
        self.stall_s = stall_s
        self.ttl_factor = ttl_factor
        self.min_ttl_ms = min_ttl_ms

        self.resources: dict[str, Resource] = {}
        self.tenants: dict[str, Tenant] = {}
        self.stats: dict[str, ResourceStats] = {}

        self._queues: dict[str, deque[_Waiter]] = {}
        self._active: dict[str, dict[int, Lease]] = {}
        self._vtime: dict[str, float] = {}
        self._run_owner: dict[str, str | None] = {}
        self._run_started: dict[str, float] = {}
        self._leases: dict[int, Lease] = {}
        self._lease_seq = itertools.count(1)
        self._wait_seq = itertools.count(1)
        self.epoch = 0

    # -- resources ---------------------------------------------------------

    def add_resource(self, res: Resource) -> None:
        self.resources[res.rid] = res
        self.stats.setdefault(res.rid, ResourceStats())
        self._queues.setdefault(res.rid, deque())
        self._active.setdefault(res.rid, {})
        self._vtime.setdefault(res.rid, 0.0)
        self._run_owner.setdefault(res.rid, None)
        self._run_started.setdefault(res.rid, 0.0)

    # -- feasibility screen -------------------------------------------------

    def feasibility(self, rid: str, needs: Mapping[str, float]) -> dict[str, Any]:
        """Check a claim against every declared dimension and show the math.

        Returns a report; raises nothing.  ``feasible`` false means "do not
        bother trying"; ``feasible`` true means "worth attempting" -- it is NOT
        a promise, because arithmetic has never been proof of fit on this
        fleet.  The report carries ``proof_required`` to say so.
        """
        res = self.resources.get(rid)
        if res is None:
            return {"feasible": False, "reason": f"unknown resource {rid!r}"}
        committed: dict[str, float] = {}
        for t in self.tenants.values():
            if t.rid != rid:
                continue
            for k, v in t.needs.items():
                committed[k] = committed.get(k, 0.0) + v
        dims: dict[str, Any] = {}
        feasible = True
        for key, want in needs.items():
            cap = res.dims.get(key)
            if cap is None:
                dims[key] = {"want": want, "capacity": None, "note": "undeclared dimension"}
                continue
            used = committed.get(key, 0.0)
            fits = used + want <= cap
            feasible = feasible and fits
            dims[key] = {
                "want": want,
                "already_committed": used,
                "capacity": cap,
                "headroom_after": cap - used - want,
                "fits": fits,
            }
        return {
            "resource": rid,
            "feasible": feasible,
            "dims": dims,
            "proof_required": "survived-prefill",
            "note": (
                "arithmetic is a screen, not a proof: weights + KV(ctx,quant) + compute "
                "buffers must be validated by an actual prefill that survived"
            ),
        }

    # -- registration -------------------------------------------------------

    def register(self, tenant: Tenant) -> Tenant:
        res = self.resources.get(tenant.rid)
        if res is None:
            raise Infeasible(f"unknown resource {tenant.rid!r}", {"known": sorted(self.resources)})
        now = self.clock()
        tenant.registered_at = now
        tenant.last_hb = now
        tenant.last_progress_at = now
        if tenant.mode == "turn":
            tenant.vfinish = self._vtime[tenant.rid]
        self.tenants[tenant.tid] = tenant
        if tenant.mode == "partition":
            self.repartition(tenant.rid)
        return tenant

    def unregister(self, tid: str) -> None:
        t = self.tenants.pop(tid, None)
        if t is None:
            return
        for lid, lease in list(self._active.get(t.rid, {}).items()):
            if lease.tid == tid:
                self.release(lid)
        q = self._queues.get(t.rid)
        if q is not None:
            self._queues[t.rid] = deque(w for w in q if w.lease.tid != tid)
        if self._run_owner.get(t.rid) == tid:
            self._run_owner[t.rid] = None
        if t.mode == "partition":
            self.repartition(t.rid)

    def touch(self, tid: str) -> None:
        """Record that we just heard from this tenant.

        ANY frame on the control connection is evidence the process is alive --
        a `lease` or a `release` no less than an explicit heartbeat.  Without
        this, a tenant whose turns are longer than hb_timeout_s gets reaped
        *while it is holding a lease*: measured on the request-granularity turn
        arm, 12 requests produced only 3 grants because both drivers were
        declared dead ~30 s in and every later lease came back "unknown tenant"
        (which the client, correctly, fails open on -- so the arm silently ran
        uncoordinated and looked like it had worked).

        Progress is deliberately NOT advanced here.  Being alive and making
        progress are different claims, and collapsing them would erase the
        stalled state, which is the one that matters (SPEC-013 s4.0).
        """
        t = self.tenants.get(tid)
        if t is not None:
            t.last_hb = self.clock()

    def heartbeat(self, tid: str, progress: int | None = None) -> TenantState:
        t = self.tenants.get(tid)
        if t is None:
            raise Infeasible(f"unknown tenant {tid!r}", {})
        now = self.clock()
        t.last_hb = now
        if progress is not None and progress > t.progress:
            t.progress = progress
            t.last_progress_at = now
        return t.state(now, hb_timeout_s=self.hb_timeout_s, stall_s=self.stall_s)

    def _holders(self) -> set[str]:
        return {lease.tid for leases in self._active.values() for lease in leases.values()}

    def reap(self) -> list[str]:
        """Drop tenants whose heartbeat went stale; return their ids.

        This is the world model decaying *explicitly* (SPEC-016 req 7).  A
        registration is a lease on the arbiter's attention, not a permanent
        entry in a table nobody re-reads.
        """
        now = self.clock()
        holders = self._holders()
        dead = [
            tid
            for tid, t in self.tenants.items()
            if t.state(
                now,
                hb_timeout_s=self.hb_timeout_s,
                stall_s=self.stall_s,
                holding=tid in holders,
            )
            == "dead"
        ]
        for tid in dead:
            self.unregister(tid)
        return dead

    # -- tier 1 -------------------------------------------------------------

    def repartition(self, rid: str) -> dict[str, tuple[int, ...]]:
        """Recompute the disjoint core sets for every partition tenant."""
        res = self.resources[rid]
        parts = sorted(
            (t for t in self.tenants.values() if t.rid == rid and t.mode == "partition"),
            key=lambda t: t.tid,
        )
        if not parts:
            return {}
        units = res.allocatable
        counts = apportion(len(units), parts)
        previous = {t.tid: t.assigned_units for t in parts}
        assigned = assign_units(
            units,
            counts,
            previous,
            classes={u: res.class_of(u) for u in units},
            prefer={t.tid: t.prefer_class for t in parts},
        )
        self.epoch += 1
        out: dict[str, tuple[int, ...]] = {}
        for t in parts:
            us = assigned[t.tid]
            cpus = tuple(sorted(c for u in us for c in u))
            if cpus != t.assigned:
                t.epoch = self.epoch
            t.assigned_units = us
            t.assigned = cpus
            out[t.tid] = cpus
        return out

    # -- tier 2 -------------------------------------------------------------

    def request_lease(self, tid: str, est_ms: float, *, nowait: bool = False) -> tuple[Lease, bool]:
        """Enqueue a turn request.  Returns ``(lease, granted_now)``.

        Weighted fair queuing over virtual time: a tenant that has consumed more
        than its weighted share has a later virtual finish, so it yields.  The
        quantum then lets whoever currently holds the resource keep it for a
        coarse run, which is the whole point -- alternating in 250ms blocks
        beats alternating per kernel.

        ``nowait`` is the polling caller (the C client in the stage-runner pump,
        which cannot block the one thread that owns all ring I/O).  A nowait
        miss must leave *no trace*: the waiter is removed and the tenant's
        virtual finish is rolled back.  Both halves matter --

        * a left-behind waiter is granted later to a caller that has long since
          moved on, so the turn is held by nobody until its TTL expires and the
          resource deadlocks in slow motion;
        * a left-behind ``vfinish`` advance charges the tenant for compute it
          never received, so the tenant that polls most often starves itself.
        """
        t = self.tenants.get(tid)
        if t is None:
            raise Infeasible(f"unknown tenant {tid!r}", {})
        if t.mode != "turn":
            raise Infeasible(f"tenant {tid!r} is mode {t.mode}, not 'turn'", {})
        now = self.clock()
        rid = t.rid
        t.demand_at = now
        self._expire(rid, now)
        lease = Lease(
            lease_id=next(self._lease_seq),
            tid=tid,
            rid=rid,
            est_ms=max(0.0, est_ms),
            requested_at=now,
        )
        self._leases[lease.lease_id] = lease
        vfinish_before = t.vfinish
        vstart = max(t.vfinish, self._vtime[rid])
        t.vfinish = vstart + lease.est_ms / max(t.weight, 1e-9)
        waiter = _Waiter(lease, vstart, next(self._wait_seq))
        self._queues[rid].append(waiter)
        granted = self._dispatch(rid, now)
        if lease.lease_id in granted:
            return lease, True
        if nowait:
            self.cancel_lease(lease.lease_id)
            t.vfinish = vfinish_before
        return lease, False

    def cancel_lease(self, lease_id: int) -> bool:
        """Withdraw a queued (not yet granted) request.  Idempotent."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return False
        q = self._queues.get(lease.rid)
        if q is not None:
            for w in list(q):
                if w.lease.lease_id == lease_id:
                    q.remove(w)
        if lease_id not in self._active.get(lease.rid, {}):
            self._leases.pop(lease_id, None)
            return True
        return False

    def release(self, lease_id: int) -> list[int]:
        """Finish a turn.  Returns lease ids granted as a consequence."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return []
        now = self.clock()
        rid = lease.rid
        active = self._active.get(rid, {})
        if lease_id in active:
            del active[lease_id]
            busy = (now - lease.granted_at) * 1000.0
            self.stats[rid].busy_ms += busy
            t = self.tenants.get(lease.tid)
            if t is not None:
                t.busy_ms += busy
        self._leases.pop(lease_id, None)
        return self._dispatch(rid, now)

    def poll(self) -> dict[str, list[int]]:
        """Advance every resource (expiry + dispatch).  Returns new grants."""
        now = self.clock()
        out: dict[str, list[int]] = {}
        for rid in self.resources:
            self._expire(rid, now)
            granted = self._dispatch(rid, now)
            if granted:
                out[rid] = granted
        return out

    def _expire(self, rid: str, now: float) -> None:
        for lid, lease in list(self._active.get(rid, {}).items()):
            if lease.expires_at and now >= lease.expires_at:
                del self._active[rid][lid]
                self._leases.pop(lid, None)
                self.stats[rid].overruns += 1
                t = self.tenants.get(lease.tid)
                if t is not None:
                    t.overruns += 1
                # An overrun is NOT a death sentence.  The tenant keeps its
                # registration; we only take the turn back.  Killing a process
                # because it was slow is exactly the silent actuation
                # requirement 10 forbids.

    def _demand_from_others(self, rid: str, tid: str, now: float) -> bool:
        """Is some *other* turn tenant on this resource asking, right now?

        A polling client (the C client in the stage-runner pump, which must not
        block the thread that owns all ring I/O) is not in the queue when it
        loses -- it asked, missed, and went away.  Without a demand marker the
        arbiter cannot tell "nobody else wants this" from "somebody else asks
        every 5ms and never wins", and the incumbent keeps renewing forever.
        That is exactly how the first version of this starved one of two
        tenants to zero waves in a 4-second run.

        Demand decays: it is a statement about the last few hundred
        milliseconds, not a registration.  A tenant that dies mid-poll stops
        blocking the incumbent within one demand window.
        """
        ttl = self.resources[rid].quantum_ms * 4.0 / 1000.0
        return any(
            t.tid != tid and t.rid == rid and t.mode == "turn" and now - t.demand_at <= ttl
            for t in self.tenants.values()
        )

    def _dispatch(self, rid: str, now: float) -> list[int]:
        res = self.resources[rid]
        q = self._queues[rid]
        active = self._active[rid]
        granted: list[int] = []
        while q and len(active) < res.concurrency:
            owner = self._run_owner[rid]
            run_age_ms = (now - self._run_started[rid]) * 1000.0
            waiters = sorted(q, key=lambda w: (w.vstart, w.seq))
            pick: _Waiter | None = None
            if owner is not None and run_age_ms < res.quantum_ms:
                # Inside the current quantum: the incumbent keeps the resource
                # if it still has work.  This is the switch-cost saver.
                pick = next((w for w in waiters if w.lease.tid == owner), None)
            if pick is None:
                pick = waiters[0]
            if (
                owner is not None
                and pick.lease.tid == owner
                and run_age_ms >= res.quantum_ms
                and self._demand_from_others(rid, owner, now)
            ):
                # The incumbent's quantum is spent and somebody else is asking.
                # Grant nothing: leave the resource free so the demanding
                # tenant's next poll wins it.  Handing it back to the incumbent
                # here is what starvation looks like from the inside.
                self.stats[rid].yields += 1
                break
            q.remove(pick)
            lease = pick.lease
            lease.granted_at = now
            ttl_ms = max(lease.est_ms * self.ttl_factor, self.min_ttl_ms)
            lease.expires_at = now + ttl_ms / 1000.0
            active[lease.lease_id] = lease
            lease.vstart = pick.vstart
            if (tp := self.tenants.get(lease.tid)) is not None:
                # Being granted a turn IS progress: the tenant asked, and the
                # resource moved for it.
                tp.last_hb = now
                tp.last_progress_at = now
            # Advance the resource's clock in VIRTUAL units -- it is the floor
            # for every new vstart, so it must share vfinish's units (accumulated
            # weighted service, ms of est_ms).  Wall-clock time works only while
            # tenants stay backlogged enough for vfinish to outrun it; once grants
            # get rarer than est_ms of real time the clock dominates, every vstart
            # collapses to the same floor, and the weighting degrades to FIFO.
            self._vtime[rid] = max(self._vtime[rid], pick.vstart)
            wait_ms = (now - lease.requested_at) * 1000.0
            st = self.stats[rid]
            st.grants += 1
            st.waits.append(wait_ms)
            if len(st.waits) > 8192:
                del st.waits[: len(st.waits) - 8192]
            t = self.tenants.get(lease.tid)
            if t is not None:
                t.grants += 1
                t.wait_ms_total += wait_ms
                t.waits.append(wait_ms)
                if len(t.waits) > 4096:
                    del t.waits[: len(t.waits) - 4096]
            if owner != lease.tid:
                st.switches += 1
                self._run_owner[rid] = lease.tid
                self._run_started[rid] = now
            elif run_age_ms >= res.quantum_ms:
                # Same tenant, fresh quantum.  Without this the incumbent's run
                # age grows without bound and every later grant is instantly
                # deniable, so it could never hold a full quantum again.
                self._run_started[rid] = now
            granted.append(lease.lease_id)
        # NOTE: ``_run_owner`` is deliberately NOT cleared when the queue and
        # the active set both empty.  A polling client releases and re-asks in
        # the same instant; if the owner reset in that gap, the incumbent would
        # always re-acquire before any other tenant could poll, and the quantum
        # would never expire against it.  That is precisely how the first
        # version starved one of two tenants to zero turns.  The field means
        # "who ran last", and it is cleared on unregister.
        return granted

    def lease_by_id(self, lease_id: int) -> Lease | None:
        return self._leases.get(lease_id)

    def active_leases(self, rid: str) -> list[Lease]:
        return list(self._active.get(rid, {}).values())

    def queue_depth(self, rid: str) -> int:
        return len(self._queues.get(rid, ()))

    # -- reporting ----------------------------------------------------------

    def occupancy(self) -> dict[str, Any]:
        """Per-resource occupancy, with freshness stamps on every number.

        SPEC-016 requirement 8: a queued request looks exactly like a slow one
        unless queue depth is published alongside latency.  Both are here.
        """
        now = self.clock()
        holders = self._holders()
        out: dict[str, Any] = {"now": now, "epoch": self.epoch, "resources": {}}
        for rid, res in self.resources.items():
            st = self.stats[rid]
            waits = sorted(st.waits)
            tens = [t for t in self.tenants.values() if t.rid == rid]
            out["resources"][rid] = {
                "kind": res.kind,
                "concurrency": res.concurrency,
                "quantum_ms": res.quantum_ms,
                "units_total": len(res.units),
                "units_allocatable": len(res.allocatable),
                "allocatable_by_class": res.allocatable_by_class(),
                "grants": st.grants,
                "switches": st.switches,
                "overruns": st.overruns,
                "yields": st.yields,
                "busy_ms": round(st.busy_ms, 3),
                "queue_depth": self.queue_depth(rid),
                "active": len(self._active.get(rid, {})),
                "wait_ms_p50": _pct(waits, 0.50),
                "wait_ms_p95": _pct(waits, 0.95),
                "wait_ms_p99": _pct(waits, 0.99),
                "run_owner": self._run_owner[rid],
                "tenants": {
                    t.tid: {
                        "mode": t.mode,
                        "weight": t.weight,
                        "pid": t.pid,
                        "state": t.state(
                            now,
                            hb_timeout_s=self.hb_timeout_s,
                            stall_s=self.stall_s,
                            holding=t.tid in holders,
                        ),
                        "progress": t.progress,
                        "hb_age_s": round(now - t.last_hb, 3),
                        "progress_age_s": round(now - t.last_progress_at, 3),
                        "assigned_cpus": list(t.assigned),
                        "assigned_by_class": _tally_units(res, t.assigned_units),
                        "prefer_class": t.prefer_class,
                        "epoch": t.epoch,
                        "grants": t.grants,
                        "overruns": t.overruns,
                        "busy_ms": round(t.busy_ms, 3),
                        "wait_ms_p50": _pct(sorted(t.waits), 0.50),
                        "wait_ms_p99": _pct(sorted(t.waits), 0.99),
                    }
                    for t in sorted(tens, key=lambda x: x.tid)
                },
            }
        return out


def _tally_units(res: Resource, units: Sequence[tuple[int, ...]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for u in units:
        k = res.class_of(u) or "uniform"
        out[k] = out.get(k, 0) + 1
    return out


def _pct(sorted_vals: Sequence[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return round(sorted_vals[idx], 3)


def cpu_resource(
    rid: str,
    cores: Mapping[tuple[int, int], tuple[int, ...]],
    *,
    kinds: Mapping[tuple[int, int], str] | None = None,
    reserved_cores: Iterable[tuple[int, int]] = (),
    quantum_ms: float = DEFAULT_QUANTUM_MS,
    dims: Mapping[str, float] | None = None,
) -> Resource:
    """Build a CPU resource from a measured ``(node, core) -> logical cpus`` map.

    ``reserved_cores`` is how a host declares "these cores belong to somebody I
    do not schedule" -- the production ring's stage processes, the OS, an
    endpoint the operator has not opted in.  Reserved capacity is visible in the
    occupancy report but never handed out.
    """
    reserved = set(reserved_cores)
    ordered = sorted(cores)
    units = tuple(cores[k] for k in ordered)
    classes = tuple((kinds or {}).get(k, "") for k in ordered)
    res_units = tuple(cores[k] for k in ordered if k in reserved)
    return Resource(
        rid=rid,
        kind="cpu",
        units=units,
        unit_classes=classes,
        reserved_units=res_units,
        concurrency=1,
        quantum_ms=quantum_ms,
        dims=dict(dims or {}),
    )
