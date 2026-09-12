# SPEC-016 — Global slot compute scheduler (`gslot`)

**Status:** phase 1 implemented and measured.
**Date:** 2026-08-26

---

## 1. The problem, stated as it actually appears

Every GPU on this fleet is allocated. `rocm-smi` on host-a reports 17–21 GiB used of
21–25 GiB on all four cards; host-b's six 5060 Ti are four ring stages plus two
endpoints; a third host's cards run the remaining stages and side
services. There is no spare
card and there has not been one for weeks. New work therefore does not get a
card — it gets *co-residency*, and co-residency on this fleet is currently
unmanaged.

Unmanaged co-residency has already cost us measurable throughput, twice, in ways
that look like different bugs but are one bug:

- **CPU (expert-disagg P1, 2026-08-22).** Two ggml processes on one host, each
  sized for the whole machine: 4.409 tok/s partitioned versus **1.2 tok/s naive**,
  with per-call latency going 0.7 ms → 8.7 ms. The mechanism is the ggml
  intra-graph spin barrier — while thread 0 blocks in an RPC custom op, the
  remaining workers spin at the node barrier and evict the co-tenant. No
  per-request politeness can fix that, because the damage happens *inside* one
  graph.
- **GPU (2026-08-19).** `qwen38` occupied a second card it did not need — the
  1.5× everyone attributed to the second GPU was the MTP head — at dsv4's
  expense. A VRAM balancer would have called that placement correct.

The ring already solved the *intra-process* version of this. `STAGE_WAVE_SCHED`
(inference/llama.cpp#4) has one pump thread owning all ring I/O, admission-time
routing, FIFO reply matching and a wave stagger, and it is deployed in the
production GLM v14 ring. What it coordinates is slots **within one
stage-runner process**.

`gslot` is the generalisation: **independent processes co-resident on the same
GPU and CPU coordinate their compute so they schedule around each other.**

## 2. Scope

In scope: a host-local arbiter; opt-in cooperative coordination between
processes; CPU core-set partitioning; turn leases on a contended device;
occupancy and per-tenant latency instrumentation; a multi-resource feasibility
screen.

Explicit non-goals, inherited from the requirements this spec derives from: live migration, automatic restart,
and **ring-internal scheduling changes — the ring keeps its own slot-router.**
`gslot` never touches a process that has not registered with it.

## 3. Design

### 3.1 Two tiers, because there are two collisions

| | tier 1 — partition lease | tier 2 — device lease |
|---|---|---|
| fixes | spin-barrier / core contention | concurrent allocation; wasted pipeline idle |
| resource | **divisible** (CPU cores) | **indivisible, shared allocator** (a GPU) |
| granularity | process lifetime | allocation phase, or one compute burst |
| steady-state RPC | **none** | one per lease |
| client | `gslot-run` shim, no code change | `gslot-run` for admission; in-process at the dispatch gate for bursts |
| status | **measured, 2.0x — section 6.1** | **designed, unvalidated — section 8** |

Tier 1 is the tool for CPU. The arbiter computes a disjoint assignment of
**whole physical cores** and returns it; *the tenant applies it to itself*.
Splitting SMT siblings across two tenants is not a partition — it is the
collision wearing a partition's clothes — so cores are allocated as units, both
siblings together. On a hybrid host cores are also dealt per **class**, because
"four cores each" is not a fair split when one tenant's four are 34% faster.

Tier 2 is the tool for a device that **cannot** be split, where two tenants
share one allocator. It does two things, in this order of importance:

1. **Bounds concurrent allocation.** Two processes sizing compute buffers
   against the same card, each believing it has the whole thing, is how a card
   OOMs and how a driver wedges. An exclusive lease makes "only one of you is
   allocating right now" true.
2. **Harvests pipeline idle.** A pipeline stage computes for a small slice of
   each token and waits out the rest — measured at S0, every ring stage but the
   bottleneck is idle >90% of every token. Another instance's compute belongs
   in those gaps. This is `STAGE_WAVE_SCHED` lifted one level: that interleaves
   slots inside one process, tier 2 interleaves processes on one device.

Section 6.2 measured a quantum-hold turn gate on a shared **CPU** and it was a
27% regression. Section 6.2.1 explains why that is evidence about a divisible
resource and the wrong lease shape, not about exclusive leasing — and section 8
proposes neither. Weighted fair queuing over virtual time keeps exclusivity
from becoming starvation.

### 3.2 Three modes

`partition` — give me a disjoint slice for my lifetime.
`turn` — grant me exclusive use of an indivisible device: across my allocation
phase, or for one compute burst (section 8.1).
`share` — I run uncoordinated; count me against capacity but never gate me.

`share` is what makes the whole thing safe to deploy on a host that also runs
production. A ring stage can be declared as a `share` tenant so the arbiter
knows its cores are spoken for, without the arbiter acquiring any ability to
gate it.

### 3.3 Resources are measured, never declared

The inventory is read from `/sys` and the vendor SMI tools at daemon start,
stamped with the time it was taken, and re-readable. There is no hand-written
table. Requirement 7 exists because every stale map on a fleet like this has
bitten us — autostart running retired recipes at boot, a console holding a v6
ring map, an endpoint registry going demo-stale. A
scheduler whose world model is a config file is the next instance of that bug.

GPUs are keyed by vendor UUID or PCI id, never by index, per the fleet GPU
identity-pinning standard.

### 3.4 Liveness: saturated ≠ stalled ≠ dead

Classified from a **measured progress counter** plus heartbeat freshness, never
from a health probe:

- `active` — heartbeat fresh, progress advancing.
- `saturated` — advancing, but the tenant waits more than it computes.
- `stalled` — heartbeat fresh, progress frozen. Alive and doing nothing useful.
- `dead` — heartbeat itself stale. Claims are reclaimed.

This is SPEC-013 §4.0 applied to tenancy. On 2026-08-18 a `/health` 200 through
a dead ring cost 90 minutes; the discriminator that worked was `/admin/stagelat`
age, i.e. a progress signal. A dropped control connection is treated as a
stronger death signal than a heartbeat timeout and reclaims capacity at once.

### 3.5 No silent actuation

Requirement 10 is enforced structurally, not by a permission check:

- The arbiter has **no code path** that touches a process which has not
  registered. Core sets are *returned to the tenant*, which calls
  `sched_setaffinity` on itself.
- The observation plane is a separate, **read-only** TCP listener. Nothing
  served there mutates state.
- An overrun reclaims the *turn*, never the registration. A slow tenant is not a
  dead tenant, and killing a process for being slow is exactly the failure mode
  the requirement forbids.
- Every client call is **fail-open**. Unreachable arbiter, timeout, malformed
  reply, arbiter restart: all return "proceed uncoordinated". There is
  deliberately no option to make this fail closed. A scheduler that can stop
  inference by dying is worse than no scheduler.

### 3.6 Feasibility is a screen, not a proof

`feasibility(resource, needs)` checks each declared dimension — `vram_bytes`,
`host_ram_bytes`, and any dimension a resource declares — against what is
already committed, and returns the arithmetic. It answers "do not bother
trying". It does **not** answer "this will fit": the report carries
`proof_required: survived-prefill`, because on this fleet arithmetic has never
been proof. A 13.45 GiB quant fits a 16 GiB card; its 40960 context does not.

Undeclared dimensions are reported as undeclared rather than assumed absent.

## 4. Protocol

Newline-delimited JSON over `AF_UNIX` SOCK_STREAM (`/run/gslotd.sock`). Ops:
`register`, `unregister`, `hb`, `lease`, `release`, `stats`, `resources`,
`tenants`, `feasibility`; one server push, `grant`, when a partition changes.

Read-only HTTP observation plane (default `127.0.0.1:8099`): `/occupancy`,
`/resources`, `/tenants`, `/inventory`, `/healthz`.

A `lease` blocks until granted — one round trip, no client state machine. A
`lease` with `"nowait": true` returns immediately; this is the polling caller
(the stage-runner pump, which must not block the one thread that owns all ring
I/O). **A nowait miss must leave no trace** — the waiter is removed and the
tenant's virtual finish is rolled back. Both halves are load-bearing:

- a left-behind waiter is granted later to a caller that has moved on, so the
  turn is held by nobody until its TTL expires and the resource deadlocks in
  slow motion;
- a left-behind virtual-time advance charges a tenant for compute it never
  received, so the tenant that polls most often starves itself.

### 4.1 Demand, and why the obvious fair-queue is not enough

Because a polling client is *not in the queue* when it loses, weighted fair
queuing alone cannot see it. The first implementation therefore let the
incumbent renew forever: measured, two identical tenants over a 4-second run
went **199 waves to 0**. The arbiter now records a decaying *demand* marker on
every request. When the incumbent's quantum is spent and another tenant has
fresh demand, the arbiter grants nothing and leaves the resource free for the
demanding tenant's next poll. Same test after the fix: 128 waves to 117,
19 alternating runs of 259 ms, zero overlap.

`_run_owner` is deliberately **not** cleared when the queue empties. A polling
client releases and re-asks in the same instant; if the owner reset in that gap
the incumbent would always re-acquire before any other tenant could poll, and
the quantum would never expire against it. The field means "who ran last".

## 5. Client integration

**`gslot-run`** — a launcher shim. `gslot-run --tenant NAME -- llama-server ...`
registers, receives a core set, applies it to the child, and re-applies it as
the partition is recomputed. No code change in the wrapped binary. This is the
on-ramp: every co-resident CPU tenant on this fleet is somebody's launcher
script, and rewriting them all to speak a protocol was never going to happen.

**`gslot_client.h`** — header-only C++17, POSIX sockets only, no library. Drop
it beside `stage-runner.cpp`; it needs no CMake change. The whole integration at
the wave-scheduler dispatch gate is one conjunct:

```cpp
const bool gate_open = (!wave_stagger || now >= t_next) && g_gslot.open(now);
```

Default OFF: with `STAGE_GSLOT_SOCKET` unset, `open()` returns `true` after one
predictable-branch bool load and issues no syscall — the same
byte-identical-when-off discipline `STAGE_WAVE_SCHED` shipped under.

The client takes **one lease per quantum**, not one per wave, and answers
locally until it expires. The pump asks the gate thousands of times a second; an
RPC per ask would cost more than the contention it prevents. Measured overhead:
one round trip per 250 ms per tenant, ≈0.04% of the quantum.

**Known gap — the quantum hold is the wrong shape for a pipeline stage.** A
stage computes for a small fraction of each token and then waits; holding the
device across that wait starves the co-tenant *and* leaves the device idle,
which is the opposite of the idle-window harvesting tier 2 is for. Section 8.1
specifies the **burst lease** that fixes it — acquire at compute start, release
at handoff, with the quantum degraded to a floor on switch rate. Not built; it
is ~30 lines plus one call site next to `ws_send_wave`, and it should follow the
validation run rather than precede it.

## 6. Measurements

### 6.1 Two co-resident llama-server CPU tenants

host-a, 2026-08-26. Two co-resident llama.cpp CPU tenants (`gemma-4-12b-it-qat-q4_0`,
dense 12B) on a 12-physical-core pool, everything confined to
`taskset -c 4-15,20-31`; cores 0-3/16-19 reserved for the ring stages and the
OS. Identical work every arm: 2 tenants x 6 requests, 256-token prompt, 64
tokens greedy, `cache_prompt:false`, non-repetitive prose (a markov-friendly
prompt inflates prefill and, with a drafter, acceptance).

| arm | `-t` each | wall s | aggregate tok/s | p50 s | p99 s | decode tok/s each |
|---|---|---|---|---|---|---|
| solo — 1 tenant, the ceiling | 12 | 111.96 | 20.42 | 18.63 | 20.49 | 5.11 |
| **naive** — 2 tenants, `-t $(nproc)` | 32 | 377.08 | 12.12 | 61.79 | 71.54 | 2.12 |
| unco — 2 tenants, `-t` = pool size | 12 | 211.37 | 21.63 | 35.50 | 35.73 | 2.51 |
| half — the obvious manual fix | 6 | 202.60 | 22.57 | 33.92 | 36.14 | 3.25 |
| **part** — arbiter partition, tier 1 | 6 | **189.58** | **24.12** | **31.62** | **34.28** | 3.05 |
| part — repeat run (control) | 6 | 191.36 | 23.89 | 32.80 | 33.94 | 3.17 |
| turn — request-granularity leases | 6 | 252.49 | 18.11 | 20.54 | 21.63 | 5.07 |

**Against `naive` — the case that actually occurs, because no launcher on this
fleet sizes itself to its share — tier 1 is 2.0x aggregate throughput, half the
p99, and half the wall.** The repeat run reproduces within 1%. Against `unco`
(+11.5%) and `half` (+6.9%) the win is real but modest; most of the prize is in
knowing how big your share is, which is what the arbiter tells you.

`turn` at request granularity is a **latency lever, not a throughput lever**.
Leasing a turn per 30-second request is serialisation, so aggregate drops to
18.11 -- but per-request decode is 5.07 tok/s against solo's 5.11, and p99 is
21.63 s against solo's 20.49 and *part's 34.28*. Each tenant runs as though
alone.

### 6.2 The gate in the real stage-runner

Two independent CPU 2-stage GLM ring instances (head `[0,2)` + tail `[76,79)`
with embd/output/nextn), built from the branch **with the gate compiled in**,
co-resident on the same 12-core pool. 2 x 8 requests, 32 tokens.

| mode | wall s | p50 s | p99 s | max s |
|---|---|---|---|---|
| gate OFF (uncoordinated) | 53.13 | 6.616 | 6.939 | 6.939 |
| **tier 1 — arbiter partition** | **52.62** | **6.566** | **6.652** | **6.652** |
| tier 2 — turn gate, 250 ms quantum | 67.43 | 8.333 | 9.242 | 9.242 |

Tier-1 grants, verified in `/proc/<pid>/status` on all four processes:
`ring-A -> 4-9,20-25`, `ring-B -> 10-15,26-31`. A small win, mostly in the tail.

**The tier-2 row is a 27% regression and it is reported as one.** The mechanism
worked exactly as designed -- `/admin/gslot` showed 122 and 115 grants,
897/916 blocked, **0 faults**; the arbiter 249 grants, 237 switches, 330 yields,
**0 overruns**, fairness 0.94 -- and it still lost.

### 6.2.1 Why this does NOT invalidate tier 2

Two independent things went wrong here, and neither is a property of exclusive
leasing as such.

**The resource was divisible.** A CPU can be genuinely split, so a core
partition gives both instances *simultaneous* progress. Serialising access to
something that did not need serialising can only cost. A GPU cannot be split
that way: two processes on one card do not get half a card each, they get
interleaved kernels and a **shared allocator**.

**The lease shape was wrong for a pipeline.** The client holds one lease for a
whole 250 ms quantum and answers locally inside it. That is right when two
tenants each want continuous throughput. It is exactly wrong for a
pipeline-parallel stage, which computes for a small fraction of each token and
then waits: holding the device across the wait both starves the co-tenant *and*
leaves the device idle. Measured on the production ring at S0 (July, v6/v14
topology -- order-of-magnitude, not current): ring period ~220 ms/wave with
per-stage busy fractions of stage-1 ~8%, stage-2 4.4%, relay 6.9%, tail 9.8%, mid
70.6%. **Every stage but the bottleneck is idle for more than 90% of every
token.** A quantum-hold lease harvests none of that. Section 8.1 defines the
lease shape that does.

So section 6.2 is evidence about *this lease shape on a divisible resource*,
which is a configuration section 8 does not propose. **Do not enable the
quantum-hold turn gate on a CPU-shared stage-runner** -- that much stands.

Also verified live on the production-lineage binary: with `STAGE_GSLOT_SOCKET`
unset, `/admin/gslot` reports `enabled:false, granted:0, blocked:0, faults:0`
and the process issues no socket calls at all.

### 6.3 Tier 2 at wave granularity, in isolation

Two tenants through the C client, 20 ms synthetic waves, 250 ms quantum, 5 s:

- **0 overlapping waves** out of 245 -- exact mutual exclusion
- fairness 0.914 (128 vs 117 waves)
- 19 alternating runs, mean run **259.4 ms**, 12.9 waves per run
- 19 grants, 19 switches, 0 overruns, **0 faults** -- one round trip per quantum

Before the demand fix in section 4.1 the same test scored **199 waves to 0**.
The primitive is sound; 6.2 says the CPU is the wrong place to spend it.

## 7. Deployment

```
python3 -m gslot --socket /run/gslotd.sock --http 127.0.0.1:8099 \
    --reserve-cpus 0-3,16-19 --quantum-ms 250
```

`--reserve-cpus` is how a host declares "these cores belong to somebody I do not
schedule" -- production ring stages, the OS, an endpoint the operator has not
opted in. A physical core is reserved if *any* of its logical CPUs is reserved:
half a core is not a usable grant.

The package is **stdlib-only** on purpose. A GPU host here has no uv, no venv
and often no pip; `scp -r gslot` plus `python3 -m gslot` has to
be the whole install story. It lives here for code ownership, CI, ruff
and mypy-strict coverage.

### 7.1 Live deployments, 2026-08-26

All three are **inert**: zero tenants, zero grants, loopback-only HTTP, and
**no autostart unit**, so a reboot removes them. None can affect a process that
has not registered.

| host | path | `--reserve-cpus` | allocatable | why that reserve |
|---|---|---|---|---|
| host-a (4× RDNA3) | `$GSLOT_HOME/gslot` | `0-3,16-19` | 12 cores | OS + live relay/tail ring stages |
| host-b (6× RTX 5060 Ti) | `$GSLOT_HOME/gslot` | `16-19` | 16 cores (8P+8E) | OS + a live head stage; leaves the exact pool the expert-server workstream hand-tuned |
| host-c (CPU-only) | `$GSLOT_HOME/gslot` | `0-3,16-19` | 12 cores | OS + the ring's bottleneck stage, which gets a wide berth |

host-b is the hybrid case: **8 P-cores (cpu 0-7) and 12 E-cores (cpu 8-19), no
SMT**, detected from `/sys/devices/cpu_{core,atom}/cpus`. host-c and host-a are
uniform SMT2, 16 physical cores each.

Post-teardown reserve values, and the per-tenant registration lines for the
expert-server topology, are in **`runbook-gslot-expert-server.md`** -- written
to be applied verbatim, and asserted end-to-end by
`test_expert_server_recipe_reproduces_the_hand_tuned_layout`.

## 8. Tier 2 — exclusive device leases: design and validation plan

**Paper only. No hardware has been touched and none will be until the run is
dispatched.** host-a's four cards are still held by ring stages.

### 8.0 What tier 2 is actually for

Operator framing, 2026-08-26: *"the idea is essentially identical to
slot-pipeline-parallel — when a stage is idle between tokens, allow another
instance to make use of the idle resources. The goal is to prevent the GPU/CPU
compute buffers from being overloaded by two processes requesting the same
resource, causing crashes, OOMs and wedges."*

That is two claims, and the second is the important one.

**Idle-window harvesting.** A pipeline stage computes for a small slice of each
token and then waits for the ring to come round. Measured at S0 on the
production ring (July, v6/v14 topology — cite as order-of-magnitude, that ring
is now being torn down): period ~220 ms/wave, busy fractions stage-1 ~8%, stage-2
4.4%, relay 6.9%, tail 9.8%, mid 70.6%. **Every stage except the bottleneck is
idle for more than 90% of every token.** Those inter-wave gaps are real,
repeating, and currently wasted. A second instance's compute belongs in them —
instances interleaving on one device exactly as slots interleave across the
ring's stages. This is the same mechanism as the wave scheduler, lifted one
level: `STAGE_WAVE_SCHED` interleaves slots inside one process, tier 2
interleaves processes on one device.

**Bounding concurrent allocation.** Two processes sizing compute buffers
against the same card, each believing it has the whole thing, is how a card
OOMs, how a process dies mid-graph, and how a driver wedges. The fleet's
history is entirely this failure class: `:8095` gated behind a VMM-OOM ceiling;
`:8094` OOM-killed by an `--ctx-checkpoints` default nobody costed; "VRAM is
lost due to GPU reset" on a mis-addressed bus-30 card; an Arc GPU wedging when `max_freq`
drifts; an MES timeout taking the ASIC down. An exclusive lease is the
mechanism that makes "only one of you is allocating right now" true.

**The success criterion is therefore safety first, harvested throughput
second.** Section 8.8 states it precisely. The earlier draft's falsifier ("if
`turn` beats neither throughput nor p99, tier 2 has no home") was wrong: equal
throughput with zero faults, against an uncoordinated arm that demonstrably
faults, is a pass.

### 8.1 Two lease shapes, because there are two failure points

This is the substantive design revision, and it comes from asking *when* a
compute buffer is actually allocated.

**(a) Allocation-admission lease — one per process lifetime.**
`llama.cpp` reserves its compute buffer once, at context creation and graph
reservation, not per wave. **A dispatch-time lease therefore does not bound
peak allocation for it at all** — by the time the first wave dispatches, the
buffer is already resident. The collision happens during *startup*, or when a
graph is re-reserved because the batch shape changed.

So the lease that carries the safety claim is: **hold a lease across the
load/warmup/graph-reservation phase, release it once the process is healthy.**
Only one process on a device may be in its allocation phase at a time. This is
cheap (one lease per process), needs **no code change** — `gslot-run` can hold
it and release on a health probe — and it directly prevents the OOM/crash
class. It is also the piece that generalises to any co-resident endpoint on the
card, not just ring stages.

**(b) Dispatch burst lease — acquire at compute start, release at handoff.**
This is the idle-window harvester, and it needs a change to the C client. The
shipped client holds one lease per **quantum** (default 250 ms) and answers
locally inside it; section 6.2.1 explains why that is exactly wrong for a
pipeline stage. What the ring needs is a lease **held only while this stage is
computing**: acquire before dispatching a wave, release when the wave is handed
off downstream. The pump already knows both instants precisely — that is why
`may_dispatch()` at the wave-pump dispatch point is the right hook — but it
currently never releases mid-quantum because `ws_active` stays non-empty while
waiting for replies.

Concretely, `gslot_client.h` gains a `burst` mode where `may_dispatch()`
acquires and a new `handoff()` releases, with the quantum degraded to a *floor*
on switch rate rather than a hold. **This is not built.** It is roughly 30
lines plus one call site next to `ws_send_wave`, and it should be written
*after* the arms in 8.4 show the exclusivity is worth having, not before.

### 8.2 Consumer

Expert-servers are **descoped** from arbiter integration (operator, 2026-08-26
— they get dedicated compute). `runbook-gslot-expert-server.md` stays as the
reference tier-1 recipe but is shelved for that topology.

Tier 2's consumer is **the future ring built on the VRAM the GLM mini-ring
frees, plus any endpoints co-resident on those cards** — i.e. exactly the
"never free cards" situation that motivated this workstream, where a ring stage
idle 90% of each token shares a card with something else.

### 8.3 Target hardware — RUN, 2026-08-26

**host-a**, card **`xt-14`** (a 20 GiB RDNA3 workstation card),
render node `renderD130`, Navi 31 XT, 19.98 GiB usable. The other three cards
were untouched. Pinned by passing that one render node into the container, and
verified: VRAM moved only on `GPU[2]` throughout.

**Prerequisite RESOLVED.** The host build trees all fail on missing ROCm
runtime libraries — the ring ran them inside containers. The working image is
our own **`ik_llama.cpp/ling-hip:0a8aebbf-gfx1100`**, the HIP sibling of the
`ik-ling-cpu` build section 6.1 used. It loads `gemma4`, offloads 49/49 layers,
`ROCm0 buffer size = 6637.69 MiB`. Graceful `TERM` gave `Exited (0)` with VRAM
fully reclaimed, every time.

### 8.4 Calibration — and the first result

Section 8.5 says calibrate before colliding. Single tenant, `-c 16384`,
`rocm-smi` sampled at 5 Hz through load and warmup:

```
settled = 14,306,144,256 B = 13.32 GiB   (weights 6.48 + KV 5.25 + ~1.6 compute)
peak    = 14,306,144,256 B = 13.32 GiB
card    = 21,458,059,264 B = 19.98 GiB
```

**Peak equals settled.** There is no transient allocation spike: llama.cpp
reserves its compute buffer once at context creation and holds it. That
confirms section 8.1's analysis directly — **a dispatch-time lease would bound
nothing on this stack**, and the lease that carries the safety claim has to be
the allocation-admission lease.

Two tenants at this size would need 26.64 GiB on a 19.98 GiB card: a guaranteed
collision, and the margin is measured rather than guessed.

### 8.5 Results — the safety arms

An incumbent serving continuously; a newcomer arrives that cannot fit.

| | `unco-pressure` | `lease-pressure` |
|---|---|---|
| newcomer | container starts, loads ~90 s, **allocation fails, `Exited (1)`** | **refused, `rc=75`, container never created** |
| newcomer wasted work | full model load | none |
| faults recorded | **2** | **0** |
| incumbent requests | 9 ok / 0 failed | 7 ok / 0 failed |
| incumbent decode | 21.33 tok/s | 21.46 tok/s |
| incumbent p99 | 3.831 s | 3.723 s |
| VRAM peak | 13.63 GiB | 13.63 GiB |

`gslot-run` output in the leased arm:

```
[gslot-run] REFUSED by the arbiter: register failed: infeasible claim
[gslot-run] not launching the child (--require-admission)
```

**The headline is a partial falsification of my own safety case, and it is the
most useful thing in this run.**

**ROCm degraded gracefully.** The uncoordinated newcomer's allocation failed
cleanly, it exited 1, and **the incumbent was not damaged** — 9 of 9 requests
completed at full rate, no wedge, no reset, VRAM reclaimed on exit. The
catastrophic outcome the design assumed — a starting process taking down the
one already serving — **did not occur on this hardware and this runtime.**
Section 8.8 predicted this possibility and called it good news about the fleet
rather than a failed experiment. It is.

So the value the admission lease actually delivered is narrower than claimed:

- a **doomed 90-second model load** became an instant refusal with the
  arithmetic attached;
- **zero faults instead of two** — nothing to find in a log later;
- the incumbent's tail was **2.9% better** (p99 3.723 vs 3.831 s). Small, and on
  9 vs 7 samples p99 is effectively the max, so treat it as suggestive rather
  than established — but it points the right way: the doomed allocation attempt
  did perturb the incumbent, just nowhere near fatally.

That is real, and it is fail-fast-with-a-reason, not crash prevention. **Claim
it as that.** Whether the catastrophic case exists on other stacks (Vulkan on
A770, the CUDA endpoints, unified memory on host-c) is untested — those are where
the fleet's actual wedge history lives, and none of them is this one.

### 8.6 Results — the harvest arms

Two tenants at `-c 2048` (8.73 GiB each, 17.43 GiB together — fits), 6 requests
each, 256-token prompt, 64 tokens greedy.

| arm | wall s | aggregate tok/s | p50 s | p99 s | decode each | arbiter |
|---|---|---|---|---|---|---|
| `solo` | 22.12 | 103.36 | 3.68 | 3.70 | 21.49 | — |
| `unco-fit` | 33.70 | **135.68** | 5.47 | 5.86 | 14.11 | — |
| `turn-fit` | 44.17 | 103.52 | **3.67** | **3.74** | 21.72 | 12 grants, 12 switches, 0 overruns |

**The idle-window harvest is already being done — by the GPU driver, not by
us.** Two uncoordinated tenants reach **1.31x** the single-tenant aggregate
(135.68 vs 103.36). ROCm time-slices the card efficiently enough that there is
no idle window left for a scheduler to harvest.

Exclusive turns convert that 31% throughput gain into **latency isolation**:
p99 5.86 -> 3.74 s, which is solo's 3.70 s to within noise, and per-tenant
decode returns to solo rate (21.72 vs 21.49). Same shape as the CPU result in
section 6.1: **turn leases are a latency lever, on GPU exactly as on CPU.**

### 8.7 Failure classes — what actually happened

Provoked, as intended: **one OOM-class allocation failure** (`failed to
allocate`, container `Exited (1)`). Recoverable exactly as predicted — the
process died, the device was untouched, VRAM returned to the 27.9 MB idle
baseline, and the co-resident tenant kept serving.

Not provoked, as intended: no GuC wedge, no MES timeout, no `amdgpu` ring
timeout, no ASIC reset, no all-ones MMIO. No HIP process was ever SIGKILLed —
every stop was `TERM` with a 45 s grace and an explicit refusal to escalate. No
clock, power or `--device` manipulation. One card; the other three never bound.

### 8.8 Verdict

Against the criterion in the plan: **safety is a qualified pass, harvest is a
clear no.**

- **Admission control: keep.** It turns a 90-second doomed load into an instant
  refusal with the math, and costs one registration. Cheap, and it is the only
  arm that produced zero faults where the baseline produced two.
- **Turn leases for throughput: reject, on both backends but for different
  reasons.** On ROCm uncoordinated is 31% faster because the driver has already
  taken the idle window. On CUDA uncoordinated is only 3% faster because there
  is barely an idle window to take — co-residency there is very slightly
  *negative* (0.95x solo). Either way exclusivity does not add aggregate
  throughput.
- **Turn leases for latency isolation: keep, and price them per backend.**
  Both backends return a leased tenant to *solo* service — 100-101% of solo
  decode, p50 indistinguishable from solo. What differs is the aggregate cost:
  **3% on CUDA, 24% on ROCm.** On NVIDIA hardware this is close to free and is
  the strongest case tier 2 has; on AMD it is a real trade to be argued per
  workload.
- **The crash-prevention claim is refuted on both backends this fleet runs.**
  ROCm/gfx1100 and CUDA/sm120 each failed an over-large allocation cleanly and
  left the serving tenant unharmed — CUDA to within 0.015% of its decode rate.
  Do not repeat the claim. What admission control delivers is fail-fast with the
  arithmetic attached, and that is worth having on its own. Vulkan and unified
  memory are still untested, but the prior should now be that they are fine too.

### 8.9 CUDA — the same arms on an NVIDIA card, 2026-08-26

Run on **host-b idx 5** (RTX 5060 Ti, 15.93 GiB), pinned by
`CUDA_VISIBLE_DEVICES=<uuid>` — never by index, because two cards on that host
belong to another workstream and can read 0 MiB during a front bounce. Same
model, same serving flags, `-t 2` host threads to stay out of that workstream's
instrumentation. Zero Xid events; their cards untouched at 9390/8484 MiB
throughout.

**Prerequisite:** the container images are CPU-only here; the working binary is
the **host** build at `$GSLOT_BUILD/bin/llama-server`. It
reports `CUDA0 model buffer size = 6637.69 MiB` — byte-identical to ROCm's
figure, so the two backends are running the same model bytes.

**Calibration difference worth recording.** CUDA's gemma4 path honours the
**sliding window**, so KV is bounded by the window rather than the context:
ctx 16384 costs **7.90 GiB** here versus **13.32 GiB** on the ROCm ik_llama
build, which allocated full KV. On CUDA the footprint lever is **parallel
slots**, not context. Configs used: fit = ctx 2048 / 1 slot = 7.64 GiB;
pressure = ctx 16384 / 8 slots = 11.14 GiB.

#### 8.9.1 Safety — CUDA is *more* forgiving than ROCm

| | `unco-pressure` | `lease-pressure` |
|---|---|---|
| newcomer | starts, **allocation fails, never healthy** | **refused, `rc=75`, never launched** |
| faults | **2** | **0** |
| incumbent | **27/27 ok**, 52.779 tok/s | 19/19 ok, 52.787 tok/s |
| incumbent p50 / p99 | 1.3879 / 1.3985 s | 1.3876 / 1.3919 s |

**The incumbent's decode rate differs by 0.015% between the two arms**, and its
per-request rates stayed flat across the newcomer's failed allocation
(52.384-52.872, a 0.9% spread that is run-to-run noise). Where ROCm showed a
2.9% tail difference that I flagged as suggestive, **CUDA shows none at all.**

**That is the crash-prevention claim falsified on a second backend, more
cleanly than on the first.** Both of the allocators this fleet actually runs
refuse an over-large allocation without harming the tenant already serving.
Section 8.8's verdict stands and is now much better supported.

#### 8.9.2 Harvest — and a large backend divergence

| arm | wall s | aggregate tok/s | p50 s | p99 s | decode each | arbiter |
|---|---|---|---|---|---|---|
| `solo` | 8.32 | **274.76** | 1.3854 | 1.3906 | 52.85 | — |
| `unco-fit` | 17.45 | 261.96 | 2.8892 | 2.9514 | 24.73 | — |
| `turn-fit` | 18.00 | 254.01 | **1.3876** | 2.7323 | **52.79** | 12 grants, 12 switches, 0 overruns |

**ROCm and CUDA behave oppositely under co-residency:**

| | ROCm (7900 XT) | CUDA (5060 Ti) |
|---|---|---|
| `unco-fit` / `solo` aggregate | **1.31x** | **0.95x** |
| per-tenant decode, uncoordinated | 66% of solo | 47% of solo |
| `turn-fit` cost vs `unco-fit` | **-24%** | **-3%** |
| per-tenant decode, turn-leased | 101% of solo | 100% of solo |

**ROCm harvests co-residency; CUDA does not.** Two uncoordinated tenants on the
AMD card produce 31% more aggregate than one; on the NVIDIA card they produce
5% *less*. So section 8.6's "the driver has already taken the idle window" is
**a ROCm property, not a general one** — the claim needed the second backend to
be scoped correctly, and it was wrong as stated.

**This flips the economics of turn leases.** Both backends restore per-tenant
solo-rate service (100-101% of solo decode, and p50 identical to solo). The
difference is what that costs in aggregate: **24% on ROCm, 3% on CUDA.** On an
NVIDIA card, giving a latency-SLO tenant its full solo rate while sharing a
card with a throughput tenant is nearly free.

### 8.9 Cost, actual

**~2 h 15 m** against a ~3 h estimate. The HIP prerequisite was the risk and it
resolved in one probe once the right container image was found — the host build
trees are all unusable outside a ROCm container, which is worth knowing.

Original estimate, for calibration of future ones: **~3 hours.** ~30 min on the HIP/`gemma4` prerequisite (the real risk — it is a
build if it fails), ~20 min rig setup, ~30 min calibrating the pressure margin
on a single tenant, ~70 min for six arms plus loads and cooldowns, ~30 min
write-up. Post to `agent-coord` when the card is claimed and when it is
released. A freed card is not proof of a working config: the only accepted
feasibility proof remains a **survived prefill**.

## 9. Full-width alternating leases for CPU pipeline stages

Operator, 2026-08-26: *"allow all of the CPU's threads to be used by one
instance while the stage is idle."*

### 9.1 The idea, and why it does not contradict section 6.1

Static partitioning (tier 1) splits the machine in **space**: each tenant gets
half the cores and both compute at once. Full-width alternating splits it in
**time**: both tenants stay sized for the *whole* pool and an exclusive
wave-granularity lease decides which one computes. Each burst then runs at full
width and finishes proportionally sooner.

The reconciliation with section 6.1's "partition beats turn" result is that
those measure different regimes. The ggml spin-barrier collapse and the 27%
turn-gate regression both come from **concurrent** compute — two tenants
colliding inside one graph, or a lease held across a wait. An exclusive
wave-granularity lease makes concurrency impossible, so full width is safe; and
section 6.2's arm was **request**-granularity leases on continuously-busy
tenants, which is a third regime again. Three different things, three different
answers.

**Implementation follows the operator's constraint: do not resize threadpools.**
Both tenants are launched full width (minus OS reserves) and never re-pinned.
`sched_setaffinity` stays out of the hot path entirely; the only per-wave cost
is the lease round trip, which is measured below.

### 9.2 Measured — host-a, two stage-runner ring instances, 14-core pool

Two independent CPU 2-stage GLM rings sharing cores 2-15,18-31 (`--reserve-cpus
0-1,16-17` for the OS and nfsd). `unco` and `burst` both run **full width** on
the whole pool; `part` gets disjoint 7-core halves from the arbiter.
`gated duty` is measured by the gate itself — `held_ms_total` summed across both
tenants over wall time.

**Caveat on scope, stated up front: the gate lives in `run_server`, so it
governs the HEAD stage only.** The tail computes ungated. The head window was
widened to `[0,5)` (two IQ2 MoE blocks) precisely so the gated stage carries a
real share, but "exclusive" here means exclusive *head* compute, not exclusive
instance compute. Read the numbers accordingly.

| mode | conc | gated duty | thru tok/s | p50 s | p99 s | grants A/B | blocked A/B | acquire µs | faults |
|---|---|---|---|---|---|---|---|---|---|
| `unco` | 1 | — | 5.169 | 3.095 | 5.025 | — | — | — | 0 |
| `part` | 1 | — | **6.616** | 4.823 | **4.855** | — | — | — | 0 |
| `burst` | 1 | 0.345 | 6.456 | 4.910 | 5.050 | 498/486 | 134/137 | 310 | 0 |
| `unco` | 4 | — | 6.841 | 18.057 | 19.425 | — | — | — | 0 |
| `part` | 4 | — | **6.931** | **17.505** | 19.438 | — | — | — | 0 |
| `burst` | 4 | 0.486 | 6.822 | 18.117 | 19.545 | 1508/1528 | 721/781 | 368 | 0 |

**Handoff overhead, measured explicitly** (the thing burst has to justify):
**310-368 µs per acquire**, ~1000 acquires at conc 1 and ~3000 at conc 4. As a
fraction of wall that is **~2%** — and it accounts for essentially the entire
gap between `burst` and `part` (2.4% at conc 1, 1.6% at conc 4). The
exclusivity itself is neither helping nor hurting; the RPC is the whole
difference. 310 µs is a Python asyncio daemon over a unix socket; a C arbiter
or a shared-memory ticket would remove most of it.

### 9.3 The low-duty arm — and why duty cycle is the wrong variable

The obvious next measurement was a genuinely low duty cycle, since that is the
regime the proposal targets. It was run: thin head `[0,2)` (two small dense
blocks instead of three dense plus two IQ2 MoE), same pool, same work, guards
added for both failure modes that ate the earlier attempts — an abort if any
port is already bound, and a **real completion** from each instance before
anything is measured.

| mode | gated duty | thru tok/s | p50 s | p99 s | wall s | grants A/B | blocked A/B | acquire µs | faults |
|---|---|---|---|---|---|---|---|---|---|
| `unco` | — | 6.485 | 4.852 | 5.079 | 14.803 | — | — | — | 0 |
| `part` | — | **6.726** | **4.645** | **4.898** | 14.273 | — | — | — | 0 |
| `burst` | 0.354 | 6.511 | 4.847 | 5.020 | 14.744 | 543/523 | 123/144 | 273 | 0 |

**The head got 60% thinner and the duty cycle did not move: 0.345 -> 0.354.**
That is the finding. The gated stage's per-wave cost on this rig is
**overhead-dominated, not layer-dominated** — sampling, embedding lookup, frame
assembly and the send itself swamp two dense blocks — so **duty cannot be
lowered by thinning the head, and ≤0.2 is not reachable here at all.**

The ring's S0 figures (headA busy ~8% of a 220 ms wave) come from a topology
where the *round trip* dwarfs the head's compute. A CPU rig with a local tail
has no such round trip. **A CPU stage-runner head is not a low-duty stage.**

### 9.4 Why alternating loses, and the criterion that decides it

At the lowest duty the rig can produce, the ordering is unchanged: `part` 6.726
> `burst` 6.511 > `unco` 6.485. Handoff accounts for two thirds of the
`part`-`burst` gap (299.9 ms of acquire over a 14.744 s wall = **2.03%**, against
a 3.2% gap).

The rest is not a tuning problem, and it generalises. Full-width alternating
yields, per unit time, the throughput of **one** tenant at full width.
Partitioning yields **two** tenants at half width, concurrently. So:

> **Alternating beats partitioning only if throughput scales SUPERLINEARLY with
> core count** — i.e. only if `T(2n) > 2·T(n)`.

Measured on this fleet, it does not. From section 6.1, same host, same model:
one tenant at `-t 12` gives **20.42** tok/s aggregate; two tenants at `-t 6`
give **22.57**. `2·T(6) > T(12)` by 10.5% — **sublinear**, as expected for a
memory-bandwidth-bound ggml workload, where doubling cores does not double
achievable bandwidth. Alternating therefore starts ~10% behind on throughput
before the lease RPC is even counted, at any duty cycle.

**So for aggregate throughput the question is settled, and duty cycle was never
the deciding variable — the scaling exponent is.** Static partitioning is the
right tool for CPU co-tenancy on this fleet, and no amount of gap-harvesting
changes that while ggml scales sublinearly.

**The remaining question was a different one, and section 9.6 now answers it.**
In a real pipeline, a stage idle most of each token is idle *waiting for the
ring*, not waiting for CPU. There, contending that stage lengthens its wave and
can push it onto the critical path, stretching the ring period for everyone.
That is a **latency-on-the-critical-path** argument, not a throughput argument,
and the CPU rig could not pose it. A GPU rig could, and did.

### 9.5 Which mode to pick

- **Static partition (tier 1) is the default for CPU co-tenancy.** It won or
  tied at every duty cycle measured (0.354, 0.345, 0.486), costs nothing per
  wave, and needs no code in the serving process.
- **Full-width alternating is not recommended for CPU** on this fleet. It is
  implemented, correct and fault-free — 543/523 grants, zero faults, clean
  alternation — and it is behind on arithmetic that does not depend on tuning.
- **Reconsider it only where `T(2n) > 2·T(n)`**, or where a gated stage sits on
  a pipeline's critical path and its wave latency, not its throughput, is what
  matters.
- **Neither mode helps a saturated machine.** At conc 4 the OS scheduler already
  has enough runnable work; coordination is worth 25-28% on an idle machine and
  ~1% on a busy one.

### 9.6 The critical-path question, answered on GPU

Posed on a purpose-built rig, **not** the production mini-ring: a 2-stage GLM
gap-ring on host-b, head `[0,2)` on idx 3 and tail `[76,79)`+embd/output/nextn
on idx 4, each stage pinned by `CUDA_VISIBLE_DEVICES=<uuid>`. The existing CUDA
`llama-stage-runner` was used unmodified — **no gate** — because the first
question does not need one. Then a gemma co-tenant was added to **idx 3 only**,
the head's card, driven continuously.

| | solo | contended |
|---|---|---|
| decode | **73.116** tok/s | **49.230** tok/s (**−32.7%**) |
| ring period | **13.7 ms** | **20.3 ms** (**+48.4%**) |
| wall per request | 0.392 s | 0.580 s |
| **head stage** (contended card) | **1684 µs** | **7588 µs** (**+350%**) |
| tail stage (uncontended card) | 115 µs | 91 µs |

**Yes: a contended pipeline stage joins the critical path and stretches the
ring period.** The contended stage's own compute went **4.5×** while the
uncontended stage stayed flat, so the entire 48% period stretch is attributable
to the one stage whose card was shared. `rtt_ewma` barely moved (9.80 → 9.49
ms), confirming the cost is compute, not transport.

This is the operator's model confirmed on real hardware, and it is the one
argument for tier 2 that survives every other measurement in this document:
**co-residency on a pipeline stage is not a local cost. It propagates to the
ring period for every consumer of that ring.** A 32% decode loss for a whole
ring, caused by one co-tenant on one stage's card, is a different class of
problem from the 3–5% aggregate effects in sections 6 and 8.

#### 9.6.1 Half B — not run, and the blocker

The follow-on ("does a burst lease keep the contended stage's waves short?")
needs the gate compiled into a **CUDA** stage-runner. The cheap route I proposed
— copy the already-CUDA-built ring tree to scratch, patch, build incrementally —
**does not exist.** The patch applied cleanly (the hook is one conjunct at the
`gate_open` boolean; brace balance verified unchanged against the original), but
a CMake build directory is not relocatable: 58 build files carried absolute
paths to the original tree, and rewriting them reset ninja's stamps into a full
**253-step nvcc rebuild**. At `-j2` on a host whose CPU another workstream is
using for instrumentation, that is 40–90 minutes of somebody else's cores.
Stopped and banked; the patched tree is kept in the scratch build dir
(172 MB) for a future attempt.

**Real cost of Half B: one clean CUDA configure-and-build of that tree
(~45–90 min at `-j2`, less on a quiet host), then ~20 min of arms.** Nothing
about it is hard; it is simply not cheap, and the rig is already built and proven.

#### 9.6.2 A design constraint Half B would have to confront

Worth recording before anyone runs it, because it is not obvious — and flagged
as **untested reasoning, not a measurement**: a burst lease on the ring's head
only helps if the *other* tenant leases at a compatible granularity. The ring's
wave is ~1.7 ms and its period 13.7 ms; a co-tenant taking a **request**-
granularity turn holds the device for ~1.4 s. Exclusivity at that granularity
would not protect the ring — it would stall it for a hundred periods at a time.

So tier 2 on a pipeline stage likely needs either (a) every tenant on the card
leasing at wave granularity, or (b) **priority preemption rather than fair
alternation** — a latency-critical stage that can interrupt, not merely take
turns.

**Section 9.7 measured this. (a) is refuted** — a finer-grained co-tenant made
things worse, because turn *frequency* hurts, not turn length. **(b) remains
unbuilt and is scoped as future work in 9.7.3**, with the warning that its real
cost is a yield hook inside llama.cpp's decode loop, not the scheduler side.
**9.7.4 recommends not building it**: admission-time refusal gets the same
outcome with machinery that already ships.

### 9.7 Half B — does wave-granularity leasing recover the loss?

The gate was compiled into a CUDA `llama-stage-runner` (patched scratch tree at
the scratch patched tree, clean 251-step build, 0 errors) and the section 9.6
rig re-run with four arms. Same cards, same model, same work.

**Control first:** with `STAGE_GSLOT_SOCKET` unset the gated binary reproduces
section 9.6 exactly — solo 72.283 tok/s / head 1711 µs against 73.116 / 1684,
and uncoordinated 49.234 / 6981 against 49.230 / 7588. Default-OFF is inert.

| arm | decode tok/s | period ms | **head µs** (contended card) | tail µs | wall mean s | gate granted/blocked |
|---|---|---|---|---|---|---|
| `solo` (gate off) | 72.283 | 13.8 | **1711** | 109 | 0.396 | — |
| `unco` (gate off) | 49.234 | 20.3 | **6981** | 92 | 0.578 | — |
| `lease-coarse` | 53.928 | 18.5 | **1694** | 105 | 9.123 | 304 / 1741 |
| `lease-fine` | 37.469 | 26.7 | **1824** | 162 | 8.703 | 258 / 1701 |

#### 9.7.1 The number

Two answers, and they point opposite ways.

**On the stage itself, leasing recovers 100%.** Head-stage compute went
6981 µs uncoordinated to **1694 µs** leased, against a solo baseline of 1711 µs
— **100.3% of the contention penalty removed.** The gate does exactly what it
was built to do: it makes the stage's compute exclusive, and exclusivity
eliminates the interference completely.

**On a request that holds the lease, leasing recovers 96%.** Comparing the
32-token requests: solo 0.4570 s, uncoordinated 0.6621 s (a **44.9%** penalty),
leased **0.4652 s** — a **1.8%** residual, i.e. **96.0% recovered**.

**And on the tail it is a catastrophe.** The per-request walls:

```
solo          0.456  0.458  0.457
unco          0.652  0.677  0.658
lease-coarse  35.34  0.466  0.466      <- one request blocked 35 s
lease-fine    17.13  16.99  0.464      <- two blocked 17 s
```

**The worst leased request is 52x the worst uncoordinated one.** The
distribution is bimodal: when the ring holds the lease it runs at *exactly*
solo speed; when the co-tenant holds it, the ring stops dead.

**So the honest headline: wave-granularity leasing recovers ~96-100% of Half A's
loss while it is running, and loses far more than it recovers to arbitration
stalls.** Net, it is a regression.

#### 9.7.2 Why, and a correction to 9.6.2

The gate was blocked on **1741 of 2045 asks (85%)**. Weighted fair queuing
splits a device by *service time*; the ring needs only ~13% of the device but
needs it **every 13.8 ms**. Fair-share arbitration cannot express "a small
slice, promptly", and a ring stage that waits is a ring stage on the critical
path — which is the problem Half A identified.

**Section 9.6.2 predicted that a finer-grained co-tenant would fix this. It does
not, and the measurement says so.** Cutting the co-tenant's turn length 16x
(64-token requests at ~1.4 s down to 4-token at ~80 ms) made things **worse**:
`lease-fine` is 37.469 tok/s against `lease-coarse`'s 53.928. The reason is that
the co-tenant then asked 164 times instead of 29, and every win blocks the ring
again. **Turn frequency, not turn length, is what hurts.** That was a reasonable
hypothesis and it was wrong.

#### 9.7.3 Future work: preemption — scoped, NOT built

Getting further needs a different policy, not a tuning parameter. Sketch only:

- **A `priority` claim.** A tenant registers as latency-critical with a deadline
  (here: 13.8 ms). Its demand makes the arbiter (a) stop granting to lower
  tenants and (b) require the current holder to yield within a bounded time.
- **The hard part is the yield.** Cooperative preemption needs the *co-tenant*
  to check a "must yield" flag at fine granularity — in llama.cpp that means a
  check inside the decode loop. That is an upstream-shaped change, not a
  scheduler change, and it is the whole cost of this direction.
- **Estimate:** protocol and arbiter side ~1-2 days; the llama.cpp yield hook is
  the open-ended part and should be scoped separately before anyone commits.

**Do not build this on the strength of these numbers alone** — see 9.7.4, which
gets the same outcome for free.

#### 9.7.4 What to do instead, using machinery that already ships

The measurement points at a simpler answer than preemption. Sharing a card with
a latency-critical pipeline stage costs that ring 32.7% (Half A) and cannot be
fixed by interleaving (Half B). So **do not interleave — refuse.**

The arbiter already has the piece for this: the **admission-time feasibility
screen** from section 8.1(a), which refused an over-large newcomer in
milliseconds with the arithmetic attached. Declaring a pipeline stage's card as
*claimed* and letting admission reject co-tenants is tier-1 thinking applied to
a GPU, it needs no new mechanism, and it is the recommendation this project
ends on for pipeline stages.

### 9.8 State at pause (operator pause, 2026-08-26)

**Half B is COMPLETE — its arms ran and are reported in 9.7 before the pause
landed. Nothing is pending and no work is half-done.** The gated CUDA
stage-runner is built and kept at
`build-cuda/bin/llama-stage-runner` in the scratch patched tree (2,096,328 B, 7
`STAGE_GSLOT` symbols); re-running or extending the arms is the
rig driver script, ~20 min for all four, with the rigs also
version-controlled here as `tools/gslot/ring2b-halfb.sh` and
`tools/gslot/cot-driver.py`.

All three `gslotd` daemons remain deployed and **inert** (zero tenants, zero
grants, loopback-only, no autostart). No rig processes anywhere; host-b idx
0/3/4/5 idle at 0 MiB. The only outstanding item in this document is the
preemption design of 9.7.3, which 9.7.4 recommends **not** building.

## 10. Phase 2 — not built

- Fleet-wide view: one arbiter per host, scraped read-only into a placement
  advisor. Cross-host placement stays advisory.
- Alias continuity on relaunch (requirement 6) — the scheduler must own the
  `--alias` string across a restart, since the server pins `compressor_model` to it.
- Spec-head detection by ratio (`tokens_predicted_total / n_decode_total`,
  ~2.14 active vs ~1.0 absent) as a placement constraint.
- Per-endpoint measured depth curves. There is no equivalent GPU: four distinct
  depth-performance shapes on one fleet, so no fleet-wide law.
- QoS classes bound to the `X-Request-Purpose` contract.
- Actuation of any kind. Phase 1 and this phase are opt-in cooperative only.
- Core-class awareness beyond P/E: NUMA distance and cache-domain (CCX)
  affinity are not modelled. host-a and host-c are single-node, so nothing on this
  fleet needs it yet.
