# Runbook — `gslot` core partitioning for the expert-server topology

> **SHELVED for the expert-server topology, 2026-08-26.** The operator descoped
> expert-servers from arbiter integration — they get dedicated compute, so
> there is nothing to arbitrate. This document is retained as **the reference
> tier-1 recipe** for any future CPU co-tenancy, and its host-b section is the
> worked example of hybrid P/E-core placement. The measurements and the
> commands are current; only the intended consumer changed.

**Originally for:** the expert-disaggregation workstream (host-b attention
stages + local expert-server; host-c expert-server). Written to be applied
verbatim, and still is — for whoever needs CPU co-tenancy next.

**Status:** the arbiter is deployed and **inert** on host-a, host-b and host-c — zero
tenants, zero grants, loopback-only, no autostart unit. It does nothing until
you register. Nothing below changes any GPU, any model, or any port.

Spec: `SPEC-016-global-slot-compute-scheduler.md`.

---

## 1. Why bother — the cost of not doing it

Measured on host-a, two co-resident llama.cpp CPU tenants on a 12-physical-core
pool, identical work every arm:

| how the tenants are sized | wall s | aggregate tok/s | p99 s |
|---|---|---|---|
| **`-t $(nproc)` each — the default everyone writes** | 377.08 | 12.12 | 71.54 |
| `-t` = pool size each | 211.37 | 21.63 | 35.73 |
| `-t` = pool/2 each, unpinned | 202.60 | 22.57 | 36.14 |
| **arbiter partition** | **189.58** | **24.12** | **34.28** |

**2.0x aggregate throughput and half the p99, against the case that actually
happens.** Most of that gap is the difference between "sized for the machine"
and "sized for your share" — which is the number the arbiter tells you. The
rest is placement.

You are already ahead of the naive case because you hand-wrote cpusets. What
this buys you on top: the sets are **recomputed when tenancy changes** (a third
tenant arrives and everyone shrinks, correctly), they are **whole physical
cores** so nothing lands on your SMT sibling, they are **class-aware** on
host-b's hybrid part, and the assignment is **published** at `/occupancy`
rather than living in a launcher comment.

## 2. host-b (six-card CUDA host)

**Topology, measured:** Core Ultra 7 265K — **8 P-cores (cpu 0-7, capacity
1012) and 12 E-cores (cpu 8-19, capacity 756), no SMT.** The kernel's `core_id`
values interleave the classes (0, 12, 24, 32, 36, 39), so anything that deals
cores "in order" hands out a silently unequal mix. The arbiter reads the class
from `/sys/devices/cpu_{core,atom}/cpus` and deals each class proportionally,
unless a tenant asks for one.

### 2.1 Restart the daemon with the right reserve

It currently runs `--reserve-cpus 16-19`, which holds 4 E-cores back for the OS
and a **still-live head stage**. Once the ring is down and this box is yours:

```bash
# host-b, after ring teardown
kill -TERM $(pgrep -f '[p]rism.gslot')      # by PID; pkill -f self-matches
rm -f /run/gslotd.sock
cd $GSLOT_HOME/gslot
PYTHONPATH=$GSLOT_HOME/gslot nohup python3 -m gslot \
  --socket /run/gslotd.sock --http 127.0.0.1:8099 --reserve-cpus '' \
  > $GSLOT_HOME/gslotd.log 2>&1 &
```

`--reserve-cpus ''` gives all 20 cores to tenants, which matches your current
layout (0-15 + 16-19 accounts for the whole box). If you want a couple of cores
kept out of the pool for the OS, `--reserve-cpus 18,19` and drop the front
tenant to `--min-cores 2 --max-cores 2`.

### 2.2 Register the tenants

**Two-tenant steady state** (expert-server + attention front). This reproduces
your hand-written `expert-server 0-15` / `client 16-19` **exactly**:

```bash
# the big local expert-server -- wants the fast cores
$GSLOT_HOME/gslot/tools/gslot/gslot-run \
  --tenant expert-server --min-cores 16 --max-cores 16 --prefer-class P \
  -- llama-expert-server --role expert-server --model $MODEL \
     --expert-layers 1-47 --listen $PORT --threads 16

# the attention front
$GSLOT_HOME/gslot/tools/gslot/gslot-run \
  --tenant attn-front --min-cores 4 --max-cores 4 \
  -- llama-server -m $MODEL --port 8090 -t 4 ...
```

Result, asserted by `test_expert_server_recipe_reproduces_the_hand_tuned_layout`:

| tenant | cores | class mix | cpus |
|---|---|---|---|
| `expert-server` | 16 | **8 P + 8 E** | 0-15 |
| `attn-front` | 4 | 4 E | 16-19 |

**Three-tenant, while qwen38 is still co-resident:**

```bash
gslot-run --tenant expert-server --min-cores 12 --max-cores 12 --prefer-class P -- ...
gslot-run --tenant qwen38       --min-cores 4  --max-cores 4  -- ...
gslot-run --tenant attn-front   --min-cores 4  --max-cores 4  -- ...
```

→ `expert-server` 8 P + 4 E (cpu 0-11), `qwen38` 4 E (12-15), `attn-front` 4 E
(16-19). qwen38 is GPU-resident, so E-cores are the right place for it.

`qwen38-nvfp4` is a systemd unit, so wrap it with a drop-in rather than editing
the unit:

```ini
# /etc/systemd/system/qwen38-nvfp4.service.d/gslot.conf
[Service]
ExecStart=
ExecStart=$GSLOT_HOME/gslot/tools/gslot/gslot-run --tenant myworkload \
  --min-cores 4 --max-cores 4 -- <the original ExecStart, verbatim>
```

`systemctl daemon-reload` then restart the unit. To back it out, delete the
drop-in and reload — the original `ExecStart` is untouched.

### 2.3 Drop `--min-cores/--max-cores` if you want the arbiter to rebalance

Fixed sizes reproduce today's layout, which is the point of a recipe. If you
would rather it adapt as tenancy changes, use weights instead:

```bash
gslot-run --tenant expert-server --weight 4 --prefer-class P -- ...
gslot-run --tenant attn-front    --weight 1 -- ...
```

→ 16 / 4 with two tenants, and a correct three-way split the moment a third
registers, with no launcher edits.

## 3. host-c (CPU-only host)

**Topology, measured:** 16 physical cores, **SMT2**, uniform (no P/E split),
one APU. Allocatable is currently 12 physical cores — `--reserve-cpus 0-3,16-19`
holds 4 whole cores back for the OS and the **live bottleneck ring stage**. Once
host-c is exclusively yours:

```bash
kill -TERM $(pgrep -f '[p]rism.gslot'); rm -f /run/gslotd.sock
cd $GSLOT_HOME/gslot
PYTHONPATH=$GSLOT_HOME/gslot nohup python3 -m gslot \
  --socket /run/gslotd.sock --http 127.0.0.1:8099 --reserve-cpus 0,16 \
  > $GSLOT_HOME/gslotd.log 2>&1 &

$GSLOT_HOME/gslot/tools/gslot/gslot-run --tenant expert-server -- \
  llama-expert-server --role expert-server --model $MODEL \
    --expert-layers 1-47 --listen $PORT --threads 15
```

**This deliberately differs from your `cores 0-15`, and you should know why.**
`0-15` is one SMT sibling of each of the 16 physical cores; the other sibling of
every one of those cores is left open for anything else that lands on the box.
The arbiter allocates **whole cores**, so a single tenant asking for 15 gets
`1-15,17-31` — 15 complete cores, both siblings, nothing else can be scheduled
onto their execution units. Run the same 15 threads; you just also own the
siblings. When a second tenant registers, both shrink to whole-core halves
instead of interleaving on shared cores.

`--threads` should be the count of **physical cores** granted, not logical CPUs.

## 4. Verify — three commands

```bash
curl -s localhost:8099/occupancy | python3 -m json.tool     # what was granted
grep Cpus_allowed_list /proc/<pid>/status                   # what was applied
curl -s localhost:8099/inventory | python3 -m json.tool     # measured topology
```

`/occupancy` publishes `allocatable_by_class` per resource and
`assigned_cpus` + `assigned_by_class` per tenant, so an unequal share is
**visible rather than inferred**. Check the two agree; the grant is what the
arbiter thinks, `/proc` is what the kernel did.

## 5. Failure modes, honestly

**The arbiter dying does nothing bad.** Every client call is fail-open with no
option to make it fail closed. `gslot-run` with no arbiter logs one line and
execs the child unpinned — exactly as if the shim were not there.

**A stale pin can outlive the shim.** `sched_setaffinity` is sticky. If
`gslot-run` is killed but its child somehow survives, the child keeps its pin
while the arbiter reclaims the cores and may hand them to someone else. In
normal operation the shim forwards SIGINT/SIGTERM/SIGHUP and the child dies
with it, and under systemd the unit's own lifecycle handles it. If you ever
kill a shim by hand, check for a surviving child before registering anything
new.

**A partition change re-pins the child within ~0.5 s** (the heartbeat
interval), which costs a ggml thread-pool warm-up. The solver keeps a tenant on
its existing cores wherever the new size allows, so steady state does not churn.

**Do not enable `STAGE_GSLOT` (the tier-2 turn gate) on a CPU tenant.**
Measured on two real co-resident stage-runner ring instances: the mechanism
worked perfectly — 237 switches, 0 faults, 0 overruns, fairness 0.94 — and it
was still a **27% regression** (wall 53.1 → 67.4 s, p99 6.94 → 9.24 s), because
a CPU is divisible and exclusive turns destroy pipeline overlap. Turn leases
are for an indivisible device. Core partitioning is the CPU tool.

## 6. Backing it out

Remove the `gslot-run` prefix from the launch line (or delete the systemd
drop-in and `daemon-reload`). Nothing else changes: no model, no port, no
alias, no GPU. Affinity resets on the next start of the process.
