# Global Slot Router Runtime Port: proj 10 main -> ik_llama fork-base

## Scope

Port the runtime client and scheduling gate for the **Global Slot Router** (`tools/gslot` / `gslot_client.h`) onto the `ik_llama` public fork (`fork-base` @ `32ea126b`), completing Lane 11 (card `cood35bi9fjbfxk8jyn58nrcj8o`).

The target fork already contains `tools/gslot` (daemon, protocol, tests, and CLI runner), but was missing the C++ runtime integration in `examples/stage-runner/` that connects pipelined stages (`llama-stage-runner`) to the host-local arbiter.

This document records:
1. The architectural mapping between `proj 10 main` and `ik_llama fork-base`.
2. The runtime integration surface in `examples/stage-runner/stage-runner.cpp` and `stage-server.h`.
3. Guarding, configuration, and fail-open discipline.
4. Sequencing relative to Lane 10 (!23, expert-server) and `llama-gsched`.
5. CPU acceptance verification via the two-stage Gemma loopback harness.

---

## Architectural Mapping

### 1. Component Boundaries

| Component | Status in fork-base | Source Reference (proj 10) | Port Action |
|---|---|---|---|
| `tools/gslot` (Arbiter daemon) | Present (`tools/gslot/`) | `tools/gslot/` | Unchanged (standard library Python daemon) |
| `c/gslot_client.h` | Present (`tools/gslot/c/`) | `tools/gslot/c/gslot_client.h` | Mirrored to `examples/stage-runner/gslot_client.h` |
| `stage-runner.cpp` gate | Missing | `examples/stage-runner/stage-runner.cpp` | Integrated across `head`, `relay`, `tail`, `server` |
| `stage-server.h` gate | Missing | `examples/stage-runner/stage-server.h` | Integrated in wave dispatch and pump loops |
| `llama-gsched` graph hooks | Missing | `src/llama-gsched*` (branch `28-gsched-gate`) | Sequenced on top of Lane 10 (!23) |

### 2. Design Invariants & Discipline

- **Default OFF**: In the absence of `STAGE_GSLOT_SOCKET`, the gate evaluates to a single branch predicting true with zero syscalls. Behavior and output remain bit-for-bit identical to baseline.
- **Strict Fail-Open**: A scheduler that blocks the inference pipeline upon failure is worse than no scheduler. Connection failures, daemon restarts, timeouts, or parse faults immediately fail open, increment fault telemetry, and back off without halting token generation.
- **Zero Third-Party Dependencies**: The C++ client is header-only C++17 utilizing standard POSIX domain sockets (`AF_UNIX`) and monotonic clock primitives. No new CMake targets or libraries are introduced.
- **Two Lease Disciplines**:
  - `quantum` (default, 250 ms): Single lease per time window for continuous batching / high-throughput workloads.
  - `burst`: Immediate acquire-before-compute and release-on-handoff (`handoff()`). Sized for pipelined stages where each node computes for a fraction of each token wave and waits while downstream stages execute.

---

## Runtime Integration Surface

### 1. Header Placement

`gslot_client.h` is placed directly alongside `stage-runner.cpp` in `examples/stage-runner/gslot_client.h` (and kept identical to `tools/gslot/c/gslot_client.h`). No modification to `examples/stage-runner/CMakeLists.txt` is required.

### 2. File-Scope Gate

At file scope in `stage-runner.cpp`:
```cpp
#include "gslot_client.h"

// Global slot compute arbiter gate (SPEC-016). Default OFF when STAGE_GSLOT_SOCKET is unset.
static gslot::gate g_gslot;
```

### 3. Pipeline Role Integration

Every stage-runner role follows the acquire -> compute -> emit -> handoff / yield lifecycle:

#### A. Head Stage (`--role head` and `--role head` with `STAGE_MTP`)
- **Prefill**: Gate acquired before `run_tokens(...)`. Handoff after `send_hidden(...)`.
- **Decode loop**: Gate acquired before each decode step:
  ```cpp
  while (!g_gslot.open(gslot::now_ms_monotonic())) { usleep(2000); }
  if (!run_tokens(b, dt, ds, dp, hd)) break;
  if (!send_hidden(fd, hd)) break;
  g_gslot.handoff();
  ```
- **Teardown / Idle**: `g_gslot.yield()` releases any held quantum.

#### B. Tail Stage (`--role tail`)
- **Prefill**: Gate acquired before initial `run_hidden(...)`. Handoff immediately after.
- **Decode loop**:
  ```cpp
  while (!g_gslot.open(gslot::now_ms_monotonic())) { usleep(2000); }
  if (!run_hidden(b, hd, false, dummy)) break;
  g_gslot.handoff();
  ```
- Token sampling (`argmax_ith`) and streaming (`send_tokens`) occur outside the compute lease hold.

#### C. Relay Stage (`--role relay`)
- **Layer Window**: Gate acquired before intermediate `run_hidden(...)`. Handoff immediately following `send_hidden(fd_down, out)`.

#### D. Server Role (`--role server`, `stage-server.h`)
- In `run_server_pipelined`: Gate checked in wave dispatch before handing off wave to forward socket.
- In `run_server`: Gate checked in `gen()` around prefill and decode wave compute.

---

## Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STAGE_GSLOT_SOCKET` | *unset* | Path to arbiter AF_UNIX socket. Unset = gate completely disabled. |
| `STAGE_GSLOT_MODE` | `quantum` | `quantum` (coarse-grained time slices) or `burst` (acquire before compute, release on emit). |
| `STAGE_GSLOT_TENANT` | `stage-runner@<host>:<pid>` | Unique tenant identifier for registration and telemetry. |
| `STAGE_GSLOT_RESOURCE` | `cpu:host` | Target managed resource partition (`cpu:host`, `gpu:0`, etc.). |
| `STAGE_GSLOT_QUANTUM_MS` | `250` | Lease duration in milliseconds (clamped 10..5000 ms). |
| `STAGE_GSLOT_RETRY_MS` | `5` | Polling retry interval when lease is denied (clamped 1..1000 ms). |
| `STAGE_GSLOT_WEIGHT` | `1.0` | Proportional fair-share scheduling weight. |

---

## Sequencing with Lane 10 and `llama-gsched`

Lane 10 (!23, `expert-server-port`) disaggregates MoE expert computation over TCP. As confirmed during review, Lane 10 operates independently of `gslot_client.h`.

`llama-gsched` introduces microsecond-scale idle window harvesting around remote expert RPC brackets (`llama_gsched::window_open()` / `window_close()` in `llama-experts-remote.cpp`). Because this idle window exists solely during remote expert RPC network round-trips:
1. **Lane 11 (this port)** provides the complete inter-process stage arbiter runtime and dispatch gate across all stage-runner roles, enabling multi-tenant coordination on CPU and GPU.
2. The intra-process expert RPC idle gate (`llama-gsched`) builds on top of Lane 10's transport once !23 is merged into `fork-base`.

---

## Verification & Proofs

Acceptance gate:
1. **Zero-Warning Compilation**: Build clean under `gcc`/`clang` with `-Wall -Wextra`.
2. **Regression (Gate OFF)**: Two-stage Gemma loopback produces the 12 reference tokens identically with `STAGE_GSLOT_SOCKET` unset.
3. **Active Arbiter Proof (Gate ON)**: `gslotd` daemon running on `/tmp/gslot.sock`. Both head and tail stages run with `STAGE_GSLOT_SOCKET=/tmp/gslot.sock` and `STAGE_GSLOT_MODE=burst`.
   - Both tenants register with the daemon.
   - Successful lease grant / release cycles recorded on both stages.
   - Generated tokens match the reference stream byte-for-byte:
     ```
     The capital of France is Europe.
     <|channel>thought
     <channel|>It appears there is a
     ```

---

## Acceptance Run Results (2026-09-11 21:51 UTC)

### Platform & Topology
- **Host**: build VM (AMD EPYC 7B12, 8 vCPUs, 32 GiB RAM)
- **Model**: Gemma-4 12B Q4_0 (a Gemma-4 12B Q4_0 per-layer library, 48 layers)
- **Branch**: the lane-11 port branch (based on the staging tip with lanes 9 and 10)
- **Pipeline Shape**:
  - Head stage: layers `[0, 24)`, `STAGE_THREADS=4`, `STAGE_EMIT=hidden`, `STAGE_GSLOT_MODE=burst`
  - Tail stage: layers `[24, 48)`, `STAGE_THREADS=4`, `STAGE_PRINT=1`, `STAGE_GSLOT_MODE=burst`
  - Arbiter: `tools/gslot` (`python3 -m gslot --socket /tmp/gslot.sock`)

### Measurements
- **Arbiter Telemetry** (`/occupancy` / `/tenants`):
  - Head tenant registered: `head-stage` (mode=turn, weight=1.00)
  - Tail tenant registered: `tail-stage` (mode=turn, weight=1.00)
  - Total grants: 28 (14 head-stage, 14 tail-stage)
  - Switches: 28, Overruns: 0, Faults: 0
- **Throughput**:
  - Head: prefilled 1 slots x 6 tok, decode 13 steps in 3.63s = 3.58 tok/s
- **Output Token Stream** (`STAGE_PRINT=1` on Tail):
  ```
  [s0] Europe
  [s0].
  [s0]

  [s0]<|channel>
  [s0]thought
  [s0]

  [s0]<channel|>
  [s0]It
  [s0] appears
  [s0] there
  [s0] is
  [s0] a
  ```
- **Byte-Exactness**:
  Matches reference comparison against single-process `llama-cli` byte-for-byte:
  `The capital of France is Europe.\n<|channel>thought\n<channel|>It appears there is a`.
