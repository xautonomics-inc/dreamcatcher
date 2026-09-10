# Canonical Bring-Up Guide: Multi-Stage Distributed Inference

This guide provides the authoritative, step-by-step procedure for orchestrating, executing, and measuring multi-stage distributed inference on the `ik_llama.cpp` fork.

It covers:
1. **Single-box loopback**: A two-stage pipeline on one machine using two GPUs to verify layer assembly, tensor handoff, and token generation before introducing network transport.
2. **Two boxes over TCP**: Distributed execution across separate physical hosts over TCP sockets.
3. **Acceptance measurement**: Explicit verification criteria and smoke testing for every phase.

---

## 1. Architecture & Pipeline Topology

In multi-stage pipeline parallelism, a model's transformer layers are partitioned across sequential stages. Communication occurs via hidden-state activation frames passed forward along the pipeline:

```
[Prompt Input]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Stage 0: Head Stage (GPU 0)                            │
│ - Embeds input prompt tokens                           │
│ - Executes layer window [0, 24)                        │
│ - Emits hidden state activations                       │
└──────────────────────────┬─────────────────────────────┘
                           │ TCP Socket (127.0.0.1:8081 or 10.0.0.11:8081)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Tail Stage (GPU 1)                            │
│ - Listens on TCP socket                                │
│ - Injects hidden activations into layer 24             │
│ - Executes layer window [24, 48)                       │
│ - Computes final norm, logits, and token sampling      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                    [Output Tokens]
```

### Layer Window vs Stage-Local Indexing
- `--layers <start>,<end>`: Defines the **absolute layer window** loaded from the per-layer sliced model directory on disk (e.g. `--layers 24,48` for layers 24 through 47).
- `STAGE_IL_START=0 STAGE_IL_END=24`: Defines the **stage-local execution range** consumed by the internal graph builder within that stage's allocated submodel.

---

## 2. Honest Current State & Baseline Status

> [!IMPORTANT]
> **Loader Flag Integration Status**:
> The public fork base incorporates the per-layer assembly API within the core library. However, the standalone `llama-stage-runner` binary CLI argument parser is currently undergoing integration for `--model-dir` and `--layers` flags.
> Running `llama-stage-runner` with `--model-dir` on older builds halts with:
> `stage: unknown arg --model-dir` (argument parsing failure at line 506).
>
> The commands and outputs presented in this guide define the canonical bring-up contract. All command blocks and captured output blocks that reflect post-integration behavior are explicitly marked with `[ILLUSTRATIVE - NOT EXECUTED]`.

---

## 3. House Rules of Operation (Verbatim)

All automated agents and human operators must strictly observe these five operational constraints:

1. **/health is not liveness (prove with a completion)**
   HTTP 200 on an endpoint does not verify generation capability. Liveness requires a verified token completion stream.
2. **arithmetic is not proof (feasibility is a screen; the proof is a survived prefill)**
   Memory accounting and planner calculations are feasibility screens only. Runtime proof requires a survived prefill and successful forward pass.
3. **the layer window is a launch flag**
   Layer splits and assignments are determined at process launch time (`--layers <start>,<end>`), decoupled from static model files.
4. **pin GPUs by UUID never by index**
   Never rely on ordinal GPU indexes (e.g. `CUDA_VISIBLE_DEVICES=0`). Always query and pin via device UUID (`CUDA_VISIBLE_DEVICES=GPU-<uuid>`).
5. **measure never declare**
   Every configuration, performance figure, or topology assertion must be accompanied by explicit measurement and verification.

---

## 4. Phase 1: Single-Box Two-Stage Loopback

The single-box loopback validates per-layer loading, tensor serialization, and token sampling on a single machine before introducing network transport.

### Step 1: Pre-Flight Stage Planning

Use `layer_distribution.plan` to compute deterministic layer boundaries and screen VRAM feasibility:

```bash
python3 -m layer_distribution.plan /models/gemma4-12b-qat-q4_0-layers \
  --node stage0:16GiB:100GiB:8080 \
  --node stage1:16GiB:100GiB:8081 \
  --model-dir /models/gemma4-12b-qat-q4_0-layers
```

#### What to Measure to Know It Worked
1. Exit code is `0`.
2. Output table reports `PASS (weights)` for all stages.
3. Layer windows continuously cover the entire model without gaps or overlaps (e.g. `[0, 24)` and `[24, 48)` for a 48-layer model).

*Example Output [ILLUSTRATIVE - NOT EXECUTED]*:
```
=== Layer Distribution Stage Plan ===
Model: gemma4-12b-qat-q4_0 (48 layers total)
Status: FEASIBLE
Summary: All 48 layers assigned and feasible across 2 active nodes
VRAM Basis: weights only (layers + parts); KV cache and compute buffers not modelled

Node         Window     Layers  Disk Req / Free        VRAM Req / Cap         Status           Launch Command
-------------------------------------------------------------------------------------------------------------------
stage0       [0, 24)    24      3.22 GiB / 100.00 GiB  2.85 GiB / 16.00 GiB   PASS (weights)   llama-stage-runner --role head --connect stage1:8081 --model-dir /models/gemma4-12b-qat-q4_0-layers --layers 0,24
stage1       [24, 48)   24      3.22 GiB / 100.00 GiB  2.85 GiB / 16.00 GiB   PASS (weights)   llama-stage-runner --role tail --listen 8081 --model-dir /models/gemma4-12b-qat-q4_0-layers --layers 24,48
```

---

### Step 2: Launch the Tail Stage (Listener)

Always start the **Tail stage first** so its TCP socket is listening before the Head attempts to dial in.

```bash
# Pin GPU by UUID (query UUIDs via your GPU driver query utility)
export CUDA_VISIBLE_DEVICES=GPU-98765432-abcd-ef01-2345-6789abcdef01

# Stage execution range
export STAGE_ACTIVE=1
export STAGE_IL_START=0
export STAGE_IL_END=24

# Launch Tail process [ILLUSTRATIVE - NOT EXECUTED]
llama-stage-runner \
  --role tail \
  --listen 8081 \
  --model-dir /models/gemma4-12b-qat-q4_0-layers \
  --layers 24,48
```

#### What to Measure to Know It Worked
1. **Socket Binding**: Verify that port `8081` is actively listening:
   ```bash
   ss -tulpn | grep 8081
   # Expected: tcp LISTEN 0 128 0.0.0.0:8081
   ```
2. **VRAM Allocation**: Verify that process VRAM matches model layer weight projections (~2.85 GiB).
3. **Process Readiness**: Verify process log output [ILLUSTRATIVE - NOT EXECUTED]:
   ```
   stage: IL=[0,24) EMIT=- role=tail slots=1
   llama_model_load_from_parts: loading layer window [24, 48) from /models/gemma4-12b-qat-q4_0-layers
   stage_conn_listen: listening on 0.0.0.0:8081
   ```

---

### Step 3: Launch the Head Stage (Driver)

Launch the Head stage to load the first layer slice, dial the Tail stage, embed the prompt, and execute inference:

```bash
# Pin dedicated GPU by UUID
export CUDA_VISIBLE_DEVICES=GPU-12345678-abcd-ef01-2345-6789abcdef00

# Stage execution range and activation emission
export STAGE_ACTIVE=1
export STAGE_IL_START=0
export STAGE_IL_END=24
export STAGE_EMIT=hidden

# Launch Head driver [ILLUSTRATIVE - NOT EXECUTED]
llama-stage-runner \
  --role head \
  --connect 127.0.0.1:8081 \
  --model-dir /models/gemma4-12b-qat-q4_0-layers \
  --layers 0,24 \
  --prompt "Explain Fermat's principle in optics." \
  --max-tokens 64
```

#### What to Measure to Know It Worked
1. **TCP Connection Handshake**:
   - Tail log confirms connection: `stage[tail]: client (forward=TCP)`.
   - Head log confirms socket dial: `stage_conn_dial: connected to 127.0.0.1:8081`.
2. **Prefill Survival**:
   - Head successfully executes initial prompt prefill through layers 0–23 without OOM or fault.
   - Hidden activation tensor (`result_output`) is transmitted over loopback TCP.
3. **End-to-End Completion Output**:
   - Head stdout streams generated completion text [ILLUSTRATIVE - NOT EXECUTED]:
     ```
     Fermat's principle states that the path taken by a ray of light between two points is the path that can be traversed in the least time...
     ```
4. **Throughput Metric**:
   - Verify non-zero decode throughput [ILLUSTRATIVE - NOT EXECUTED]:
     ```
     stage[head]: DECODE 64 steps x 1 slots in 1.42s = 45.07 tok/s agg
     ```

---

## 5. Phase 2: Two Boxes Over TCP

Once loopback execution is verified, transition to multi-node distributed inference across two physical hosts.

### Topology
- **Host A (Head Stage)**: IP `10.0.0.10`, carries layers `[0, 24)`.
- **Host B (Tail Stage)**: IP `10.0.0.11`, carries layers `[24, 48)`.
- Interconnect: Dedicated 10GbE+ network interface.

```
       Host A (10.0.0.10)                        Host B (10.0.0.11)
┌───────────────────────────────┐         ┌───────────────────────────────┐
│ Stage 0 (Head)                │         │ Stage 1 (Tail)                │
│ CUDA_VISIBLE_DEVICES=GPU-<u0> │         │ CUDA_VISIBLE_DEVICES=GPU-<u1> │
│ Dials 10.0.0.11:8081          ├────────►│ Listens on 0.0.0.0:8081       │
└───────────────────────────────┘   TCP   └───────────────────────────────┘
```

### Step 1: Network & Port Verification

Before launching runners, verify network path integrity and latency:

```bash
# On Host A: test TCP reachability to Host B port 8081
nc -zv 10.0.0.11 8081

# Check network round-trip time and MTU
ping -c 4 10.0.0.11
```

#### What to Measure
- Ping RTT < 0.2 ms on local LAN / direct connect.
- TCP port connectivity confirmed without firewall drops.

---

### Step 2: Start Tail Stage on Host B

```bash
export CUDA_VISIBLE_DEVICES=GPU-98765432-abcd-ef01-2345-6789abcdef01
export STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=24

llama-stage-runner \
  --role tail \
  --listen 8081 \
  --model-dir /models/gemma4-12b-qat-q4_0-layers \
  --layers 24,48
```

#### What to Measure
- Socket status: `ss -tulpn | grep 8081` confirms listening on `0.0.0.0:8081`.
- VRAM on Host B: matches layer slice allocation.

---

### Step 3: Start Head Stage on Host A

```bash
export CUDA_VISIBLE_DEVICES=GPU-12345678-abcd-ef01-2345-6789abcdef00
export STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=24 STAGE_EMIT=hidden

llama-stage-runner \
  --role head \
  --connect 10.0.0.11:8081 \
  --model-dir /models/gemma4-12b-qat-q4_0-layers \
  --layers 0,24 \
  --prompt "Describe the physical principles of fiber optic transmission." \
  --max-tokens 64
```

#### What to Measure to Know It Worked
1. **Network Socket State**:
   ```bash
   ss -tn 'sport = :8081 or dport = :8081'
   # Expected: ESTAB state between 10.0.0.10 and 10.0.0.11
   ```
2. **Zero Packet Retransmits**:
   ```bash
   netstat -s | grep -i retrans
   # Verify TCP retransmits do not spike during tensor transmission
   ```
3. **Token Stream & Tok/s**:
   - Complete completion stream received on Host A.
   - Aggregate tokens-per-second recorded.

---

## 6. Phase 3: Acceptance & Release Smoke Verification

When running an HTTP serving endpoint (`llama-server`), verify service liveness using the release smoke tool:

```bash
ci/smoke-serve.sh http://127.0.0.1:8080 --min-tps 1.0
```

### Acceptance Checklist
| Item | Verification Target | Pass Criteria |
| :--- | :--- | :--- |
| **Feasibility** | `layer_distribution.plan` | `Status: FEASIBLE`, `PASS (weights)` |
| **Port Binding** | `ss -tulpn` | Stage listener socket in `LISTEN` state |
| **Connection** | Process logs | TCP forward handshake established |
| **Liveness Proof** | Token completion stream | Survived prefill + non-empty completion tokens |
| **Throughput** | `tokens_per_second` | Measured tok/s $\ge$ required threshold |
| **Clean Teardown** | Process exit / SIGINT | Socket released, 0 orphaned processes |

---

## 7. Operational Traps & Troubleshooting

- **Stage Launch Sequence**: Always start the listener stage (`tail`) prior to the dialing stage (`head`). Starting Head first causes immediate connection failure (`Connection refused`).
- **GPU Pinning by UUID**: Ordinal GPU index numbers (0, 1, 2) can shift across driver reloads or system reboots. Always query the persistent hardware UUID and set `CUDA_VISIBLE_DEVICES=GPU-<uuid>`.
- **VRAM Reserve for Context**: The feasibility screen checks model weights. Real inference requires memory headroom for KV cache buffers and compute scratchpads. Keep 2–4 GiB of unallocated VRAM headroom per GPU.
- **CPU Thread Contention**: When running on CPU or hybrid cores, bind each stage runner to disjoint CPU core masks using `taskset -c <cores>` to prevent threadpool starvation.
