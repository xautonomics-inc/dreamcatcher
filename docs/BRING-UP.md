# Canonical Bring-Up Guide: Multi-Stage Distributed Inference

This guide provides the authoritative, step-by-step procedure for orchestrating, executing, and measuring multi-stage distributed inference on the `ik_llama.cpp` fork.

It covers:
1. **Architecture & pipeline topology**: How transformer layers are partitioned across sequential stages and how hidden activations flow.
2. **Execution nuances & runner flags**: The 9 critical runtime settings, flags, and conventions required for stable stage runner execution.
3. **Single-box loopback**: A two-stage pipeline on one machine verifying layer assembly, tensor handoff, and token generation before introducing network transport.
4. **Two boxes over TCP**: Distributed execution across separate physical hosts over TCP sockets.
5. **Acceptance measurement**: Explicit verification criteria, captured logs, and smoke testing for every phase.

---

## 1. Architecture & Pipeline Topology

In multi-stage pipeline parallelism, a model's transformer layers are partitioned across sequential stages. Communication occurs via hidden-state activation frames passed forward along the pipeline:

```
[Prompt Input]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Stage 0: Head Stage (GPU 0 / Core Mask 0)              │
│ - Embeds input prompt tokens                           │
│ - Executes layer window [0, 24)                        │
│ - Emits hidden state activations (STAGE_EMIT=hidden)   │
└──────────────────────────┬─────────────────────────────┘
                           │ TCP Socket (127.0.0.1:8081 or 10.0.0.11:8081)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Tail Stage (GPU 1 / Core Mask 1)              │
│ - Listens on TCP socket (--listen 8081)                │
│ - Injects hidden activations into layer 24             │
│ - Executes layer window [24, 48)                       │
│ - Computes final norm, logits, and token sampling      │
│ - Streams generated tokens to stdout (STAGE_PRINT=1)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                    [Output Tokens]
```

### Layer Window vs Stage-Local Indexing
- `--layers <start>,<end>`: Defines the **absolute layer window** loaded from the per-layer sliced model directory on disk (e.g. `--layers 24,48` for layers 24 through 47).
- `STAGE_IL_START=0 STAGE_IL_END=24`: Defines the **stage-local execution range** consumed by the internal graph builder within that stage's allocated submodel. Because each stage runner loads only its specified layer slice, local layers are always 0-indexed relative to the slice.

---

## 2. Execution Nuances & Runner Flag Conventions

Executing multi-stage inference reliably requires observing 9 operational nuances and runner flags:

1. **Listener Flag Syntax (`--listen PORT`)**
   The runner listener flag accepts only the port number (e.g. `--listen 8081` or `--listen 53600`). Do not pass an IP address or hostname to `--listen`.
2. **Thread Allocation (`STAGE_THREADS=...`)**
   Set `STAGE_THREADS` in the environment to explicitly bound compute worker threads per stage process (e.g. `export STAGE_THREADS=3` or `STAGE_THREADS=8`). When co-locating multiple stages on a single machine or CPU host, this bounds thread consumption and prevents CPU starvation.
3. **Single-Sequence Pipeline Mode (`--n-seq-max 1`)**
   The stage runner executes single-sequence pipelines. Passing `--n-seq-max 1` is required to ensure batch scheduling invariants are preserved during distributed forward passes.
4. **Explicit Context Size (`--n-ctx <N>`)**
   Always specify an explicit context limit (e.g. `--n-ctx 512`). Models frequently declare large maximum training contexts in GGUF metadata (such as 262,144 tokens in Gemma-4); omitting `--n-ctx` causes excessive KV cache buffer allocations.
5. **Tail Output Emission (`STAGE_PRINT=1`)**
   In pipeline parallelism, final layer execution, RMS normalization, logits evaluation, and autoregressive sampling take place exclusively on the **Tail stage**. To stream generated tokens to standard output, set `STAGE_PRINT=1` on the Tail process. The Tail runner emits tokens prefixed by slot index (e.g. `[s0] Europe`). The Head stage drives prompt ingestion and decode steps (`STAGE_EMIT=hidden`), but does not emit completion text.
6. **Direct Socket Transport (No HTTP Head for v1 Tail)**
   Stage runners establish direct peer-to-peer TCP socket connections (`--connect <ip>:<port>`). In the v1 pipeline architecture, no HTTP proxy or intermediate gateway head is used for the Tail stage.
7. **CPU MoE Auto-Fit Flag (`-cmoe`)**
   When executing Mixture-of-Experts (MoE) architectures on CPU-only or GPU-less hosts, specify `-cmoe`. Without this flag, the device auto-fitting allocator halts when attempting to place sparse expert layers without GPU devices.
8. **Teardown Buffer Protection (`stdbuf -o0`)**
   During process shutdown or SIGINT termination, teardown memory cleanup can trigger an abort (meta#77) before standard C library I/O buffers flush. Wrapping stage runner invocations with `stdbuf -o0` disables stdout buffering and ensures all generated tokens and logs are written immediately.
9. **Slice-Relative Layer Indexing (`STAGE_IL_START=0`, `STAGE_IL_END=<N>`)**
   Because each stage runner builds its local computation graph solely from the layers loaded via `--layers <start>,<end>`, internal layer indexing is zero-based. For example, a 24-layer tail stage loaded with `--layers 24,48` executes layers `[0, 24)` internally, requiring `STAGE_IL_START=0 STAGE_IL_END=24`.

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

> [!IMPORTANT]
> **Gemma-4 Quantization Variant & CPU Execution (meta#80 / meta#81)**:
> - **Quantization (meta#80)**: Build the layer library from the **Unsloth Q4_0 quant** (`unsloth/gemma-4-12b-it-GGUF`, where `token_embd` is quantized as `Q4_K`). Google's official `gemma-4-12b-it-qat-q4_0.gguf` carries a `Q6_K` `token_embd` tensor which currently triggers degenerate output (`011111111111`) across both CPU and CUDA (tracked in meta#80).
> - **Execution Mode (meta#81)**: Run the Gemma-4 stage runner with `-ngl 0` (CPU build). The Gemma-4 loopback logs and completions documented in this guide are captured from CPU execution; the fork's Gemma-4 CUDA graph is currently being stabilized under meta#81.

```bash
python3 -m layer_distribution.plan /models/gemma4-12b-q4_0-layers \
  --node stage0:16GiB:100GiB:8080 \
  --node stage1:16GiB:100GiB:8081 \
  --model-dir /models/gemma4-12b-q4_0-layers
```

#### What to Measure to Know It Worked
1. Exit code is `0`.
2. Output table reports `PASS (weights)` for all stages.
3. Layer windows continuously cover the entire model without gaps or overlaps (e.g. `[0, 24)` and `[24, 48)` for a 48-layer model).

*Example Output*:
```
=== Layer Distribution Stage Plan ===
Model: gemma4-12b-q4_0 (48 layers total)
Status: FEASIBLE
Summary: All 48 layers assigned and feasible across 2 active nodes
VRAM Basis: weights only (layers + parts); KV cache and compute buffers not modelled

Node         Window     Layers  Disk Req / Free        VRAM Req / Cap         Status           Launch Command
-------------------------------------------------------------------------------------------------------------------
stage0       [0, 24)    24      3.22 GiB / 100.00 GiB  2.85 GiB / 16.00 GiB   PASS (weights)   llama-stage-runner --role head --connect stage1:8081 --model-dir /models/gemma4-12b-q4_0-layers --layers 0,24
stage1       [24, 48)   24      3.22 GiB / 100.00 GiB  2.85 GiB / 16.00 GiB   PASS (weights)   llama-stage-runner --role tail --listen 8081 --model-dir /models/gemma4-12b-q4_0-layers --layers 24,48
```

---

### Step 2: Launch the Tail Stage (Listener)

Always start the **Tail stage first** so its TCP socket is listening before the Head attempts to dial in.

```bash
# Pin GPU by UUID (if using GPU; for CPU execution, omit or set -ngl 0)
export CUDA_VISIBLE_DEVICES=GPU-98765432-abcd-ef01-2345-6789abcdef01

# Stage execution range, thread budget, and stdout token emission
export STAGE_ACTIVE=1
export STAGE_IL_START=0
export STAGE_IL_END=24
export STAGE_THREADS=3
export STAGE_PRINT=1

# Launch Tail listener under stdbuf -o0
stdbuf -o0 llama-stage-runner \
  --role tail \
  --listen 8081 \
  --model-dir /models/gemma4-12b-q4_0-layers \
  --layers 24,48 \
  --max-tokens 12 \
  --n-ctx 512 \
  --n-seq-max 1 \
  -ngl 0
```

#### What to Measure to Know It Worked
1. **Socket Binding**: Verify that port `8081` is actively listening:
   ```bash
   ss -tulpn | grep 8081
   # Expected: tcp LISTEN 0 128 0.0.0.0:8081
   ```
2. **Process Readiness & Layer Assembly**: Verify captured startup log output:
   ```
   stage: IL=[0,24) EMIT=- role=tail slots=1
   stage: including parts-other.gguf (present in library)
   stage: assembling window [24,48) from 27 part files in /models/gemma4-12b-q4_0-layers (embd=1 output=1 other=1)
   llama_model_loader: assembled 27 parts: block_count=24 leading_dense_block_count=0 nextn_predict_layers=0 (335 tensors)
   llm_load_print_meta: arch             = gemma4
   llm_load_print_meta: n_layer          = 24
   llama_init_from_model: n_ctx         = 512
   llama_init_from_model: graph nodes  = 723
   stage: listening on :8081
   ```

---

### Step 3: Launch the Head Stage (Driver)

Launch the Head stage to load the first layer slice, dial the Tail stage, embed the prompt, and execute inference:

```bash
# Pin dedicated GPU by UUID (if using GPU; for CPU execution, omit or set -ngl 0)
export CUDA_VISIBLE_DEVICES=GPU-12345678-abcd-ef01-2345-6789abcdef00

# Stage execution range, activation emission, and thread allocation
export STAGE_ACTIVE=1
export STAGE_IL_START=0
export STAGE_IL_END=24
export STAGE_EMIT=hidden
export STAGE_THREADS=3

# Launch Head driver under stdbuf -o0
stdbuf -o0 llama-stage-runner \
  --role head \
  --connect 127.0.0.1:8081 \
  --model-dir /models/gemma4-12b-q4_0-layers \
  --layers 0,24 \
  --prompt "The capital of France is" \
  --n-ctx 512 \
  --n-seq-max 1 \
  -ngl 0
```

#### What to Measure to Know It Worked
1. **TCP Connection Handshake**:
   - Head log confirms connection to `127.0.0.1:8081`.
   - Tail log transitions from listening to active decode.
2. **Prefill Survival**:
   - Head successfully executes initial prompt prefill through layers 0–23 without fault:
     ```
     stage[head]: prefilled 1 slots x 6 tok
     ```
   - Hidden activation tensor (`result_norm`) is transmitted over loopback TCP.
3. **Throughput Metric**:
   - Verify non-zero decode throughput recorded across both stages:
     ```
     stage[head]: DECODE 11 steps x 1 slots in 3.84s = 2.87 tok/s agg, 2.87 t/s/slot
     stage[tail]: decode 11 steps x 1 slots in 3.32s = 3.31 tok/s agg, 3.31 t/s/slot
     ```
4. **Captured End-to-End Completion Output**:
   - Under `STAGE_PRINT=1`, Tail stdout streams generated tokens:
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
   - Reference comparison against single-process `llama-cli` on the full model:
     ```
     The capital of France is Europe.
     <|channel>thought
     <channel|>It appears there is a
     ```
     The generated token stream from the two-stage pipeline matches the single-process reference byte-for-byte.

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
│ Dials 10.0.0.11:8081          ├────────►│ Listens on :8081              │
│ STAGE_EMIT=hidden             │   TCP   │ STAGE_PRINT=1                 │
└───────────────────────────────┘         └───────────────────────────────┘
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
export STAGE_THREADS=8 STAGE_PRINT=1

stdbuf -o0 llama-stage-runner \
  --role tail \
  --listen 8081 \
  --model-dir /models/gemma4-12b-q4_0-layers \
  --layers 24,48 \
  --max-tokens 64 \
  --n-ctx 512 \
  --n-seq-max 1
```

#### What to Measure
- Socket status: `ss -tulpn | grep 8081` confirms listening on port `8081`.
- VRAM on Host B: matches layer slice allocation.
- Log output confirms `stage: listening on :8081`.

---

### Step 3: Start Head Stage on Host A

```bash
export CUDA_VISIBLE_DEVICES=GPU-12345678-abcd-ef01-2345-6789abcdef00
export STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=24 STAGE_EMIT=hidden
export STAGE_THREADS=8

stdbuf -o0 llama-stage-runner \
  --role head \
  --connect 10.0.0.11:8081 \
  --model-dir /models/gemma4-12b-q4_0-layers \
  --layers 0,24 \
  --prompt "The capital of France is" \
  --max-tokens 64 \
  --n-ctx 512 \
  --n-seq-max 1
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
   - Under `STAGE_PRINT=1`, generated tokens stream on Host B standard output.
   - Aggregate tokens-per-second recorded in runner logs.

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
| **Liveness Proof** | Token completion stream | Survived prefill + non-empty completion tokens emitted on Tail stdout (`STAGE_PRINT=1`) |
| **Throughput** | `tokens_per_second` | Measured tok/s $\ge$ required threshold |
| **Clean Teardown** | Process exit / SIGINT | Socket released, 0 orphaned processes |

---

## 7. Operational Traps & Troubleshooting

- **Stage Launch Sequence**: Always start the listener stage (`tail`) prior to the dialing stage (`head`). Starting Head first causes immediate connection failure (`Connection refused`).
- **Runner Listener Flag**: Use `--listen PORT` (e.g. `--listen 8081`). Specifying hostnames or IP addresses in `--listen` causes argument parsing errors.
- **GPU Pinning by UUID**: Ordinal GPU index numbers (0, 1, 2) can shift across driver reloads or system reboots. Always query the persistent hardware UUID and set `CUDA_VISIBLE_DEVICES=GPU-<uuid>`.
- **Explicit Context Allocation**: Always specify `--n-ctx` (e.g. `--n-ctx 512`). Unset `--n-ctx` defaults to full model training context (e.g. 262k), leading to unnecessary memory consumption.
- **Single Sequence Invariant**: Pass `--n-seq-max 1` for all stage runner processes. Stage runner execution requires single-sequence batches.
- **Token Output Emission**: Tokens are generated and sampled on the Tail stage. Enable `STAGE_PRINT=1` on the Tail runner to observe generated tokens on stdout. The Head driver outputs decode metrics but does not stream tokens.
- **Direct Connection Topology**: The v1 stage runner pipeline communicates via direct TCP sockets (`--connect <host>:<port>`). No intermediate HTTP server or gateway is placed in front of the tail stage.
- **MoE on CPU**: When running Mixture-of-Experts architectures on CPU or GPU-less hosts, pass `-cmoe` to prevent device auto-fit allocation aborts.
- **Unbuffered Stdout on Teardown**: Wrap stage runner invocations with `stdbuf -o0` to prevent loss of buffered stdout during process teardown or signal termination (meta#77).
- **Slice-Relative Stage Layer Indexing**: Internal layer ranges (`STAGE_IL_START`, `STAGE_IL_END`) are indexed relative to the loaded stage slice (`0` to `N`), not global model layer indices.
- **Gemma-4 Quantization Variant (meta#80)**: Until meta#80 is resolved, use the Unsloth Q4_0 quant (`token_embd` quantized as `Q4_K`) rather than Google's official `gemma-4-12b-it-qat-q4_0.gguf` (which quantizes `token_embd` as `Q6_K`). On the current fork base, Q6_K embedding tensors on Gemma-4 trigger degenerate `011111111111` token output on both CPU and CUDA, whereas Q4_K embeddings run reliably.
- **CPU Thread Contention**: When running on CPU or shared cores, bound threads using `STAGE_THREADS=<N>` and bind each stage runner to disjoint CPU core masks using `taskset -c <cores>` to prevent threadpool starvation.
