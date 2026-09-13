# Known Issues

Known-bad behaviour in the current publication snapshot. Each entry gives a
one-line reproduction and the observed symptom. All reproductions are
deterministic (`--temp 0 --seed 0`) and were captured on the `fork-base`
snapshot.

> **Note on identifiers.** The `meta#NN` handles below are internal tracking
> references. Every entry is self-contained — you do not need to resolve the
> handle to reproduce or understand the issue.

---

## `meta#80` — Gemma-4 `Q6_K` `token_embd` produces degenerate output (CPU and CUDA)

A Gemma-4 GGUF whose `token_embd.weight` is quantized as `Q6_K` (for example
Google's official QAT `gemma-4-12b-it-qat-q4_0.gguf`) produces degenerate
output on both CPU and CUDA. The same model with a `Q4_K` `token_embd` (for
example the Unsloth `gemma-4-12b-it-Q4_0` quant) is fine. The defect is in the
embedding / `get_rows` dequant path for a `Q6_K` `token_embd` in the `gemma4`
graph (which scales the embedding by `sqrt(n_embd)` after the lookup).

**Repro:**
```
llama-cli -m gemma-4-12b-it-qat-q4_0.gguf -p "The capital of France is" -n 12 --temp 0 --seed 0 -ngl 99
```
**Symptom:** `011111111111` (degenerate) on both `-ngl 99` (CUDA) and `-ngl 0` (CPU).
**Workaround:** use a `Q4_K`-embedding quant (e.g. the Unsloth `gemma-4-12b-it-Q4_0`).

---

## `meta#81` — Gemma-4 on CUDA: batched prefill produces wrong tokens

On a CUDA build, Gemma-4's default batched prefill (`-fa on`) produces wrong
tokens, while the CPU path (`-ngl 0`) and mainline CUDA are correct. The
`-b 1 -ub 1` (single-token batch) near-miss points at the batched prefill path
(sliding-window mask / per-layer-input embeddings / softcap in the batched
kernels), with a smaller secondary error in the decode path. Other archs on the
same CUDA build (e.g. Mellum) are correct, so CUDA works in general.

**Repro:**
```
llama-cli -m gemma-4-12b-it-Q4_0.gguf -p "The capital of France is" --temp 0 --seed 0 -n 12 -ngl 99
```
**Symptom:** ` not a good a a good a good` (wrong) on CUDA; the CPU reference
(`-ngl 0`) is ` Europe.\n<|channel>thought\n<channel|>It appears there is a`.
**Workaround:** run Gemma-4 on the CPU build (`-ngl 0`) until the CUDA graph is fixed.

---

## `meta#84` — Vulkan backend drifts on NVIDIA Vulkan ICD; empty `GGML_VK_VISIBLE_DEVICES` segfaults

The Vulkan backend (inherited from upstream, not a fork regression) drifts
from the CPU reference on the NVIDIA Vulkan ICD. The stage split is self-consistent
on Vulkan (head/tail == single-process), so the transport is fine; the backend
kernels are not. The Vulkan driver on the NVIDIA ICD reports "not a conformant
Vulkan implementation" and no matrix cores, and is also slower than CPU there.

**Repro (wrong tokens):**
```
llama-cli -m gemma-4-12b-it-Q4_0.gguf -p "The capital of France is" -n 12 --temp 0 --seed 0 -ngl 99   # Vulkan build, NVIDIA ICD
```
**Symptom:** wrong tokens, not matching the CPU reference; `-b 1 -ub 1` gives byte-identical wrong output (so it is not
the `meta#81` batched-prefill signature).

**Repro (segfault):**
```
GGML_VK_VISIBLE_DEVICES= llama-cli -m <any-model.gguf> ...
```
**Symptom:** exit 139 (segfault) when `GGML_VK_VISIBLE_DEVICES` is set to an empty string.
**Note:** `GGML_VK_VISIBLE_DEVICES` indexes the raw `vkEnumeratePhysicalDevices` order, before the
discrete-GPU filter — document when selecting a device.
**Status:** the Vulkan backend is **not** claimed as working on the NVIDIA Vulkan ICD in this snapshot;
the fix is expected to come from upstream first.

> **Update (meta#85):** the RDNA4 (RADV) wrong-token issue previously tracked (note: on the current tip RDNA4's bare greedy prompt can still differ from the CPU by rounding, like RDNA3 — the flash-attention sweep is clean on both)
> under this handle has been fixed by the lane 9 graft — see the measured device
> matrix in `docs/VULKAN-BACKEND.md`. RDNA4 (RADV) now reproduces the CPU output
> exactly. The NVIDIA ICD drift and the empty-`GGML_VK_VISIBLE_DEVICES` segfault
> remain as described above.

---

## `meta#85` — Vulkan on AMD RDNA3 (RX 7900 XT, RADV): flash attention verified, two fp16 overflows fixed, one prompt still rounding-sensitive

**What was reported.** The grafted Vulkan backend gave ` 寿司<|channel>…` for the bare
prompt below on an RX 7900 XT while RDNA4 and NVIDIA reproduced the CPU's
` Europe.\n<|channel>thought\n<channel|>It appears there is a`, and `test-backend-ops`
failed 208 of its 362 `FLASH_ATTN_EXT` cases on the device (NMSE 0.17-0.19 on every
K/V type and head size, plus `inf` for `q8_0` K/V at batch 1).

**What it was.** Three separate things, none of them the RDNA3 flash-attention kernel
the report pointed at:

1. *The sweep's reference was wrong, not the backend.* The old harness filled the
   attention mask with uniform random values; ik's CPU flash-attention kernels only
   implement 0 / -inf masks (they binarise or mis-scale anything else), so every
   `mask=1, max_bias=0` case on the head sizes those kernels cover failed against a
   wrong CPU answer -- on this device and on every other one. With a 0 / -inf mask the
   same shapes pass. Current mainline `test-backend-ops` on the same card passed
   5179/5179 `FLASH_ATTN_EXT` cases against its own CPU, and a mainline build from the
   graft's era (identical flash-attention SPIR-V, byte for byte) passed 4420/4420, which
   is what pointed at the reference rather than the kernel.
2. *Two real fp16 overflows in the integer-dot (MMQ) scalar flash-attention path*, the
   path taken for `q8_0` / `q4_0` K/V at batch 1 with the default fp16 accumulator:
   the Q block scale was computed in fp16 (a denormal `qd` made `1/qd` infinite;
   mainline fixed this later as "FA MMQ should use fp32 for Q quantization
   calculations"), and the 32-wide int8 dot product (up to 32*127*127 for `q8_0`) was
   converted to fp16 before scaling. Both now compute in fp32. These are device
   independent; they only showed here because nothing had run the sweep with a valid
   reference before. They do not touch f16 K/V, which is what the Gemma-4 run uses.
3. *The bare prompt is rounding-order sensitive on Gemma-4.* With the backend
   op-exact (`GGML_VULKAN_CHECK_RESULTS` over the whole Gemma-4 run, prefill and four
   decode steps: flash attention within 2e-4 of the CPU, every other op within 1e-6,
   mat-vec at the quantized-activation noise floor of a few 1e-3, and every one of
   those mat-vec shapes passing the op sweep standalone), the bare prompt still gives
   ` 寿司…` on Vulkan.
   Switching the Vulkan mat-vec kernels (`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`) moves
   it to ` Europe<|channel>…` -- a different rounding order, not a different answer --
   without reaching the CPU string, and the same switch leaves a mainline build's
   output unchanged. A well-posed prompt is exact: the chat-formatted `What is the
   capital of France?` gives `<|channel>thought\n<channel|>The capital of France is
   Paris.` on the CPU and on Vulkan, and Qwen3 8B `The capital of France is` gives
   ` Paris. The capital of the United States is Washington, D` on both, with `-fa on`
   and `-fa off`. The remaining "wrong" tokens are the model's low-confidence
   continuation of a prompt it was not trained on, decided by rounding order.

**Repro (still differs, by the nature of the prompt):**
```
llama-cli -m gemma-4-12b-it-Q4_0.gguf -p "The capital of France is" -n 12 --temp 0 --seed 0 -ngl 99   # Vulkan build, RX 7900 XT, RADV
```
**Symptom:** ` 寿司<|channel><|channel>thought…` vs the CPU's ` Europe.\n<|channel>thought…`.
The head/tail loopback on the same card gives the single-process Vulkan output, so the
stage transport is not involved.

**Repro (exact):**
```
llama-cli -m gemma-4-12b-it-Q4_0.gguf -p "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n" -n 12 --temp 0 --seed 0 -ngl 99
build/bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT    # 830 cases, all pass, on the RX 7900 XT
build/bin/test-backend-ops -b Vulkan0                      # 2144/2152; the 8 left are the pre-existing CPY f32->q4_0/iq4_nl, iq4_xs / bf16 MUL_MAT, iq4_xs MUL_MAT_ID and PAD cases
```
**Status:** the Vulkan backend **is** claimed as working on RDNA3 (RADV) in this
snapshot, with the caveat above about bare-prompt token matching. `-fa off` remains
broken for Gemma-4 on the CPU (`meta#86`), so it is not a usable control for this
model. Still owed on the device, cut short by a host outage: a perplexity comparison
CPU vs Vulkan, and the CPU-only batch-size / thread-count variation of the bare prompt
that would show the flip is not specific to the GPU.

**CPU-side observations from the same work** (not fixed here): ik's generic CPU flash
attention returns NaN for `q8_0` / `q4_0` K/V when there is no mask or when
`max_bias > 0`; the iqk kernels' mask contract is 0 / -inf only (see the harness notes
in `docs/VULKAN-BACKEND.md`). Neither combination is emitted by a graph.

## `meta#88` — qwen4exp on Vulkan (any vendor): degenerate output on UD-IQ4_XS, flash-attention-independent

**What was reported.** On the published tree (`20c308ca`), RX 7900 XT x4 (RADV):
Qwen3.8-Flash-Next UD-IQ4_XS via Vulkan full offload emits `**:** **:** ;**;**…`;
`-ngl 12` emits empty/immediate-EOG. Both with flash attention on and off. On the
same host and build, CPU (`-ngl 0`) is fully coherent (4.3 t/s), and an out-of-fork
HIP build on the same card is coherent (20.1 t/s) — evidence that the quant and the
weights are fine and the fault is in the fork's Vulkan compute path for this arch.
The library itself is exact: 1224/1224 tensor hashes match the monolith it was
sliced from and the full 48-block hidden state is byte-identical. The same build
reproduces the **identical** degenerate output on NVIDIA Vulkan (4x RTX 5060 Ti,
coopmat1 path) while the CUDA path on the same box is coherent (23.2 tok/s), so
this is not an AMD-specific fault. Raw logs are attached to `meta#88`.

**How the docs came to claim otherwise.** The "qwen4exp: Vulkan verified — RDNA3"
row (README, `docs/HF-MODEL-CARDS.md`) was extrapolated from the lane-9 RDNA3
evidence, which covered flash-attention shapes emitted by Gemma-4 12B (830 cases)
and a Qwen3 8B run — qwen4exp itself was never token-run on RDNA3 (or RDNA4) before
this matrix. Not a regression: an unsupported generalization, now falsified by the
published-tree check. All per-arch backend rows now state the model + quant + host
actually token-run on the published tree.

**What it is.** Under triage, and broader than the title suggests. Forcing all 24
BF16 `indexer.k_proj` tensors to CPU on the NVIDIA run changes nothing — the BF16
placement hypothesis is refuted. Flash attention on and off both reproduce, so the
meta#85 reduction path is not the culprit either. The identical output across
vendors points at a qwen4exp-specific fault in the Vulkan backend itself (op
coverage or graph construction for the SSM/gated-delta path), not a device quirk
in either driver. Discriminators still worth running on an RDNA3 host: disable
integer-dot product; force SSM tensors to CPU via `-ot`; single-GPU `--device` to
exclude the cross-card tensor-split mapping (device order differs from HIP order);
op-level localization against the CPU reference with `GGML_VULKAN_CHECK_RESULTS`.

**Workaround.** Run this model where it is coherent on the published tree — CUDA
(monolith or ring) or CPU; the HIP result above is an out-of-fork data point, not
a supported backend. **Status.** Open. Related: `meta#85` (different path, same
device class).

---

## `meta#88` — Vulkan: hybrid MoE models with a hyper-connection mixer (Qwen3.8-Flash-Next) produced degenerate text on every vendor — fixed

**What was reported.** The `qwen4exp` architecture (Qwen3.8-Flash-Next, `UD-IQ4_XS`,
experts and the per-layer token embedding on the CPU) generated
` **:** **:**;**;** andell;**ell;** ** ** …` on the Vulkan backend on AMD RDNA3 (RADV)
and on NVIDIA (`KHR_coopmat`), with `-fa on` and `-fa off`, while the same file gave
`:\n\n1. **First Law** – Energy cannot be created or destroyed…` on the CPU and on CUDA.
`-ngl 12` gave an empty completion. Forcing the model's only `bf16` tensors (the
sparse-attention indexer projections) to the CPU changed nothing, and the per-layer
library reproduced the monolith exactly.

**What it was.** One op, `GGML_OP_MULTI_ADD`, ik's "sum `n_add` consecutive column
chunks of every row". The graph builders hand it a `[ne0, nrows]` *view* whose row
stride spans the `n_add` chunks — the MoE expert combine, the hyper-connection stream
mix (`hc = 4` streams of `n_embd`), the pooled indexer keys. `ggml_nbytes()` of that
view ends after the last row's *first* chunk, and the generic Vulkan dispatcher bound
exactly that many bytes as the `src0` descriptor range. The kernel's reads of chunks
`1..n_add-1` of the last row fell outside the bound range, which the hardware clamps to
zero on both vendors. At decode there is one row, so every hyper-connection mix in
every layer returned `stream_0 / hc` instead of the mean of the four streams; at
prefill only the last token's row was wrong. Nothing had exercised the op on the GPU
before: the MoE combine runs on the CPU whenever the experts do, and no earlier
Vulkan-verified model carried a mixer that runs it on the GPU with one row.

The `-ngl 12` empty completion was the same fault seen through a partial offload
(the mixer of the offloaded layers still ran on the GPU); it is gone with the fix.

**Fix.** `ggml_vk_op_f32` binds `src0` for `MULTI_ADD` over the range the kernel reads,
`nb[1]*(nrows-1) + n_add*ne0*sizeof(float)` (bounded by the buffer). Also found on the
way: the ik `MULTI_ADD` / `MUL_MULTI_ADD` pipelines were created inside the
`device->multi_add` gate of mainline's fused-ADD chain (a device property, plus
`GGML_VK_DISABLE_MULTI_ADD=1`), while `supports_op` accepted the ops unconditionally,
so on a device without `shaderRoundingModeRTEFloat16` or with that knob set the run
aborted with `Missing op: MULTI_ADD`. The pipelines are now created unconditionally.

**Repro (op level; failed before, passes now):**
```
build/bin/test-backend-ops -b Vulkan0 -o MULTI_ADD
#   MULTI_ADD(type=f32,ne0=2560,n_add=4,nrows=1):  NMSE = 2.94  -> OK   (one row: every value wrong)
#   MULTI_ADD(type=f32,ne0=2560,n_add=4,nrows=32): NMSE = 0.024 -> OK   (only the last row wrong)
```
**Repro (model):**
```
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf -ngl 999 -ot "blk\.[0-9]+\.ffn_.*_exps\.=CPU" -ot "per_layer_token_embd\.weight=CPU" -c 2048
# prompt "The three laws of thermodynamics are", greedy, 48 tokens, one RTX 50-class card under Vulkan
```
Before: ` **:** **:**;**;** andell;**ell;** ** ** ** **;** …`.
After: ` often summarized as follows:  \n1. You can't win (you can't get something for
nothing).  \n2. You can't break even …` — coherent, and not the CUDA string, for the
reason the next paragraph gives. With `output.weight` on the CPU, or `-ngl 36`, or
`-ngl 12`, the same build gives `:\n\n1. **First law** – energy cannot be created or
destroyed, only transformed. \n2. **Second law** – in any energy transformation, some
energy becomes unavailable for work, increasing entropy. \n3. **Third law`, which is the
CUDA text for 48 tokens up to a `Law`/`law` step that CUDA itself scores 0.500 / 0.498.

**Why the full-offload text still differs from CUDA.** The prompt's first token is a
three-way tie. CUDA on the same card scores `':' 0.0981, ' fundamental' 0.0971,
' often' 0.0967`; the fixed Vulkan build scores the same three at `0.0947 / 0.0928 /
0.0974` and, with `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`, at `0.0971 / 0.0993 /
0.0959` — each mat-vec rounding order picks a different member of the tie, and the
continuation is decided from there. Every per-step top-5 agrees with CUDA to within
about 0.5 percentage points. This is the `meta#85` class of difference, not a kernel
fault; a well-posed prompt does not show it.

**Second defect, found while chasing DeepSeek-V4-Flash (`deepseek4`) under the same
handle — real, but not that model's cause.** `ggml_get_rows_ext(a, idx,
same_type, dim0 = true)` is a `GET_ROWS` node with `op_params[0] = 1` meaning a gather
*along dim 0 within each row* (`out[i, r] = a[idx[i, r], r]`), which `deepseek4` uses at
decode to cut the sparse-attention mask to the selected cells. The Vulkan `supports_op`
looked only at the source type and the get_rows shaders never read `op_params`, so the
backend accepted the node and computed a plain row gather over rows that do not exist —
a wrong mask on every decode step (prefill takes the other, `n_tokens > 1`, path). Both
ik-only forms are now declined (`op_params[0] == 1`, and a
same-type gather whose output type has no pipeline, which would have aborted) and run on
the CPU. Op level: `GET_ROWS(type=f16, n_kv=4096, n_tokens=1, top_k=2048, dim0=1)` failed
with NMSE = inf and is now reported unsupported; the same-type row gather passes.
DeepSeek-V4-Flash itself (`UD-Q4_K_XL`, experts on the CPU, `-fa on -ctk q8_0 -amb 512`)
still generates `\n\n\nimportimportimport\n##…` on the build with both fixes (7.8 tokens/s
on one RTX 50-class card), so that graph has at least one more Vulkan-side fault. It is not
in the mixer path: `deepseek4` mixes through `HC_PRE` / `HC_POST` (declined, CPU) and
`MUL_MULTI_ADD` (passes at its shapes, see the harness), and never emits `MULTI_ADD` with
experts on the CPU. Localizing it needs a `GGML_VULKAN_CHECK_RESULTS` pass over that graph;
not done here.

**Status:** fixed. The Vulkan backend on the `qwen4exp` graph is claimed as working
for the single-card + CPU-experts configuration above (NVIDIA, `KHR_coopmat`, measured
at 8.8-10.1 tokens/s decode on a shared host against 23.4 tokens/s for CUDA on the same
card). Not re-measured here: the RDNA3 host that showed the same symptom; the fix is
device independent (a descriptor range), and the op test above is the check to run
there.

---

## CPU: `iq4_xs`, `iq4_kss`, `iq5_ks` mat-vec kernels disagree with their own dequantizers at `n < 32` (found under `meta#88`)

The `test-backend-ops` sweep has always failed `MUL_MAT(type_a=iq4_xs, n=1)` and
`MUL_MAT_ID(type_a=iq4_xs, n=1)` on every Vulkan device, recorded here and in
`docs/VULKAN-BACKEND.md` as pre-existing backend failures. They are not. The harness
compares each backend against the CPU, and on an AVX2 host the CPU's own `iq4_xs`
kernel disagrees with `iq4_xs -> f32` dequantization by NMSE ~3e-2 at `n = 1` and
`n = 7` — seven times the 4-bit quantization noise of ~4e-3 measured against the
original weights — and agrees (1.5e-5) at `n = 32`. The Vulkan `iq4_xs` mat-mat case
(`n = 32`) passes against the same CPU exactly where the CPU agrees with itself;
`GET_ROWS(iq4_xs)` passes, so the shaders read the blocks correctly. `iq4_kss` and
`iq5_ks` show the same pattern; `iq2_kt` sits at 2.5e-3 at every `n`. Every other
quantized type the CPU can quantize on its own agrees with its dequantizer at ~1.5e-5.

**Repro:**
```
build/bin/test-quant-matmul -v        # CPU only; prints NMSE(kernel vs dequant) per type at n = 1, 7, 32
```
**Symptom:** `iq4_xs m=2560 n=1 k=2560 NMSE(kernel vs dequant) = 3.19e-02 FAIL` (and the
`iq4_kss`, `iq5_ks`, `iq2_kt` lines); `62/70 type/shape combinations consistent`.
**Status:** not fixed here. It affects any `iq4_xs` tensor computed on the CPU at small
batch — the routed experts of a `UD-IQ4_XS` model run with `-ot exps=CPU` at decode are
the common case, on every backend build, since the CPU does that matmul. The Vulkan
sweep entries for `iq4_xs` should be read as "reference disagrees", not "backend wrong".

---

## `meta#96` — DeepSeek-V4-Flash on NVIDIA Vulkan (`coopmat1`) produces degenerate token output

On NVIDIA Vulkan (`coopmat1`), DeepSeek-V4-Flash (`deepseek4`, measured with the UD-Q4_K_XL monolith, one RTX 5060 Ti + CPU experts, `-fa on -ctk q8_0 -amb 512`) generates degenerate repetitive token output (RDNA3/ANV not measured; not quant-related). From the issue's exclusion list, the defect is not in MULTI_ADD (`meta#88`), not dim-0 GET_ROWS (a separate decline change that left output unchanged), not MXFP4, and not the quant. The CUDA backend on the same host produces coherent generation (CPU generation was not measured; only CPU hidden states were compared).

**Measured configuration:**
DeepSeek-V4-Flash UD-Q4_K_XL monolith run on NVIDIA Vulkan (`coopmat1`, one RTX 5060 Ti 16 GB) with experts offloaded to CPU via `-ot` (the `-ot` regex must include block 0 on 16 GB cards), `-fa on -ctk q8_0 -amb 512`.
**Symptom:** Emits degenerate repetitive sequences (e.g. `\n\n\nimportimportimport\n##…`) at ~7.8 tokens/s instead of coherent completions.
**Workaround:** Run DeepSeek-V4 on CUDA.

---

## `meta#100` — `llama-server --model-dir` leaves `model_name` and `model_path` empty (conversation header and assistant-message badge show no name)

When `llama-server` is launched with `--model-dir <path>` instead of `-m <file>`, `params_base.model` is empty, so `/props` leaves `model_name` and `model_path` empty (`examples/server/server.cpp:1079-1082, 1116-1117`). In the WebUI, the conversation header and assistant-message badge show no name (`examples/server/webui/src/components/ChatScreen.tsx:261`). (Note: the WebUI start screen never shows a model name for any server). While model generation is unaffected, the OpenAI-compatible response `model` fallback (`examples/server/server.cpp:1162`) uses the same empty source and is unverified.

**Expected behavior (from code reading, not yet reproduced):**
```bash
llama-server --model-dir /path/to/layer-library -ngl 99 --port 8080
curl -s http://127.0.0.1:8080/props | jq '{model_name: .model_name, model_path: .model_path}'
```
**Expected symptom:** Expected to return empty strings `""` for both `model_name` and `model_path`. In the WebUI, the conversation header and assistant-message badge will have no model name displayed.
**Status:** Generation is unaffected; the `model` field in OpenAI-compatible completions is unverified.

---

## `meta#102` — DeepSeek-V4 GGUFs with unsloth vision bias tensors fail to load ("wrong number of tensors")

On 2026-09-04 (commit `e1efe867`), upstream Unsloth added 43 vision expert-routing bias tensors (`blk.N.exp_probs_b_vl.bias`) to `DeepSeek-V4-Flash-Vision-Exp-GGUF`, increasing total tensor count from 1328 to 1371. The fork model loader enforces strict tensor count validation (`src/llama-model-loader.cpp:1456-1458`). Because the fork loader does not construct these vision tensors, loading files containing them is expected from code reading to fail.

**Expected behavior (from code reading, not yet reproduced):**
```bash
llama-cli -m DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00001-of-00005.gguf   # Unsloth rev >= e1efe867 (post-09-04)
```
**Expected symptom:** Expected from code reading (`src/llama-model-loader.cpp:1457`) to abort during model load with:
```
done_getting_tensors: wrong number of tensors; expected 1371, got 1328
```
(where expected is the tensor count in the file and got is the number of tensors created by the loader).
**Supported distribution:** Use the xAutonomics DSv4 library (built from revision `37044a3c`, pre-09-04 1328-tensor layout).
