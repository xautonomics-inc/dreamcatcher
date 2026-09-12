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
this is not an AMD-specific fault. Raw logs: `amd:` and `nvidia:`
`/fast/build/p29-public/verify/`.

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
in either driver. Discriminators still worth running on the AMD host: disable
integer-dot product; force SSM tensors to CPU via `-ot`; single-GPU `--device` to
exclude the cross-card tensor-split mapping (device order differs from HIP order);
op-level localization against the CPU reference with `GGML_VULKAN_CHECK_RESULTS`.

**Workaround.** Run this model where it is coherent on the published tree — CUDA
(monolith or ring) or CPU; the HIP result above is an out-of-fork data point, not
a supported backend. **Status.** Open. Related: `meta#85` (different path, same
device class).
