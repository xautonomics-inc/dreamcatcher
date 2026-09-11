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

## `meta#84` — Vulkan backend is wrong on RADV GFX1201 (RDNA4); drifts on NVIDIA Vulkan; empty `GGML_VK_VISIBLE_DEVICES` segfaults

The Vulkan backend (inherited from upstream, not a fork regression) produces
wrong tokens on RADV GFX1201 (RDNA4) for `gemma4` and other archs, and drifts
from the CPU reference on the NVIDIA Vulkan ICD. The stage split is self-consistent
on Vulkan (head/tail == single-process), so the transport is fine; the backend
kernels are not. The Vulkan driver on GFX1201 reports "not a conformant Vulkan
implementation" and no matrix cores, and is also slower than CPU there.

**Repro (wrong tokens):**
```
llama-cli -m gemma-4-12b-it-Q4_0.gguf -p "The capital of France is" -n 12 --temp 0 --seed 0 -ngl 99   # Vulkan build, RADV GFX1201
```
**Symptom:** wrong tokens (e.g. ` 寿司<|channel><|channel>thought\n<channel|>The capital of France is`),
not matching the CPU reference; `-b 1 -ub 1` gives byte-identical wrong output (so it is not
the `meta#81` batched-prefill signature).

**Repro (segfault):**
```
GGML_VK_VISIBLE_DEVICES= llama-cli -m <any-model.gguf> ...
```
**Symptom:** exit 139 (segfault) when `GGML_VK_VISIBLE_DEVICES` is set to an empty string.
**Note:** `GGML_VK_VISIBLE_DEVICES` indexes the raw `vkEnumeratePhysicalDevices` order, before the
discrete-GPU filter — document when selecting a device.
**Status:** the Vulkan backend is **not** claimed as working for the archs above in this snapshot;
the fix is expected to come from upstream first.
