# Expert-tensor disaggregation: port map onto this fork

Source lineage: the mainline-lineage llama.cpp tree, commits `8e5cf15c`
(initial P1/P2 port), `0d345b0c` (transport hardening: CAPS handshake +
reconnect/retry), `582a8afb` (capability-handshake completion) and `773df43e`
(connection-local allocators).

This document is the written mapping produced **before** any code was edited,
so a reviewer can check each hook against its source rather than reading the
diff cold.

## What the feature is

A MoE model's routed-expert FFN is the bulk of its weights but only a thin
slice of its compute per token. Expert-tensor disaggregation moves the routed
experts of a configured layer range into a separate process (`expert-server`),
which may live on another host. The attention side keeps the router, the
top-k selection, the expert weights, the shared/dense experts and the KV
cache; per covered layer it ships `{ hidden, topk ids, topk weights }` and
receives the accumulated routed-expert output.

The design gate is **token exactness**: a split run must produce bit-identical
logits to the same model run single-process. That is achievable because the
split is placed after weight normalisation, `f32` crosses the wire unrounded,
and the server mirrors the local graph's op sequence on the same CPU kernels.

## Touch-point map (source hook -> this fork)

| # | Source (mainline lineage) | This fork | Nature of the change |
|---|---|---|---|
| 1 | `src/llama-experts-remote.h` | same path | verbatim; no fork-specific types in it |
| 2 | `src/llama-experts-remote.cpp` | same path | near-verbatim; only the `ggml.h` / `llama-impl.h` includes are lineage-checked |
| 3 | `src/CMakeLists.txt` (add the TU) | same file | 1 line |
| 4 | `llm_graph_context::llm_graph_context()` warmup line in `src/llama-graph.cpp` | `llm_build_context::llm_build_context()` initialiser list, `src/llama-build-context.cpp` | same one-line guard, different class |
| 5 | `llm_graph_context::build_moe_ffn()` hook in `src/llama-graph.cpp` | `llm_build_context::llm_build_moe_ffn()` in `src/llama-build-context.cpp` | same split point (right after `ggml_build_forward_expand(graph, weights)`, before `cur` is reshaped to 3d); extra guards for this fork's wider MoE surface |
| 6 | central `create_tensor` lambda in `llama_model::load_tensors()`, `src/llama-model.cpp` | `create_tensors_helper::create_tensor()`, `src/llama-load-tensors.cpp` | same idea (OR in `TENSOR_SKIP`), but keyed off the tensor *name* because this fork's helper does not carry a `bid` |
| 7 | `examples/stage-runner/expert-server.cpp` | same path | adapted: this fork's ggml has no threadpool API and an older backend-device API; the mirrored op sequence is this fork's, not mainline's |
| 8 | `examples/stage-runner/expert-check.cpp` | same path | rewritten against this fork's older `gpt_params` common API |
| 9 | `examples/stage-runner/CMakeLists.txt` targets | same file | additive; this fork's `llama-stage-runner` target is untouched |
| 10 | `examples/CMakeLists.txt` `add_subdirectory(stage-runner)` | already present | no change needed |

Nothing else in the source surface is needed. `gslot_client.h` is *not*
required: the expert server and its client talk their own framed TCP
protocol and never reference the gslot runtime. That stays a separate lane.

## Per-hook detail

### 4. Warmup expert count

Both lineages force `n_expert_used = n_expert` during the warmup decode so
every expert page is faulted in. With a remote expert server that is both
pointless (those weights live in another process, prewarmed there) and fatal:
the server rejects a call whose `k` exceeds its `n_topk` bound. The guard is
the same in both trees; only the constructor differs.

### 5. The MoE graph hook

The split point is identical in both lineages, and it is the only point at
which it can be bit-exact: everything before it (router matmul, gating
function, selection bias, top-k, softmax-of-weights, sum-normalisation,
weight scaling) runs locally through untouched code, and everything after it
is what the server mirrors.

This fork's `llm_build_moe_ffn` supports a wider surface than the source
lineage's `build_moe_ffn`, so the remote path guards more:

* fused up/gate expert tensor (`up_gate_exps`) - unsupported, assert
* per-expert biases (`up/gate/down_exps_b`) - unsupported, assert
* per-expert output scales (`down_exps_s`) - unsupported, assert
* `weight_before_ffn` archs - unsupported, assert
* grouped expert routing / expert groups - unsupported, assert
* activation must be SILU
* this fork's `add_input` residual add is applied **after** the remote call,
  so the remote path returns the same thing the local path would

The local (non-remote) path is not touched at all, including this fork's
fused MoE kernels: an uncovered layer takes exactly the code it took before.

### 6. Loader placement

The source lineage hooks the one `create_tensor` lambda and tests
`tn.bid` + a `_exps.weight` name suffix. This fork routes every architecture
through `create_tensors_helper::create_tensor(ctx, name, ne, flags)`, which
receives the fully-formed tensor name and no layer id, so the hook parses
`blk.<N>.` out of the name and applies the same `_exps.weight` suffix test.
It ORs in `TENSOR_SKIP`, which in this fork's model loader returns `nullptr`
and subtracts the tensor from the load budget - exactly the source behaviour.

One fork-specific benefit falls out for free: this fork can *merge* separate
`ffn_up_exps` / `ffn_gate_exps` into a fused tensor at load time, but only
when `flags == 0`. A covered layer therefore never takes the merge path, and
`up_gate_exps` is reliably null on the remote path.

### 7. The expert server

The server is where the two lineages diverge most, because it is written
directly against ggml rather than against llama:

| Source lineage uses | This fork | Resolution |
|---|---|---|
| `ggml-cpu.h`, `gguf.h` as separate headers | both folded into `ggml.h` | drop the includes |
| `ggml_graph_plan(gf, n_threads, threadpool)` + `ggml_threadpool_*` | no threadpool API; `ggml_graph_plan(gf, n_threads)` | drop the threadpool and the `--poll` flag |
| `ggml_backend_dev_count()` / `ggml_backend_dev_get()` / `ggml_backend_dev_init()` | older registry API: `ggml_backend_reg_get_count()` / `ggml_backend_reg_get_name()` / `ggml_backend_reg_init_backend()` | enumerate through the registry, skip the CPU backend |
| `gguf_find_tensor` / `gguf_find_key` return `int64_t` | return `int` | narrow the locals |
| `ggml_swiglu_split(gate, up)` for the activation | the local graph uses `ggml_fused_mul_unary(gate, up, SILU)` | **mirror this fork**, not the source |
| chained `ggml_add` of per-expert 2d views for the weighted sum | the local graph uses `ggml_multi_add` (or fused `ggml_mul_multi_add`) | **mirror this fork**, selectable |

The last two rows are the substance of the port. Byte-exactness is a property
of matching the *client's* op sequence, so the server grows two switches,
`--fmoe` and `--mmad`, defaulting to on to match this fork's client defaults
(`fused_moe_up_gate` and `fused_mmad` are both on by default here, and the
client can turn them off with its existing flags). The wire protocol is
unchanged and stays interoperable with the source lineage at version 1.

### 8. The exactness harness

`expert-check` is a thin greedy-decode harness that prints, per step, the
token id and an FNV-1a 64 hash of the whole logits vector, and can dump raw
logits. It carries no feature logic, so it is re-expressed against this
fork's older common API (`gpt_params`, `gpt_params_parse`,
`llama_init_from_gpt_params`, `llama_n_vocab`, `llama_token_is_eog`) rather
than ported line by line.

## Guarding

With no expert server configured the feature is inert:

* config is parsed once from the environment into a function-local static;
  with `LLAMA_EXPERTS_REMOTE` unset it returns `enabled = false` immediately
* the loader hook is behind `cfg.enabled` and short-circuits before it looks
  at the tensor name
* the graph hook is behind `cfg.layer_covered(il)`, which is `false` for
  every layer when disabled, so no node is added and no op is replaced
* the warmup guard only weakens when a server is configured
* the two new executables are additive targets; no existing target changes

## Deferred

* GPU-side expert compute (`--device gpu`) is ported but untested here - it
  needs a GPU-enabled build and a free device.
* The gslot runtime is a separate lane and is not required by this feature.
* Fused up/gate expert tensors, expert biases, per-expert scales, grouped
  routing and non-SILU experts remain unsupported on the remote path in both
  lineages; they assert rather than silently mis-compute.

## What the port actually needed, beyond the map

Two things the map above did not predict.

**Fusion is not bit-neutral in this tree.** The source lineage has one
routed-expert tail, so its server mirrors one op sequence. This tree has
three forms of it and turns two of them on by default. Measured on a small
MoE with a 12-step greedy decode:

| client | server | logits |
|---|---|---|
| fused (default) | fused (default) | byte-identical |
| unfused (`-no-fmoe -no-mmad`) | unfused (`--fmoe 0 --mmad 0`) | byte-identical |
| fused | unfused | **differ** |
| unfused | fused | **differ** |

and the two *local* forms differ from each other as well, with no expert
server anywhere - a fused local run and an unfused local run do not produce
the same logits on this tree. So the server's `--fmoe` / `--mmad` switches
are load-bearing, not belt and braces: an expert server must be told which
form its client built, and the defaults are chosen so that the common case
(client defaults on both sides) needs no flags at all.

**One pre-existing null dereference.** `llm_build_std_moe_ffn` guarded its
`up_exps->extra` and `gate_exps->extra` lookups but not `down_exps->extra`.
Nothing could reach that before, because the three tensors were always
present together; a layer served remotely makes all three null, and the
unguarded one segfaulted during graph build. Fixed with the same guard the
other two already had.

## Proofs

All on CPU, all with the feature built in.

**Regression, feature off.** A greedy completion from a small MoE, pre-port
binary and post-port binary, no expert server configured: byte-identical
stdout. Repeated with `-no-fmoe -no-mmad`: byte-identical. The local path
does not move.

**Dense path.** The head/tail layer-library loopback over a 48-layer dense
model, split 24/48, still emits its recorded completion piece for piece
(` Europe`, `.`, `\n`, `<|channel>`, `thought`, `\n`, `<channel|>`, `It`,
` appears`, ` there`, ` is`, ` a`), and the pre-port and post-port binaries
agree on an unrelated prompt as well.

**Token exactness, single server.** Small MoE, 28 layers all served
remotely over loopback. The attention side skipped 84 tensors (6.89 GiB of
routed experts) at load. Twelve decode steps: per-step token ids and logits
hashes identical to the single-process baseline, and the raw logits dumps
compare equal byte for byte (4.5 MiB).

**Token exactness, large model.** A 48-layer, 512-expert MoE of about 94 GB
across three shards, all 48 layers served remotely. The attention side
skipped 144 tensors (55.4 GiB) and loaded in 4.7 s instead of 67.7 s.
Twelve decode steps, 11.4 MiB of raw logits: byte-identical.

**Multi-endpoint (P2).** The same small MoE split across two expert servers,
layers 0-13 and 14-27: byte-identical logits again. A deliberately
overlapping coverage spec aborts at config-parse time naming the layer and
both endpoints, as designed.

**Build and tests.** The four required targets build clean with zero
warnings in any touched file, and the existing model-loader-metadata and
stage-manifest tests pass.
