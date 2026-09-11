# The Vulkan backend in this fork

This fork's Vulkan backend is a graft: `ggml/src/ggml-vulkan.cpp` and the whole
`ggml/src/vulkan-shaders/` tree were replaced with the 2026 mainline ggml backend and
adapted to ik, rather than merged hunk by hunk. The backend they replaced was last
synced from mainline in mid-2025 and had drifted far enough that several devices
produced wrong output rather than a clean refusal.

This document is the part of that work that is not in the code: what the graft brings,
what it deliberately refuses to do, what it costs, and how to check it on a new device.

## What the graft contains

- The 2026 mainline backend: per-shader compilation, the current coopmat / coopmat2 /
  integer-dot / bfloat16 feature tests, and a `supports_op` table that matches what
  current graphs actually emit.
- ik's own quantisation types `IQ4_K`, `IQ5_K` and `IQ6_K`, with dequant, `get_rows`,
  `mul_mat_vec` and `mul_mm` kernels and entries in the `supports_op` type allowlists.
  These are marked `ik-port` in the source.
- ik's fused ops, which have no mainline equivalent: `FUSED_UP_GATE`,
  `MOE_FUSED_UP_GATE`, `MUL_MULTI_ADD` and `FUSED_MUL_UNARY`, the last mapped onto the
  mainline GLU split path for `SIGMOID`.
- `ggml-vulkan-ik-compat.h`, which holds the shims the grafted code needs here. ik is
  pre-module-split, so mainline's `ggml-cpu.h`, `GGML_LOG_*` and backend-registry entry
  points do not exist and are provided there.

The fused up/gate kernels work, but they are slower than the decomposed graph at decode
(measured on an MoE model: 121.9 t/s fused against 206.8 t/s split; 11-17% slower for
dense). So when any layer is offloaded to Vulkan, `fused_up_gate` and `-fmoe` default off
and the split graph is used; `GGML_VK_ALLOW_FUG=1` and `GGML_VK_ALLOW_FMOE=1` opt back in.
Both choices are logged once at context creation.

## Building

```
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j
```

Build dependencies, in addition to the Vulkan loader and `glslc`:

- **SPIR-V headers** (`spirv-headers` on Debian/Ubuntu). The refreshed backend includes
  `<spirv/unified1/spirv.hpp>`. The include has `__has_include` fallbacks for the LunarG
  and flat layouts, but one of them has to be present. This is new; the pre-graft backend
  did not need it.

The shader generator now emits one translation unit per shader with depfile tracking, so
the Vulkan section of `ggml/src/CMakeLists.txt` differs substantially from the old one.

## What the backend declines, and why

A backend that returns wrong numbers is worse than a backend that says no: `supports_op`
returning `false` costs a kernel, and the op falls back to the CPU. Four things are
declined here.

### `SSM_CONV` and `SSM_SCAN`

ik's `ssm_conv` is a five-tensor stateful op (state, x, conv weights, sequence ids, saved
steps). The grafted mainline kernel reads `src0` and `src1` only and ignores the rest, so
every state-space layer silently produced garbage instead of failing. `ssm_scan` has the
same kind of source-layout mismatch. Both are declined, so they run on the CPU: measured
on a small hybrid model after the fix, perplexity 3.1226 on Vulkan against 3.1247 on the
CPU, where before the fix the output was visibly broken.

### GLU ops whose operands need broadcasting

In the split (two-source) form, the mainline GLU pipelines read `src0` and `src1` through
one shared index, so the two operands must have the same shape. ik's CPU kernel is more
permissive and broadcasts, and some models rely on it -- a per-head attention gate that
multiplies a `[1, H]` gate against a `[D, H]` value, for instance. Accepting that shape
made the shader read past the end of the smaller operand. `supports_op` now requires
`ggml_are_same_shape(src0, src1)` for the split form; the broadcasting cases fall back to
the CPU and everything else keeps running on the GPU.

### `VK_NV_cooperative_matrix2` (off by default)

The grafted coopmat2 matmul and flash-attention pipelines do not compute the right thing
here. On an NVIDIA card that advertises the extension, a 12B model at temperature 0
answers a capital-city prompt with unrelated text; the same build on the same device with
coopmat2 declined reproduces the CPU reference token for token, and so does the
`KHR_coopmat` path on that device.

So the extension is declined unless `GGML_VK_ENABLE_COOPMAT2=1` is set, and a line is
logged when a device advertises it. The device keeps `KHR_coopmat`, so this costs a
faster matmul on NVIDIA rather than acceleration itself. Whoever fixes the coopmat2
kernels against ik's graph shapes can flip the default back.

### `IQ4_KS` and `IQ4_KT` (no Vulkan offload)

This is the one capability the graft loses outright. Upstream added Vulkan support for
these two types in six dedicated `.comp` shaders, because both put a per-row f32 scale
ahead of the row's blocks and so cannot use block-indexed addressing.

The graft replaces ik's per-type-shader architecture with the 2026 mainline
*parameterized* idiom: one `mul_mat_vec.comp` / `get_rows.comp` / `mul_mm.comp` driven by
`DATA_A_<TYPE>` defines and per-type `dequantize()` functions in `dequant_funcs.glsl`,
whose signature (`a_offset, ib, iqs`) has no way to reach a per-row prefix. Upstream's
shaders therefore cannot simply be carried alongside. A correct port means adding a
per-row-scale addressing mode to the refreshed mat-vec / mat-mat / `get_rows` pipelines,
plus the `ggml_row_size` handling that the graft also drops.

Neither type is in the refreshed `supports_op` allowlists, so they are declined and fall
back to the CPU. That is a **loss of GPU offload for `IQ4_KS` / `IQ4_KT` models, not a
correctness problem.** `test-backend-ops` can validate such a port synthetically when
someone does it.

## Graph placement: what runs on the GPU when nothing is offloaded

The scheduler asks a backend two questions before it moves an op whose weights live in a
CPU buffer onto that backend: `supports_op`, and `offload_op`. The second one therefore
decides how much of a run with **no** offloaded layers (`-ngl 0`, or `-ot` forcing tensors
to the CPU) still executes on the GPU.

Mainline measures that batch as `ggml_nrows(op)` -- `ne[1]*ne[2]*ne[3]` -- for everything
outside a short op list. A three-dimensional activation such as
`[head_dim, n_head, n_tokens]` clears the default threshold of 32 on head count alone, at
any prompt length, so adopting mainline's predicate moved ops that ik has always computed
on the CPU onto the GPU. That is observable from the CPU side: on a 48-layer 12B model at
`-ngl 0 -ot ".*=CPU"` it took the graph from 621 splits to 733 and changed the generated
tokens, because the moved ops came back from the GPU rather than from the CPU kernels,
and the two do not agree bit for bit.

This backend keeps ik's predicate (`ne[1]`, plus `ne[2]` for `MUL_MAT_ID`), so a run that
did not ask for the GPU gets the same tokens it got before the graft.
`GGML_OP_OFFLOAD_MIN_BATCH` overrides the threshold; setting it very high pins the whole
graph to the CPU, which is the quickest way to tell a placement change from a kernel
change.

## Ops that exist only as enum ids

The graft needs to *name* ops and a type that ik's `ggml` does not declare, so the enum
commit at the base of this work adds, purely additively:

- `GGML_TYPE_NVFP4` (mainline's id 40; block layout only, no CPU kernels),
- eleven ops: `SIN`, `COS`, `ROLL`, `TOP_K`, `COUNT_EQUAL`, `IM2COL_3D`,
  `OPT_STEP_ADAMW`, `OPT_STEP_SGD`, `RWKV_WKV6`, `RWKV_WKV7`, `GATED_DELTA_NET`,
- five unary ops: `CEIL`, `FLOOR`, `ROUND`, `TRUNC`, `XIELU`.

No graph builder here emits any of them. There is no CPU kernel for any of them either,
and `ggml_compute_forward` aborts the process on an id it does not handle, so the CPU
backend's `supports_op` declines them explicitly. Anything that walks the enum -- the op
harness does exactly that -- gets a usable "no" rather than an abort part-way through.

## Measured device matrix

Gemma-4 12B Q4_0, greedy (`--temp 0 --seed 0`), 12 tokens from the same short prompt,
compared against the same build's CPU output.

| Device / driver | Matrix path | Result with this backend |
|---|---|---|
| AMD RDNA4 (RX 9070 class, RADV) | none | **Fixed.** Reproduces the CPU output exactly. The pre-graft backend produced garbage on this device. |
| NVIDIA Blackwell (RTX 50 class) | `KHR_coopmat` | **Exact.** Reproduces the CPU output token for token -- the only GPU configuration measured that does so on this hardware. |
| NVIDIA Blackwell (RTX 50 class) | `NV_coopmat2` | Wrong. Declined by default; see above. |
| Intel Arc B-series (BMG, ANV) | `KHR_coopmat` | Self-consistent and unchanged from the pre-graft backend, but does not reproduce the CPU output token for token. Not investigated further. |
| AMD RDNA3 | -- | **Known-bad in this snapshot.** Wrong output (` 寿司<|channel>…` vs the CPU reference); `test-backend-ops` pins it on `FLASH_ATTN_EXT` (208 of 216 failures, NMSE ~0.18 across all KV types incl. Gemma's head size 256). `MUL_MAT` is clean for the model's types. The head/tail loopback reproduces single-process Vulkan output byte-for-byte, so the stage transport is fine — it is the flash-attention kernel on RDNA3. See `meta#85`. |

A CPU-only run (`-ngl 0`, every tensor forced to the CPU with `-ot`) produces
byte-identical output to the pre-graft build -- with the Vulkan backend compiled in and
with `-DGGML_VULKAN=OFF`, on both an AMD Zen 3 and an Intel hybrid host. That is a
property worth keeping a test on: before the op-offload predicate was restored (see
above), it did not hold.

## Checking the backend on a new device

```
# one op family at a time
build/bin/test-backend-ops -o MUL_MAT -b Vulkan0
# the whole sweep
build/bin/test-backend-ops -b Vulkan0
```

`tests/test-backend-ops.cpp` compares every registered backend against the CPU reference
per op. It is the tool the `SSM_*` and GLU mismatches above were found with, and it is
also the only thing that exercises ik's `IQ4_K` / `IQ5_K` / `IQ6_K` Vulkan kernels. It is
wired into the test CMakeLists here; before this work it was in the tree but never built.

Measured on an Arc B580 (ANV), same host and same harness, comparing the pre-graft
backend against this one:

| | pre-graft backend | this backend |
|---|---|---|
| cases | 1628 | 1648 (the 20 extra are the newly declared unary ids, skipped) |
| passed | 865 OK + 378 not supported | 1177 OK + 207 not supported = 1384 |
| failed | 16 | 264 |
| completes? | **no** -- segfaults at `UPSCALE(ne=[512,512,3,1],scale_factor=2)` | yes |

The failure counts are not comparable head-on, because the pre-graft sweep dies before it
reaches flash attention. Per op, on the same device:

| Op | pre-graft | this backend |
|---|---|---|
| `CPY` | 22 OK / 4 FAIL | 22 OK / 4 FAIL -- **the same four cases** (f32 to q4_0 and iq4_nl, NMSE 3e-4 to 1.3e-3) |
| `FLASH_ATTN_EXT` | 88 OK / 274 FAIL | 104 OK / 258 FAIL |
| `MUL_MAT` | 11 FAIL | 2 FAIL |

So the two large failure sets are **not introduced by the refresh**: `CPY` fails
identically before and after, and flash attention fails fewer cases after. They are worth
fixing, but they are not what breaks model output on a bad path -- a device can reproduce
the CPU reference token for token while failing both.

The remaining full-sweep failures on that device are `FLASH_ATTN_EXT` (256), `CPY` (4),
`MUL_MAT` (2, `iq4_xs` and a degenerate `bf16` k=1 case), `MUL_MAT_ID` (1, `iq4_xs`) and
`PAD` (1). They are device-specific: on an NVIDIA card the same build fails only `CPY` and
`FLASH_ATTN_EXT`.

For a suspected numerical mismatch rather than a missing op, build with
`-DGGML_VULKAN_CHECK_RESULTS=ON`: every Vulkan result is then recomputed on the CPU and
compared node by node. That harness had bit-rotted against ik and is fixed here.

## Environment variables

| Variable | Effect |
|---|---|
| `GGML_VK_VISIBLE_DEVICES` | Comma-separated device indices to use. Selecting nothing is handled: host-memory pinning falls back to plain CPU buffers instead of indexing an empty list. |
| `GGML_VK_ENABLE_COOPMAT2` | Opt into `NV_cooperative_matrix2`, which is off by default because it miscomputes. |
| `GGML_VK_DISABLE_COOPMAT2` | Force it off (still honoured, and wins over the opt-in). |
| `GGML_VK_DISABLE_COOPMAT` | Disable `KHR_cooperative_matrix`. |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`, `GGML_VK_DISABLE_BFLOAT16`, `GGML_VK_DISABLE_F16` | Disable the corresponding feature path. |
| `GGML_VK_DISABLE_MMVQ`, `GGML_VK_FORCE_MMVQ` | Force the quantised mat-vec path off or on. Useful for bisecting a wrong-output device. |
| `GGML_VK_DISABLE_FUSION`, `GGML_VK_DISABLE_MULTI_ADD`, `GGML_VK_DISABLE_GRAPH_OPTIMIZE` | Turn off graph-level fusion and reordering. |
| `GGML_VK_ALLOW_FUG`, `GGML_VK_ALLOW_FMOE` | Re-enable the fused up/gate and fused-MoE graphs when offloading to Vulkan. |
| `GGML_OP_OFFLOAD_MIN_BATCH` | Batch threshold above which the scheduler may move a CPU-resident op to the GPU (default 32). A very large value pins the graph to the CPU. |
| `LLAMA_NO_HOST_OVERRIDES` | Allocate `-ot` / `--cpu-moe` / `-ncmoe` tensors from plain CPU buffers instead of the backend host buffer type. Under Vulkan that host buffer is device-visible host memory, capped by the GTT aperture, and a large expert set exhausts it long before system RAM runs out. |
| `GGML_VK_PERF_LOGGER`, `GGML_VK_MEMORY_LOGGER`, `GGML_VK_PIPELINE_STATS` | Per-op timings, allocation tracing, pipeline statistics. |
