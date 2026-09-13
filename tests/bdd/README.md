# Dreamcatcher BDD Acceptance Scenarios

This directory contains executable Behavior-Driven Development (BDD) scenarios written in standard [Gherkin](https://cucumber.io/docs/gherkin/) syntax (`.feature` files), specifying user-facing interfaces, operational behaviors, error-handling contracts, and release gates for the **Dreamcatcher** distributed inference engine.

---

## 1. Feature Specifications

| Feature File | Target Interface & Subsystem | Authoritative Documentation |
|:---|:---|:---|
| [`features/serve-library.feature`](features/serve-library.feature) | Single-process per-layer model library serving (`llama-server --model-dir`) | [`docs/BRING-UP.md`](../../docs/BRING-UP.md) §0, [`LAYER_SLICES.md`](../../examples/stage-runner/LAYER_SLICES.md) §3 |
| [`features/stage-ring.feature`](features/stage-ring.feature) | Multi-host stage runner pipeline rings (`llama-stage-runner --role head\|relay\|tail`) | [`docs/BRING-UP.md`](../../docs/BRING-UP.md) §1–§4 |
| [`features/layer-library.feature`](features/layer-library.feature) | Per-layer slicer (`slice_gguf_layers.py`), manifest schema, and distribution planner | [`LAYER_SLICES.md`](../../examples/stage-runner/LAYER_SLICES.md) §1, [`tools/layer-distribution`](../../tools/layer-distribution/README.md) |
| [`features/expert-server.feature`](features/expert-server.feature) | Disaggregated MoE expert server (`llama-expert-server`) and remote client hook | [`docs/EXPERT-SERVER-PORT.md`](../../docs/EXPERT-SERVER-PORT.md) |
| [`features/gslot-runtime.feature`](features/gslot-runtime.feature) | Global Slot Router client gate (`STAGE_GSLOT_*`), fail-open resilience, and lease arbiter | [`docs/GSLOT-RUNTIME-PORT.md`](../../docs/GSLOT-RUNTIME-PORT.md), [`tools/gslot`](../../tools/gslot/README.md) |
| [`features/release-checks.feature`](features/release-checks.feature) | Serving smoke verification (`ci/smoke-serve.sh`), checksum verification, and manifest parser | [`docs/RELEASE-INTEGRITY.md`](../../docs/RELEASE-INTEGRITY.md), [`docs/KNOWN-ISSUES.md`](../../docs/KNOWN-ISSUES.md) |

---

## 2. Placeholder Binding & Environment Configuration

In strict compliance with the **Evidence Rule**, scenarios contain zero hardcoded machine hostnames, internal IP addresses, or private filesystem paths. All environment-specific parameters are expressed as standardized bracketed placeholders (`<placeholder>`).

When binding these scenarios to a live cluster, test harness, or CI runner, map the placeholders to target environment variables or runner fixtures as follows:

| Placeholder | Meaning & Description | Example Value / Binding |
|:---|:---|:---|
| `<lib_dir>` | Directory containing partitioned `gguf-layer-library/v1` files and `manifest.json` | `/models/gemma-4-12b-layers` |
| `<monolith_path>` | Path to monolithic reference GGUF model file | `/models/gemma-4-12b.gguf` |
| `<output_dir>` | Destination directory for slicer output | `/tmp/sliced-library` |
| `<host>` | Hostname or IP address for HTTP server binding | `127.0.0.1` |
| `<port>` | TCP port for `llama-server` HTTP API | `8080` |
| `<head_port>` | TCP listening port for stage runner Head stage | `9001` |
| `<relay_port>` | TCP listening port for stage runner Relay stage | `9002` |
| `<tail_port>` | TCP listening port for stage runner Tail stage | `9003` |
| `<http_port>` | External HTTP completion endpoint exposed by Head stage | `8080` |
| `<host_head>` | Network hostname or IP of Head compute node | `stage-head.local` |
| `<host_relay>` | Network hostname or IP of Relay compute node | `stage-relay.local` |
| `<host_tail>` | Network hostname or IP of Tail compute node | `stage-tail.local` |
| `<expert_host>` | Hostname or IP of running `llama-expert-server` | `127.0.0.1` |
| `<expert_port>` | TCP port of running `llama-expert-server` | `9090` |
| `<arbiter_socket>` | Path to UNIX domain socket or TCP address of `gslot` arbiter daemon | `/tmp/gslot.sock` |
| `<total_layers>` | Total number of transformer blocks in the target model | `48` |
| `<total_blocks>` | Total number of blocks declared in model manifest | `48` |

---

## 3. Scenario Tags & Filtering

Scenarios are categorized using standard Gherkin tags to enable targeted execution across different testing phases:

- **`@smoke`**: Fast end-to-end sanity tests verifying process startup, HTTP availability, and basic completion.
- **`@parity`**: Mathematical equivalence gates verifying bit-exact or token-for-token parity against monolithic baselines.
- **`@windowing`**: Layer slice boundary validation (`--layers A,B`).
- **`@error-handling`**: Validation that malformed arguments, missing files, or out-of-bounds inputs produce explicit, non-zero exits with descriptive errors.
- **`@fail-open`**: Resilience checks validating that scheduler, arbiter, or socket interruptions fail open without stalling the pipeline.
- **`@known-issue`**: Marks scenarios that document active upstream or backend investigations tracked in [`docs/KNOWN-ISSUES.md`](../../docs/KNOWN-ISSUES.md). These scenarios are tagged with their tracking issue handle:
  - `@meta-80`: Gemma-4 `Q6_K` `token_embd` dequantization defect.
  - `@meta-81`: Gemma-4 on CUDA batched prefill token mismatch.
  - `@meta-84`: NVIDIA Vulkan ICD kernel drift.
  - `@meta-88`: Vulkan hybrid MoE hyper-connection mixer.
  - `@meta-96`: DeepSeek-V4 Vulkan backend kernel planning.

---

## 4. Execution Guidelines

These feature files are designed to be executed with standard BDD test runners such as [behave](https://behave.readthedocs.io/):

```bash
# Run all smoke tests against a local test deployment
behave tests/bdd/features/ --tags=smoke

# Run per-layer library serving scenarios
behave tests/bdd/features/serve-library.feature

# Exclude open known issues during release promotion gates
behave tests/bdd/features/ --tags="~known-issue"
```

Step definitions (`tests/bdd/steps/`) will be mapped in a subsequent implementation pass to bind these scenarios directly to test runner instances and the cluster automation harness.
