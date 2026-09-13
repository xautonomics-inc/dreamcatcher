# Dreamcatcher BDD feature specifications

The `.feature` files in this directory describe acceptance behavior for
Dreamcatcher's user-facing model-library, stage-ring, remote-expert, global-slot,
and release-verification interfaces. They use Gherkin syntax so a later adapter
can execute them with Behave, Cucumber, or another compatible runner.

This pass intentionally contains no step definitions. Until an adapter is added,
the feature files are reviewable specifications rather than an automated release
gate.

## Feature map

| File | Interface | Primary reference |
| --- | --- | --- |
| [`serve-library.feature`](serve-library.feature) | `llama-server --model-dir` | [`docs/BRING-UP.md`](../../../docs/BRING-UP.md) |
| [`stage-ring.feature`](stage-ring.feature) | `llama-stage-runner` ring roles | [`docs/BRING-UP.md`](../../../docs/BRING-UP.md) |
| [`layer-library.feature`](layer-library.feature) | Slicer, manifest, and planner | [`LAYER_SLICES.md`](../../../examples/stage-runner/LAYER_SLICES.md) |
| [`expert-server.feature`](expert-server.feature) | `llama-expert-server` and `LLAMA_EXPERTS_REMOTE` | [`docs/EXPERT-SERVER-PORT.md`](../../../docs/EXPERT-SERVER-PORT.md) |
| [`gslot-runtime.feature`](gslot-runtime.feature) | `STAGE_GSLOT_*` gate and arbiter | [`docs/GSLOT-RUNTIME-PORT.md`](../../../docs/GSLOT-RUNTIME-PORT.md) |
| [`release-checks.feature`](release-checks.feature) | Serving, checksum, and manifest checks | [`docs/RELEASE-INTEGRITY.md`](../../../docs/RELEASE-INTEGRITY.md) |

## Binding placeholders

Values in angle brackets are environment-specific inputs. They are literal text
in the feature files; the future step implementation must resolve each one from a
fixture, command-line parameter, or environment variable before launching a
process. An unbound placeholder should stop the scenario before it touches a
model or service.

| Placeholder | Binding |
| --- | --- |
| `<lib_dir>` | A `gguf-layer-library/v1` directory with `manifest.json` |
| `<monolith_path>` | The monolithic GGUF used as the reference for the library |
| `<model_path>` | A GGUF suitable for the scenario's executable |
| `<host>` / `<port>` | Host and port of an owned HTTP endpoint |
| `<head_host>` / `<head_port>` | Head-stage ring address |
| `<relay_host>` / `<relay_port>` | Relay-stage ring address |
| `<tail_host>` / `<tail_port>` | Tail-stage ring address |
| `<return_port>` | Token return-edge port |
| `<expert_host>` / `<expert_port>` | Remote expert-server address |
| `<arbiter_socket>` | Filesystem path of an owned `gslot` Unix socket |
| `<coherent_text_file>` | Saved completion expected to pass the coherence screen |
| `<degenerate_text_file>` | Saved completion expected to fail the coherence screen |
| `<fixture_checkout>` | Disposable checkout used for checksum fixtures |
| `<build_dir>` | Build directory containing the requested test executable |

Do not bind scenarios to a shared production endpoint. Model paths, addresses,
and generated evidence belong in the runner's private configuration or artifacts,
not in these feature files.

## Deterministic comparisons

Parity scenarios require the same model bytes, prompt, tokenizer and chat
template, context settings, sampling settings, backend placement, and relevant
fusion flags on both sides. A scenario that says *byte-identical* compares raw
logits or files byte for byte. A scenario that says *token-identical* compares the
ordered token IDs. Text alone is insufficient for either claim.

For remote experts, client and server execution modes are a pair:

| Client | Expert server |
| --- | --- |
| defaults | `--fmoe 1 --mmad 1` (defaults) |
| `-no-fmoe -no-mmad` | `--fmoe 0 --mmad 0` |

Mixing these rows is expected to change logits and must fail the parity gate.

## Tags

- `@smoke` covers a short process or endpoint check.
- `@parity` requires a deterministic reference comparison.
- `@error-handling` exercises a documented rejection path.
- `@fail-open` verifies that an unavailable optional scheduler does not halt the
  ring.
- `@known-issue @meta-NN` records an open item from
  [`docs/KNOWN-ISSUES.md`](../../../docs/KNOWN-ISSUES.md). Known-issue scenarios are
  evidence captures, not passing release requirements.

Once step definitions exist, a runner can select scenarios along these lines:

```bash
behave tests/bdd/features --tags=@smoke
behave tests/bdd/features/expert-server.feature
behave tests/bdd/features --tags='not @known-issue'
```

The adapter should preserve command output, exit status, model identity, source
revision, backend and device selection, and comparison digests as artifacts. A
successful `/health` response does not replace a generated completion, and memory
estimates do not replace a survived prefill.
