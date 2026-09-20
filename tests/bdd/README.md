# Dreamcatcher BDD Acceptance Scenario Suite

This directory contains Behavior-Driven Development (BDD) acceptance scenarios, executable step definitions, and test fixtures for Dreamcatcher's user-facing interfaces, distribution engine, stage-runner pipeline rings, remote expert serving, compute arbitration, release checks, and built-in WebUI.

---

## 1. Core Rule: Real Binaries or Explicit Skip

> [!IMPORTANT]
> **Strict Fail-Closed Policy**:
> - Every test step must exercise a **real product binary, model library, or live API endpoint**. Silent mock fallbacks are strictly prohibited.
> - If an environment prerequisite (e.g. compiled `llama-server` binary, `llama-stage-runner`, model weights, or Playwright browser dependencies) is not present on the test host, the scenario must fail closed or explicitly **SKIP with a descriptive reason** (e.g., `pytest.skip("Prerequisite unmet: llama-server binary not found")`).
> - Mock fixtures are permitted only behind an explicit, off-by-default `@contract-mock` marker and are reported in their own category, never as passing product runs.
> - `tests/bdd/RESULTS.md` is generated exclusively from actual JUnit XML run records, with zero fabricated or hardcoded counts.

---

## 2. Directory Layout

```
tests/bdd/
├── features/               # Executable Gherkin feature files
│   ├── layer-library.feature  # Slicing, Blake2b-128 hashing, distribution planner
│   ├── expert-server.feature  # Remote MoE handshake, parity, routing, failures
│   ├── gslot-runtime.feature  # Compute leases, handoff, contention, fail-open
│   ├── inkling.feature        # Inkling arch lanes D1-D6; D6 = CPU synthetic-fixture smoke (CI)
│   ├── release-checks.feature # Serve smoke, coherence, checksums, manifest test
│   ├── serve-library.feature  # llama-server --model-dir serving and windowing
│   ├── stage-ring.feature     # Multi-host pipeline rings and handoffs
│   └── web-ui.feature         # llama-server built-in WebUI via headless Playwright
├── steps/                  # Python step definitions (pytest-bdd)
│   ├── common_steps.py        # Shared library and HTTP steps
│   ├── expert_server_steps.py # Real expert-server and expert-check processes
│   ├── gslot_runtime_steps.py # Real gslot daemon and C++ client-header probe
│   ├── inkling_steps.py       # D2 envelope gate; D6 in-step synthetic GGUF + llama-perplexity/llama-cli smoke
│   ├── layer_library_steps.py # Slicer and distribution planner steps
│   ├── release_checks_steps.py # Real release scripts and manifest test binary
│   ├── serve_library_steps.py # Live llama-server process checks
│   ├── stage_ring_steps.py    # Live stage-runner ring steps
│   └── web_ui_steps.py        # Playwright headless browser E2E steps
├── conftest.py             # Pytest fixtures, placeholder resolvers, xfail tags
├── generate_results_md.py  # Strict JUnit XML parser -> RESULTS.md generator
├── requirements.txt        # Pinned Python dependencies
└── run-local.sh            # Local & GitLab docker runner entrypoint
```

---

## 3. Running Scenarios

### Local / Docker Runner Execution
Execute all scenarios using the local runner script:
```bash
./tests/bdd/run-local.sh [path/to/env_file]
```

The script:
1. Sources any external environment bindings without leaking fleet paths into git.
2. Sets `PYTHONPATH` to include the repository root, `tools/layer-distribution`, and dependencies.
3. Executes `pytest` across all feature files, or the path selected by `BDD_TEST_TARGET`, capturing exit codes without masking.
4. Generates `tests/bdd/report.xml` and formats it into `tests/bdd/RESULTS.md`.
5. Exits with the true pytest exit code.

### Running with Pytest Directly
```bash
python3 -m pytest tests/bdd/ -v
```

To run a specific feature:
```bash
python3 -m pytest tests/bdd/test_layer_library.py -v
```

---

## 4. Environment Variables & Placeholders

Scenario steps support the following environment overrides:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LLAMA_SERVER_BIN` | Absolute path to built `llama-server` | `build/bin/llama-server` |
| `LLAMA_STAGE_RUNNER_BIN` | Absolute path to built `llama-stage-runner` | `build/bin/llama-stage-runner` |
| `LLAMA_PERPLEXITY_BIN` | Absolute path to built `llama-perplexity` (Inkling D6 fixture smoke) | `build/bin/llama-perplexity`, then `PATH` |
| `LLAMA_CLI_BIN` | Absolute path to built `llama-cli` (Inkling D6 fixture smoke) | `build/bin/llama-cli`, then `PATH` |
| `BDD_INKLING_D6_LOG_DIR` | Directory where the D6 smoke persists the loader/graph logs of each binary run | Unset; logs stay in the pytest tmp dir |
| `LLAMA_EXPERT_SERVER_BIN` | Absolute path to built `llama-expert-server` | `build/bin/llama-expert-server`, then `PATH` |
| `LLAMA_EXPERT_CHECK_BIN` | Absolute path to built `llama-expert-check` | `build/bin/llama-expert-check`, then `PATH` |
| `BDD_MOE_MODEL` | Path to a real MoE GGUF used by expert-server scenarios | Unbound; scenarios skip |
| `BDD_MOE_LAYERS` | Number of model layers; current scenarios require at least 32 | Unbound; scenarios skip |
| `BDD_EXPERT_PROMPT` | Deterministic prompt used for local/remote parity | `The capital of France is` |
| `BDD_EXPERT_TOKENS` | Number of generated tokens compared for parity | `12` |
| `BDD_EXPERT_TIMEOUT` | Timeout in seconds for each expert-check process | `900` |
| `INKLING_REMOTE_EXPERTS_MODEL` | Inkling GGUF for the `@d4c` remote-experts scenario (e.g. Inkling-Small in a booked window) | Unbound; the tiny synthetic fixture is generated in-step by `tools/make-inkling-test-gguf.py` (needs numpy, pyyaml) |
| `INKLING_REMOTE_EXPERTS_CLIENT_ARGS` | Extra `llama-expert-check` arguments for BOTH the in-process and the remote `@d4c` run (e.g. `-fa off`) | Empty |
| `INKLING_REMOTE_EXPERTS_SERVER_ARGS` | Extra `llama-expert-server` arguments for the `@d4c` run; negative control only (`--moe-form moe_ffn` must fail the byte gate) | Empty; `--moe-form auto` |
| `INKLING_MODEL_PATH` | Inkling-Small GGUF for the model-heavy lanes (`@d2`, `@d3`); also overrides the in-step fixture for `@d1` and is the monolith of a bound `INKLING_LIB_DIR` | Unbound; runner Whens skip |
| `INKLING_ORACLE_DIR` | D0 oracle artifact directory (sha256-pinned `greedy-64x8.json`, `kld-base-4x2048.bin`, `ppl.log`) | Unbound; `@d2`/`@d3` skip, `@fail-closed` runs |
| `INKLING_ENVELOPE_JSON` | Pre-registered cross-lineage envelope (`@d2`, `@d3` gate) | `tests/bdd/fixtures/inkling-envelope.json` |
| `INKLING_GREEDY_CMD` | `@d2` greedy runner template, placeholders `{model}` `{oracle_dir}`; stdout = the D0-schema dump. Wired: `python3 tools/inkling-greedy-dump.py --server BIN --model {model} --prompts {oracle_dir}/greedy-64x8.json --server-args "-ngl 0 -t 20 -c 4096 -np 1 -fa off"` | Unbound; skips |
| `INKLING_PPL_CMD` | `@d2` KLD runner template, placeholders `{model}` `{text}` `{base}` (a `llama-perplexity -fa off --kl-divergence --kl-divergence-base {base}` run over the 4x2048 split; `{base}` must be a working COPY of the oracle base, see `tests/bdd/fixtures/inkling-window.env.example`) | Unbound; skips |
| `INKLING_D3_PPL_CMD` | `@d3` cache-lane runner template, same placeholders as `INKLING_PPL_CMD` but running the banded cache path (`-fa on`) | Unbound; skips |
| `INKLING_PPL_TEXT` | Perplexity text file for the KLD runs (the D0 4-chunk split source) | Unbound; skips |
| `INKLING_ORACLE_FEATURES` / `INKLING_BUILD_FEATURES` | Comma-separated CPU feature sets of the oracle and the runner build; token-for-token equality gates only when both are declared and equal | Unbound; diagnostic |
| `INKLING_LIB_DIR` | `@d4a`: a layer library sliced with `tools/layer-distribution` (parts carry `inkling.dense_block_count`); needs `INKLING_MODEL_PATH` as its monolith | Unbound; the fixture is generated and sliced in-step |
| `INKLING_LIBRARY_SERVER_ARGS` | `@d4a`: extra `llama-server` arguments for BOTH the `--model-dir` and the `-m` server (e.g. `-fa off`) | Empty |
| `INKLING_STAGE_RING_MODEL` | `@d4b`: Inkling GGUF for the head+tail ring (split at its `dense_block_count`) | Unbound; the fixture is generated and sliced in-step |
| `INKLING_STAGE_RING_CLIENT_ARGS` | `@d4b`: extra `llama-expert-check` arguments for the monolith greedy reference | Empty |
| `INKLING_BDD_THREADS` | CPU threads for the servers / stages the Inkling fixture lanes start (`-t`, `STAGE_THREADS`) | `4` |
| `INKLING_SERVER_STARTUP_TIMEOUT` / `INKLING_RUNNER_TIMEOUT` | Seconds to wait for a server `/health` / for one runner command (a 152 GB load and a 4-chunk KLD run take minutes) | `1800` / `7200` |
| `INKLING_D1_SERVER_ARGS` | `@d1`: extra `llama-server` arguments for the load-only run | Empty |
| `BDD_SERVER_URL` | Owned live `llama-server` used by serve smoke checks | Unbound; scenarios skip |
| `BDD_REASONING_ONLY_URL` | Owned fixture endpoint for reasoning-only smoke behavior | Unbound; scenario skips |
| `BDD_LIB_DIR` | Directory of pre-sliced layer library | Auto-generated synthetic library |
| `BDD_MONOLITH_PATH` | Path to source monolithic GGUF model | Auto-generated synthetic model |
| `BDD_HOST` | Host address for server binding | `127.0.0.1` |
| `BDD_PORT` | Port for server binding | `8080` |
| `BDD_MODEL` | Human-readable model identifier | `unbound` |
| `BDD_TEST_TARGET` | Pytest file or directory to execute | Entire `tests/bdd` suite |
| `BDD_BINARY_COMMIT` | Exact source commit used to build the tested binaries | Current checkout |
| `BDD_GITHUB_COMMIT` | Equivalent Dreamcatcher commit for the public report link | Binary commit |

---

## 5. Known Issues Tracking

Scenarios representing tracked issues are tagged with `@known-issue` and their meta issue identifier. The test runner automatically applies `pytest.mark.xfail` so that expected defects are reported truthfully without breaking CI:
- **`@known-issue @meta-80`**: Gemma-4 Q6_K `token_embd.weight` degenerate token repetition.
- **`@known-issue @meta-97`**: Multi-stage pipeline rings unsupported with per-layer input embeddings in non-zero stage windows.
- **`@known-issue @meta-100`**: A server loaded with `--model-dir` reports an empty `model_name`, so the conversation header and assistant badge show no model name.
