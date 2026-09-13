# Dreamcatcher BDD Acceptance Scenario Suite

This directory contains Behavior-Driven Development (BDD) acceptance scenarios, executable step definitions, and test fixtures for Dreamcatcher's user-facing interfaces, distribution engine, stage-runner pipeline rings, and built-in WebUI.

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
│   ├── serve-library.feature  # llama-server --model-dir serving and windowing
│   ├── stage-ring.feature     # Multi-host pipeline rings and handoffs
│   └── web-ui.feature         # llama-server built-in WebUI via headless Playwright
├── steps/                  # Python step definitions (pytest-bdd)
│   ├── common_steps.py        # Shared library and HTTP steps
│   ├── layer_library_steps.py # Slicer and distribution planner steps
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
3. Executes `pytest` across all feature files, capturing exit codes without masking.
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
| `BDD_LIB_DIR` | Directory of pre-sliced layer library | Auto-generated synthetic library |
| `BDD_MONOLITH_PATH` | Path to source monolithic GGUF model | Auto-generated synthetic model |
| `BDD_HOST` | Host address for server binding | `127.0.0.1` |
| `BDD_PORT` | Port for server binding | `8080` |
| `BDD_MODEL` | Human-readable model identifier | `synthetic-cpu-4b` |

---

## 5. Known Issues Tracking

Scenarios representing known upstream issues are tagged with `@known-issue` and their meta issue identifier. The test runner automatically applies `pytest.mark.xfail` so that expected defects are reported truthfully without breaking CI:
- **`@known-issue @meta-80`**: Gemma-4 Q6_K `token_embd.weight` degenerate token repetition.
- **`@known-issue @meta-97`**: Multi-stage pipeline rings unsupported with per-layer input embeddings in non-zero stage windows.
