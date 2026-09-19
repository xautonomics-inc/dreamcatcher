"""Bind the implemented Inkling scenarios.

The two @d2 parity scenarios, the @d6 @ci synthetic-fixture smoke, and
the @fail-closed scenario have step definitions in steps/inkling_steps.py;
the @d4c remote-experts scenario in steps/inkling_remote_experts_steps.py.
The D1/D3/D4a/D4b lanes stay unbound until their steps exist — a
whole-file scenarios() bind here would fail collection with
StepDefinitionNotFoundError for the whole feature, so bind by scenario
title. A later whole-file scenarios() call skips these as already bound
(pytest-bdd dedupes on (feature filename, scenario name)).
"""

from pytest_bdd import scenario


# D2 amendment (2026-09-19): the equality gate became a pre-registered
# cross-lineage envelope, so the scenario titles below track the feature
# file — a stale title here fails collection with ScenarioNotFound.
@scenario(
    "features/inkling.feature",
    "CPU greedy generation stays within the cross-lineage envelope of the D0 oracle",
)
def test_inkling_d2_greedy_parity():
    """D2: greedy metrics inside the pre-registered cross-lineage envelope."""


@scenario(
    "features/inkling.feature",
    "4-chunk perplexity falls within the envelope around the oracle per-chunk values",
)
def test_inkling_d2_ppl_parity():
    """D2: per-chunk perplexity inside the envelope band around the oracle log."""


@scenario(
    "features/inkling.feature",
    "Missing D0 artifacts stop the scenario before any model is touched",
)
def test_inkling_fail_closed():
    """Fail closed: invalid INKLING_ORACLE_DIR aborts before any launch."""


# D6 (@d6 @ci): the CI job bdd-inkling-fixture in .gitlab-ci.yml targets this
# node id alone. It needs no oracle and no full model — only the in-repo
# generator plus CPU-built llama-perplexity / llama-cli (LLAMA_PERPLEXITY_BIN
# / LLAMA_CLI_BIN); without them it skips, and the CI job refuses a skip.
@scenario(
    "features/inkling.feature",
    "The synthetic-GGUF CI fixture smoke-tests arch load and graph",
)
def test_inkling_d6_ci_fixture():
    """D6: the synthetic Inkling fixture loads and its graph runs on CPU."""


@scenario(
    "features/inkling.feature",
    "Remote experts are byte-exact against in-process experts",
)
def test_inkling_d4c_remote_experts():
    """D4c: expert-server --moe-form inkling vs in-process, byte-identical
    tokens, per-step logits hashes and raw logits (steps/inkling_remote_experts_steps.py)."""
