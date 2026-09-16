"""Bind the implemented Inkling scenarios.

Only the two @d2 parity scenarios and the @fail-closed scenario have step
definitions (steps/inkling_steps.py). The D1/D3/D4a/D4b/D4c/D6 lanes stay
unbound until their steps exist — a whole-file scenarios() bind here would
fail collection with StepDefinitionNotFoundError for the whole feature, so
bind by scenario title. A later whole-file scenarios() call skips these as
already bound (pytest-bdd dedupes on (feature filename, scenario name)).
"""

from pytest_bdd import scenario


@scenario("features/inkling.feature", "CPU greedy generation matches the D0 oracle token-for-token")
def test_inkling_d2_greedy_parity():
    """D2: greedy token IDs and per-position log-probabilities vs the oracle."""


@scenario("features/inkling.feature", "4-chunk perplexity reproduces the oracle per-chunk values")
def test_inkling_d2_ppl_parity():
    """D2: the four per-chunk perplexity values vs the oracle log."""


@scenario(
    "features/inkling.feature",
    "Missing D0 artifacts stop the scenario before any model is touched",
)
def test_inkling_fail_closed():
    """Fail closed: invalid INKLING_ORACLE_DIR aborts before any launch."""
