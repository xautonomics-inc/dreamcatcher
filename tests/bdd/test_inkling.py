"""Bind the implemented Inkling scenarios.

Every scenario of features/inkling.feature has a step implementation:
@d1/@d2/@d3/@d6/@fail-closed in steps/inkling_steps.py, @d4c in
steps/inkling_remote_experts_steps.py, @d4a/@d4b in
steps/inkling_library_steps.py. Scenarios are bound by title (a stale title
here fails collection with ScenarioNotFound; a whole-file scenarios() bind
would hide which lane lost its steps behind a single
StepDefinitionNotFoundError). The lanes and what they need:

  @d1  arch load, no compute      fixture in-step (INKLING_MODEL_PATH overrides)
  @d2  greedy / KLD envelope      Inkling-Small + D0 oracle (booked window)
  @d3  hybrid cache envelope      Inkling-Small + D0 oracle (booked window)
  @d4a library == monolith        fixture in-step (INKLING_LIB_DIR + model override)
  @d4b head+tail ring == monolith fixture in-step (INKLING_STAGE_RING_MODEL override)
  @d4c remote experts byte-exact  fixture in-step (INKLING_REMOTE_EXPERTS_MODEL override)
  @d6  CI fixture smoke           fixture in-step
"""

from pytest_bdd import scenario


# D2 amendment (2026-09-19): the equality gate became the pre-registered
# cross-lineage envelope (gate rule in the feature header), so the scenario
# titles below track the feature file — a stale title here fails collection
# with ScenarioNotFound.
@scenario(
    "features/inkling.feature",
    "CPU greedy generation runs coherently on the D0 fixed prompt set",
)
def test_inkling_d2_greedy_coherent():
    """D2: greedy generation runs coherently (finite, well-formed); divergence is diagnostic."""


@scenario(
    "features/inkling.feature",
    "4-chunk KLD / top-1 agreement stays within the cross-lineage envelope of the banded D0 oracle",
)
def test_inkling_d2_kld_parity():
    """D2: KLD / top-1 agreement vs the banded base inside the working band."""


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


@scenario(
    "features/inkling.feature",
    "The architecture loads metadata and creates every tensor, no compute",
)
def test_inkling_d1_arch_load():
    """D1: llama-server --no-warmup on CPU; header KVs, every tensor, zero forward passes."""


@scenario(
    "features/inkling.feature",
    "The hybrid attention+recurrent cache serves banded SWA windows",
)
def test_inkling_d3_cache():
    """D3: banded cache path over the D0 4-chunk split; finite state across windows, envelope gate."""


@scenario(
    "features/inkling.feature",
    "A sliced layer library reproduces the monolith exactly",
)
def test_inkling_d4a_layer_library():
    """D4a: llama-server --model-dir on the arch-aware library == -m monolith, greedy dump for dump."""


@scenario(
    "features/inkling.feature",
    "Head and tail ring stages reproduce the monolith",
)
def test_inkling_d4b_stage_ring():
    """D4b: head [0,dense) + tail [dense,n) stage-runner ring vs the monolith: ids + boundary state."""
