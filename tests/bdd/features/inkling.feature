@inkling @pending
Feature: Serve the Inkling (TML) architecture end to end
  As an operator running Inkling-Small on dreamcatcher
  I want the architecture supported across the runtime, cache, layer library,
  stage rings, and remote experts
  So that every execution mode produces results identical to the reference port

  Expected values bind to the D0 oracle artifacts (the D0 oracle lane): greedy
  token IDs, FNV/raw logits dumps, and 4-chunk perplexity recorded from the
  internal llama.cpp-lineage port on Inkling-Small, checked in with the exact
  commands and hashes. Until those artifacts land, every scenario below is
  pending and MUST fail closed: an unbound `<oracle_dir>` stops the scenario
  before it touches a model or service (the same rule as the other feature
  files in this directory).

  Lane gates (from the 2026-09-15 Inkling plan):
  D1 arch plumbing, D2 `build_inkling.cpp` CPU parity, D3 cache + kernels,
  D4a layer library, D4b stage rings, D4c remote experts, D6 CI fixture.
  Public feature request: https://github.com/xautonomics-inc/dreamcatcher/issues/43

  Background:
    Given Inkling-Small GGUF metadata for "<model_path>"
    And a D0 oracle artifact directory "<oracle_dir>" with recorded hashes

  @d1 @arch @pending
  Scenario: The architecture loads metadata and creates every tensor, no compute
    Given a dreamcatcher build with LLM_ARCH_INKLING registered
    When the model is loaded on CPU with compute disabled
    Then every tensor is created with the Inkling names and shapes
    And the KV keys include dense_block_count, d_rel, rel_extent, rel_extent_swa,
      shortconv_kernel, logit_scale_denom, log_scaling_n_floor, log_scaling_alpha,
      unpadded_vocab_size, the per-layer SWA pattern, and the expert FFN length
    And no forward pass has run

  @d2 @parity @pending
  Scenario: CPU greedy generation matches the D0 oracle token-for-token
    When the model runs greedy generation on CPU for the D0 fixed prompt set
    Then every generated token ID equals the oracle token IDs
    And the raw logits are within tolerance of the oracle logits dumps

  @d2 @parity @pending
  Scenario: 4-chunk perplexity equals the oracle within error
    When the model computes perplexity on CPU over the D0 4-chunk split
    Then the perplexity equals the oracle value within the recorded error

  @d3 @cache @pending
  Scenario: The hybrid attention+recurrent cache serves banded SWA windows
    Given a hybrid Inkling cache with iswa windows over a 5:1 pattern
    When generation runs past several window boundaries with int64 expert indexing
    Then attention state, recurrent short-conv state, and expert indices stay consistent
    And token IDs continue to equal the oracle

  @d4a @layer-library @pending
  Scenario: A sliced layer library reproduces the monolith exactly
    Given an Inkling layer library sliced with arch-aware dense_block_count
    When llama-server runs with "--model-dir <lib_dir>"
    Then the served monolith equals the GGUF monolith
    And token IDs continue to equal the oracle

  @d4b @stage-ring @pending
  Scenario: Head and tail ring stages reproduce the monolith
    Given a stage ring with head and tail roles hosting Inkling stages
    When generation runs for at least 64 tokens through the ring
    Then head and tail token IDs equal the monolith run
    And the boundary hidden state is byte-equal to the monolith run

  @d4c @remote-experts @pending
  Scenario: Remote experts are byte-exact against in-process experts
    Given the expert server hosting the routed expert bank
    And the shared expert bank stays local
    When generation runs through the remote-expert path on CPU
    Then token IDs and logits are byte-exact against the in-process run

  @d6 @ci @pending
  Scenario: The synthetic-GGUF CI fixture smoke-tests arch load and graph
    Given a tiny synthetic Inkling GGUF fixture "<fixture_gguf>"
    When the CI job loads the architecture and runs a graph smoke test on CPU
    Then the load succeeds and generation completes without the full model
    And no scenario in this file requires the 160 GiB model outside CI-booked windows

  @fail-closed @pending
  Scenario: Missing D0 artifacts stop the scenario before any model is touched
    Given an "<oracle_dir>" that does not exist or fails hash verification
    When the scenario attempts to bind expected values
    Then the scenario aborts before loading any model or starting any service
