@inkling @pending
Feature: Serve the Inkling (TML) architecture end to end
  As an operator running Inkling-Small on dreamcatcher
  I want the architecture supported across the runtime, cache, layer library,
  stage rings, and remote experts
  So that every execution mode produces results identical to the reference port

  Expected values bind to the D0 oracle artifacts (the D0 oracle lane), recorded
  from the internal llama.cpp-lineage port on Inkling-Small. Steps resolve the
  INKLING_ORACLE_DIR, INKLING_MODEL_PATH and INKLING_LIB_DIR paths from the
  environment and fail closed when one is unset, missing, or fails hash
  verification (the same rule as BDD_SERVER_URL in release-checks.feature).
  Recorded oracle run (do not re-run it):
    oracle dir:  /fast/build/agents/noah/p1/d0-oracle-20260915T2024/
    greedy-64x8.json     — greedy token IDs plus per-position log-probabilities
                           (fields logprob and top_logprobs, n_probs: 10; the
                           dump holds no full logit vectors)
    kld-base-4x2048.bin  — base logits dump over the D0 4-chunk perplexity split
    ppl.log              — per-chunk and final perplexity log
    ppl 72.6183 ± 5.49154 — the ± value is the oracle's own error bar, not the
                           parity tolerance; parity asserts the four per-chunk
                           values below
    oracle host/build: nvidia, build 0db1d97c3ac1f9882b7b8b1719373bb76a52d353,
    CUDA-built but run CPU-only (CUDA_VISIBLE_DEVICES=""); a D2 parity run must
    reproduce this build and its CPU feature set (AVX2, FMA, LLAMAFILE, REPACK),
    not just the command line, -t 20, Inkling-Small UD-Q4_K_M
  sha256 greedy-64x8.json     1b5e5c4bff5b98cbe91345e5df290a804667e25ff728fbbba78b3d08cef923f0
  sha256 kld-base-4x2048.bin  aeab11429a2567db2b36f210754a444ce97f6a9fa851bf0389b236129838b505
  sha256 ppl.log              98f7349cb50c92500b5cd79171e0fcf34453a79abec7c7fccfd881f1f4ab4edc
  Reference behaviour: oracle prompt 4 degenerates into repeated "changed"
  tokens. That is the reference behaviour; parity means reproducing it, not
  fixing it.

  Until the artifact directory exists on the host under test, every scenario
  below is pending and MUST fail closed: resolution of INKLING_ORACLE_DIR stops
  the scenario before it touches a model or service (the same rule as the
  other feature files in this directory).

  Lane gates (from the 2026-09-15 Inkling plan):
  D1 arch plumbing, D2 `build_inkling.cpp` CPU parity, D3 cache + kernels,
  D4a layer library, D4b stage rings, D4c remote experts, D6 CI fixture.
  Public feature request: https://github.com/xautonomics-inc/dreamcatcher/issues/43

  CI policy (D6): no scenario in this file requires the 160 GiB full model
  outside CI-booked windows; CI uses only the tiny synthetic Inkling GGUF
  fixture generated in-step. Full-model runs need a booked window.

  Background:
    Given Inkling-Small GGUF metadata for the model named by INKLING_MODEL_PATH
    And a D0 oracle artifact directory named by INKLING_ORACLE_DIR with recorded hashes

  @d1 @arch @pending
  Scenario: The architecture loads metadata and creates every tensor, no compute
    Given a dreamcatcher build with LLM_ARCH_INKLING registered
    When the model is loaded on CPU with compute disabled
    Then every tensor is created with the Inkling names and shapes
    And the KV keys include dense_block_count, d_rel, rel_extent, rel_extent_swa,
      shortconv_kernel, logit_scale_denom, log_scaling_n_floor, log_scaling_alpha,
      vocab_size, unpadded_vocab_size, block_count, attention.sliding_window,
      attention.sliding_window_pattern, attention.head_count_kv,
      feed_forward_length, expert_feed_forward_length, expert_count,
      expert_used_count, and expert_shared_count
    And no forward pass has run

  @d2 @parity @pending
  Scenario: CPU greedy generation matches the D0 oracle token-for-token
    Given the D0 oracle dump "greedy-64x8.json" exists in the INKLING_ORACLE_DIR directory
    When the model runs greedy generation on CPU for the D0 fixed prompt set
    Then every generated token ID equals the oracle token IDs
    And the per-position log-probabilities (logprob, top_logprobs) match the oracle within tolerance

  @d2 @parity @pending
  Scenario: 4-chunk perplexity reproduces the oracle per-chunk values
    Given the D0 oracle dump "kld-base-4x2048.bin" exists in the INKLING_ORACLE_DIR directory
    When the model computes perplexity on CPU over the D0 4-chunk split
    Then the four per-chunk perplexity values equal the oracle 94.3665, 83.6386, 78.6291, 72.6183
    And the final chunk equals the recorded overall perplexity 72.6183

  @d3 @cache @pending
  Scenario: The hybrid attention+recurrent cache serves banded SWA windows
    Given a hybrid Inkling cache with iswa windows over a 5:1 pattern
    When generation runs past several window boundaries with int64 expert indexing
    Then attention state, recurrent short-conv state, and expert indices stay consistent
    And token IDs continue to equal the oracle

  @d4a @layer-library @pending
  Scenario: A sliced layer library reproduces the monolith exactly
    Given an Inkling layer library sliced with arch-aware dense_block_count
    When llama-server runs with "--model-dir" pointing at the sliced INKLING_LIB_DIR directory
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
    Given a tiny synthetic Inkling GGUF fixture generated by the D6 generator
    When the CI job loads the architecture and runs a graph smoke test on CPU
    Then the load succeeds and generation completes without the full model

  @fail-closed @pending
  Scenario: Missing D0 artifacts stop the scenario before any model is touched
    Given a D0 oracle artifact directory named by INKLING_ORACLE_DIR with recorded hashes
    And that directory does not exist or fails hash verification
    When the scenario attempts to bind expected values
    Then the scenario aborts before loading any model or starting any service
