@inkling
Feature: Serve the Inkling (TML) architecture end to end
  As an operator running Inkling-Small on dreamcatcher
  I want the architecture supported across the runtime, cache, layer library,
  stage rings, and remote experts
  So that every execution mode reproduces the reference port: identically
  within a lineage, and inside the cross-lineage gate rule below across lineages

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
    ppl.log              — per-chunk and final perplexity log: 94.3665,
                           83.6386, 78.6291, 72.6183 (the final estimate IS
                           chunk 4); recorded for reference, not gated
    ppl 72.6183 ± 5.49154 — the ± value is the oracle's own error bar, not a
                           parity tolerance; the D2 gate is the cross-lineage
                           gate rule below
    oracle host/build: nvidia, build 0db1d97c3ac1f9882b7b8b1719373bb76a52d353,
    CUDA-built but run CPU-only (CUDA_VISIBLE_DEVICES=""); a same-lineage D2
    parity run must reproduce this build and its CPU feature set (AVX2, FMA,
    LLAMAFILE, REPACK), not just the command line, -t 20, Inkling-Small
    UD-Q4_K_M; a cross-lineage run is gated by the rule below instead
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

  Cross-lineage gate rule (candreev-blessed, 2026-09-19; source: noah's D3
  envelope rule, built on toshi's D2 envelope start): the @d2 and @d3 lanes
  are cross-lineage ports of the reference's banded path. Token-exact and
  4-decimal-place perplexity equality against the banded D0 oracle is a
  same-kernel / same-lineage assumption a correct port cannot meet (the
  calibration showed the reference disagreeing with its own masked-vs-banded
  paths by ~17 % top-1), so those lanes gate on a bounded agreement envelope
  — KLD and top-1 agreement against the banded D0 oracle — under this
  pre-registered rule:
    1. Working band = measured banded-vs-banded drift × 2 (a pre-registered
       multiplier, fixed before the numbers land, not fit to them).
    2. The working band must sit under the masked-path ceiling (the agreement
       the reference's own masked path shows vs the banded oracle); if it
       doesn't, that's a finding.
    3. A working band wider than half the masked band is surfaced as a
       finding, not absorbed.
  Per metric, the measured drift, the masked-path ceiling, the band mode and
  the calibration provenance are registered in the INKLING_ENVELOPE_JSON file
  (env-bound, fail closed when unset); the ×2 multiplier is pinned in the
  step file, not in the envelope. The calibration, its finding and the
  practical gate that follows from it are in the BAND record below; the
  envelope registration must match that record.
  # BAND: measured 2026-09-19 (ben's calibration: llama-perplexity KL-divergence
  # run vs the banded D0 oracle, 4×2048 chunks; pending noah/emma's final encode)
  #   banded-vs-banded cross-kernel drift (the working-band basis):
  #     top-1 agreement 84.75 ± 0.56 %  (top-1 disagreement 15.25 %),
  #     RMS Δp 9.18 ± 0.48 %,  Mean Δp -0.28 ± 0.14 %
  #   masked-path ceiling (the reference's own masked path vs the banded oracle):
  #     top-1 agreement 83.50 ± 0.58 %  (top-1 disagreement 16.50 %),
  #     RMS Δp 10.01 ± 0.50 %
  #   cross-check (ik build, masked, -rtr vs its own masked): top-1 83.16 %,
  #     RMS Δp 9.68 % — same order, confirms it is kernel drift
  # FINDING (recorded verbatim, not absorbed): the banded-vs-banded drift is
  # nearly as large as the masked ceiling, so working band = drift × 2
  # (≈ RMS 18 % / top-1 disagreement 30 %) EXCEEDS the masked-path ceiling —
  # which fires noah's pre-registered rule 3 (band wider than half the masked
  # band → surfaced as a finding, not absorbed) and rule 2 (band must sit
  # under the ceiling).
  # PRACTICAL GATE (D2 and D3, pending noah/emma's final encode): the working
  # band is the masked-path ceiling itself — D3 must agree with the banded
  # oracle at least as well as the reference's masked path: top-1 ≥ ~83.5 %
  # within CI, RMS Δp ≤ ~10 % within CI. Encoded as band = ceiling + its ± as
  # printed: top-1 disagreement ≤ 16.50 + 0.58 = 17.08 %, RMS Δp ≤ 10.01 +
  # 0.50 = 10.51 %. drift × 2 stays recorded above as the rule-3 finding; the
  # envelope registers these entries with band "masked_ceiling" and that
  # finding, and the steps refuse a masked_ceiling entry that carries no
  # finding or whose drift × 2 would in fact sit under half the ceiling.

  Background:
    Given Inkling-Small GGUF metadata for the model named by INKLING_MODEL_PATH
    And a D0 oracle artifact directory named by INKLING_ORACLE_DIR with recorded hashes

  @d1 @arch
  Scenario: The architecture loads metadata and creates every tensor, no compute
    Given a dreamcatcher build with LLM_ARCH_INKLING registered
    When the model is loaded on CPU with compute disabled
    Then every tensor is created with the Inkling names and shapes
    And the KV keys include dense_block_count, d_rel, rel_extent, rel_extent_swa, shortconv_kernel, logit_scale_denom, log_scaling_n_floor, log_scaling_alpha, vocab_size, unpadded_vocab_size, block_count, attention.sliding_window, attention.sliding_window_pattern, attention.head_count_kv, feed_forward_length, expert_feed_forward_length, expert_count, expert_used_count, expert_shared_count, expert_weights_scale, and expert_gating_func
    And no forward pass has run

  # Cross-lineage envelope (D2 amendment, 2026-09-19; the gate rule is in the
  # feature header): the ik-lineage build is NOT expected to reproduce the
  # banded D0 oracle token-for-token — the old token-for-token and 4-dp
  # perplexity assertions are replaced by the KLD / top-1 agreement envelope.
  # Per metric, the INKLING_ENVELOPE_JSON file (env-bound like every runner
  # variable here, fail closed when unset) registers the measured
  # banded-vs-banded drift, the masked-path ceiling and the calibration
  # provenance; the working band is drift × 2 and is capped by the ceiling —
  # a band that is not under the ceiling, or wider than half of it, is a
  # finding (rules 2 and 3), never absorbed. The feature file pins no
  # calibration figure beyond the BAND line in the header; the only other
  # numbers here are the sha-pinned oracle values. Token-for-token equality
  # stays a gate ONLY for builds matching the oracle CPU feature set
  # (GGML_LLAMAFILE, GGML_CPU_REPACK, declared via INKLING_ORACLE_FEATURES);
  # otherwise the run records it as a diagnostic. The envelope metrics are the
  # KL-divergence run's top-1 agreement and RMS Δp against the banded base
  # (the BAND record in the header). greedy-64x8.json holds no logit vectors,
  # so the greedy scenario observes top-1 agreement only; RMS Δp binds in the
  # kld-base scenario.

  @d2 @parity
  Scenario: CPU greedy generation runs coherently on the D0 fixed prompt set
    Given the D0 oracle dump "greedy-64x8.json" exists in the INKLING_ORACLE_DIR directory
    And a pre-registered cross-lineage envelope exists in the INKLING_ENVELOPE_JSON file
    When the model runs greedy generation on CPU for the D0 fixed prompt set
    Then greedy generation completes on every prompt with finite, well-formed logprobs
    And token-for-token equality is asserted only when the build features match the oracle build (GGML_LLAMAFILE, GGML_CPU_REPACK) and is otherwise recorded as a diagnostic, not a gate

  @d2 @parity
  Scenario: 4-chunk KLD / top-1 agreement stays within the cross-lineage envelope of the banded D0 oracle
    Given the D0 oracle dump "kld-base-4x2048.bin" exists in the INKLING_ORACLE_DIR directory
    And a pre-registered cross-lineage envelope exists in the INKLING_ENVELOPE_JSON file
    When the model computes KL-divergence on CPU over the D0 4-chunk split against the banded oracle base
    Then the KLD / top-1 agreement against the banded D0 oracle stays inside the pre-registered working band, capped by the masked-path ceiling
    And a working band that is not under the masked-path ceiling, or wider than half the masked band, is surfaced as a finding, not absorbed

  # D3 gate (2026-09-19, noah's envelope rule, candreev-blessed): the cache
  # lane is gated by the same KLD / top-1 envelope as D2, against the banded
  # D0 oracle, at the practical band in the header's BAND record — the
  # masked-path ceiling itself (top-1 ≥ ~83.5 % within CI, RMS Δp ≤ ~10 %
  # within CI), with drift × 2 recorded there as a rule-3 finding pending
  # noah/emma's final encode. The D3 When records its observation on
  # inkling_observed when its steps land; the gate Then fails closed without
  # one.
  @d3 @cache
  Scenario: The hybrid attention+recurrent cache serves banded SWA windows
    Given a hybrid Inkling cache with iswa windows over a 5:1 pattern
    And a pre-registered cross-lineage envelope exists in the INKLING_ENVELOPE_JSON file
    When generation runs past several window boundaries with int64 expert indexing
    Then attention state, recurrent short-conv state, and expert indices stay consistent
    And the KLD / top-1 agreement against the banded D0 oracle stays inside the pre-registered working band, capped by the masked-path ceiling

  @d4a @layer-library
  Scenario: A sliced layer library reproduces the monolith exactly
    Given an Inkling layer library sliced with arch-aware dense_block_count
    When llama-server runs with "--model-dir" pointing at the sliced INKLING_LIB_DIR directory
    Then the served monolith equals the GGUF monolith
    And token IDs continue to equal the oracle

  @d4b @stage-ring
  Scenario: Head and tail ring stages reproduce the monolith
    Given a stage ring with head and tail roles hosting Inkling stages
    When generation runs for at least 64 tokens through the ring
    Then head and tail token IDs equal the monolith run
    And the boundary hidden state is byte-equal to the monolith run

  @d4c @remote-experts
  Scenario: Remote experts are byte-exact against in-process experts
    Given the expert server hosting the routed expert bank
    And the shared expert bank stays local
    When generation runs through the remote-expert path on CPU
    Then token IDs and logits are byte-exact against the in-process run

  @d6 @ci
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
