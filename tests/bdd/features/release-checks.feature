@release-checks @integrity
Feature: Run the repository release verification interfaces
  As a release operator
  I want machine-readable serving and artifact checks
  So that promotion decisions use recorded runtime and identity evidence

  @smoke @serving
  Scenario: A real chat completion passes the serving smoke check
    Given an owned "llama-server" endpoint is available at "http://<host>:<port>"
    And model discovery returns exactly one model
    When I run "ci/smoke-serve.sh http://<host>:<port> --min-tps 1.0"
    Then the script should request one non-streaming 32-token chat completion
    And the response should contain non-empty answer text
    And "usage.completion_tokens" should be a positive integer
    And the completion should pass the coherence screen
    And measured completion tokens per request second should be at least 1.0
    And the script should emit one JSON result with "ok" set to true
    And the process exit code should be 0

  @smoke @error-handling
  Scenario: A reasoning-only response fails the serving smoke check
    Given an owned endpoint returns empty answer text and non-empty "reasoning_content"
    When I run "ci/smoke-serve.sh http://<host>:<port>"
    Then the script should emit one JSON result with "ok" set to false
    And the result error should be "reasoning_only"
    And the process exit code should be non-zero

  @smoke @check-text
  Scenario: Check-text mode accepts a coherent saved completion without contacting a server
    Given "<coherent_text_file>" contains at least 5 distinct words
    And at least 40 percent of its words are distinct
    And at least 60 percent of its characters are alphanumeric or whitespace
    When I run "ci/smoke-serve.sh --check-text <coherent_text_file>"
    Then no HTTP request should be sent
    And the script should emit one JSON result with "mode" set to "check_text"
    And the result should have "ok" set to true
    And the process exit code should be 0

  @smoke @check-text @error-handling
  Scenario: Check-text mode rejects a degenerate saved completion
    Given "<degenerate_text_file>" contains "01111111111111111111111111111111"
    When I run "ci/smoke-serve.sh --check-text <degenerate_text_file>"
    Then no HTTP request should be sent
    And the script should emit one JSON result with "ok" set to false
    And the result error should be "degenerate_too_few_distinct_words"
    And the process exit code should be non-zero

  @checksum @artifact-integrity
  Scenario: Report a model artifact whose SHA-256 checksum matches
    Given "<fixture_checkout>/SHA256SUMS" names a model artifact and its SHA-256 digest
    And the named artifact exists under "<fixture_checkout>" with matching bytes
    When I run "<fixture_checkout>/scripts/verify-checksum-models.py"
    Then the artifact row should contain "V" in the valid checksum column
    And the artifact row should have an empty missing-file column
    And the process exit code should be 0

  @checksum @error-handling
  Scenario: Reject a checksum run without the repository SHA256SUMS file
    Given "<fixture_checkout>/SHA256SUMS" does not exist
    When I run "<fixture_checkout>/scripts/verify-checksum-models.py"
    Then the diagnostic should identify the missing SHA256SUMS path
    And the process exit code should be non-zero

  @stage-manifest @parser
  Scenario: The stage-manifest test executable enforces the source block-count schema
    Given "test-stage-manifest" has been built under "<build_dir>"
    When I run "<build_dir>/bin/test-stage-manifest"
    Then its fixtures should accept a positive 32-bit integer at "source.block_count"
    And its fixtures should ignore decoy block counts outside "source.block_count"
    And its fixtures should reject missing, non-integer, non-positive, overflowing, and trailing values
    And stdout should contain "stage manifest: 20 cases passed"
    And the process exit code should be 0
