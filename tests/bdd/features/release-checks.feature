@release-checks @integrity-suite @ci
Feature: Release Integrity, Checksum Verification, and Serving Quality Gates
  As a release engineering and quality assurance lead
  I want to validate artifact integrity, manifest schemas, and runtime serving coherence
  So that only verified, bit-accurate, and non-degenerate builds are promoted to production

  Background:
    Given the repository root contains verification scripts in "ci/" and "scripts/"

  @smoke-serve @coherent-completion
  Scenario: Automated serving smoke test verifies throughput and coherence
    Given an active "llama-server" endpoint reachable at "http://<host>:<port>"
    When I run the smoke test script "ci/smoke-serve.sh" with arguments:
      | argument  | value                    |
      | url       | http://<host>:<port>     |
      | --min-tps | 1.0                      |
    Then the process exit code should be 0
    And the stdout JSON should contain:
      """
      "ok": true
      """
    And the generated completion text should pass vocabulary entropy and distinct-token checks

  @smoke-serve @degeneracy-detection
  Scenario: Check-text mode detects and rejects degenerate repetitive text
    Given a text file "<degenerate_file>" containing repetitive degenerate output:
      """
      01111111111111111111111111111111
      """
    When I run "ci/smoke-serve.sh" with arguments:
      | argument     | value              |
      | --check-text | <degenerate_file>  |
    Then the process exit code should be 1
    And the stdout JSON should contain:
      """
      "ok": false
      """
    And the error field should indicate a degeneracy condition:
      """
      "error": "degenerate_low_distinct_ratio"
      """

  @smoke-serve @valid-text
  Scenario: Check-text mode approves coherent natural language completion
    Given a text file "<coherent_file>" containing natural text:
      """
      The capital of France is Paris, located along the Seine River in northern France.
      """
    When I run "ci/smoke-serve.sh" with arguments:
      | argument     | value             |
      | --check-text | <coherent_file>   |
    Then the process exit code should be 0
    And the stdout JSON should contain:
      """
      "ok": true
      """

  @checksum-verification @artifact-integrity
  Scenario: Model checksum verification passes on authentic weights
    Given an authentic model artifact directory "<model_dir>"
    And an authoritative digest file "<checksums_file>" with SHA-256 digests
    When I run "scripts/verify-checksum-models.py" against "<model_dir>"
    Then all artifact checksums should match
    And the verification script should report:
      """
      All model checksums verified successfully
      """
    And the exit code should be 0

  @checksum-verification @tamper-detection
  Scenario: Model checksum verification fails on corrupted or modified weights
    Given a model artifact directory with a modified or corrupted tensor file
    When I run "scripts/verify-checksum-models.py" against the corrupted directory
    Then the verification script should detect a checksum mismatch
    And the process should exit with a non-zero status code

  @stage-manifest @parser-unit
  Scenario Outline: Stage manifest parser validates source block count schema
    Given a manifest string with source configuration "<manifest_input>"
    When I execute the stage manifest parser "test-stage-manifest"
    Then the parser should "<expected_outcome>"

    Examples:
      | manifest_input                                                      | expected_outcome                           |
      | {"source":{"block_count":43}}                                       | accept and return block count 43           |
      | {"files":[{"window_kv":{"block_count":1}}],"source":{"block_count":43}} | accept and ignore file-level window kv |
      | {"source":{"block_count":2147483647}}                               | accept maximum 32-bit integer              |
      | {"source":{"block_count":0}}                                        | reject non-positive block count            |
      | {"source":{"block_count":-1}}                                       | reject negative block count                |
      | {"source":{"block_count":43.5}}                                     | reject floating-point count                |
      | {"source":{"block_count":"43"}}                                     | reject string value count                  |
      | {"source":{"block_count":2147483648}}                               | reject integer overflow > 2^31-1           |
      | {"source":{}}                                                       | reject missing block count                 |

  @known-issue @meta-84
  Scenario: Known issue: Vulkan ICD drift on NVIDIA Vulkan driver
    Given a Vulkan backend build running on an NVIDIA Vulkan ICD
    When I execute token generation with greedy decoding
    Then generated tokens drift from the CPU reference output as tracked under meta#84
    And an empty GGML_VK_VISIBLE_DEVICES environment variable causes a crash

  @known-issue @meta-96
  Scenario: Known issue: DeepSeek-V4 Vulkan backend evaluation
    Given a DeepSeek-V4 model evaluated using the Vulkan compute backend
    When prompt prefill and token generation are initiated
    Then the backend encounters kernel planning errors tracked under meta#96
    And the current workaround is executing DeepSeek-V4 stages on the CPU or CUDA backend
