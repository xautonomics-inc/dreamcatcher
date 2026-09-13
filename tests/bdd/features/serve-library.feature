@serve-library @library-assembly
Feature: Single-Process Per-Layer Model Library Serving
  As a distributed systems operator or inference engineer
  I want to serve partitioned per-layer model libraries directly via `llama-server --model-dir`
  So that I can achieve single-process serving without monolithic conversion, pipeline overhead, or multi-host networking

  Background:
    Given a valid model library exists at "<lib_dir>"
    And the library contains a valid "manifest.json" of format "gguf-layer-library/v1"
    And the library contains parts for all layers from 0 to "<total_layers>"
    And the library contains mandatory stage parts "parts-embd.gguf" and "parts-output.gguf"

  @smoke @happy-path
  Scenario: Successfully load and serve a complete layer library
    When I launch "llama-server" with arguments:
      | flag        | value        |
      | --model-dir | <lib_dir>    |
      | --host      | <host>       |
      | --port      | <port>       |
    Then the server log should report model assembly:
      """
      model-dir: assembling window [0,<total_layers>)
      """
    And the server log should confirm part assembly:
      """
      llama_model_loader: assembled
      """
    And the server should report "HTTP server listening"
    When I send a GET request to "http://<host>:<port>/health"
    Then the HTTP status code should be 200
    And the response JSON should indicate health status "ok"
    When I send a GET request to "http://<host>:<port>/v1/models"
    Then the HTTP status code should be 200
    And the response JSON should list the model ID matching the library manifest

  @parity @verification
  Scenario: Verify greedy completion token parity against monolithic baseline
    Given an identical reference model running from monolithic GGUF file "<monolith_path>"
    When I launch "llama-server" with arguments:
      | flag        | value        |
      | --model-dir | <lib_dir>    |
      | --host      | <host>       |
      | --port      | <port>       |
    And I submit a chat completion request to "http://<host>:<port>/v1/chat/completions" with payload:
      """
      {
        "messages": [{"role": "user", "content": "Write a poem about distributed systems."}],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 64
      }
      """
    Then the HTTP status code should be 200
    And the completion tokens should be bit-for-bit identical to the monolithic reference output

  @windowing
  Scenario: Load a bounded layer window slice via --layers
    When I launch "llama-server" with arguments:
      | flag        | value        |
      | --model-dir | <lib_dir>    |
      | --layers    | <start>,<end>|
      | --host      | <host>       |
      | --port      | <port>       |
    Then the server log should report window assembly:
      """
      model-dir: assembling window [<start>,<end>)
      """
    And only part files corresponding to blocks "<start>" through "<end_minus_one>" should be memory-mapped
    And the mandatory files "parts-embd.gguf" and "parts-output.gguf" should be memory-mapped

  @error-handling
  Scenario: Reject mutually exclusive -m and --model-dir flags
    When I launch "llama-server" with arguments:
      | flag        | value          |
      | -m          | <monolith_path>|
      | --model-dir | <lib_dir>      |
    Then the process should exit with a non-zero status code
    And the standard error should contain:
      """
      error: -m/--model and --model-dir are mutually exclusive
      """

  @error-handling
  Scenario: Reject non-existent library directory
    When I launch "llama-server" with arguments:
      | flag        | value                |
      | --model-dir | /non/existent/path   |
    Then the process should exit with a non-zero status code
    And the standard error should report that the directory could not be opened

  @error-handling
  Scenario: Reject library missing manifest.json
    Given a directory "<corrupt_dir>" containing part files but lacking "manifest.json"
    When I launch "llama-server" with arguments:
      | flag        | value         |
      | --model-dir | <corrupt_dir> |
    Then the process should exit with a non-zero status code
    And the standard error should report missing manifest

  @error-handling
  Scenario Outline: Reject invalid layer window specifications
    When I launch "llama-server" with arguments:
      | flag        | value        |
      | --model-dir | <lib_dir>    |
      | --layers    | <window>     |
    Then the process should exit with a non-zero status code
    And the standard error should report an invalid layer window

    Examples:
      | window     | reason                        |
      | 20,10      | start index greater than end  |
      | 15,15      | zero-length window            |
      | -1,10      | negative start index          |
      | 0,99999    | end index exceeds block count |
      | invalid    | non-numeric input             |

  @error-handling
  Scenario: Abort load on missing part file within assigned window
    Given a library directory with block "blk-00003.gguf" deleted
    When I launch "llama-server" with arguments:
      | flag        | value        |
      | --model-dir | <lib_dir>    |
      | --layers    | 0,10         |
    Then the process should fail during part enumeration
    And the standard error should indicate missing block file "blk-00003.gguf"

  @known-issue @meta-80
  Scenario: Known issue: Gemma-4 Q6_K token_embd degenerate output
    Given a Gemma-4 model library where "token_embd.weight" is quantized as "Q6_K"
    When I launch "llama-server" with "--model-dir <lib_dir>"
    And I submit a completion prompt to the server
    Then the output exhibits degenerate token repetition tracked under meta#80
    And the suggested workaround is using a "Q4_K" token embedding quant
