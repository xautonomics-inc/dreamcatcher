@expert-server @moe-disaggregation
Feature: Route MoE expert computation to an expert server
  As an operator of a sparse MoE model
  I want routed-expert computation to run in a separate process
  So that the attention process can omit covered expert tensors without changing the result

  Background:
    Given a supported MoE model at "<model_path>"
    And a deterministic greedy prompt and generation length

  @smoke @connection
  Scenario: Connect an attention client to the layers served by an expert server
    Given "llama-expert-server" is started with:
      """
      --role expert-server --model <model_path> --expert-layers 0-31 --host <expert_host> --listen <expert_port>
      """
    When "llama-expert-check" starts with "LLAMA_EXPERTS_REMOTE=<expert_host>:<expert_port>@0-31"
    Then the client and server complete the version 1 capability handshake
    And the client verifies the model embedding width and served layer set
    And covered layers send hidden values, top-k expert IDs, and top-k weights
    And covered layers receive the accumulated routed-expert output

  @parity @byte-exact
  Scenario Outline: Remote CPU experts are byte-exact when client and server fusion modes match
    Given a local "llama-expert-check" reference uses client arguments "<client_arguments>"
    And a CPU expert server uses server arguments "<server_arguments>"
    When the same check runs remotely for every MoE layer with client arguments "<client_arguments>"
    Then every generated token ID should equal the local reference
    And every per-step logits hash should equal the local reference
    And the remote and local raw logits dumps should be byte-identical

    Examples:
      | client_arguments       | server_arguments       |
      |                        | --fmoe 1 --mmad 1      |
      | -no-fmoe -no-mmad      | --fmoe 0 --mmad 0      |

  @parity @fusion-contract
  Scenario: The parity gate rejects mismatched client and server fusion modes
    Given the local reference uses the default fused client path
    And the expert server runs with "--fmoe 0 --mmad 0"
    When the same deterministic check runs through the expert server
    Then at least one logits hash should differ from the local reference
    And the run should not be reported as byte-exact

  @routing @multi-endpoint
  Scenario: Route disjoint layer ranges to multiple expert servers
    Given one expert server serves layers 0 through 15
    And another expert server serves layers 16 through 31
    When the client starts with "LLAMA_EXPERTS_REMOTE=<expert_host>:<expert_port>@0-15;<host>:<port>@16-31"
    Then each covered layer should be routed to the endpoint that owns it
    And the deterministic logits should be byte-identical to local CPU experts

  @error-handling @routing
  Scenario: Reject overlapping multi-endpoint coverage
    When the client starts with two remote endpoints that both claim layer 15
    Then configuration parsing should abort before model execution
    And the diagnostic should identify overlapping layer coverage

  @reconnect @fault-tolerance
  Scenario: Retry a transient expert-server connection failure
    Given a client has completed the capability handshake with an expert server
    When the connection fails during an expert RPC and the server becomes available again
    Then the client should tear down the failed connection
    And the client should retry the RPC at most 3 times with reconnect and backoff
    And a successful reconnect should repeat the capability handshake before another expert call

  @default-off @regression
  Scenario: Leave the local expert path unchanged when remote experts are not configured
    Given "LLAMA_EXPERTS_REMOTE" is unset
    When the deterministic check runs with all experts local
    Then no remote expert connection should be attempted
    And the generated token IDs and logits hashes should equal the local baseline
