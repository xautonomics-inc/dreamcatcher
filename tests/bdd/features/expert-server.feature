@expert-server @moe-disaggregation
Feature: Disaggregated MoE Expert Server and Remote Client
  As an inference systems architect
  I want to disaggregate sparse MoE expert feed-forward computation onto dedicated expert servers
  So that large parameter MoE models execute on cost-effective hardware while maintaining dense attention in accelerator VRAM

  Background:
    Given a Mixture-of-Experts (MoE) model with 64 total routed experts per layer
    And a running "llama-expert-server" process listening on "<expert_host>:<expert_port>"
    And the expert server has loaded routed expert weights for layers 0 through 31

  @connection @handshake
  Scenario: Establish connection and capability handshake with remote expert server
    When a primary stage runner starts with environment:
      | variable             | value                               |
      | LLAMA_EXPERTS_REMOTE | <expert_host>:<expert_port>@0-31    |
    Then the stage runner should connect to "<expert_host>:<expert_port>" over TCP
    And both processes should complete the capability handshake successfully
    And the stage runner should log:
      """
      remote-experts: connected to <expert_host>:<expert_port> covering layers 0-31
      """

  @parity @bit-exact
  Scenario: Verify byte-exact logit parity between remote expert server and local CPU reference
    Given an identical MoE model running with all experts evaluated locally on CPU
    When I submit a deterministic evaluation request with prompt "Scientific inquiry requires hypothesis testing"
    And the request routes through the remote expert server for layers 0 to 31
    Then the computed hidden activation outputs from the expert server should match local CPU execution bit-for-bit
    And the final output logits should be byte-identical to the local CPU baseline

  @moe-kernels @fmoe-mmad
  Scenario Outline: Maintain mathematical parity across fused MoE and MMAD execution modes
    When the primary stage runner runs with expert offload mode "<mode>"
    And the expert server evaluates top-k routed expert FFN blocks
    Then the output tokens should match the reference unquantized greedy baseline

    Examples:
      | mode   | description                   |
      | --fmoe | Fused MoE kernel path         |
      | --mmad | Matrix multiply-add path      |

  @guards @bounds-checking
  Scenario: Reject top-k routing request exceeding configured expert bound
    When the client transmits an activation wave requesting top-k value "16"
    But the expert server is configured with maximum top-k bound "8"
    Then the expert server should reject the request with an explicit bounds error
    And the primary runner should report a capability violation

  @guards @unsupported-arch
  Scenario Outline: Refuse unsupported MoE architectural variants with explicit assertion
    Given a model architecture specifying "<unsupported_feature>"
    When the primary stage runner attempts to bind remote experts
    Then initialization should abort with an explicit guard message:
      """
      unsupported remote MoE feature
      """

    Examples:
      | unsupported_feature          |
      | per-expert biases            |
      | grouped expert routing       |
      | non-SILU activation function |
      | weight-before-ffn layout     |

  @fault-tolerance @reconnect
  Scenario: Graceful recovery and reconnect on transient socket disruption
    Given an active connection between stage runner and expert server
    When a temporary network interruption occurs on the expert socket
    Then the client transport should execute configured reconnect and retry attempts
    And upon socket reconnection, expert tensor dispatch should resume without memory corruption
