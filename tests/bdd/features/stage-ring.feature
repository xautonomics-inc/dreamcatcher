@stage-ring @distributed-inference
Feature: Multi-Host Stage-Runner Pipeline Rings
  As a distributed systems engineer
  I want to orchestrate pipelined inference rings across sequential stage runners
  So that large language models exceeding single-host VRAM capacity execute reliably across compute stages

  Background:
    Given the model contains 48 total transformer layers
    And a valid partitioned model library exists at "<lib_dir>"

  @loopback @smoke
  Scenario: Bring up a two-stage loopback ring on a single host
    # Canonical invocation per docs/BRING-UP.md: tail listens first, head dials in.
    # Only real llama-stage-runner flags are used (--listen, --connect, --prompt,
    # --max-tokens); generation over HTTP requires --role server and is out of scope
    # for a plain head/tail ring, so the completion steps below skip gracefully.
    When I start a "tail" stage runner with arguments:
      | flag         | value     |
      | --role       | tail      |
      | --model-dir  | <lib_dir> |
      | --layers     | 24,48     |
      | --listen     | <port>    |
      | --n-ctx      | 512       |
      | --max-tokens | 12        |
    And I start a "head" stage runner with arguments:
      | flag           | value             |
      | --role         | head              |
      | --model-dir    | <lib_dir>         |
      | --layers       | 0,24              |
      | --connect      | 127.0.0.1:<port>  |
      | --prompt       | Ping              |
      | --max-tokens   | 12                |
      | env:STAGE_EMIT | hidden            |
    Then the tail stage should report "listening on :<port>"
    And the head stage should report "IL=[0,24)"
    And the tail stage should report "IL=[24,48)"
    And the head stage should transmit hidden activation tensors to the tail stage
    And the tail stage should evaluate layers 24 through 47 and compute final logits

  @three-stage @multi-host
  Scenario: Bring up a three-stage heterogeneous pipeline ring
    Given three networked compute stages "<host_head>", "<host_relay>", and "<host_tail>"
    When I launch stage "tail" on "<host_tail>" covering layers 32 to 48 connecting to "<host_head>"
    And I launch stage "relay" on "<host_relay>" covering layers 16 to 32 connecting to "<host_tail>"
    And I launch stage "head" on "<host_head>" covering layers 0 to 16 connecting to "<host_relay>"
    Then the ring topology "head -> relay -> tail -> head" should be established
    And activation tensor handoffs should flow sequentially across stages without dropped waves

  @per-layer-embedding @ple @known-issue @meta-97
  Scenario: Known issue: Per-layer-embedding blocks in later stage window unsupported in ring transport
    Given a model architecture with per-layer input embeddings
    When I launch a tail stage runner with layers "<tail_start>,<tail_end>" covering a PLE block
    Then the tail stage log should emit an advisory warning regarding per-layer embedding slice placement:
      """
      warning: per-layer-embedding tensor in non-zero stage window
      """
    And the ring transport cannot forward raw token IDs to non-head stages as tracked under meta#97
    And multi-stage partitioning across PLE blocks is unsupported until token ID transport is added

  @error-handling @network-faults
  Scenario: Handle unreachable downstream stage during ring bring-up
    When I start a "head" stage runner pointing to unreachable downstream address "<bad_host>:<bad_port>"
    Then the stage runner should retry connection up to the configured connection timeout
    And if the downstream stage remains unreachable, the runner should exit with a descriptive connection failure error

  @shutdown @teardown
  Scenario: Clean shutdown and socket resource cleanup on ring termination
    Given an active two-stage pipeline ring
    When I send SIGTERM to the head stage runner process
    Then the head stage should forward a termination wave to the tail stage
    And both stages should close their TCP sockets without address binding leaks
    And both processes should exit cleanly with return code 0
