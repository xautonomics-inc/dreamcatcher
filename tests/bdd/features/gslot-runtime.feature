@gslot-runtime @scheduling
Feature: Coordinate stage compute through the optional gslot gate
  As an operator sharing compute between stage processes
  I want stages to acquire and release host-local leases
  So that scheduling can coordinate dispatch without becoming a generation dependency

  Background:
    Given a stage runner with an instrumented "gslot" gate

  @default-off @parity
  Scenario: The gate is disabled when STAGE_GSLOT_SOCKET is unset
    Given "STAGE_GSLOT_SOCKET" is unset
    When the stage runner performs prefill and greedy generation
    Then every gate check should permit compute without a socket call
    And the generated token IDs should equal the ungated reference

  @lease @heartbeat
  Scenario: Register, acquire, heartbeat, and release against a live arbiter
    Given a gslot arbiter listens on "<arbiter_socket>"
    And the stage runner environment contains:
      | variable             | value              |
      | STAGE_GSLOT_SOCKET   | <arbiter_socket>   |
      | STAGE_GSLOT_TENANT   | bdd-stage          |
      | STAGE_GSLOT_RESOURCE | cpu:host           |
      | STAGE_GSLOT_MODE     | quantum            |
    When the stage runner requests permission to compute across multiple heartbeat intervals
    Then the tenant should register with the arbiter
    And the client should acquire a positive lease ID before dispatch
    And the client should send progress heartbeats while registered
    And the client should release its held lease when the stage becomes idle

  @lease @quantum
  Scenario: Reuse a quantum lease until its local deadline
    Given a live arbiter grants a lease to a stage using "STAGE_GSLOT_MODE=quantum"
    And "STAGE_GSLOT_QUANTUM_MS" is unset
    When the stage checks the gate repeatedly during the next 250 milliseconds
    Then compute should remain permitted under the held lease
    And the stage should not request another lease before the quantum expires
    When the quantum expires and the stage still wants compute
    Then the old lease should be released before another lease is requested

  @lease @burst
  Scenario: Release a burst lease when a pipeline wave is handed off
    Given a live arbiter grants a lease to a stage using "STAGE_GSLOT_MODE=burst"
    When the stage receives a wave and is ready to dispatch compute
    Then the stage should acquire a lease before compute starts
    When the stage emits the wave to the next stage
    Then the stage should release the lease immediately through handoff

  @lease @backpressure
  Scenario: A denied lease delays dispatch without counting as a transport fault
    Given a live arbiter reports that another tenant holds the resource
    When the stage asks the gate for permission to compute
    Then the gate should deny dispatch for the configured retry interval
    And the blocked counter should increment
    And the fault counter should remain unchanged

  @fail-open @resilience
  Scenario Outline: An unavailable Unix-socket arbiter cannot halt the ring
    Given "STAGE_GSLOT_SOCKET" points to "<arbiter_socket>"
    And the socket condition is "<condition>"
    When the stage asks the gate for permission to compute
    Then the gate should permit compute after the bounded socket attempt fails
    And the fault counter should increment
    And the client should disconnect and wait 2 seconds before another connection attempt

    Examples:
      | condition                         |
      | the Unix socket does not exist    |
      | the arbiter does not reply in time|
      | the arbiter returns malformed data|

  @fail-open @reconnect
  Scenario: Re-register after the arbiter restarts
    Given a registered stage has previously acquired a lease
    When the arbiter restarts and the next RPC fails
    Then that gate check should fail open
    And generation should continue during the client backoff
    When the arbiter is listening again after the backoff
    Then the client should reconnect and register the tenant again
    And later compute should use newly granted leases
