@gslot-runtime @slot-arbiter @scheduling
Feature: Global Slot Router Runtime Gate and Arbiter Client
  As an infrastructure reliability engineer
  I want to arbitrate compute slots and request leases across pipelined stages via `gslot-runtime`
  So that multi-tenant concurrency is fairly scheduled without pipeline stalls or deadlocks

  Background:
    Given a pipeline stage runner configured with model layers

  @default-off @zero-overhead
  Scenario: Default OFF behavior when STAGE_GSLOT_SOCKET is unset
    Given environment variable "STAGE_GSLOT_SOCKET" is unset
    When the stage runner executes prompt prefill and token generation
    Then the gslot scheduling gate should evaluate to true with zero syscall overhead
    And token generation behavior and logits should be bit-for-bit identical to baseline

  @fail-open @resilience
  Scenario Outline: Strict fail-open discipline when arbiter is unreachable or faults
    Given "STAGE_GSLOT_SOCKET" points to "<fault_condition>"
    When the stage runner attempts to acquire a compute slot lease
    Then the gslot gate should immediately fail open
    And the fault telemetry counter should increment
    And token generation must proceed without blocking or stalling the pipeline

    Examples:
      | fault_condition           | description                             |
      | /tmp/nonexistent_gslot.sock | missing unix domain socket              |
      | /tmp/stale_hung_gslot.sock  | socket connected but daemon not reading |
      | 127.0.0.1:59999           | unreachable TCP port                    |

  @fail-open @daemon-restart
  Scenario: Maintain continuous token generation during arbiter daemon restart
    Given an active gslot arbiter daemon running on "<arbiter_socket>"
    And a stage runner successfully acquiring leases
    When the gslot arbiter daemon process is terminated mid-generation
    Then the stage runner should detect lease timeout and fail open
    And ongoing token generation should continue without dropped tokens
    When a new gslot arbiter daemon restarts on "<arbiter_socket>"
    Then the stage runner should automatically reconnect and resume normal lease acquisition

  @lease-discipline @quantum
  Scenario: Quantum lease discipline for high-throughput continuous batching
    Given an active gslot arbiter daemon
    And environment variable "STAGE_GSLOT_DISCIPLINE" is set to "quantum"
    When the stage runner receives a batch of generation waves
    Then the runner should acquire a time-bounded lease of duration 250 milliseconds
    And subsequent token steps within the active quantum should proceed without re-requesting leases
    And the lease should automatically renew or release upon quantum expiration

  @lease-discipline @burst
  Scenario: Burst lease discipline with acquire-before-compute and release-on-handoff
    Given an active gslot arbiter daemon
    And environment variable "STAGE_GSLOT_DISCIPLINE" is set to "burst"
    When the stage runner prepares to compute its assigned layer slice
    Then it must acquire a burst compute lease prior to kernel dispatch
    And immediately upon completion and handoff to the next stage, the lease must be released

  @concurrency @fairness
  Scenario: Fair compute slot scheduling across competing pipeline stages
    Given two competing stage runner processes configured with lease weights 1 and 2
    When both processes submit high-concurrency requests to the arbiter
    Then the arbiter should grant slot leases in accordance with configured tenant weights
    And neither stage runner should experience starvation
