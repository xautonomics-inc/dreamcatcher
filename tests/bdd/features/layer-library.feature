@layer-library @slicer @distribution-planner
Feature: Per-Layer Model Library Slicing and Distribution Planning
  As a machine learning platform engineer
  I want to partition monolithic GGUF weights into verifiable per-layer model libraries and plan stage assignments
  So that worker nodes download and map only the exact layer slices required by their pipeline stages

  Background:
    Given a monolithic GGUF model file exists at "<monolith_path>"
    And the model contains "<total_blocks>" transformer blocks

  @slicer @manifest
  Scenario: Slice a monolithic GGUF model into a per-layer library
    When I run the layer slicer "slice_gguf_layers.py" with arguments:
      | argument     | value           |
      | model_source | <monolith_path> |
      | output_dir   | <output_dir>    |
    Then the output directory should contain block files "blk-00000.gguf" through "blk-<last_block>.gguf"
    And the output directory should contain mandatory header file "parts-embd.gguf"
    And the output directory should contain mandatory tail file "parts-output.gguf"
    And the output directory should contain "manifest.json"
    And the "manifest.json" format field should be "gguf-layer-library/v1"
    And the "manifest.json" source block count should equal "<total_blocks>"
    And each file entry in the manifest should record valid "n_bytes_file", "n_tensors", and "hash" fields

  @slicer @tensor-integrity
  Scenario: Verify cryptographic tensor-hash equality between sliced parts and source monolith
    Given a per-layer library generated from "<monolith_path>" in "<output_dir>"
    When I compute the Blake2b-128 digest of each tensor across all sliced block files
    Then every tensor digest should match the corresponding tensor digest in the source monolithic GGUF
    And the manifest "source.content_hash" should match the aggregate hash over sorted source tensor digests

  @slicer @options
  Scenario: Slicing with --no-hash option skips hash computation
    When I run the layer slicer "slice_gguf_layers.py" with arguments:
      | argument     | value           |
      | model_source | <monolith_path> |
      | output_dir   | <fast_out_dir>  |
      | --no-hash    |                 |
    Then the output directory should contain all block and part files
    And in "manifest.json" the file hash fields should be recorded as empty or skipped

  @distribution-planner @planner-window
  Scenario: Derive exact required file set for an arbitrary stage window
    Given a valid layer library manifest with 64 total blocks
    When I call the distribution planner "files_for_window" for layer window "[16, 32)"
    Then the resolved file set should include blocks "blk-00016.gguf" through "blk-00031.gguf"
    And the resolved file set must include "parts-embd.gguf"
    And the resolved file set must include "parts-output.gguf"
    And blocks outside the window "blk-00000.gguf" to "blk-00015.gguf" should be excluded

  @distribution-planner @planner-precheck
  Scenario: Storage precheck calculates exact disk space feasibility
    Given a layer library manifest requiring 12.5 GB for window "[0, 24)"
    When I execute storage precheck against a target filesystem with 20.0 GB free space
    Then precheck should return feasible "true" with a surplus of 7.5 GB
    When I execute storage precheck against a target filesystem with 10.0 GB free space
    Then precheck should return feasible "false" with a deficit of 2.5 GB

  @distribution-planner @planner-rebalance
  Scenario: Rebalance calculates migration delta across pipeline topology changes
    Given an existing stage allocation covering window "[0, 16)"
    When the pipeline is reconfigured to assign window "[0, 24)" to the node
    Then the rebalance planner should report "files_to_add" containing blocks "blk-00016.gguf" through "blk-00023.gguf"
    And "files_removable" should be empty
    And existing files for blocks 0 through 15 should not be re-downloaded
