---
name: ik-fork-bring-up
description: Orchestrate and verify multi-stage inference pipelines (single-box loopback and two-box TCP) using llama-stage-runner and layer distribution tooling on the ik_llama.cpp fork.
---

# Multi-Stage Bring-Up Skill (ik_llama.cpp)

This skill guides autonomous coding and orchestration agents in bringing up, configuring, and verifying multi-stage distributed inference on the `ik_llama.cpp` fork.

## Canonical Reference

For the comprehensive step-by-step procedure, measurement instructions, and failure handling, refer to:
- **[`docs/BRING-UP.md`](../../../docs/BRING-UP.md)**

## House Rules of Operation (Verbatim)

All agents executing bring-up or verification must follow these mandatory rules:

1. **/health is not liveness (prove with a completion)**
   HTTP 200 on an endpoint does not verify generation capability. Liveness requires a verified token completion stream.
2. **arithmetic is not proof (feasibility is a screen; the proof is a survived prefill)**
   Memory accounting and planner calculations are feasibility screens only. Runtime proof requires a survived prefill and successful forward pass.
3. **the layer window is a launch flag**
   Layer splits and assignments are determined at process launch time (`--layers <start>,<end>`), decoupled from static model files.
4. **pin GPUs by UUID never by index**
   Never rely on ordinal GPU indexes (e.g. `CUDA_VISIBLE_DEVICES=0`). Always query and pin via device UUID (`CUDA_VISIBLE_DEVICES=GPU-<uuid>`).
5. **measure never declare**
   Every configuration, performance figure, or topology assertion must be accompanied by explicit measurement and verification.

## Core Workflows

### 1. Pre-Flight Stage Allocation
Use `layer_distribution.plan` to compute deterministic stage windows and screen VRAM feasibility:
```bash
python3 -m layer_distribution.plan /models/model-layers \
  --node stage0:16GiB:100GiB:8080 \
  --node stage1:16GiB:100GiB:8081 \
  --model-dir /models/model-layers
```
Confirm:
- Status reports `PASS (weights)`
- Layer windows partition the total layer count without gaps or overlaps

### 2. Single-Box Loopback Pipeline
1. **Tail Stage (GPU 1, Listener)**:
   ```bash
   export CUDA_VISIBLE_DEVICES=GPU-<tail-uuid>
   export STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=24
   llama-stage-runner --role tail --listen 8081 --model-dir /models/model-layers --layers 24,48
   ```
   Measure: socket listening on `127.0.0.1:8081` (`ss -tulpn | grep 8081`).
2. **Head Stage (GPU 0, Driver)**:
   ```bash
   export CUDA_VISIBLE_DEVICES=GPU-<head-uuid>
   export STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=24 STAGE_EMIT=hidden
   llama-stage-runner --role head --connect 127.0.0.1:8081 --model-dir /models/model-layers --layers 0,24 --prompt "Hello" --max-tokens 32
   ```
   Measure: successful TCP handshake, forward pass execution, token stream on stdout with tok/s.

### 3. Release Smoke Verification
Verify OpenAI-compatible endpoints using the release smoke tool:
```bash
ci/smoke-serve.sh http://127.0.0.1:8080 --min-tps 1.0
```

## Skill Mirroring Note
This skill is duplicated at `.claude/skills/ik-fork-bring-up/SKILL.md` as an exact file copy (not a symlink) to ensure deterministic discovery across Antigravity, OpenAI Codex, and Claude Code regardless of container sandbox symlink handling or platform file system configurations.
