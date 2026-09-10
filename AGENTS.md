# Agent Guide — ik_llama.cpp

Operational orientation and house rules for autonomous coding and orchestration agents operating in this repository.

## Canonical Bring-Up Guide

The authoritative procedure for multi-stage pipeline configuration, orchestration, and verification is documented in:

- **[`docs/BRING-UP.md`](docs/BRING-UP.md)**: Two-stage loopback and multi-node TCP pipeline runbook.

## House Rules (Verbatim)

Agents operating in this repository must strictly adhere to the following five rules:

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

## Architecture & Tooling

- **Stage Runner (`llama-stage-runner`)**: Head/Tail pipeline executor exchanging hidden activations over TCP sockets.
- **Layer Distribution (`layer_distribution.plan`)**: Deterministic partitioner and feasibility calculator for layer assignment.
- **Release Verification (`ci/smoke-serve.sh`)**: Scripted OpenAI-compatible completion verification.
- **Agent Discovery**: See `.agents/skills/ik-fork-bring-up/SKILL.md` (and `.claude/skills/ik-fork-bring-up/SKILL.md`) for agent skill registration.
