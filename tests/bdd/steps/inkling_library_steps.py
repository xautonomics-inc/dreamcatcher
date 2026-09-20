"""BDD steps for the @d4a @layer-library and @d4b @stage-ring Inkling scenarios.

Both lanes prove the same thing in different shapes: a per-layer library of an
Inkling model, sliced with the arch-aware dense_block_count key
(tools/layer-distribution: `inkling.dense_block_count` per part, re-summed by
the loader's parts assembly), reproduces the GGUF monolith. The comparand is
always this build's OWN monolith run on the same host — one lineage, same
kernels — so the gate is exact equality (D4c uses the same rule), not the
cross-lineage envelope of the @d2/@d3 lanes.

Model under test: INKLING_LIB_DIR (D4a) / INKLING_STAGE_RING_MODEL (D4b) when
bound — a booked-window run on Inkling-Small — otherwise the tiny synthetic
fixture from tools/make-inkling-test-gguf.py (2 blocks: blk.0 dense + SWA,
blk.1 MoE + global), generated and sliced in-step (CI policy in the feature
header: no full model outside a window). A bound INKLING_LIB_DIR needs
INKLING_MODEL_PATH as its monolith.

D4a (llama-server --model-dir):
- Given: llama-server + slicer deps resolve; a bound library must carry
  `inkling.dense_block_count` in its block parts (the arch-aware key), or the
  scenario fails as a defect — a library sliced with the generic
  leading_dense_block_count spelling would assemble blk.0 as an MoE layer.
- When: two servers, same binary, same arguments (INKLING_LIBRARY_SERVER_ARGS
  adds to both): one on the library, one on the monolith. Each is driven
  through the greedy D0 request body (tools/inkling-greedy-dump.py) over the
  D0 prompt set when the oracle directory resolves, else a fixed prompt set.
- Then "served monolith equals GGUF monolith": the assembled model has the
  monolith's tensor count and dense block count, and the greedy dumps are
  identical position for position — token ids, logprobs and every listed
  candidate's logprob (the whole vocabulary in fixture mode; exact floats:
  same binary, same CPU, any deviation means the assembled graph differs).
- Then "token IDs continue to equal the oracle": in a window run with the D0
  oracle bound, the library's tokens are compared with greedy-64x8.json under
  the D2 rule (a gate only when INKLING_ORACLE_FEATURES / INKLING_BUILD_FEATURES
  are declared and equal, otherwise a diagnostic) and the top-1 disagreement
  is gated by the envelope when INKLING_ENVELOPE_JSON resolves. On the
  fixture there is no oracle; the monolith of the same build stands in and
  the ids must match it exactly (recorded as such, never silently).

D4b (llama-stage-runner ring, head + tail):
- Given: llama-stage-runner (LLAMA_STAGE_RUNNER_BIN), llama-expert-check
  (LLAMA_EXPERT_CHECK_BIN; the monolith greedy reference: a pure argmax
  decoder that prints every token id) and the slicer deps resolve.
- When: the library is split at the header's dense_block_count (head = the
  dense lead [0,D), tail = the MoE blocks [D,n_layer)); the tail listens, the
  head dials in with STAGE_EMIT=hidden and the ring decodes >= 64 tokens
  (STAGE_IGNORE_EOG, like the reference decoder). Then the boundary state:
  a head-window FILE run writes its emitted hidden rows (--out), and a
  monolith FILE run dumps the residual after the last head layer
  (STAGE_DUMP=l_out-<D-1>).
- Then: the ring's token ids equal the monolith reference's; the emitted
  boundary rows are byte-equal to the monolith's l_out-<D-1> tensor.

Fail closed: the Givens only resolve and record; the runner Whens re-check
and pytest.skip() with the precise reason before generating a fixture,
loading a model or starting a service. Every Then fails when its When
recorded nothing.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, then, when

from steps.expert_server_steps import _binary, _free_port
from steps.inkling_steps import (
    D6_ARCH,
    D6_GGUF_PY,
    D6_LOAD_ERROR_PATTERNS,
    REPO_ROOT,
    _d6_first_match,
    _declared_features,
    _envelope_fails,
    _envelope_unmet,
    _find_binary,
    _gguf_header_full,
    _greedy_comparison,
    _greedy_positions,
    _greedy_tool,
    _load_greedy_records,
    _oracle_unmet,
    _record_diagnostic,
    _surface_band_findings,
    generate_inkling_fixture,
)

LAYER_DIST = REPO_ROOT / "tools" / "layer-distribution"
DENSE_KEY = "inkling.dense_block_count"
THREADS = os.environ.get("INKLING_BDD_THREADS", "4")
STARTUP_TIMEOUT_S = float(os.environ.get("INKLING_SERVER_STARTUP_TIMEOUT", "1800"))
REQUEST_TIMEOUT_S = float(os.environ.get("INKLING_RUNNER_TIMEOUT", "7200"))
# fixture-mode prompt set: three prompts, distinct lengths, >= one window (32) on the fixture
FIXTURE_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Inkling is a hybrid attention and short-convolution mixture-of-experts architecture; "
    "this synthetic fixture only checks that its graph runs the same way from a library.",
    "0123456789 abcdefghijklmnopqrstuvwxyz",
]
FIXTURE_N_PREDICT = 16
# The fixture's byte-level vocab emits raw bytes under random weights; llama-server
# cannot JSON-encode a non-UTF-8 "content" (nlohmann type_error.316 -> HTTP 500).
# Fixture runs therefore constrain sampling to printable ASCII with a grammar —
# applied identically to the library and the monolith server, so the comparison
# stays exact (same logits, same mask, same argmax). Never applied to an oracle
# (window) run: the D0 body has no grammar.
FIXTURE_GRAMMAR = "root ::= [ -~]*"
D0_N_PREDICT = 64
N_PROBS = 10
RING_MIN_TOKENS = 64
RING_N_CTX = 512
STAGE_MAGIC = 0x53544732  # "STG2" — examples/stage-runner/stage-runner.cpp hidden_blob wire header (v2)
TOK_LINE = re.compile(r"^TOK step=(\d+) id=(\d+) logits_fnv=(0x[0-9a-f]+)$", re.MULTILINE)
RING_ID_LINE = re.compile(r"^\[id s(\d+)\](\d+)$", re.MULTILINE)
ASSEMBLED_LINE = re.compile(
    r"assembled (\d+) parts: block_count=(\d+) leading_dense_block_count=(\d+) "
    r"nextn_predict_layers=(\d+) \((\d+) tensors\)"
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _slicer_deps_unmet() -> str | None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return "numpy is not importable (fixture generator + slicer need it)"
    for path in (D6_GGUF_PY, LAYER_DIST):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import gguf  # noqa: F401
    except ImportError as error:
        return f"gguf-py is not importable from {D6_GGUF_PY}: {error}"
    try:
        from layer_distribution import slice_model  # noqa: F401
    except ImportError as error:
        return f"tools/layer-distribution is not importable: {error}"
    return None


def _gguf_field(path: Path, name: str):
    from gguf import GGUFReader

    entry = GGUFReader(str(path)).fields.get(name)
    return None if entry is None else entry.contents()


def _slice_library(model: Path, out_dir: Path) -> Path:
    """Slice with the arch-aware slicer and prove the arch key landed in the parts."""
    from layer_distribution import slice_model

    manifest = slice_model(str(model), out_dir, force=True)
    n_blocks = int(manifest["source"]["block_count"])
    for index in range(n_blocks):
        part = out_dir / f"blk-{index:05d}.gguf"
        if not part.is_file():
            pytest.fail(f"slicer wrote no {part.name}")
        if _gguf_field(part, DENSE_KEY) is None:
            pytest.fail(
                f"{part.name} carries no {DENSE_KEY}: the slicer is not arch-aware for Inkling "
                f"(it would fall back to leading_dense_block_count, which the Inkling loader never reads)"
            )
    return out_dir


def _check_bound_library(lib_dir: Path) -> str | None:
    """A bound INKLING_LIB_DIR must be a library whose parts carry the arch key."""
    if not (lib_dir / "manifest.json").is_file():
        return f"INKLING_LIB_DIR {lib_dir} has no manifest.json"
    first = lib_dir / "blk-00000.gguf"
    if not first.is_file():
        return f"INKLING_LIB_DIR {lib_dir} has no blk-00000.gguf"
    if _gguf_field(first, DENSE_KEY) is None:
        pytest.fail(
            f"{first} carries no {DENSE_KEY}: INKLING_LIB_DIR was not sliced with the arch-aware "
            f"dense_block_count (re-slice with tools/layer-distribution)"
        )
    return None


def _prompt_set(bdd_context: Any, mode: str, n_vocab: int) -> dict:
    """The request set: prompts, n_predict, n_probs, extra request fields, provenance.

    A window run with the D0 oracle bound drives the D0 prompt set with the
    D0 body (n_probs 10, as recorded). Anything else drives the fixed prompt
    set under the ASCII grammar with n_probs = n_vocab: the ik server reports
    the raw (pre-grammar) top-n, and the masked argmax of a near-uniform
    random-weight model is rarely inside a top-10, so the whole distribution
    is requested — which also makes the library-vs-monolith comparison a
    full per-position logit comparison, not a top-10 one.
    """
    if mode == "window" and not _oracle_unmet(bdd_context):
        oracle = bdd_context.inkling_oracle["dir"] / "greedy-64x8.json"
        records = _load_greedy_records(oracle.read_bytes(), str(oracle))
        return {"prompts": [record["prompt"] for record in records], "n_predict": D0_N_PREDICT,
                "n_probs": N_PROBS, "extra": None, "provenance": f"D0 prompt set ({oracle})"}
    return {"prompts": list(FIXTURE_PROMPTS), "n_predict": FIXTURE_N_PREDICT, "n_probs": n_vocab,
            "extra": {"grammar": FIXTURE_GRAMMAR}, "provenance": f"fixed prompt set, ASCII grammar, n_probs {n_vocab}"}


def _server_greedy(bdd_context: Any, tool, server: str, model_args: list[str], label: str, request: dict) -> dict:
    """Start llama-server on model_args, drive the greedy set, stop it; log + dump."""
    port = _free_port()
    log_path = bdd_context.tmp_path / f"{label}-llama-server.log"
    bdd_context.inkling_launches = getattr(bdd_context, "inkling_launches", 0) + 1
    server_args = f"-ngl 0 -t {THREADS} --no-warmup " + os.environ.get("INKLING_LIBRARY_SERVER_ARGS", "")
    process = tool.start_server(server, model_args, port, log_path, server_args)
    bdd_context.processes.append(process)
    try:
        tool.wait_health(process, port, STARTUP_TIMEOUT_S)
        records = tool.greedy_dump(port, request["prompts"], request["n_predict"], request["n_probs"],
                                   REQUEST_TIMEOUT_S, request["extra"])
    except RuntimeError as error:
        tail = log_path.read_bytes()[-3000:].decode("utf-8", "replace") if log_path.is_file() else ""
        pytest.fail(f"{label}: {error}\n--- server log tail ---\n{tail}")
    finally:
        tool.stop_server(process)
    return {"log": log_path.read_bytes().decode("utf-8", "replace"), "records": records, "args": model_args}


def _run(argv: list[str], env: dict[str, str], cwd: Path, timeout: float, log_name: str, bdd_context: Any) -> dict:
    bdd_context.inkling_launches = getattr(bdd_context, "inkling_launches", 0) + 1
    completed = subprocess.run(
        argv, capture_output=True, text=True, errors="replace", timeout=timeout,
        env={**os.environ, **env, "CUDA_VISIBLE_DEVICES": ""}, cwd=cwd, stdin=subprocess.DEVNULL, check=False,
    )
    (cwd / f"{log_name}.log").write_text("$ " + shlex.join(argv) + f"\n(exit {completed.returncode})\n\n"
                                         + completed.stderr + "\n--- stdout ---\n" + completed.stdout)
    return {"argv": argv, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


# ---------------------------------------------------------------------------
# D4a — llama-server --model-dir
# ---------------------------------------------------------------------------


@given(parsers.parse("an Inkling layer library sliced with arch-aware dense_block_count"))
def given_d4a_library(bdd_context: Any) -> None:
    state: dict[str, Any] = {"unmet": None, "lib_dir": None, "monolith": None, "mode": "fixture"}
    server, unmet = _find_binary("llama-server", "LLAMA_SERVER_BIN")
    state["server"] = server
    if unmet:
        state["unmet"] = unmet
    elif (deps := _slicer_deps_unmet()) is not None:
        state["unmet"] = deps
    raw = os.environ.get("INKLING_LIB_DIR")
    if raw:
        lib_dir = Path(raw)
        if not lib_dir.is_dir():
            state["unmet"] = state["unmet"] or f"INKLING_LIB_DIR {raw} is not a directory"
        elif state["unmet"] is None:
            state["unmet"] = _check_bound_library(lib_dir)
        monolith = getattr(bdd_context, "inkling_model_path", None)
        if monolith is None and state["unmet"] is None:
            state["unmet"] = "INKLING_LIB_DIR is bound but INKLING_MODEL_PATH (its monolith) is not"
        state.update(lib_dir=lib_dir, monolith=monolith, mode="window")
    bdd_context.inkling_d4a = state


@when(parsers.parse('llama-server runs with "--model-dir" pointing at the sliced INKLING_LIB_DIR directory'))
def when_d4a_serve(bdd_context: Any) -> None:
    state = getattr(bdd_context, "inkling_d4a", None)
    if state is None:
        pytest.skip("Prerequisite unmet: the D4a library Given did not run")
    if state["unmet"]:
        pytest.skip(f"Prerequisite unmet: {state['unmet']}")
    tool = _greedy_tool()
    if state["mode"] == "fixture":
        fixture = generate_inkling_fixture(bdd_context, "inkling-d4a.gguf")
        state["monolith"] = fixture["path"]
        state["lib_dir"] = _slice_library(fixture["path"], bdd_context.tmp_path / "inkling-d4a-lib")
    monolith, lib_dir = Path(state["monolith"]), Path(state["lib_dir"])
    header = _gguf_header_full(monolith)
    if header["arch"] != D6_ARCH:
        pytest.fail(f"{monolith}: general.architecture is {header['arch']!r}, not {D6_ARCH!r}")
    n_vocab = len(_gguf_field(monolith, "tokenizer.ggml.tokens") or [])
    request = _prompt_set(bdd_context, state["mode"], n_vocab)
    library = _server_greedy(bdd_context, tool, state["server"], ["--model-dir", str(lib_dir)], "d4a-library", request)
    mono = _server_greedy(bdd_context, tool, state["server"], ["-m", str(monolith)], "d4a-monolith", request)
    bdd_context.inkling_d4a_run = {
        "header": header,
        "dense": int(_gguf_field(monolith, DENSE_KEY) or 0),
        "library": library,
        "monolith": mono,
        "prompts": request["provenance"],
        "n_predict": request["n_predict"],
        "mode": state["mode"],
    }


def _d4a_run_or_fail(bdd_context: Any) -> dict:
    run = getattr(bdd_context, "inkling_d4a_run", None)
    if not run:
        pytest.fail("no --model-dir run recorded: the When must serve the library before this gate (fail closed)")
    return run


def _positions_equal(a: dict, b: dict) -> str | None:
    """Exact per-position equality of two greedy records; a report line or None."""
    pa, pb = _greedy_positions(a), _greedy_positions(b)
    if len(pa) != len(pb):
        return f"{len(pa)} vs {len(pb)} positions"
    for index, (x, y) in enumerate(zip(pa, pb)):
        if x["id"] != y["id"]:
            return f"position {index}: token {x['id']} vs {y['id']}"
        if x["logprob"] != y["logprob"]:
            return f"position {index}: logprob {x['logprob']!r} vs {y['logprob']!r}"
        if x["top_logprobs"] != y["top_logprobs"]:
            return f"position {index}: top_logprobs differ"
    return None


@then(parsers.parse("the served monolith equals the GGUF monolith"))
def then_d4a_equals_monolith(bdd_context: Any) -> None:
    run = _d4a_run_or_fail(bdd_context)
    failures: list[str] = []
    library_log, header = run["library"]["log"], run["header"]

    assembled = ASSEMBLED_LINE.search(library_log)
    if assembled is None:
        failures.append("the --model-dir server never reported 'assembled N parts ...' (no library assembly happened)")
    else:
        n_parts, block_count, n_dense, _, n_tensors = (int(value) for value in assembled.groups())
        if n_tensors != header["n_tensors"]:
            failures.append(f"assembled {n_tensors} tensors from {n_parts} parts, the monolith has {header['n_tensors']}")
        if n_dense != run["dense"]:
            failures.append(f"assembled dense block count {n_dense} != monolith {DENSE_KEY} {run['dense']}")
        n_layer = len([name for name in header["tensor_shapes"] if name.endswith(".attn_norm.weight")])
        if block_count != n_layer:
            failures.append(f"assembled block_count {block_count} != monolith block count {n_layer}")
    if re.search(rf"arch\s*=\s*{D6_ARCH}\b", library_log) is None:
        failures.append(f"the --model-dir server never reported 'arch = {D6_ARCH}'")
    if line := _d6_first_match(D6_LOAD_ERROR_PATTERNS, library_log):
        failures.append(f"--model-dir loader error: {line}")

    lib_records, mono_records = run["library"]["records"], run["monolith"]["records"]
    if len(lib_records) != len(mono_records) or not lib_records:
        failures.append(f"{len(lib_records)} library records vs {len(mono_records)} monolith records")
    else:
        for index, (a, b) in enumerate(zip(lib_records, mono_records)):
            if report := _positions_equal(a, b):
                failures.append(f"prompt {index}: library differs from monolith at {report}")
    if failures:
        pytest.fail("served library != GGUF monolith:\n" + "\n".join(f"  - {line}" for line in failures))
    _record_diagnostic(
        f"D4a: --model-dir greedy dump identical to the monolith over {len(lib_records)} prompts x "
        f"{run['n_predict']} tokens ({run['prompts']}, mode {run['mode']})"
    )


@then(parsers.parse("token IDs continue to equal the oracle"))
def then_d4a_tokens_vs_oracle(bdd_context: Any) -> None:
    run = _d4a_run_or_fail(bdd_context)
    lib_records, mono_records = run["library"]["records"], run["monolith"]["records"]
    if run["mode"] != "window" or _oracle_unmet(bdd_context) or not run["prompts"].startswith("D0 prompt set"):
        # no D0 oracle for the synthetic fixture (or no oracle bound): the
        # monolith of the same build is the comparand and the ids must match it
        for index, (a, b) in enumerate(zip(lib_records, mono_records)):
            if a["tokens"] != b["tokens"]:
                pytest.fail(f"prompt {index}: library ids {a['tokens']} != monolith ids {b['tokens']}")
        _record_diagnostic(
            "D4a: no D0 oracle applies "
            f"({'synthetic fixture' if run['mode'] == 'fixture' else _oracle_unmet(bdd_context)}); "
            "token ids gated against the same build's monolith run instead"
        )
        return
    oracle = bdd_context.inkling_oracle["dir"] / "greedy-64x8.json"
    bdd_context.inkling_greedy_oracle = _load_greedy_records(oracle.read_bytes(), str(oracle))
    bdd_context.inkling_greedy_run = lib_records
    _, mismatches, positions = _greedy_comparison(bdd_context)
    tally = f"{mismatches}/{positions} token positions differ from the oracle (library run)"
    oracle_features = _declared_features("INKLING_ORACLE_FEATURES")
    build_features = _declared_features("INKLING_BUILD_FEATURES")
    if oracle_features is not None and build_features is not None and oracle_features == build_features:
        if mismatches:
            pytest.fail(f"feature parity proven ({', '.join(sorted(build_features))}) but {tally}")
    else:
        _record_diagnostic(f"token-for-token equality NOT asserted as a gate (feature parity unproven); diagnostic: {tally}")
    if _envelope_unmet(bdd_context):
        _record_diagnostic(f"envelope not resolved ({_envelope_unmet(bdd_context)}); top-1 disagreement {100.0 * mismatches / positions:.2f} % recorded only")
        return
    entry = bdd_context.inkling_envelope["data"]["metrics"]["top1_disagreement"]
    _surface_band_findings("top1_disagreement", entry)
    if report := _envelope_fails(entry, 100.0 * mismatches / positions):
        pytest.fail(f"envelope metric top1_disagreement (library run): {report}")


# ---------------------------------------------------------------------------
# D4b — llama-stage-runner ring, head + tail
# ---------------------------------------------------------------------------


@given(parsers.parse("a stage ring with head and tail roles hosting Inkling stages"))
def given_d4b_ring(bdd_context: Any) -> None:
    state: dict[str, Any] = {"unmet": None, "model": None}
    runner, unmet = _find_binary("llama-stage-runner", "LLAMA_STAGE_RUNNER_BIN")
    check = _binary("LLAMA_EXPERT_CHECK_BIN", "llama-expert-check")
    state["runner"], state["check"] = runner, check
    if unmet:
        state["unmet"] = unmet
    elif check is None:
        state["unmet"] = "llama-expert-check binary not found (LLAMA_EXPERT_CHECK_BIN or build/bin); it is the monolith greedy reference"
    elif (deps := _slicer_deps_unmet()) is not None:
        state["unmet"] = deps
    override = os.environ.get("INKLING_STAGE_RING_MODEL")
    if override:
        if Path(override).is_file():
            state["model"] = Path(override)
        elif state["unmet"] is None:
            state["unmet"] = f"INKLING_STAGE_RING_MODEL {override} is not a readable file"
    bdd_context.inkling_d4b = state


def _wait_for_line(log_path: Path, process: subprocess.Popen, pattern: str, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        if re.search(pattern, text):
            return text
        if process.poll() is not None:
            pytest.fail(f"{log_path.name}: process exited {process.returncode} before {pattern!r}:\n{text[-3000:]}")
        time.sleep(0.1)
    pytest.fail(f"{log_path.name}: {pattern!r} not seen within {timeout:.0f} s:\n{text[-3000:]}")


def _start_stage(bdd_context: Any, argv: list[str], env: dict[str, str], name: str) -> tuple[subprocess.Popen, Path]:
    log_path = bdd_context.tmp_path / f"{name}.log"
    handle = log_path.open("w", encoding="utf-8")
    handle.write("$ " + shlex.join(argv) + "\n" + " ".join(f"{k}={v}" for k, v in sorted(env.items())) + "\n\n")
    handle.flush()
    bdd_context.inkling_launches = getattr(bdd_context, "inkling_launches", 0) + 1
    process = subprocess.Popen(
        argv, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
        env={**os.environ, **env, "CUDA_VISIBLE_DEVICES": "", "STAGE_THREADS": THREADS},
        cwd=bdd_context.tmp_path,
    )
    handle.close()
    bdd_context.processes.append(process)
    return process, log_path


def _read_hidden_blob(path: Path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    if len(raw) < 12:
        pytest.fail(f"{path}: truncated hidden blob ({len(raw)} bytes)")
    magic, n_rows, n_embd = struct.unpack("<3i", raw[:12])
    if magic != STAGE_MAGIC:
        pytest.fail(f"{path}: bad hidden-blob magic {magic:#x}")
    offset = 12 + 2 * 4 * n_rows
    data = raw[offset:]
    if len(data) != 4 * n_rows * n_embd:
        pytest.fail(f"{path}: {len(data)} data bytes for {n_rows} x {n_embd} floats")
    return n_rows, n_embd, data


def _read_dump(path: Path) -> tuple[list[int], bytes]:
    raw = path.read_bytes()
    if len(raw) < 32:
        pytest.fail(f"{path}: truncated STAGE_DUMP file ({len(raw)} bytes)")
    ne = list(struct.unpack("<4q", raw[:32]))
    return ne, raw[32:]


@when(parsers.parse("generation runs for at least 64 tokens through the ring"))
def when_d4b_ring(bdd_context: Any) -> None:
    state = getattr(bdd_context, "inkling_d4b", None)
    if state is None:
        pytest.skip("Prerequisite unmet: the D4b ring Given did not run")
    if state["unmet"]:
        pytest.skip(f"Prerequisite unmet: {state['unmet']}")
    model = state["model"] or generate_inkling_fixture(bdd_context, "inkling-d4b.gguf")["path"]
    header = _gguf_header_full(model)
    if header["arch"] != D6_ARCH:
        pytest.fail(f"{model}: general.architecture is {header['arch']!r}, not {D6_ARCH!r}")
    n_layer = int(_gguf_field(model, "inkling.block_count"))
    split = int(_gguf_field(model, DENSE_KEY) or 0)
    if not 0 < split < n_layer:
        pytest.fail(f"{model}: {DENSE_KEY} {split} of {n_layer} blocks leaves no head/tail split")
    lib_dir = _slice_library(model, bdd_context.tmp_path / "inkling-d4b-lib")
    prompt = os.environ.get("BDD_EXPERT_PROMPT", FIXTURE_PROMPTS[0])
    runner, check = str(state["runner"]), str(state["check"])
    tmp = bdd_context.tmp_path

    # monolith greedy reference: pure argmax, every id printed
    reference = _run(
        [check, "-m", str(model), "-p", prompt, "-n", str(RING_MIN_TOKENS), "-ngl", "0", "-t", THREADS]
        + shlex.split(os.environ.get("INKLING_STAGE_RING_CLIENT_ARGS", "")),
        {}, tmp, REQUEST_TIMEOUT_S, "d4b-monolith-check", bdd_context,
    )
    if reference["returncode"] != 0:
        pytest.fail(f"monolith llama-expert-check exited {reference['returncode']}:\n{reference['stderr'][-3000:]}")
    reference_ids = [int(token) for _, token, _ in TOK_LINE.findall(reference["stdout"])]

    # the ring: tail listens first, head dials in (docs/BRING-UP.md)
    port = _free_port()
    head_env = {"STAGE_EMIT": "hidden", "STAGE_ACTIVE": "1", "STAGE_IL_START": "0", "STAGE_IL_END": str(split)}
    tail_env = {"STAGE_ACTIVE": "1", "STAGE_IL_START": str(split), "STAGE_IL_END": str(n_layer),
                "STAGE_PRINT": "1", "STAGE_PRINT_IDS": "1", "STAGE_IGNORE_EOG": "1"}
    tail, tail_log = _start_stage(bdd_context, [
        runner, "--role", "tail", "--model-dir", str(lib_dir), "--layers", f"{split},{n_layer}",
        "--listen", str(port), "--n-ctx", str(RING_N_CTX), "--max-tokens", str(RING_MIN_TOKENS), "-ngl", "0",
    ], tail_env, "d4b-tail")
    _wait_for_line(tail_log, tail, r"listening on", STARTUP_TIMEOUT_S)
    head, head_log = _start_stage(bdd_context, [
        runner, "--role", "head", "--model-dir", str(lib_dir), "--layers", f"0,{split}",
        "--connect", f"127.0.0.1:{port}", "--prompt", prompt, "--max-tokens", str(RING_MIN_TOKENS),
        "--n-ctx", str(RING_N_CTX), "-ngl", "0",
    ], head_env, "d4b-head")
    try:
        head.wait(timeout=REQUEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        head.kill()
        pytest.fail(f"head stage did not finish within {REQUEST_TIMEOUT_S:.0f} s:\n{head_log.read_text(errors='replace')[-3000:]}")
    if head.returncode != 0:
        head_text = head_log.read_text(errors="replace")
        detail = _d6_first_match(D6_LOAD_ERROR_PATTERNS, head_text) or head_text[-1500:]
        pytest.fail(
            f"head stage [0,{split}) exited {head.returncode} (the ring never decoded): {detail}\n"
            f"--- head log tail ---\n{head_text[-2500:]}"
        )
    _wait_for_line(tail_log, tail, r"stage\[tail\]: decode", 60.0)
    tail.terminate()
    try:
        tail.wait(timeout=10)
    except subprocess.TimeoutExpired:
        tail.kill()
    ring_ids = [int(token) for slot, token in RING_ID_LINE.findall(tail_log.read_text(errors="replace")) if slot == "0"]

    # boundary hidden state: head window FILE run (emit) vs monolith FILE run (dump)
    head_out = tmp / "d4b-head-boundary.bin"
    head_file = _run(
        [runner, "--model-dir", str(lib_dir), "--layers", f"0,{split}", "--prompt", prompt, "--out", str(head_out), "-ngl", "0", "--n-ctx", str(RING_N_CTX)],
        head_env, tmp, REQUEST_TIMEOUT_S, "d4b-head-file", bdd_context,
    )
    dump_dir = tmp / "d4b-dump"
    dump_dir.mkdir(exist_ok=True)
    # The monolith runs in FILE mode like the head: STAGE_EMIT=hidden turns embeddings on
    # (run_tokens extracts every row through llama_get_embeddings_ith, which an embeddings-off
    # context refuses with "no embeddings"), and --out stands in for --last because an
    # embeddings context produces no logits for --last to read (it segfaults). The emitted
    # file is not compared; the STAGE_DUMP of l_out-<split-1> is what the boundary gate reads.
    mono_file = _run(
        [runner, "-m", str(model), "--prompt", prompt, "--out", str(tmp / "d4b-monolith-emit.bin"), "-ngl", "0", "--n-ctx", str(RING_N_CTX)],
        {"STAGE_EMIT": "hidden", "STAGE_DUMP": f"l_out-{split - 1}", "STAGE_DUMP_DIR": str(dump_dir)}, tmp, REQUEST_TIMEOUT_S, "d4b-monolith-file", bdd_context,
    )
    bdd_context.inkling_d4b_run = {
        "model": model, "split": split, "n_layer": n_layer, "prompt": prompt,
        "reference": reference, "reference_ids": reference_ids,
        "head_log": head_log.read_text(errors="replace"), "tail_log": tail_log.read_text(errors="replace"),
        "ring_ids": ring_ids, "head_file": head_file, "head_out": head_out,
        "mono_file": mono_file, "dump_path": dump_dir / f"l_out-{split - 1}.bin",
    }


def _d4b_run_or_fail(bdd_context: Any) -> dict:
    run = getattr(bdd_context, "inkling_d4b_run", None)
    if not run:
        pytest.fail("no ring run recorded: the When must run the ring before this gate (fail closed)")
    return run


@then(parsers.parse("head and tail token IDs equal the monolith run"))
def then_d4b_tokens(bdd_context: Any) -> None:
    run = _d4b_run_or_fail(bdd_context)
    failures: list[str] = []
    reference, ring = run["reference_ids"], run["ring_ids"]
    if len(reference) < RING_MIN_TOKENS:
        failures.append(f"monolith reference decoded {len(reference)} tokens, {RING_MIN_TOKENS} needed")
    if len(ring) < RING_MIN_TOKENS:
        failures.append(f"the ring decoded {len(ring)} tokens, {RING_MIN_TOKENS} needed (tail log tail):\n{run['tail_log'][-1500:]}")
    for log_name in ("head_log", "tail_log"):
        if line := _d6_first_match(D6_LOAD_ERROR_PATTERNS, run[log_name]):
            failures.append(f"{log_name}: {line}")
    if re.search(rf"IL=\[0,{run['split']}\)", run["head_log"]) is None:
        failures.append(f"head did not report IL=[0,{run['split']})")
    if re.search(rf"IL=\[{run['split']},{run['n_layer']}\)", run["tail_log"]) is None:
        failures.append(f"tail did not report IL=[{run['split']},{run['n_layer']})")
    n = min(len(reference), len(ring), RING_MIN_TOKENS)
    if n and ring[:n] != reference[:n]:
        first = next(index for index in range(n) if ring[index] != reference[index])
        failures.append(
            f"ring ids diverge from the monolith at token {first}: ring {ring[first:first + 6]} vs "
            f"monolith {reference[first:first + 6]} ({sum(1 for i in range(n) if ring[i] != reference[i])}/{n} differ)"
        )
    if failures:
        pytest.fail("ring != monolith (token ids):\n" + "\n".join(f"  - {line}" for line in failures))
    _record_diagnostic(f"D4b: {n} ring tokens identical to the monolith (split at block {run['split']})")


@then(parsers.parse("the boundary hidden state is byte-equal to the monolith run"))
def then_d4b_boundary(bdd_context: Any) -> None:
    run = _d4b_run_or_fail(bdd_context)
    if run["head_file"]["returncode"] != 0:
        pytest.fail(f"head-window FILE run exited {run['head_file']['returncode']}:\n{run['head_file']['stderr'][-3000:]}")
    if run["mono_file"]["returncode"] != 0:
        pytest.fail(f"monolith FILE run exited {run['mono_file']['returncode']}:\n{run['mono_file']['stderr'][-3000:]}")
    if not run["head_out"].is_file():
        pytest.fail(f"head-window run wrote no {run['head_out']}")
    if not run["dump_path"].is_file():
        pytest.fail(
            f"monolith run dumped no {run['dump_path'].name} (STAGE_DUMP=l_out-{run['split'] - 1}); "
            f"stderr tail:\n{run['mono_file']['stderr'][-1500:]}"
        )
    n_rows, n_embd, head_bytes = _read_hidden_blob(run["head_out"])
    ne, mono_bytes = _read_dump(run["dump_path"])
    if (ne[0], ne[1]) != (n_embd, n_rows):
        pytest.fail(f"monolith l_out-{run['split'] - 1} is {ne}, the head emitted {n_rows} rows x {n_embd}")
    if head_bytes != mono_bytes:
        import math

        head_f = struct.unpack(f"<{n_rows * n_embd}f", head_bytes)
        mono_f = struct.unpack(f"<{n_rows * n_embd}f", mono_bytes)
        max_abs = max(abs(a - b) for a, b in zip(head_f, mono_f))
        n_diff = sum(1 for a, b in zip(head_f, mono_f) if a != b)
        head_rms = math.sqrt(sum(a * a for a in head_f) / len(head_f))
        mono_rms = math.sqrt(sum(b * b for b in mono_f) / len(mono_f))
        pytest.fail(
            f"boundary hidden state differs from the monolith's l_out-{run['split'] - 1}: {n_diff}/{len(head_f)} "
            f"floats differ, max |delta| {max_abs:.6g}, RMS head {head_rms:.6g} vs monolith {mono_rms:.6g} "
            f"(a normalised-vs-residual mismatch reads as RMS ~1 vs the residual scale)"
        )
    _record_diagnostic(f"D4b: boundary hidden state byte-equal ({n_rows} rows x {n_embd})")
