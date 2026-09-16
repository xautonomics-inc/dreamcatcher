"""BDD step definitions for the Inkling feature — D2 oracle parity.

Scope: the two @d2 parity scenarios and the @fail-closed scenario of
features/inkling.feature. The other lanes (D1, D3, D4a/b/c, D6) bind when
their lanes implement them (see test_inkling.py).

Binding contract (feature header):
- Paths resolve from the environment — INKLING_MODEL_PATH, INKLING_ORACLE_DIR,
  plus INKLING_GREEDY_CMD / INKLING_PPL_CMD / INKLING_PPL_TEXT for the
  runners — and fail closed when unset, missing or hash-mismatched.
- Fail closed means the scenario stops before touching a model or starting a
  service. Givens only record resolution state; every runner When re-checks
  it and pytest.skip()s with the precise reason before launching anything.

Oracle artifacts (sha256 pinned in the feature header) and asserted values:
- greedy-64x8.json: per-prompt greedy token IDs with per-position logprob and
  top_logprobs (n_probs: 10).
- kld-base-4x2048.bin + ppl.log: four per-chunk perplexity values
  94.3665, 83.6386, 78.6291, 72.6183 — the final estimate IS chunk 4.
The oracle's own error bar (± 5.49154) is NOT a tolerance.

Tolerances (deliberately tight; widen only with a recorded decision):
- Perplexity: the oracle log prints four decimals, so parity means equal to
  the last printed digit (PPL_TOLERANCE).
- Log-probabilities: LOGPROB_TOLERANCE absolute — same-build, same-feature-
  set parity should sit far below it.

Runner contract (until the D2 binary lands these env vars are unbound and
the scenarios skip):
- INKLING_GREEDY_CMD: shell-free command template with a {model} placeholder;
  stdout must be the greedy dump in the oracle's schema (see
  _greedy_positions for the required record shape).
- INKLING_PPL_CMD: command template with {model} and {text} placeholders;
  stdout must carry llama.cpp-style "[i]value" per-chunk entries and a
  "Final estimate: PPL = X +/- Y" line.
- Both run with CUDA_VISIBLE_DEVICES forced empty: the oracle binary was
  CUDA-built but run CPU-only, and its CPU feature set (AVX2, FMA, LLAMAFILE,
  REPACK) is part of the environment a parity run must reproduce. The -t 20
  thread count is the command's own responsibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when

PPL_TOLERANCE = 5e-5
LOGPROB_TOLERANCE = 1e-4
N_PROBS = 10

ORACLE_FILES = {
    "greedy-64x8.json": "1b5e5c4bff5b98cbe91345e5df290a804667e25ff728fbbba78b3d08cef923f0",
    "kld-base-4x2048.bin": "aeab11429a2567db2b36f210754a444ce97f6a9fa851bf0389b236129838b505",
    "ppl.log": "98f7349cb50c92500b5cd79171e0fcf34453a79abec7c7fccfd881f1f4ab4edc",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_oracle_dir(bdd_context) -> None:
    """Resolve INKLING_ORACLE_DIR and hash-verify the pinned artifacts.

    Records {"dir": Path | None, "unmet": str | None} on the context. Never
    touches a model or a service; the runner Whens do the fail-closed skip.
    """
    state: dict = {"dir": None, "unmet": None}
    raw = os.environ.get("INKLING_ORACLE_DIR")
    if not raw:
        state["unmet"] = "INKLING_ORACLE_DIR is not bound"
    elif not (directory := Path(raw)).is_dir():
        state["unmet"] = f"INKLING_ORACLE_DIR {raw} does not exist"
    else:
        for name, expected in ORACLE_FILES.items():
            path = directory / name
            if not path.is_file():
                state["unmet"] = f"INKLING_ORACLE_DIR is missing {name}"
                break
            if (actual := _sha256(path)) != expected:
                state["unmet"] = f"{name} failed hash verification: {actual}"
                break
        else:
            state["dir"] = directory
    bdd_context.inkling_oracle = state


def _oracle_unmet(bdd_context) -> str | None:
    if not hasattr(bdd_context, "inkling_oracle"):
        _resolve_oracle_dir(bdd_context)
    return bdd_context.inkling_oracle["unmet"]


def _greedy_positions(record: dict) -> list[dict]:
    """Extract per-position entries from one greedy dump record.

    Required record shape (greedy-64x8.json): {"prompt": str, "tokens": [
    {"id": int, "logprob": float, "top_logprobs": [float, ... <= 10]}, ...]}.
    A mismatch fails loudly with the schema so the adaptation happens here
    and nowhere else when the real dump lands.
    """
    try:
        positions = record["tokens"]
        return [
            {
                "id": int(entry["id"]),
                "logprob": float(entry["logprob"]),
                "top_logprobs": [float(value) for value in entry["top_logprobs"]],
            }
            for entry in positions
        ]
    except (KeyError, TypeError, ValueError) as error:
        pytest.fail(
            "greedy dump record does not match the documented schema "
            f'({{"prompt": str, "tokens": [{{"id", "logprob", "top_logprobs"}}]}}): '
            f"{error!r}"
        )


def _load_greedy_records(payload: bytes, source: str) -> list[dict]:
    data = json.loads(payload)
    records = data.get("prompts", data) if isinstance(data, dict) else data
    if not isinstance(records, list) or not records:
        pytest.fail(f"{source}: expected a non-empty list of prompt records")
    return records


def _require(bdd_context, command_env: str) -> tuple[Path, Path, str]:
    """Fail-closed gate shared by the runner Whens, before any launch."""
    if unmet := _oracle_unmet(bdd_context):
        pytest.skip(f"Prerequisite unmet: {unmet}")
    model = getattr(bdd_context, "inkling_model_path", None)
    if model is None:
        pytest.skip("Prerequisite unmet: INKLING_MODEL_PATH is not bound to a readable file")
    command = os.environ.get(command_env)
    if not command:
        pytest.skip(f"Prerequisite unmet: {command_env} is not bound (D2 runner not wired)")
    bdd_context.inkling_launches = getattr(bdd_context, "inkling_launches", 0) + 1
    return bdd_context.inkling_oracle["dir"], model, command


def _run_command(command: str, **placeholders: str) -> str:
    argv = [part.format(**placeholders) for part in shlex.split(command)]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=3600, env=environment, check=False
    )
    if completed.returncode != 0:
        pytest.fail(
            f"{argv[0]} exited {completed.returncode}; stderr tail:\n{completed.stderr[-2000:]}"
        )
    return completed.stdout


@given(parsers.parse("Inkling-Small GGUF metadata for the model named by INKLING_MODEL_PATH"))
def given_inkling_model_path(bdd_context):
    """Record the model path; the runner Whens gate on it."""
    raw = os.environ.get("INKLING_MODEL_PATH")
    path = Path(raw) if raw else None
    bdd_context.inkling_model_path = path if path and path.is_file() else None


@given(
    parsers.parse("a D0 oracle artifact directory named by INKLING_ORACLE_DIR with recorded hashes")
)
def given_inkling_oracle_dir(bdd_context):
    """Resolve and hash-verify the D0 artifacts; recording only, no skip."""
    _resolve_oracle_dir(bdd_context)


@given(
    parsers.parse(
        'the D0 oracle dump "{name}" exists in the INKLING_ORACLE_DIR directory'
    )
)
def given_inkling_dump_exists(bdd_context, name):
    """Record the dump path; the runner Whens gate on it."""
    state = {"path": None, "unmet": _oracle_unmet(bdd_context)}
    if not state["unmet"]:
        path = bdd_context.inkling_oracle["dir"] / name
        state["path"] = path if path.is_file() else None
        if state["path"] is None:
            state["unmet"] = f"oracle dump {name} is not a readable file"
    bdd_context.inkling_dump = state


@when(parsers.parse("the model runs greedy generation on CPU for the D0 fixed prompt set"))
def when_greedy_run(bdd_context):
    directory, model, command = _require(bdd_context, "INKLING_GREEDY_CMD")
    oracle_records = _load_greedy_records(
        (directory / "greedy-64x8.json").read_bytes(), "oracle greedy-64x8.json"
    )
    stdout = _run_command(command, model=str(model))
    bdd_context.inkling_greedy_oracle = oracle_records
    bdd_context.inkling_greedy_run = _load_greedy_records(stdout.encode(), "greedy runner stdout")


@then(parsers.parse("every generated token ID equals the oracle token IDs"))
def then_greedy_tokens_equal(bdd_context):
    oracle_records = bdd_context.inkling_greedy_oracle
    run_records = bdd_context.inkling_greedy_run
    if len(run_records) != len(oracle_records):
        pytest.fail(
            f"greedy run produced {len(run_records)} prompt records, "
            f"oracle has {len(oracle_records)}"
        )
    for index, (oracle, run) in enumerate(zip(oracle_records, run_records)):
        oracle_positions = _greedy_positions(oracle)
        run_positions = _greedy_positions(run)
        if len(run_positions) != len(oracle_positions):
            pytest.fail(
                f"prompt {index}: run generated {len(run_positions)} positions, "
                f"oracle has {len(oracle_positions)}"
            )
        for position, (expected, actual) in enumerate(zip(oracle_positions, run_positions)):
            if expected["id"] != actual["id"]:
                pytest.fail(
                    f"prompt {index} position {position}: token ID {actual['id']} "
                    f"!= oracle {expected['id']} (first divergence)"
                )


@then(
    parsers.parse(
        "the per-position log-probabilities (logprob, top_logprobs) match the oracle "
        "within tolerance"
    )
)
def then_greedy_logprobs_within_tolerance(bdd_context):
    oracle_records = bdd_context.inkling_greedy_oracle
    run_records = bdd_context.inkling_greedy_run
    for index, (oracle, run) in enumerate(zip(oracle_records, run_records)):
        for position, (expected, actual) in enumerate(
            zip(_greedy_positions(oracle), _greedy_positions(run))
        ):
            if abs(expected["logprob"] - actual["logprob"]) > LOGPROB_TOLERANCE:
                pytest.fail(
                    f"prompt {index} position {position}: logprob {actual['logprob']!r} "
                    f"!= oracle {expected['logprob']!r} (tolerance {LOGPROB_TOLERANCE})"
                )
            if len(actual["top_logprobs"]) != len(expected["top_logprobs"]):
                pytest.fail(
                    f"prompt {index} position {position}: {len(actual['top_logprobs'])} "
                    f"top_logprobs != oracle {len(expected['top_logprobs'])} "
                    f"(n_probs: {N_PROBS})"
                )
            for rank, (want, got) in enumerate(
                zip(expected["top_logprobs"], actual["top_logprobs"])
            ):
                if abs(want - got) > LOGPROB_TOLERANCE:
                    pytest.fail(
                        f"prompt {index} position {position} rank {rank}: "
                        f"top_logprob {got!r} != oracle {want!r} "
                        f"(tolerance {LOGPROB_TOLERANCE})"
                    )


@when(parsers.parse("the model computes perplexity on CPU over the D0 4-chunk split"))
def when_ppl_run(bdd_context):
    directory, model, command = _require(bdd_context, "INKLING_PPL_CMD")
    text = os.environ.get("INKLING_PPL_TEXT")
    if not text or not Path(text).is_file():
        pytest.skip("Prerequisite unmet: INKLING_PPL_TEXT is not bound to a readable file")
    stdout = _run_command(command, model=str(model), text=text)
    chunks = [
        float(value)
        for _, value in sorted(
            (int(index), float(value))
            for index, value in re.findall(r"\[(\d+)\]([0-9]+\.[0-9]+)", stdout)
        )
    ]
    final = re.search(r"Final estimate: PPL = ([0-9.]+) (?:\+/-|±) ([0-9.]+)", stdout)
    if len(chunks) != 4 or final is None:
        pytest.fail(
            f"perplexity run did not yield 4 chunks + final estimate "
            f"(got {len(chunks)} chunks, final={'yes' if final else 'no'})"
        )
    bdd_context.inkling_ppl = {
        "chunks": chunks,
        "final": float(final.group(1)),
        "error_bar": float(final.group(2)),
    }


@then(
    parsers.parse(
        "the four per-chunk perplexity values equal the oracle "
        "{first:g}, {second:g}, {third:g}, {fourth:g}"
    )
)
def then_ppl_chunks_equal(bdd_context, first: float, second: float, third: float, fourth: float):
    expected = [first, second, third, fourth]
    for index, (want, got) in enumerate(zip(expected, bdd_context.inkling_ppl["chunks"]), start=1):
        if abs(want - got) > PPL_TOLERANCE:
            pytest.fail(
                f"chunk {index}: perplexity {got} != oracle {want} "
                f"(tolerance {PPL_TOLERANCE})"
            )


@then(parsers.parse("the final chunk equals the recorded overall perplexity {final:g}"))
def then_ppl_final_chunk_equals(bdd_context, final: float):
    ppl = bdd_context.inkling_ppl
    if abs(ppl["chunks"][-1] - final) > PPL_TOLERANCE:
        pytest.fail(
            f"final chunk {ppl['chunks'][-1]} != recorded overall perplexity {final}"
        )
    if abs(ppl["final"] - final) > PPL_TOLERANCE:
        pytest.fail(
            f"run final estimate {ppl['final']} != recorded overall perplexity {final}"
        )


@given(parsers.parse("that directory does not exist or fails hash verification"))
def given_fail_closed_bad_oracle_dir(bdd_context):
    """The fail-closed scenario: only provable on a host without valid artifacts."""
    if not _oracle_unmet(bdd_context):
        pytest.skip(
            "Prerequisite unmet: INKLING_ORACLE_DIR is present and hash-valid; "
            "the fail-closed path is not exercised on this host"
        )


@when(parsers.parse("the scenario attempts to bind expected values"))
def when_fail_closed_bind(bdd_context):
    _resolve_oracle_dir(bdd_context)
    if not bdd_context.inkling_oracle["unmet"]:
        pytest.fail("binding succeeded on an invalid oracle directory; fail-closed is broken")


@then(parsers.parse("the scenario aborts before loading any model or starting any service"))
def then_fail_closed_aborted(bdd_context):
    assert bdd_context.inkling_oracle["unmet"], "resolution must record the unmet prerequisite"
    launches = getattr(bdd_context, "inkling_launches", 0)
    assert launches == 0, f"{launches} model/service launch(es) happened before the abort"
