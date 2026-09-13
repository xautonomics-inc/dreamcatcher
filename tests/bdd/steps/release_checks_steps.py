"""Executable steps for the repository release-check interfaces."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import requests
from pytest_bdd import given, parsers, then, when

REPO_ROOT = Path(__file__).resolve().parents[3]


def _result(context: Any) -> dict[str, object]:
    assert context.last_stdout.strip(), "command produced no JSON output"
    lines = context.last_stdout.strip().splitlines()
    assert len(lines) == 1, f"expected one JSON line, got {len(lines)}"
    value = json.loads(lines[0])
    assert isinstance(value, dict)
    return value


def _replace_command_placeholders(context: Any, command: str) -> str:
    replacements = {
        "http://<host>:<port>": getattr(
            context, "release_server_url", f"http://{context.host}:{context.port}"
        )
    }
    optional_paths = {
        "<coherent_text_file>": "coherent_text_file",
        "<degenerate_text_file>": "degenerate_text_file",
        "<fixture_checkout>": "fixture_checkout",
        "<build_dir>": "build_dir",
    }
    for placeholder, attribute in optional_paths.items():
        if hasattr(context, attribute):
            replacements[placeholder] = str(getattr(context, attribute))
    for placeholder, value in replacements.items():
        command = command.replace(placeholder, value)
    return command


@given(parsers.parse('an owned "llama-server" endpoint is available at "{url}"'))
def owned_llama_server(bdd_context: Any, url: str) -> None:
    configured = os.environ.get("BDD_SERVER_URL")
    if not configured:
        pytest.skip("Prerequisite unmet: BDD_SERVER_URL is not bound to an owned llama-server")
    bdd_context.release_server_url = configured.rstrip("/")
    try:
        response = requests.get(
            f"{bdd_context.release_server_url}/v1/models", timeout=5.0
        )
    except requests.RequestException as exc:
        pytest.skip(f"Prerequisite unmet: owned llama-server is unreachable: {exc}")
    if response.status_code != 200:
        pytest.skip(
            f"Prerequisite unmet: owned llama-server returned {response.status_code}"
        )


@given("model discovery returns exactly one model")
def exactly_one_model(bdd_context: Any) -> None:
    response = requests.get(
        f"{bdd_context.release_server_url}/v1/models", timeout=5.0
    )
    data = response.json().get("data")
    assert isinstance(data, list) and len(data) == 1


@given(
    'an owned endpoint returns empty answer text and non-empty "reasoning_content"'
)
def reasoning_only_endpoint(bdd_context: Any) -> None:
    configured = os.environ.get("BDD_REASONING_ONLY_URL")
    if not configured:
        pytest.skip(
            "Prerequisite unmet: BDD_REASONING_ONLY_URL is not bound to an owned fixture endpoint"
        )
    bdd_context.release_server_url = configured.rstrip("/")


@given(parsers.parse('"<coherent_text_file>" contains at least {count:d} distinct words'))
def coherent_text_file(bdd_context: Any, count: int) -> None:
    path = bdd_context.tmp_path / "coherent.txt"
    path.write_text(
        "A small bird crossed the river and returned safely before sunset.\n",
        encoding="utf-8",
    )
    assert len(set(path.read_text(encoding="utf-8").lower().split())) >= count
    bdd_context.coherent_text_file = path


@given(parsers.parse('at least {percent:d} percent of its words are distinct'))
def coherent_distinct_ratio(bdd_context: Any, percent: int) -> None:
    words = bdd_context.coherent_text_file.read_text(encoding="utf-8").lower().split()
    assert 100 * len(set(words)) / len(words) >= percent


@given(
    parsers.parse(
        'at least {percent:d} percent of its characters are alphanumeric or whitespace'
    )
)
def coherent_character_ratio(bdd_context: Any, percent: int) -> None:
    text = bdd_context.coherent_text_file.read_text(encoding="utf-8")
    friendly = sum(character.isalnum() or character.isspace() for character in text)
    assert 100 * friendly / len(text) >= percent


@given(parsers.parse('"<degenerate_text_file>" contains "{text}"'))
def degenerate_text_file(bdd_context: Any, text: str) -> None:
    path = bdd_context.tmp_path / "degenerate.txt"
    path.write_text(text, encoding="utf-8")
    bdd_context.degenerate_text_file = path


@given(
    '"<fixture_checkout>/SHA256SUMS" names a model artifact and its SHA-256 digest'
)
def checksum_fixture(bdd_context: Any) -> None:
    checkout = bdd_context.tmp_path / "checksum-checkout"
    scripts = checkout / "scripts"
    models = checkout / "models"
    scripts.mkdir(parents=True)
    models.mkdir()
    shutil.copy2(REPO_ROOT / "scripts" / "verify-checksum-models.py", scripts)
    artifact = models / "model.gguf"
    artifact.write_bytes(b"dreamcatcher-bdd-model-fixture")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (checkout / "SHA256SUMS").write_text(
        f"{digest}  models/model.gguf\n", encoding="utf-8"
    )
    bdd_context.fixture_checkout = checkout
    bdd_context.checksum_artifact = artifact


@given("the named artifact exists under \"<fixture_checkout>\" with matching bytes")
def checksum_artifact_matches(bdd_context: Any) -> None:
    assert bdd_context.checksum_artifact.is_file()


@given('"<fixture_checkout>/SHA256SUMS" does not exist')
def missing_checksum_fixture(bdd_context: Any) -> None:
    checkout = bdd_context.tmp_path / "checksum-missing"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "scripts" / "verify-checksum-models.py", scripts)
    bdd_context.fixture_checkout = checkout


@given(parsers.parse('"test-stage-manifest" has been built under "<build_dir>"'))
def build_stage_manifest_test(bdd_context: Any) -> None:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("Prerequisite unmet: no C++ compiler is available")
    build_dir = bdd_context.tmp_path / "build"
    binary = build_dir / "bin" / "test-stage-manifest"
    binary.parent.mkdir(parents=True)
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-I{REPO_ROOT / 'vendor'}",
            str(REPO_ROOT / "tests" / "test-stage-manifest.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    bdd_context.build_dir = build_dir


@when(parsers.parse('I run "{command}"'))
def run_release_command(bdd_context: Any, command: str) -> None:
    resolved = _replace_command_placeholders(bdd_context, command)
    completed = subprocess.run(
        shlex.split(resolved),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    bdd_context.last_returncode = completed.returncode
    bdd_context.last_stdout = completed.stdout
    bdd_context.last_stderr = completed.stderr


@then("the script should request one non-streaming 32-token chat completion")
def smoke_request_contract(bdd_context: Any) -> None:
    source = (REPO_ROOT / "ci" / "smoke_serve.py").read_text(encoding="utf-8")
    assert '"max_tokens": 32' in source
    assert '"stream": False' in source


@then("the response should contain non-empty answer text")
def nonempty_smoke_answer(bdd_context: Any) -> None:
    text = _result(bdd_context).get("text")
    assert isinstance(text, str) and text.strip()


@then('"usage.completion_tokens" should be a positive integer')
def positive_completion_tokens(bdd_context: Any) -> None:
    tokens = _result(bdd_context).get("completion_tokens")
    assert isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0


@then("the completion should pass the coherence screen")
def smoke_coherence_passes(bdd_context: Any) -> None:
    assert "error" not in _result(bdd_context)


@then(parsers.parse("measured completion tokens per request second should be at least {minimum:g}"))
def minimum_measured_tps(bdd_context: Any, minimum: float) -> None:
    measured = _result(bdd_context).get("tokens_per_second")
    assert isinstance(measured, (int, float)) and measured >= minimum


@then(parsers.parse('the script should emit one JSON result with "ok" set to {expected}'))
def json_result_ok(bdd_context: Any, expected: str) -> None:
    assert _result(bdd_context).get("ok") is (expected == "true")


@then(parsers.parse('the script should emit one JSON result with "mode" set to "{mode}"'))
def json_result_mode(bdd_context: Any, mode: str) -> None:
    assert _result(bdd_context).get("mode") == mode


@then(parsers.parse('the result should have "ok" set to {expected}'))
def result_ok(bdd_context: Any, expected: str) -> None:
    assert _result(bdd_context).get("ok") is (expected == "true")


@then(parsers.parse('the result error should be "{error}"'))
def result_error(bdd_context: Any, error: str) -> None:
    assert _result(bdd_context).get("error") == error


@then("no HTTP request should be sent")
def check_text_mode_has_no_endpoint(bdd_context: Any) -> None:
    assert _result(bdd_context).get("mode") == "check_text"


@then("the process exit code should be 0")
def process_exit_zero(bdd_context: Any) -> None:
    assert bdd_context.last_returncode == 0, bdd_context.last_stderr


@then("the process exit code should be non-zero")
def process_exit_nonzero_release(bdd_context: Any) -> None:
    assert bdd_context.last_returncode not in (None, 0)


@then('the artifact row should contain "V" in the valid checksum column')
def checksum_row_valid(bdd_context: Any) -> None:
    line = next(
        line for line in bdd_context.last_stdout.splitlines() if "models/model.gguf" in line
    )
    assert "V" in line


@then("the artifact row should have an empty missing-file column")
def checksum_row_not_missing(bdd_context: Any) -> None:
    line = next(
        line for line in bdd_context.last_stdout.splitlines() if "models/model.gguf" in line
    )
    assert not line.rstrip().endswith("X")


@then("the diagnostic should identify the missing SHA256SUMS path")
def missing_checksum_diagnostic(bdd_context: Any) -> None:
    assert "SHA256SUMS" in bdd_context.last_stderr


def _manifest_fixture_source() -> str:
    return (REPO_ROOT / "tests" / "test-stage-manifest.cpp").read_text(encoding="utf-8")


@then('its fixtures should accept a positive 32-bit integer at "source.block_count"')
def manifest_accepts_positive_source_count(bdd_context: Any) -> None:
    assert bdd_context.last_returncode == 0
    assert '"source":{"block_count":43}' in _manifest_fixture_source()


@then('its fixtures should ignore decoy block counts outside "source.block_count"')
def manifest_ignores_decoys(bdd_context: Any) -> None:
    assert '"files":[{"window_kv":{"block_count":1}}]' in _manifest_fixture_source()


@then(
    "its fixtures should reject missing, non-integer, non-positive, overflowing, and trailing values"
)
def manifest_rejects_invalid_values(bdd_context: Any) -> None:
    source = _manifest_fixture_source()
    for fragment in ('"block_count":0', '"block_count":43.0', '"block_count":"43"', '2147483648', "trailing"):
        assert fragment in source


@then(parsers.parse('stdout should contain "{text}"'))
def stdout_contains(bdd_context: Any, text: str) -> None:
    assert text in bdd_context.last_stdout
