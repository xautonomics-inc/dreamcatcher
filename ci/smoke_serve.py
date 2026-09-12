"""One real chat completion; JSON evidence with the generated text, never credentials."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import cast


class SmokeError(ValueError):
    """A bounded, public-safe diagnostic code."""


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SmokeError("expected_json_object")
    return cast(dict[str, object], value)


def request(
    url: str, key: str, timeout: float, payload: dict[str, object] | None = None
) -> dict[str, object]:
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )

    # Do not forward bearer credentials to a redirect destination.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(
            self,
            req: urllib.request.Request,
            fp: object,
            code: int,
            msg: str,
            headers: object,
            newurl: str,
        ) -> None:
            return None

    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(req, timeout=timeout) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise SmokeError("response_too_large")
    try:
        value: object = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise SmokeError("invalid_json") from exc
    return obj(value)


def api_base(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SmokeError("invalid_base_url")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SmokeError("base_url_must_not_contain_credentials_query_or_fragment")
    base = value.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


# Coherence thresholds. A server that loads but decodes garbage still answers with
# 200 + non-empty content at a healthy token rate, so tok/s alone passes on output
# like "**:** **:**;**;**". These are deliberately loose: they reject degenerate
# loops, not terse-but-real prose.
MIN_DISTINCT_WORDS = 5      # (a) distinct alphanumeric words required
MIN_DISTINCT_RATIO = 0.4    # (a) distinct / total words
MAX_SYMBOL_REPEATS = 6      # (b) consecutive repeats of one non-alphanumeric unit
MIN_ALNUM_SPACE_RATIO = 0.6 # (c) alphanumeric-or-space share of all characters

# One non-alphanumeric unit (punctuation, and any whitespace glue inside it)
# repeated MAX_SYMBOL_REPEATS+1 times back to back: ";;;;;;;" or "**:** **:** ...".
SYMBOL_RUN = re.compile(r"([^0-9A-Za-z]{1,8})\1{%d,}" % MAX_SYMBOL_REPEATS)
WORD = re.compile(r"[0-9A-Za-z']+")


def coherence(text: str) -> dict[str, object]:
    """Degeneracy screen for generated text. Reports an "error" key when it fails."""
    stripped = text.strip()
    words = WORD.findall(stripped)
    distinct = len({word.lower() for word in words})
    distinct_ratio = distinct / len(words) if words else 0.0
    friendly = sum(1 for ch in stripped if ch.isalnum() or ch.isspace())
    alnum_space_ratio = friendly / len(stripped) if stripped else 0.0
    run = SYMBOL_RUN.search(stripped)
    report: dict[str, object] = {
        "word_count": len(words),
        "distinct_words": distinct,
        "distinct_ratio": distinct_ratio,
        "alnum_space_ratio": alnum_space_ratio,
        "min_distinct_words": MIN_DISTINCT_WORDS,
        "min_distinct_ratio": MIN_DISTINCT_RATIO,
        "max_symbol_repeats": MAX_SYMBOL_REPEATS,
        "min_alnum_space_ratio": MIN_ALNUM_SPACE_RATIO,
        "repeated_symbol_run": run.group(1) if run else "",
    }
    # Most-diagnostic failure first: a symbol loop explains itself, a word count
    # does not.
    if not stripped:
        report["error"] = "degenerate_empty_text"
    elif run is not None:
        report["error"] = "degenerate_repeated_symbol_run"
    elif alnum_space_ratio < MIN_ALNUM_SPACE_RATIO:
        report["error"] = "degenerate_low_alphanumeric_ratio"
    elif distinct < MIN_DISTINCT_WORDS:
        report["error"] = "degenerate_too_few_distinct_words"
    elif distinct_ratio < MIN_DISTINCT_RATIO:
        report["error"] = "degenerate_low_distinct_ratio"
    return report


def check_text(text: str) -> dict[str, object]:
    """Run only the coherence screen, over text from a file or stdin."""
    started = time.monotonic()
    result: dict[str, object] = {
        "schema_version": 1,
        "started_at": timestamp(),
        "ok": False,
        "mode": "check_text",
        "text": text,
        "text_characters": len(text),
    }
    report = coherence(text)
    result.update(report)
    result["ok"] = "error" not in report
    result.update(
        {"completed_at": timestamp(), "elapsed_seconds": time.monotonic() - started}
    )
    return result


def smoke(
    base: str, min_tps: float, model: str, key: str, timeout: float
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "started_at": timestamp(),
        "ok": False,
    }
    started = time.monotonic()
    try:
        if not math.isfinite(min_tps) or min_tps < 0:
            raise SmokeError("min_tps_must_be_finite_and_nonnegative")
        if not math.isfinite(timeout) or timeout <= 0:
            raise SmokeError("timeout_must_be_finite_and_positive")
        base = api_base(base)
        if not model:
            listing = request(base + "/models", key, timeout).get("data")
            if not isinstance(listing, list) or len(listing) != 1:
                raise SmokeError("set_SMOKE_MODEL_when_model_list_is_not_unique")
            candidate = obj(listing[0]).get("id")
            if not isinstance(candidate, str) or not candidate:
                raise SmokeError("missing_model_id")
            model = candidate
        request_started = time.monotonic()
        reply = request(
            base + "/chat/completions",
            key,
            timeout,
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Tell a short story about a bird."}
                ],
                "max_tokens": 32,
                "stream": False,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        seconds = time.monotonic() - request_started
        choices = reply.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SmokeError("missing_choices")
        choice = obj(choices[0])
        message = obj(choice.get("message"))
        content = message.get("content")
        if isinstance(content, list):
            pieces = [obj(item).get("text") for item in content]
            content = "".join(p for p in pieces if isinstance(p, str))
        if not isinstance(content, str) or not content.strip():
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                raise SmokeError("reasoning_only")
            raise SmokeError("no_completion_text")
        usage = obj(reply.get("usage"))
        tokens = usage.get("completion_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise SmokeError("positive_completion_token_usage_required")
        if seconds <= 0:
            raise SmokeError("invalid_elapsed_time")
        tps = tokens / seconds
        report = coherence(content)
        result.update(report)
        result.update(
            {
                "completion_tokens": tokens,
                "text": content,
                "text_characters": len(content),
                "request_seconds": seconds,
                "tokens_per_second": tps,
                "min_tps": min_tps,
                "tps_basis": "completion_tokens / request_seconds",
                "includes_prefill_and_network": True,
                "ok": tps >= min_tps and "error" not in report,
            }
        )
        if "error" not in report and tps < min_tps:
            result["error"] = "below_min_tps"
    except urllib.error.HTTPError as exc:
        result.update({"error": "http_error", "http_status": exc.code})
    except SmokeError as exc:
        result["error"] = str(exc)
    except (OSError, urllib.error.URLError, TimeoutError):
        result["error"] = "transport_error_or_timeout"
    finally:
        result.update(
            {"completed_at": timestamp(), "elapsed_seconds": time.monotonic() - started}
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?")
    parser.add_argument("--min-tps", type=float, default=0)
    parser.add_argument(
        "--check-text",
        metavar="FILE",
        help="screen a saved completion for degeneracy instead of calling a server"
        " (FILE may be - for stdin); no request is sent",
    )
    args = parser.parse_args()
    if args.check_text is not None:
        if args.check_text == "-":
            text = sys.stdin.read()
        else:
            with open(args.check_text, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        result = check_text(text)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["ok"] else 1
    if not args.base_url:
        parser.error("base_url is required unless --check-text is given")
    try:
        timeout = float(os.environ.get("SMOKE_TIMEOUT", "60"))
    except ValueError:
        timeout = float("nan")
    result = smoke(
        args.base_url,
        args.min_tps,
        os.environ.get("SMOKE_MODEL", ""),
        os.environ.get("SMOKE_API_KEY", ""),
        timeout,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
