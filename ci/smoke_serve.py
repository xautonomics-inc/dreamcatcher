"""One real chat completion; JSON evidence without response text or credentials."""

from __future__ import annotations

import argparse
import json
import math
import os
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
            },
        )
        seconds = time.monotonic() - request_started
        choices = reply.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SmokeError("missing_choices")
        choice = obj(choices[0])
        content = obj(choice.get("message")).get("content")
        if isinstance(content, list):
            pieces = [obj(item).get("text") for item in content]
            content = "".join(p for p in pieces if isinstance(p, str))
        if not isinstance(content, str) or not content.strip():
            raise SmokeError("no_completion_text")
        usage = obj(reply.get("usage"))
        tokens = usage.get("completion_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise SmokeError("positive_completion_token_usage_required")
        if seconds <= 0:
            raise SmokeError("invalid_elapsed_time")
        tps = tokens / seconds
        result.update(
            {
                "completion_tokens": tokens,
                "text_characters": len(content),
                "request_seconds": seconds,
                "tokens_per_second": tps,
                "min_tps": min_tps,
                "tps_basis": "completion_tokens / request_seconds",
                "includes_prefill_and_network": True,
                "ok": tps >= min_tps,
            }
        )
        if tps < min_tps:
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
    parser.add_argument("base_url")
    parser.add_argument("--min-tps", type=float, default=0)
    args = parser.parse_args()
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
