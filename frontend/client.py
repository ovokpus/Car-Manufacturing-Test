"""Small HTTP client for the prediction API.

The UI process calls the API. The browser never needs the model, and it never
talks to port 8000 directly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ApiError(Exception):
    """The API was unreachable or rejected the call."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def get_json(url: str, timeout: float = 5) -> dict[str, Any]:
    return _request("GET", url, None, timeout)


def post_json(url: str, payload: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
    return _request("POST", url, payload, timeout)


def explain_failure(status: int, body: object) -> str:
    """Turn a FastAPI error body into one sentence for the result panel."""

    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            message = str(detail.get("message") or "The API rejected this configuration.")
            missing = detail.get("missing") or []
            if missing:
                preview = ", ".join(str(item) for item in list(missing)[:8])
                suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
                message = f"{message} Missing: {preview}{suffix}."
            return message
        if isinstance(detail, str) and detail:
            return detail
    return f"The API returned HTTP {status}."


def _request(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None
    headers = {"accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_error = json.loads(raw)
        except json.JSONDecodeError:
            parsed_error = {"detail": raw or exc.reason}
        raise ApiError(explain_failure(exc.code, parsed_error), status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"Could not reach the API at {url}. {exc.reason}") from exc
    if not isinstance(parsed, dict):
        raise ApiError(f"The API at {url} returned a {type(parsed).__name__}, not an object.")
    return parsed
