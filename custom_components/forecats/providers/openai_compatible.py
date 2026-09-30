"""OpenAI-compatible activity provider (Ollama, LM Studio, Unsloth, vLLM, etc.)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin

from ..models import GenerateRequest
from .base import SelectOption
from .prompts import build_activity_prompt

_LOGGER = logging.getLogger(__name__)


def list_models(
    base_url: str, api_key: str | None = None, timeout: int = 15
) -> list[SelectOption]:
    """Return models from GET /v1/models."""
    payload = _request_json(
        "GET",
        _api_url(base_url, "/v1/models"),
        api_key=api_key,
        timeout=timeout,
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    options: list[SelectOption] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("name")
        if not model_id or model_id in seen:
            continue
        seen.add(str(model_id))
        options.append({"value": str(model_id), "label": str(model_id)})
    return sorted(options, key=lambda option: option["label"].lower())


def _api_url(base_url: str, path: str) -> str:
    """Join base URL with an API path, tolerating a trailing /v1 on the base."""
    root = base_url.rstrip("/")
    if root.endswith("/v1") and path.startswith("/v1/"):
        path = path[3:]
    return urljoin(f"{root}/", path.lstrip("/"))


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> Any:
    data = None
    headers = _headers(api_key)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body.strip():
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible HTTP {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"OpenAI-compatible request failed: {err}") from err
    except json.JSONDecodeError as err:
        raise RuntimeError(f"OpenAI-compatible returned invalid JSON: {err}") from err


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts).strip()
    return ""


class OpenAICompatibleActivityProvider:
    """Generate activity text via OpenAI-compatible chat completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_seconds: int,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or None
        self._model = model
        self._timeout = timeout_seconds

    def generate_activity(
        self,
        data: GenerateRequest,
        prompt_history: list[str],
    ) -> str:
        prompt = build_activity_prompt(data, prompt_history)
        result = _request_json(
            "POST",
            _api_url(self._base_url, "/v1/chat/completions"),
            api_key=self._api_key,
            payload={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=self._timeout,
        )
        choices = result.get("choices") if isinstance(result, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                f"OpenAI-compatible chat returned no choices: {result}"
            )
        text = _message_text(choices[0].get("message") if isinstance(choices[0], dict) else None)
        if not text:
            raise RuntimeError(
                f"OpenAI-compatible chat returned no activity text: {result}"
            )
        return text.replace("\n", " ")
