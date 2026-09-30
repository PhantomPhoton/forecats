"""Unsloth Studio activity and image providers."""

from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from urllib.parse import urljoin

from PIL import Image

from ..models import GenerateRequest
from .base import SelectOption
from .prompts import build_image_prompt

_LOGGER = logging.getLogger(__name__)

# FLUX.2 reference workflow: 1 primary (init_image) + up to 3 extras = 4 total.
_MAX_EXTRA_REFERENCE_IMAGES = 3
_LOAD_POLL_SECONDS = 2.0

# Mirrors Studio Images picker task filter (IMAGE_GEN_TASKS).
_IMAGE_GEN_TASKS = frozenset(
    {
        "text-to-image",
        "image-to-image",
        "image-text-to-image",
    }
)
_UNSUPPORTED_DIFFUSION_TASK = "image-diffusion-unsupported"

# Repo-id :: gguf-filename separator for config values that need a quant.
_MODEL_SPEC_SEP = "::"

# Curated Studio Images catalog entries (recommended tab), so the HA dropdown
# matches the desktop picker even when a model is not yet in the local cache.
_CURATED_IMAGE_MODELS: tuple[str, ...] = (
    "unsloth/Z-Image-Turbo",
    "unsloth/Z-Image",
    "unsloth/Qwen-Image-2.1",
    "unsloth/Qwen-Image-2512",
    "unsloth/Qwen-Image",
    "unsloth/FLUX.1-schnell",
    "unsloth/FLUX.1-dev",
    "black-forest-labs/FLUX.1-Krea-dev",
    "unsloth/FLUX.2-klein-4B",
    "unsloth/FLUX.2-klein-9B",
    "unsloth/Qwen-Image-Edit-2511",
    "unsloth/FLUX.1-Kontext-dev",
    "krea/Krea-2-Turbo",
    "Alpha-VLLM/Lumina-Image-2.0",
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    "HiDream-ai/HiDream-I1-Full",
    "ideogram-ai/ideogram-4",
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0",
)


def list_chat_models(
    base_url: str, api_key: str | None = None, timeout: int = 15
) -> list[SelectOption]:
    """Return chat models from hub inventory + OpenAI /v1/models."""
    options: list[SelectOption] = []
    seen: set[str] = set()

    def add(value: str, label: str | None = None) -> None:
        if not value or value in seen:
            return
        seen.add(value)
        options.append({"value": value, "label": label or value})

    for repo_id, label, meta in _iter_hub_models(base_url, api_key, timeout):
        if not _is_chat_candidate(repo_id, meta):
            continue
        add(repo_id, label)

    try:
        payload = _request_json(
            "GET",
            f"{base_url.rstrip('/')}/v1/models",
            api_key=api_key,
            timeout=timeout,
        )
    except RuntimeError as err:
        _LOGGER.debug("Unsloth /v1/models unavailable: %s", err)
        payload = {}

    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id") or item.get("name")
            if model_id:
                add(str(model_id))

    return sorted(options, key=lambda option: option["label"].lower())


def list_image_models(
    base_url: str, api_key: str | None = None, timeout: int = 15
) -> list[SelectOption]:
    """Return image models from Studio hub inventory + curated catalog.

    Matches the desktop Images model picker sources: on-device cache/local
    inventory (filtered to text-to-image tasks) plus Studio's recommended list.
    """
    options: list[SelectOption] = []
    seen: set[str] = set()
    loaded_repo: str | None = None

    def add(value: str, label: str | None = None) -> None:
        if not value or value in seen:
            return
        seen.add(value)
        options.append({"value": value, "label": label or value})

    try:
        status = _request_json(
            "GET",
            f"{base_url.rstrip('/')}/api/inference/images/status",
            api_key=api_key,
            timeout=timeout,
        )
    except RuntimeError as err:
        _LOGGER.debug("Unsloth image status unavailable: %s", err)
        status = {}

    if isinstance(status, dict) and status.get("loaded"):
        loaded_repo = (
            status.get("repo_id")
            or status.get("model")
            or status.get("model_path")
            or status.get("loaded_model")
        )
        if loaded_repo:
            filename = status.get("gguf_filename")
            value = _encode_model_spec(str(loaded_repo), filename)
            label = str(loaded_repo)
            if filename:
                label = f"{loaded_repo} [{filename}]"
            add(value, f"{label} (loaded)")

    for repo_id, label, meta in _iter_hub_models(base_url, api_key, timeout):
        if not _is_image_candidate(repo_id, meta):
            continue
        add(repo_id, label)

    for repo_id in _CURATED_IMAGE_MODELS:
        add(repo_id)

    return sorted(options, key=lambda option: option["label"].lower())


def _encode_model_spec(repo_id: str, filename: str | None = None) -> str:
    if filename:
        return f"{repo_id}{_MODEL_SPEC_SEP}{filename}"
    return repo_id


def _parse_model_spec(spec: str) -> tuple[str, str | None]:
    if _MODEL_SPEC_SEP in spec:
        repo_id, filename = spec.split(_MODEL_SPEC_SEP, 1)
        return repo_id.strip(), filename.strip() or None
    return spec.strip(), None


def _looks_like_gguf_repo(repo_id: str) -> bool:
    return bool(repo_id) and "-GGUF" in repo_id.upper()


def _task_of(meta: dict[str, Any]) -> str | None:
    for key in ("task", "pipeline_tag"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_image_candidate(repo_id: str, meta: dict[str, Any]) -> bool:
    if meta.get("companion") or meta.get("companion_prefetch") or meta.get("partial"):
        return False
    task = _task_of(meta)
    if task == _UNSUPPORTED_DIFFUSION_TASK:
        return False
    if task in _IMAGE_GEN_TASKS:
        return True
    # Catalog / cache rows sometimes omit task; keep obvious diffusion repos.
    lowered = repo_id.lower()
    needles = (
        "flux",
        "qwen-image",
        "sdxl",
        "stable-diffusion",
        "z-image",
        "lumina-image",
        "hidream",
        "ideogram",
        "hunyuanimage",
        "krea",
    )
    return any(n in lowered for n in needles)


def _is_chat_candidate(repo_id: str, meta: dict[str, Any]) -> bool:
    if meta.get("companion") or meta.get("companion_prefetch") or meta.get("partial"):
        return False
    task = _task_of(meta)
    if task in _IMAGE_GEN_TASKS or task == _UNSUPPORTED_DIFFUSION_TASK:
        return False
    if task in {
        "text-to-video",
        "image-to-video",
        "image-text-to-video",
        "text-to-speech",
        "automatic-speech-recognition",
    }:
        return False
    caps = meta.get("capabilities")
    if isinstance(caps, dict) and caps.get("can_chat") is False:
        return False
    if meta.get("diffusers") or meta.get("single_file"):
        return False
    return True


def _iter_hub_models(
    base_url: str, api_key: str | None, timeout: int
) -> list[tuple[str, str, dict[str, Any]]]:
    """Collect (repo_or_load_id, label, meta) from Studio /api/hub inventory."""
    root = base_url.rstrip("/")
    rows: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()

    def take(repo_id: str | None, meta: dict[str, Any], label: str | None = None) -> None:
        if not repo_id or repo_id in seen:
            return
        seen.add(repo_id)
        rows.append((repo_id, label or repo_id, meta))

    # /api/hub/local — models_dir, HF cache, LM Studio, Ollama, custom folders
    try:
        payload = _request_json(
            "GET", f"{root}/api/hub/local", api_key=api_key, timeout=timeout
        )
        models = payload.get("models") if isinstance(payload, dict) else None
        if isinstance(models, list):
            for item in models:
                if not isinstance(item, dict):
                    continue
                repo_id = (
                    item.get("load_id")
                    or item.get("model_id")
                    or item.get("id")
                    or item.get("display_name")
                )
                label = item.get("display_name") or item.get("model_id") or repo_id
                take(str(repo_id) if repo_id else None, item, str(label) if label else None)
    except RuntimeError as err:
        _LOGGER.debug("Unsloth /api/hub/local unavailable: %s", err)

    for path in ("/api/hub/cached-models", "/api/hub/cached-gguf"):
        try:
            payload = _request_json(
                "GET", f"{root}{path}", api_key=api_key, timeout=timeout
            )
            cached = payload.get("cached") if isinstance(payload, dict) else None
            if not isinstance(cached, list):
                continue
            for item in cached:
                if not isinstance(item, dict):
                    continue
                if path.endswith("cached-gguf"):
                    item = {**item, "model_format": item.get("model_format") or "gguf"}
                repo_id = item.get("load_id") or item.get("repo_id")
                take(str(repo_id) if repo_id else None, item)
        except RuntimeError as err:
            _LOGGER.debug("Unsloth %s unavailable: %s", path, err)

    return rows


def _resolve_gguf_filename(
    base_url: str,
    api_key: str | None,
    repo_id: str,
    *,
    preferred: str | None = None,
    timeout: int = 30,
) -> str | None:
    """Pick a GGUF filename for load, mirroring Studio resolveDiffusionGgufFilename."""
    if preferred and preferred.lower().endswith(".gguf"):
        return preferred

    try:
        params = urllib.parse.urlencode(
            {"repo_id": repo_id, "prefer_local_cache": "true"}
        )
        payload = _request_json(
            "GET",
            f"{base_url.rstrip('/')}/api/hub/gguf-variants?{params}",
            api_key=api_key,
            timeout=timeout,
        )
    except RuntimeError as err:
        _LOGGER.debug("Unsloth gguf-variants resolve failed for %s: %s", repo_id, err)
        return preferred

    variants = payload.get("variants") if isinstance(payload, dict) else None
    if not isinstance(variants, list) or not variants:
        return preferred

    default = payload.get("default_variant") if isinstance(payload, dict) else None
    downloaded = [
        v
        for v in variants
        if isinstance(v, dict) and v.get("downloaded") and v.get("filename")
    ]
    pool = downloaded or [v for v in variants if isinstance(v, dict) and v.get("filename")]

    if preferred:
        for variant in pool:
            quant = str(variant.get("quant") or "")
            filename = str(variant.get("filename") or "")
            if preferred in {quant, filename} or preferred.lower() in filename.lower():
                return filename

    if isinstance(default, str) and default:
        for variant in pool:
            quant = str(variant.get("quant") or "")
            filename = str(variant.get("filename") or "")
            if default in {quant, filename}:
                return filename

    if pool:
        return str(pool[0]["filename"])
    return preferred


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
        raise RuntimeError(f"Unsloth HTTP {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Unsloth request failed: {err}") from err
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Unsloth returned invalid JSON: {err}") from err


def _request_bytes(
    url: str,
    *,
    api_key: str | None = None,
    timeout: int = 120,
) -> bytes:
    request = urllib.request.Request(url, headers=_headers(api_key), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Unsloth HTTP {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Unsloth request failed: {err}") from err


def _pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class UnslothActivityProvider:
    """Generate activity text via Unsloth's OpenAI-compatible chat API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_seconds: int,
    ) -> None:
        from .openai_compatible import OpenAICompatibleActivityProvider

        self._inner = OpenAICompatibleActivityProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )

    def generate_activity(
        self,
        data: GenerateRequest,
        prompt_history: list[str],
    ) -> str:
        return self._inner.generate_activity(data, prompt_history)


class UnslothImageProvider:
    """Generate images via Unsloth Studio diffusion API (WebUI-compatible)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_seconds: int,
        *,
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or None
        self._model = model
        self._timeout = timeout_seconds
        self._width = width
        self._height = height
        self._seed = seed
        self._steps = steps
        self._guidance = guidance

    def generate_image(
        self,
        data: GenerateRequest,
        activity: str,
        input_images: dict[str, Image.Image],
        art_style: str,
    ) -> Image.Image:
        prompt = build_image_prompt(
            data, activity, list(input_images.keys()), art_style
        )
        self._ensure_image_model_loaded()

        refs = [_pil_to_data_url(image) for image in input_images.values()]
        body: dict[str, Any] = {
            "prompt": prompt,
            "width": self._width,
            "height": self._height,
            "steps": self._steps,
            "guidance": self._guidance,
            "seed": self._effective_seed(),
            "batch_size": 1,
        }
        if refs:
            body["init_image"] = refs[0]
            extras = refs[1 : 1 + _MAX_EXTRA_REFERENCE_IMAGES]
            if extras:
                body["reference_images"] = extras
            if len(refs) > 1 + _MAX_EXTRA_REFERENCE_IMAGES:
                _LOGGER.warning(
                    "Unsloth FLUX.2 reference supports 4 images max "
                    "(1 primary + 3 refs); using the first 4 of %s",
                    len(refs),
                )
            body["workflow"] = "reference"

        result = _request_json(
            "POST",
            f"{self._base_url}/api/inference/images/generate",
            api_key=self._api_key,
            payload=body,
            timeout=self._timeout,
        )
        images = result.get("images") if isinstance(result, dict) else None
        if not isinstance(images, list) or not images:
            raise RuntimeError(f"Unsloth image generate returned no images: {result}")

        first = images[0]
        if not isinstance(first, dict):
            raise RuntimeError(f"Unsloth image generate returned invalid image: {first}")

        if isinstance(first.get("b64_json"), str) and first["b64_json"]:
            raw = base64.b64decode(first["b64_json"])
            return Image.open(io.BytesIO(raw)).copy()

        url = first.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError(f"Unsloth image missing url/b64_json: {first}")
        absolute = urljoin(f"{self._base_url}/", url.lstrip("/"))
        raw = _request_bytes(
            absolute, api_key=self._api_key, timeout=min(120, self._timeout)
        )
        return Image.open(io.BytesIO(raw)).copy()

    def _effective_seed(self) -> int:
        if self._seed and self._seed > 0:
            return self._seed
        return int(time.time()) % 2_147_483_647

    def _ensure_image_model_loaded(self) -> None:
        repo_id, filename = _parse_model_spec(self._model) if self._model else ("", None)

        status = _request_json(
            "GET",
            f"{self._base_url}/api/inference/images/status",
            api_key=self._api_key,
            timeout=min(30, self._timeout),
        )
        if isinstance(status, dict) and status.get("loaded"):
            loaded_repo = status.get("repo_id") or status.get("model_path")
            # Already loaded and either no preferred model, or it matches.
            if not repo_id or not loaded_repo or str(loaded_repo) == repo_id:
                if not filename or status.get("gguf_filename") in {None, filename}:
                    return

        if not repo_id:
            raise RuntimeError(
                "Unsloth has no diffusion model loaded. Load one in Studio's "
                "Images tab, or set unsloth_image_model so Forecats can load it."
            )

        is_gguf = bool(filename) or _looks_like_gguf_repo(repo_id)
        if is_gguf:
            filename = _resolve_gguf_filename(
                self._base_url,
                self._api_key,
                repo_id,
                preferred=filename,
                timeout=min(30, self._timeout),
            )
            if not filename:
                raise RuntimeError(
                    f"Unsloth GGUF model {repo_id!r} needs a quantization "
                    "filename; pick one in Studio or set "
                    f"repo::file.gguf (e.g. {repo_id}::Q4_K_M.gguf)."
                )
            model_kind = "gguf"
        else:
            model_kind = "pipeline"

        payload: dict[str, Any] = {
            "model_path": repo_id,
            "model_kind": model_kind,
        }
        if filename:
            payload["gguf_filename"] = filename

        _LOGGER.info(
            "Loading Unsloth diffusion model %s kind=%s filename=%s",
            repo_id,
            model_kind,
            filename,
        )
        _request_json(
            "POST",
            f"{self._base_url}/api/inference/images/load",
            api_key=self._api_key,
            payload=payload,
            timeout=min(120, self._timeout),
        )
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            status = _request_json(
                "GET",
                f"{self._base_url}/api/inference/images/status",
                api_key=self._api_key,
                timeout=min(30, self._timeout),
            )
            if isinstance(status, dict) and status.get("loaded"):
                return
            if isinstance(status, dict) and status.get("error"):
                raise RuntimeError(f"Unsloth model load failed: {status.get('error')}")
            progress = None
            try:
                progress = _request_json(
                    "GET",
                    f"{self._base_url}/api/inference/images/load-progress",
                    api_key=self._api_key,
                    timeout=min(30, self._timeout),
                )
            except RuntimeError:
                pass
            if isinstance(progress, dict) and progress.get("error"):
                raise RuntimeError(
                    f"Unsloth model load failed: {progress.get('error')}"
                )
            time.sleep(_LOAD_POLL_SECONDS)
        raise RuntimeError(
            f"Unsloth model load timed out after {self._timeout}s ({self._model!r})."
        )
