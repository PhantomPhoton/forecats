"""InvokeAI activity and image providers (CPU-friendly defaults)."""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from ..models import GenerateRequest
from .base import SelectOption
from .prompts import build_activity_prompt, build_image_prompt

_LOGGER = logging.getLogger(__name__)

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "workflows" / "invokeai_reference_image.json"

ACTIVITY_MODEL_TYPES = ["text_llm"]
IMAGE_MODEL_TYPES = ["main"]
VAE_MODEL_TYPES = ["vae"]
# Companion text/vision encoders used across Flux, Flux2, SD3, Z-Image, etc.
ENCODER_MODEL_TYPES = [
    "t5_encoder",
    "clip_embed",
    "clip_vision",
    "qwen3_encoder",
    "qwen_vl_encoder",
    "qwen3_vl_encoder",
    "mistral_encoder",
    "gemma2_encoder",
    "wan_t5_encoder",
    "siglip",
]


def list_models(
    base_url: str,
    api_token: str | None = None,
    *,
    model_types: list[str] | None = None,
    timeout: int = 15,
) -> list[SelectOption]:
    """Return InvokeAI models from GET /api/v2/models/, optionally by type."""
    client = InvokeAIClient(base_url, api_token, timeout_seconds=timeout)
    options: list[SelectOption] = []
    seen: set[str] = set()
    types_to_query: list[str | None] = list(model_types) if model_types else [None]

    for index, model_type in enumerate(types_to_query):
        path = "/api/v2/models/"
        if model_type:
            path = f"{path}?model_type={model_type}"
        try:
            payload = client.get_json(path)
        except RuntimeError as err:
            # Host/timeout failures: stop immediately so config discovery does not
            # retry every model type with a full timeout each.
            if index == 0 and not options:
                raise
            _LOGGER.warning(
                "InvokeAI model list failed for type=%s: %s", model_type, err
            )
            continue
        for model in _extract_models(payload):
            value, label = _model_option(model)
            if not value:
                continue
            if value in seen and label not in seen:
                value = label
            if value in seen:
                continue
            seen.add(value)
            options.append({"value": value, "label": label})

    return sorted(options, key=lambda option: option["label"].lower())


def list_activity_models(
    base_url: str, api_token: str | None = None, timeout: int = 15
) -> list[SelectOption]:
    """Return InvokeAI Text LLM models for activity generation."""
    return list_models(
        base_url, api_token, model_types=ACTIVITY_MODEL_TYPES, timeout=timeout
    )


def list_image_models(
    base_url: str, api_token: str | None = None, timeout: int = 15
) -> list[SelectOption]:
    """Return InvokeAI main image models."""
    return list_models(
        base_url, api_token, model_types=IMAGE_MODEL_TYPES, timeout=timeout
    )


def _extract_models(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        models = payload.get("models")
        if isinstance(models, list):
            return [model for model in models if isinstance(model, dict)]
    if isinstance(payload, list):
        return [model for model in payload if isinstance(model, dict)]
    return []


def _model_label(model: dict[str, Any]) -> str:
    """Human-readable label: name plus base/type when present."""
    name = model.get("name") or model.get("model_name") or model.get("key") or ""
    base = model.get("base") or ""
    mtype = model.get("type") or ""
    label = str(name)
    extras = [str(part) for part in (base, mtype) if part]
    if extras:
        label = f"{name} ({', '.join(extras)})"
    return label


def _model_option(model: dict[str, Any]) -> tuple[str, str]:
    """Return (display value, label) using the model name, not the UUID key."""
    label = _model_label(model)
    name = str(model.get("name") or model.get("model_name") or "")
    value = name or label
    return value, label


def resolve_model(
    client: "InvokeAIClient",
    configured: str,
    *,
    model_types: list[str] | None = None,
) -> dict[str, Any] | None:
    """Map a configured name/label/key to a full InvokeAI model record."""
    if not configured:
        return None

    types_to_query: list[str | None] = list(model_types) if model_types else [None]
    for model_type in types_to_query:
        path = "/api/v2/models/"
        if model_type:
            path = f"{path}?model_type={model_type}"
        try:
            payload = client.get_json(path)
        except RuntimeError as err:
            _LOGGER.warning(
                "InvokeAI model resolve failed for type=%s: %s", model_type, err
            )
            continue
        for model in _extract_models(payload):
            key = str(model.get("key") or model.get("id") or "")
            name = str(model.get("name") or model.get("model_name") or "")
            label = _model_label(model)
            if configured in {key, name, label}:
                return model
    return None


def resolve_model_key(
    client: "InvokeAIClient",
    configured: str,
    *,
    model_types: list[str] | None = None,
) -> str:
    """Map a configured name/label/key to the InvokeAI model key UUID."""
    model = resolve_model(client, configured, model_types=model_types)
    if model:
        return str(model.get("key") or model.get("id") or configured)
    return configured


def _model_identifier(model: dict[str, Any] | None, fallback_key: str | None = None) -> dict[str, str]:
    """Build an InvokeAI ModelIdentifierField from a model record."""
    if model:
        ident: dict[str, str] = {}
        key = str(model.get("key") or model.get("id") or "")
        if key:
            ident["key"] = key
        for field in ("hash", "name", "base", "type"):
            value = model.get(field)
            if value is not None and value != "":
                ident[field] = str(value)
        if "name" not in ident:
            name = model.get("model_name") or model.get("name")
            if name:
                ident["name"] = str(name)
        if ident:
            return ident
    if fallback_key:
        return {"key": fallback_key}
    raise RuntimeError("Cannot build InvokeAI model identifier without a model key.")


def _list_model_records(
    client: "InvokeAIClient",
    model_types: list[str],
) -> list[dict[str, Any]]:
    """Fetch model records for the given types."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model_type in model_types:
        try:
            payload = client.get_json(f"/api/v2/models/?model_type={model_type}")
        except RuntimeError as err:
            _LOGGER.warning(
                "InvokeAI companion list failed for type=%s: %s", model_type, err
            )
            continue
        for model in _extract_models(payload):
            key = str(model.get("key") or model.get("id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            records.append(model)
    return records


def _pick_model(
    models: list[dict[str, Any]],
    *,
    configured: str | None = None,
    model_type: str | None = None,
    base: str | None = None,
    name_contains: str | None = None,
    exclude_bases: set[str] | None = None,
) -> dict[str, Any] | None:
    """Pick a model by configured name/key, type, base, or name substring."""
    typed = [
        model
        for model in models
        if model_type is None or str(model.get("type") or "") == model_type
    ]
    if base is not None:
        typed = [
            model for model in typed if str(model.get("base") or "").lower() == base
        ]
    if exclude_bases:
        excluded = {item.lower() for item in exclude_bases}
        typed = [
            model
            for model in typed
            if str(model.get("base") or "").lower() not in excluded
        ]
    if configured:
        for model in typed:
            key = str(model.get("key") or model.get("id") or "")
            name = str(model.get("name") or model.get("model_name") or "")
            label = _model_label(model)
            if configured in {key, name, label}:
                return model
        # Configured value may exist but was filtered out by base; still try unfiltered
        # only when no base constraint matched, so callers can detect a mismatch.
        if base is not None or exclude_bases:
            return None
    if name_contains:
        needle = name_contains.lower()
        for model in typed:
            name = str(model.get("name") or model.get("model_name") or "").lower()
            if needle in name:
                return model
    return typed[0] if typed else None


def _pick_vae_for_base(
    companions: list[dict[str, Any]],
    *,
    main_base: str,
) -> dict[str, Any] | None:
    """Pick a VAE compatible with the main model family (flux vs flux2)."""
    family = "flux2" if main_base == "flux2" else "flux"
    # Prefer an exact-base VAE; for FLUX.1 also accept base=any and classic ae names.
    vae = _pick_model(companions, model_type="vae", base=family)
    if vae:
        return vae
    if family == "flux":
        return (
            _pick_model(
                companions,
                model_type="vae",
                name_contains="schnell_ae",
                exclude_bases={"flux2"},
            )
            or _pick_model(
                companions,
                model_type="vae",
                name_contains="flux.1",
                exclude_bases={"flux2"},
            )
            or _pick_model(companions, model_type="vae", exclude_bases={"flux2"})
        )
    return _pick_model(companions, model_type="vae", name_contains="flux.2")


def _edge(source_id: str, source_field: str, dest_id: str, dest_field: str) -> dict:
    return {
        "source": {"node_id": source_id, "field": source_field},
        "destination": {"node_id": dest_id, "field": dest_field},
    }


def _request(
    method: str,
    url: str,
    *,
    timeout: int,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    content_type: str | None = None,
) -> Any:
    req_headers = dict(headers or {})
    if content_type:
        req_headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            if not body:
                return None
            content_type_header = response.headers.get("Content-Type", "")
            if "application/json" in content_type_header or body[:1] in (b"{", b"["):
                return json.loads(body.decode("utf-8"))
            return body
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"InvokeAI HTTP {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"InvokeAI request failed: {err}") from err


class InvokeAIClient:
    """Thin HTTP client for InvokeAI."""

    def __init__(
        self,
        base_url: str,
        api_token: str | None,
        timeout_seconds: int,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._headers: dict[str, str] = {}
        if api_token:
            self._headers["Authorization"] = f"Bearer {api_token}"

    def post_json(self, path: str, payload: dict) -> Any:
        return _request(
            "POST",
            f"{self._base_url}{path}",
            timeout=self._timeout,
            headers=self._headers,
            data=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )

    def get_json(self, path: str) -> Any:
        return _request(
            "GET",
            f"{self._base_url}{path}",
            timeout=min(60, self._timeout),
            headers=self._headers,
        )

    def get_bytes(self, path: str) -> bytes:
        result = _request(
            "GET",
            f"{self._base_url}{path}",
            timeout=min(120, self._timeout),
            headers=self._headers,
        )
        if not isinstance(result, (bytes, bytearray)):
            raise RuntimeError("InvokeAI expected binary image response.")
        return bytes(result)

    def upload_image(self, name: str, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        boundary = "----ForecatsBoundary7MA4YWxkTrZu0gW"
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'.encode()
        )
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(buffer.read())
        body.extend(f"\r\n--{boundary}--\r\n".encode())

        result = _request(
            "POST",
            f"{self._base_url}/api/v1/images/upload?image_category=user&is_intermediate=true",
            timeout=min(120, self._timeout),
            headers=self._headers,
            data=bytes(body),
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        if not isinstance(result, dict) or "image_name" not in result:
            raise RuntimeError(f"InvokeAI image upload failed: {result}")
        return result["image_name"]

    def wait_for_queue_item(self, item_id: int | str) -> dict:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            item = self.get_json(f"/api/v1/queue/default/i/{item_id}")
            status = item.get("status")
            if status == "completed":
                return item
            if status == "failed":
                raise RuntimeError(
                    f"InvokeAI queue item failed: {item.get('error_message') or item}"
                )
            if status == "canceled":
                raise RuntimeError("InvokeAI queue item was canceled.")
            time.sleep(2)
        raise RuntimeError(
            f"InvokeAI job timed out after {self._timeout} seconds (item {item_id})."
        )


class InvokeAIActivityProvider:
    """Generate activity text via InvokeAI Text LLM expand-prompt API."""

    def __init__(
        self,
        base_url: str,
        api_token: str | None,
        model: str,
        timeout_seconds: int,
    ) -> None:
        self._client = InvokeAIClient(base_url, api_token, timeout_seconds)
        self._model = model

    def generate_activity(
        self,
        data: GenerateRequest,
        prompt_history: list[str],
    ) -> str:
        prompt = build_activity_prompt(data, prompt_history)
        model_key = resolve_model_key(
            self._client, self._model, model_types=ACTIVITY_MODEL_TYPES
        )
        result = self._client.post_json(
            "/api/v1/utilities/expand-prompt",
            {
                "prompt": prompt,
                "model_key": model_key,
                "max_tokens": 300,
            },
        )
        # API may return string or object with expanded prompt
        if isinstance(result, str):
            text = result.strip()
        elif isinstance(result, dict):
            text = (
                result.get("expanded_prompt")
                or result.get("prompt")
                or result.get("response")
                or ""
            ).strip()
        else:
            text = ""
        if not text:
            # Fallback: enqueue a Text LLM workflow graph
            text = self._generate_via_text_llm_graph(prompt, model_key)
        return text.replace("\n", " ")

    def _generate_via_text_llm_graph(self, prompt: str, model_key: str) -> str:
        graph = {
            "id": "forecats-activity",
            "nodes": {
                "text-llm": {
                    "id": "text-llm",
                    "type": "text_llm",
                    "prompt": prompt,
                    "model": {"key": model_key},
                    "max_new_tokens": 300,
                    "is_intermediate": False,
                    "use_cache": False,
                }
            },
            "edges": [],
        }
        enqueue = self._client.post_json(
            "/api/v1/queue/default/enqueue_batch",
            {"batch": {"graph": graph, "runs": 1, "origin": "forecats"}},
        )
        item_ids = enqueue.get("item_ids") or []
        if not item_ids:
            raise RuntimeError(f"InvokeAI enqueue returned no item ids: {enqueue}")
        item = self._client.wait_for_queue_item(item_ids[0])
        results = (item.get("session") or {}).get("results") or {}
        for value in results.values():
            if isinstance(value, dict):
                for key in ("text", "response", "value", "output"):
                    if isinstance(value.get(key), str) and value[key].strip():
                        return value[key].strip()
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise RuntimeError(f"InvokeAI Text LLM returned no text: {results}")


class InvokeAIImageProvider:
    """Generate images via a checked-in InvokeAI graph template with reference uploads."""

    def __init__(
        self,
        base_url: str,
        api_token: str | None,
        model: str,
        timeout_seconds: int,
        *,
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance: float,
    ) -> None:
        self._client = InvokeAIClient(base_url, api_token, timeout_seconds)
        self._model = model
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

        uploaded_names: list[str] = []
        for name, image in input_images.items():
            uploaded_names.append(self._client.upload_image(name, image))

        main_model = resolve_model(
            self._client, self._model, model_types=IMAGE_MODEL_TYPES
        )
        if not main_model:
            raise RuntimeError(
                f"InvokeAI main image model not found: {self._model!r}"
            )

        graph = self._build_graph(prompt, uploaded_names, main_model)
        enqueue = self._client.post_json(
            "/api/v1/queue/default/enqueue_batch",
            {"batch": {"graph": graph, "runs": 1, "origin": "forecats"}},
        )
        item_ids = enqueue.get("item_ids") or []
        if not item_ids:
            raise RuntimeError(f"InvokeAI enqueue returned no item ids: {enqueue}")

        item = self._client.wait_for_queue_item(item_ids[0])
        image_name = self._extract_image_name(item)
        image_bytes = self._client.get_bytes(f"/api/v1/images/i/{image_name}/full")
        return Image.open(io.BytesIO(image_bytes)).copy()

    def _effective_seed(self) -> int:
        if self._seed and self._seed > 0:
            return self._seed
        return int(time.time()) % 2_147_483_647

    def _build_graph(
        self,
        prompt: str,
        uploaded_names: list[str],
        main_model: dict[str, Any],
    ) -> dict:
        """Build an executable Flux / Flux Kontext generation graph."""
        base = str(main_model.get("base") or "flux").lower()
        name = str(main_model.get("name") or main_model.get("model_name") or "")
        is_kontext = "kontext" in name.lower()
        seed = self._effective_seed()

        if base == "flux2":
            graph = self._build_flux2_graph(
                prompt, uploaded_names, main_model, seed=seed
            )
        else:
            graph = self._build_flux_graph(
                prompt,
                uploaded_names,
                main_model,
                seed=seed,
                with_kontext=is_kontext or bool(uploaded_names),
            )

        # Optional checked-in template can still document the shape; generation
        # always uses the programmatic denoise graph above.
        if WORKFLOW_PATH.exists():
            try:
                notes = json.loads(WORKFLOW_PATH.read_text()).get("notes")
                if notes:
                    graph["notes"] = notes
            except (OSError, json.JSONDecodeError):
                pass
        return graph

    def _build_flux_graph(
        self,
        prompt: str,
        uploaded_names: list[str],
        main_model: dict[str, Any],
        *,
        seed: int,
        with_kontext: bool,
    ) -> dict:
        companions = _list_model_records(
            self._client, [*VAE_MODEL_TYPES, *ENCODER_MODEL_TYPES]
        )
        t5 = _pick_model(companions, model_type="t5_encoder")
        clip = _pick_model(companions, model_type="clip_embed")
        main_base = str(main_model.get("base") or "flux").lower()
        vae = _pick_vae_for_base(
            companions, main_base=main_base
        )

        loader_fields: dict[str, Any] = {
            "id": "flux_model_loader",
            "type": "flux_model_loader",
            "model": _model_identifier(main_model),
            "is_intermediate": True,
            "use_cache": False,
        }
        # Omit companions when missing so self-contained pipelines can resolve
        # them from the main model; otherwise require them for standard Flux.
        if t5:
            loader_fields["t5_encoder_model"] = _model_identifier(t5)
        if clip:
            loader_fields["clip_embed_model"] = _model_identifier(clip)
        if vae:
            loader_fields["vae_model"] = _model_identifier(vae)
        if not (t5 and clip and vae):
            _LOGGER.warning(
                "InvokeAI Flux companions incomplete (t5=%s clip=%s vae=%s); "
                "relying on model-loader defaults",
                bool(t5),
                bool(clip),
                bool(vae),
            )

        nodes: dict[str, Any] = {
            "positive_prompt": {
                "id": "positive_prompt",
                "type": "string",
                "value": prompt,
                "is_intermediate": True,
                "use_cache": False,
            },
            "seed": {
                "id": "seed",
                "type": "integer",
                "value": seed,
                "is_intermediate": True,
                "use_cache": False,
            },
            "flux_model_loader": loader_fields,
            "flux_text_encoder": {
                "id": "flux_text_encoder",
                "type": "flux_text_encoder",
                "is_intermediate": True,
                "use_cache": False,
            },
            "pos_cond_collect": {
                "id": "pos_cond_collect",
                "type": "collect",
                "is_intermediate": True,
                "use_cache": False,
            },
            "flux_denoise": {
                "id": "flux_denoise",
                "type": "flux_denoise",
                "guidance": self._guidance,
                "num_steps": self._steps,
                "width": self._width,
                "height": self._height,
                "denoising_start": 0.0,
                "denoising_end": 1.0,
                "is_intermediate": True,
                "use_cache": False,
            },
            "flux_vae_decode": {
                "id": "flux_vae_decode",
                "type": "flux_vae_decode",
                "is_intermediate": False,
                "use_cache": False,
            },
        }
        edges = [
            _edge("flux_model_loader", "clip", "flux_text_encoder", "clip"),
            _edge("flux_model_loader", "t5_encoder", "flux_text_encoder", "t5_encoder"),
            _edge(
                "flux_model_loader", "max_seq_len", "flux_text_encoder", "t5_max_seq_len"
            ),
            _edge("flux_model_loader", "transformer", "flux_denoise", "transformer"),
            _edge("flux_model_loader", "vae", "flux_denoise", "controlnet_vae"),
            _edge("flux_model_loader", "vae", "flux_vae_decode", "vae"),
            _edge("positive_prompt", "value", "flux_text_encoder", "prompt"),
            _edge("flux_text_encoder", "conditioning", "pos_cond_collect", "item"),
            _edge(
                "pos_cond_collect",
                "collection",
                "flux_denoise",
                "positive_text_conditioning",
            ),
            _edge("seed", "value", "flux_denoise", "seed"),
            _edge("flux_denoise", "latents", "flux_vae_decode", "latents"),
        ]

        if with_kontext and uploaded_names:
            nodes["flux_kontext_collect"] = {
                "id": "flux_kontext_collect",
                "type": "collect",
                "is_intermediate": True,
                "use_cache": False,
            }
            for index, image_name in enumerate(uploaded_names):
                prep_id = f"flux_kontext_image_prep_{index}"
                kontext_id = f"flux_kontext_{index}"
                nodes[prep_id] = {
                    "id": prep_id,
                    "type": "flux_kontext_image_prep",
                    "images": [{"image_name": image_name}],
                    "is_intermediate": True,
                    "use_cache": False,
                }
                nodes[kontext_id] = {
                    "id": kontext_id,
                    "type": "flux_kontext",
                    "is_intermediate": True,
                    "use_cache": False,
                }
                edges.append(_edge(prep_id, "image", kontext_id, "image"))
                edges.append(
                    _edge(kontext_id, "kontext_cond", "flux_kontext_collect", "item")
                )
            edges.append(
                _edge(
                    "flux_kontext_collect",
                    "collection",
                    "flux_denoise",
                    "kontext_conditioning",
                )
            )

        return {
            "id": "forecats-flux-image",
            "nodes": nodes,
            "edges": edges,
        }

    def _build_flux2_graph(
        self,
        prompt: str,
        uploaded_names: list[str],
        main_model: dict[str, Any],
        *,
        seed: int,
    ) -> dict:
        """Build a FLUX.2 (Klein / Dev) generation graph with optional Kontext refs."""
        companions = _list_model_records(
            self._client, [*VAE_MODEL_TYPES, *ENCODER_MODEL_TYPES]
        )
        variant = str(main_model.get("variant") or "").lower()
        is_dev = variant == "dev"
        vae = _pick_vae_for_base(
            companions, main_base="flux2"
        )

        if is_dev:
            loader_type = "flux2_dev_model_loader"
            text_type = "flux2_dev_text_encoder"
            encoder = _pick_model(companions, model_type="mistral_encoder")
            loader: dict[str, Any] = {
                "id": "flux2_model_loader",
                "type": loader_type,
                "model": _model_identifier(main_model),
                "is_intermediate": True,
                "use_cache": False,
            }
            if vae:
                loader["vae_model"] = _model_identifier(vae)
            if encoder:
                loader["mistral_encoder_model"] = _model_identifier(encoder)
            text_edges = [
                _edge(
                    "flux2_model_loader",
                    "mistral_encoder",
                    "flux2_text_encoder",
                    "mistral_encoder",
                ),
                _edge(
                    "flux2_model_loader",
                    "max_seq_len",
                    "flux2_text_encoder",
                    "max_seq_len",
                ),
            ]
        else:
            loader_type = "flux2_klein_model_loader"
            text_type = "flux2_klein_text_encoder"
            encoder = _pick_model(companions, model_type="qwen3_encoder")
            loader = {
                "id": "flux2_model_loader",
                "type": loader_type,
                "model": _model_identifier(main_model),
                "is_intermediate": True,
                "use_cache": False,
            }
            if vae:
                loader["vae_model"] = _model_identifier(vae)
            if encoder:
                loader["qwen3_encoder_model"] = _model_identifier(encoder)
            text_edges = [
                _edge(
                    "flux2_model_loader",
                    "qwen3_encoder",
                    "flux2_text_encoder",
                    "qwen3_encoder",
                ),
                _edge(
                    "flux2_model_loader",
                    "max_seq_len",
                    "flux2_text_encoder",
                    "max_seq_len",
                ),
            ]

        nodes: dict[str, Any] = {
            "positive_prompt": {
                "id": "positive_prompt",
                "type": "string",
                "value": prompt,
                "is_intermediate": True,
                "use_cache": False,
            },
            "seed": {
                "id": "seed",
                "type": "integer",
                "value": seed,
                "is_intermediate": True,
                "use_cache": False,
            },
            "flux2_model_loader": loader,
            "flux2_text_encoder": {
                "id": "flux2_text_encoder",
                "type": text_type,
                "is_intermediate": True,
                "use_cache": False,
            },
            "pos_cond_collect": {
                "id": "pos_cond_collect",
                "type": "collect",
                "is_intermediate": True,
                "use_cache": False,
            },
            "flux2_denoise": {
                "id": "flux2_denoise",
                "type": "flux2_denoise",
                "num_steps": self._steps,
                "width": self._width,
                "height": self._height,
                "denoising_start": 0.0,
                "denoising_end": 1.0,
                "is_intermediate": True,
                "use_cache": False,
            },
            "flux2_vae_decode": {
                "id": "flux2_vae_decode",
                "type": "flux2_vae_decode",
                "is_intermediate": False,
                "use_cache": False,
            },
        }
        if is_dev:
            nodes["flux2_denoise"]["guidance"] = self._guidance

        edges = [
            *text_edges,
            _edge("flux2_model_loader", "transformer", "flux2_denoise", "transformer"),
            _edge("flux2_model_loader", "vae", "flux2_denoise", "vae"),
            _edge("flux2_model_loader", "vae", "flux2_vae_decode", "vae"),
            _edge("positive_prompt", "value", "flux2_text_encoder", "prompt"),
            _edge("flux2_text_encoder", "conditioning", "pos_cond_collect", "item"),
            _edge(
                "pos_cond_collect",
                "collection",
                "flux2_denoise",
                "positive_text_conditioning",
            ),
            _edge("seed", "value", "flux2_denoise", "seed"),
            _edge("flux2_denoise", "latents", "flux2_vae_decode", "latents"),
        ]

        if uploaded_names:
            # FLUX.2 packs refs via flux_kontext (image on the node) + collect chain.
            prev_collect: str | None = None
            for index, image_name in enumerate(uploaded_names):
                kontext_id = f"flux_kontext_{index}"
                collect_id = f"flux2_kontext_collect_{index}"
                nodes[kontext_id] = {
                    "id": kontext_id,
                    "type": "flux_kontext",
                    "image": {"image_name": image_name},
                    "is_intermediate": True,
                    "use_cache": False,
                }
                nodes[collect_id] = {
                    "id": collect_id,
                    "type": "collect",
                    "is_intermediate": True,
                    "use_cache": False,
                }
                edges.append(
                    _edge(kontext_id, "kontext_cond", collect_id, "item")
                )
                if prev_collect is not None:
                    edges.append(
                        _edge(prev_collect, "collection", collect_id, "collection")
                    )
                prev_collect = collect_id
            if prev_collect is not None:
                edges.append(
                    _edge(
                        prev_collect,
                        "collection",
                        "flux2_denoise",
                        "kontext_conditioning",
                    )
                )

        return {
            "id": "forecats-flux2-image",
            "nodes": nodes,
            "edges": edges,
        }

    def _extract_image_name(self, item: dict) -> str:
        session = item.get("session") or {}
        results = session.get("results") or {}
        nodes = (session.get("graph") or {}).get("nodes") or {}
        preferred_types = {
            "flux_vae_decode",
            "flux2_vae_decode",
            "l2i",
            "latents_to_image",
            "sd3_l2i",
        }
        preferred_ids = ("flux_vae_decode", "flux2_vae_decode")

        def _image_from(value: Any) -> str | None:
            if not isinstance(value, dict):
                return None
            image = value.get("image")
            if isinstance(image, dict) and image.get("image_name"):
                return str(image["image_name"])
            if value.get("image_name"):
                return str(value["image_name"])
            return None

        for node_id in preferred_ids:
            name = _image_from(results.get(node_id))
            if name:
                return name

        for node_id, value in results.items():
            node = nodes.get(node_id) or {}
            if isinstance(node, dict) and node.get("type") in preferred_types:
                name = _image_from(value)
                if name:
                    return name

        # Prefer non-intermediate image outputs over reference/pass-through nodes.
        for node_id, value in results.items():
            node = nodes.get(node_id) or {}
            if isinstance(node, dict) and node.get("is_intermediate"):
                continue
            name = _image_from(value)
            if name:
                return name

        for value in results.values():
            name = _image_from(value)
            if name:
                return name
        raise RuntimeError(f"InvokeAI completed without an image result: {results}")
