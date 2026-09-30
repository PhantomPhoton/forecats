"""Gemini activity and image providers."""

from __future__ import annotations

import io

from google import genai
from google.genai import types
from PIL import Image

from ..models import GenerateRequest
from .base import SelectOption
from .prompts import build_activity_prompt, build_image_prompt

# Suggested models when the Gemini API is not queried for a live catalog.
ACTIVITY_MODEL_OPTIONS: list[SelectOption] = [
    {"value": "gemini-2.5-flash-lite", "label": "gemini-2.5-flash-lite"},
    {"value": "gemini-2.5-flash", "label": "gemini-2.5-flash"},
    {"value": "gemini-2.0-flash", "label": "gemini-2.0-flash"},
]

IMAGE_MODEL_OPTIONS: list[SelectOption] = [
    {
        "value": "gemini-3-pro-image-preview",
        "label": "gemini-3-pro-image-preview (Nano Banana Pro)",
    },
    {
        "value": "gemini-3.1-flash-image",
        "label": "gemini-3.1-flash-image (Nano Banana 2)",
    },
    {
        "value": "gemini-2.5-flash-image",
        "label": "gemini-2.5-flash-image (Nano Banana)",
    },
]


def list_activity_models() -> list[SelectOption]:
    """Return suggested Gemini activity models."""
    return list(ACTIVITY_MODEL_OPTIONS)


def list_image_models() -> list[SelectOption]:
    """Return suggested Gemini image models."""
    return list(IMAGE_MODEL_OPTIONS)


def _client(api_key: str, timeout_seconds: int) -> genai.Client:
    """Create a Gemini client, applying timeout when the SDK supports it."""
    try:
        return genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_seconds * 1000},
        )
    except TypeError:
        return genai.Client(api_key=api_key)


class GeminiActivityProvider:
    """Generate activity text with Gemini."""

    def __init__(self, api_key: str, model: str, timeout_seconds: int) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def generate_activity(
        self,
        data: GenerateRequest,
        prompt_history: list[str],
    ) -> str:
        client = _client(self._api_key, self._timeout)
        prompt = build_activity_prompt(data, prompt_history)
        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        if not response.text:
            raise RuntimeError("Gemini returned no activity.")
        return response.text


class GeminiImageProvider:
    """Generate images with Gemini Nano Banana models."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int,
        *,
        aspect_ratio: str,
        resolution: str,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._aspect_ratio = aspect_ratio
        self._resolution = resolution

    def generate_image(
        self,
        data: GenerateRequest,
        activity: str,
        input_images: dict[str, Image.Image],
        art_style: str,
    ) -> Image.Image:
        client = _client(self._api_key, self._timeout)
        prompt = build_image_prompt(
            data, activity, list(input_images.keys()), art_style
        )
        response = client.models.generate_content(
            model=self._model,
            contents=[prompt, *input_images.values()],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=self._aspect_ratio,
                    image_size=self._resolution,
                ),
            ),
        )
        if not response.parts:
            raise RuntimeError("Gemini returned no image.")

        for part in response.parts:
            img = part.as_image()
            if img and img.image_bytes:
                img_pil = Image.open(io.BytesIO(img.image_bytes))
                return img_pil.copy()

        raise RuntimeError("No image part in Gemini response.")
