"""Build activity/image providers from config entry data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..const import (
    CONF_ACTIVITY_PROVIDER,
    CONF_GEMINI_ACTIVITY_MODEL,
    CONF_GEMINI_ACTIVITY_TIMEOUT,
    CONF_GEMINI_API_KEY,
    CONF_GEMINI_IMAGE_ASPECT_RATIO,
    CONF_GEMINI_IMAGE_MODEL,
    CONF_GEMINI_IMAGE_RESOLUTION,
    CONF_GEMINI_IMAGE_TIMEOUT,
    CONF_IMAGE_PROVIDER,
    CONF_INVOKEAI_ACTIVITY_MODEL,
    CONF_INVOKEAI_ACTIVITY_TIMEOUT,
    CONF_INVOKEAI_API_TOKEN,
    CONF_INVOKEAI_BASE_URL,
    CONF_INVOKEAI_IMAGE_GUIDANCE,
    CONF_INVOKEAI_IMAGE_HEIGHT,
    CONF_INVOKEAI_IMAGE_MODEL,
    CONF_INVOKEAI_IMAGE_SEED,
    CONF_INVOKEAI_IMAGE_STEPS,
    CONF_INVOKEAI_IMAGE_TIMEOUT,
    CONF_INVOKEAI_IMAGE_WIDTH,
    CONF_NAME,
    CONF_OPENAI_ACTIVITY_MODEL,
    CONF_OPENAI_ACTIVITY_TIMEOUT,
    CONF_OPENAI_API_KEY,
    CONF_OPENAI_BASE_URL,
    CONF_UNSLOTH_ACTIVITY_MODEL,
    CONF_UNSLOTH_ACTIVITY_TIMEOUT,
    CONF_UNSLOTH_API_KEY,
    CONF_UNSLOTH_BASE_URL,
    CONF_UNSLOTH_IMAGE_GUIDANCE,
    CONF_UNSLOTH_IMAGE_HEIGHT,
    CONF_UNSLOTH_IMAGE_MODEL,
    CONF_UNSLOTH_IMAGE_SEED,
    CONF_UNSLOTH_IMAGE_STEPS,
    CONF_UNSLOTH_IMAGE_TIMEOUT,
    CONF_UNSLOTH_IMAGE_WIDTH,
    DEFAULT_GEMINI_ACTIVITY_MODEL,
    DEFAULT_GEMINI_ACTIVITY_TIMEOUT,
    DEFAULT_GEMINI_IMAGE_ASPECT_RATIO,
    DEFAULT_GEMINI_IMAGE_MODEL,
    DEFAULT_GEMINI_IMAGE_RESOLUTION,
    DEFAULT_GEMINI_IMAGE_TIMEOUT,
    DEFAULT_INVOKEAI_ACTIVITY_MODEL,
    DEFAULT_INVOKEAI_ACTIVITY_TIMEOUT,
    DEFAULT_INVOKEAI_BASE_URL,
    DEFAULT_INVOKEAI_IMAGE_GUIDANCE,
    DEFAULT_INVOKEAI_IMAGE_HEIGHT,
    DEFAULT_INVOKEAI_IMAGE_MODEL,
    DEFAULT_INVOKEAI_IMAGE_SEED,
    DEFAULT_INVOKEAI_IMAGE_STEPS,
    DEFAULT_INVOKEAI_IMAGE_TIMEOUT,
    DEFAULT_INVOKEAI_IMAGE_WIDTH,
    DEFAULT_OPENAI_ACTIVITY_MODEL,
    DEFAULT_OPENAI_ACTIVITY_TIMEOUT,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_UNSLOTH_ACTIVITY_MODEL,
    DEFAULT_UNSLOTH_ACTIVITY_TIMEOUT,
    DEFAULT_UNSLOTH_BASE_URL,
    DEFAULT_UNSLOTH_IMAGE_GUIDANCE,
    DEFAULT_UNSLOTH_IMAGE_HEIGHT,
    DEFAULT_UNSLOTH_IMAGE_MODEL,
    DEFAULT_UNSLOTH_IMAGE_SEED,
    DEFAULT_UNSLOTH_IMAGE_STEPS,
    DEFAULT_UNSLOTH_IMAGE_TIMEOUT,
    DEFAULT_UNSLOTH_IMAGE_WIDTH,
    PROVIDER_GEMINI,
    PROVIDER_INVOKEAI,
    PROVIDER_OPENAI,
    PROVIDER_UNSLOTH,
)
from .base import ActivityProvider, ImageProvider
from .gemini import GeminiActivityProvider, GeminiImageProvider
from .invokeai import InvokeAIActivityProvider, InvokeAIImageProvider
from .openai_compatible import OpenAICompatibleActivityProvider
from .unsloth import UnslothActivityProvider, UnslothImageProvider


@dataclass
class InstanceRuntime:
    """Runtime objects for one Forecats config entry."""

    name: str
    entry_id: str
    activity_provider: ActivityProvider
    image_provider: ImageProvider


def build_runtime(config: dict[str, Any], entry_id: str = "default") -> InstanceRuntime:
    """Create providers for a single Forecats instance config."""
    activity = _build_activity_provider(config)
    image = _build_image_provider(config)
    return InstanceRuntime(
        name=config.get(CONF_NAME, "Forecats"),
        entry_id=entry_id,
        activity_provider=activity,
        image_provider=image,
    )


def _build_activity_provider(config: dict[str, Any]) -> ActivityProvider:
    provider = config.get(CONF_ACTIVITY_PROVIDER, PROVIDER_GEMINI)
    if provider == PROVIDER_GEMINI:
        return GeminiActivityProvider(
            api_key=config[CONF_GEMINI_API_KEY],
            model=config.get(CONF_GEMINI_ACTIVITY_MODEL, DEFAULT_GEMINI_ACTIVITY_MODEL),
            timeout_seconds=int(
                config.get(
                    CONF_GEMINI_ACTIVITY_TIMEOUT, DEFAULT_GEMINI_ACTIVITY_TIMEOUT
                )
            ),
        )
    if provider == PROVIDER_OPENAI:
        return OpenAICompatibleActivityProvider(
            base_url=config.get(CONF_OPENAI_BASE_URL, DEFAULT_OPENAI_BASE_URL),
            api_key=config.get(CONF_OPENAI_API_KEY) or None,
            model=config.get(
                CONF_OPENAI_ACTIVITY_MODEL, DEFAULT_OPENAI_ACTIVITY_MODEL
            ),
            timeout_seconds=int(
                config.get(
                    CONF_OPENAI_ACTIVITY_TIMEOUT, DEFAULT_OPENAI_ACTIVITY_TIMEOUT
                )
            ),
        )
    if provider == PROVIDER_INVOKEAI:
        return InvokeAIActivityProvider(
            base_url=config.get(CONF_INVOKEAI_BASE_URL, DEFAULT_INVOKEAI_BASE_URL),
            api_token=config.get(CONF_INVOKEAI_API_TOKEN) or None,
            model=config.get(
                CONF_INVOKEAI_ACTIVITY_MODEL, DEFAULT_INVOKEAI_ACTIVITY_MODEL
            ),
            timeout_seconds=int(
                config.get(
                    CONF_INVOKEAI_ACTIVITY_TIMEOUT, DEFAULT_INVOKEAI_ACTIVITY_TIMEOUT
                )
            ),
        )
    if provider == PROVIDER_UNSLOTH:
        return UnslothActivityProvider(
            base_url=config.get(CONF_UNSLOTH_BASE_URL, DEFAULT_UNSLOTH_BASE_URL),
            api_key=config.get(CONF_UNSLOTH_API_KEY) or None,
            model=config.get(
                CONF_UNSLOTH_ACTIVITY_MODEL, DEFAULT_UNSLOTH_ACTIVITY_MODEL
            ),
            timeout_seconds=int(
                config.get(
                    CONF_UNSLOTH_ACTIVITY_TIMEOUT, DEFAULT_UNSLOTH_ACTIVITY_TIMEOUT
                )
            ),
        )
    raise ValueError(f"Unknown activity provider: {provider}")


def _build_image_provider(config: dict[str, Any]) -> ImageProvider:
    provider = config.get(CONF_IMAGE_PROVIDER, PROVIDER_GEMINI)
    if provider == PROVIDER_GEMINI:
        return GeminiImageProvider(
            api_key=config[CONF_GEMINI_API_KEY],
            model=config.get(CONF_GEMINI_IMAGE_MODEL, DEFAULT_GEMINI_IMAGE_MODEL),
            timeout_seconds=int(
                config.get(CONF_GEMINI_IMAGE_TIMEOUT, DEFAULT_GEMINI_IMAGE_TIMEOUT)
            ),
            aspect_ratio=config.get(
                CONF_GEMINI_IMAGE_ASPECT_RATIO, DEFAULT_GEMINI_IMAGE_ASPECT_RATIO
            ),
            resolution=config.get(
                CONF_GEMINI_IMAGE_RESOLUTION, DEFAULT_GEMINI_IMAGE_RESOLUTION
            ),
        )
    if provider == PROVIDER_INVOKEAI:
        return InvokeAIImageProvider(
            base_url=config.get(CONF_INVOKEAI_BASE_URL, DEFAULT_INVOKEAI_BASE_URL),
            api_token=config.get(CONF_INVOKEAI_API_TOKEN) or None,
            model=config.get(CONF_INVOKEAI_IMAGE_MODEL, DEFAULT_INVOKEAI_IMAGE_MODEL),
            timeout_seconds=int(
                config.get(
                    CONF_INVOKEAI_IMAGE_TIMEOUT, DEFAULT_INVOKEAI_IMAGE_TIMEOUT
                )
            ),
            width=int(
                config.get(CONF_INVOKEAI_IMAGE_WIDTH, DEFAULT_INVOKEAI_IMAGE_WIDTH)
            ),
            height=int(
                config.get(CONF_INVOKEAI_IMAGE_HEIGHT, DEFAULT_INVOKEAI_IMAGE_HEIGHT)
            ),
            seed=int(config.get(CONF_INVOKEAI_IMAGE_SEED, DEFAULT_INVOKEAI_IMAGE_SEED)),
            steps=int(
                config.get(CONF_INVOKEAI_IMAGE_STEPS, DEFAULT_INVOKEAI_IMAGE_STEPS)
            ),
            guidance=float(
                config.get(
                    CONF_INVOKEAI_IMAGE_GUIDANCE, DEFAULT_INVOKEAI_IMAGE_GUIDANCE
                )
            ),
        )
    if provider == PROVIDER_UNSLOTH:
        return UnslothImageProvider(
            base_url=config.get(CONF_UNSLOTH_BASE_URL, DEFAULT_UNSLOTH_BASE_URL),
            api_key=config.get(CONF_UNSLOTH_API_KEY) or None,
            model=config.get(CONF_UNSLOTH_IMAGE_MODEL, DEFAULT_UNSLOTH_IMAGE_MODEL)
            or "",
            timeout_seconds=int(
                config.get(CONF_UNSLOTH_IMAGE_TIMEOUT, DEFAULT_UNSLOTH_IMAGE_TIMEOUT)
            ),
            width=int(
                config.get(CONF_UNSLOTH_IMAGE_WIDTH, DEFAULT_UNSLOTH_IMAGE_WIDTH)
            ),
            height=int(
                config.get(CONF_UNSLOTH_IMAGE_HEIGHT, DEFAULT_UNSLOTH_IMAGE_HEIGHT)
            ),
            seed=int(config.get(CONF_UNSLOTH_IMAGE_SEED, DEFAULT_UNSLOTH_IMAGE_SEED)),
            steps=int(
                config.get(CONF_UNSLOTH_IMAGE_STEPS, DEFAULT_UNSLOTH_IMAGE_STEPS)
            ),
            guidance=float(
                config.get(
                    CONF_UNSLOTH_IMAGE_GUIDANCE, DEFAULT_UNSLOTH_IMAGE_GUIDANCE
                )
            ),
        )
    raise ValueError(f"Unknown image provider: {provider}")
