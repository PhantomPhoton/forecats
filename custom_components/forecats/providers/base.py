"""Provider protocols for Forecats."""

from __future__ import annotations

from typing import Protocol, TypedDict

from PIL import Image

from ..models import GenerateRequest


class SelectOption(TypedDict):
    """Dropdown option for config flow model selectors."""

    value: str
    label: str


class ActivityProvider(Protocol):
    """Generate an activity description for the daily illustration."""

    def generate_activity(
        self,
        data: GenerateRequest,
        prompt_history: list[str],
    ) -> str:
        """Return a single-line activity description."""


class ImageProvider(Protocol):
    """Generate the daily pet illustration."""

    def generate_image(
        self,
        data: GenerateRequest,
        activity: str,
        input_images: dict[str, Image.Image],
        art_style: str,
    ) -> Image.Image:
        """Return a PIL image of the generated illustration."""
