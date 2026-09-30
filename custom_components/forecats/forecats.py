"""Generate pet pictures based on weather forecasts."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from PIL import Image

try:
    from .image_processing import recolor_image, resize_image
    from .models import GenerateRequest
    from .providers import InstanceRuntime
except ImportError:  # For local testing
    from image_processing import recolor_image, resize_image
    from models import GenerateRequest
    from providers import InstanceRuntime

_LOGGER = logging.getLogger(__name__)

DEFAULT_ORIGINAL_NAME = "forecats_original.png"
DEFAULT_OPTIMIZED_NAME = "forecats_optimized.png"


def _resolve_output_paths(output_dir: str, config_path: Path) -> tuple[Path, Path]:
    """Map output directory to (original, optimized) file paths."""
    path = Path(output_dir)
    if not path.is_absolute():
        path = config_path / path
    path.mkdir(parents=True, exist_ok=True)
    return path / DEFAULT_ORIGINAL_NAME, path / DEFAULT_OPTIMIZED_NAME


def generate_pet_pic(
    data: GenerateRequest,
    config_dir: str,
    runtime: InstanceRuntime,
) -> tuple[str, str]:
    """Generate, crop, and dither a pet picture using the instance providers.

    Args:
        data: Per-run generation parameters (including output_dir)
        config_dir: Home Assistant configuration directory for prompt history
        runtime: Providers for the targeted config entry

    Returns:
        tuple[str, str]: Paths of the saved original and optimized PNG images.

    """
    config_path = Path(config_dir)
    data_dir = config_path / "forecats_data" / runtime.entry_id
    data_dir.mkdir(parents=True, exist_ok=True)
    original_filepath, optimized_filepath = _resolve_output_paths(
        data.output_dir, config_path
    )

    prompt_history_filepath = data_dir / "forecats_prompt_history.txt"

    prompt_history = load_prompt_history(prompt_history_filepath)
    images: dict[str, Image.Image] = {}
    if data.input_image_dir:
        images.update(load_images_from_dir(data.input_image_dir))
    if data.input_image_paths:
        images.update(load_images(data.input_image_paths))
    if not images:
        _LOGGER.warning("No images loaded — generation may produce generic results.")

    art_style = random.choice(data.art_styles)
    _LOGGER.info(
        "Instance %s: selected art style %s",
        runtime.name,
        art_style,
    )

    activity = runtime.activity_provider.generate_activity(data, prompt_history)
    _LOGGER.info("Generated activity: %s", activity)

    prompt_history.append(activity)
    prompt_history = prompt_history[-20:]
    save_prompt_history(prompt_history_filepath, prompt_history)

    _LOGGER.info("Generating image for: %s", activity)
    image = runtime.image_provider.generate_image(data, activity, images, art_style)

    resized_image = resize_image(image.copy(), data.final_image_size)
    optimized_image = recolor_image(resized_image, data.display_profile)

    image.save(original_filepath)
    optimized_image.save(optimized_filepath)

    _LOGGER.info(
        "Images saved: original=%s optimized=%s",
        original_filepath,
        optimized_filepath,
    )
    return str(original_filepath), str(optimized_filepath)


def load_prompt_history(filepath: Path) -> list[str]:
    """Load past prompts from file."""
    if filepath.exists():
        return filepath.read_text().splitlines()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return []


def save_prompt_history(filepath: Path, history: list[str]) -> None:
    """Save prompt history to file."""
    Path(filepath).write_text("\n".join(history))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def _resize_image(img_path: Path, max_size: int = 1024) -> Image.Image:
    img = Image.open(img_path)
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def load_images(image_paths: list[str], max_size: int = 1024) -> dict[str, Image.Image]:
    """Load and resize images."""
    paths = [Path(p) for p in image_paths]
    valid_paths = [p for p in paths if p.exists()]
    if not valid_paths:
        _LOGGER.warning("No valid image paths found from: %s", paths)
    return {p.name: _resize_image(p, max_size) for p in valid_paths}


def load_images_from_dir(dir_path: str, max_size: int = 1024) -> dict[str, Image.Image]:
    """Load and resize all images from a directory."""
    path = Path(dir_path)
    if not path.is_dir():
        _LOGGER.warning("input_image_dir not found: %s", path)
        return {}
    files = [p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        _LOGGER.warning("No image files found in: %s", path)
    return {p.name: _resize_image(p, max_size) for p in files}
