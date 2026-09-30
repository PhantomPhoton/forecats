from pydantic import BaseModel


class Pet(BaseModel):
    name: str
    type: str
    description: str


class GenerateRequest(BaseModel):
    """Request model for generating pet pictures (per-run inputs only)."""

    location: str
    forecast: dict
    temperature_unit: str
    pets: list[Pet]
    input_image_paths: list[str] | None = None
    input_image_dir: str | None = None
    art_styles: list[str]

    final_image_size: str
    # Resolved filesystem directory for forecats_original/optimized.png.
    output_dir: str

    display_profile: str | None = None
