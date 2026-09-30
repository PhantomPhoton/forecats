# test.py
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "custom_components"))

from dotenv import load_dotenv

from forecats.const import (
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
from forecats.forecats import generate_pet_pic
from forecats.models import GenerateRequest, Pet
from forecats.providers import build_runtime

load_dotenv()

TEST_DIR = "."

input_image_paths = [
    "./forecats_data/input_images/milo.jpg",
    "./forecats_data/input_images/milo_and_tolmie.jpg",
    "./forecats_data/input_images/tolmie.jpg",
    "./forecats_data/input_images/milo_and_tolmie_2.jpg",
    "./forecats_data/input_images/tolmie_2.jpg",
]

activity_provider = os.getenv("ACTIVITY_PROVIDER", PROVIDER_GEMINI)
image_provider = os.getenv("IMAGE_PROVIDER", PROVIDER_GEMINI)
output_dir = os.getenv("OUTPUT_DIR", "www/daily_forecats")

config = {
    CONF_NAME: os.getenv("INSTANCE_NAME", "Local Test"),
    CONF_ACTIVITY_PROVIDER: activity_provider,
    CONF_IMAGE_PROVIDER: image_provider,
}

if PROVIDER_GEMINI in (activity_provider, image_provider):
    config[CONF_GEMINI_API_KEY] = os.getenv("GEMINI_API_KEY")
    config[CONF_GEMINI_ACTIVITY_MODEL] = os.getenv(
        "GEMINI_ACTIVITY_MODEL", DEFAULT_GEMINI_ACTIVITY_MODEL
    )
    config[CONF_GEMINI_IMAGE_MODEL] = os.getenv(
        "GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL
    )
    config[CONF_GEMINI_ACTIVITY_TIMEOUT] = int(
        os.getenv("GEMINI_ACTIVITY_TIMEOUT", DEFAULT_GEMINI_ACTIVITY_TIMEOUT)
    )
    config[CONF_GEMINI_IMAGE_TIMEOUT] = int(
        os.getenv("GEMINI_IMAGE_TIMEOUT", DEFAULT_GEMINI_IMAGE_TIMEOUT)
    )
    config[CONF_GEMINI_IMAGE_ASPECT_RATIO] = os.getenv(
        "GEMINI_IMAGE_ASPECT_RATIO", DEFAULT_GEMINI_IMAGE_ASPECT_RATIO
    )
    config[CONF_GEMINI_IMAGE_RESOLUTION] = os.getenv(
        "GEMINI_IMAGE_RESOLUTION", DEFAULT_GEMINI_IMAGE_RESOLUTION
    )

if PROVIDER_OPENAI == activity_provider:
    config[CONF_OPENAI_BASE_URL] = os.getenv(
        "OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL
    )
    config[CONF_OPENAI_API_KEY] = os.getenv("OPENAI_API_KEY", "")
    config[CONF_OPENAI_ACTIVITY_MODEL] = os.getenv(
        "OPENAI_ACTIVITY_MODEL", DEFAULT_OPENAI_ACTIVITY_MODEL
    )
    config[CONF_OPENAI_ACTIVITY_TIMEOUT] = int(
        os.getenv("OPENAI_ACTIVITY_TIMEOUT", DEFAULT_OPENAI_ACTIVITY_TIMEOUT)
    )

if PROVIDER_INVOKEAI in (activity_provider, image_provider):
    config[CONF_INVOKEAI_BASE_URL] = os.getenv(
        "INVOKEAI_BASE_URL", DEFAULT_INVOKEAI_BASE_URL
    )
    config[CONF_INVOKEAI_API_TOKEN] = os.getenv("INVOKEAI_API_TOKEN", "")
    config[CONF_INVOKEAI_ACTIVITY_MODEL] = os.getenv(
        "INVOKEAI_ACTIVITY_MODEL", DEFAULT_INVOKEAI_ACTIVITY_MODEL
    )
    config[CONF_INVOKEAI_IMAGE_MODEL] = os.getenv(
        "INVOKEAI_IMAGE_MODEL", DEFAULT_INVOKEAI_IMAGE_MODEL
    )
    config[CONF_INVOKEAI_ACTIVITY_TIMEOUT] = int(
        os.getenv("INVOKEAI_ACTIVITY_TIMEOUT", DEFAULT_INVOKEAI_ACTIVITY_TIMEOUT)
    )
    config[CONF_INVOKEAI_IMAGE_TIMEOUT] = int(
        os.getenv("INVOKEAI_IMAGE_TIMEOUT", DEFAULT_INVOKEAI_IMAGE_TIMEOUT)
    )
    config[CONF_INVOKEAI_IMAGE_WIDTH] = int(
        os.getenv("INVOKEAI_IMAGE_WIDTH", DEFAULT_INVOKEAI_IMAGE_WIDTH)
    )
    config[CONF_INVOKEAI_IMAGE_HEIGHT] = int(
        os.getenv("INVOKEAI_IMAGE_HEIGHT", DEFAULT_INVOKEAI_IMAGE_HEIGHT)
    )
    config[CONF_INVOKEAI_IMAGE_SEED] = int(
        os.getenv("INVOKEAI_IMAGE_SEED", DEFAULT_INVOKEAI_IMAGE_SEED)
    )
    config[CONF_INVOKEAI_IMAGE_STEPS] = int(
        os.getenv("INVOKEAI_IMAGE_STEPS", DEFAULT_INVOKEAI_IMAGE_STEPS)
    )
    config[CONF_INVOKEAI_IMAGE_GUIDANCE] = float(
        os.getenv("INVOKEAI_IMAGE_GUIDANCE", DEFAULT_INVOKEAI_IMAGE_GUIDANCE)
    )

if PROVIDER_UNSLOTH in (activity_provider, image_provider):
    config[CONF_UNSLOTH_BASE_URL] = os.getenv(
        "UNSLOTH_BASE_URL", DEFAULT_UNSLOTH_BASE_URL
    )
    config[CONF_UNSLOTH_API_KEY] = os.getenv("UNSLOTH_API_KEY", "")
    config[CONF_UNSLOTH_ACTIVITY_MODEL] = os.getenv(
        "UNSLOTH_ACTIVITY_MODEL", DEFAULT_UNSLOTH_ACTIVITY_MODEL
    )
    config[CONF_UNSLOTH_IMAGE_MODEL] = os.getenv(
        "UNSLOTH_IMAGE_MODEL", DEFAULT_UNSLOTH_IMAGE_MODEL
    )
    config[CONF_UNSLOTH_ACTIVITY_TIMEOUT] = int(
        os.getenv("UNSLOTH_ACTIVITY_TIMEOUT", DEFAULT_UNSLOTH_ACTIVITY_TIMEOUT)
    )
    config[CONF_UNSLOTH_IMAGE_TIMEOUT] = int(
        os.getenv("UNSLOTH_IMAGE_TIMEOUT", DEFAULT_UNSLOTH_IMAGE_TIMEOUT)
    )
    config[CONF_UNSLOTH_IMAGE_WIDTH] = int(
        os.getenv("UNSLOTH_IMAGE_WIDTH", DEFAULT_UNSLOTH_IMAGE_WIDTH)
    )
    config[CONF_UNSLOTH_IMAGE_HEIGHT] = int(
        os.getenv("UNSLOTH_IMAGE_HEIGHT", DEFAULT_UNSLOTH_IMAGE_HEIGHT)
    )
    config[CONF_UNSLOTH_IMAGE_SEED] = int(
        os.getenv("UNSLOTH_IMAGE_SEED", DEFAULT_UNSLOTH_IMAGE_SEED)
    )
    config[CONF_UNSLOTH_IMAGE_STEPS] = int(
        os.getenv("UNSLOTH_IMAGE_STEPS", DEFAULT_UNSLOTH_IMAGE_STEPS)
    )
    config[CONF_UNSLOTH_IMAGE_GUIDANCE] = float(
        os.getenv("UNSLOTH_IMAGE_GUIDANCE", DEFAULT_UNSLOTH_IMAGE_GUIDANCE)
    )

runtime = build_runtime(config, entry_id="local-test")

data = GenerateRequest(
    location="Toronto, Ontario, Canada",
    forecast={"datetime": "2025-6-07", "temperature": -10, "templow": -15, "condition": "sunny"},
    temperature_unit="C",
    pets=[
        Pet(
            name="Milo",
            type="cat",
            description="Milo is a domestic short-haired tabby. He is seven years old and we like to imagine that he is serious but a little goofy on the outside. He has green eyes, with big pupils when he is excited. He loves cuddling, sitting on his moms lap, his little brother Tolmie, holding little pom-poms in his mouth, and stealing kibbles from his brother. You can just *barely* see a little bit of his pink skin beneath his tummy fur, but it is cute and fuzzy with some paunch. He has cute white paws and a white tummy.",
        ),
        Pet(
            name="Tolmie",
            type="cat",
            description="Tolmie is a ragdoll, he has pretty blue eyes, and his black pupils get big when he's excited. We like to imagine that he is goofy. He loves his big brother Milo, playing with strings, rolling on the ground belly-up, and sprinting full speed around the house yelling. He is a little bigger (longer) than Milo, and all fluff. When he is happy he likes to stick his tail up.",
        ),
        Pet(
            name="Brutus",
            type="dog",
            description="Brutus is a golden retriever. He is a year old and we like to imagine that he is playful and energetic. He has brown eyes, with big pupils when he is excited. He loves playing fetch, running around the yard, and cuddling with his family. He is a little bigger (longer) than Tolmie, and all fluff. When he is happy he likes to stick his tail up.",
        ),
    ],
    input_image_paths=input_image_paths,
    art_styles=[
        "comic book",
        "anime",
        "pixar",
        "watercolor",
        "ghibli",
        "disney",
    ],
    final_image_size="800x480",
    output_dir=output_dir,
    display_profile="spectra6",
)

img = generate_pet_pic(data, TEST_DIR, runtime)
print(img)
