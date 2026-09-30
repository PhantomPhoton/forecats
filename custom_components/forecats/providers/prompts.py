"""Shared prompt builders for Forecats providers."""

from __future__ import annotations

import textwrap

from ..models import GenerateRequest


def build_activity_prompt(data: GenerateRequest, prompt_history: list[str]) -> str:
    """Build the activity-generation prompt shared across providers."""
    pet_types = ", ".join(f"{pet.name}, {pet.type}" for pet in data.pets)
    return textwrap.dedent(
        f"""
        You are a prompt generator for static AI cartoon art generation model. Your task is to generate a seasonally appropriate activity for {len(data.pets)} pets to do based on the date and weather conditions provided, which will be used to draw a single picture.

        The pets are: {pet_types}.

        The date is {data.forecast.get("datetime", "")}.

        The weather forecast is for {data.location}.

        The forecast is:
        {data.forecast}

        The last 20 activities you generated were:
        {prompt_history}

        ************************************************

        Follow this prompt:

        Generate a fun activity for {len(data.pets)} pets to do together that fits the weather conditions and time of year.

        Heuristics:
        - You can anthropomorphize the pets to do human-like activities, or you can make them do more pet-like activities occasionally.
        - The activity can be either indoors or outdoors
        - The activity should be seasonally appropriate
        - Activities should be 30% set in locations in {data.location}, and 20% set in other specific locations with similar weather, and 50% set in generic locations.
        - The mix of indoor/outdoor should be seasonally appropriate. Summer is mostly outdoor, winter is 50/50 indoor/outdoor.
        - It can be a mundane activity (waiting for the bus, commuting, shopping, reading, etc.) or it can be exciting (playing in the snow, sports, going to a festival, playing tag, games, etc.).

        Rules:
        - You must come up with the following:
            - Activity: A short (<5 words) description of the activity
            - Foreground: A description of what the pets are doing, including any clothing or accessories they are wearing.
            - Background: A description of the background elements (e.g. buildings, landmarks, trees, furniture, etc.)
        - You don't have to describe the weather
        - Do not describe the general appearance of the pets, with the exception of clothing or accessories needed for the activity
        - The activity should involve all {len(data.pets)} pets
        - The activity must not be similar to any of the last 20 activities you generated.
        - Respond in a single line, no more than 100 words
        - Do not use newlines
        - Respond with "Activity: <activity>, Foreground: <foreground>, Background: <background>"
        """,
    ).strip()


def build_image_prompt(
    data: GenerateRequest,
    activity: str,
    input_image_names: list[str],
    art_style: str,
) -> str:
    """Build the image-generation prompt shared across providers."""
    pets_information = "\n".join(
        f"- {pet.name}, {pet.type}, {pet.description}" for pet in data.pets
    )
    return textwrap.dedent(
        f"""
        You are an AI artist creating daily weather illustrations featuring pets based on a weather forecast and an activity that will be given to you. Your task is to generate a vibrant and engaging illustration that captures the essence of the weather conditions and the pets' activity in a specific art style.

        The weather forecast is for {data.location} on {data.forecast.get("datetime", "")}:
        {data.forecast}

        You have {len(data.pets)} pets to illustrate, here are their name, type, and descriptions:
        {pets_information}

        The activity they are doing today is:
        {activity}

        The art style should be:
        {art_style}

        You will be given {len(input_image_names)} input images with these names (in this order):
        {", ".join(input_image_names)}

        ***********************************************

        Create a vibrant and engaging illustration in the recommended style that captures the essence of the weather conditions and the pets' activity. Use colors and elements that reflect the forecasted weather, making the scene lively and appropriate for the time of year.

        Additionally, create a small box in the bottom left corner, in the style of the image. This box should contain:
        - A < three word description of the weather conditions
        - The daily high temperature (forecast field: temperature)
        - The daily low temperature (forecast field: templow)
        - Note that the temperatures are in {data.temperature_unit}

        Heuristics:
        - Try to capture the mood of the weather, season, and activity in the illustration (e.g., bright and sunny, cozy indoors during snow, etc.).
        - Try to capture the pets personalities, but it is okay if they change based on the weather and activity.
        - This is for a color e-ink screen. I will handle the dithering later, but try to avoid very fine details that may be lost in dithering.

        Rules:
        - Only generate a single image.
        - The final image will be cropped in postprocessing to {data.final_image_size}, so compose the image accordingly and DON'T place anything near the edges.
        - The weather is important, so include elements that clearly indicate the weather conditions
        - Do not place the temperature box too close to the edge, or overlapping any important details.
        - Style the temperature box to fit the overall image and art style.
        - Use the input images as references for the pets' appearances.
        - Style the pets to fit the activity and weather conditions.
        """,
    ).strip()
