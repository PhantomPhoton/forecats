"""Forecats Home Assistant integration."""

from __future__ import annotations

import logging

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

try:
    from pathlib import Path

    import homeassistant.helpers.config_validation as cv
    import voluptuous as vol
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant, ServiceCall
    from homeassistant.exceptions import HomeAssistantError
    from homeassistant.helpers.typing import ConfigType

    from .forecats import generate_pet_pic
    from .models import GenerateRequest
    from .providers import build_runtime

    CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

    PET_SCHEMA = vol.Schema(
        {
            vol.Required("name"): cv.string,
            vol.Required("description"): cv.string,
            vol.Required("type", default="cat"): cv.string,
        }
    )

    MEDIA_ITEM_SCHEMA = vol.Schema(
        {
            vol.Required("media_content_id"): cv.string,
            vol.Required("media_content_type"): cv.string,
            vol.Optional("metadata"): dict,
        }
    )

    # Media picker (files/dirs), legacy filesystem path, or list of either.
    INPUT_IMAGE_DIR_SCHEMA = vol.Any(
        cv.string,
        MEDIA_ITEM_SCHEMA,
        [vol.Any(cv.string, MEDIA_ITEM_SCHEMA)],
    )

    ART_STYLE_SCHEMA = vol.Any(
        cv.string,
        vol.Schema({vol.Required("name"): cv.string}),
    )

    SERVICE_SCHEMA = vol.Schema(
        {
            vol.Required("config_entry_id"): cv.string,
            vol.Required("weather_entity"): cv.entity_domain("weather"),
            vol.Required("location"): cv.string,
            vol.Required("pets"): [PET_SCHEMA],
            vol.Optional("input_image_paths", default=[]): [cv.string],
            vol.Required("input_image_dir", default=[]): INPUT_IMAGE_DIR_SCHEMA,
            vol.Required("art_styles"): [ART_STYLE_SCHEMA],
            vol.Required("final_image_size"): cv.string,
            vol.Required(
                "output_dir", default="/config/www/daily_forecats"
            ): cv.string,
            vol.Required("display_profile", default=""): vol.Any(None, cv.string),
        }
    )

    async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
        """Set up the Forecats integration (YAML is config-entry only)."""
        hass.data.setdefault(DOMAIN, {})
        return True

    async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
        """Set up one Forecats instance from a config entry."""
        hass.data.setdefault(DOMAIN, {})
        hass.data[DOMAIN][entry.entry_id] = build_runtime(entry.data, entry.entry_id)
        entry.async_on_unload(entry.add_update_listener(_async_update_listener))

        if not hass.services.has_service(DOMAIN, "generate_pet_picture"):
            hass.services.async_register(
                DOMAIN,
                "generate_pet_picture",
                _handle_generate,
                schema=SERVICE_SCHEMA,
            )

        return True

    async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
        """Unload a Forecats config entry."""
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        if not hass.data.get(DOMAIN):
            hass.services.async_remove(DOMAIN, "generate_pet_picture")
        return True

    async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Reload providers when options/data change."""
        await hass.config_entries.async_reload(entry.entry_id)

    def _resolve_entry_id(hass: HomeAssistant, requested: str) -> str:
        entries = list(hass.config_entries.async_entries(DOMAIN))
        if not entries:
            raise HomeAssistantError(
                "No Forecats instances configured. Add one under Settings → Devices & Services."
            )
        if requested not in hass.data.get(DOMAIN, {}):
            raise HomeAssistantError(f"Unknown Forecats instance: {requested}")
        return requested

    async def _forecast_from_weather(
        hass: HomeAssistant, entity_id: str
    ) -> tuple[dict, str]:
        """Fetch today's daily forecast and temperature unit from a weather entity."""
        response = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"type": "daily"},
            target={"entity_id": entity_id},
            blocking=True,
            return_response=True,
        )
        if not response or entity_id not in response:
            raise HomeAssistantError(f"No forecast returned for {entity_id}")
        forecasts = response[entity_id].get("forecast") or []
        if not forecasts:
            raise HomeAssistantError(f"Empty daily forecast for {entity_id}")

        state = hass.states.get(entity_id)
        unit = state.attributes.get("temperature_unit") if state else None
        if not unit:
            raise HomeAssistantError(
                f"Weather entity {entity_id} has no temperature_unit attribute"
            )
        return forecasts[0], str(unit)

    def _local_media_path(hass: HomeAssistant, media_content_id: str) -> Path | None:
        """Map a media-source://media_source/... id to a filesystem path."""
        prefix = "media-source://media_source/"
        if not media_content_id.startswith(prefix):
            return None
        identifier = media_content_id[len(prefix) :]
        source_dir_id, _, location = identifier.partition("/")
        base = hass.config.media_dirs.get(source_dir_id)
        if not base:
            return None
        return Path(base) / location

    async def _resolve_media_path(
        hass: HomeAssistant, media_content_id: str
    ) -> Path:
        """Resolve a media content id or filesystem path to a Path."""
        # Plain filesystem path (YAML / advanced)
        if not media_content_id.startswith("media-source://"):
            return Path(media_content_id)

        local = _local_media_path(hass, media_content_id)
        if local is not None:
            return local

        # Non-local media sources: resolve playable files only
        from homeassistant.components import media_source

        try:
            play = await media_source.async_resolve_media(
                hass, media_content_id, None
            )
        except Exception as err:
            raise HomeAssistantError(
                f"Could not resolve media: {media_content_id}"
            ) from err
        if getattr(play, "path", None):
            return Path(play.path)
        raise HomeAssistantError(
            f"Media has no local path (folders must be under Local Media): {media_content_id}"
        )

    def _normalize_art_styles(styles: list) -> list[str]:
        """Accept list[str] or list[{name: str}] from the object selector."""
        out: list[str] = []
        for item in styles:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    out.append(str(name))
            else:
                out.append(str(item))
        if not out:
            raise HomeAssistantError("art_styles must include at least one style")
        return out

    _IMAGE_SUFFIXES = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
    }

    async def _normalize_input_images(
        hass: HomeAssistant, call_data: dict
    ) -> None:
        """Resolve media selector / paths into input_image_paths + input_image_dir."""
        raw = call_data.pop("input_image_dir", None)
        paths: list[str] = list(call_data.get("input_image_paths") or [])
        dirs: list[str] = []

        if raw is None:
            items: list = []
        elif isinstance(raw, list):
            items = raw
        else:
            items = [raw]

        for item in items:
            if isinstance(item, dict):
                content_id = item["media_content_id"]
            else:
                content_id = str(item)
            resolved = await _resolve_media_path(hass, content_id)
            if resolved.is_dir():
                dirs.append(str(resolved))
            else:
                paths.append(str(resolved))

        # GenerateRequest accepts a single dir; fold extra dirs into paths.
        if dirs:
            call_data["input_image_dir"] = dirs[0]
            for extra in dirs[1:]:
                for child in Path(extra).iterdir():
                    if child.suffix.lower() in _IMAGE_SUFFIXES:
                        paths.append(str(child))
        else:
            call_data["input_image_dir"] = None
        call_data["input_image_paths"] = paths

    async def _normalize_output(hass: HomeAssistant, call_data: dict) -> None:
        """Normalize output_dir to an absolute filesystem directory path."""
        raw = call_data.get("output_dir") or "/config/www/daily_forecats"
        # Allow a media-source id from older YAML, otherwise treat as a path.
        if isinstance(raw, dict):
            content_id = raw["media_content_id"]
            resolved = await _resolve_media_path(hass, content_id)
        elif str(raw).startswith("media-source://"):
            resolved = await _resolve_media_path(hass, str(raw))
        else:
            resolved = Path(str(raw))
            if not resolved.is_absolute():
                resolved = Path(hass.config.path()) / resolved
        if resolved.exists() and not resolved.is_dir():
            raise HomeAssistantError(
                f"output_dir must be a directory, got file: {resolved}"
            )
        call_data["output_dir"] = str(resolved)

    async def _handle_generate(call: ServiceCall) -> None:
        """Handle generate_pet_picture service calls."""
        hass = call.hass
        call_data = dict(call.data)
        entry_id = _resolve_entry_id(hass, call_data.pop("config_entry_id"))
        runtime = hass.data[DOMAIN][entry_id]

        weather_entity = call_data.pop("weather_entity")
        forecast, temperature_unit = await _forecast_from_weather(
            hass, weather_entity
        )
        call_data["forecast"] = forecast
        call_data["temperature_unit"] = temperature_unit

        call_data["art_styles"] = _normalize_art_styles(call_data["art_styles"])
        await _normalize_input_images(hass, call_data)
        await _normalize_output(hass, call_data)

        display_profile = call_data.get("display_profile")
        if display_profile == "":
            call_data["display_profile"] = None

        data = GenerateRequest(**call_data)
        _LOGGER.info(
            "generate_pet_picture for instance %s → %s",
            runtime.name,
            data.output_dir,
        )
        try:
            original_path, optimized_path = await hass.async_add_executor_job(
                generate_pet_pic,
                data,
                hass.config.path(),
                runtime,
            )
            _LOGGER.info(
                "Generated pet pictures: %s, %s", original_path, optimized_path
            )
        except Exception:
            _LOGGER.exception("Failed to generate pet picture")
            raise

except ImportError:
    # Allow importing submodules for local testing without Home Assistant installed.
    pass
