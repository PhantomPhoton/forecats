"""Config flow for Forecats."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    ACTIVITY_PROVIDERS,
    ASPECT_RATIO_OPTIONS,
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
    DOMAIN,
    GEMINI_RESOLUTION_OPTIONS,
    IMAGE_PROVIDERS,
    PROVIDER_GEMINI,
    PROVIDER_INVOKEAI,
    PROVIDER_OPENAI,
    PROVIDER_UNSLOTH,
    STEP_ACTIVITY_CREDENTIALS,
    STEP_ACTIVITY_SETTINGS,
    STEP_IMAGE_CREDENTIALS,
    STEP_IMAGE_SETTINGS,
    STEP_USER,
)
from .providers import gemini as gemini_provider
from .providers import invokeai as invokeai_provider
from .providers import openai_compatible as openai_provider
from .providers import unsloth as unsloth_provider
from .providers.base import SelectOption

_LOGGER = logging.getLogger(__name__)
_DISCOVERY_TIMEOUT = 5


def _activity_provider_selector() -> selector.SelectSelector:
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=[
                {"value": PROVIDER_GEMINI, "label": "Gemini (Google cloud)"},
                {"value": PROVIDER_OPENAI, "label": "OpenAI-compatible (Ollama, LM Studio, …)"},
                {"value": PROVIDER_INVOKEAI, "label": "InvokeAI (local)"},
                {"value": PROVIDER_UNSLOTH, "label": "Unsloth Studio (local)"},
            ],
            mode=selector.SelectSelectorMode.DROPDOWN,
        )
    )


def _image_provider_selector() -> selector.SelectSelector:
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=[
                {"value": PROVIDER_GEMINI, "label": "Gemini (Google cloud)"},
                {"value": PROVIDER_INVOKEAI, "label": "InvokeAI (local)"},
                {"value": PROVIDER_UNSLOTH, "label": "Unsloth Studio (local)"},
            ],
            mode=selector.SelectSelectorMode.DROPDOWN,
        )
    )


def _aspect_selector() -> selector.SelectSelector:
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=ASPECT_RATIO_OPTIONS,
            mode=selector.SelectSelectorMode.DROPDOWN,
            custom_value=True,
        )
    )


def _resolution_selector() -> selector.SelectSelector:
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=GEMINI_RESOLUTION_OPTIONS,
            mode=selector.SelectSelectorMode.DROPDOWN,
            custom_value=True,
        )
    )


def _model_selector(options: list[SelectOption]) -> selector.SelectSelector:
    """Dropdown that also allows typing a custom model id."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=options or [{"value": "", "label": "(none found — type a value)"}],
            mode=selector.SelectSelectorMode.DROPDOWN,
            custom_value=True,
        )
    )


def _number_box(
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    step: float = 1,
) -> selector.NumberSelector:
    """Integer/float input box (avoids the default slider when min+max are set)."""
    config: selector.NumberSelectorConfig = {
        "mode": selector.NumberSelectorMode.BOX,
        "step": step,
    }
    if min_value is not None:
        config["min"] = min_value
    if max_value is not None:
        config["max"] = max_value
    return selector.NumberSelector(config)


def _ensure_option(
    options: list[SelectOption], value: str | None
) -> list[SelectOption]:
    """Keep the current saved value selectable even if discovery missed it."""
    if not value:
        return options
    if any(option["value"] == value for option in options):
        return options
    return [{"value": value, "label": f"{value} (saved)"}, *options]


def _default_model(options: list[SelectOption], preferred: str) -> str:
    values = {option["value"] for option in options}
    if preferred in values:
        return preferred
    if options:
        return options[0]["value"]
    return preferred


def _credentials_fields_for_provider(
    provider: str, defaults: dict[str, Any]
) -> dict[Any, Any]:
    """Return voluptuous fields for a single provider's access config."""
    fields: dict[Any, Any] = {}
    if provider == PROVIDER_GEMINI:
        fields[
            vol.Required(
                CONF_GEMINI_API_KEY,
                default=defaults.get(CONF_GEMINI_API_KEY, ""),
            )
        ] = selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        )
    elif provider == PROVIDER_OPENAI:
        fields[
            vol.Required(
                CONF_OPENAI_BASE_URL,
                default=defaults.get(CONF_OPENAI_BASE_URL, DEFAULT_OPENAI_BASE_URL),
            )
        ] = str
        fields[
            vol.Optional(
                CONF_OPENAI_API_KEY,
                default=defaults.get(CONF_OPENAI_API_KEY, ""),
            )
        ] = selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        )
    elif provider == PROVIDER_INVOKEAI:
        fields[
            vol.Required(
                CONF_INVOKEAI_BASE_URL,
                default=defaults.get(
                    CONF_INVOKEAI_BASE_URL, DEFAULT_INVOKEAI_BASE_URL
                ),
            )
        ] = str
        fields[
            vol.Optional(
                CONF_INVOKEAI_API_TOKEN,
                default=defaults.get(CONF_INVOKEAI_API_TOKEN, ""),
            )
        ] = selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        )
    elif provider == PROVIDER_UNSLOTH:
        fields[
            vol.Required(
                CONF_UNSLOTH_BASE_URL,
                default=defaults.get(CONF_UNSLOTH_BASE_URL, DEFAULT_UNSLOTH_BASE_URL),
            )
        ] = str
        fields[
            vol.Optional(
                CONF_UNSLOTH_API_KEY,
                default=defaults.get(CONF_UNSLOTH_API_KEY, ""),
            )
        ] = selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        )
    return fields


async def _async_discover_models(
    hass: HomeAssistant,
    data: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, list[SelectOption]], str]:
    """Fetch model dropdown options for activity and/or image stage."""
    discovered: dict[str, list[SelectOption]] = {
        "gemini_activity": gemini_provider.list_activity_models(),
        "gemini_image": gemini_provider.list_image_models(),
        "openai": [],
        "invokeai_activity": [],
        "invokeai_image": [],
        "unsloth_activity": [],
        "unsloth_image": [],
    }
    notes: list[str] = []

    activity = data.get(CONF_ACTIVITY_PROVIDER)
    image = data.get(CONF_IMAGE_PROVIDER)
    need_activity = stage in {"activity", "all"}
    need_image = stage in {"image", "all"}

    openai_needed = need_activity and activity == PROVIDER_OPENAI
    invokeai_needed = (need_activity and activity == PROVIDER_INVOKEAI) or (
        need_image and image == PROVIDER_INVOKEAI
    )
    unsloth_needed = (need_activity and activity == PROVIDER_UNSLOTH) or (
        need_image and image == PROVIDER_UNSLOTH
    )

    if openai_needed:
        base_url = data.get(CONF_OPENAI_BASE_URL, DEFAULT_OPENAI_BASE_URL)
        api_key = data.get(CONF_OPENAI_API_KEY) or None

        def _openai() -> list[SelectOption]:
            return openai_provider.list_models(
                base_url, api_key, timeout=_DISCOVERY_TIMEOUT
            )

        try:
            discovered["openai"] = await hass.async_add_executor_job(_openai)
        except Exception as err:  # noqa: BLE001 — soft-fail; type models manually
            _LOGGER.warning(
                "Could not discover OpenAI-compatible models from %s: %s",
                base_url,
                err,
            )
            notes.append(
                f"Could not reach OpenAI-compatible server at `{base_url}` — "
                "type model names manually."
            )

    if invokeai_needed:
        base_url = data.get(CONF_INVOKEAI_BASE_URL, DEFAULT_INVOKEAI_BASE_URL)
        token = data.get(CONF_INVOKEAI_API_TOKEN) or None

        def _invokeai() -> dict[str, list[SelectOption]]:
            result: dict[str, list[SelectOption]] = {}
            if need_activity and activity == PROVIDER_INVOKEAI:
                result["invokeai_activity"] = invokeai_provider.list_activity_models(
                    base_url, token, timeout=_DISCOVERY_TIMEOUT
                )
            if need_image and image == PROVIDER_INVOKEAI:
                result["invokeai_image"] = invokeai_provider.list_image_models(
                    base_url, token, timeout=_DISCOVERY_TIMEOUT
                )
            return result

        try:
            discovered.update(await hass.async_add_executor_job(_invokeai))
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning(
                "Could not discover InvokeAI models from %s: %s", base_url, err
            )
            notes.append(
                f"Could not reach InvokeAI at `{base_url}` — type model names manually."
            )

    if unsloth_needed:
        base_url = data.get(CONF_UNSLOTH_BASE_URL, DEFAULT_UNSLOTH_BASE_URL)
        api_key = data.get(CONF_UNSLOTH_API_KEY) or None

        def _unsloth() -> dict[str, list[SelectOption]]:
            result: dict[str, list[SelectOption]] = {}
            if need_activity and activity == PROVIDER_UNSLOTH:
                result["unsloth_activity"] = unsloth_provider.list_chat_models(
                    base_url, api_key, timeout=_DISCOVERY_TIMEOUT
                )
            if need_image and image == PROVIDER_UNSLOTH:
                result["unsloth_image"] = unsloth_provider.list_image_models(
                    base_url, api_key, timeout=_DISCOVERY_TIMEOUT
                )
            return result

        try:
            discovered.update(await hass.async_add_executor_job(_unsloth))
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning(
                "Could not discover Unsloth models from %s: %s", base_url, err
            )
            notes.append(
                f"Could not reach Unsloth Studio at `{base_url}` — type model names manually."
            )

    if not notes and (openai_needed or invokeai_needed or unsloth_needed):
        notes.append(
            "Model lists were loaded from the configured servers. "
            "You can still type a custom model id."
        )

    return discovered, " ".join(notes)


class ForecatsFlowMixin:
    """Shared schema helpers and step navigation for config and options flows."""

    hass: HomeAssistant
    _data: dict[str, Any]
    _discovered: dict[str, list[SelectOption]]

    def _providers_same(self) -> bool:
        return self._data.get(CONF_ACTIVITY_PROVIDER) == self._data.get(
            CONF_IMAGE_PROVIDER
        )

    def _activity_credentials_schema(
        self, defaults: dict[str, Any] | None = None
    ) -> vol.Schema:
        defaults = defaults or {}
        provider = self._data.get(CONF_ACTIVITY_PROVIDER, PROVIDER_GEMINI)
        return vol.Schema(_credentials_fields_for_provider(provider, defaults))

    def _image_credentials_schema(
        self, defaults: dict[str, Any] | None = None
    ) -> vol.Schema:
        defaults = defaults or {}
        provider = self._data.get(CONF_IMAGE_PROVIDER, PROVIDER_GEMINI)
        return vol.Schema(_credentials_fields_for_provider(provider, defaults))

    def _activity_settings_schema(
        self, defaults: dict[str, Any] | None = None
    ) -> vol.Schema:
        defaults = defaults or {}
        discovered = getattr(self, "_discovered", {}) or {}
        fields: dict[Any, Any] = {}
        activity = self._data.get(CONF_ACTIVITY_PROVIDER)

        if activity == PROVIDER_GEMINI:
            options = _ensure_option(
                discovered.get(
                    "gemini_activity", gemini_provider.list_activity_models()
                ),
                defaults.get(CONF_GEMINI_ACTIVITY_MODEL),
            )
            fields[
                vol.Required(
                    CONF_GEMINI_ACTIVITY_MODEL,
                    default=_default_model(
                        options,
                        defaults.get(
                            CONF_GEMINI_ACTIVITY_MODEL, DEFAULT_GEMINI_ACTIVITY_MODEL
                        ),
                    ),
                )
            ] = _model_selector(options)
            fields[
                vol.Required(
                    CONF_GEMINI_ACTIVITY_TIMEOUT,
                    default=defaults.get(
                        CONF_GEMINI_ACTIVITY_TIMEOUT, DEFAULT_GEMINI_ACTIVITY_TIMEOUT
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1))

        elif activity == PROVIDER_OPENAI:
            options = _ensure_option(
                discovered.get("openai", []),
                defaults.get(CONF_OPENAI_ACTIVITY_MODEL),
            )
            fields[
                vol.Required(
                    CONF_OPENAI_ACTIVITY_MODEL,
                    default=_default_model(
                        options,
                        defaults.get(
                            CONF_OPENAI_ACTIVITY_MODEL, DEFAULT_OPENAI_ACTIVITY_MODEL
                        ),
                    ),
                )
            ] = _model_selector(options)
            fields[
                vol.Required(
                    CONF_OPENAI_ACTIVITY_TIMEOUT,
                    default=defaults.get(
                        CONF_OPENAI_ACTIVITY_TIMEOUT, DEFAULT_OPENAI_ACTIVITY_TIMEOUT
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1))

        elif activity == PROVIDER_INVOKEAI:
            options = _ensure_option(
                discovered.get("invokeai_activity", []),
                defaults.get(CONF_INVOKEAI_ACTIVITY_MODEL),
            )
            fields[
                vol.Required(
                    CONF_INVOKEAI_ACTIVITY_MODEL,
                    default=_default_model(
                        options,
                        defaults.get(
                            CONF_INVOKEAI_ACTIVITY_MODEL,
                            DEFAULT_INVOKEAI_ACTIVITY_MODEL,
                        ),
                    ),
                )
            ] = _model_selector(options)
            fields[
                vol.Required(
                    CONF_INVOKEAI_ACTIVITY_TIMEOUT,
                    default=defaults.get(
                        CONF_INVOKEAI_ACTIVITY_TIMEOUT,
                        DEFAULT_INVOKEAI_ACTIVITY_TIMEOUT,
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1))

        elif activity == PROVIDER_UNSLOTH:
            options = _ensure_option(
                discovered.get("unsloth_activity", []),
                defaults.get(CONF_UNSLOTH_ACTIVITY_MODEL),
            )
            fields[
                vol.Required(
                    CONF_UNSLOTH_ACTIVITY_MODEL,
                    default=_default_model(
                        options,
                        defaults.get(
                            CONF_UNSLOTH_ACTIVITY_MODEL,
                            DEFAULT_UNSLOTH_ACTIVITY_MODEL,
                        ),
                    ),
                )
            ] = _model_selector(options)
            fields[
                vol.Required(
                    CONF_UNSLOTH_ACTIVITY_TIMEOUT,
                    default=defaults.get(
                        CONF_UNSLOTH_ACTIVITY_TIMEOUT,
                        DEFAULT_UNSLOTH_ACTIVITY_TIMEOUT,
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1))

        return vol.Schema(fields)

    def _image_settings_schema(
        self, defaults: dict[str, Any] | None = None
    ) -> vol.Schema:
        defaults = defaults or {}
        discovered = getattr(self, "_discovered", {}) or {}
        fields: dict[Any, Any] = {}
        image = self._data.get(CONF_IMAGE_PROVIDER)

        if image == PROVIDER_GEMINI:
            options = _ensure_option(
                discovered.get("gemini_image", gemini_provider.list_image_models()),
                defaults.get(CONF_GEMINI_IMAGE_MODEL),
            )
            fields[
                vol.Required(
                    CONF_GEMINI_IMAGE_MODEL,
                    default=_default_model(
                        options,
                        defaults.get(
                            CONF_GEMINI_IMAGE_MODEL, DEFAULT_GEMINI_IMAGE_MODEL
                        ),
                    ),
                )
            ] = _model_selector(options)
            fields[
                vol.Required(
                    CONF_GEMINI_IMAGE_TIMEOUT,
                    default=defaults.get(
                        CONF_GEMINI_IMAGE_TIMEOUT, DEFAULT_GEMINI_IMAGE_TIMEOUT
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1))
            fields[
                vol.Required(
                    CONF_GEMINI_IMAGE_ASPECT_RATIO,
                    default=defaults.get(
                        CONF_GEMINI_IMAGE_ASPECT_RATIO,
                        DEFAULT_GEMINI_IMAGE_ASPECT_RATIO,
                    ),
                )
            ] = _aspect_selector()
            fields[
                vol.Required(
                    CONF_GEMINI_IMAGE_RESOLUTION,
                    default=defaults.get(
                        CONF_GEMINI_IMAGE_RESOLUTION, DEFAULT_GEMINI_IMAGE_RESOLUTION
                    ),
                )
            ] = _resolution_selector()

        elif image == PROVIDER_INVOKEAI:
            image_options = _ensure_option(
                discovered.get("invokeai_image", []),
                defaults.get(CONF_INVOKEAI_IMAGE_MODEL),
            )
            fields[
                vol.Required(
                    CONF_INVOKEAI_IMAGE_MODEL,
                    default=_default_model(
                        image_options,
                        defaults.get(
                            CONF_INVOKEAI_IMAGE_MODEL, DEFAULT_INVOKEAI_IMAGE_MODEL
                        ),
                    ),
                )
            ] = _model_selector(image_options)
            fields[
                vol.Required(
                    CONF_INVOKEAI_IMAGE_TIMEOUT,
                    default=defaults.get(
                        CONF_INVOKEAI_IMAGE_TIMEOUT, DEFAULT_INVOKEAI_IMAGE_TIMEOUT
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1))
            fields[
                vol.Required(
                    CONF_INVOKEAI_IMAGE_WIDTH,
                    default=defaults.get(
                        CONF_INVOKEAI_IMAGE_WIDTH, DEFAULT_INVOKEAI_IMAGE_WIDTH
                    ),
                )
            ] = _number_box(min_value=64, max_value=4096)
            fields[
                vol.Required(
                    CONF_INVOKEAI_IMAGE_HEIGHT,
                    default=defaults.get(
                        CONF_INVOKEAI_IMAGE_HEIGHT, DEFAULT_INVOKEAI_IMAGE_HEIGHT
                    ),
                )
            ] = _number_box(min_value=64, max_value=4096)
            fields[
                vol.Required(
                    CONF_INVOKEAI_IMAGE_STEPS,
                    default=defaults.get(
                        CONF_INVOKEAI_IMAGE_STEPS, DEFAULT_INVOKEAI_IMAGE_STEPS
                    ),
                )
            ] = _number_box(min_value=1, max_value=200)
            fields[
                vol.Required(
                    CONF_INVOKEAI_IMAGE_GUIDANCE,
                    default=defaults.get(
                        CONF_INVOKEAI_IMAGE_GUIDANCE, DEFAULT_INVOKEAI_IMAGE_GUIDANCE
                    ),
                )
            ] = _number_box(min_value=0.1, max_value=30.0, step=0.1)
            fields[
                vol.Required(
                    CONF_INVOKEAI_IMAGE_SEED,
                    default=defaults.get(
                        CONF_INVOKEAI_IMAGE_SEED, DEFAULT_INVOKEAI_IMAGE_SEED
                    ),
                )
            ] = _number_box(min_value=0)

        elif image == PROVIDER_UNSLOTH:
            image_options = _ensure_option(
                discovered.get("unsloth_image", []),
                defaults.get(CONF_UNSLOTH_IMAGE_MODEL),
            )
            fields[
                vol.Optional(
                    CONF_UNSLOTH_IMAGE_MODEL,
                    default=defaults.get(
                        CONF_UNSLOTH_IMAGE_MODEL, DEFAULT_UNSLOTH_IMAGE_MODEL
                    )
                    or "",
                )
            ] = _model_selector(image_options)
            fields[
                vol.Required(
                    CONF_UNSLOTH_IMAGE_TIMEOUT,
                    default=defaults.get(
                        CONF_UNSLOTH_IMAGE_TIMEOUT, DEFAULT_UNSLOTH_IMAGE_TIMEOUT
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1))
            fields[
                vol.Required(
                    CONF_UNSLOTH_IMAGE_WIDTH,
                    default=defaults.get(
                        CONF_UNSLOTH_IMAGE_WIDTH, DEFAULT_UNSLOTH_IMAGE_WIDTH
                    ),
                )
            ] = _number_box(min_value=64, max_value=4096)
            fields[
                vol.Required(
                    CONF_UNSLOTH_IMAGE_HEIGHT,
                    default=defaults.get(
                        CONF_UNSLOTH_IMAGE_HEIGHT, DEFAULT_UNSLOTH_IMAGE_HEIGHT
                    ),
                )
            ] = _number_box(min_value=64, max_value=4096)
            fields[
                vol.Required(
                    CONF_UNSLOTH_IMAGE_STEPS,
                    default=defaults.get(
                        CONF_UNSLOTH_IMAGE_STEPS, DEFAULT_UNSLOTH_IMAGE_STEPS
                    ),
                )
            ] = _number_box(min_value=1, max_value=200)
            fields[
                vol.Required(
                    CONF_UNSLOTH_IMAGE_GUIDANCE,
                    default=defaults.get(
                        CONF_UNSLOTH_IMAGE_GUIDANCE, DEFAULT_UNSLOTH_IMAGE_GUIDANCE
                    ),
                )
            ] = _number_box(min_value=0.0, max_value=30.0, step=0.1)
            fields[
                vol.Required(
                    CONF_UNSLOTH_IMAGE_SEED,
                    default=defaults.get(
                        CONF_UNSLOTH_IMAGE_SEED, DEFAULT_UNSLOTH_IMAGE_SEED
                    ),
                )
            ] = _number_box(min_value=0)

        return vol.Schema(fields)

    async def _async_goto_after_activity_settings(self) -> FlowResult:
        if self._providers_same():
            return await self.async_step_image_settings()
        return await self.async_step_image_credentials()

    async def async_step_activity_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect access config for the activity provider."""
        schema = self._activity_credentials_schema(self._data)
        if not schema.schema:
            return await self.async_step_activity_settings()

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_activity_settings()

        return self.async_show_form(
            step_id=STEP_ACTIVITY_CREDENTIALS,
            data_schema=schema,
            last_step=False,
            description_placeholders={
                "provider": self._data.get(CONF_ACTIVITY_PROVIDER, "")
            },
        )

    async def async_step_activity_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect activity model/timeout settings."""
        if user_input is not None:
            self._data.update(user_input)
            return await self._async_goto_after_activity_settings()

        self._discovered, discovery_note = await _async_discover_models(
            self.hass, self._data, stage="activity"
        )
        schema = self._activity_settings_schema(self._data)
        if not schema.schema:
            return await self._async_goto_after_activity_settings()

        return self.async_show_form(
            step_id=STEP_ACTIVITY_SETTINGS,
            data_schema=schema,
            last_step=False,
            description_placeholders={"discovery_note": discovery_note},
        )

    async def async_step_image_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect access config for the image provider when it differs."""
        if self._providers_same():
            return await self.async_step_image_settings()

        schema = self._image_credentials_schema(self._data)
        if not schema.schema:
            return await self.async_step_image_settings()

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_image_settings()

        return self.async_show_form(
            step_id=STEP_IMAGE_CREDENTIALS,
            data_schema=schema,
            last_step=False,
            description_placeholders={
                "provider": self._data.get(CONF_IMAGE_PROVIDER, "")
            },
        )

    async def async_step_image_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect image model and generation settings (final step)."""
        raise NotImplementedError


class ForecatsConfigFlow(ForecatsFlowMixin, config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Forecats."""

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._discovered: dict[str, list[SelectOption]] = {}

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> ForecatsOptionsFlow:
        return ForecatsOptionsFlow()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect name and both providers."""
        if user_input is not None:
            if user_input[CONF_ACTIVITY_PROVIDER] not in ACTIVITY_PROVIDERS:
                return self.async_abort(reason="invalid_provider")
            if user_input[CONF_IMAGE_PROVIDER] not in IMAGE_PROVIDERS:
                return self.async_abort(reason="invalid_provider")
            self._data[CONF_NAME] = user_input[CONF_NAME]
            self._data[CONF_ACTIVITY_PROVIDER] = user_input[CONF_ACTIVITY_PROVIDER]
            self._data[CONF_IMAGE_PROVIDER] = user_input[CONF_IMAGE_PROVIDER]
            return await self.async_step_activity_credentials()

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME, default="Cloud Gemini"): str,
                vol.Required(
                    CONF_ACTIVITY_PROVIDER, default=PROVIDER_GEMINI
                ): _activity_provider_selector(),
                vol.Required(
                    CONF_IMAGE_PROVIDER, default=PROVIDER_GEMINI
                ): _image_provider_selector(),
            }
        )
        return self.async_show_form(
            step_id=STEP_USER,
            data_schema=schema,
            last_step=False,
        )

    async def async_step_image_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is not None:
            self._data.update(user_input)
            self._data.pop("output_slug", None)
            return self.async_create_entry(
                title=self._data[CONF_NAME],
                data=self._data,
            )

        self._discovered, discovery_note = await _async_discover_models(
            self.hass, self._data, stage="image"
        )
        return self.async_show_form(
            step_id=STEP_IMAGE_SETTINGS,
            data_schema=self._image_settings_schema(self._data),
            last_step=True,
            description_placeholders={"discovery_note": discovery_note},
        )


class ForecatsOptionsFlow(ForecatsFlowMixin, config_entries.OptionsFlow):
    """Handle options for an existing Forecats instance."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._discovered: dict[str, list[SelectOption]] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        self._data = dict(self.config_entry.data)
        return await self.async_step_user(user_input)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is not None:
            if user_input[CONF_ACTIVITY_PROVIDER] not in ACTIVITY_PROVIDERS:
                return self.async_abort(reason="invalid_provider")
            if user_input[CONF_IMAGE_PROVIDER] not in IMAGE_PROVIDERS:
                return self.async_abort(reason="invalid_provider")
            self._data[CONF_NAME] = user_input[CONF_NAME]
            self._data[CONF_ACTIVITY_PROVIDER] = user_input[CONF_ACTIVITY_PROVIDER]
            self._data[CONF_IMAGE_PROVIDER] = user_input[CONF_IMAGE_PROVIDER]
            return await self.async_step_activity_credentials()

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_NAME, default=self._data.get(CONF_NAME, "Forecats")
                ): str,
                vol.Required(
                    CONF_ACTIVITY_PROVIDER,
                    default=self._data.get(CONF_ACTIVITY_PROVIDER, PROVIDER_GEMINI),
                ): _activity_provider_selector(),
                vol.Required(
                    CONF_IMAGE_PROVIDER,
                    default=self._data.get(CONF_IMAGE_PROVIDER, PROVIDER_GEMINI),
                ): _image_provider_selector(),
            }
        )
        return self.async_show_form(
            step_id=STEP_USER,
            data_schema=schema,
            last_step=False,
        )

    async def async_step_image_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is not None:
            self._data.update(user_input)
            self._data.pop("output_slug", None)
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                title=self._data[CONF_NAME],
                data=self._data,
            )
            return self.async_create_entry(title="", data={})

        self._discovered, discovery_note = await _async_discover_models(
            self.hass, self._data, stage="image"
        )
        return self.async_show_form(
            step_id=STEP_IMAGE_SETTINGS,
            data_schema=self._image_settings_schema(self._data),
            last_step=True,
            description_placeholders={"discovery_note": discovery_note},
        )