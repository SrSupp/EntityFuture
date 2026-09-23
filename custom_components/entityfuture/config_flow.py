"""Config and options flow for EntityFuture."""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_HELPER_ENTITIES,
    CONF_HORIZON_MINUTES,
    CONF_PREDICTION_THRESHOLD,
    CONF_SAMPLING_INTERVAL_MINUTES,
    CONF_TARGET_ENTITY,
    CONF_TARGET_STATE,
    CONF_USE_RECORDER_HISTORY,
    DEFAULT_PREDICTION_THRESHOLD,
    DEFAULT_SAMPLING_INTERVAL_MINUTES,
    DEFAULT_USE_RECORDER_HISTORY,
    DOMAIN,
    MAX_HORIZON_MINUTES,
    MAX_SAMPLING_INTERVAL_MINUTES,
    MIN_HORIZON_MINUTES,
    MIN_SAMPLING_INTERVAL_MINUTES,
)

CONF_NAME = "name"


class EntityFutureConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle creating a new EntityFuture predictor."""

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> Any:
        """Pick the target entity and prediction horizon."""
        if user_input is not None:
            self._data[CONF_NAME] = user_input[CONF_NAME]
            self._data[CONF_TARGET_ENTITY] = user_input[CONF_TARGET_ENTITY]
            self._data[CONF_HORIZON_MINUTES] = int(user_input[CONF_HORIZON_MINUTES])
            return await self.async_step_target_state()

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): selector.TextSelector(),
                vol.Required(CONF_TARGET_ENTITY): selector.EntitySelector(),
                vol.Required(CONF_HORIZON_MINUTES, default=30): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=MIN_HORIZON_MINUTES,
                        max=MAX_HORIZON_MINUTES,
                        mode=selector.NumberSelectorMode.BOX,
                        unit_of_measurement="min",
                    )
                ),
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema)

    async def async_step_target_state(
        self, user_input: dict[str, Any] | None = None
    ) -> Any:
        """Ask which state of the target entity counts as the "yes" outcome."""
        target_entity_id = self._data[CONF_TARGET_ENTITY]
        current_state = self.hass.states.get(target_entity_id)
        current_state_value = current_state.state if current_state else None

        errors: dict[str, str] = {}
        if user_input is not None:
            target_state = user_input[CONF_TARGET_STATE].strip()
            if not target_state:
                errors["base"] = "target_state_required"
            else:
                self._data[CONF_TARGET_STATE] = target_state
                await self.async_set_unique_id(
                    f"{target_entity_id}_{self._data[CONF_HORIZON_MINUTES]}"
                )
                self._abort_if_unique_id_configured()
                title = self._data.pop(CONF_NAME)
                return self.async_create_entry(title=title, data=self._data)

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_TARGET_STATE, default=current_state_value or ""
                ): selector.TextSelector(),
            }
        )
        return self.async_show_form(
            step_id="target_state",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "entity_id": target_entity_id,
                "current_state": current_state_value or "unbekannt / unknown",
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return EntityFutureOptionsFlow()


class EntityFutureOptionsFlow(OptionsFlow):
    """Let the user add helper entities and tune sampling after setup."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> Any:
        errors: dict[str, str] = {}
        target_entity_id = self.config_entry.data[CONF_TARGET_ENTITY]

        if user_input is not None:
            helper_entities = user_input.get(CONF_HELPER_ENTITIES, [])
            if target_entity_id in helper_entities:
                errors["base"] = "target_is_helper"
            else:
                return self.async_create_entry(data=user_input)

        options = self.config_entry.options
        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_HELPER_ENTITIES,
                    default=options.get(CONF_HELPER_ENTITIES, []),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(multiple=True)
                ),
                vol.Optional(
                    CONF_SAMPLING_INTERVAL_MINUTES,
                    default=options.get(
                        CONF_SAMPLING_INTERVAL_MINUTES,
                        DEFAULT_SAMPLING_INTERVAL_MINUTES,
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=MIN_SAMPLING_INTERVAL_MINUTES,
                        max=MAX_SAMPLING_INTERVAL_MINUTES,
                        mode=selector.NumberSelectorMode.BOX,
                        unit_of_measurement="min",
                    )
                ),
                vol.Optional(
                    CONF_PREDICTION_THRESHOLD,
                    default=options.get(
                        CONF_PREDICTION_THRESHOLD, DEFAULT_PREDICTION_THRESHOLD
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1, max=99, mode=selector.NumberSelectorMode.SLIDER, unit_of_measurement="%"
                    )
                ),
                vol.Optional(
                    CONF_USE_RECORDER_HISTORY,
                    default=options.get(
                        CONF_USE_RECORDER_HISTORY, DEFAULT_USE_RECORDER_HISTORY
                    ),
                ): selector.BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)
