"""The EntityFuture integration."""
from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .const import (
    ATTR_CONFIG_ENTRY_ID,
    CONF_HELPER_ENTITIES,
    CONF_INCLUDE_TARGET_STATE_FEATURE,
    CONF_LABEL_MODE,
    DOMAIN,
    LABEL_MODE_EXACT,
    LABEL_MODE_WITHIN,
    MAX_TARGET_HOLD_MINUTES,
    PLATFORMS,
    SERVICE_BACKTEST,
    SERVICE_RESET_MODEL,
)
from .coordinator import EntityFutureCoordinator

_LOGGER = logging.getLogger(__name__)

RESET_MODEL_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Optional("rerun_warmstart", default=False): cv.boolean,
    }
)

BACKTEST_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Optional("days"): vol.All(vol.Coerce(int), vol.Range(min=1, max=30)),
        vol.Optional(CONF_LABEL_MODE): vol.In([LABEL_MODE_WITHIN, LABEL_MODE_EXACT]),
        vol.Optional("hold_minutes"): vol.All(
            vol.Coerce(int), vol.Range(min=0, max=MAX_TARGET_HOLD_MINUTES)
        ),
        vol.Optional(CONF_INCLUDE_TARGET_STATE_FEATURE): cv.boolean,
        vol.Optional(CONF_HELPER_ENTITIES): cv.entity_ids,
        vol.Optional("threshold"): vol.All(vol.Coerce(int), vol.Range(min=1, max=99)),
    }
)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up EntityFuture from a config entry."""
    coordinator = EntityFutureCoordinator(hass, entry)
    await coordinator.async_setup()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    _async_register_services(hass)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        coordinator: EntityFutureCoordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_shutdown()
    return unload_ok


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the entry when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)


def _get_coordinator(hass: HomeAssistant, entry_id: str) -> EntityFutureCoordinator:
    coordinator: EntityFutureCoordinator | None = hass.data.get(DOMAIN, {}).get(entry_id)
    if coordinator is None:
        raise HomeAssistantError(f"No EntityFuture predictor found for config entry {entry_id}")
    return coordinator


def _async_register_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, SERVICE_RESET_MODEL):
        return

    async def _async_handle_reset(call: ServiceCall) -> None:
        coordinator = _get_coordinator(hass, call.data[ATTR_CONFIG_ENTRY_ID])
        await coordinator.async_reset(rerun_warmstart=call.data["rerun_warmstart"])

    async def _async_handle_backtest(call: ServiceCall) -> ServiceResponse:
        coordinator = _get_coordinator(hass, call.data[ATTR_CONFIG_ENTRY_ID])
        return await coordinator.async_run_backtest(
            days=call.data.get("days"),
            label_mode=call.data.get(CONF_LABEL_MODE),
            hold_minutes=call.data.get("hold_minutes"),
            include_target_state_feature=call.data.get(CONF_INCLUDE_TARGET_STATE_FEATURE),
            helper_entities=call.data.get(CONF_HELPER_ENTITIES),
            threshold=call.data.get("threshold"),
        )

    hass.services.async_register(
        DOMAIN, SERVICE_RESET_MODEL, _async_handle_reset, schema=RESET_MODEL_SCHEMA
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_BACKTEST,
        _async_handle_backtest,
        schema=BACKTEST_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
