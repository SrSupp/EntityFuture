"""The EntityFuture integration."""
from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .const import ATTR_CONFIG_ENTRY_ID, DOMAIN, PLATFORMS, SERVICE_RESET_MODEL
from .coordinator import EntityFutureCoordinator

_LOGGER = logging.getLogger(__name__)

RESET_MODEL_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Optional("rerun_warmstart", default=False): cv.boolean,
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


def _async_register_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, SERVICE_RESET_MODEL):
        return

    async def _async_handle_reset(call: ServiceCall) -> None:
        entry_id = call.data[ATTR_CONFIG_ENTRY_ID]
        coordinator: EntityFutureCoordinator | None = hass.data.get(DOMAIN, {}).get(
            entry_id
        )
        if coordinator is None:
            raise HomeAssistantError(
                f"No EntityFuture predictor found for config entry {entry_id}"
            )
        await coordinator.async_reset(rerun_warmstart=call.data["rerun_warmstart"])

    hass.services.async_register(
        DOMAIN, SERVICE_RESET_MODEL, _async_handle_reset, schema=RESET_MODEL_SCHEMA
    )
