"""Shared base entity for EntityFuture platforms."""
from __future__ import annotations

from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceEntryType
from homeassistant.helpers.entity import DeviceInfo, Entity
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN
from .coordinator import EntityFutureCoordinator, signal_update


class EntityFutureEntity(Entity):
    """Base entity tied to one EntityFuture predictor device."""

    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(self, coordinator: EntityFutureCoordinator, unique_id_suffix: str) -> None:
        self.coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{unique_id_suffix}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name=coordinator.entry.title,
            manufacturer="EntityFuture",
            model="Naive Bayes Predictor",
            entry_type=DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                signal_update(self.coordinator.entry.entry_id),
                self._handle_coordinator_update,
            )
        )

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()
