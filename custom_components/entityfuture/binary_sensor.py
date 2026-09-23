"""Binary sensor entity for EntityFuture: thresholded yes/no prediction."""
from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import EntityFutureCoordinator
from .entity import EntityFutureEntity


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: EntityFutureCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([EntityFuturePredictionBinarySensor(coordinator)])


class EntityFuturePredictionBinarySensor(EntityFutureEntity, BinarySensorEntity):
    """On if the predicted probability is at/above the configured threshold."""

    _attr_translation_key = "prediction"
    _attr_icon = "mdi:crystal-ball"

    def __init__(self, coordinator: EntityFutureCoordinator) -> None:
        super().__init__(coordinator, "prediction")

    @property
    def is_on(self) -> bool | None:
        probability = self.coordinator.current_probability()
        if probability is None:
            return None
        return (probability * 100) >= self.coordinator.prediction_threshold

    @property
    def extra_state_attributes(self) -> dict:
        return {
            "prediction_threshold": self.coordinator.prediction_threshold,
            "training_samples": self.coordinator.total_samples,
        }
