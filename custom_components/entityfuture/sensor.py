"""Sensor entities for EntityFuture: probability and rolling accuracy."""
from __future__ import annotations

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import EntityFutureCoordinator
from .entity import EntityFutureEntity


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: EntityFutureCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            EntityFutureProbabilitySensor(coordinator),
            EntityFutureAccuracySensor(coordinator),
        ]
    )


class EntityFutureProbabilitySensor(EntityFutureEntity, SensorEntity):
    """Probability (%) that the target entity will match its target state."""

    _attr_translation_key = "probability"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:crystal-ball"

    def __init__(self, coordinator: EntityFutureCoordinator) -> None:
        super().__init__(coordinator, "probability")

    @property
    def native_value(self) -> float | None:
        probability = self.coordinator.current_probability()
        if probability is None:
            return None
        return round(probability * 100, 1)

    @property
    def extra_state_attributes(self) -> dict:
        return {
            "target_entity": self.coordinator.target_entity,
            "target_state": self.coordinator.target_state,
            "horizon_minutes": int(self.coordinator.horizon.total_seconds() // 60),
            "helper_entities": self.coordinator.helper_entities,
            "training_samples": self.coordinator.total_samples,
        }


class EntityFutureAccuracySensor(EntityFutureEntity, SensorEntity):
    """Rolling accuracy (%) of past predictions, verified against reality."""

    _attr_translation_key = "accuracy"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:target"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: EntityFutureCoordinator) -> None:
        super().__init__(coordinator, "accuracy")

    @property
    def native_value(self) -> float | None:
        accuracy = self.coordinator.current_accuracy()
        if accuracy is None:
            return None
        return round(accuracy, 1)

    @property
    def extra_state_attributes(self) -> dict:
        return {
            "verified_predictions": self.coordinator.verified_count,
            "pending_predictions": self.coordinator.pending_count,
        }
