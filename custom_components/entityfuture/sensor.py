"""Sensor entities for EntityFuture: probability, accuracy, and discrimination (AUC)."""
from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, MIN_SAMPLES_FOR_AUC
from .coordinator import EntityFutureCoordinator
from .entity import EntityFutureEntity


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: EntityFutureCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            EntityFutureProbabilitySensor(coordinator),
            EntityFutureAccuracySensor(coordinator),
            EntityFutureAucSensor(coordinator),
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
            "label_mode": self.coordinator.label_mode,
            "hold_minutes": self.coordinator.hold_minutes,
            "helper_entities": self.coordinator.helper_entities,
            "training_samples": self.coordinator.total_samples,
        }


class EntityFutureAccuracySensor(EntityFutureEntity, SensorEntity):
    """Rolling accuracy (%) at the configured threshold, next to the trivial baseline.

    Accuracy alone is misleading for rare events (a predictor that always
    says "no" looks great), so `baseline_accuracy` is exposed right next to
    it - the number to beat.
    """

    _attr_translation_key = "accuracy"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:target"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: EntityFutureCoordinator) -> None:
        super().__init__(coordinator, "accuracy")

    @property
    def native_value(self) -> float | None:
        metrics = self.coordinator.current_metrics()
        if metrics is None:
            return None
        return round(metrics["accuracy"] * 100, 1)

    @property
    def extra_state_attributes(self) -> dict:
        attributes: dict[str, Any] = {
            "verified_predictions": self.coordinator.verified_count,
            "pending_predictions": self.coordinator.pending_count,
        }
        metrics = self.coordinator.current_metrics()
        if metrics is not None:
            attributes.update(
                {
                    "baseline_accuracy": round(metrics["baseline_accuracy"] * 100, 1),
                    "yes_rate": round(metrics["positive_rate"] * 100, 1),
                    "precision": _rounded(metrics["precision"]),
                    "recall": _rounded(metrics["recall"]),
                    "flagged_rate": _rounded(metrics["flagged_rate"]),
                }
            )
        return attributes


class EntityFutureAucSensor(EntityFutureEntity, SensorEntity):
    """How well the probabilities separate "yes" from "no" (ROC AUC).

    0.5 means no better than a coin flip, 1.0 is perfect. Unlike accuracy,
    this doesn't get inflated by a rare "yes".
    """

    _attr_translation_key = "auc"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-bell-curve-cumulative"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator: EntityFutureCoordinator) -> None:
        super().__init__(coordinator, "auc")

    def _metrics(self) -> dict[str, Any] | None:
        metrics = self.coordinator.current_metrics()
        if metrics is None or metrics["n"] < MIN_SAMPLES_FOR_AUC:
            return None
        return metrics

    @property
    def native_value(self) -> float | None:
        metrics = self._metrics()
        if metrics is None or metrics["roc_auc"] is None:
            return None
        return round(metrics["roc_auc"], 2)

    @property
    def extra_state_attributes(self) -> dict:
        attributes: dict[str, Any] = {"evaluated_predictions": self.coordinator.verified_count}
        metrics = self._metrics()
        if metrics is not None:
            attributes["brier_skill_score"] = _rounded(metrics["brier_skill_score"])
            attributes["lift"] = _rounded(metrics["lift"], 2)
        if self.coordinator.last_backtest is not None:
            attributes["last_backtest"] = self.coordinator.last_backtest
        return attributes
