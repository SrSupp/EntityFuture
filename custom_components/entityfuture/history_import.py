"""Optional warm-start: pre-train a predictor from existing recorder history.

This is best-effort. If the recorder is disabled, has no data yet, or
anything goes wrong, the caller (coordinator) simply continues without a
warm start and learns live from now on.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State
import homeassistant.util.dt as dt_util

from .const import (
    FEATURE_TARGET_CURRENT_STATE,
    FEATURE_TIME_BUCKET,
    FEATURE_WEEKDAY,
    HISTORY_IMPORT_LOOKBACK_DAYS,
    STATE_UNKNOWN_VALUE,
    TIME_BUCKET_MINUTES,
)

if TYPE_CHECKING:
    from .coordinator import EntityFutureCoordinator

_LOGGER = logging.getLogger(__name__)


def _state_at(states: list[State], at: datetime) -> str | None:
    """Return the entity's state active at time `at`, or None if unknown."""
    result: str | None = None
    for state in states:
        if state.last_changed > at:
            break
        result = state.state
    return result


async def async_import_history(hass: HomeAssistant, coordinator: EntityFutureCoordinator) -> int:
    """Reconstruct (features, label) training pairs from recorder history.

    Returns the number of samples learned.
    """
    from homeassistant.components.recorder import get_instance
    from homeassistant.components.recorder.history import get_significant_states

    horizon = coordinator.horizon
    sampling_interval = coordinator.sampling_interval
    if horizon <= timedelta(0) or sampling_interval <= timedelta(0):
        return 0

    entity_ids = [coordinator.target_entity, *coordinator.helper_entities]
    end_time = dt_util.utcnow()
    start_time = end_time - timedelta(days=HISTORY_IMPORT_LOOKBACK_DAYS)

    recorder_instance = get_instance(hass)

    def _fetch() -> dict[str, list[State]]:
        return get_significant_states(
            hass,
            start_time,
            end_time,
            entity_ids,
            include_start_time_state=True,
            significant_changes_only=False,
            no_attributes=True,
        )

    states_by_entity = await recorder_instance.async_add_executor_job(_fetch)

    target_states = states_by_entity.get(coordinator.target_entity)
    if not target_states:
        _LOGGER.debug(
            "EntityFuture %s: no recorder history for %s, skipping warm start",
            coordinator.entry.title,
            coordinator.target_entity,
        )
        return 0

    helper_states = {
        entity_id: states_by_entity.get(entity_id, [])
        for entity_id in coordinator.helper_entities
    }

    def _learn() -> int:
        learned = 0
        target_state_value = coordinator.target_state
        current = start_time
        while current + horizon <= end_time:
            verify_time = current + horizon
            actual = _state_at(target_states, verify_time)
            if actual is not None and actual not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                local_now = dt_util.as_local(current)
                minute_of_day = local_now.hour * 60 + local_now.minute
                features = {
                    FEATURE_WEEKDAY: str(local_now.weekday()),
                    FEATURE_TIME_BUCKET: str(minute_of_day // TIME_BUCKET_MINUTES),
                    FEATURE_TARGET_CURRENT_STATE: (
                        _state_at(target_states, current) or STATE_UNKNOWN_VALUE
                    ),
                }
                for entity_id, states in helper_states.items():
                    features[entity_id] = _state_at(states, current) or STATE_UNKNOWN_VALUE

                label = 1 if actual == target_state_value else 0
                coordinator.model.learn_one(features, label)
                learned += 1
            current += sampling_interval
        return learned

    return await recorder_instance.async_add_executor_job(_learn)
