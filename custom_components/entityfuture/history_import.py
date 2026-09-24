"""Recorder-backed helpers: warm start, and window-based label checks.

Warm start is best-effort. If the recorder is disabled, has no data yet,
or anything goes wrong, the caller (coordinator) simply continues without
a warm start and learns live from now on.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING

from homeassistant.core import HomeAssistant, State
import homeassistant.util.dt as dt_util

from .const import (
    FEATURE_TARGET_CURRENT_STATE,
    FEATURE_TARGET_RECENCY,
    FEATURE_TIME_BUCKET,
    FEATURE_WEEKDAY,
    HISTORY_IMPORT_LOOKBACK_DAYS,
    MIN_SAMPLES_FOR_WARMSTART_EVAL,
    RECENCY_RECENT,
    RECENCY_STABLE,
    RECENCY_SUFFIX,
    STATE_UNKNOWN_VALUE,
    TIME_BUCKET_MINUTES,
    WARMSTART_EVAL_FRACTION,
)

if TYPE_CHECKING:
    from .coordinator import EntityFutureCoordinator

_LOGGER = logging.getLogger(__name__)


def _state_object_at(states: list[State], at: datetime) -> State | None:
    """Return the State object active at time `at`, or None if unknown."""
    result: State | None = None
    for state in states:
        if state.last_changed > at:
            break
        result = state
    return result


def _state_at(states: list[State], at: datetime) -> str | None:
    """Return the entity's state value active at time `at`, or None."""
    state = _state_object_at(states, at)
    return state.state if state else None


def _recency_at(states: list[State], at: datetime, window: timedelta) -> str:
    """Return "recent" if the state active at `at` started within `window`."""
    state = _state_object_at(states, at)
    if state is None or state.last_changed is None:
        return STATE_UNKNOWN_VALUE
    return RECENCY_RECENT if (at - state.last_changed) <= window else RECENCY_STABLE


def _any_state_active_in_window(
    states: list[State], start: datetime, end: datetime, target_value: str
) -> bool:
    """Return whether `target_value` was active at any point in [start, end)."""
    for index, state in enumerate(states):
        if state.last_changed >= end:
            break
        if state.state != target_value:
            continue
        interval_end = states[index + 1].last_changed if index + 1 < len(states) else None
        if interval_end is None or interval_end > start:
            return True
    return False


async def async_check_target_state_in_window(
    hass: HomeAssistant,
    entity_id: str,
    window_start: datetime,
    window_end: datetime,
    target_value: str,
) -> bool | None:
    """Check via the recorder whether `entity_id` held `target_value` at any
    point during [window_start, window_end).

    Returns None if this could not be determined (recorder disabled, query
    failed, or no history at all for the entity) - the caller should then
    fall back to an instantaneous state check.
    """
    from homeassistant.components.recorder import get_instance
    from homeassistant.components.recorder.history import get_significant_states

    try:
        recorder_instance = get_instance(hass)

        def _fetch() -> dict[str, list[State]]:
            return get_significant_states(
                hass,
                window_start,
                window_end,
                [entity_id],
                include_start_time_state=True,
                significant_changes_only=False,
                no_attributes=True,
            )

        states_by_entity = await recorder_instance.async_add_executor_job(_fetch)
    except Exception:  # noqa: BLE001 - this must never break live verification
        _LOGGER.debug(
            "EntityFuture: recorder window check failed for %s", entity_id, exc_info=True
        )
        return None

    states = states_by_entity.get(entity_id)
    if not states:
        return None

    return _any_state_active_in_window(states, window_start, window_end, target_value)


async def async_import_history(
    hass: HomeAssistant, coordinator: EntityFutureCoordinator
) -> tuple[int, int]:
    """Reconstruct (features, label) training pairs from recorder history.

    A pair's label reflects whether the target state occurred at any point
    during the sampling-interval-wide window starting at `t + horizon`
    (not just a single instant), matching how live verification works.

    Uses a chronological train/test split: the earliest pairs are learned
    directly, and the most recent slice is evaluated walk-forward style
    (predict, then learn) so the accuracy counter already has an honest,
    held-out estimate right after setup instead of staying empty until
    live predictions start coming in.

    Returns (total_pairs_learned, pairs_used_for_the_accuracy_backtest).
    """
    from homeassistant.components.recorder import get_instance
    from homeassistant.components.recorder.history import get_significant_states

    horizon = coordinator.horizon
    sampling_interval = coordinator.sampling_interval
    if horizon <= timedelta(0) or sampling_interval <= timedelta(0):
        return 0, 0

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
        return 0, 0

    helper_states = {
        entity_id: states_by_entity.get(entity_id, [])
        for entity_id in coordinator.helper_entities
    }

    def _build_pairs() -> list[tuple[dict[str, str], int]]:
        target_state_value = coordinator.target_state
        pairs: list[tuple[dict[str, str], int]] = []
        current = start_time
        while current + horizon + sampling_interval <= end_time:
            window_start = current + horizon
            window_end = window_start + sampling_interval

            local_now = dt_util.as_local(current)
            minute_of_day = local_now.hour * 60 + local_now.minute
            features = {
                FEATURE_WEEKDAY: str(local_now.weekday()),
                FEATURE_TIME_BUCKET: str(minute_of_day // TIME_BUCKET_MINUTES),
                FEATURE_TARGET_CURRENT_STATE: (
                    _state_at(target_states, current) or STATE_UNKNOWN_VALUE
                ),
                FEATURE_TARGET_RECENCY: _recency_at(target_states, current, sampling_interval),
            }
            for entity_id, states in helper_states.items():
                features[entity_id] = _state_at(states, current) or STATE_UNKNOWN_VALUE
                features[f"{entity_id}{RECENCY_SUFFIX}"] = _recency_at(
                    states, current, sampling_interval
                )

            label = 1 if _any_state_active_in_window(
                target_states, window_start, window_end, target_state_value
            ) else 0
            pairs.append((features, label))
            current += sampling_interval
        return pairs

    def _learn(pairs: list[tuple[dict[str, str], int]]) -> tuple[int, int]:
        total = len(pairs)
        if total == 0:
            return 0, 0

        eval_count = 0
        if total >= MIN_SAMPLES_FOR_WARMSTART_EVAL:
            eval_count = min(int(total * WARMSTART_EVAL_FRACTION), total // 2)
        train_only_count = total - eval_count

        for features, label in pairs[:train_only_count]:
            coordinator.model.learn_one(features, label)

        for features, label in pairs[train_only_count:]:
            predicted = coordinator.model.predict_proba(features)
            if predicted is not None:
                predicted_label = 1 if predicted >= 0.5 else 0
                coordinator.accuracy_window.append(1 if predicted_label == label else 0)
            coordinator.model.learn_one(features, label)

        return total, eval_count

    pairs = await recorder_instance.async_add_executor_job(_build_pairs)
    return await recorder_instance.async_add_executor_job(_learn, pairs)
