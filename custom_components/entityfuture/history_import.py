"""Recorder-backed helpers: warm start, live window checks, and the backtest.

The actual feature/label/evaluation logic lives in features.py and
backtest.py (no Home Assistant imports); this module only talks to the
recorder and hands the data over.

Warm start is best-effort. If the recorder is disabled, has no data yet,
or anything goes wrong, the caller (coordinator) simply continues without
a warm start and learns live from now on.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant, State
import homeassistant.util.dt as dt_util

from .backtest import build_samples, evaluate, walk_forward
from .const import (
    BACKTEST_BURN_IN_FRACTION,
    HISTORY_IMPORT_LOOKBACK_DAYS,
    LAPLACE_ALPHA,
    MIN_SAMPLES_FOR_BACKTEST,
    MIN_SAMPLES_FOR_WARMSTART_EVAL,
    SESSION_GAP_MINUTES,
    WARMSTART_EVAL_FRACTION,
)
from .features import any_state_active_in_window, find_session_starts

if TYPE_CHECKING:
    from .coordinator import EntityFutureCoordinator

_LOGGER = logging.getLogger(__name__)


async def _async_fetch_states(
    hass: HomeAssistant, entity_ids: list[str], start: datetime, end: datetime
) -> tuple[Any, dict[str, list[State]]]:
    """Fetch state history from the recorder; returns (recorder_instance, states)."""
    from homeassistant.components.recorder import get_instance
    from homeassistant.components.recorder.history import get_significant_states

    recorder_instance = get_instance(hass)

    def _fetch() -> dict[str, list[State]]:
        return get_significant_states(
            hass,
            start,
            end,
            entity_ids,
            include_start_time_state=True,
            significant_changes_only=False,
            no_attributes=True,
        )

    return recorder_instance, await recorder_instance.async_add_executor_job(_fetch)


async def async_check_target_state_in_window(
    hass: HomeAssistant,
    entity_id: str,
    window_start: datetime,
    window_end: datetime,
    target_value: str,
    hold: timedelta = timedelta(0),
) -> bool | None:
    """Check via the recorder whether `entity_id` held `target_value` at any
    point during [window_start, window_end), counting `hold` after each
    active stretch.

    Returns None if this could not be determined (recorder disabled, query
    failed, or no history at all for the entity) - the caller should then
    fall back to an instantaneous state check.
    """
    try:
        # Look back by `hold` so an active stretch that ended just before the
        # window still counts.
        _, states_by_entity = await _async_fetch_states(
            hass, [entity_id], window_start - hold, window_end
        )
    except Exception:  # noqa: BLE001 - this must never break live verification
        _LOGGER.debug(
            "EntityFuture: recorder window check failed for %s", entity_id, exc_info=True
        )
        return None

    states = states_by_entity.get(entity_id)
    if not states:
        return None

    return any_state_active_in_window(states, window_start, window_end, target_value, hold)


async def async_import_history(
    hass: HomeAssistant, coordinator: EntityFutureCoordinator
) -> tuple[int, int]:
    """Warm start: replay recorder history into the coordinator's model.

    A sample's label follows the coordinator's label mode and hold time, so
    it means exactly what a live verification would mean.

    Uses a chronological train/test split: the earliest samples are learned
    directly, and the most recent slice is evaluated walk-forward style
    (predict, then learn) and recorded in the coordinator's evaluation
    window, so the quality metrics already have an honest, held-out estimate
    right after setup.

    Returns (total_samples_learned, samples_used_for_the_backtest_slice).
    """
    horizon = coordinator.horizon
    sampling_interval = coordinator.sampling_interval
    if horizon <= timedelta(0) or sampling_interval <= timedelta(0):
        return 0, 0

    entity_ids = [coordinator.target_entity, *coordinator.helper_entities]
    end_time = dt_util.utcnow()
    start_time = end_time - timedelta(days=HISTORY_IMPORT_LOOKBACK_DAYS)

    recorder_instance, states_by_entity = await _async_fetch_states(
        hass, entity_ids, start_time, end_time
    )

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

    def _work() -> tuple[int, int]:
        samples = build_samples(
            target_states=target_states,
            helper_states=helper_states,
            target_state=coordinator.target_state,
            horizon=horizon,
            sampling_interval=sampling_interval,
            label_mode=coordinator.label_mode,
            hold=coordinator.hold,
            include_target=coordinator.include_target_state_feature,
            start=start_time,
            end=end_time,
            to_local=dt_util.as_local,
        )
        total = len(samples)
        if total == 0:
            return 0, 0

        eval_count = 0
        if total >= MIN_SAMPLES_FOR_WARMSTART_EVAL:
            eval_count = min(int(total * WARMSTART_EVAL_FRACTION), total // 2)
        train_only_count = total - eval_count

        for sample in samples[:train_only_count]:
            coordinator.model.learn_one(sample.features, sample.label)

        for sample in samples[train_only_count:]:
            predicted = coordinator.model.predict_proba(sample.features)
            if predicted is not None:
                coordinator.eval_window.append((predicted, sample.label))
            coordinator.model.learn_one(sample.features, sample.label)

        return total, eval_count

    return await recorder_instance.async_add_executor_job(_work)


async def async_run_backtest(
    hass: HomeAssistant,
    coordinator: EntityFutureCoordinator,
    *,
    days: int,
    label_mode: str,
    hold_minutes: int,
    include_target: bool,
    helper_entities: list[str],
    threshold_pct: int,
) -> dict[str, Any]:
    """Replay up to `days` of history and report how good the predictions would have been.

    Every parameter can differ from the live configuration, so alternative
    setups (other helper entities, label mode, ...) can be compared without
    touching the running predictor or waiting days for live data.
    """
    horizon = coordinator.horizon
    sampling_interval = coordinator.sampling_interval
    hold = timedelta(minutes=hold_minutes)
    end_time = dt_util.utcnow()
    start_time = end_time - timedelta(days=days)

    parameters = {
        "days": days,
        "label_mode": label_mode,
        "hold_minutes": hold_minutes,
        "include_target_state_feature": include_target,
        "helper_entities": helper_entities,
        "threshold_pct": threshold_pct,
        "horizon_minutes": int(horizon.total_seconds() // 60),
        "sampling_interval_minutes": int(sampling_interval.total_seconds() // 60),
    }

    recorder_instance, states_by_entity = await _async_fetch_states(
        hass, [coordinator.target_entity, *helper_entities], start_time, end_time
    )
    target_states = states_by_entity.get(coordinator.target_entity)
    if not target_states:
        return {"parameters": parameters, "error": "no recorder history for the target entity"}

    helper_states = {
        entity_id: states_by_entity.get(entity_id, []) for entity_id in helper_entities
    }

    def _work() -> dict[str, Any]:
        samples = build_samples(
            target_states=target_states,
            helper_states=helper_states,
            target_state=coordinator.target_state,
            horizon=horizon,
            sampling_interval=sampling_interval,
            label_mode=label_mode,
            hold=hold,
            include_target=include_target,
            start=start_time,
            end=end_time,
            to_local=dt_util.as_local,
        )
        if len(samples) < MIN_SAMPLES_FOR_BACKTEST:
            return {
                "error": f"not enough history ({len(samples)} samples, need {MIN_SAMPLES_FOR_BACKTEST})"
            }

        rows = walk_forward(
            samples, alpha=LAPLACE_ALPHA, burn_in_fraction=BACKTEST_BURN_IN_FRACTION
        )
        if not rows:
            return {"error": "no samples left after the burn-in period"}

        sessions = [
            moment
            for moment in find_session_starts(
                target_states,
                coordinator.target_state,
                hold,
                timedelta(minutes=SESSION_GAP_MINUTES),
            )
            if moment >= rows[0].t
        ]
        return evaluate(
            rows,
            threshold=threshold_pct / 100,
            sessions=sessions,
            horizon=horizon,
            sampling_interval=sampling_interval,
        )

    result = await recorder_instance.async_add_executor_job(_work)
    return {"parameters": parameters, **result}
