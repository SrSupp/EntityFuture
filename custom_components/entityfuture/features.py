"""Feature and label logic shared by the live coordinator and the history replay.

Nothing in here imports Home Assistant, so it can be unit tested directly.
"State" objects are duck-typed: anything with `.state` and `.last_changed`.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta
from typing import Any

from .const import (
    FEATURE_TARGET_CURRENT_STATE,
    FEATURE_TARGET_RECENCY,
    FEATURE_TIME_BUCKET,
    FEATURE_WEEKDAY,
    LABEL_MODE_EXACT,
    RECENCY_RECENT,
    RECENCY_STABLE,
    RECENCY_SUFFIX,
    STATE_UNKNOWN_VALUE,
    TIME_BUCKET_MINUTES,
)

_UNAVAILABLE_STATES = ("unavailable", "unknown")


def _changed(state: Any) -> datetime:
    return state.last_changed


def state_object_at(states: list[Any], at: datetime) -> Any | None:
    """Return the state object active at `at` (states sorted by last_changed)."""
    index = bisect_right(states, at, key=_changed) - 1
    return states[index] if index >= 0 else None


def state_value_at(states: list[Any], at: datetime) -> str | None:
    state = state_object_at(states, at)
    return state.state if state else None


def feature_pair(state: Any | None, now: datetime, window: timedelta) -> tuple[str, str]:
    """Return (value, recency) for one entity.

    `recency` is "recent" if the state changed within `window` of `now`,
    otherwise "stable" - so a fresh transition (a blind that was *just*
    opened) is a different signal from a state that was simply held.
    Unavailable/unknown/missing states are normalized to one value so live
    data and replayed history look the same to the model.
    """
    if state is None or state.state in _UNAVAILABLE_STATES:
        return STATE_UNKNOWN_VALUE, STATE_UNKNOWN_VALUE
    if state.last_changed is None:
        return state.state, STATE_UNKNOWN_VALUE
    recency = RECENCY_RECENT if (now - state.last_changed) <= window else RECENCY_STABLE
    return state.state, recency


def time_features(local_now: datetime) -> dict[str, str]:
    minute_of_day = local_now.hour * 60 + local_now.minute
    return {
        FEATURE_WEEKDAY: str(local_now.weekday()),
        FEATURE_TIME_BUCKET: str(minute_of_day // TIME_BUCKET_MINUTES),
    }


def make_features(
    local_now: datetime,
    target_pair: tuple[str, str] | None,
    helper_pairs: dict[str, tuple[str, str]],
) -> dict[str, str]:
    """Build the model's feature dict. `target_pair=None` omits the target's own state."""
    features = time_features(local_now)
    if target_pair is not None:
        features[FEATURE_TARGET_CURRENT_STATE], features[FEATURE_TARGET_RECENCY] = target_pair
    for entity_id, (value, recency) in helper_pairs.items():
        features[entity_id] = value
        features[f"{entity_id}{RECENCY_SUFFIX}"] = recency
    return features


def label_window(
    mode: str, horizon: timedelta, sampling_interval: timedelta
) -> tuple[timedelta, timedelta]:
    """Return (start_offset, end_offset) of the "yes" window relative to a snapshot at T."""
    if mode == LABEL_MODE_EXACT:
        return horizon, horizon + sampling_interval
    return timedelta(0), horizon


def any_state_active_in_window(
    states: list[Any],
    start: datetime,
    end: datetime,
    target_value: str,
    hold: timedelta = timedelta(0),
) -> bool:
    """Return whether `target_value` was active at any point in [start, end).

    With `hold`, each active stretch counts as lasting `hold` longer, so
    pulses separated by less than that merge into one.
    """
    first = max(bisect_right(states, start - hold, key=_changed) - 1, 0)
    for index in range(first, len(states)):
        state = states[index]
        if state.last_changed >= end:
            break
        if state.state != target_value:
            continue
        interval_end = states[index + 1].last_changed if index + 1 < len(states) else None
        if interval_end is None or interval_end + hold > start:
            return True
    return False


def find_session_starts(
    states: list[Any], target_value: str, hold: timedelta, gap: timedelta
) -> list[datetime]:
    """Return the times the target became active after at least `gap` of quiet."""
    starts: list[datetime] = []
    for index, state in enumerate(states):
        if index == 0 or state.state != target_value:
            continue
        if states[index - 1].state == target_value:
            continue
        moment = state.last_changed
        if not any_state_active_in_window(states, moment - gap, moment, target_value, hold):
            starts.append(moment)
    return starts
