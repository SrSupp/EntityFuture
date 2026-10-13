"""Replay history through the model and report how good it really is.

Pure Python (no Home Assistant imports). The recorder fetch and the service
glue live in history_import.py / __init__.py.

The evaluation is walk-forward ("predict, then learn"): every prediction is
made using only what the model could have known at that moment, exactly like
live operation - no peeking at the future.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .const import BACKTEST_THRESHOLDS
from .features import (
    any_state_active_in_window,
    feature_pair,
    label_window,
    make_features,
    state_object_at,
    time_features,
)
from .metrics import brier_skill_score, roc_auc, summarize
from .model import NaiveBayesModel


@dataclass(frozen=True)
class Sample:
    t: datetime
    features: dict[str, str]
    time_features: dict[str, str]
    label: int
    currently_active: int


@dataclass(frozen=True)
class EvalRow:
    t: datetime
    p_full: float
    p_time_only: float
    currently_active: int
    label: int


def build_samples(
    *,
    target_states: list[Any],
    helper_states: dict[str, list[Any]],
    target_state: str,
    horizon: timedelta,
    sampling_interval: timedelta,
    label_mode: str,
    hold: timedelta,
    include_target: bool,
    start: datetime,
    end: datetime,
    to_local: Callable[[datetime], datetime],
) -> list[Sample]:
    """Reconstruct one (features, label) sample per sampling interval."""
    window_start, window_end = label_window(label_mode, horizon, sampling_interval)
    one_second = timedelta(seconds=1)
    samples: list[Sample] = []

    current = start
    while current + window_end <= end:
        local_now = to_local(current)
        target_pair = (
            feature_pair(state_object_at(target_states, current), current, sampling_interval)
            if include_target
            else None
        )
        helper_pairs = {
            entity_id: feature_pair(state_object_at(states, current), current, sampling_interval)
            for entity_id, states in helper_states.items()
        }
        label = any_state_active_in_window(
            target_states, current + window_start, current + window_end, target_state, hold
        )
        currently_active = any_state_active_in_window(
            target_states, current, current + one_second, target_state, hold
        )
        samples.append(
            Sample(
                t=current,
                features=make_features(local_now, target_pair, helper_pairs),
                time_features=time_features(local_now),
                label=int(label),
                currently_active=int(currently_active),
            )
        )
        current += sampling_interval
    return samples


def walk_forward(
    samples: list[Sample], *, alpha: float, burn_in_fraction: float
) -> list[EvalRow]:
    """Predict-then-learn through the samples; return rows after the burn-in.

    Two models run side by side: the configured one, and a "time of day /
    weekday only" one, so you can see what the helper entities actually add.
    """
    burn_in = int(len(samples) * burn_in_fraction)
    full = NaiveBayesModel(alpha=alpha)
    time_only = NaiveBayesModel(alpha=alpha)
    rows: list[EvalRow] = []

    for index, sample in enumerate(samples):
        if index >= burn_in:
            p_full = full.predict_proba(sample.features)
            p_time = time_only.predict_proba(sample.time_features)
            if p_full is not None and p_time is not None:
                rows.append(
                    EvalRow(
                        t=sample.t,
                        p_full=p_full,
                        p_time_only=p_time,
                        currently_active=sample.currently_active,
                        label=sample.label,
                    )
                )
        full.learn_one(sample.features, sample.label)
        time_only.learn_one(sample.time_features, sample.label)
    return rows


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _sessions_caught(
    rows: list[EvalRow],
    sessions: list[datetime],
    threshold: float,
    horizon: timedelta,
    sampling_interval: timedelta,
) -> int:
    """Count session starts flagged at least one interval (and up to `horizon`) ahead."""
    caught = 0
    for session in sessions:
        earliest = session - horizon
        latest = session - sampling_interval
        if any(
            earliest <= row.t <= latest and row.p_full >= threshold for row in rows
        ):
            caught += 1
    return caught


def evaluate(
    rows: list[EvalRow],
    *,
    threshold: float,
    sessions: list[datetime],
    horizon: timedelta,
    sampling_interval: timedelta,
) -> dict[str, Any]:
    full_pairs = [(row.p_full, row.label) for row in rows]
    time_pairs = [(row.p_time_only, row.label) for row in rows]
    persistence_pairs = [(float(row.currently_active), row.label) for row in rows]

    summary = summarize(full_pairs, threshold)
    session_count = len(sessions)

    table = []
    for percent in BACKTEST_THRESHOLDS:
        at = summarize(full_pairs, percent / 100)
        caught = _sessions_caught(rows, sessions, percent / 100, horizon, sampling_interval)
        table.append(
            {
                "threshold_pct": percent,
                "flagged_rate": _round(at["flagged_rate"]),
                "precision": _round(at["precision"]),
                "recall": _round(at["recall"]),
                "lift": _round(at["lift"], 2),
                "sessions_caught": caught,
            }
        )

    return {
        "evaluated_samples": len(rows),
        "positive_rate": _round(summary["positive_rate"]),
        "baseline_accuracy": _round(summary["baseline_accuracy"]),
        "at_configured_threshold": {
            "threshold_pct": round(threshold * 100),
            "accuracy": _round(summary["accuracy"]),
            "precision": _round(summary["precision"]),
            "recall": _round(summary["recall"]),
            "flagged_rate": _round(summary["flagged_rate"]),
            "lift": _round(summary["lift"], 2),
        },
        "roc_auc": {
            "model": _round(summary["roc_auc"]),
            "time_of_day_only": _round(roc_auc(time_pairs)),
            "just_current_activity": _round(roc_auc(persistence_pairs)),
        },
        "brier_skill_score": {
            "model": _round(summary["brier_skill_score"]),
            "time_of_day_only": _round(brier_skill_score(time_pairs)),
        },
        "session_starts_in_period": session_count,
        "lead_window_minutes": [
            int(sampling_interval.total_seconds() // 60),
            int(horizon.total_seconds() // 60),
        ],
        "by_threshold": table,
    }
