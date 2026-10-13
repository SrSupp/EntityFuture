from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ef.backtest import build_samples, evaluate, walk_forward
from ef.features import find_session_starts

DAY0 = datetime(2026, 1, 5, tzinfo=timezone.utc)  # a Monday
SAMPLING = timedelta(minutes=5)
HORIZON = timedelta(minutes=15)


class FakeState:
    def __init__(self, state: str, last_changed: datetime) -> None:
        self.state = state
        self.last_changed = last_changed


def _daily_routine(days: int, *, hour: int = 7, helper_lead_min: int | None = None):
    """Target is "on" for 5 minutes at `hour`:00 every day.

    Optionally a helper pulses `helper_lead_min` minutes earlier.
    """
    target = [FakeState("off", DAY0)]
    helper = [FakeState("off", DAY0)]
    for day in range(days):
        on = DAY0 + timedelta(days=day, hours=hour)
        target.append(FakeState("on", on))
        target.append(FakeState("off", on + timedelta(minutes=5)))
        if helper_lead_min is not None:
            pulse = on - timedelta(minutes=helper_lead_min)
            helper.append(FakeState("on", pulse))
            helper.append(FakeState("off", pulse + timedelta(minutes=1)))
    return target, helper


def _samples(target, helper_states, *, mode="within", include_target=False, days=10):
    return build_samples(
        target_states=target,
        helper_states=helper_states,
        target_state="on",
        horizon=HORIZON,
        sampling_interval=SAMPLING,
        label_mode=mode,
        hold=timedelta(0),
        include_target=include_target,
        start=DAY0,
        end=DAY0 + timedelta(days=days),
        to_local=lambda value: value,
    )


def test_within_mode_labels_the_whole_run_up_to_the_event() -> None:
    target, _ = _daily_routine(days=1)
    samples = _samples(target, {}, days=1)
    positive_times = {s.t.strftime("%H:%M") for s in samples if s.label}
    # Window is [T, T+15): the 06:50, 06:55 and 07:00 snapshots see the event
    # inside it. 06:45's window ends exactly at 07:00 (exclusive); 07:05's
    # starts exactly when the event ends.
    assert positive_times == {"06:50", "06:55", "07:00"}


def test_exact_mode_labels_only_the_window_at_the_horizon() -> None:
    target, _ = _daily_routine(days=1)
    samples = _samples(target, {}, mode="exact", days=1)
    positive_times = {s.t.strftime("%H:%M") for s in samples if s.label}
    assert positive_times == {"06:45"}  # window [07:00, 07:05)


def test_walk_forward_learns_a_daily_routine() -> None:
    target, _ = _daily_routine(days=10)
    samples = _samples(target, {})
    rows = walk_forward(samples, alpha=1.0, burn_in_fraction=0.3)
    result = evaluate(
        rows,
        threshold=0.3,
        sessions=[],
        horizon=HORIZON,
        sampling_interval=SAMPLING,
    )
    assert result["roc_auc"]["model"] > 0.85
    assert result["roc_auc"]["model"] >= result["roc_auc"]["time_of_day_only"] - 1e-9
    assert result["baseline_accuracy"] > 0.9


def test_a_leading_helper_adds_signal_beyond_time_of_day() -> None:
    # The routine is not at a fixed time here: the helper is on for the 12
    # minutes before every event, but the hour itself varies by day.
    target = [FakeState("off", DAY0)]
    helper = [FakeState("off", DAY0)]
    for day in range(14):
        hour = 6 + (day * 5) % 11  # scattered hours 6..16
        on = DAY0 + timedelta(days=day, hours=hour)
        helper += [FakeState("on", on - timedelta(minutes=12)), FakeState("off", on)]
        target += [FakeState("on", on), FakeState("off", on + timedelta(minutes=5))]

    samples = _samples(target, {"binary_sensor.helper": helper}, days=14)
    rows = walk_forward(samples, alpha=1.0, burn_in_fraction=0.3)
    result = evaluate(rows, threshold=0.3, sessions=[], horizon=HORIZON, sampling_interval=SAMPLING)
    assert result["roc_auc"]["model"] > result["roc_auc"]["time_of_day_only"] + 0.1


def test_session_catch_counts_early_flags() -> None:
    target, _ = _daily_routine(days=10)
    samples = _samples(target, {})
    rows = walk_forward(samples, alpha=1.0, burn_in_fraction=0.3)
    sessions = [
        s for s in find_session_starts(target, "on", timedelta(0), timedelta(minutes=30)) if s >= rows[0].t
    ]
    result = evaluate(rows, threshold=0.3, sessions=sessions, horizon=HORIZON, sampling_interval=SAMPLING)
    by_threshold = {row["threshold_pct"]: row for row in result["by_threshold"]}
    assert result["session_starts_in_period"] == len(sessions) > 0
    # Probabilities stay modest on a short history (the weekday feature splits
    # a 10-day history into thin slices), so check a low threshold.
    assert by_threshold[5]["sessions_caught"] == len(sessions)
