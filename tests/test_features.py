from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ef.features import (
    any_state_active_in_window,
    day_type,
    feature_pair,
    find_session_starts,
    is_idle,
    is_idle_now,
    label_window,
    make_features,
    state_object_at,
    time_features,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeState:
    def __init__(self, state: str, last_changed: datetime) -> None:
        self.state = state
        self.last_changed = last_changed


def _states(*pairs: tuple[str, int]) -> list[FakeState]:
    return [FakeState(state, T0 + timedelta(minutes=minutes)) for state, minutes in pairs]


def _min(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def test_peak_within_window_counts_as_active() -> None:
    states = _states(("off", 0), ("on", 2), ("off", 3))
    assert any_state_active_in_window(states, _min(0), _min(5), "on") is True


def test_no_occurrence_in_window_is_false() -> None:
    states = _states(("off", 0), ("on", 10))
    assert any_state_active_in_window(states, _min(0), _min(5), "on") is False


def test_state_already_active_before_window_counts() -> None:
    states = _states(("on", -10))
    assert any_state_active_in_window(states, _min(0), _min(5), "on") is True


def test_state_ending_exactly_at_window_start_does_not_count() -> None:
    states = _states(("on", -5), ("off", 0))
    assert any_state_active_in_window(states, _min(0), _min(5), "on") is False


def test_hold_extends_an_active_stretch_past_its_end() -> None:
    # Active until minute -1; window starts at 0. Only counts with a hold.
    states = _states(("on", -5), ("off", -1))
    assert any_state_active_in_window(states, _min(0), _min(5), "on") is False
    assert any_state_active_in_window(states, _min(0), _min(5), "on", timedelta(minutes=2)) is True
    assert any_state_active_in_window(states, _min(0), _min(5), "on", timedelta(minutes=1)) is False


def test_state_object_at() -> None:
    states = _states(("off", 0), ("on", 10))
    assert state_object_at(states, _min(-1)) is None
    assert state_object_at(states, _min(5)).state == "off"
    assert state_object_at(states, _min(10)).state == "on"


def test_feature_pair_recent_stable_and_unavailable() -> None:
    window = timedelta(minutes=5)
    assert feature_pair(FakeState("on", _min(0)), _min(3), window) == ("on", "recent")
    assert feature_pair(FakeState("on", _min(0)), _min(30), window) == ("on", "stable")
    assert feature_pair(FakeState("unavailable", _min(0)), _min(1), window) == ("_unknown_", "_unknown_")
    assert feature_pair(None, _min(1), window) == ("_unknown_", "_unknown_")


def test_label_window_modes() -> None:
    horizon, sampling = timedelta(minutes=15), timedelta(minutes=5)
    assert label_window("within", horizon, sampling) == (timedelta(0), horizon)
    assert label_window("exact", horizon, sampling) == (horizon, horizon + sampling)


def test_make_features_can_omit_the_targets_own_state() -> None:
    local_now = datetime(2026, 1, 5, 7, 10)  # a Monday
    helpers = {"binary_sensor.door": ("on", "recent")}
    with_target = make_features(local_now, ("off", "stable"), helpers)
    without_target = make_features(local_now, None, helpers)

    assert "_target_current_state" in with_target
    assert "_target_current_state" not in without_target
    assert without_target["_weekday"] == "0"
    assert without_target["_time_bucket"] == "14"  # 07:10 -> 430 min // 30
    assert without_target["binary_sensor.door"] == "on"
    assert without_target["binary_sensor.door__recency"] == "recent"


def test_find_session_starts_needs_a_quiet_gap() -> None:
    # pulses at 10, 12 (same session), then 60 (new session after a long gap)
    states = _states(("off", 0), ("on", 10), ("off", 11), ("on", 12), ("off", 13), ("on", 60), ("off", 61))
    starts = find_session_starts(states, "on", timedelta(0), timedelta(minutes=30))
    assert starts == [_min(10), _min(60)]


def test_find_session_starts_with_hold_merges_nearby_pulses() -> None:
    states = _states(("off", 0), ("on", 10), ("off", 11), ("on", 40), ("off", 41))
    assert find_session_starts(states, "on", timedelta(0), timedelta(minutes=20)) == [_min(10), _min(40)]
    # With a 25 minute hold the first pulse still "counts" at minute 40.
    assert find_session_starts(states, "on", timedelta(minutes=25), timedelta(minutes=20)) == [_min(10)]


def test_time_modes() -> None:
    monday_0710 = datetime(2026, 1, 5, 7, 10)
    saturday_0710 = datetime(2026, 1, 10, 7, 10)

    assert time_features(monday_0710) == {"_weekday": "0", "_time_bucket": "14"}
    assert time_features(monday_0710, "time_only") == {"_time_bucket": "14"}
    assert time_features(monday_0710, "daytype") == {"_daytype_time": "workday_14"}
    assert time_features(saturday_0710, "daytype") == {"_daytype_time": "weekend_14"}
    assert time_features(monday_0710, "daytype_and_time") == {
        "_daytype_time": "workday_14",
        "_time_bucket": "14",
    }
    # Tuesday and Monday are the same day type, so they share one joint feature value.
    tuesday = datetime(2026, 1, 6, 7, 10)
    assert time_features(tuesday, "daytype") == time_features(monday_0710, "daytype")
    assert day_type(datetime(2026, 1, 11)) == "weekend"


def test_is_idle_needs_a_quiet_gap_before() -> None:
    gap = timedelta(minutes=30)
    states = _states(("off", 0), ("on", 10), ("off", 11))
    assert is_idle(states, _min(60), "on", timedelta(0), gap) is True
    assert is_idle(states, _min(30), "on", timedelta(0), gap) is False  # on at 10..11, within 30 min
    assert is_idle(states, _min(5), "on", timedelta(0), gap) is True
    assert is_idle(states, _min(10), "on", timedelta(0), gap) is False  # active right now
    # A hold extends the activity, so the room stays "not idle" for longer.
    assert is_idle(states, _min(50), "on", timedelta(minutes=15), gap) is False


def test_is_idle_now_uses_the_current_state_object() -> None:
    gap, hold = timedelta(minutes=30), timedelta(minutes=5)
    now = _min(100)
    assert is_idle_now(FakeState("off", _min(40)), now, "on", hold, gap) is True  # 60 min quiet
    assert is_idle_now(FakeState("off", _min(80)), now, "on", hold, gap) is False  # only 20 min quiet
    assert is_idle_now(FakeState("off", _min(66)), now, "on", hold, gap) is False  # 34 < 30 + 5
    assert is_idle_now(FakeState("on", _min(0)), now, "on", hold, gap) is False  # active now
    assert is_idle_now(FakeState("unavailable", _min(0)), now, "on", hold, gap) is False
    assert is_idle_now(None, now, "on", hold, gap) is False
