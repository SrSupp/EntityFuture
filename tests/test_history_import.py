"""Unit tests for the pure interval/recency logic in history_import.py.

history_import.py imports a couple of names from `homeassistant` at module
level (just for type hints and dt utilities), and uses a relative import
for `.const`. Since a full Home Assistant install isn't available in this
environment, we stub the tiny slice of `homeassistant` that's actually
needed, and load `const.py`/`history_import.py` directly via importlib
under a fake package name - this avoids executing the real
`custom_components/entityfuture/__init__.py` (which needs much more of
Home Assistant than this test cares about).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
import types

if "homeassistant" not in sys.modules:
    ha = types.ModuleType("homeassistant")
    ha_core = types.ModuleType("homeassistant.core")
    ha_core.HomeAssistant = object
    ha_core.State = object
    ha_util = types.ModuleType("homeassistant.util")
    ha_util_dt = types.ModuleType("homeassistant.util.dt")
    ha_util_dt.utcnow = lambda: datetime.now(timezone.utc)
    ha_util_dt.as_local = lambda value: value
    sys.modules["homeassistant"] = ha
    sys.modules["homeassistant.core"] = ha_core
    sys.modules["homeassistant.util"] = ha_util
    sys.modules["homeassistant.util.dt"] = ha_util_dt

_ENTITYFUTURE_DIR = Path(__file__).resolve().parents[1] / "custom_components" / "entityfuture"
_PKG = "_entityfuture_stub"


def _load(module_name: str, filename: str):
    full_name = f"{_PKG}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, _ENTITYFUTURE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PKG
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


if _PKG not in sys.modules:
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_ENTITYFUTURE_DIR)]
    sys.modules[_PKG] = pkg

_load("const", "const.py")
history_import = _load("history_import", "history_import.py")

_any_state_active_in_window = history_import._any_state_active_in_window
_recency_at = history_import._recency_at

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeState:
    def __init__(self, state: str, last_changed: datetime) -> None:
        self.state = state
        self.last_changed = last_changed


def _states(*pairs: tuple[str, int]) -> list[FakeState]:
    return [FakeState(state, T0 + timedelta(minutes=minutes)) for state, minutes in pairs]


def test_peak_within_window_counts_as_active() -> None:
    # off -> on (minute 2) -> off (minute 3); window is [0, 5)
    states = _states(("off", 0), ("on", 2), ("off", 3))
    assert _any_state_active_in_window(states, T0, T0 + timedelta(minutes=5), "on") is True


def test_no_occurrence_in_window_is_false() -> None:
    states = _states(("off", 0), ("on", 10))
    assert _any_state_active_in_window(states, T0, T0 + timedelta(minutes=5), "on") is False


def test_state_already_active_before_window_counts() -> None:
    # Became "on" before the window started and never changed since.
    states = _states(("on", -10))
    assert _any_state_active_in_window(states, T0, T0 + timedelta(minutes=5), "on") is True


def test_state_ending_exactly_at_window_start_does_not_count() -> None:
    # Half-open interval semantics: [start, end) - touching the boundary
    # doesn't count as overlap.
    states = _states(("on", -5), ("off", 0))
    assert _any_state_active_in_window(states, T0, T0 + timedelta(minutes=5), "on") is False


def test_recency_recent_vs_stable() -> None:
    states = _states(("on", 0))
    window = timedelta(minutes=5)
    assert _recency_at(states, T0 + timedelta(minutes=1), window) == "recent"
    assert _recency_at(states, T0 + timedelta(minutes=10), window) == "stable"
