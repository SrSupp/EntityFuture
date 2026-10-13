"""Runtime coordination for a single EntityFuture predictor.

Responsibilities:
- take a feature snapshot every `sampling_interval_minutes`
- schedule a "verification" for when that snapshot's "yes" window has fully
  passed, then compare the prediction against what actually happened and
  learn from it (prequential / test-then-train evaluation, so the quality
  metrics are honest)
- persist model + pending samples + rolling evaluation window across restarts
- notify entities (sensor/binary_sensor) when something changes
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
import logging
import uuid
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from .const import (
    ACCURACY_WINDOW_SIZE,
    CONF_HELPER_ENTITIES,
    CONF_HORIZON_MINUTES,
    CONF_INCLUDE_TARGET_STATE_FEATURE,
    CONF_LABEL_MODE,
    CONF_PREDICTION_THRESHOLD,
    CONF_SAMPLING_INTERVAL_MINUTES,
    CONF_TARGET_ENTITY,
    CONF_TARGET_HOLD_MINUTES,
    CONF_TARGET_STATE,
    CONF_USE_RECORDER_HISTORY,
    DEFAULT_INCLUDE_TARGET_STATE_FEATURE,
    DEFAULT_LABEL_MODE,
    DEFAULT_PREDICTION_THRESHOLD,
    DEFAULT_SAMPLING_INTERVAL_MINUTES,
    DEFAULT_TARGET_HOLD_MINUTES,
    DEFAULT_USE_RECORDER_HISTORY,
    DOMAIN,
    HISTORY_IMPORT_LOOKBACK_DAYS,
    LAPLACE_ALPHA,
    MIN_SAMPLES_FOR_ACCURACY,
    STATE_UNKNOWN_VALUE,
    STORAGE_VERSION,
)
from .features import feature_pair, label_window, make_features
from .metrics import summarize
from .model import NaiveBayesModel

_LOGGER = logging.getLogger(__name__)


def signal_update(entry_id: str) -> str:
    """Return the dispatcher signal used to notify this entry's entities."""
    return f"{DOMAIN}_{entry_id}_update"


class EntityFutureCoordinator:
    """Owns the model, scheduling and persistence for one config entry."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self.model = NaiveBayesModel(alpha=LAPLACE_ALPHA)
        # Rolling (predicted probability, actual 0/1) pairs of verified predictions.
        self.eval_window: deque[tuple[float, int]] = deque(maxlen=ACCURACY_WINDOW_SIZE)
        self.warmstart_done = False
        self.pending_samples: dict[str, dict] = {}
        self.last_backtest: dict[str, Any] | None = None

        self._store: Store = Store(hass, STORAGE_VERSION, f"{DOMAIN}_{entry.entry_id}")
        self._unsub_interval: Callable[[], None] | None = None
        self._unsub_state_change: Callable[[], None] | None = None
        self._unsub_pending: dict[str, Callable[[], None]] = {}

    # -- configuration helpers -------------------------------------------------

    @property
    def target_entity(self) -> str:
        return self.entry.data[CONF_TARGET_ENTITY]

    @property
    def target_state(self) -> str:
        return self.entry.data[CONF_TARGET_STATE]

    @property
    def horizon(self) -> timedelta:
        return timedelta(minutes=self.entry.data[CONF_HORIZON_MINUTES])

    @property
    def helper_entities(self) -> list[str]:
        return list(self.entry.options.get(CONF_HELPER_ENTITIES, []))

    @property
    def sampling_interval(self) -> timedelta:
        minutes = self.entry.options.get(
            CONF_SAMPLING_INTERVAL_MINUTES, DEFAULT_SAMPLING_INTERVAL_MINUTES
        )
        return timedelta(minutes=int(minutes))

    @property
    def prediction_threshold(self) -> int:
        return int(
            self.entry.options.get(CONF_PREDICTION_THRESHOLD, DEFAULT_PREDICTION_THRESHOLD)
        )

    @property
    def use_recorder_history(self) -> bool:
        return self.entry.options.get(
            CONF_USE_RECORDER_HISTORY, DEFAULT_USE_RECORDER_HISTORY
        )

    @property
    def include_target_state_feature(self) -> bool:
        return self.entry.options.get(
            CONF_INCLUDE_TARGET_STATE_FEATURE, DEFAULT_INCLUDE_TARGET_STATE_FEATURE
        )

    @property
    def label_mode(self) -> str:
        return self.entry.options.get(CONF_LABEL_MODE, DEFAULT_LABEL_MODE)

    @property
    def hold_minutes(self) -> int:
        return int(self.entry.options.get(CONF_TARGET_HOLD_MINUTES, DEFAULT_TARGET_HOLD_MINUTES))

    @property
    def hold(self) -> timedelta:
        return timedelta(minutes=self.hold_minutes)

    @property
    def model_signature(self) -> str:
        """Settings that change what the learned counts *mean*.

        If these change, the existing model can no longer be trusted and is
        rebuilt (counts learned for a different question would mislead).
        """
        return (
            f"{self.label_mode}|hold={self.hold_minutes}"
            f"|target={int(self.include_target_state_feature)}"
        )

    # -- lifecycle ---------------------------------------------------------

    async def async_setup(self) -> None:
        """Load persisted state, optionally warm-start, start scheduling."""
        await self._async_load()

        if not self.warmstart_done and self.use_recorder_history:
            await self._async_try_warmstart()
            # Persist immediately: without this, a restart before the first
            # live sample is verified would silently lose the warm-started
            # model and its backtest (delayed saves only happen from the
            # snapshot/verify loop).
            await self._store.async_save(self._data_to_save())

        self._unsub_interval = async_track_time_interval(
            self.hass, self._async_take_snapshot, self.sampling_interval
        )
        self._unsub_state_change = async_track_state_change_event(
            self.hass,
            [self.target_entity, *self.helper_entities],
            self._async_handle_tracked_state_change,
        )

    async def async_shutdown(self) -> None:
        """Cancel scheduled callbacks and persist final state."""
        if self._unsub_interval is not None:
            self._unsub_interval()
            self._unsub_interval = None
        if self._unsub_state_change is not None:
            self._unsub_state_change()
            self._unsub_state_change = None
        for unsub in list(self._unsub_pending.values()):
            unsub()
        self._unsub_pending.clear()
        await self._store.async_save(self._data_to_save())

    @callback
    def _async_handle_tracked_state_change(self, event) -> None:
        """Refresh the displayed probability when a relevant entity changes.

        This does not create a new training sample (those stay on the fixed
        sampling grid) - it just makes the probability/prediction entities
        feel live instead of only updating every `sampling_interval`.
        """
        self._notify()

    async def async_reset(self, *, rerun_warmstart: bool = False) -> None:
        """Wipe the learned model and pending samples."""
        for unsub in list(self._unsub_pending.values()):
            unsub()
        self._unsub_pending.clear()
        self.pending_samples.clear()
        self.model = NaiveBayesModel(alpha=LAPLACE_ALPHA)
        self.eval_window.clear()
        self.warmstart_done = False

        if rerun_warmstart and self.use_recorder_history:
            await self._async_try_warmstart()

        await self._store.async_save(self._data_to_save())
        self._notify()

    # -- persistence ---------------------------------------------------------

    async def _async_load(self) -> None:
        data = await self._store.async_load()
        if not data:
            return

        if data.get("model_signature") != self.model_signature:
            _LOGGER.info(
                "EntityFuture %s: label/feature settings changed since the model was "
                "trained (%s -> %s); rebuilding the model from scratch",
                self.entry.title,
                data.get("model_signature", "legacy"),
                self.model_signature,
            )
            return

        self.model = NaiveBayesModel.from_dict(data.get("model", {}))
        self.eval_window = deque(
            ((float(p), int(y)) for p, y in data.get("eval_window", [])),
            maxlen=ACCURACY_WINDOW_SIZE,
        )
        self.warmstart_done = bool(data.get("warmstart_done", False))

        now = dt_util.utcnow()
        for sample in data.get("pending_samples", []):
            window_end = dt_util.parse_datetime(sample.get("window_end", ""))
            if window_end is None:
                _LOGGER.debug(
                    "Dropping pending sample in an unrecognized format: %s",
                    sample.get("id"),
                )
                continue
            sample_id = sample["id"]
            self.pending_samples[sample_id] = sample
            if window_end <= now:
                # Home Assistant was offline when this sample's window ended.
                # Best effort: verify right away if we're not too far past
                # due, otherwise drop it.
                if now - window_end <= self.sampling_interval * 3:
                    await self._async_verify_sample(sample_id, now)
                else:
                    _LOGGER.debug(
                        "Dropping stale pending sample %s (window_end %s)",
                        sample_id,
                        window_end,
                    )
                    self.pending_samples.pop(sample_id, None)
            else:
                self._schedule_verification(sample_id, window_end)

    def _data_to_save(self) -> dict:
        return {
            "model_signature": self.model_signature,
            "model": self.model.to_dict(),
            "eval_window": [[probability, outcome] for probability, outcome in self.eval_window],
            "warmstart_done": self.warmstart_done,
            "pending_samples": list(self.pending_samples.values()),
        }

    def _async_persist(self) -> None:
        self._store.async_delay_save(self._data_to_save, 30)

    # -- warm start ----------------------------------------------------------

    async def _async_try_warmstart(self) -> None:
        try:
            from .history_import import async_import_history

            learned, evaluated = await async_import_history(self.hass, self)
            _LOGGER.info(
                "EntityFuture %s: learned %d samples from recorder history "
                "(%d held out for an initial backtest)",
                self.entry.title,
                learned,
                evaluated,
            )
        except Exception:  # noqa: BLE001 - warm start must never break setup
            _LOGGER.exception(
                "EntityFuture %s: recorder warm-start failed, continuing without it",
                self.entry.title,
            )
        finally:
            self.warmstart_done = True

    # -- backtest ---------------------------------------------------------------

    async def async_run_backtest(self, **overrides: Any) -> dict[str, Any]:
        """Replay recorder history with the live settings (or overrides)."""
        from .history_import import async_run_backtest

        threshold = overrides.get("threshold")
        result = await async_run_backtest(
            self.hass,
            self,
            days=overrides.get("days") or HISTORY_IMPORT_LOOKBACK_DAYS,
            label_mode=overrides.get("label_mode") or self.label_mode,
            hold_minutes=(
                self.hold_minutes
                if overrides.get("hold_minutes") is None
                else int(overrides["hold_minutes"])
            ),
            include_target=(
                self.include_target_state_feature
                if overrides.get("include_target_state_feature") is None
                else bool(overrides["include_target_state_feature"])
            ),
            helper_entities=(
                self.helper_entities
                if overrides.get("helper_entities") is None
                else list(overrides["helper_entities"])
            ),
            threshold_pct=int(threshold) if threshold else self.prediction_threshold,
        )
        self.last_backtest = result
        _LOGGER.info("EntityFuture %s: backtest result: %s", self.entry.title, result)
        self._notify()
        return result

    # -- feature engineering ---------------------------------------------------

    def _state_of(self, entity_id: str) -> str:
        state = self.hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return STATE_UNKNOWN_VALUE
        return state.state

    def build_features(self) -> dict[str, str]:
        """Build the feature vector for right now."""
        now_utc = dt_util.utcnow()
        window = self.sampling_interval
        target_pair = (
            feature_pair(self.hass.states.get(self.target_entity), now_utc, window)
            if self.include_target_state_feature
            else None
        )
        helper_pairs = {
            entity_id: feature_pair(self.hass.states.get(entity_id), now_utc, window)
            for entity_id in self.helper_entities
        }
        return make_features(dt_util.as_local(now_utc), target_pair, helper_pairs)

    # -- prediction / quality exposed to entities ----------------------------

    def current_probability(self) -> float | None:
        return self.model.predict_proba(self.build_features())

    def current_metrics(self) -> dict[str, Any] | None:
        """Quality of the verified predictions so far, at the configured threshold."""
        if len(self.eval_window) < MIN_SAMPLES_FOR_ACCURACY:
            return None
        return summarize(list(self.eval_window), self.prediction_threshold / 100)

    @property
    def verified_count(self) -> int:
        return len(self.eval_window)

    @property
    def pending_count(self) -> int:
        return len(self.pending_samples)

    @property
    def total_samples(self) -> int:
        return self.model.total_samples

    # -- snapshot / verify loop ------------------------------------------------

    @callback
    def _async_take_snapshot(self, now: datetime) -> None:
        features = self.build_features()
        predicted = self.model.predict_proba(features)

        sample_id = uuid.uuid4().hex
        now_utc = dt_util.utcnow()
        start_offset, end_offset = label_window(
            self.label_mode, self.horizon, self.sampling_interval
        )
        window_start = now_utc + start_offset
        window_end = now_utc + end_offset
        self.pending_samples[sample_id] = {
            "id": sample_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "features": features,
            "predicted_prob": predicted,
        }
        self._schedule_verification(sample_id, window_end)
        self._async_persist()
        self._notify()

    def _schedule_verification(self, sample_id: str, check_time: datetime) -> None:
        # This must be a coroutine function (not a plain callback or a bare
        # lambda) so Home Assistant schedules it safely on the event loop
        # instead of routing it to a worker thread - the recorder lookup
        # inside _async_verify_sample needs to run from there.
        async def _verify(now: datetime) -> None:
            await self._async_verify_sample(sample_id, now)

        unsub = async_track_point_in_time(self.hass, _verify, check_time)
        self._unsub_pending[sample_id] = unsub

    async def _async_verify_sample(self, sample_id: str, now: datetime) -> None:
        unsub = self._unsub_pending.pop(sample_id, None)
        if unsub is not None:
            unsub()

        sample = self.pending_samples.pop(sample_id, None)
        if sample is None:
            return

        occurred: bool | None = None
        window_start = dt_util.parse_datetime(sample.get("window_start", ""))
        window_end = dt_util.parse_datetime(sample.get("window_end", ""))
        if window_start is not None and window_end is not None:
            from .history_import import async_check_target_state_in_window

            occurred = await async_check_target_state_in_window(
                self.hass,
                self.target_entity,
                window_start,
                window_end,
                self.target_state,
                self.hold,
            )

        if occurred is None:
            # Recorder unavailable (or no data yet) - fall back to an
            # instantaneous check instead of losing the sample entirely.
            actual_state = self._state_of(self.target_entity)
            if actual_state == STATE_UNKNOWN_VALUE:
                _LOGGER.debug(
                    "Skipping sample %s: target entity unavailable at verification time",
                    sample_id,
                )
                self._async_persist()
                return
            occurred = actual_state == self.target_state

        label = 1 if occurred else 0

        predicted_prob = sample.get("predicted_prob")
        if predicted_prob is not None:
            self.eval_window.append((float(predicted_prob), label))

        self.model.learn_one(sample["features"], label)
        self._async_persist()
        self._notify()

    def _notify(self) -> None:
        async_dispatcher_send(self.hass, signal_update(self.entry.entry_id))
