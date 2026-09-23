"""Runtime coordination for a single EntityFuture predictor.

Responsibilities:
- take a feature snapshot every `sampling_interval_minutes`
- schedule a "verification" for `horizon_minutes` later, when we can
  compare the prediction against what actually happened and learn from it
  (prequential / test-then-train evaluation, so accuracy is honest)
- persist model + pending samples + rolling accuracy across restarts
- notify entities (sensor/binary_sensor) when something changes
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
import logging
import uuid

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
    CONF_PREDICTION_THRESHOLD,
    CONF_SAMPLING_INTERVAL_MINUTES,
    CONF_TARGET_ENTITY,
    CONF_TARGET_STATE,
    CONF_USE_RECORDER_HISTORY,
    DEFAULT_PREDICTION_THRESHOLD,
    DEFAULT_SAMPLING_INTERVAL_MINUTES,
    DEFAULT_USE_RECORDER_HISTORY,
    DOMAIN,
    FEATURE_TARGET_CURRENT_STATE,
    FEATURE_TIME_BUCKET,
    FEATURE_WEEKDAY,
    LAPLACE_ALPHA,
    MIN_SAMPLES_FOR_ACCURACY,
    STATE_UNKNOWN_VALUE,
    STORAGE_VERSION,
    TIME_BUCKET_MINUTES,
)
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
        self.accuracy_window: deque[int] = deque(maxlen=ACCURACY_WINDOW_SIZE)
        self.warmstart_done = False
        self.pending_samples: dict[str, dict] = {}

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
        return timedelta(minutes=minutes)

    @property
    def prediction_threshold(self) -> int:
        return self.entry.options.get(
            CONF_PREDICTION_THRESHOLD, DEFAULT_PREDICTION_THRESHOLD
        )

    @property
    def use_recorder_history(self) -> bool:
        return self.entry.options.get(
            CONF_USE_RECORDER_HISTORY, DEFAULT_USE_RECORDER_HISTORY
        )

    # -- lifecycle ---------------------------------------------------------

    async def async_setup(self) -> None:
        """Load persisted state, optionally warm-start, start scheduling."""
        await self._async_load()

        if not self.warmstart_done and self.use_recorder_history:
            await self._async_try_warmstart()

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
        self.accuracy_window.clear()
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

        self.model = NaiveBayesModel.from_dict(data.get("model", {}))
        self.accuracy_window = deque(
            data.get("accuracy_window", []), maxlen=ACCURACY_WINDOW_SIZE
        )
        self.warmstart_done = bool(data.get("warmstart_done", False))

        now = dt_util.utcnow()
        for sample in data.get("pending_samples", []):
            due = dt_util.parse_datetime(sample["due"])
            if due is None:
                continue
            sample_id = sample["id"]
            self.pending_samples[sample_id] = sample
            if due <= now:
                # Home Assistant was offline when this sample was due.
                # Best effort: verify against the current state right away
                # if we're not too far past due, otherwise drop it.
                if now - due <= self.sampling_interval * 3:
                    await self._async_verify_sample(sample_id, now)
                else:
                    _LOGGER.debug(
                        "Dropping stale pending sample %s (due %s)", sample_id, due
                    )
                    self.pending_samples.pop(sample_id, None)
            else:
                self._schedule_verification(sample_id, due)

    def _data_to_save(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "accuracy_window": list(self.accuracy_window),
            "warmstart_done": self.warmstart_done,
            "pending_samples": list(self.pending_samples.values()),
        }

    def _async_persist(self) -> None:
        self._store.async_delay_save(self._data_to_save, 30)

    # -- warm start ----------------------------------------------------------

    async def _async_try_warmstart(self) -> None:
        try:
            from .history_import import async_import_history

            learned = await async_import_history(self.hass, self)
            _LOGGER.info(
                "EntityFuture %s: learned %d samples from recorder history",
                self.entry.title,
                learned,
            )
        except Exception:  # noqa: BLE001 - warm start must never break setup
            _LOGGER.exception(
                "EntityFuture %s: recorder warm-start failed, continuing without it",
                self.entry.title,
            )
        finally:
            self.warmstart_done = True

    # -- feature engineering ---------------------------------------------------

    def _state_of(self, entity_id: str) -> str:
        state = self.hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return STATE_UNKNOWN_VALUE
        return state.state

    def build_features(self, at: datetime | None = None) -> dict[str, str]:
        """Build the feature vector for "now" (or a given local time)."""
        now = at or dt_util.now()
        minute_of_day = now.hour * 60 + now.minute
        features: dict[str, str] = {
            FEATURE_WEEKDAY: str(now.weekday()),
            FEATURE_TIME_BUCKET: str(minute_of_day // TIME_BUCKET_MINUTES),
            FEATURE_TARGET_CURRENT_STATE: self._state_of(self.target_entity),
        }
        for entity_id in self.helper_entities:
            features[entity_id] = self._state_of(entity_id)
        return features

    # -- prediction / accuracy exposed to entities ----------------------------

    def current_probability(self) -> float | None:
        return self.model.predict_proba(self.build_features())

    def current_accuracy(self) -> float | None:
        if len(self.accuracy_window) < MIN_SAMPLES_FOR_ACCURACY:
            return None
        return 100.0 * sum(self.accuracy_window) / len(self.accuracy_window)

    @property
    def verified_count(self) -> int:
        return len(self.accuracy_window)

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
        due = dt_util.utcnow() + self.horizon
        self.pending_samples[sample_id] = {
            "id": sample_id,
            "due": due.isoformat(),
            "features": features,
            "predicted_prob": predicted,
        }
        self._schedule_verification(sample_id, due)
        self._async_persist()
        self._notify()

    def _schedule_verification(self, sample_id: str, due: datetime) -> None:
        unsub = async_track_point_in_time(
            self.hass,
            lambda now, sample_id=sample_id: self.hass.async_create_task(
                self._async_verify_sample(sample_id, now)
            ),
            due,
        )
        self._unsub_pending[sample_id] = unsub

    async def _async_verify_sample(self, sample_id: str, now: datetime) -> None:
        unsub = self._unsub_pending.pop(sample_id, None)
        if unsub is not None:
            unsub()

        sample = self.pending_samples.pop(sample_id, None)
        if sample is None:
            return

        actual_state = self._state_of(self.target_entity)
        if actual_state == STATE_UNKNOWN_VALUE:
            _LOGGER.debug(
                "Skipping sample %s: target entity unavailable at verification time",
                sample_id,
            )
            self._async_persist()
            return

        label = 1 if actual_state == self.target_state else 0

        predicted_prob = sample.get("predicted_prob")
        if predicted_prob is not None:
            predicted_label = 1 if predicted_prob >= 0.5 else 0
            self.accuracy_window.append(1 if predicted_label == label else 0)

        self.model.learn_one(sample["features"], label)
        self._async_persist()
        self._notify()

    def _notify(self) -> None:
        async_dispatcher_send(self.hass, signal_update(self.entry.entry_id))
