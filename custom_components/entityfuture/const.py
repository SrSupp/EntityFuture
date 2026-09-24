"""Constants for the EntityFuture integration."""
from __future__ import annotations

DOMAIN = "entityfuture"
PLATFORMS = ["sensor", "binary_sensor"]

# Config entry data (fixed at creation time)
CONF_TARGET_ENTITY = "target_entity"
CONF_TARGET_STATE = "target_state"
CONF_HORIZON_MINUTES = "horizon_minutes"

# Options (changeable after creation)
CONF_HELPER_ENTITIES = "helper_entities"
CONF_SAMPLING_INTERVAL_MINUTES = "sampling_interval_minutes"
CONF_PREDICTION_THRESHOLD = "prediction_threshold"
CONF_USE_RECORDER_HISTORY = "use_recorder_history"

DEFAULT_SAMPLING_INTERVAL_MINUTES = 5
DEFAULT_PREDICTION_THRESHOLD = 50
DEFAULT_USE_RECORDER_HISTORY = True

MIN_HORIZON_MINUTES = 1
MAX_HORIZON_MINUTES = 24 * 60

MIN_SAMPLING_INTERVAL_MINUTES = 1
MAX_SAMPLING_INTERVAL_MINUTES = 60

# Internal model tuning (not exposed in the UI to keep it simple)
TIME_BUCKET_MINUTES = 30
ACCURACY_WINDOW_SIZE = 200
LAPLACE_ALPHA = 1.0
HISTORY_IMPORT_LOOKBACK_DAYS = 7
MIN_SAMPLES_FOR_ACCURACY = 5

# Warm start uses a chronological train/test split so the accuracy sensor
# already has an honest (held-out, walk-forward) estimate right after setup,
# instead of only ever reflecting live predictions made from now on.
WARMSTART_EVAL_FRACTION = 0.2
MIN_SAMPLES_FOR_WARMSTART_EVAL = 30

# Feature keys used internally by the model
FEATURE_WEEKDAY = "_weekday"
FEATURE_TIME_BUCKET = "_time_bucket"
FEATURE_TARGET_CURRENT_STATE = "_target_current_state"

# Suffix appended to an entity's feature key for its companion "did this
# just change?" feature, e.g. "cover.bedroom_blind__recency". A change
# within the last sampling interval is "recent", otherwise "stable" - this
# lets the model weigh a fresh transition differently from a state that
# has simply been held for a while.
RECENCY_SUFFIX = "__recency"
FEATURE_TARGET_RECENCY = FEATURE_TARGET_CURRENT_STATE + RECENCY_SUFFIX
RECENCY_RECENT = "recent"
RECENCY_STABLE = "stable"

STATE_UNKNOWN_VALUE = "_unknown_"

STORAGE_VERSION = 1

SERVICE_RESET_MODEL = "reset_model"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
