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
CONF_INCLUDE_TARGET_STATE_FEATURE = "include_target_state_feature"
CONF_LABEL_MODE = "label_mode"
CONF_TARGET_HOLD_MINUTES = "target_hold_minutes"

# What "yes" means for a prediction made at time T:
#   within: the target state occurs at any point in [T, T + horizon)
#   exact:  the target state occurs at any point in [T + horizon, T + horizon + sampling interval)
LABEL_MODE_WITHIN = "within"
LABEL_MODE_EXACT = "exact"
DEFAULT_LABEL_MODE = LABEL_MODE_WITHIN

# Treat the target as still "active" this long after its last active moment,
# so short sensor pulses (e.g. a PIR that holds for ~1 minute) merge into one
# session instead of looking like unrelated blips.
DEFAULT_TARGET_HOLD_MINUTES = 0
MAX_TARGET_HOLD_MINUTES = 60

DEFAULT_SAMPLING_INTERVAL_MINUTES = 5
DEFAULT_PREDICTION_THRESHOLD = 50
DEFAULT_USE_RECORDER_HISTORY = True
# The target's own current state (and whether it just changed) is often the
# single strongest predictor of its near-future state - but for a predictor
# meant to *anticipate* an event from other, earlier signals (e.g. a blind
# opening elsewhere in the house before motion starts), that dominant signal
# can drown out the weaker leading indicators entirely. Default stays True
# for backward compatibility / the "will this still be on" use case, but
# anticipatory use cases should turn it off.
DEFAULT_INCLUDE_TARGET_STATE_FEATURE = True

MIN_HORIZON_MINUTES = 1
MAX_HORIZON_MINUTES = 24 * 60

MIN_SAMPLING_INTERVAL_MINUTES = 1
MAX_SAMPLING_INTERVAL_MINUTES = 60

# Internal model tuning (not exposed in the UI to keep it simple)
TIME_BUCKET_MINUTES = 30
ACCURACY_WINDOW_SIZE = 500
LAPLACE_ALPHA = 1.0
HISTORY_IMPORT_LOOKBACK_DAYS = 7
MIN_SAMPLES_FOR_ACCURACY = 5
MIN_SAMPLES_FOR_AUC = 20

# Backtest (entityfuture.backtest service)
BACKTEST_BURN_IN_FRACTION = 0.3
MIN_SAMPLES_FOR_BACKTEST = 50
BACKTEST_THRESHOLDS = (5, 10, 15, 20, 30, 40, 50)
SESSION_GAP_MINUTES = 30

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
SERVICE_BACKTEST = "backtest"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
