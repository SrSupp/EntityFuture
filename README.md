# EntityFuture

A [HACS](https://hacs.xyz/) integration for [Home Assistant](https://www.home-assistant.io/) that estimates whether an entity will be in a given state **within the next X minutes** — using a deliberately simple, self-learning model (Naive Bayes) that runs directly inside Home Assistant, with no external dependencies (no NumPy, no scikit-learn, no cloud).

## Disclaimer

This project was built together with [Claude](https://claude.com/claude-code) (Anthropic's AI coding assistant) — most of the code, architecture decisions, and this README were generated with AI assistance and reviewed by the repository owner, but **not independently audited line by line**, and it has not been tested against the full range of Home Assistant versions/configurations. It is provided **as-is, without any warranty of any kind**, express or implied. Use it at your own risk, review the code yourself before relying on it (especially before wiring it up to actuators like heaters), and expect rough edges — this is a hobby project, not a certified product.

## What does EntityFuture do?

You create a "predictor":

- **Target entity** – e.g. `binary_sensor.bathroom_motion`, `person.sven`, `switch.heating`
- **Target state** – which state counts as "yes", e.g. `on` or `home`
- **Horizon** – how many minutes ahead to look, e.g. 15

Optionally you add **helper entities** that help the model learn (e.g. `binary_sensor.school_holiday`, `person.partner`, `cover.bedroom_blind`, `binary_sensor.front_door`) — these contribute as features but are never themselves predicted.

This creates a device with these entities:

| Entity | Description |
|---|---|
| `sensor.<name>_probability` | Probability (0–100%) that the target will be in its target state (see "What counts as yes" below) |
| `binary_sensor.<name>_prediction` | Yes/no, based on a configurable threshold (default 50%) |
| `sensor.<name>_accuracy` | Rolling accuracy at your threshold. Attributes show `baseline_accuracy` — what a predictor that *always* says the more common answer would score — plus precision, recall and the share of time it says "yes". **Compare accuracy to the baseline, not to 100%.** |
| `sensor.<name>_auc` | Discrimination (ROC AUC) of the verified predictions. 0.5 = coin flip, 1.0 = perfect. Unlike accuracy it isn't inflated by a rare "yes". |
| `sensor.<name>_auc_idle` | The same, but counting only predictions made while the target had been quiet for 30+ minutes. That is the case anticipation (e.g. pre-heating) is about: the room is idle now — does the probability rise *before* it is used again? Continuing activity can't flatter this number. |

## What counts as "yes"? (Options → "What counts as yes")

- **Within the horizon** *(default)*: yes if the target state occurs **at any point in the next X minutes**. This is usually the right question for something like pre-heating: "will this room be used soon?"
- **Exactly at the horizon**: yes only if it occurs in the short window (one sampling interval wide) X minutes from now. A much sparser, noisier event for pulse-like sensors such as a PIR motion sensor.

**Hold time** treats the target as still active for N minutes after its last active moment, so short sensor pulses (a PIR that reports "on" for ~1 minute) merge into one session instead of looking like unrelated blips.

Changing either rebuilds the model (the learned counts would otherwise answer a different question), then re-runs the recorder warm start.

## Time model (Options → "Time model")

How the clock enters the model:

- **Weekday + time of day (separate)** *(default)*: seven weekdays and 48 half-hour buckets as two independent features.
- **Workday/weekend × time of day (joint)**: one feature, e.g. `workday_14` (Mon–Fri, 07:00–07:30). Each value gets far more examples, so a routine is learned much faster on a short history.
- **Workday/weekend × time of day, plus time of day**: the joint feature plus the plain time-of-day bucket as a fallback.
- **Time of day only**.

Not sure which fits your household? Don't guess — compare them with the backtest (`time_mode`). Changing the option rebuilds the model.

## How does the model work?

**Naive Bayes**, hand-implemented, with no ML library at all:

- Features: day of week, time of day (30-minute buckets), optionally the target entity's own current state (see below), and the current state of every configured helper entity. Each entity-state feature also has a companion "recency" feature ("recent" if it changed within the last sampling interval, otherwise "stable").
- Every few minutes (sampling interval, default 5 min), the integration takes a snapshot of these features and remembers: "check back later to see what became of the target entity".
- Once the "yes" window of that snapshot has fully passed, the outcome is looked up (a lightweight single-entity [Recorder](https://www.home-assistant.io/integrations/recorder/) query; instantaneous check as a fallback), compared against the earlier prediction, and then learned. Predicting *before* learning is what keeps the quality metrics honest.
- Only counters per feature value and class are kept — memory grows with the number of distinct observed values, not with time. A feature that has never been observed at all (e.g. a helper entity you just added) contributes exactly nothing until it has real data.
- During setup, existing recorder history (last 7 days) can optionally "warm-start" the model. The most recent ~20% is replayed walk-forward (predict, then learn) to fill the quality metrics with an honest, held-out estimate right away.

**Adding helper entities later:** a newly added helper starts with zero observations, so it has no effect until it accumulates live data. To backfill it immediately, call `entityfuture.reset_model` with `rerun_warmstart: true`.

**Anticipatory predictors vs. "will this still be true" predictors:** the target's own current state is usually by far the strongest predictor of its own near future. That suits "will this switch still be on", but works *against* an anticipatory predictor, because Naive Bayes just sums every feature's evidence and that one self-referential signal drowns out weaker leading indicators. For anticipation, turn off **"Include the target's own current state as a feature"** in Options.

## Is it any good? Use the backtest

Don't wait days to find out. The `entityfuture.backtest` service replays your recorder history through the model — predict first, then learn, exactly like live — and reports:

- ROC AUC of your setup, of a **time-of-day-only** model (what do the helpers actually add?), and of "just use whether it is active right now"
- the positive rate and the **always-"no" baseline accuracy**
- a **"room idle" block**: the same quality numbers counting only moments when the target had been quiet for 30+ minutes — i.e. whether the probability rises *before* a new session starts
- a **by-threshold table**: share of time the predictor would say "yes" (≈ how long a heater would run), precision, recall, lift, and how many **session starts were flagged early enough** (at least one sampling interval, up to the horizon, ahead)

Every setting can be overridden per run (label mode, hold time, time model, helper entities, include-target-state, threshold, days), so you can compare candidate leading signals without touching the live predictor. The result is returned as the service response (Developer Tools → Actions shows it), written to the log, and shown on the AUC sensor's `last_backtest` attribute.

## Installation

### Via HACS (custom repository)

1. HACS → Integrations → Menu (⋮) → *Custom repositories*
2. Repository URL: `https://github.com/SrSupp/EntityFuture`, category: *Integration*
3. Install "EntityFuture", restart Home Assistant

### Manually

Copy `custom_components/entityfuture` into your `config/custom_components/` directory and restart Home Assistant.

## Setup

1. Settings → Devices & Services → Add Integration → *EntityFuture*
2. Provide a name, the target entity, and the horizon (minutes)
3. Confirm/adjust the target state (pre-filled with the entity's current state, shown as a dropdown of its known states)
4. Afterwards, use *Configure* on the device at any time to add helper entities and adjust sampling interval, threshold, label mode, hold time and warm start

## Services

- `entityfuture.reset_model` — wipes a predictor's learned model and can optionally re-run the recorder warm start.
- `entityfuture.backtest` — see above.

## Limitations

- Only discrete states as a target (no "temperature > 22°C" threshold on numeric sensors)
- One predictor = one target entity + one horizon; create multiple predictors for multiple horizons
- With the default time model, weekday and time of day are separate features: with only a week or two of history each weekday has very few examples, which can hold probabilities down even for very regular routines. Try a workday/weekend time model (see above).
- Naive Bayes assumes features are conditionally independent — tightly correlated features (e.g. a helper's state and its "recency") are partly double-counted, so probabilities tend to be overconfident. Use the backtest's precision/lift rather than the raw percentage.
- A prediction can only be as good as its signals. If nothing in your house reliably happens a while *before* the target, no model can foresee it 15 minutes ahead — the backtest will tell you.

## Development

```bash
pip install pytest
pytest tests/
```

The model, features, metrics and backtest logic have no Home Assistant imports and are unit tested directly. The rest (config flow, coordinator, recorder access) should be tested against a real Home Assistant instance.
