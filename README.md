# EntityFuture

A [HACS](https://hacs.xyz/) integration for [Home Assistant](https://www.home-assistant.io/) that predicts whether an entity will be in a given state **X minutes from now** — using a deliberately simple, self-learning model (Naive Bayes) that runs directly inside Home Assistant, with no external dependencies (no NumPy, no scikit-learn, no cloud).

## Disclaimer

This project was built together with [Claude](https://claude.com/claude-code) (Anthropic's AI coding assistant) — most of the code, architecture decisions, and this README were generated with AI assistance and reviewed by the repository owner, but **not independently audited line by line**, and it has not been tested against the full range of Home Assistant versions/configurations. It is provided **as-is, without any warranty of any kind**, express or implied. Use it at your own risk, review the code yourself before relying on it (especially before wiring it up to actuators like heaters), and expect rough edges — this is a hobby project, not a certified product.

## What does EntityFuture do?

You create a "predictor":

- **Target entity** – e.g. `binary_sensor.bathroom_humidity`, `person.sven`, `switch.heating`
- **Target state** – which state counts as "yes", e.g. `on` or `home`
- **Horizon** – how many minutes into the future to predict, e.g. 30

Optionally you add **helper entities** that help the model learn (e.g. `binary_sensor.school_holiday`, `person.partner`, `sensor.weather_condition`, `input_boolean.home_office`) — these contribute as features but are never themselves predicted.

This creates a device with three entities:

| Entity | Description |
|---|---|
| `sensor.<name>_probability` | Probability (0–100%) that the target entity will be in its target state in *X* minutes |
| `binary_sensor.<name>_prediction` | Yes/no, based on a configurable threshold (default 50%) |
| `sensor.<name>_accuracy` | Rolling hit-rate of past predictions (diagnostic entity) |

## How does the model work?

**Naive Bayes**, hand-implemented, with no ML library at all:

- Features used: day of week, time of day (30-minute buckets), the target entity's own current state, and the current state of every configured helper entity.
- Every few minutes (sampling interval, default 5 min), the integration takes a snapshot of these features and remembers: "check back in *X* minutes to see what became of the target entity".
- Once that time arrives, the actual state is compared against the earlier prediction — this is how **accuracy** is computed, honestly, without any test-data trickery ("predict then verify") — and the feature vector together with the observed outcome is learned into the model.
- Only counters per feature value and class are kept — memory usage grows with the number of distinct observed values, not with time. Lightweight enough for a Raspberry Pi.
- During setup, existing [Recorder](https://www.home-assistant.io/integrations/recorder/) history (last 7 days by default) can optionally be used to "warm-start" the model instead of starting from zero.

**Note on the accuracy sensor after a warm start:** the warm start only feeds `model.learn_one(...)` — it does *not* populate the accuracy counter. That's intentional: testing the model against data it was just trained on would make accuracy look artificially good. So `sensor.<name>_accuracy` stays unknown until at least 5 *live* predictions have actually been verified (roughly `horizon + 4 × sampling_interval` after setup), even though `training_samples` on the probability sensor's attributes may already be high right after a warm start.

The model and all pending (not-yet-verified) predictions are persisted and survive a Home Assistant restart.

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
4. Afterwards, use *Configure* on the device at any time to add helper entities, and adjust the sampling interval, threshold, and recorder warm start

## Service `entityfuture.reset_model`

Wipes a predictor's learned model (e.g. after adding new helper entities, if you want to start learning "from scratch") and can optionally re-run the recorder warm start.

## Limitations (deliberate, for v1)

- Only discrete states as a target (no "temperature > 22°C" threshold on numeric sensors)
- One predictor = one target entity + one horizon; create multiple predictors for multiple horizons
- Naive Bayes assumes features are conditionally independent — for tightly correlated helper entities this can slightly skew the probability, but in practice it's a good, very cheap trade-off for this use case

## Development

```bash
pip install pytest
pytest tests/
```

`tests/test_model.py` checks the core Naive Bayes model logic independently of Home Assistant. The rest of the integration (config flow, coordinator, recorder import) should additionally be tested against a real Home Assistant instance.
