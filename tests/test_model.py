"""Unit tests for the pure-Python NaiveBayesModel (no Home Assistant needed)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "custom_components" / "entityfuture"))

from model import NaiveBayesModel  # noqa: E402


def test_predict_proba_none_before_any_training() -> None:
    model = NaiveBayesModel()
    assert model.predict_proba({"weekday": "0"}) is None


def test_learns_strong_correlation() -> None:
    model = NaiveBayesModel()
    for _ in range(50):
        model.learn_one({"weekday": "mon", "helper": "home"}, 1)
        model.learn_one({"weekday": "mon", "helper": "away"}, 0)

    prob_home = model.predict_proba({"weekday": "mon", "helper": "home"})
    prob_away = model.predict_proba({"weekday": "mon", "helper": "away"})

    assert prob_home is not None and prob_away is not None
    assert prob_home > 0.9
    assert prob_away < 0.1


def test_unseen_feature_value_is_neutral_ish() -> None:
    model = NaiveBayesModel()
    for _ in range(20):
        model.learn_one({"helper": "a"}, 1)
        model.learn_one({"helper": "b"}, 0)

    # A brand-new feature name that was never trained on should not
    # dramatically swing the prediction either way.
    baseline = model.predict_proba({"helper": "a"})
    with_new_feature = model.predict_proba({"helper": "a", "new_feature": "x"})

    assert baseline is not None and with_new_feature is not None
    assert abs(baseline - with_new_feature) < 0.05


def test_unseen_feature_is_exactly_neutral_even_with_imbalanced_classes() -> None:
    # Regression test: with a skewed base rate (very common for e.g. a
    # motion sensor that is "off" far more often than "on"), a brand-new,
    # never-learned feature must not silently push the prediction toward
    # one class just because the classes are imbalanced.
    model = NaiveBayesModel()
    for _ in range(45):
        model.learn_one({"weekday": "mon"}, 0)
    for _ in range(5):
        model.learn_one({"weekday": "mon"}, 1)

    baseline = model.predict_proba({"weekday": "mon"})
    with_new_helper = model.predict_proba({"weekday": "mon", "new_helper": "on"})

    assert baseline is not None and with_new_helper is not None
    assert baseline == with_new_helper


def test_serialization_round_trip_preserves_predictions() -> None:
    model = NaiveBayesModel(alpha=1.0)
    for i in range(30):
        label = 1 if i % 3 == 0 else 0
        model.learn_one({"weekday": str(i % 7), "helper": "on" if label else "off"}, label)

    features = {"weekday": "3", "helper": "on"}
    original_prediction = model.predict_proba(features)

    restored = NaiveBayesModel.from_dict(model.to_dict())
    restored_prediction = restored.predict_proba(features)

    assert original_prediction == restored_prediction
    assert restored.total_samples == model.total_samples


def test_label_must_be_binary() -> None:
    model = NaiveBayesModel()
    try:
        model.learn_one({"a": "b"}, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-binary label")
