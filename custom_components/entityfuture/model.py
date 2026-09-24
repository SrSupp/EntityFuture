"""A small online Naive Bayes classifier.

This is intentionally simple: pure Python, no NumPy/scikit-learn/etc. It
keeps running counts per feature value and class, and combines them with
the Naive Bayes (conditional independence) assumption. Learning is
incremental ("one sample at a time"), so there is no batch retraining
step and the memory footprint only grows with the number of distinct
feature values actually observed.
"""
from __future__ import annotations

import math
from typing import Any

FeatureVector = dict[str, str]

_POSITIVE = 1
_NEGATIVE = 0


class NaiveBayesModel:
    """Bernoulli-target Naive Bayes classifier, learned online."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize with a Laplace smoothing factor `alpha`."""
        self.alpha = alpha
        self.class_counts: dict[int, int] = {_NEGATIVE: 0, _POSITIVE: 0}
        # feature name -> feature value -> class -> count
        self.feature_counts: dict[str, dict[str, dict[int, int]]] = {}
        # feature name -> set of distinct values ever observed
        self.feature_domains: dict[str, set[str]] = {}

    @property
    def total_samples(self) -> int:
        """Return the number of samples learned so far."""
        return self.class_counts[_NEGATIVE] + self.class_counts[_POSITIVE]

    def learn_one(self, features: FeatureVector, label: int) -> None:
        """Update the model with a single observed (features, label) pair."""
        if label not in (_NEGATIVE, _POSITIVE):
            raise ValueError("label must be 0 or 1")

        self.class_counts[label] += 1
        for name, raw_value in features.items():
            value = str(raw_value)
            self.feature_domains.setdefault(name, set()).add(value)
            counts = self.feature_counts.setdefault(name, {}).setdefault(
                value, {_NEGATIVE: 0, _POSITIVE: 0}
            )
            counts[label] += 1

    def predict_proba(self, features: FeatureVector) -> float | None:
        """Return P(label=1 | features), or None if nothing was learned yet."""
        if self.total_samples == 0:
            return None

        log_odds = math.log(
            (self.class_counts[_POSITIVE] + self.alpha)
            / (self.class_counts[_NEGATIVE] + self.alpha)
        )

        for name, raw_value in features.items():
            if name not in self.feature_domains:
                # We have never learned this feature at all (e.g. a helper
                # entity that was just added). Zero observations means zero
                # information - it must not shift the odds either way, even
                # if the two classes are imbalanced overall.
                continue

            value = str(raw_value)
            domain_size = len(self.feature_domains[name]) or 1
            counts = self.feature_counts.get(name, {}).get(
                value, {_NEGATIVE: 0, _POSITIVE: 0}
            )
            p_given_positive = (counts[_POSITIVE] + self.alpha) / (
                self.class_counts[_POSITIVE] + self.alpha * domain_size
            )
            p_given_negative = (counts[_NEGATIVE] + self.alpha) / (
                self.class_counts[_NEGATIVE] + self.alpha * domain_size
            )
            log_odds += math.log(p_given_positive) - math.log(p_given_negative)

        try:
            return 1.0 / (1.0 + math.exp(-log_odds))
        except OverflowError:
            return 0.0 if log_odds < 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a JSON-compatible dict."""
        return {
            "alpha": self.alpha,
            "class_counts": {
                "0": self.class_counts[_NEGATIVE],
                "1": self.class_counts[_POSITIVE],
            },
            "feature_counts": {
                name: {
                    value: {"0": counts[_NEGATIVE], "1": counts[_POSITIVE]}
                    for value, counts in values.items()
                }
                for name, values in self.feature_counts.items()
            },
            "feature_domains": {
                name: sorted(domain) for name, domain in self.feature_domains.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NaiveBayesModel:
        """Reconstruct a model previously serialized with `to_dict`."""
        model = cls(alpha=data.get("alpha", 1.0))

        class_counts = data.get("class_counts", {})
        model.class_counts = {
            _NEGATIVE: int(class_counts.get("0", 0)),
            _POSITIVE: int(class_counts.get("1", 0)),
        }

        for name, values in data.get("feature_counts", {}).items():
            model.feature_counts[name] = {
                value: {
                    _NEGATIVE: int(counts.get("0", 0)),
                    _POSITIVE: int(counts.get("1", 0)),
                }
                for value, counts in values.items()
            }

        for name, domain in data.get("feature_domains", {}).items():
            model.feature_domains[name] = set(domain)

        return model
