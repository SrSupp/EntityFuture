from __future__ import annotations

import math

from ef.metrics import brier_skill_score, confusion, roc_auc, summarize


def test_auc_perfect_and_inverted() -> None:
    perfect = [(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)]
    inverted = [(0.1, 1), (0.2, 1), (0.8, 0), (0.9, 0)]
    assert roc_auc(perfect) == 1.0
    assert roc_auc(inverted) == 0.0


def test_auc_ties_get_half_credit_and_single_class_is_none() -> None:
    assert roc_auc([(0.5, 0), (0.5, 1)]) == 0.5
    assert roc_auc([(0.5, 1), (0.6, 1)]) is None
    assert roc_auc([]) is None


def test_confusion_counts() -> None:
    pairs = [(0.9, 1), (0.8, 0), (0.2, 1), (0.1, 0)]
    assert confusion(pairs, 0.5) == (1, 1, 1, 1)


def test_always_no_baseline_beats_naive_accuracy_for_rare_events() -> None:
    # 9% positives, model predicts "no" everywhere: accuracy 91%, but no skill.
    pairs = [(0.05, 1)] * 9 + [(0.05, 0)] * 91
    summary = summarize(pairs, 0.5)
    assert math.isclose(summary["accuracy"], 0.91)
    assert math.isclose(summary["baseline_accuracy"], 0.91)
    assert summary["recall"] == 0.0
    assert summary["precision"] is None
    assert roc_auc(pairs) == 0.5


def test_skill_score_positive_when_better_than_base_rate() -> None:
    informative = [(0.9, 1)] * 5 + [(0.1, 0)] * 15
    uninformative = [(0.25, 1)] * 5 + [(0.25, 0)] * 15
    assert brier_skill_score(informative) > 0.5
    assert abs(brier_skill_score(uninformative)) < 1e-9


def test_lift_and_flagged_rate() -> None:
    # base rate 25%; flagged half of the time with precision 50% -> lift 2.
    pairs = [(0.9, 1)] * 5 + [(0.9, 0)] * 5 + [(0.1, 1)] * 0 + [(0.1, 0)] * 10
    summary = summarize(pairs, 0.5)
    assert math.isclose(summary["positive_rate"], 0.25)
    assert math.isclose(summary["flagged_rate"], 0.5)
    assert math.isclose(summary["precision"], 0.5)
    assert math.isclose(summary["lift"], 2.0)
