"""Pure-Python evaluation metrics for probabilistic yes/no predictions.

A plain accuracy number is misleading for rare events: if only 9% of
windows are "yes", a model that always says "no" scores 91%. These metrics
are chosen so a prediction has to beat that kind of trivial baseline to
look good.
"""
from __future__ import annotations

from typing import Any

# (predicted probability of "yes", actual outcome 0/1)
Pair = tuple[float, int]


def roc_auc(pairs: list[Pair]) -> float | None:
    """Rank-based ROC AUC: chance a random "yes" outranks a random "no".

    0.5 is a coin flip, 1.0 is perfect. Ties get half credit. Returns None
    when only one outcome class is present.
    """
    positives = sum(1 for _, outcome in pairs if outcome)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(pairs, key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    total = len(ordered)
    while index < total:
        end = index
        while end + 1 < total and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        rank_sum += average_rank * sum(1 for k in range(index, end + 1) if ordered[k][1])
        index = end + 1

    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def brier_score(pairs: list[Pair]) -> float | None:
    if not pairs:
        return None
    return sum((probability - outcome) ** 2 for probability, outcome in pairs) / len(pairs)


def brier_skill_score(pairs: list[Pair]) -> float | None:
    """1 - Brier / Brier of always predicting the base rate. >0 beats the baseline."""
    if not pairs:
        return None
    base_rate = sum(outcome for _, outcome in pairs) / len(pairs)
    reference = base_rate * (1 - base_rate)
    if reference == 0:
        return None
    return 1 - brier_score(pairs) / reference


def confusion(pairs: list[Pair], threshold: float) -> tuple[int, int, int, int]:
    """Return (true_pos, false_pos, false_neg, true_neg) at the threshold."""
    tp = fp = fn = tn = 0
    for probability, outcome in pairs:
        flagged = probability >= threshold
        if flagged and outcome:
            tp += 1
        elif flagged:
            fp += 1
        elif outcome:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def summarize(pairs: list[Pair], threshold: float) -> dict[str, Any]:
    """Summarize predictions at a probability threshold (0..1).

    `flagged_rate` is the share of the time the predictor says "yes" - for a
    heater that is roughly the share of time it would be running.
    """
    count = len(pairs)
    tp, fp, fn, tn = confusion(pairs, threshold)
    positives = tp + fn
    positive_rate = positives / count if count else None
    flagged = tp + fp

    precision = tp / flagged if flagged else None
    recall = tp / positives if positives else None
    lift = (
        precision / positive_rate
        if precision is not None and positive_rate
        else None
    )
    return {
        "n": count,
        "positive_rate": positive_rate,
        "baseline_accuracy": max(positive_rate, 1 - positive_rate) if count else None,
        "accuracy": (tp + tn) / count if count else None,
        "precision": precision,
        "recall": recall,
        "flagged_rate": flagged / count if count else None,
        "lift": lift,
        "roc_auc": roc_auc(pairs),
        "brier_skill_score": brier_skill_score(pairs),
    }
