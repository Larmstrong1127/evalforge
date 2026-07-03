"""Pure metric functions: precision/recall/F1, confusion matrix, and
Expected Calibration Error (ECE). Kept dependency-free of the model/training
code so they're testable with hand-computed fixtures.
"""
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def compute_classification_metrics(
    y_true: list[int], y_pred: list[int]
) -> ClassificationMetrics:
    """Computes precision/recall/F1 and confusion matrix.

    Assumes binary labels {0, 1} with 1 as the positive class (matches the
    training.data.prepare.Example labeling convention: 0=faithful,
    1=hallucinated).
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return ClassificationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion_matrix=cm.tolist(),
    )


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> float:
    """Bins predictions by confidence and averages |accuracy - confidence|
    per bin, weighted by bin size — the standard ECE formulation."""
    if not confidences:
        raise ValueError("confidences must be non-empty")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if any(c < 0.0 or c > 1.0 for c in confidences):
        raise ValueError("confidences must be in [0.0, 1.0]")
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    total = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [
            (c, ok)
            for c, ok in zip(confidences, correct, strict=True)
            if (lo <= c < hi) or (i == n_bins - 1 and c == hi)
        ]
        if not in_bin:
            continue
        bin_confidence = sum(c for c, _ in in_bin) / len(in_bin)
        bin_accuracy = sum(1 for _, ok in in_bin if ok) / len(in_bin)
        ece += (len(in_bin) / total) * abs(bin_accuracy - bin_confidence)
    return ece
