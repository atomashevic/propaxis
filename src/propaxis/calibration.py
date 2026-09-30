"""Binary thresholds fitted on labeled calibration data and evaluated separately."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .instrument import ScoreResult


def _labels(labels, count):
    labels = np.asarray(labels)
    if labels.shape != (count,) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain one binary 0/1 value per document")
    if len(np.unique(labels)) != 2:
        raise ValueError("both classes are required for calibration/evaluation")
    return labels.astype(int)


def _values(result, column):
    values = np.asarray([row[column] for row in result.scores], dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be nonempty and finite")
    return values


@dataclass(frozen=True)
class BinaryClassifier:
    """A saved decision threshold tied to one scoring instrument and score column."""

    threshold: float
    column: str
    instrument_id: str
    objective: str

    def predict(self, result: ScoreResult) -> np.ndarray:
        """Return 0/1 predictions using score >= threshold; no parameter fitting."""
        if result.instrument_id != self.instrument_id:
            raise ValueError(
                "scores come from a different instrument; recalibrate after changing propositions"
            )
        return (_values(result, self.column) >= self.threshold).astype(int)

    def evaluate(self, result: ScoreResult, labels) -> dict:
        """Held-out binary metrics. Precision and F1 are zero when no positives are predicted."""
        predictions = self.predict(result)
        y = _labels(labels, len(predictions))
        values = _values(result, self.column)
        tp = int(((predictions == 1) & (y == 1)).sum())
        tn = int(((predictions == 0) & (y == 0)).sum())
        fp = int(((predictions == 1) & (y == 0)).sum())
        fn = int(((predictions == 0) & (y == 1)).sum())
        # Group score ties before computing ROC area and noninterpolated AP.
        order = np.argsort(-values, kind="stable")
        sorted_y, sorted_values = y[order], values[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_values)) + 1, len(y)] - 1
        true_positives = np.cumsum(sorted_y)[ends]
        false_positives = ends + 1 - true_positives
        recall = np.r_[0.0, true_positives / y.sum()]
        fpr = np.r_[0.0, false_positives / (len(y) - y.sum())]
        auroc = float(np.sum(np.diff(fpr) * (recall[:-1] + recall[1:]) / 2))
        ap = float(np.sum(np.diff(recall) * true_positives / (ends + 1)))
        return {
            "n": len(y),
            "positive_count": int(y.sum()),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / (tp + fn),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "auroc": auroc,
            "average_precision": ap,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"schema": "propaxis.binary.v1", **asdict(self)}, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.pop("schema", None) != "propaxis.binary.v1":
            raise ValueError("unsupported classifier schema")
        if not np.isfinite(data["threshold"]) or data["objective"] not in {
            "balanced_accuracy",
            "f1",
        }:
            raise ValueError("invalid classifier parameters")
        return cls(**data)


def calibrate(
    result: ScoreResult, labels, *, column="score", objective="balanced_accuracy"
) -> BinaryClassifier:
    """Choose a binary threshold on a dedicated calibration sample.

    Higher scores must correspond to label 1. All distinct cutoffs are evaluated;
    objective ties select the higher (more conservative) threshold. Use separate
    test documents, and keep repeated speakers/authors within the same split.
    """
    if objective not in {"balanced_accuracy", "f1"}:
        raise ValueError("objective must be balanced_accuracy or f1")
    values = _values(result, column)
    y = _labels(labels, len(values))
    order = np.argsort(values, kind="stable")
    values, y = values[order], y[order]
    distinct, starts = np.unique(values, return_index=True)
    positive_prefix = np.r_[0, np.cumsum(y)]
    positives, negatives = y.sum(), len(y) - y.sum()
    starts = np.r_[starts, len(y)]
    thresholds = np.r_[distinct, np.nextafter(values[-1], np.inf)]
    tp = positives - positive_prefix[starts]
    fn = positives - tp
    fp = len(y) - starts - tp
    tn = negatives - fp
    metric = (
        (tp / positives + tn / negatives) / 2
        if objective == "balanced_accuracy"
        else 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    )
    best = np.flatnonzero(metric == metric.max())[-1]
    if not np.isfinite(thresholds[best]):
        raise ValueError("scores are too large for a finite decision threshold")
    return BinaryClassifier(float(thresholds[best]), column, result.instrument_id, objective)
