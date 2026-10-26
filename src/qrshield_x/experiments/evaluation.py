"""Evaluation metrics for QRShield-X baseline experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ExperimentEvaluator:
    """Calculate validation metrics for quishing detection models."""

    LABELS = (0, 1)

    def evaluate(
        self,
        y_true: Any,
        y_pred: Any,
        *,
        y_probability_malicious: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluate predictions, treating label 1 as malicious."""

        actual = np.asarray(y_true)
        predicted = np.asarray(y_pred)

        if actual.ndim != 1 or predicted.ndim != 1:
            raise ValueError("y_true and y_pred must be one-dimensional.")

        if actual.size == 0:
            raise ValueError("Evaluation data must not be empty.")

        if actual.size != predicted.size:
            raise ValueError("y_true and y_pred must have equal lengths.")

        if not np.isin(actual, self.LABELS).all():
            raise ValueError("y_true must contain only labels 0 and 1.")

        if not np.isin(predicted, self.LABELS).all():
            raise ValueError("y_pred must contain only labels 0 and 1.")

        matrix = confusion_matrix(
            actual,
            predicted,
            labels=list(self.LABELS),
        )
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())

        # Probability-based metrics are undefined when actual labels
        # contain only one class, but supplied probabilities must still
        # pass validation.
        roc_auc: float | None = None
        average_precision: float | None = None

        if y_probability_malicious is not None:
            probabilities = np.asarray(
                y_probability_malicious,
                dtype=float,
            )

            if probabilities.ndim != 1:
                raise ValueError(
                    "Malicious-class probabilities must be one-dimensional."
                )

            if probabilities.size != actual.size:
                raise ValueError(
                    "Probabilities must have the same length as y_true."
                )

            if not np.isfinite(probabilities).all():
                raise ValueError("Probabilities must be finite.")

            if ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Probabilities must be between 0 and 1.")

            if np.unique(actual).size == 2:
                roc_auc = float(roc_auc_score(actual, probabilities))
                average_precision = float(
                    average_precision_score(actual, probabilities)
                )

        has_benign_samples = (tn + fp) > 0
        has_malicious_samples = (tp + fn) > 0
        has_both_classes = has_benign_samples and has_malicious_samples

        specificity = (
            float(tn / (tn + fp)) if has_benign_samples else None
        )
        false_positive_rate = (
            float(fp / (tn + fp)) if has_benign_samples else None
        )
        false_negative_rate = (
            float(fn / (tp + fn)) if has_malicious_samples else None
        )
        balanced_accuracy = (
            float(balanced_accuracy_score(actual, predicted))
            if has_both_classes
            else None
        )

        return {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision_malicious": float(
                precision_score(
                    actual,
                    predicted,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "recall_malicious": float(
                recall_score(
                    actual,
                    predicted,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "f1_malicious": float(
                f1_score(
                    actual,
                    predicted,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "balanced_accuracy": balanced_accuracy,
            "average_precision": average_precision,
            "confusion_matrix": {
                "labels": [0, 1],
                "rows": "actual",
                "columns": "predicted",
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "roc_auc": roc_auc,
        }
