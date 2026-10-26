"""Adapter for using trained baselines through QRShield-X MLModel."""

from __future__ import annotations

import numpy as np

from qrshield_x.experiments.logistic_regression import (
    LogisticRegressionBaseline,
)
from qrshield_x.experiments.random_forest import (
    RandomForestBaseline,
)
from qrshield_x.ml.model import MLPrediction


class BaselineMLAdapter:
    """Adapt a trained baseline to the QRShield-X prediction interface."""

    def __init__(
        self,
        model: LogisticRegressionBaseline | RandomForestBaseline,
    ) -> None:
        if not isinstance(
            model,
            (LogisticRegressionBaseline, RandomForestBaseline),
        ):
            raise TypeError(
                "Model must be a supported QRShield-X baseline."
            )

        if not model.is_trained:
            raise RuntimeError(
                "Baseline model must be trained before use."
            )

        self._model = model

    def predict(self, features: list[float]) -> MLPrediction:
        """Predict a QR label and confidence from extracted features."""

        if not isinstance(features, list):
            raise TypeError("Features must be provided as a list.")

        if not features:
            raise ValueError("Features cannot be empty.")

        if len(features) != 4761:
            raise ValueError(
                "Baseline expects exactly 4761 QR-image features."
            )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            for value in features
        ):
            raise TypeError(
                "All features must be finite numeric values."
            )

        feature_array = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, -1)

        predicted_label = int(self._model.predict(feature_array)[0])
        probabilities = self._model.predict_proba(feature_array)[0]
        classes = self._model.classes_

        class_indices = np.flatnonzero(classes == predicted_label)
        if len(class_indices) != 1:
            raise RuntimeError(
                "Predicted class was not found in model classes."
            )

        confidence = float(probabilities[int(class_indices[0])])

        label_mapping = {
            0: "BENIGN",
            1: "MALICIOUS",
        }

        if predicted_label not in label_mapping:
            raise ValueError(
                f"Unsupported baseline class: {predicted_label}"
            )

        return MLPrediction(
            label=label_mapping[predicted_label],
            confidence=confidence,
        )