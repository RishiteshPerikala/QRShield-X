"""Machine-learning model abstraction for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MLPrediction:
    """Prediction produced by an ML model."""

    label: str
    confidence: float


class MLModel(Protocol):
    """Protocol defining the QRShield-X ML model interface."""

    def predict(
        self,
        features: list[float],
    ) -> MLPrediction:
        """Generate a prediction from extracted features."""
        ...


class DummyMLModel:
    """Deterministic placeholder model for pipeline development."""

    def predict(
        self,
        features: list[float],
    ) -> MLPrediction:
        """Generate a deterministic placeholder prediction."""

        if not isinstance(features, list):
            raise TypeError("Features must be provided as a list.")

        if not features:
            raise ValueError("Features cannot be empty.")

        if not all(
            isinstance(feature, (int, float))
            for feature in features
        ):
            raise TypeError("All features must be numeric.")

        return MLPrediction(
            label="UNKNOWN",
            confidence=0.0,
        )