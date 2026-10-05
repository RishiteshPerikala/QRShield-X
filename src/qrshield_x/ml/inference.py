"""ML inference adapter for QRShield-X."""

from __future__ import annotations

from qrshield_x.ml.features import MLFeatures
from qrshield_x.ml.model import MLModel, MLPrediction


class MLInference:
    """Run ML predictions through the QRShield-X model boundary."""

    def __init__(self, model: MLModel) -> None:
        self._model = model

    def predict(self, features: MLFeatures) -> MLPrediction:
        """Generate a prediction from validated ML features."""

        if not isinstance(features, MLFeatures):
            raise TypeError("Features must be provided as MLFeatures.")

        return self._model.predict(features.as_list())