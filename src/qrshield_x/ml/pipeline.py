"""ML pipeline integration for QRShield-X."""

from __future__ import annotations

from qrshield_x.ml.feature_extractor import FeatureExtractor
from qrshield_x.ml.inference import MLInference
from qrshield_x.ml.model import MLModel, MLPrediction


class MLPipeline:
    """Coordinate feature extraction and model inference."""

    def __init__(self, model: MLModel) -> None:
        self._extractor = FeatureExtractor()
        self._inference = MLInference(model)

    def predict(self, values: list[float]) -> MLPrediction:
        """Extract features and generate an ML prediction."""

        features = self._extractor.extract(values)
        return self._inference.predict(features)