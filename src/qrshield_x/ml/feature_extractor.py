"""Feature extraction boundary for QRShield-X."""

from __future__ import annotations

from qrshield_x.ml.features import FeatureBuilder, MLFeatures


class FeatureExtractor:
    """Extract a validated ML feature vector from numeric inputs."""

    def __init__(self) -> None:
        self._builder = FeatureBuilder()

    def extract(self, values: list[float]) -> MLFeatures:
        """Convert numeric investigation features into MLFeatures."""
        return self._builder.build(values)