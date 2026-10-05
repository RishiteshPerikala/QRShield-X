"""Feature representation for QRShield-X ML analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLFeatures:
    """Numeric feature vector supplied to an ML model."""

    values: tuple[float, ...]

    def as_list(self) -> list[float]:
        """Return features as a list suitable for model inference."""
        return list(self.values)


class FeatureBuilder:
    """Build validated ML feature representations."""

    def build(self, values: list[float]) -> MLFeatures:
        """Build an ML feature vector from numeric values."""

        if not isinstance(values, list):
            raise TypeError("Feature values must be provided as a list.")

        if not values:
            raise ValueError("Feature values cannot be empty.")

        if not all(
            isinstance(value, (int, float))
            for value in values
        ):
            raise TypeError("All feature values must be numeric.")

        return MLFeatures(
            values=tuple(float(value) for value in values)
        )