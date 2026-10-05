"""Dataset sample models for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QRDatasetSample:
    """A single labeled QR sample from the primary dataset."""

    image: np.ndarray
    label: int

    def __post_init__(self) -> None:
        """Validate the dataset sample."""

        if not isinstance(self.image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")

        if self.image.shape != (69, 69):
            raise ValueError("QR image must have shape (69, 69).")

        if self.label not in {0, 1}:
            raise ValueError("Label must be either 0 or 1.")

        if not np.issubdtype(self.image.dtype, np.number):
            raise TypeError("QR image must contain numeric values.")