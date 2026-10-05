"""Loader for the QRShield-X primary QR dataset."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from qrshield_x.datasets.sample import QRDatasetSample


class QRDatasetLoader:
    """Load and validate the primary QRShield-X dataset."""

    def __init__(
        self,
        data_path: str | Path,
        labels_path: str | Path,
    ) -> None:
        self._data_path = Path(data_path)
        self._labels_path = Path(labels_path)

    def load(self) -> list[QRDatasetSample]:
        """Load QR images and labels into dataset samples."""

        if not self._data_path.exists():
            raise FileNotFoundError(
                f"QR dataset file not found: {self._data_path}"
            )

        if not self._labels_path.exists():
            raise FileNotFoundError(
                f"QR labels file not found: {self._labels_path}"
            )

        images = self._load_pickle(self._data_path)
        labels = self._load_pickle(self._labels_path)

        if not isinstance(images, np.ndarray):
            raise TypeError("QR dataset must be a NumPy array.")

        if not isinstance(labels, np.ndarray):
            raise TypeError("QR labels must be a NumPy array.")

        if len(images) != len(labels):
            raise ValueError(
                "Number of QR samples must match number of labels."
            )

        return [
            QRDatasetSample(
                image=image,
                label=int(label),
            )
            for image, label in zip(images, labels)
        ]

    @staticmethod
    def _load_pickle(path: Path) -> object:
        """Load a trusted local pickle dataset file."""

        with path.open("rb") as file:
            return pickle.load(file)  # noqa: S301