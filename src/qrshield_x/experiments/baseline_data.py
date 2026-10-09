"""Baseline training data preparation for QRShield-X."""

from __future__ import annotations

import numpy as np

from qrshield_x.datasets.feature_preparation import QRFeaturePreparer
from qrshield_x.datasets.sample import QRDatasetSample
from qrshield_x.datasets.splitter import DatasetSplit


class BaselineData:
    """Prepared NumPy arrays for baseline ML experiments."""

    def __init__(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.x_test = x_test
        self.y_test = y_test


class BaselineDataPreparer:
    """Convert a dataset split into scikit-learn-ready arrays."""

    def __init__(self) -> None:
        self._feature_preparer = QRFeaturePreparer()

    def prepare(
        self,
        split: DatasetSplit,
    ) -> BaselineData:
        """Prepare train, validation, and test arrays."""

        if not isinstance(split, DatasetSplit):
            raise TypeError(
                "Split must be a DatasetSplit."
            )

        return BaselineData(
            x_train=self._features(split.train),
            y_train=self._labels(split.train),
            x_validation=self._features(split.validation),
            y_validation=self._labels(split.validation),
            x_test=self._features(split.test),
            y_test=self._labels(split.test),
        )

    def _features(
        self,
        samples: list[QRDatasetSample],
    ) -> np.ndarray:
        """Convert samples into a float32 feature matrix."""

        if not samples:
            raise ValueError("Cannot prepare an empty sample collection.")

        return np.asarray(
            [
                self._feature_preparer.prepare(sample).values
                for sample in samples
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _labels(
        samples: list[QRDatasetSample],
    ) -> np.ndarray:
        """Convert sample labels into an integer target array."""

        if not samples:
            raise ValueError("Cannot prepare labels from empty samples.")

        return np.asarray(
            [sample.label for sample in samples],
            dtype=np.int32,
        )