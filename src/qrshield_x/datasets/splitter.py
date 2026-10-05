"""Dataset splitting utilities for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qrshield_x.datasets.sample import QRDatasetSample


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test dataset split."""

    train: list[QRDatasetSample]
    validation: list[QRDatasetSample]
    test: list[QRDatasetSample]


class QRDatasetSplitter:
    """Create deterministic stratified dataset splits."""

    def split(
        self,
        samples: list[QRDatasetSample],
        *,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> DatasetSplit:
        """Split samples into stratified train, validation, and test sets."""

        if not isinstance(samples, list):
            raise TypeError("Samples must be provided as a list.")

        if not samples:
            raise ValueError("Cannot split an empty dataset.")

        if any(
            not isinstance(sample, QRDatasetSample)
            for sample in samples
        ):
            raise TypeError(
                "All samples must be QRDatasetSample instances."
            )

        if any(
            ratio <= 0
            for ratio in (
                train_ratio,
                validation_ratio,
                test_ratio,
            )
        ):
            raise ValueError("All split ratios must be positive.")

        if not np.isclose(
            train_ratio + validation_ratio + test_ratio,
            1.0,
        ):
            raise ValueError("Split ratios must sum to 1.0.")

        rng = np.random.default_rng(random_seed)

        labels = np.array([sample.label for sample in samples])

        train_indices: list[int] = []
        validation_indices: list[int] = []
        test_indices: list[int] = []

        for label in np.unique(labels):
            label_indices = np.where(labels == label)[0]
            rng.shuffle(label_indices)

            total = len(label_indices)
            train_count = int(total * train_ratio)
            validation_count = int(total * validation_ratio)

            train_end = train_count
            validation_end = train_count + validation_count

            train_indices.extend(label_indices[:train_end])
            validation_indices.extend(
                label_indices[train_end:validation_end]
            )
            test_indices.extend(label_indices[validation_end:])

        rng.shuffle(train_indices)
        rng.shuffle(validation_indices)
        rng.shuffle(test_indices)

        return DatasetSplit(
            train=[samples[index] for index in train_indices],
            validation=[samples[index] for index in validation_indices],
            test=[samples[index] for index in test_indices],
        )