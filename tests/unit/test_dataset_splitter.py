"""Tests for the QRShield-X dataset splitter."""

from __future__ import annotations

import numpy as np
import pytest

from qrshield_x.datasets.sample import QRDatasetSample
from qrshield_x.datasets.splitter import QRDatasetSplitter


def _samples() -> list[QRDatasetSample]:
    """Create a small balanced dataset for testing."""

    return [
        QRDatasetSample(
            image=np.zeros((69, 69), dtype=np.int16),
            label=index % 2,
        )
        for index in range(20)
    ]


def test_split_preserves_total_sample_count() -> None:
    """All samples should appear in exactly one split."""

    samples = _samples()

    result = QRDatasetSplitter().split(samples)

    assert (
        len(result.train)
        + len(result.validation)
        + len(result.test)
        == len(samples)
    )


def test_split_is_reproducible() -> None:
    """The same seed should produce the same split."""

    samples = _samples()
    splitter = QRDatasetSplitter()

    first = splitter.split(samples, random_seed=42)
    second = splitter.split(samples, random_seed=42)

    assert [sample.label for sample in first.train] == [
        sample.label for sample in second.train
    ]
    assert [sample.label for sample in first.validation] == [
        sample.label for sample in second.validation
    ]
    assert [sample.label for sample in first.test] == [
        sample.label for sample in second.test
    ]

    assert [id(sample) for sample in first.train] == [
        id(sample) for sample in second.train
    ]


def test_split_is_stratified() -> None:
    """Each split should contain both classes."""

    result = QRDatasetSplitter().split(_samples())

    for split in (
        result.train,
        result.validation,
        result.test,
    ):
        labels = {sample.label for sample in split}
        assert labels == {0, 1}


def test_invalid_ratios_are_rejected() -> None:
    """Split ratios must be positive and sum to one."""

    splitter = QRDatasetSplitter()
    samples = _samples()

    with pytest.raises(ValueError):
        splitter.split(
            samples,
            train_ratio=0.8,
            validation_ratio=0.2,
            test_ratio=0.2,
        )

    with pytest.raises(ValueError):
        splitter.split(
            samples,
            train_ratio=0.0,
            validation_ratio=0.5,
            test_ratio=0.5,
        )


def test_empty_dataset_is_rejected() -> None:
    """An empty dataset cannot be split."""

    with pytest.raises(ValueError):
        QRDatasetSplitter().split([])


def test_non_list_input_is_rejected() -> None:
    """Splitter should reject non-list input."""

    with pytest.raises(TypeError):
        QRDatasetSplitter().split(None)  # type: ignore[arg-type]