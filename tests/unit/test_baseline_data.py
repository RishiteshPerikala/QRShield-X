"""Tests for QRShield-X baseline data preparation."""

from __future__ import annotations

import numpy as np
import pytest

from qrshield_x.datasets.sample import QRDatasetSample
from qrshield_x.datasets.splitter import DatasetSplit
from qrshield_x.experiments.baseline_data import BaselineDataPreparer


def _sample(label: int) -> QRDatasetSample:
    """Create a deterministic QR dataset sample."""

    image = np.zeros((69, 69), dtype=np.int16)
    image[0, 0] = label

    return QRDatasetSample(
        image=image,
        label=label,
    )


def _split() -> DatasetSplit:
    """Create a small deterministic dataset split."""

    return DatasetSplit(
        train=[_sample(0), _sample(1)],
        validation=[_sample(0)],
        test=[_sample(1)],
    )


def test_prepare_creates_expected_feature_shapes() -> None:
    """Prepared feature matrices should have 4761 columns."""

    result = BaselineDataPreparer().prepare(_split())

    assert result.x_train.shape == (2, 4761)
    assert result.x_validation.shape == (1, 4761)
    assert result.x_test.shape == (1, 4761)


def test_prepare_creates_expected_label_shapes() -> None:
    """Prepared label arrays should match split sizes."""

    result = BaselineDataPreparer().prepare(_split())

    assert result.y_train.shape == (2,)
    assert result.y_validation.shape == (1,)
    assert result.y_test.shape == (1,)


def test_prepare_uses_float32_features() -> None:
    """Feature matrices should use memory-efficient float32 values."""

    result = BaselineDataPreparer().prepare(_split())

    assert result.x_train.dtype == np.float32


def test_prepare_uses_integer_labels() -> None:
    """Target arrays should contain integer labels."""

    result = BaselineDataPreparer().prepare(_split())

    assert result.y_train.dtype == np.int32


def test_prepare_preserves_labels() -> None:
    """Dataset labels must remain separate from feature vectors."""

    result = BaselineDataPreparer().prepare(_split())

    assert result.y_train.tolist() == [0, 1]
    assert result.y_validation.tolist() == [0]
    assert result.y_test.tolist() == [1]


def test_prepare_preserves_pixel_features() -> None:
    """Pixel values should be preserved in the feature matrix."""

    result = BaselineDataPreparer().prepare(_split())

    assert result.x_train[0, 0] == 0.0
    assert result.x_train[1, 0] == 1.0


def test_invalid_split_type_is_rejected() -> None:
    """Invalid split objects should be rejected."""

    with pytest.raises(TypeError):
        BaselineDataPreparer().prepare(None)  # type: ignore[arg-type]