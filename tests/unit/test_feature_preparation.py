"""Tests for QRShield-X feature preparation."""

from __future__ import annotations

import numpy as np
import pytest

from qrshield_x.datasets.feature_preparation import QRFeaturePreparer
from qrshield_x.datasets.sample import QRDatasetSample


def _sample() -> QRDatasetSample:
    """Create a deterministic QR sample for testing."""

    image = np.zeros((69, 69), dtype=np.int16)
    image[0, 0] = 1
    image[10, 20] = 1

    return QRDatasetSample(
        image=image,
        label=1,
    )


def test_prepare_returns_expected_feature_count() -> None:
    """A 69x69 QR image should produce 4761 features."""

    result = QRFeaturePreparer().prepare(_sample())

    assert len(result.values) == 69 * 69
    assert len(result.values) == 4761


def test_prepare_converts_values_to_float() -> None:
    """Prepared feature values should be floating-point values."""

    result = QRFeaturePreparer().prepare(_sample())

    assert all(isinstance(value, float) for value in result.values)


def test_prepare_preserves_pixel_values() -> None:
    """Flattening should preserve the original pixel values."""

    result = QRFeaturePreparer().prepare(_sample())

    assert result.values[0] == 1.0
    assert result.values[10 * 69 + 20] == 1.0
    assert sum(result.values) == 2.0


def test_prepare_does_not_include_label() -> None:
    """The dataset label must not appear in the feature vector."""

    sample = _sample()
    result = QRFeaturePreparer().prepare(sample)

    assert len(result.values) == 4761
    assert result.values[-1] == float(sample.image[-1, -1])


def test_invalid_sample_type_is_rejected() -> None:
    """Feature preparation should reject invalid sample types."""

    with pytest.raises(TypeError):
        QRFeaturePreparer().prepare(None)  # type: ignore[arg-type]