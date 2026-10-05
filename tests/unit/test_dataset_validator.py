"""Tests for the QRShield-X dataset validator."""

from __future__ import annotations

import numpy as np

from qrshield_x.datasets.sample import QRDatasetSample
from qrshield_x.datasets.validator import QRDatasetValidator


def _sample(label: int) -> QRDatasetSample:
    return QRDatasetSample(
        image=np.zeros((69, 69), dtype=np.int16),
        label=label,
    )


def test_valid_dataset() -> None:
    """Validator should accept a valid dataset."""

    samples = [_sample(0), _sample(1), _sample(0)]

    result = QRDatasetValidator().validate(samples)

    assert result.valid is True
    assert result.sample_count == 3
    assert result.benign_count == 2
    assert result.phishing_count == 1
    assert result.errors == ()


def test_empty_dataset_is_structurally_valid() -> None:
    """Validator should report an empty dataset without structural errors."""

    result = QRDatasetValidator().validate([])

    assert result.valid is True
    assert result.sample_count == 0
    assert result.benign_count == 0
    assert result.phishing_count == 0
    assert result.errors == ()


def test_non_list_input_is_rejected() -> None:
    """Validator should reject non-list input."""

    result = QRDatasetValidator().validate(
        None,  # type: ignore[arg-type]
    )

    assert result.valid is False
    assert result.sample_count == 0
    assert result.benign_count == 0
    assert result.phishing_count == 0
    assert "Dataset samples must be provided as a list." in result.errors


def test_invalid_sample_type_is_reported() -> None:
    """Validator should report objects that are not dataset samples."""

    result = QRDatasetValidator().validate(
        [object()],  # type: ignore[list-item]
    )

    assert result.valid is False
    assert result.sample_count == 1
    assert result.benign_count == 0
    assert result.phishing_count == 0
    assert "Sample at index 0 is not a QRDatasetSample." in result.errors