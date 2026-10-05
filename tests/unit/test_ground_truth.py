"""Tests for QRShield-X ground-truth handling."""

from __future__ import annotations

import pytest

from qrshield_x.core.enums import Confidence, GroundTruthLabel
from qrshield_x.datasets.ground_truth import GroundTruthBuilder


def test_benign_label_creates_benign_ground_truth() -> None:
    """Dataset label 0 should map to BENIGN."""

    result = GroundTruthBuilder().build(0)

    assert result.label == GroundTruthLabel.BENIGN


def test_phishing_label_creates_malicious_ground_truth() -> None:
    """Dataset label 1 should map to MALICIOUS."""

    result = GroundTruthBuilder().build(1)

    assert result.label == GroundTruthLabel.MALICIOUS


def test_ground_truth_metadata_is_preserved() -> None:
    """Ground-truth provenance metadata should be explicit."""

    result = GroundTruthBuilder().build(1)

    assert result.source == "Trad & Chehab QuishingDataset"
    assert result.status == "DATASET_LABEL"
    assert result.verification_method == "dataset-provided-label"
    assert result.confidence == Confidence.HIGH


def test_ground_truth_does_not_claim_verification_timestamp() -> None:
    """Dataset labels should not claim independent verification."""

    result = GroundTruthBuilder().build(0)

    assert result.verification_timestamp is None


@pytest.mark.parametrize("label", [-1, 2, 99])
def test_invalid_dataset_label_is_rejected(label: int) -> None:
    """Invalid dataset labels should be rejected."""

    with pytest.raises(ValueError):
        GroundTruthBuilder().build(label)