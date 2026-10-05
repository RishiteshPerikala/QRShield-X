"""Ground-truth handling for the QRShield-X primary dataset."""

from __future__ import annotations

from qrshield_x.core.enums import Confidence, GroundTruthLabel
from qrshield_x.core.models import GroundTruth


class GroundTruthBuilder:
    """Convert dataset-provided labels into QRShield-X ground truth."""

    SOURCE = "Trad & Chehab QuishingDataset"
    STATUS = "DATASET_LABEL"
    VERIFICATION_METHOD = "dataset-provided-label"

    def build(self, label: int) -> GroundTruth:
        """Build ground truth from a dataset label."""

        if label == 0:
            ground_truth_label = GroundTruthLabel.BENIGN
        elif label == 1:
            ground_truth_label = GroundTruthLabel.MALICIOUS
        else:
            raise ValueError(
                "Dataset label must be either 0 or 1."
            )

        return GroundTruth(
            label=ground_truth_label,
            source=self.SOURCE,
            status=self.STATUS,
            verification_method=self.VERIFICATION_METHOD,
            verification_timestamp=None,
            confidence=Confidence.HIGH,
        )