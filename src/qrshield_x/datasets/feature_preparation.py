"""Feature preparation for the QRShield-X primary dataset."""

from __future__ import annotations

import numpy as np

from qrshield_x.datasets.sample import QRDatasetSample
from qrshield_x.ml.features import FeatureBuilder, MLFeatures


class QRFeaturePreparer:
    """Convert QR dataset samples into ML feature representations."""

    def __init__(self) -> None:
        self._feature_builder = FeatureBuilder()

    def prepare(
        self,
        sample: QRDatasetSample,
    ) -> MLFeatures:
        """Convert one QR dataset sample into ML features."""

        if not isinstance(sample, QRDatasetSample):
            raise TypeError(
                "Sample must be a QRDatasetSample."
            )

        values = sample.image.astype(
            np.float32,
            copy=False,
        ).reshape(-1).tolist()

        return self._feature_builder.build(values)