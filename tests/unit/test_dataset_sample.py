from __future__ import annotations

import numpy as np
import pytest

from qrshield_x.datasets.sample import QRDatasetSample


def test_valid_qr_dataset_sample() -> None:
    image = np.zeros((69, 69), dtype=np.int16)

    sample = QRDatasetSample(
        image=image,
        label=0,
    )

    assert sample.image.shape == (69, 69)
    assert sample.label == 0


def test_rejects_invalid_image_shape() -> None:
    image = np.zeros((28, 28), dtype=np.int16)

    with pytest.raises(ValueError):
        QRDatasetSample(
            image=image,
            label=0,
        )


def test_rejects_invalid_label() -> None:
    image = np.zeros((69, 69), dtype=np.int16)

    with pytest.raises(ValueError):
        QRDatasetSample(
            image=image,
            label=2,
        )


def test_rejects_non_numpy_image() -> None:
    with pytest.raises(TypeError):
        QRDatasetSample(
            image=[[0] * 69] * 69,  # type: ignore[arg-type]
            label=0,
        )


def test_accepts_both_dataset_labels() -> None:
    image = np.zeros((69, 69), dtype=np.int16)

    benign = QRDatasetSample(image=image, label=0)
    phishing = QRDatasetSample(image=image, label=1)

    assert benign.label == 0
    assert phishing.label == 1