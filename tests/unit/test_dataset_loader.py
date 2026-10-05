"""Tests for the QRShield-X dataset loader."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from qrshield_x.datasets.loader import QRDatasetLoader


def _write_pickle(path, value) -> None:
    with path.open("wb") as file:
        pickle.dump(value, file)


def test_load_valid_dataset(tmp_path) -> None:
    """Loader should convert valid arrays into dataset samples."""

    images = np.zeros((2, 69, 69), dtype=np.int16)
    labels = np.array([0, 1], dtype=np.int32)

    data_path = tmp_path / "images.pickle"
    labels_path = tmp_path / "labels.pickle"

    _write_pickle(data_path, images)
    _write_pickle(labels_path, labels)

    loader = QRDatasetLoader(data_path, labels_path)
    samples = loader.load()

    assert len(samples) == 2
    assert samples[0].label == 0
    assert samples[1].label == 1
    assert samples[0].image.shape == (69, 69)


def test_missing_dataset_file_raises(tmp_path) -> None:
    """Missing QR dataset file should raise FileNotFoundError."""

    labels_path = tmp_path / "labels.pickle"
    _write_pickle(
        labels_path,
        np.array([0], dtype=np.int32),
    )

    loader = QRDatasetLoader(
        tmp_path / "missing.pickle",
        labels_path,
    )

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_missing_labels_file_raises(tmp_path) -> None:
    """Missing labels file should raise FileNotFoundError."""

    data_path = tmp_path / "images.pickle"
    _write_pickle(
        data_path,
        np.zeros((1, 69, 69), dtype=np.int16),
    )

    loader = QRDatasetLoader(
        data_path,
        tmp_path / "missing_labels.pickle",
    )

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_non_numpy_dataset_raises(tmp_path) -> None:
    """Non-NumPy dataset content should be rejected."""

    data_path = tmp_path / "images.pickle"
    labels_path = tmp_path / "labels.pickle"

    _write_pickle(data_path, [[0, 1]])
    _write_pickle(
        labels_path,
        np.array([0], dtype=np.int32),
    )

    loader = QRDatasetLoader(data_path, labels_path)

    with pytest.raises(TypeError):
        loader.load()


def test_mismatched_sample_and_label_counts_raise(tmp_path) -> None:
    """Different sample and label counts should be rejected."""

    data_path = tmp_path / "images.pickle"
    labels_path = tmp_path / "labels.pickle"

    _write_pickle(
        data_path,
        np.zeros((2, 69, 69), dtype=np.int16),
    )
    _write_pickle(
        labels_path,
        np.array([0], dtype=np.int32),
    )

    loader = QRDatasetLoader(data_path, labels_path)

    with pytest.raises(ValueError):
        loader.load()