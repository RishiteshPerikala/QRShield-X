"""Tests for QRShield-X model persistence."""

import numpy as np
import pytest

from qrshield_x.experiments.logistic_regression import (
    LogisticRegressionBaseline,
)
from qrshield_x.experiments.model_persistence import (  # type: ignore[reportMissingImports]
    ModelPersistence,
)


@pytest.fixture
def trained_model():
    """Create a small trained Logistic Regression model."""

    features = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.2],
            [0.9, 0.8],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [0, 0, 1, 1, 0, 1],
        dtype=np.int32,
    )

    return LogisticRegressionBaseline().fit(features, labels), features


def test_save_trained_model(trained_model, tmp_path):
    model, _ = trained_model
    destination = tmp_path / "models" / "baseline.joblib"

    result = ModelPersistence.save(model, destination)

    assert result == destination
    assert destination.is_file()


def test_load_saved_model(trained_model, tmp_path):
    model, _ = trained_model
    destination = tmp_path / "baseline.joblib"

    ModelPersistence.save(model, destination)
    loaded_model = ModelPersistence.load(destination)

    assert loaded_model.is_trained is True


def test_loaded_model_predictions_match(trained_model, tmp_path):
    model, features = trained_model
    destination = tmp_path / "baseline.joblib"

    ModelPersistence.save(model, destination)
    loaded_model = ModelPersistence.load(destination)

    expected = model.predict(features)
    actual = loaded_model.predict(features)

    assert np.array_equal(actual, expected)


def test_cannot_save_untrained_model(tmp_path):
    model = LogisticRegressionBaseline()

    with pytest.raises(
        RuntimeError,
        match="not been trained",
    ):
        ModelPersistence.save(
            model,
            tmp_path / "untrained.joblib",
        )


def test_loading_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError):
        ModelPersistence.load(missing_path)


def test_loaded_model_is_trained(trained_model, tmp_path):
    model, _ = trained_model
    destination = tmp_path / "baseline.joblib"

    ModelPersistence.save(model, destination)
    loaded_model = ModelPersistence.load(destination)

    assert loaded_model.is_trained