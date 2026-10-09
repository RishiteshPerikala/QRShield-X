
"""Tests for the QRShield-X Logistic Regression baseline."""

from __future__ import annotations

import numpy as np
import pytest

from qrshield_x.experiments.logistic_regression import (
    LogisticRegressionBaseline,
)


def _training_data():
    """Create a small deterministic binary-classification dataset."""

    features = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [1.0, 1.0, 1.0],
            [0.9, 1.0, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    return features, labels


def test_model_starts_untrained() -> None:
    """A newly created model should not be trained."""

    model = LogisticRegressionBaseline()

    assert not model.is_trained


def test_fit_trains_model() -> None:
    """Fitting should mark the model as trained."""

    features, labels = _training_data()

    model = LogisticRegressionBaseline().fit(features, labels)

    assert model.is_trained
    assert set(model.classes_) == {0, 1}


def test_predict_returns_binary_labels() -> None:
    """Predictions should use the learned class labels."""

    features, labels = _training_data()
    model = LogisticRegressionBaseline().fit(features, labels)

    predictions = model.predict(features)

    assert predictions.shape == (4,)
    assert set(predictions).issubset({0, 1})


def test_predict_proba_returns_valid_probabilities() -> None:
    """Predicted probabilities should be valid and sum to one."""

    features, labels = _training_data()
    model = LogisticRegressionBaseline().fit(features, labels)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (4, 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_prediction_before_training_raises_error() -> None:
    """Inference must not run before model training."""

    model = LogisticRegressionBaseline()

    with pytest.raises(RuntimeError, match="has not been trained"):
        model.predict(np.zeros((1, 3), dtype=np.float32))


def test_probability_prediction_before_training_raises_error() -> None:
    """Probability inference must require a trained model."""

    model = LogisticRegressionBaseline()

    with pytest.raises(RuntimeError, match="has not been trained"):
        model.predict_proba(np.zeros((1, 3), dtype=np.float32))


@pytest.mark.parametrize(
    ("max_iterations", "c"),
    [
        (0, 1.0),
        (-1, 1.0),
        (100, 0.0),
        (100, -1.0),
    ],
)
def test_invalid_configuration_is_rejected(
    max_iterations: int,
    c: float,
) -> None:
    """Invalid model configuration should be rejected."""

    with pytest.raises(ValueError):
        LogisticRegressionBaseline(
            max_iterations=max_iterations,
            c=c,
        )