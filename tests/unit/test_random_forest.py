"""Tests for the Random Forest baseline."""

import numpy as np
import pytest

from qrshield_x.experiments.random_forest import (
    RandomForestBaseline,
)


@pytest.fixture
def training_data():
    """Provide a small deterministic binary classification dataset."""

    features = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [0, 0, 1, 1, 0, 0, 1, 1],
        dtype=np.int32,
    )
    return features, labels


def test_model_starts_untrained():
    model = RandomForestBaseline()
    assert model.is_trained is False


def test_fit_returns_model(training_data):
    features, labels = training_data
    model = RandomForestBaseline(n_estimators=10)

    result = model.fit(features, labels)

    assert result is model
    assert model.is_trained is True


def test_predict_returns_binary_labels(training_data):
    features, labels = training_data
    model = RandomForestBaseline(n_estimators=10)
    model.fit(features, labels)

    predictions = model.predict(features)

    assert predictions.shape == labels.shape
    assert set(predictions).issubset({0, 1})


def test_predict_proba_has_valid_shape_and_values(training_data):
    features, _ = training_data
    model = RandomForestBaseline(n_estimators=10)
    model.fit(features, np.asarray([0, 0, 1, 1, 0, 0, 1, 1]))

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (len(features), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_predict_before_fit_raises_error():
    model = RandomForestBaseline()

    with pytest.raises(RuntimeError, match="has not been trained"):
        model.predict(np.asarray([[0.0, 1.0]]))


def test_predict_proba_before_fit_raises_error():
    model = RandomForestBaseline()

    with pytest.raises(RuntimeError, match="has not been trained"):
        model.predict_proba(np.asarray([[0.0, 1.0]]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"n_estimators": -1},
        {"max_depth": 0},
        {"max_depth": -2},
    ],
)
def test_invalid_configuration_raises_error(kwargs):
    with pytest.raises(ValueError):
        RandomForestBaseline(**kwargs)


def test_classes_property_requires_training():
    model = RandomForestBaseline()

    with pytest.raises(RuntimeError, match="has not been trained"):
        _ = model.classes_


def test_classes_property_after_training(training_data):
    features, labels = training_data
    model = RandomForestBaseline(n_estimators=10)
    model.fit(features, labels)

    assert np.array_equal(model.classes_, np.asarray([0, 1]))
