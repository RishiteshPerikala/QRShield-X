"""Tests for the QRShield-X baseline ML adapter."""

import numpy as np
import pytest

from qrshield_x.experiments.logistic_regression import (
    LogisticRegressionBaseline,
)
from qrshield_x.experiments.random_forest import (
    RandomForestBaseline,
)
from qrshield_x.ml.baseline_adapter import BaselineMLAdapter
from qrshield_x.ml.model import MLPrediction


@pytest.fixture
def training_data():
    """Create a small dataset with 4761 features per sample."""

    features = np.zeros((8, 4761), dtype=np.float32)
    features[4:, :] = 1.0

    labels = np.asarray(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=np.int32,
    )

    return features, labels


@pytest.fixture
def trained_logistic_model(training_data):
    features, labels = training_data
    return LogisticRegressionBaseline().fit(features, labels)


@pytest.fixture
def trained_forest_model(training_data):
    features, labels = training_data
    return RandomForestBaseline(
        n_estimators=10,
        random_seed=42,
    ).fit(features, labels)


def test_adapter_rejects_untrained_logistic_model():
    with pytest.raises(RuntimeError, match="must be trained"):
        BaselineMLAdapter(LogisticRegressionBaseline())


def test_adapter_rejects_untrained_random_forest():
    with pytest.raises(RuntimeError, match="must be trained"):
        BaselineMLAdapter(RandomForestBaseline())


def test_adapter_rejects_unsupported_model():
    with pytest.raises(TypeError, match="supported QRShield-X baseline"):
        BaselineMLAdapter(object())


@pytest.mark.parametrize(
    "model_fixture",
    ["trained_logistic_model", "trained_forest_model"],
)
def test_predict_returns_ml_prediction(
    request,
    model_fixture,
    training_data,
):
    model = request.getfixturevalue(model_fixture)
    features, _ = training_data
    adapter = BaselineMLAdapter(model)

    result = adapter.predict(features[0].tolist())

    assert isinstance(result, MLPrediction)
    assert result.label in {"BENIGN", "MALICIOUS"}
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize(
    "model_fixture",
    ["trained_logistic_model", "trained_forest_model"],
)
def test_predict_identifies_clear_training_examples(
    request,
    model_fixture,
    training_data,
):
    model = request.getfixturevalue(model_fixture)
    features, _ = training_data
    adapter = BaselineMLAdapter(model)

    benign = adapter.predict(features[0].tolist())
    malicious = adapter.predict(features[-1].tolist())

    assert benign.label == "BENIGN"
    assert malicious.label == "MALICIOUS"


def test_predict_rejects_non_list(trained_logistic_model):
    adapter = BaselineMLAdapter(trained_logistic_model)

    with pytest.raises(TypeError, match="provided as a list"):
        adapter.predict(np.zeros(4761, dtype=np.float32))


def test_predict_rejects_empty_features(trained_logistic_model):
    adapter = BaselineMLAdapter(trained_logistic_model)

    with pytest.raises(ValueError, match="cannot be empty"):
        adapter.predict([])


@pytest.mark.parametrize("length", [2, 4760, 4762])
def test_predict_rejects_wrong_feature_length(
    trained_logistic_model,
    length,
):
    adapter = BaselineMLAdapter(trained_logistic_model)

    with pytest.raises(ValueError, match="exactly 4761"):
        adapter.predict([0.0] * length)


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf"), True, "invalid"],
)
def test_predict_rejects_invalid_feature_values(
    trained_logistic_model,
    bad_value,
):
    adapter = BaselineMLAdapter(trained_logistic_model)
    features = [0.0] * 4761
    features[0] = bad_value

    with pytest.raises(TypeError, match="finite numeric values"):
        adapter.predict(features)


def test_confidence_matches_predicted_class_probability(
    trained_logistic_model,
    training_data,
):
    features, _ = training_data
    adapter = BaselineMLAdapter(trained_logistic_model)

    result = adapter.predict(features[-1].tolist())

    probabilities = trained_logistic_model.predict_proba(
        features[-1:].astype(np.float32)
    )[0]
    classes = trained_logistic_model.classes_
    predicted_label = int(
        trained_logistic_model.predict(
            features[-1:].astype(np.float32)
        )[0]
    )
    class_index = int(np.flatnonzero(classes == predicted_label)[0])

    assert result.confidence == pytest.approx(
        float(probabilities[class_index])
    )