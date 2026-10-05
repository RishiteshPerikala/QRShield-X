import pytest

from qrshield_x.ml.model import DummyMLModel, MLPrediction


def test_dummy_model_returns_prediction():
    model = DummyMLModel()

    result = model.predict([0.1, 0.2, 0.3])

    assert isinstance(result, MLPrediction)
    assert result.label == "UNKNOWN"
    assert result.confidence == 0.0


def test_dummy_model_rejects_empty_features():
    model = DummyMLModel()

    with pytest.raises(ValueError):
        model.predict([])


def test_dummy_model_rejects_non_list_features():
    model = DummyMLModel()

    with pytest.raises(TypeError):
        model.predict((0.1, 0.2))


def test_dummy_model_rejects_non_numeric_features():
    model = DummyMLModel()

    with pytest.raises(TypeError):
        model.predict([0.1, "invalid"])


def test_prediction_is_immutable():
    prediction = MLPrediction(
        label="BENIGN",
        confidence=0.9,
    )

    with pytest.raises(AttributeError):
        prediction.label = "MALICIOUS"