from __future__ import annotations

import pytest

from qrshield_x.ml.features import FeatureBuilder
from qrshield_x.ml.inference import MLInference
from qrshield_x.ml.model import DummyMLModel


def test_predict_uses_model_with_features() -> None:
    features = FeatureBuilder().build([1.0, 2.0, 3.0])
    inference = MLInference(DummyMLModel())

    result = inference.predict(features)

    assert result.label == "UNKNOWN"
    assert result.confidence == 0.0


def test_predict_rejects_invalid_features() -> None:
    inference = MLInference(DummyMLModel())

    with pytest.raises(TypeError):
        inference.predict([1.0, 2.0])  # type: ignore[arg-type]


def test_predict_returns_prediction_object() -> None:
    features = FeatureBuilder().build([0.5, 1.5])
    inference = MLInference(DummyMLModel())

    result = inference.predict(features)

    assert result.__class__.__name__ == "MLPrediction"