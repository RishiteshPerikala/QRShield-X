
"""Integration tests for trained baselines through the ML pipeline."""

import numpy as np
import pytest

from qrshield_x.experiments.logistic_regression import (
    LogisticRegressionBaseline,
)
from qrshield_x.experiments.random_forest import RandomForestBaseline
from qrshield_x.ml.baseline_adapter import BaselineMLAdapter
from qrshield_x.ml.model import MLPrediction
from qrshield_x.ml.pipeline import MLPipeline


def make_trained_baseline(model_kind: str):
    """Create a baseline trained on a small synthetic dataset."""

    features = np.zeros((8, 4761), dtype=np.float32)
    features[4:] = 1.0
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    if model_kind == "logistic_regression":
        model = LogisticRegressionBaseline(random_seed=42)
    elif model_kind == "random_forest":
        model = RandomForestBaseline(
            random_seed=42,
            n_estimators=10,
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    model.fit(features, labels)
    return model


@pytest.mark.parametrize(
    "model_kind",
    ["logistic_regression", "random_forest"],
)
def test_pipeline_predicts_with_trained_baseline(model_kind):
    """Verify feature extraction, inference, and baseline prediction."""

    baseline = make_trained_baseline(model_kind)
    adapter = BaselineMLAdapter(baseline)
    pipeline = MLPipeline(adapter)

    prediction = pipeline.predict([0.0] * 4761)

    assert isinstance(prediction, MLPrediction)
    assert prediction.label in {"BENIGN", "MALICIOUS"}
    assert 0.0 <= prediction.confidence <= 1.0


@pytest.mark.parametrize(
    "model_kind",
    ["logistic_regression", "random_forest"],
)
def test_pipeline_rejects_wrong_feature_count(model_kind):
    """Verify the baseline rejects input with the wrong feature count."""

    baseline = make_trained_baseline(model_kind)
    pipeline = MLPipeline(BaselineMLAdapter(baseline))

    with pytest.raises(ValueError, match="4761"):
        pipeline.predict([0.0] * 10)