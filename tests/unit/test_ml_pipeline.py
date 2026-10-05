from __future__ import annotations

import pytest

from qrshield_x.ml.model import DummyMLModel
from qrshield_x.ml.pipeline import MLPipeline


def test_pipeline_extracts_and_predicts() -> None:
    pipeline = MLPipeline(DummyMLModel())

    result = pipeline.predict([1.0, 2.0, 3.0])

    assert result.label == "UNKNOWN"
    assert result.confidence == 0.0


def test_pipeline_rejects_empty_features() -> None:
    pipeline = MLPipeline(DummyMLModel())

    with pytest.raises(ValueError):
        pipeline.predict([])


def test_pipeline_rejects_non_numeric_features() -> None:
    pipeline = MLPipeline(DummyMLModel())

    with pytest.raises(TypeError):
        pipeline.predict([1.0, "invalid"])  # type: ignore[list-item]