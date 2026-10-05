from __future__ import annotations

import pytest

from qrshield_x.ml.feature_extractor import FeatureExtractor


def test_extract_builds_ml_features() -> None:
    extractor = FeatureExtractor()

    result = extractor.extract([1, 2.5, 3])

    assert result.values == (1.0, 2.5, 3.0)
    assert result.as_list() == [1.0, 2.5, 3.0]


def test_extract_rejects_empty_features() -> None:
    extractor = FeatureExtractor()

    with pytest.raises(ValueError):
        extractor.extract([])


def test_extract_rejects_non_numeric_features() -> None:
    extractor = FeatureExtractor()

    with pytest.raises(TypeError):
        extractor.extract([1, "invalid"])  # type: ignore[list-item]