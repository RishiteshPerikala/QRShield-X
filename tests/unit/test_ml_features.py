import pytest

from qrshield_x.ml.features import FeatureBuilder, MLFeatures


def test_feature_builder_creates_features():
    builder = FeatureBuilder()

    result = builder.build([1, 2.5, 3])

    assert isinstance(result, MLFeatures)
    assert result.values == (1.0, 2.5, 3.0)


def test_features_can_be_converted_to_list():
    builder = FeatureBuilder()

    result = builder.build([1, 2.5])

    assert result.as_list() == [1.0, 2.5]


def test_empty_features_are_rejected():
    builder = FeatureBuilder()

    with pytest.raises(ValueError):
        builder.build([])


def test_non_list_features_are_rejected():
    builder = FeatureBuilder()

    with pytest.raises(TypeError):
        builder.build((1, 2))


def test_non_numeric_features_are_rejected():
    builder = FeatureBuilder()

    with pytest.raises(TypeError):
        builder.build([1, "invalid"])


def test_features_are_immutable():
    builder = FeatureBuilder()

    result = builder.build([1, 2])

    with pytest.raises(AttributeError):
        result.values = (3.0, 4.0)