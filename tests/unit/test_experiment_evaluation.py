
"""Unit tests for experiment evaluation metrics."""

import pytest

from qrshield_x.experiments.evaluation import ExperimentEvaluator


def test_evaluator_calculates_metrics_and_confusion_matrix() -> None:
    evaluator = ExperimentEvaluator()

    result = evaluator.evaluate(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 0],
        y_probability_malicious=[0.1, 0.7, 0.8, 0.4],
    )

    assert result["accuracy"] == 0.5
    assert result["precision_malicious"] == pytest.approx(0.5)
    assert result["recall_malicious"] == pytest.approx(0.5)
    assert result["f1_malicious"] == pytest.approx(0.5)

    matrix = result["confusion_matrix"]
    assert matrix["true_negative"] == 1
    assert matrix["false_positive"] == 1
    assert matrix["false_negative"] == 1
    assert matrix["true_positive"] == 1

    assert result["roc_auc"] == pytest.approx(0.75)


def test_evaluator_handles_no_predicted_malicious_samples() -> None:
    result = ExperimentEvaluator().evaluate(
        y_true=[0, 1, 1],
        y_pred=[0, 0, 0],
    )

    assert result["precision_malicious"] == 0.0
    assert result["recall_malicious"] == 0.0
    assert result["f1_malicious"] == 0.0
    assert result["roc_auc"] is None


def test_roc_auc_is_none_when_actual_labels_contain_one_class() -> None:
    result = ExperimentEvaluator().evaluate(
        y_true=[0, 0, 0],
        y_pred=[0, 1, 0],
        y_probability_malicious=[0.1, 0.8, 0.2],
    )

    assert result["roc_auc"] is None


def test_evaluator_rejects_mismatched_prediction_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        ExperimentEvaluator().evaluate(
            y_true=[0, 1],
            y_pred=[1],
        )


def test_evaluator_rejects_invalid_class_labels() -> None:
    with pytest.raises(ValueError, match="only labels 0 and 1"):
        ExperimentEvaluator().evaluate(
            y_true=[0, 2],
            y_pred=[0, 1],
        )


def test_evaluator_rejects_mismatched_probability_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ExperimentEvaluator().evaluate(
            y_true=[0, 1],
            y_pred=[0, 1],
            y_probability_malicious=[0.2],
        )


def test_evaluator_rejects_probabilities_outside_valid_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ExperimentEvaluator().evaluate(
            y_true=[0, 1],
            y_pred=[0, 1],
            y_probability_malicious=[0.2, 1.5],
        )

def test_evaluator_calculates_additional_metrics() -> None:
    result = ExperimentEvaluator().evaluate(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 0],
        y_probability_malicious=[0.1, 0.7, 0.8, 0.4],
    )

    assert result["specificity"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.5)
    assert result["false_negative_rate"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["average_precision"] == pytest.approx(0.8333333333)


def test_probability_metrics_are_none_for_single_actual_class() -> None:
    result = ExperimentEvaluator().evaluate(
        y_true=[0, 0, 0],
        y_pred=[0, 1, 0],
        y_probability_malicious=[0.1, 0.8, 0.2],
    )

    assert result["roc_auc"] is None
    assert result["average_precision"] is None
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == pytest.approx(1 / 3)
    assert result["false_negative_rate"] is None
    assert result["balanced_accuracy"] is None


def test_evaluator_rejects_non_finite_probabilities() -> None:
    with pytest.raises(ValueError, match="finite"):
        ExperimentEvaluator().evaluate(
            y_true=[0, 1],
            y_pred=[0, 1],
            y_probability_malicious=[0.2, float("nan")],
        )


def test_evaluator_rejects_non_1d_probabilities() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        ExperimentEvaluator().evaluate(
            y_true=[0, 1],
            y_pred=[0, 1],
            y_probability_malicious=[[0.2], [0.8]],
        )
