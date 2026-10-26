"""Tests for exporting validation predictions."""

import csv

import pytest

from qrshield_x.experiments.runner import BaselineExperimentRunner


def test_save_validation_predictions_creates_csv(tmp_path) -> None:
    output_path = tmp_path / "predictions" / "model_validation_predictions.csv"

    BaselineExperimentRunner._save_predictions(
        output_path,
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 1, 0],
        y_probability_malicious=[0.1, 0.9, 0.7, 0.4],
    )

    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 4
    assert rows[0]["y_true"] == "0"
    assert rows[0]["y_pred"] == "0"
    assert float(rows[0]["y_probability_malicious"]) == pytest.approx(0.1)
    assert rows[1]["y_true"] == "1"
    assert float(rows[1]["y_probability_malicious"]) == pytest.approx(0.9)


def test_save_validation_predictions_rejects_mismatched_lengths(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        BaselineExperimentRunner._save_predictions(
            tmp_path / "predictions.csv",
            y_true=[0, 1],
            y_pred=[0],
            y_probability_malicious=[0.2, 0.8],
        )
