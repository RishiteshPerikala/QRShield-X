"""Visualization utilities for QRShield-X experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"


MODELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def load_predictions(model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test-set predictions exported by the experiment runner."""

    path = (
        PREDICTIONS_DIR
        / f"{model_name}_test_predictions.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[float] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "y_true",
            "y_pred",
            "y_probability_malicious",
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                f"Prediction file is missing required columns: "
                f"{required_columns}"
            )

        for row in reader:
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
            probabilities.append(
                float(row["y_probability_malicious"])
            )

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(probabilities),
    )


def load_results() -> dict:
    """Load the baseline experiment results."""

    results_path = (
        RESULTS_DIR / "baseline_results.json"
    )

    if not results_path.is_file():
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def plot_roc_curve() -> None:
    """Generate ROC curve comparison for the test set."""

    plt.figure(figsize=(8, 6))

    for model_name, display_name in MODELS.items():
        (
            y_true,
            _,
            probabilities,
        ) = load_predictions(model_name)

        fpr, tpr, _ = roc_curve(
            y_true,
            probabilities,
        )

        results = load_results()

        auc = results[
            "models"
        ][
            model_name
        ][
            "test_metrics"
        ][
            "roc_auc"
        ]

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{display_name} (AUC = {auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR / "roc_curve_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_confusion_matrix(
    model_name: str,
) -> None:
    """Generate a confusion matrix heatmap."""

    (
        y_true,
        y_pred,
        _,
    ) = load_predictions(model_name)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    display_name = MODELS[model_name]

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
    )

    figure.colorbar(image)

    axis.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=[
            "Benign",
            "Malicious",
        ],
        yticklabels=[
            "Benign",
            "Malicious",
        ],
        xlabel="Predicted Label",
        ylabel="Actual Label",
        title=(
            f"Confusion Matrix — "
            f"{display_name} Test Set"
        ),
    )

    threshold = matrix.max() / 2.0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
                fontsize=14,
                fontweight="bold",
            )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"confusion_matrix_{model_name}_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_metric_comparison() -> None:
    """Generate a test-set metric comparison chart."""

    results = load_results()

    metric_names = [
        "accuracy",
        "precision_malicious",
        "recall_malicious",
        "f1_malicious",
        "roc_auc",
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    values = {
        model_name: [
            results[
                "models"
            ][
                model_name
            ][
                "test_metrics"
            ][metric]
            for metric in metric_names
        ]
        for model_name in MODELS
    }

    x = np.arange(
        len(metric_labels)
    )

    width = 0.35

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.bar(
        x - width / 2,
        values["logistic_regression"],
        width,
        label="Logistic Regression",
    )

    axis.bar(
        x + width / 2,
        values["random_forest"],
        width,
        label="Random Forest",
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_xlabel(
        "Evaluation Metric"
    )

    axis.set_title(
        "Baseline Model Comparison — Test Set"
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        metric_labels
    )

    axis.set_ylim(
        0,
        1.0
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    for index, model_name in enumerate(
        MODELS
    ):
        model_values = values[
            model_name
        ]

        offset = (
            -width / 2
            if model_name == "logistic_regression"
            else width / 2
        )

        for metric_index, value in enumerate(
            model_values
        ):
            axis.text(
                metric_index + offset,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "model_metric_comparison_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def main() -> None:
    """Generate all QRShield-X test-set figures."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Generating QRShield-X test-set figures..."
    )

    plot_roc_curve()

    plot_confusion_matrix(
        "logistic_regression"
    )

    plot_confusion_matrix(
        "random_forest"
    )

    plot_metric_comparison()

    print(
        "\nAll figures generated successfully."
    )


if __name__ == "__main__":
    main()