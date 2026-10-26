
"""Generate validation confusion matrices and ROC curves for QRShield-X."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "predictions"
)
PLOTS_DIR = PROJECT_ROOT / "experiments" / "results" / "plots"

MODELS = {
    "Logistic Regression": (
        PREDICTIONS_DIR / "logistic_regression_validation_predictions.csv"
    ),
    "Random Forest": (
        PREDICTIONS_DIR / "random_forest_validation_predictions.csv"
    ),
}


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate saved validation predictions."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    data = pd.read_csv(path)

    required = {
        "y_true",
        "y_pred",
        "y_probability_malicious",
    }

    if not required.issubset(data.columns):
        raise ValueError(
            f"Missing required columns in {path}"
        )

    if data.empty:
        raise ValueError(f"Prediction file is empty: {path}")

    if data[list(required)].isna().any().any():
        raise ValueError(f"Missing values in {path}")

    if not data["y_true"].isin([0, 1]).all():
        raise ValueError(f"Invalid actual labels in {path}")

    if not data["y_pred"].isin([0, 1]).all():
        raise ValueError(f"Invalid predicted labels in {path}")

    probabilities = data["y_probability_malicious"]

    if (
        not np.isfinite(probabilities.to_numpy()).all()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError(f"Invalid probabilities in {path}")

    if data["y_true"].nunique() != 2:
        raise ValueError(
            f"ROC curve requires both classes in {path}"
        )

    return data


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        name: load_predictions(path)
        for name, path in MODELS.items()
    }

    # 1. Side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, (name, data) in zip(axes, datasets.items()):
        ConfusionMatrixDisplay.from_predictions(
            data["y_true"],
            data["y_pred"],
            labels=[0, 1],
            display_labels=["Benign (0)", "Malicious (1)"],
            cmap="Blues",
            values_format="d",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(name)

    fig.suptitle("QRShield-X — Validation Confusion Matrices")
    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 2. Combined ROC curve
    fig, axis = plt.subplots(figsize=(7, 6))

    for name, data in datasets.items():
        fpr, tpr, _ = roc_curve(
            data["y_true"],
            data["y_probability_malicious"],
            pos_label=1,
        )

        roc_auc = auc(fpr, tpr)

        axis.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    axis.set(
        title="QRShield-X — Validation ROC Curve",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate (Recall)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )

    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Plots saved to: {PLOTS_DIR}")
    print(" - confusion_matrices.png")
    print(" - roc_curve.png")


if __name__ == "__main__":
    main()

