"""Save and load trained QRShield-X baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class ModelPersistence:
    """Persist trained machine-learning models to local files."""

    @staticmethod
    def save(model: Any, path: str | Path) -> Path:
        """Save a trained model and return its destination path."""

        if not getattr(model, "is_trained", False):
            raise RuntimeError(
                "Cannot save a model that has not been trained."
            )

        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(model, destination)

        return destination

    @staticmethod
    def load(path: str | Path) -> Any:
        """Load a previously saved model from a trusted local file."""

        source = Path(path)

        if not source.is_file():
            raise FileNotFoundError(
                f"Model file does not exist: {source}"
            )

        model = joblib.load(source)

        if not getattr(model, "is_trained", False):
            raise ValueError(
                "Loaded object is not a trained QRShield-X model."
            )

        return model