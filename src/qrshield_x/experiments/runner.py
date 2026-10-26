"""Configuration-driven baseline experiment runner for QRShield-X."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qrshield_x.datasets.loader import QRDatasetLoader
from qrshield_x.datasets.splitter import QRDatasetSplitter
from qrshield_x.experiments.baseline_data import BaselineDataPreparer
from qrshield_x.experiments.evaluation import ExperimentEvaluator
from qrshield_x.experiments.logistic_regression import (
    LogisticRegressionBaseline,
)
from qrshield_x.experiments.model_persistence import ModelPersistence
from qrshield_x.experiments.random_forest import RandomForestBaseline


LOGGER = logging.getLogger(__name__)


class BaselineExperimentRunner:
    """Load configuration, train baseline models, and evaluate them."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.project_root = self._find_project_root(self.config_path)
        self.config = self._load_config(self.config_path)
        self.evaluator = ExperimentEvaluator()

    @staticmethod
    def _find_project_root(config_path: Path) -> Path:
        """Find the project root using pyproject.toml."""

        for candidate in config_path.parents:
            if (candidate / "pyproject.toml").is_file():
                return candidate

        raise FileNotFoundError(
            "Could not find project root containing pyproject.toml "
            f"for configuration: {config_path}"
        )

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        """Load and minimally validate the YAML configuration."""

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Experiment configuration not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "Experiment configuration must be a YAML mapping."
            )

        required_sections = ("dataset", "split", "models", "output")

        for section in required_sections:
            if not isinstance(config.get(section), dict):
                raise ValueError(
                    f"Missing or invalid configuration section: {section}"
                )

        dataset = config["dataset"]

        for key in ("data_path", "labels_path"):
            if (
                not isinstance(dataset.get(key), str)
                or not dataset[key].strip()
            ):
                raise ValueError(
                    f"dataset.{key} must be a non-empty path string."
                )

        split = config["split"]

        for key in ("train_ratio", "validation_ratio", "test_ratio"):
            if key not in split:
                raise ValueError(
                    f"Missing split.{key} in configuration."
                )

        output = config["output"]

        for key in ("models_dir", "results_dir"):
            value = output.get(key)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"output.{key} must be a non-empty path string."
                )

        return config

    def _resolve_path(self, configured_path: str) -> Path:
        """Resolve relative paths against the project root."""

        path = Path(configured_path)

        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()

    @staticmethod
    def _save_predictions(
        output_path: Path,
        y_true: Any,
        y_pred: Any,
        y_probability_malicious: Any,
    ) -> None:
        """Save labels, predictions, and malicious probabilities to CSV."""

        actual = np.asarray(y_true)
        predicted = np.asarray(y_pred)
        probabilities = np.asarray(
            y_probability_malicious,
            dtype=float,
        )

        if not (
            actual.ndim == predicted.ndim == probabilities.ndim == 1
        ):
            raise ValueError(
                "Prediction arrays must be one-dimensional."
            )

        if not (
            actual.size == predicted.size == probabilities.size
        ):
            raise ValueError(
                "Prediction arrays must have equal lengths."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = output_path.with_suffix(".csv.tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "y_true",
                    "y_pred",
                    "y_probability_malicious",
                ]
            )

            writer.writerows(
                zip(
                    actual.tolist(),
                    predicted.tolist(),
                    probabilities.tolist(),
                )
            )

        temporary_path.replace(output_path)

    def run(self) -> dict[str, Any]:
        """Train enabled models and evaluate validation and test data."""

        dataset_config = self.config["dataset"]
        split_config = self.config["split"]
        output_config = self.config["output"]

        data_path = self._resolve_path(
            dataset_config["data_path"]
        )

        labels_path = self._resolve_path(
            dataset_config["labels_path"]
        )

        LOGGER.info("Loading QR dataset...")

        samples = QRDatasetLoader(
            data_path=data_path,
            labels_path=labels_path,
        ).load()

        LOGGER.info(
            "Loaded %d samples.",
            len(samples),
        )

        split = QRDatasetSplitter().split(
            samples,
            train_ratio=float(
                split_config["train_ratio"]
            ),
            validation_ratio=float(
                split_config["validation_ratio"]
            ),
            test_ratio=float(
                split_config["test_ratio"]
            ),
            random_seed=int(
                split_config.get("random_seed", 42)
            ),
        )

        prepared = BaselineDataPreparer().prepare(split)

        LOGGER.info(
            "Split sizes: train=%d, validation=%d, test=%d",
            len(prepared.y_train),
            len(prepared.y_validation),
            len(prepared.y_test),
        )

        if len(np.unique(prepared.y_train)) < 2:
            raise ValueError(
                "Training data must contain at least two distinct classes."
            )

        models_dir = self._resolve_path(
            output_config["models_dir"]
        )

        results_dir = self._resolve_path(
            output_config["results_dir"]
        )

        models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results: dict[str, Any] = {
            "project": "QRShield-X",
            "experiment": "baseline_comparison",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "config_path": str(self.config_path),
            "dataset": {
                "sample_count": len(samples),
                "feature_count": int(
                    prepared.x_train.shape[1]
                ),
                "class_counts": {
                    str(int(label)): int(count)
                    for label, count in zip(
                        *np.unique(
                            np.asarray(
                                [
                                    sample.label
                                    for sample in samples
                                ]
                            ),
                            return_counts=True,
                        )
                    )
                },
            },
            "split": {
                "random_seed": int(
                    split_config.get("random_seed", 42)
                ),
                "train_count": int(
                    len(prepared.y_train)
                ),
                "validation_count": int(
                    len(prepared.y_validation)
                ),
                "test_count": int(
                    len(prepared.y_test)
                ),
                "test_evaluated": True,
            },
            "models": {},
        }

        model_configs = self.config["models"]

        model_specs = {
            "logistic_regression": (
                LogisticRegressionBaseline,
                {
                    "random_seed": int(
                        split_config.get(
                            "random_seed",
                            42,
                        )
                    ),
                    "max_iterations": int(
                        model_configs.get(
                            "logistic_regression",
                            {},
                        ).get(
                            "max_iterations",
                            1000,
                        )
                    ),
                    "c": float(
                        model_configs.get(
                            "logistic_regression",
                            {},
                        ).get(
                            "c",
                            1.0,
                        )
                    ),
                },
            ),
            "random_forest": (
                RandomForestBaseline,
                {
                    "random_seed": int(
                        split_config.get(
                            "random_seed",
                            42,
                        )
                    ),
                    "n_estimators": int(
                        model_configs.get(
                            "random_forest",
                            {},
                        ).get(
                            "n_estimators",
                            100,
                        )
                    ),
                    "max_depth": model_configs.get(
                        "random_forest",
                        {},
                    ).get(
                        "max_depth"
                    ),
                },
            ),
        }

        enabled_count = 0

        for (
            model_name,
            (model_class, model_kwargs),
        ) in model_specs.items():

            model_config = model_configs.get(
                model_name,
                {},
            )

            if not isinstance(
                model_config,
                dict,
            ):
                raise ValueError(
                    f"models.{model_name} must be a configuration mapping."
                )

            if not model_config.get(
                "enabled",
                False,
            ):
                LOGGER.info(
                    "Skipping disabled model: %s",
                    model_name,
                )
                continue

            enabled_count += 1

            LOGGER.info(
                "Training %s...",
                model_name,
            )

            model = model_class(
                **model_kwargs
            )

            # Train ONLY on the training set.
            model.fit(
                prepared.x_train,
                prepared.y_train,
            )

            # ---------------------------------------------------------
            # Validation evaluation
            # ---------------------------------------------------------

            validation_predictions = model.predict(
                prepared.x_validation
            )

            validation_probabilities = model.predict_proba(
                prepared.x_validation
            )

            classes = list(model.classes_)

            if 1 not in classes:
                raise ValueError(
                    f"{model_name} does not expose probabilities "
                    "for malicious class 1."
                )

            malicious_class_index = classes.index(1)

            validation_malicious_probabilities = (
                validation_probabilities[
                    :,
                    malicious_class_index,
                ]
            )

            validation_prediction_path = (
                results_dir
                / "predictions"
                / f"{model_name}_validation_predictions.csv"
            )

            self._save_predictions(
                validation_prediction_path,
                prepared.y_validation,
                validation_predictions,
                validation_malicious_probabilities,
            )

            validation_metrics = self.evaluator.evaluate(
                prepared.y_validation,
                validation_predictions,
                y_probability_malicious=(
                    validation_malicious_probabilities
                ),
            )

            # ---------------------------------------------------------
            # Test evaluation
            # ---------------------------------------------------------

            test_predictions = model.predict(
                prepared.x_test
            )

            test_probabilities = model.predict_proba(
                prepared.x_test
            )

            test_malicious_probabilities = (
                test_probabilities[
                    :,
                    malicious_class_index,
                ]
            )

            test_prediction_path = (
                results_dir
                / "predictions"
                / f"{model_name}_test_predictions.csv"
            )

            self._save_predictions(
                test_prediction_path,
                prepared.y_test,
                test_predictions,
                test_malicious_probabilities,
            )

            test_metrics = self.evaluator.evaluate(
                prepared.y_test,
                test_predictions,
                y_probability_malicious=(
                    test_malicious_probabilities
                ),
            )

            # ---------------------------------------------------------
            # Persist model
            # ---------------------------------------------------------

            model_path = (
                models_dir
                / f"{model_name}.joblib"
            )

            ModelPersistence.save(
                model,
                model_path,
            )

            results["models"][model_name] = {
                "parameters": model_kwargs,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "model_path": str(model_path),
                "validation_predictions_path": str(
                    validation_prediction_path
                ),
                "test_predictions_path": str(
                    test_prediction_path
                ),
            }

            LOGGER.info(
                "%s validation metrics: %s",
                model_name,
                validation_metrics,
            )

            LOGGER.info(
                "%s test metrics: %s",
                model_name,
                test_metrics,
            )

        if enabled_count == 0:
            raise ValueError(
                "No baseline models are enabled in the configuration."
            )

        results_path = (
            results_dir
            / "baseline_results.json"
        )

        temporary_path = results_path.with_suffix(
            ".json.tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=2,
                allow_nan=False,
            )

            file.write("\n")

        temporary_path.replace(
            results_path
        )

        LOGGER.info(
            "Experiment results saved to %s",
            results_path,
        )

        return results


def main() -> None:
    """Run the experiment runner from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Run QRShield-X baseline experiments."
        )
    )

    parser.add_argument(
        "--config",
        default="config/experiment_config.yaml",
        help=(
            "Path to the experiment YAML configuration."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        help="Logging verbosity.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),
    )

    runner = BaselineExperimentRunner(
        args.config
    )

    results = runner.run()

    results_path = (
        runner._resolve_path(
            runner.config["output"]["results_dir"]
        )
        / "baseline_results.json"
    )

    print(
        "\nBaseline experiment completed."
    )

    print(
        f"Results: {results_path}"
    )

    for (
        name,
        details,
    ) in results["models"].items():

        print(f"\n{name}")

        print("\n  Validation metrics:")

        for (
            metric,
            value,
        ) in details[
            "validation_metrics"
        ].items():

            if isinstance(
                value,
                dict,
            ):
                print(f"    {metric}:")

                for (
                    key,
                    item,
                ) in value.items():
                    print(
                        f"      {key}: {item}"
                    )

            elif value is None:
                print(
                    f"    {metric}: unavailable"
                )

            else:
                print(
                    f"    {metric}: {value:.4f}"
                )

        print("\n  Test metrics:")

        for (
            metric,
            value,
        ) in details[
            "test_metrics"
        ].items():

            if isinstance(
                value,
                dict,
            ):
                print(f"    {metric}:")

                for (
                    key,
                    item,
                ) in value.items():
                    print(
                        f"      {key}: {item}"
                    )

            elif value is None:
                print(
                    f"    {metric}: unavailable"
                )

            else:
                print(
                    f"    {metric}: {value:.4f}"
                )


if __name__ == "__main__":
    main()