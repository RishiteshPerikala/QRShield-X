
"""Logistic Regression baseline for QRShield-X."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


class LogisticRegressionBaseline:
    """Train and use a Logistic Regression QR classifier."""

    def __init__(
        self,
        *,
        random_seed: int = 42,
        max_iterations: int = 1000,
        c: float = 1.0,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")

        if c <= 0:
            raise ValueError("c must be positive.")

        self._model = LogisticRegression(
            random_state=random_seed,
            max_iter=max_iterations,
            C=c,
            solver="lbfgs",
        )
        self._is_trained = False

    def fit(
        self,
        features,
        labels,
    ) -> LogisticRegressionBaseline:
        """Train the model on the supplied training data."""

        self._model.fit(features, labels)
        self._is_trained = True
        return self

    def predict(self, features):
        """Predict labels for supplied feature vectors."""

        self._require_trained()
        return self._model.predict(features)

    def predict_proba(self, features):
        """Return class probabilities for supplied feature vectors."""

        self._require_trained()
        return self._model.predict_proba(features)

    def _require_trained(self) -> None:
        """Prevent inference before model training."""

        if not self._is_trained:
            raise RuntimeError(
                "Logistic Regression model has not been trained."
            )

    @property
    def classes_(self):
        """Return the class labels learned during training."""

        self._require_trained()
        return self._model.classes_

    @property
    def is_trained(self) -> bool:
        """Return whether the model has been trained."""

        return self._is_trained