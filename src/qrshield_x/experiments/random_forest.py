from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


class RandomForestBaseline:
    """Train and use a Random Forest QR classifier."""

    def __init__(
        self,
        *,
        random_seed: int = 42,
        n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")

        if max_depth is not None and max_depth <= 0:
            raise ValueError(
                "max_depth must be positive when specified."
            )

        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_seed,
            max_depth=max_depth,
            n_jobs=-1,
        )
        self._is_trained = False

    def fit(
        self,
        features,
        labels,
    ) -> RandomForestBaseline:
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
                "Random Forest model has not been trained."
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
