"""Dataset validation utilities for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.datasets.sample import QRDatasetSample


@dataclass(frozen=True)
class DatasetValidationResult:
    """Result of validating a collection of QR dataset samples."""

    valid: bool
    sample_count: int
    benign_count: int
    phishing_count: int
    errors: tuple[str, ...]


class QRDatasetValidator:
    """Validate structural and label-level dataset properties."""

    def validate(
        self,
        samples: list[QRDatasetSample],
    ) -> DatasetValidationResult:
        """Validate dataset samples and return a validation result."""

        errors: list[str] = []

        if not isinstance(samples, list):
            errors.append("Dataset samples must be provided as a list.")
            return DatasetValidationResult(
                valid=False,
                sample_count=0,
                benign_count=0,
                phishing_count=0,
                errors=tuple(errors),
            )

        sample_count = len(samples)
        benign_count = 0
        phishing_count = 0

        for index, sample in enumerate(samples):
            if not isinstance(sample, QRDatasetSample):
                errors.append(
                    f"Sample at index {index} is not a QRDatasetSample."
                )
                continue

            if sample.image.shape != (69, 69):
                errors.append(
                    f"Sample at index {index} has invalid image shape."
                )

            if sample.label == 0:
                benign_count += 1
            elif sample.label == 1:
                phishing_count += 1
            else:
                errors.append(
                    f"Sample at index {index} has an invalid label."
                )

        return DatasetValidationResult(
            valid=not errors,
            sample_count=sample_count,
            benign_count=benign_count,
            phishing_count=phishing_count,
            errors=tuple(errors),
        )