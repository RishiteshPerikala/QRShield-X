"""A2 QR structure and image-quality analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class QRStructureResult:
    """Result produced by the A2 QR structure analyzer."""

    success: bool
    detected: bool
    image_width: int
    image_height: int
    image_area: int
    grayscale_mean: float | None
    grayscale_std: float | None
    sharpness: float | None
    error: str | None = None


class A2QRStructureAnalyzer:
    """Analyze structural and basic image-quality properties of a QR image."""

    def __init__(self) -> None:
        """Initialize the OpenCV QR detector."""
        self._detector = cv2.QRCodeDetector()

    def analyze(self, image_path: str | Path) -> QRStructureResult:
        """
        Analyze a QR image.

        Args:
            image_path: Path to the QR image.

        Returns:
            Measurable structural and image-quality information.
        """
        path = Path(image_path)

        if not path.is_file():
            return QRStructureResult(
                success=False,
                detected=False,
                image_width=0,
                image_height=0,
                image_area=0,
                grayscale_mean=None,
                grayscale_std=None,
                sharpness=None,
                error=f"Image file does not exist: {path}",
            )

        image = cv2.imread(str(path))

        if image is None:
            return QRStructureResult(
                success=False,
                detected=False,
                image_width=0,
                image_height=0,
                image_area=0,
                grayscale_mean=None,
                grayscale_std=None,
                sharpness=None,
                error=f"Unable to read image: {path}",
            )

        height, width = image.shape[:2]

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mean = float(grayscale.mean())
        standard_deviation = float(grayscale.std())

        # Variance of the Laplacian is a commonly used image-sharpness
        # measurement. Higher values generally indicate stronger edges.
        sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())

        detected, _, _ = self._detector.detectAndDecode(image)

        return QRStructureResult(
            success=True,
            detected=bool(detected),
            image_width=width,
            image_height=height,
            image_area=width * height,
            grayscale_mean=mean,
            grayscale_std=standard_deviation,
            sharpness=sharpness,
            error=None,
        )