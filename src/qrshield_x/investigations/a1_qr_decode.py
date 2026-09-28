"""A1 QR decoding analyzer for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class QRDecodeResult:
    """Result produced by the A1 QR decoder."""

    success: bool
    decoded_urls: tuple[str, ...]
    decoder_count: int
    error: str | None = None


class A1QRDecoder:
    """Decode one or more QR codes from an image."""

    def __init__(self) -> None:
        """Initialize the OpenCV QR detector."""
        self._detector = cv2.QRCodeDetector()

    def decode(self, image_path: str | Path) -> QRDecodeResult:
        """
        Decode QR codes from an image.

        Args:
            image_path: Path to the QR image.

        Returns:
            A QRDecodeResult containing decoded URLs or an error.
        """
        path = Path(image_path)

        if not path.is_file():
            return QRDecodeResult(
                success=False,
                decoded_urls=(),
                decoder_count=0,
                error=f"Image file does not exist: {path}",
            )

        image = cv2.imread(str(path))

        if image is None:
            return QRDecodeResult(
                success=False,
                decoded_urls=(),
                decoder_count=0,
                error=f"Unable to read image: {path}",
            )

        decoded_urls: list[str] = []

        try:
            success, decoded_info, _, _ = self._detector.detectAndDecodeMulti(
                image
            )

            if success:
                decoded_urls = [
                    value.strip()
                    for value in decoded_info
                    if value and value.strip()
                ]
        except cv2.error:
            # Fall back to single-code decoding when multi-code decoding
            # is unavailable or fails for the supplied image.
            try:
                value, _, _ = self._detector.detectAndDecode(image)

                if value and value.strip():
                    decoded_urls = [value.strip()]
            except cv2.error as exc:
                return QRDecodeResult(
                    success=False,
                    decoded_urls=(),
                    decoder_count=0,
                    error=f"QR decoding failed: {exc}",
                )

        return QRDecodeResult(
            success=bool(decoded_urls),
            decoded_urls=tuple(decoded_urls),
            decoder_count=len(decoded_urls),
            error=None if decoded_urls else "No QR code could be decoded.",
        )