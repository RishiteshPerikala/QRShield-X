from pathlib import Path

from PIL import Image
from qrcode import QRCode

from qrshield_x.investigations.a2_qr_structure import (
    A2QRStructureAnalyzer,
)


def create_qr_image(path: Path, data: str) -> None:
    """Create a QR image for testing."""
    qr = QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)

    image = qr.make_image()
    image.save(path)


def test_structure_analyzer_reads_qr_image(tmp_path: Path) -> None:
    image_path = tmp_path / "qr.png"

    create_qr_image(
        image_path,
        "https://example.com",
    )

    analyzer = A2QRStructureAnalyzer()
    result = analyzer.analyze(image_path)

    assert result.success is True
    assert result.detected is True
    assert result.image_width > 0
    assert result.image_height > 0
    assert result.image_area > 0
    assert result.grayscale_mean is not None
    assert result.grayscale_std is not None
    assert result.sharpness is not None


def test_structure_analyzer_handles_non_qr_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "normal.png"

    image = Image.new("RGB", (300, 200), "white")
    image.save(image_path)

    analyzer = A2QRStructureAnalyzer()
    result = analyzer.analyze(image_path)

    assert result.success is True
    assert result.detected is False
    assert result.image_width == 300
    assert result.image_height == 200
    assert result.image_area == 60000


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "missing.png"

    analyzer = A2QRStructureAnalyzer()
    result = analyzer.analyze(image_path)

    assert result.success is False
    assert result.detected is False
    assert result.error is not None
    assert "does not exist" in result.error