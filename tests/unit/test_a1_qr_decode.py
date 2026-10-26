from pathlib import Path

from PIL import Image
from qrcode import QRCode

from qrshield_x.investigations.a1_qr_decode import A1QRDecoder

from unittest.mock import patch


def create_qr_image(path: Path, data: str) -> None:
    """Create a QR image for testing."""
    qr = QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)

    image = qr.make_image()
    image.save(path)


def test_decoder_decodes_qr_image(tmp_path: Path) -> None:
    image_path = tmp_path / "test_qr.png"
    expected_url = "https://example.com"

    create_qr_image(image_path, expected_url)

    decoder = A1QRDecoder()
    result = decoder.decode(image_path)

    assert result.success is True
    assert expected_url in result.decoded_urls
    assert result.decoder_count == 1
    assert result.error is None


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "missing.png"

    decoder = A1QRDecoder()
    result = decoder.decode(image_path)

    assert result.success is False
    assert result.decoded_urls == ()
    assert result.decoder_count == 0
    assert "does not exist" in result.error


def test_non_qr_image_returns_decode_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "not_qr.png"

    image = Image.new("RGB", (300, 300), "white")
    image.save(image_path)

    decoder = A1QRDecoder()
    result = decoder.decode(image_path)

    assert result.success is False
    assert result.decoded_urls == ()
    assert result.decoder_count == 0
    assert result.error == "No QR code could be decoded."

def test_unreadable_image_returns_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "unreadable.png"
    image_path.write_text("not a valid image", encoding="utf-8")

    decoder = A1QRDecoder()

    with patch(
        "qrshield_x.investigations.a1_qr_decode.cv2.imread",
        return_value=None,
    ):
        result = decoder.decode(image_path)

    assert result.success is False
    assert result.decoded_urls == ()
    assert result.decoder_count == 0
    assert result.error == f"Unable to read image: {image_path}"