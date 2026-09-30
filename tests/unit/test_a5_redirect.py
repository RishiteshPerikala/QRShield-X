"""Unit tests for A5 redirect analysis."""

from unittest.mock import MagicMock, patch

from qrshield_x.investigations.a5_redirect import A5RedirectAnalyzer


def test_rejects_empty_url() -> None:
    analyzer = A5RedirectAnalyzer()

    result = analyzer.analyze("")

    assert result.success is False
    assert result.redirect_chain == ()
    assert result.final_destination is None
    assert result.error == "URL must be a non-empty string."


def test_rejects_unsupported_scheme() -> None:
    analyzer = A5RedirectAnalyzer()

    result = analyzer.analyze("ftp://example.com/file")

    assert result.success is False
    assert result.final_destination is None
    assert result.error == "URL must use HTTP or HTTPS."


def test_rejects_url_without_hostname() -> None:
    analyzer = A5RedirectAnalyzer()

    result = analyzer.analyze("https:///path")

    assert result.success is False
    assert result.final_destination is None
    assert result.error == "URL must contain a hostname."


@patch("qrshield_x.investigations.a5_redirect.build_opener")
def test_direct_url_without_redirects(
    mock_build_opener: MagicMock,
) -> None:
    analyzer = A5RedirectAnalyzer()

    mock_response = MagicMock()
    mock_response.geturl.return_value = "https://example.com"

    mock_opener = MagicMock()
    mock_opener.open.return_value.__enter__.return_value = mock_response
    mock_build_opener.return_value = mock_opener

    result = analyzer.analyze("https://example.com")

    assert result.success is True
    assert result.initial_url == "https://example.com"
    assert result.redirect_chain == ()
    assert result.final_destination == "https://example.com"
    assert result.redirect_count == 0
    assert result.status_codes == ()


@patch("qrshield_x.investigations.a5_redirect.build_opener")
def test_redirect_chain_is_recorded(
    mock_build_opener: MagicMock,
) -> None:
    analyzer = A5RedirectAnalyzer()

    mock_response = MagicMock()
    mock_response.geturl.return_value = "https://example.com/final"

    mock_handler = MagicMock()
    mock_handler.redirect_chain = [
        "https://example.com/redirect1",
        "https://example.com/redirect2",
    ]
    mock_handler.status_codes = [302, 301]

    mock_opener = MagicMock()
    mock_opener.open.return_value.__enter__.return_value = mock_response
    mock_build_opener.return_value = mock_opener

    with patch(
        "qrshield_x.investigations.a5_redirect._BoundedRedirectHandler",
        return_value=mock_handler,
    ):
        result = analyzer.analyze("https://example.com/start")

    assert result.success is True
    assert result.redirect_chain == (
        "https://example.com/redirect1",
        "https://example.com/redirect2",
    )
    assert result.final_destination == "https://example.com/final"
    assert result.redirect_count == 2
    assert result.status_codes == (302, 301)


@patch("qrshield_x.investigations.a5_redirect.build_opener")
def test_redirect_failure_is_recorded(
    mock_build_opener: MagicMock,
) -> None:
    analyzer = A5RedirectAnalyzer()

    mock_opener = MagicMock()
    mock_opener.open.side_effect = TimeoutError("request timed out")
    mock_build_opener.return_value = mock_opener

    result = analyzer.analyze("https://example.com")

    assert result.success is False
    assert result.final_destination is None
    assert result.redirect_chain == ()
    assert result.redirect_count == 0
    assert result.error == (
        "Redirect analysis failed: request timed out"
    )


def test_invalid_configuration_is_rejected() -> None:
    try:
        A5RedirectAnalyzer(timeout=0)
        assert False
    except ValueError as exc:
        assert str(exc) == "Timeout must be greater than zero."

    try:
        A5RedirectAnalyzer(max_redirects=-1)
        assert False
    except ValueError as exc:
        assert str(exc) == "Maximum redirects cannot be negative."