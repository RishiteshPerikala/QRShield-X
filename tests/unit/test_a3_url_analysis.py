"""Unit tests for the A3 URL analyzer."""

from qrshield_x.investigations.a3_url_analysis import A3URLAnalyzer


def test_analyze_normal_url() -> None:
    """A normal URL should produce structured lexical features."""
    analyzer = A3URLAnalyzer()

    result = analyzer.analyze(
        "https://example.com/products/item?id=123"
    )

    assert result.success is True
    assert result.scheme == "https"
    assert result.hostname == "example.com"
    assert result.url_length > 0
    assert result.path_length == len("/products/item")
    assert result.query_length == len("id=123")
    assert result.has_query_parameters is True
    assert result.has_ip_host is False


def test_detect_ip_host_and_obfuscation() -> None:
    """An IP-based URL should expose measurable obfuscation indicators."""
    analyzer = A3URLAnalyzer()

    result = analyzer.analyze(
        "http://192.168.1.10/login%2Fverify"
    )

    assert result.success is True
    assert result.has_ip_host is True
    assert result.has_percent_encoding is True
    assert "IP_HOST" in result.obfuscation_indicators
    assert "PERCENT_ENCODING" in result.obfuscation_indicators


def test_detect_lexical_indicators() -> None:
    """Security-sensitive lexical keywords should be reported."""
    analyzer = A3URLAnalyzer()

    result = analyzer.analyze(
        "https://secure-login.example.com/account/verify"
    )

    assert result.success is True
    assert "login" in result.suspicious_lexical_indicators
    assert "secure" in result.suspicious_lexical_indicators
    assert "account" in result.suspicious_lexical_indicators
    assert "verify" in result.suspicious_lexical_indicators


def test_detect_url_structure_features() -> None:
    """Structural URL characteristics should be measured."""
    analyzer = A3URLAnalyzer()

    result = analyzer.analyze(
        "https://a.b.example.com/path//test#section"
    )

    assert result.success is True
    assert result.subdomain_count == 2
    assert result.has_double_slash_in_path is True
    assert result.fragment_length == len("section")


def test_empty_url_fails_cleanly() -> None:
    """An empty URL should return a structured failure."""
    analyzer = A3URLAnalyzer()

    result = analyzer.analyze("")

    assert result.success is False
    assert result.error is not None