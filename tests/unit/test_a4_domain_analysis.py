"""Unit tests for the A4 domain analyzer."""

from qrshield_x.investigations.a4_domain_analysis import A4DomainAnalyzer


def test_analyze_normal_domain() -> None:
    """A normal domain should produce structured characteristics."""
    analyzer = A4DomainAnalyzer()

    result = analyzer.analyze("www.example.com")

    assert result.success is True
    assert result.hostname == "www.example.com"
    assert result.registered_domain == "example.com"
    assert result.tld == "com"
    assert result.subdomain_count == 1
    assert result.is_ip_address is False
    assert result.is_punycode is False


def test_detect_ip_address() -> None:
    """An IP hostname should be identified separately."""
    analyzer = A4DomainAnalyzer()

    result = analyzer.analyze("192.168.1.10")

    assert result.success is True
    assert result.is_ip_address is True
    assert result.registered_domain == "192.168.1.10"
    assert result.tld is None
    assert result.subdomain_count == 0


def test_detect_punycode() -> None:
    """Punycode hostnames should be explicitly identified."""
    analyzer = A4DomainAnalyzer()

    result = analyzer.analyze("xn--example-9za.com")

    assert result.success is True
    assert result.is_punycode is True
    assert result.tld == "com"


def test_detect_lexical_indicators() -> None:
    """Security-sensitive lexical indicators should be reported."""
    analyzer = A4DomainAnalyzer()

    result = analyzer.analyze(
        "secure-login-account.example.com"
    )

    assert result.success is True
    assert "secure" in result.suspicious_lexical_indicators
    assert "login" in result.suspicious_lexical_indicators
    assert "account" in result.suspicious_lexical_indicators


def test_domain_characteristics_are_measured() -> None:
    """Basic domain character counts should be calculated."""
    analyzer = A4DomainAnalyzer()

    result = analyzer.analyze("shop-123.example.com")

    assert result.success is True
    assert result.hyphen_count == 1
    assert result.digit_count == 3
    assert result.dot_count == 2
    assert result.hostname_length == len("shop-123.example.com")


def test_empty_hostname_fails_cleanly() -> None:
    """An empty hostname should return a structured failure."""
    analyzer = A4DomainAnalyzer()

    result = analyzer.analyze("")

    assert result.success is False
    assert result.error is not None