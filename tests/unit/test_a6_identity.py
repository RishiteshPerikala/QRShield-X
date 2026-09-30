"""Unit tests for A6 identity analysis."""

from qrshield_x.investigations.a6_identity import A6IdentityAnalyzer


def test_matching_domains_produce_identity_match() -> None:
    analyzer = A6IdentityAnalyzer()

    result = analyzer.analyze(
        "example.com",
        "example.com",
    )

    assert result.success is True
    assert result.identity_match is True
    assert result.evidence_type == "DOMAIN_IDENTITY_MATCH"


def test_different_domains_produce_identity_mismatch() -> None:
    analyzer = A6IdentityAnalyzer()

    result = analyzer.analyze(
        "evil-example.com",
        "example.com",
    )

    assert result.success is True
    assert result.identity_match is False
    assert result.evidence_type == "DOMAIN_IDENTITY_MISMATCH"


def test_domain_comparison_is_case_insensitive() -> None:
    analyzer = A6IdentityAnalyzer()

    result = analyzer.analyze(
        "Example.COM",
        "example.com",
    )

    assert result.success is True
    assert result.identity_match is True
    assert result.normalized_observed_domain == "example.com"
    assert result.normalized_expected_domain == "example.com"


def test_url_is_normalized_to_hostname() -> None:
    analyzer = A6IdentityAnalyzer()

    result = analyzer.analyze(
        "https://example.com/login",
        "example.com",
    )

    assert result.success is True
    assert result.identity_match is True
    assert result.normalized_observed_domain == "example.com"


def test_trailing_dot_is_normalized() -> None:
    analyzer = A6IdentityAnalyzer()

    result = analyzer.analyze(
        "example.com.",
        "example.com",
    )

    assert result.success is True
    assert result.identity_match is True


def test_missing_expected_identity_is_unavailable() -> None:
    analyzer = A6IdentityAnalyzer()

    result = analyzer.analyze(
        "example.com",
        None,
    )

    assert result.success is True
    assert result.identity_match is None
    assert result.evidence_type is None
    assert result.normalized_observed_domain == "example.com"


def test_empty_observed_domain_fails_cleanly() -> None:
    analyzer = A6IdentityAnalyzer()

    result = analyzer.analyze(
        "",
        "example.com",
    )

    assert result.success is False
    assert result.identity_match is None
    assert result.error == (
        "Observed domain must be a non-empty string."
    )


def test_invalid_expected_domain_fails_cleanly() -> None:
    analyzer = A6IdentityAnalyzer()

    result = analyzer.analyze(
        "example.com",
        "https://",
    )

    assert result.success is False
    assert result.identity_match is None
    assert result.error == "Expected domain is invalid."