"""A6 domain-identity analysis for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class A6IdentityResult:
    """Result of domain-identity matching."""

    success: bool
    observed_domain: str
    expected_domain: str | None
    identity_match: bool | None
    normalized_observed_domain: str | None
    normalized_expected_domain: str | None
    evidence_type: str | None
    error: str | None = None


class A6IdentityAnalyzer:
    """Compare an observed domain against an expected identity domain."""

    def analyze(
        self,
        observed_domain: str,
        expected_domain: str | None,
    ) -> A6IdentityResult:
        """Determine whether the observed domain matches the expected identity."""
        if not isinstance(observed_domain, str) or not observed_domain.strip():
            return self._failure(
                observed_domain=observed_domain,
                expected_domain=expected_domain,
                error="Observed domain must be a non-empty string.",
            )

        normalized_observed = self._normalize_domain(observed_domain)

        if normalized_observed is None:
            return self._failure(
                observed_domain=observed_domain,
                expected_domain=expected_domain,
                error="Observed domain is invalid.",
            )

        if expected_domain is None or not expected_domain.strip():
            return A6IdentityResult(
                success=True,
                observed_domain=observed_domain,
                expected_domain=expected_domain,
                identity_match=None,
                normalized_observed_domain=normalized_observed,
                normalized_expected_domain=None,
                evidence_type=None,
                error=None,
            )

        normalized_expected = self._normalize_domain(expected_domain)

        if normalized_expected is None:
            return self._failure(
                observed_domain=observed_domain,
                expected_domain=expected_domain,
                error="Expected domain is invalid.",
            )

        identity_match = (
            normalized_observed == normalized_expected
        )

        evidence_type = (
            "DOMAIN_IDENTITY_MATCH"
            if identity_match
            else "DOMAIN_IDENTITY_MISMATCH"
        )

        return A6IdentityResult(
            success=True,
            observed_domain=observed_domain,
            expected_domain=expected_domain,
            identity_match=identity_match,
            normalized_observed_domain=normalized_observed,
            normalized_expected_domain=normalized_expected,
            evidence_type=evidence_type,
            error=None,
        )

    @staticmethod
    def _normalize_domain(domain: str) -> str | None:
        """Normalize a hostname or URL to a lowercase hostname."""
        value = domain.strip().lower()

        if not value:
            return None

        parsed = urlparse(
            value if "://" in value else f"https://{value}"
        )

        hostname = parsed.hostname

        if not hostname:
            return None

        return hostname.rstrip(".")

    @staticmethod
    def _failure(
        *,
        observed_domain: str,
        expected_domain: str | None,
        error: str,
    ) -> A6IdentityResult:
        return A6IdentityResult(
            success=False,
            observed_domain=(
                observed_domain
                if isinstance(observed_domain, str)
                else ""
            ),
            expected_domain=expected_domain,
            identity_match=None,
            normalized_observed_domain=None,
            normalized_expected_domain=None,
            evidence_type=None,
            error=error,
        )