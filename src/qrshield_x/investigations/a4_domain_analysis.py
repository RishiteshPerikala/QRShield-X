"""A4 domain analysis for QRShield-X."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class A4DomainAnalysisResult:
    """Structured result produced by the A4 domain analyzer."""

    success: bool
    hostname: str
    registered_domain: str | None
    tld: str | None

    hostname_length: int
    domain_length: int
    subdomain_count: int
    dot_count: int
    hyphen_count: int
    digit_count: int

    is_ip_address: bool
    is_punycode: bool

    suspicious_lexical_indicators: tuple[str, ...]

    error: str | None = None


class A4DomainAnalyzer:
    """Analyze structural and lexical characteristics of a domain."""

    _SUSPICIOUS_KEYWORDS = (
        "login",
        "signin",
        "verify",
        "verification",
        "account",
        "secure",
        "update",
        "confirm",
        "password",
        "wallet",
        "payment",
    )

    def analyze(self, hostname: str) -> A4DomainAnalysisResult:
        """
        Analyze a hostname and return structured domain characteristics.

        Args:
            hostname: Domain or hostname to analyze.

        Returns:
            A4DomainAnalysisResult containing measurable characteristics.
        """
        if not isinstance(hostname, str) or not hostname.strip():
            return self._failure(
                hostname=hostname,
                error="Hostname must be a non-empty string.",
            )

        normalized_hostname = hostname.strip().lower().rstrip(".")

        if not normalized_hostname:
            return self._failure(
                hostname=hostname,
                error="Hostname must contain a valid value.",
            )

        is_ip_address = self._is_ip_address(normalized_hostname)

        labels = [
            label
            for label in normalized_hostname.split(".")
            if label
        ]

        registered_domain = self._extract_registered_domain(
            normalized_hostname
        )

        tld = self._extract_tld(normalized_hostname)

        subdomain_count = self._count_subdomains(
            normalized_hostname,
            is_ip_address,
        )

        suspicious_indicators = self._find_suspicious_keywords(
            normalized_hostname
        )

        return A4DomainAnalysisResult(
            success=True,
            hostname=normalized_hostname,
            registered_domain=registered_domain,
            tld=tld,
            hostname_length=len(normalized_hostname),
            domain_length=len(registered_domain or normalized_hostname),
            subdomain_count=subdomain_count,
            dot_count=normalized_hostname.count("."),
            hyphen_count=normalized_hostname.count("-"),
            digit_count=sum(character.isdigit() for character in normalized_hostname),
            is_ip_address=is_ip_address,
            is_punycode=any(
                label.startswith("xn--")
                for label in labels
            ),
            suspicious_lexical_indicators=tuple(
                suspicious_indicators
            ),
            error=None,
        )

    @staticmethod
    def _is_ip_address(hostname: str) -> bool:
        """Return True when the hostname is an IPv4 or IPv6 address."""
        try:
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_registered_domain(hostname: str) -> str | None:
        """
        Extract a simple registered-domain approximation.

        This intentionally does not perform live public-suffix lookups.
        """
        if A4DomainAnalyzer._is_ip_address(hostname):
            return hostname

        labels = [
            label
            for label in hostname.split(".")
            if label
        ]

        if len(labels) < 2:
            return hostname if labels else None

        return ".".join(labels[-2:])

    @staticmethod
    def _extract_tld(hostname: str) -> str | None:
        """Extract the final hostname label as a basic TLD approximation."""
        if A4DomainAnalyzer._is_ip_address(hostname):
            return None

        labels = [
            label
            for label in hostname.split(".")
            if label
        ]

        return labels[-1] if labels else None

    @staticmethod
    def _count_subdomains(
        hostname: str,
        is_ip_address: bool,
    ) -> int:
        """Count labels preceding the registered-domain approximation."""
        if is_ip_address:
            return 0

        labels = [
            label
            for label in hostname.split(".")
            if label
        ]

        if len(labels) <= 2:
            return 0

        return len(labels) - 2

    @classmethod
    def _find_suspicious_keywords(
        cls,
        hostname: str,
    ) -> list[str]:
        """Find predefined lexical indicators in the hostname."""
        return [
            keyword
            for keyword in cls._SUSPICIOUS_KEYWORDS
            if keyword in hostname
        ]

    @staticmethod
    def _failure(
        *,
        hostname: str,
        error: str,
    ) -> A4DomainAnalysisResult:
        """Create a structured failed-analysis result."""
        return A4DomainAnalysisResult(
            success=False,
            hostname=hostname if isinstance(hostname, str) else "",
            registered_domain=None,
            tld=None,
            hostname_length=0,
            domain_length=0,
            subdomain_count=0,
            dot_count=0,
            hyphen_count=0,
            digit_count=0,
            is_ip_address=False,
            is_punycode=False,
            suspicious_lexical_indicators=(),
            error=error,
        )