"""A3 URL lexical and obfuscation analysis for QRShield-X."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class A3URLAnalysisResult:
    """Structured result produced by the A3 URL analyzer."""

    success: bool
    url: str
    scheme: str
    hostname: str | None

    url_length: int
    hostname_length: int
    subdomain_count: int
    path_length: int
    query_length: int
    fragment_length: int

    has_ip_host: bool
    has_percent_encoding: bool
    has_at_symbol: bool
    has_double_slash_in_path: bool
    has_query_parameters: bool

    suspicious_lexical_indicators: tuple[str, ...]
    obfuscation_indicators: tuple[str, ...]

    error: str | None = None


class A3URLAnalyzer:
    """Analyze lexical and basic obfuscation characteristics of a URL."""

    _SUSPICIOUS_KEYWORDS = (
        "login",
        "signin",
        "verify",
        "verification",
        "account",
        "update",
        "secure",
        "confirm",
        "password",
        "wallet",
        "payment",
    )

    def analyze(self, url: str) -> A3URLAnalysisResult:
        """
        Analyze a URL and return structured lexical features.

        Args:
            url: URL to analyze.

        Returns:
            A3URLAnalysisResult containing measurable URL characteristics.
        """
        if not isinstance(url, str) or not url.strip():
            return self._failure(
                url=url,
                error="URL must be a non-empty string.",
            )

        normalized_url = url.strip()

        try:
            parsed = urlparse(normalized_url)
        except ValueError as exc:
            return self._failure(
                url=normalized_url,
                error=f"URL parsing failed: {exc}",
            )

        hostname = parsed.hostname

        if hostname is not None:
            hostname = hostname.lower()

        scheme = parsed.scheme.lower()

        path = parsed.path
        query = parsed.query
        fragment = parsed.fragment

        has_ip_host = self._is_ip_address(hostname)

        subdomain_count = self._count_subdomains(hostname)

        has_percent_encoding = bool(
            re.search(r"%[0-9A-Fa-f]{2}", normalized_url)
        )

        has_at_symbol = "@" in normalized_url

        has_double_slash_in_path = "//" in path

        query_parameters = parse_qs(query, keep_blank_values=True)

        suspicious_indicators = self._find_suspicious_keywords(
            normalized_url
        )

        obfuscation_indicators = self._find_obfuscation_indicators(
            normalized_url=normalized_url,
            hostname=hostname,
            has_ip_host=has_ip_host,
            has_percent_encoding=has_percent_encoding,
            has_at_symbol=has_at_symbol,
            has_double_slash_in_path=has_double_slash_in_path,
        )

        return A3URLAnalysisResult(
            success=True,
            url=normalized_url,
            scheme=scheme,
            hostname=hostname,
            url_length=len(normalized_url),
            hostname_length=len(hostname or ""),
            subdomain_count=subdomain_count,
            path_length=len(path),
            query_length=len(query),
            fragment_length=len(fragment),
            has_ip_host=has_ip_host,
            has_percent_encoding=has_percent_encoding,
            has_at_symbol=has_at_symbol,
            has_double_slash_in_path=has_double_slash_in_path,
            has_query_parameters=bool(query_parameters),
            suspicious_lexical_indicators=tuple(suspicious_indicators),
            obfuscation_indicators=tuple(obfuscation_indicators),
            error=None,
        )

    @staticmethod
    def _is_ip_address(hostname: str | None) -> bool:
        """Return True when the hostname is an IPv4 or IPv6 address."""
        if not hostname:
            return False

        try:
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            return False

    @staticmethod
    def _count_subdomains(hostname: str | None) -> int:
        """Count hostname labels before the registered-domain portion."""
        if not hostname:
            return 0

        labels = [label for label in hostname.split(".") if label]

        if len(labels) <= 2:
            return 0

        return len(labels) - 2

    @classmethod
    def _find_suspicious_keywords(cls, url: str) -> list[str]:
        """Find predefined lexical indicators in the URL."""
        lowered_url = url.lower()

        return [
            keyword
            for keyword in cls._SUSPICIOUS_KEYWORDS
            if keyword in lowered_url
        ]

    @staticmethod
    def _find_obfuscation_indicators(
        *,
        normalized_url: str,
        hostname: str | None,
        has_ip_host: bool,
        has_percent_encoding: bool,
        has_at_symbol: bool,
        has_double_slash_in_path: bool,
    ) -> list[str]:
        """Find measurable indicators commonly associated with obfuscation."""
        indicators: list[str] = []

        if has_ip_host:
            indicators.append("IP_HOST")

        if has_percent_encoding:
            indicators.append("PERCENT_ENCODING")

        if has_at_symbol:
            indicators.append("AT_SYMBOL")

        if has_double_slash_in_path:
            indicators.append("DOUBLE_SLASH_IN_PATH")

        if hostname and hostname.count("-") >= 3:
            indicators.append("MULTIPLE_HYPHENS_IN_HOST")

        if len(normalized_url) >= 150:
            indicators.append("LONG_URL")

        return indicators

    @staticmethod
    def _failure(*, url: str, error: str) -> A3URLAnalysisResult:
        """Create a failed analysis result."""
        return A3URLAnalysisResult(
            success=False,
            url=url,
            scheme="",
            hostname=None,
            url_length=len(url) if isinstance(url, str) else 0,
            hostname_length=0,
            subdomain_count=0,
            path_length=0,
            query_length=0,
            fragment_length=0,
            has_ip_host=False,
            has_percent_encoding=False,
            has_at_symbol=False,
            has_double_slash_in_path=False,
            has_query_parameters=False,
            suspicious_lexical_indicators=(),
            obfuscation_indicators=(),
            error=error,
        )