"""A5 redirect-chain analysis for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)
from urllib.error import HTTPError, URLError


@dataclass(frozen=True)
class A5RedirectResult:
    """Result of bounded URL redirect analysis."""

    success: bool
    initial_url: str
    redirect_chain: tuple[str, ...]
    final_destination: str | None
    redirect_count: int
    status_codes: tuple[int, ...]
    error: str | None = None


class _BoundedRedirectHandler(HTTPRedirectHandler):
    """Record redirects while enforcing a redirect limit."""

    def __init__(self, max_redirects: int) -> None:
        super().__init__()
        self.max_redirections = max_redirects
        self.redirect_chain: list[str] = []
        self.status_codes: list[int] = []

    def redirect_request(
        self,
        req: Request,
        fp,
        code: int,
        msg: str,
        headers,
        newurl: str,
    ) -> Request | None:
        """Record and validate a redirect before following it."""
        parsed = urlparse(newurl)

        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError(
                "Redirect target must use HTTP or HTTPS."
            )

        self.redirect_chain.append(newurl)
        self.status_codes.append(code)

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


class A5RedirectAnalyzer:
    """Analyze a URL's redirect chain with bounded network access."""

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        max_redirects: int = 10,
    ) -> None:
        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

        if max_redirects < 0:
            raise ValueError("Maximum redirects cannot be negative.")

        self._timeout = timeout
        self._max_redirects = max_redirects

    def analyze(self, url: str) -> A5RedirectResult:
        """Follow a bounded HTTP/HTTPS redirect chain."""
        if not isinstance(url, str) or not url.strip():
            return self._failure(
                url=url,
                error="URL must be a non-empty string.",
            )

        normalized_url = url.strip()
        parsed = urlparse(normalized_url)

        if parsed.scheme.lower() not in {"http", "https"}:
            return self._failure(
                url=normalized_url,
                error="URL must use HTTP or HTTPS.",
            )

        if not parsed.hostname:
            return self._failure(
                url=normalized_url,
                error="URL must contain a hostname.",
            )

        handler = _BoundedRedirectHandler(
            max_redirects=self._max_redirects
        )
        opener = build_opener(handler)

        request = Request(
            normalized_url,
            method="GET",
            headers={
                "User-Agent": "QRShield-X/0.1",
            },
        )

        try:
            with opener.open(
                request,
                timeout=self._timeout,
            ) as response:
                final_destination = response.geturl()

            return A5RedirectResult(
                success=True,
                initial_url=normalized_url,
                redirect_chain=tuple(handler.redirect_chain),
                final_destination=final_destination,
                redirect_count=len(handler.redirect_chain),
                status_codes=tuple(handler.status_codes),
                error=None,
            )

        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            return A5RedirectResult(
                success=False,
                initial_url=normalized_url,
                redirect_chain=tuple(handler.redirect_chain),
                final_destination=None,
                redirect_count=len(handler.redirect_chain),
                status_codes=tuple(handler.status_codes),
                error=f"Redirect analysis failed: {exc}",
            )

    @staticmethod
    def _failure(
        *,
        url: str,
        error: str,
    ) -> A5RedirectResult:
        return A5RedirectResult(
            success=False,
            initial_url=url if isinstance(url, str) else "",
            redirect_chain=(),
            final_destination=None,
            redirect_count=0,
            status_codes=(),
            error=error,
        )