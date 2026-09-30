"""A7 web-content analysis for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse
from pathlib import Path

import cv2


@dataclass(frozen=True)
class A7DOMAnalysisResult:
    """Result of HTML/DOM structure analysis."""

    success: bool
    html_length: int
    has_html_tag: bool
    has_head_tag: bool
    has_body_tag: bool
    has_title: bool
    link_count: int
    form_count: int
    script_count: int
    image_count: int
    element_count: int
    error: str | None = None


@dataclass(frozen=True)
class A7FormAnalysisResult:
    """Result of HTML form and credential-collection analysis."""

    success: bool
    form_count: int
    input_count: int
    password_input_count: int
    email_input_count: int
    username_input_count: int
    submit_control_count: int
    credential_collection_detected: bool
    indicators: tuple[str, ...]
    error: str | None = None

@dataclass(frozen=True)
class A7ScriptLinkAnalysisResult:
    """Result of script and link analysis."""

    success: bool
    script_count: int
    inline_script_count: int
    external_script_count: int
    link_count: int
    internal_link_count: int
    external_link_count: int
    javascript_link_count: int
    indicators: tuple[str, ...]
    error: str | None = None

@dataclass(frozen=True)
class A7VisualBrandEvidenceResult:
    """Result of visual brand evidence analysis."""

    success: bool
    image_width: int
    image_height: int
    image_area: int
    grayscale_mean: float | None
    grayscale_std: float | None
    edge_density: float | None
    reference_similarity: float | None
    visual_indicators: tuple[str, ...]
    error: str | None = None


class _DOMParser(HTMLParser):
    """Collect basic structural HTML statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.has_html_tag = False
        self.has_head_tag = False
        self.has_body_tag = False
        self.has_title = False

        self.link_count = 0
        self.form_count = 0
        self.script_count = 0
        self.image_count = 0
        self.element_count = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()

        self.element_count += 1

        if tag == "html":
            self.has_html_tag = True
        elif tag == "head":
            self.has_head_tag = True
        elif tag == "body":
            self.has_body_tag = True
        elif tag == "title":
            self.has_title = True
        elif tag == "a":
            self.link_count += 1
        elif tag == "form":
            self.form_count += 1
        elif tag == "script":
            self.script_count += 1
        elif tag == "img":
            self.image_count += 1


class _FormParser(HTMLParser):
    """Collect form and credential-related HTML indicators."""

    _USERNAME_NAMES = {
        "username",
        "user",
        "userid",
        "user_id",
        "login",
        "loginid",
        "login_id",
    }

    _EMAIL_NAMES = {
        "email",
        "emailaddress",
        "email_address",
    }

    def __init__(self) -> None:
        super().__init__()

        self.form_count = 0
        self.input_count = 0
        self.password_input_count = 0
        self.email_input_count = 0
        self.username_input_count = 0
        self.submit_control_count = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()

        attributes = {
            key.lower(): value.lower() if value else ""
            for key, value in attrs
        }

        if tag == "form":
            self.form_count += 1
            return

        if tag == "input":
            self.input_count += 1

            input_type = attributes.get("type", "text")
            name = attributes.get("name", "").strip().lower()
            input_id = attributes.get("id", "").strip().lower()
            autocomplete = attributes.get("autocomplete", "").strip().lower()

            if input_type == "password":
                self.password_input_count += 1

            if input_type == "email" or autocomplete == "email":
                self.email_input_count += 1

            if (
                name in self._USERNAME_NAMES
                or input_id in self._USERNAME_NAMES
                or autocomplete in {"username", "current-username"}
            ):
                self.username_input_count += 1

            if input_type in {"submit", "button"}:
                self.submit_control_count += 1

            return

        if tag == "button":
            button_type = attributes.get("type", "submit")

            if button_type in {"submit", "button"}:
                self.submit_control_count += 1


class A7DOMAnalyzer:
    """Analyze basic HTML/DOM structure without executing web content."""

    def analyze(self, html: str) -> A7DOMAnalysisResult:
        """Analyze an HTML document."""
        if not isinstance(html, str) or not html.strip():
            return self._failure(
                error="HTML content must be a non-empty string."
            )

        parser = _DOMParser()

        try:
            parser.feed(html)
            parser.close()
        except (ValueError, TypeError) as exc:
            return self._failure(
                error=f"HTML parsing failed: {exc}"
            )

        return A7DOMAnalysisResult(
            success=True,
            html_length=len(html),
            has_html_tag=parser.has_html_tag,
            has_head_tag=parser.has_head_tag,
            has_body_tag=parser.has_body_tag,
            has_title=parser.has_title,
            link_count=parser.link_count,
            form_count=parser.form_count,
            script_count=parser.script_count,
            image_count=parser.image_count,
            element_count=parser.element_count,
            error=None,
        )

    @staticmethod
    def _failure(
        *,
        error: str,
    ) -> A7DOMAnalysisResult:
        return A7DOMAnalysisResult(
            success=False,
            html_length=0,
            has_html_tag=False,
            has_head_tag=False,
            has_body_tag=False,
            has_title=False,
            link_count=0,
            form_count=0,
            script_count=0,
            image_count=0,
            element_count=0,
            error=error,
        )


class A7FormAnalyzer:
    """Analyze forms and credential-collection indicators."""

    def analyze(self, html: str) -> A7FormAnalysisResult:
        """Analyze HTML forms without executing page content."""
        if not isinstance(html, str) or not html.strip():
            return self._failure(
                error="HTML content must be a non-empty string."
            )

        parser = _FormParser()

        try:
            parser.feed(html)
            parser.close()
        except (ValueError, TypeError) as exc:
            return self._failure(
                error=f"HTML form parsing failed: {exc}"
            )

        indicators: list[str] = []

        if parser.form_count > 0:
            indicators.append("FORM_PRESENT")

        if parser.password_input_count > 0:
            indicators.append("PASSWORD_INPUT")

        if parser.email_input_count > 0:
            indicators.append("EMAIL_INPUT")

        if parser.username_input_count > 0:
            indicators.append("USERNAME_INPUT")

        if parser.submit_control_count > 0:
            indicators.append("SUBMIT_CONTROL")

        credential_collection_detected = (
            parser.password_input_count > 0
            or (
                parser.username_input_count > 0
                and parser.email_input_count > 0
            )
        )

        if credential_collection_detected:
            indicators.append("CREDENTIAL_COLLECTION")

        return A7FormAnalysisResult(
            success=True,
            form_count=parser.form_count,
            input_count=parser.input_count,
            password_input_count=parser.password_input_count,
            email_input_count=parser.email_input_count,
            username_input_count=parser.username_input_count,
            submit_control_count=parser.submit_control_count,
            credential_collection_detected=credential_collection_detected,
            indicators=tuple(indicators),
            error=None,
        )

    @staticmethod
    def _failure(
        *,
        error: str,
    ) -> A7FormAnalysisResult:
        return A7FormAnalysisResult(
            success=False,
            form_count=0,
            input_count=0,
            password_input_count=0,
            email_input_count=0,
            username_input_count=0,
            submit_control_count=0,
            credential_collection_detected=False,
            indicators=(),
            error=error,
        )


class _ScriptLinkParser(HTMLParser):
    """Collect static script and link indicators."""

    def __init__(self, page_hostname: str | None) -> None:
        super().__init__()

        self.page_hostname = (
            page_hostname.lower().strip().rstrip(".")
            if page_hostname
            else None
        )

        self.script_count = 0
        self.inline_script_count = 0
        self.external_script_count = 0

        self.link_count = 0
        self.internal_link_count = 0
        self.external_link_count = 0
        self.javascript_link_count = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()

        attributes = {
            key.lower(): value.strip() if value else ""
            for key, value in attrs
        }

        if tag == "script":
            self.script_count += 1

            src = attributes.get("src", "")

            if src:
                self.external_script_count += 1
            else:
                self.inline_script_count += 1

            return

        if tag != "a":
            return

        self.link_count += 1

        href = attributes.get("href", "").strip()

        if not href:
            return

        if href.lower().startswith("javascript:"):
            self.javascript_link_count += 1
            return

        parsed = urlparse(href)

        if parsed.scheme.lower() in {"http", "https"}:
            hostname = parsed.hostname

            if (
                hostname
                and self.page_hostname
                and hostname.lower().rstrip(".") == self.page_hostname
            ):
                self.internal_link_count += 1
            else:
                self.external_link_count += 1

        elif href.startswith(("/", "#", "?")):
            self.internal_link_count += 1

class A7ScriptLinkAnalyzer:
    """Analyze scripts and links without executing or fetching content."""

    def analyze(
        self,
        html: str,
        *,
        page_hostname: str | None = None,
    ) -> A7ScriptLinkAnalysisResult:
        """Analyze static script and link structure."""
        if not isinstance(html, str) or not html.strip():
            return self._failure(
                error="HTML content must be a non-empty string."
            )

        parser = _ScriptLinkParser(page_hostname)

        try:
            parser.feed(html)
            parser.close()
        except (ValueError, TypeError) as exc:
            return self._failure(
                error=f"Script/link parsing failed: {exc}"
            )

        indicators: list[str] = []

        if parser.script_count > 0:
            indicators.append("SCRIPT_PRESENT")

        if parser.inline_script_count > 0:
            indicators.append("INLINE_SCRIPT")

        if parser.external_script_count > 0:
            indicators.append("EXTERNAL_SCRIPT")

        if parser.external_link_count > 0:
            indicators.append("EXTERNAL_LINK")

        if parser.javascript_link_count > 0:
            indicators.append("JAVASCRIPT_LINK")

        return A7ScriptLinkAnalysisResult(
            success=True,
            script_count=parser.script_count,
            inline_script_count=parser.inline_script_count,
            external_script_count=parser.external_script_count,
            link_count=parser.link_count,
            internal_link_count=parser.internal_link_count,
            external_link_count=parser.external_link_count,
            javascript_link_count=parser.javascript_link_count,
            indicators=tuple(indicators),
            error=None,
        )

    @staticmethod
    def _failure(
        *,
        error: str,
    ) -> A7ScriptLinkAnalysisResult:
        return A7ScriptLinkAnalysisResult(
            success=False,
            script_count=0,
            inline_script_count=0,
            external_script_count=0,
            link_count=0,
            internal_link_count=0,
            external_link_count=0,
            javascript_link_count=0,
            indicators=(),
            error=error,
        )

class A7VisualBrandAnalyzer:
    """Analyze visual evidence without making identity claims."""

    def analyze(
        self,
        image_path: str | Path,
        *,
        reference_image_path: str | Path | None = None,
    ) -> A7VisualBrandEvidenceResult:
        """Analyze an image and optionally compare it with a reference image."""

        path = Path(image_path)

        if not path.is_file():
            return self._failure(
                error=f"Image file does not exist: {path}"
            )

        image = cv2.imread(str(path))

        if image is None:
            return self._failure(
                error=f"Unable to read image: {path}"
            )

        height, width = image.shape[:2]

        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        grayscale_mean = float(grayscale.mean())
        grayscale_std = float(grayscale.std())

        edges = cv2.Canny(
            grayscale,
            100,
            200,
        )

        edge_density = float(
            (edges > 0).mean()
        )

        reference_similarity = None

        if reference_image_path is not None:
            reference_path = Path(reference_image_path)

            if reference_path.is_file():
                reference_image = cv2.imread(
                    str(reference_path)
                )

                if reference_image is not None:
                    reference_similarity = (
                        self._calculate_similarity(
                            image,
                            reference_image,
                        )
                    )

        indicators: list[str] = []

        if grayscale_std < 20:
            indicators.append("LOW_VISUAL_CONTRAST")

        if edge_density < 0.02:
            indicators.append("LOW_EDGE_DENSITY")

        if reference_similarity is not None:
            indicators.append("REFERENCE_COMPARISON_AVAILABLE")

        return A7VisualBrandEvidenceResult(
            success=True,
            image_width=width,
            image_height=height,
            image_area=width * height,
            grayscale_mean=grayscale_mean,
            grayscale_std=grayscale_std,
            edge_density=edge_density,
            reference_similarity=reference_similarity,
            visual_indicators=tuple(indicators),
            error=None,
        )

    @staticmethod
    def _calculate_similarity(
        image: object,
        reference_image: object,
    ) -> float:
        """Calculate a simple normalized grayscale similarity."""

        image_array = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        reference_array = cv2.cvtColor(
            reference_image,
            cv2.COLOR_BGR2GRAY,
        )

        reference_array = cv2.resize(
            reference_array,
            (
                image_array.shape[1],
                image_array.shape[0],
            ),
        )

        image_float = image_array.astype("float32")
        reference_float = reference_array.astype("float32")

        difference = cv2.absdiff(
            image_float,
            reference_float,
        )

        mean_difference = float(
            difference.mean()
        )

        similarity = 1.0 - (
            mean_difference / 255.0
        )

        return max(
            0.0,
            min(1.0, similarity),
        )

    @staticmethod
    def _failure(
        *,
        error: str,
    ) -> A7VisualBrandEvidenceResult:
        return A7VisualBrandEvidenceResult(
            success=False,
            image_width=0,
            image_height=0,
            image_area=0,
            grayscale_mean=None,
            grayscale_std=None,
            edge_density=None,
            reference_similarity=None,
            visual_indicators=(),
            error=error,
        )