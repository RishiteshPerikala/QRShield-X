"""Unit tests for A7 web analysis."""

import cv2
import numpy as np

from qrshield_x.investigations.a7_web_analysis import (
    A7DOMAnalyzer,
    A7FormAnalyzer,
    A7ScriptLinkAnalyzer,
    A7VisualBrandAnalyzer,
)




def test_analyzes_basic_html_structure() -> None:
    analyzer = A7DOMAnalyzer()

    html = """
    <html>
        <head>
            <title>Example Page</title>
        </head>
        <body>
            <a href="/home">Home</a>
            <a href="/about">About</a>
            <img src="logo.png">
            <form action="/login">
                <input type="text">
            </form>
            <script src="app.js"></script>
        </body>
    </html>
    """

    result = analyzer.analyze(html)

    assert result.success is True
    assert result.has_html_tag is True
    assert result.has_head_tag is True
    assert result.has_body_tag is True
    assert result.has_title is True
    assert result.link_count == 2
    assert result.form_count == 1
    assert result.script_count == 1
    assert result.image_count == 1
    assert result.element_count > 0
    assert result.html_length == len(html)


def test_counts_multiple_element_types() -> None:
    analyzer = A7DOMAnalyzer()

    html = """
    <html>
        <head>
            <title>Test</title>
        </head>
        <body>
            <a href="/one">One</a>
            <a href="/two">Two</a>
            <a href="/three">Three</a>
            <form></form>
            <form></form>
            <script></script>
            <script></script>
            <img>
            <img>
        </body>
    </html>
    """

    result = analyzer.analyze(html)

    assert result.success is True
    assert result.link_count == 3
    assert result.form_count == 2
    assert result.script_count == 2
    assert result.image_count == 2


def test_structure_flags_can_be_false() -> None:
    analyzer = A7DOMAnalyzer()

    html = "<div>Hello</div>"

    result = analyzer.analyze(html)

    assert result.success is True
    assert result.has_html_tag is False
    assert result.has_head_tag is False
    assert result.has_body_tag is False
    assert result.has_title is False
    assert result.element_count == 1


def test_empty_html_fails_cleanly() -> None:
    analyzer = A7DOMAnalyzer()

    result = analyzer.analyze("")

    assert result.success is False
    assert result.error == (
        "HTML content must be a non-empty string."
    )


def test_whitespace_only_html_fails_cleanly() -> None:
    analyzer = A7DOMAnalyzer()

    result = analyzer.analyze("   ")

    assert result.success is False
    assert result.error == (
        "HTML content must be a non-empty string."
    )


def test_malformed_html_is_handled() -> None:
    analyzer = A7DOMAnalyzer()

    html = """
    <html>
        <body>
            <div>
            <a href="/test">Test
    """

    result = analyzer.analyze(html)

    assert result.success is True
    assert result.has_html_tag is True
    assert result.has_body_tag is True
    assert result.link_count == 1


def test_script_content_is_not_executed() -> None:
    analyzer = A7DOMAnalyzer()

    html = """
    <html>
        <body>
            <script>
                raise RuntimeError("This must never execute")
            </script>
        </body>
    </html>
    """

    result = analyzer.analyze(html)

    assert result.success is True
    assert result.script_count == 1

    from qrshield_x.investigations.a7_web_analysis import A7FormAnalyzer


def test_form_analysis_detects_password_and_username_fields():
    html = """
    <html>
        <body>
            <form>
                <input type="text" name="username">
                <input type="password" name="password">
                <button type="submit">Login</button>
            </form>
        </body>
    </html>
    """

    result = A7FormAnalyzer().analyze(html)

    assert result.success is True
    assert result.form_count == 1
    assert result.input_count == 2
    assert result.password_input_count == 1
    assert result.username_input_count == 1
    assert result.submit_control_count == 1
    assert result.credential_collection_detected is True
    assert "PASSWORD_INPUT" in result.indicators
    assert "USERNAME_INPUT" in result.indicators
    assert "CREDENTIAL_COLLECTION" in result.indicators


def test_form_analysis_detects_email_field():
    html = """
    <form>
        <input type="email" name="email">
        <button type="submit">Continue</button>
    </form>
    """

    result = A7FormAnalyzer().analyze(html)

    assert result.success is True
    assert result.form_count == 1
    assert result.email_input_count == 1
    assert result.submit_control_count == 1
    assert result.credential_collection_detected is False
    assert "EMAIL_INPUT" in result.indicators


def test_form_analysis_detects_multiple_forms_and_inputs():
    html = """
    <form>
        <input type="text" name="username">
        <input type="password" name="password">
    </form>

    <form>
        <input type="email" name="email">
        <input type="submit" value="Submit">
    </form>
    """

    result = A7FormAnalyzer().analyze(html)

    assert result.form_count == 2
    assert result.input_count == 4
    assert result.password_input_count == 1
    assert result.email_input_count == 1
    assert result.username_input_count == 1
    assert result.submit_control_count == 1
    assert result.credential_collection_detected is True


def test_form_analysis_detects_autocomplete_username():
    html = """
    <form>
        <input
            type="text"
            autocomplete="username"
        >
    </form>
    """

    result = A7FormAnalyzer().analyze(html)

    assert result.username_input_count == 1
    assert "USERNAME_INPUT" in result.indicators


def test_form_analysis_handles_non_credential_form():
    html = """
    <form>
        <input type="text" name="search">
        <button type="submit">Search</button>
    </form>
    """

    result = A7FormAnalyzer().analyze(html)

    assert result.success is True
    assert result.form_count == 1
    assert result.input_count == 1
    assert result.password_input_count == 0
    assert result.email_input_count == 0
    assert result.username_input_count == 0
    assert result.credential_collection_detected is False

from qrshield_x.investigations.a7_web_analysis import (
    A7DOMAnalyzer,
    A7FormAnalyzer,
    A7ScriptLinkAnalyzer,
)

def test_form_analysis_rejects_empty_html():
    result = A7FormAnalyzer().analyze("")

    assert result.success is False
    assert result.error is not None
    assert result.credential_collection_detected is False
def test_script_link_analysis_detects_inline_and_external_scripts():
    html = """
    <html>
        <head>
            <script>
                console.log("test");
            </script>
            <script src="https://cdn.example.com/app.js"></script>
        </head>
    </html>
    """

    result = A7ScriptLinkAnalyzer().analyze(
        html,
        page_hostname="example.com",
    )

    assert result.success is True
    assert result.script_count == 2
    assert result.inline_script_count == 1
    assert result.external_script_count == 1
    assert "SCRIPT_PRESENT" in result.indicators
    assert "INLINE_SCRIPT" in result.indicators
    assert "EXTERNAL_SCRIPT" in result.indicators


def test_script_link_analysis_classifies_internal_and_external_links():
    html = """
    <a href="/login">Login</a>
    <a href="https://example.com/account">Account</a>
    <a href="https://other.example.com/help">Help</a>
    <a href="#section">Section</a>
    """

    result = A7ScriptLinkAnalyzer().analyze(
        html,
        page_hostname="example.com",
    )

    assert result.success is True
    assert result.link_count == 4
    assert result.internal_link_count == 3
    assert result.external_link_count == 1
    assert "EXTERNAL_LINK" in result.indicators


def test_script_link_analysis_detects_javascript_links():
    html = """
    <a href="javascript:void(0)">Click</a>
    <a href="/safe">Safe link</a>
    """

    result = A7ScriptLinkAnalyzer().analyze(
        html,
        page_hostname="example.com",
    )

    assert result.success is True
    assert result.link_count == 2
    assert result.javascript_link_count == 1
    assert result.internal_link_count == 1
    assert "JAVASCRIPT_LINK" in result.indicators


def test_script_link_analysis_handles_no_scripts_or_links():
    html = """
    <html>
        <body>
            <p>Hello</p>
        </body>
    </html>
    """

    result = A7ScriptLinkAnalyzer().analyze(
        html,
        page_hostname="example.com",
    )

    assert result.success is True
    assert result.script_count == 0
    assert result.inline_script_count == 0
    assert result.external_script_count == 0
    assert result.link_count == 0
    assert result.internal_link_count == 0
    assert result.external_link_count == 0
    assert result.javascript_link_count == 0
    assert result.indicators == ()


def test_script_link_analysis_handles_empty_html():
    result = A7ScriptLinkAnalyzer().analyze("")

    assert result.success is False
    assert result.error is not None

def test_visual_brand_analysis_reads_image(tmp_path):
    image = np.full(
        (100, 200, 3),
        128,
        dtype=np.uint8,
    )

    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), image)

    result = A7VisualBrandAnalyzer().analyze(image_path)

    assert result.success is True
    assert result.image_width == 200
    assert result.image_height == 100
    assert result.image_area == 20000
    assert result.grayscale_mean is not None
    assert result.grayscale_std is not None
    assert result.edge_density is not None
    assert result.reference_similarity is None


def test_visual_brand_analysis_detects_low_contrast(tmp_path):
    image = np.full(
        (100, 100, 3),
        128,
        dtype=np.uint8,
    )

    image_path = tmp_path / "low_contrast.png"
    cv2.imwrite(str(image_path), image)

    result = A7VisualBrandAnalyzer().analyze(image_path)

    assert result.success is True
    assert "LOW_VISUAL_CONTRAST" in result.visual_indicators


def test_visual_brand_analysis_detects_reference_similarity(tmp_path):
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    reference = np.zeros(
        (50, 50, 3),
        dtype=np.uint8,
    )

    image_path = tmp_path / "page.png"
    reference_path = tmp_path / "reference.png"

    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(reference_path), reference)

    result = A7VisualBrandAnalyzer().analyze(
        image_path,
        reference_image_path=reference_path,
    )

    assert result.success is True
    assert result.reference_similarity == 1.0
    assert "REFERENCE_COMPARISON_AVAILABLE" in result.visual_indicators


def test_visual_brand_analysis_handles_different_images(tmp_path):
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    reference = np.full(
        (100, 100, 3),
        255,
        dtype=np.uint8,
    )

    image_path = tmp_path / "page.png"
    reference_path = tmp_path / "reference.png"

    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(reference_path), reference)

    result = A7VisualBrandAnalyzer().analyze(
        image_path,
        reference_image_path=reference_path,
    )

    assert result.success is True
    assert result.reference_similarity == 0.0


def test_visual_brand_analysis_handles_missing_image(tmp_path):
    image_path = tmp_path / "missing.png"

    result = A7VisualBrandAnalyzer().analyze(image_path)

    assert result.success is False
    assert result.error is not None
    assert result.reference_similarity is None