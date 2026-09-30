from qrshield_x.investigations.a8_behavioral_analysis import (
    A8NetworkActivity,
    A8NetworkAnalyzer,
)


def test_network_activity_is_analyzed() -> None:
    analyzer = A8NetworkAnalyzer()

    activities = [
        A8NetworkActivity(
            destination="https://example.com",
            method="GET",
            resource_type="document",
            status_code=200,
        ),
        A8NetworkActivity(
            destination="https://example.com/login",
            method="GET",
            resource_type="document",
            status_code=200,
        ),
    ]

    result = analyzer.analyze(activities)

    assert result.success is True
    assert result.request_count == 2
    assert result.unique_destinations == (
        "https://example.com",
        "https://example.com/login",
    )
    assert "NETWORK_ACTIVITY_PRESENT" in result.indicators
    assert "MULTIPLE_DESTINATIONS" in result.indicators


def test_failed_network_requests_are_detected() -> None:
    analyzer = A8NetworkAnalyzer()

    activities = [
        A8NetworkActivity(
            destination="https://example.com",
            method="GET",
            resource_type="document",
            status_code=200,
        ),
        A8NetworkActivity(
            destination="https://example.com/collector",
            method="POST",
            resource_type="xhr",
            status_code=403,
        ),
    ]

    result = analyzer.analyze(activities)

    assert result.success is True
    assert result.failed_request_count == 1
    assert "FAILED_NETWORK_REQUEST" in result.indicators


def test_empty_network_activity_is_valid() -> None:
    analyzer = A8NetworkAnalyzer()

    result = analyzer.analyze([])

    assert result.success is True
    assert result.request_count == 0
    assert result.failed_request_count == 0
    assert result.indicators == ()


def test_invalid_network_activity_input_fails() -> None:
    analyzer = A8NetworkAnalyzer()

    result = analyzer.analyze(None)  # type: ignore[arg-type]

    assert result.success is False
    assert result.error == "Activities must be provided as a list."

def test_behavioral_activities_are_analyzed() -> None:
    from qrshield_x.investigations.a8_behavioral_analysis import (
        A8BehavioralActivity,
        A8BehavioralAnalyzer,
    )

    analyzer = A8BehavioralAnalyzer()

    activities = [
        A8BehavioralActivity(
            activity_type="navigation",
            target="https://example.com/login",
        ),
        A8BehavioralActivity(
            activity_type="form_submission",
            target="/login",
        ),
        A8BehavioralActivity(
            activity_type="script_execution",
            details="inline script executed",
        ),
    ]

    result = analyzer.analyze(activities)

    assert result.success is True
    assert result.activity_count == 3
    assert result.activity_types == (
        "navigation",
        "form_submission",
        "script_execution",
    )
    assert "BEHAVIORAL_ACTIVITY_PRESENT" in result.indicators
    assert "NAVIGATION_ACTIVITY" in result.indicators
    assert "FORM_SUBMISSION_ACTIVITY" in result.indicators
    assert "SCRIPT_EXECUTION_ACTIVITY" in result.indicators


def test_download_activity_is_detected() -> None:
    from qrshield_x.investigations.a8_behavioral_analysis import (
        A8BehavioralActivity,
        A8BehavioralAnalyzer,
    )

    analyzer = A8BehavioralAnalyzer()

    activities = [
        A8BehavioralActivity(
            activity_type="download",
            target="document.pdf",
        ),
    ]

    result = analyzer.analyze(activities)

    assert result.success is True
    assert result.activity_count == 1
    assert "DOWNLOAD_ACTIVITY" in result.indicators


def test_empty_behavioral_activity_is_valid() -> None:
    from qrshield_x.investigations.a8_behavioral_analysis import (
        A8BehavioralAnalyzer,
    )

    analyzer = A8BehavioralAnalyzer()

    result = analyzer.analyze([])

    assert result.success is True
    assert result.activity_count == 0
    assert result.activity_types == ()
    assert result.indicators == ()


def test_invalid_behavioral_activity_input_fails() -> None:
    from qrshield_x.investigations.a8_behavioral_analysis import (
        A8BehavioralAnalyzer,
    )

    analyzer = A8BehavioralAnalyzer()

    result = analyzer.analyze(None)  # type: ignore[arg-type]

    assert result.success is False
    assert result.error == "Activities must be provided as a list."