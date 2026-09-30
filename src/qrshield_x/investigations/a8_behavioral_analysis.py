"""A8 behavioral analysis for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class A8NetworkActivity:
    """One observed runtime network activity."""

    destination: str
    method: str
    resource_type: str
    status_code: int | None = None


@dataclass(frozen=True)
class A8BehavioralActivity:
    """One observed runtime behavioral activity."""

    activity_type: str
    target: str | None = None
    details: str | None = None


@dataclass(frozen=True)
class A8NetworkAnalysisResult:
    """Result of runtime network activity analysis."""

    success: bool
    activities: tuple[A8NetworkActivity, ...]
    unique_destinations: tuple[str, ...]
    request_count: int
    failed_request_count: int
    indicators: tuple[str, ...]
    error: str | None = None


class A8NetworkAnalyzer:
    """Analyze bounded runtime network observations."""

    def analyze(
        self,
        activities: list[A8NetworkActivity],
    ) -> A8NetworkAnalysisResult:
        """Analyze observed runtime network activities."""

        if not isinstance(activities, list):
            return A8NetworkAnalysisResult(
                success=False,
                activities=(),
                unique_destinations=(),
                request_count=0,
                failed_request_count=0,
                indicators=(),
                error="Activities must be provided as a list.",
            )

        unique_destinations = tuple(
            dict.fromkeys(
                activity.destination
                for activity in activities
                if activity.destination
            )
        )

        failed_request_count = sum(
            1
            for activity in activities
            if activity.status_code is not None
            and activity.status_code >= 400
        )

        indicators: list[str] = []

        if activities:
            indicators.append("NETWORK_ACTIVITY_PRESENT")

        if len(unique_destinations) > 1:
            indicators.append("MULTIPLE_DESTINATIONS")

        if failed_request_count:
            indicators.append("FAILED_NETWORK_REQUEST")

        return A8NetworkAnalysisResult(
            success=True,
            activities=tuple(activities),
            unique_destinations=unique_destinations,
            request_count=len(activities),
            failed_request_count=failed_request_count,
            indicators=tuple(indicators),
            error=None,
        )

@dataclass(frozen=True)
class A8BehavioralAnalysisResult:
    """Result of runtime behavioral activity analysis."""

    success: bool
    activities: tuple[A8BehavioralActivity, ...]
    activity_types: tuple[str, ...]
    activity_count: int
    indicators: tuple[str, ...]
    error: str | None = None


class A8BehavioralAnalyzer:
    """Analyze bounded runtime behavioral observations."""

    def analyze(
        self,
        activities: list[A8BehavioralActivity],
    ) -> A8BehavioralAnalysisResult:
        """Analyze observed runtime behavioral activities."""

        if not isinstance(activities, list):
            return A8BehavioralAnalysisResult(
                success=False,
                activities=(),
                activity_types=(),
                activity_count=0,
                indicators=(),
                error="Activities must be provided as a list.",
            )

        activity_types = tuple(
            dict.fromkeys(
                activity.activity_type
                for activity in activities
                if activity.activity_type
            )
        )

        indicators: list[str] = []

        if activities:
            indicators.append("BEHAVIORAL_ACTIVITY_PRESENT")

        if "navigation" in activity_types:
            indicators.append("NAVIGATION_ACTIVITY")

        if "form_submission" in activity_types:
            indicators.append("FORM_SUBMISSION_ACTIVITY")

        if "download" in activity_types:
            indicators.append("DOWNLOAD_ACTIVITY")

        if "script_execution" in activity_types:
            indicators.append("SCRIPT_EXECUTION_ACTIVITY")

        return A8BehavioralAnalysisResult(
            success=True,
            activities=tuple(activities),
            activity_types=activity_types,
            activity_count=len(activities),
            indicators=tuple(indicators),
            error=None,
        )