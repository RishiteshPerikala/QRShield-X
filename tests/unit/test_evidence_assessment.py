"""Unit tests for the evidence assessment layer."""

from datetime import datetime, timezone

from qrshield_x.assessment.evidence_assessment import EvidenceAssessor
from qrshield_x.core.enums import (
    Availability,
    Confidence,
    EvidenceType,
    RelationshipID,
    Reliability,
    RiskContribution,
)
from qrshield_x.core.models import EvidenceRecord


def make_evidence(
    *,
    evidence_id: str = "E1",
    risk_contribution: RiskContribution = RiskContribution.SUSPICIOUS,
    confidence: Confidence = Confidence.MEDIUM,
    reliability: Reliability = Reliability.MEDIUM,
    availability: Availability = Availability.AVAILABLE,
) -> EvidenceRecord:
    """Create a test evidence record."""
    return EvidenceRecord(
        evidence_id=evidence_id,
        case_id="CASE-001",
        evidence_type=EvidenceType.URL_LEXICAL_FEATURES,
        source_action="A3",
        observation={"feature": "test"},
        relationship_id=RelationshipID.R2,
        risk_contribution=risk_contribution,
        confidence=confidence,
        reliability=reliability,
        availability=availability,
        timestamp=datetime.now(timezone.utc),
        provenance_id="PROV-001",
    )


def test_available_known_evidence_is_usable() -> None:
    """Available evidence with a known contribution should be usable."""
    assessor = EvidenceAssessor()

    result = assessor.assess(make_evidence())

    assert result.evidence_id == "E1"
    assert result.risk_contribution == RiskContribution.SUSPICIOUS
    assert result.confidence == Confidence.MEDIUM
    assert result.reliability == Reliability.MEDIUM
    assert result.availability == Availability.AVAILABLE
    assert result.usable is True


def test_unavailable_evidence_is_not_usable() -> None:
    """Unavailable evidence should not be usable."""
    assessor = EvidenceAssessor()

    evidence = make_evidence(
        availability=Availability.UNAVAILABLE,
    )

    result = assessor.assess(evidence)

    assert result.availability == Availability.UNAVAILABLE
    assert result.usable is False


def test_unknown_contribution_is_not_usable() -> None:
    """Evidence with an unknown risk contribution should not be usable."""
    assessor = EvidenceAssessor()

    evidence = make_evidence(
        risk_contribution=RiskContribution.UNKNOWN,
    )

    result = assessor.assess(evidence)

    assert result.risk_contribution == RiskContribution.UNKNOWN
    assert result.usable is False


def test_neutral_available_evidence_is_usable() -> None:
    """Neutral evidence is still usable when it is available."""
    assessor = EvidenceAssessor()

    evidence = make_evidence(
        risk_contribution=RiskContribution.NEUTRAL,
    )

    result = assessor.assess(evidence)

    assert result.risk_contribution == RiskContribution.NEUTRAL
    assert result.usable is True


def test_high_confidence_and_reliability_are_preserved() -> None:
    """Explicit confidence and reliability should be preserved."""
    assessor = EvidenceAssessor()

    evidence = make_evidence(
        confidence=Confidence.HIGH,
        reliability=Reliability.HIGH,
        risk_contribution=RiskContribution.HIGH_RISK,
    )

    result = assessor.assess(evidence)

    assert result.confidence == Confidence.HIGH
    assert result.reliability == Reliability.HIGH
    assert result.risk_contribution == RiskContribution.HIGH_RISK
    assert result.usable is True