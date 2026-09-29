"""Unit tests for the relationship assessment layer."""

from datetime import datetime, timezone

from qrshield_x.assessment.relationship_assessment import RelationshipAssessor
from qrshield_x.core.enums import (
    Availability,
    Confidence,
    EvidenceType,
    RelationshipID,
    RelationshipState,
    Reliability,
    RiskContribution,
)
from qrshield_x.core.models import EvidenceRecord


def make_evidence(
    *,
    evidence_id: str,
    relationship_id: RelationshipID,
    risk_contribution: RiskContribution,
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
        relationship_id=relationship_id,
        risk_contribution=risk_contribution,
        confidence=confidence,
        reliability=reliability,
        availability=availability,
        timestamp=datetime.now(timezone.utc),
        provenance_id="PROV-001",
    )


def test_no_evidence_is_unassessed() -> None:
    """A relationship without evidence remains unassessed."""
    assessor = RelationshipAssessor()

    result = assessor.assess(
        RelationshipID.R1,
        [],
    )

    assert result.state == RelationshipState.UNASSESSED
    assert result.confidence == Confidence.LOW
    assert result.supporting_evidence_ids == ()


def test_unavailable_evidence_is_unavailable() -> None:
    """Unavailable evidence should produce an unavailable relationship."""
    assessor = RelationshipAssessor()

    evidence = [
        make_evidence(
            evidence_id="E1",
            relationship_id=RelationshipID.R1,
            risk_contribution=RiskContribution.UNKNOWN,
            availability=Availability.UNAVAILABLE,
        )
    ]

    result = assessor.assess(RelationshipID.R1, evidence)

    assert result.state == RelationshipState.UNAVAILABLE
    assert result.supporting_evidence_ids == ("E1",)


def test_neutral_evidence_is_insufficient() -> None:
    """Only neutral/unknown evidence should remain insufficient."""
    assessor = RelationshipAssessor()

    evidence = [
        make_evidence(
            evidence_id="E1",
            relationship_id=RelationshipID.R1,
            risk_contribution=RiskContribution.NEUTRAL,
        )
    ]

    result = assessor.assess(RelationshipID.R1, evidence)

    assert result.state == RelationshipState.INSUFFICIENT
    assert result.confidence == Confidence.LOW


def test_consistent_evidence_is_sufficient() -> None:
    """Consistent non-neutral evidence should establish the relationship."""
    assessor = RelationshipAssessor()

    evidence = [
        make_evidence(
            evidence_id="E1",
            relationship_id=RelationshipID.R2,
            risk_contribution=RiskContribution.SUSPICIOUS,
            confidence=Confidence.HIGH,
            reliability=Reliability.HIGH,
        ),
        make_evidence(
            evidence_id="E2",
            relationship_id=RelationshipID.R2,
            risk_contribution=RiskContribution.SUSPICIOUS,
            confidence=Confidence.MEDIUM,
            reliability=Reliability.HIGH,
        ),
    ]

    result = assessor.assess(RelationshipID.R2, evidence)

    assert result.state == RelationshipState.SUFFICIENT
    assert result.confidence == Confidence.HIGH
    assert result.supporting_evidence_ids == ("E1", "E2")


def test_conflicting_evidence_is_conflicting() -> None:
    """Suspicious and high-risk evidence should expose conflict."""
    assessor = RelationshipAssessor()

    evidence = [
        make_evidence(
            evidence_id="E1",
            relationship_id=RelationshipID.R3,
            risk_contribution=RiskContribution.SUSPICIOUS,
        ),
        make_evidence(
            evidence_id="E2",
            relationship_id=RelationshipID.R3,
            risk_contribution=RiskContribution.HIGH_RISK,
        ),
    ]

    result = assessor.assess(RelationshipID.R3, evidence)

    assert result.state == RelationshipState.CONFLICTING
    assert result.confidence == Confidence.MEDIUM


def test_evidence_for_other_relationship_is_ignored() -> None:
    """Evidence belonging to another relationship must not affect assessment."""
    assessor = RelationshipAssessor()

    evidence = [
        make_evidence(
            evidence_id="E1",
            relationship_id=RelationshipID.R2,
            risk_contribution=RiskContribution.HIGH_RISK,
        )
    ]

    result = assessor.assess(RelationshipID.R1, evidence)

    assert result.state == RelationshipState.UNASSESSED
    assert result.supporting_evidence_ids == ()