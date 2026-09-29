"""Evidence assessment layer for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.core.enums import (
    Availability,
    Confidence,
    Reliability,
    RiskContribution,
)
from qrshield_x.core.models import EvidenceRecord


@dataclass(frozen=True)
class EvidenceAssessmentResult:
    """Assessment result for one evidence record."""

    evidence_id: str
    risk_contribution: RiskContribution
    confidence: Confidence
    reliability: Reliability
    availability: Availability
    usable: bool


class EvidenceAssessor:
    """Assess whether an evidence record is usable and characterize it."""

    def assess(
        self,
        evidence: EvidenceRecord,
    ) -> EvidenceAssessmentResult:
        """
        Assess one evidence record.

        The assessor preserves the evidence record's explicit assessment
        dimensions and derives whether the evidence is currently usable.

        Evidence is usable only when it is available and has a known
        risk contribution.
        """
        usable = (
            evidence.availability == Availability.AVAILABLE
            and evidence.risk_contribution != RiskContribution.UNKNOWN
        )

        return EvidenceAssessmentResult(
            evidence_id=evidence.evidence_id,
            risk_contribution=evidence.risk_contribution,
            confidence=evidence.confidence,
            reliability=evidence.reliability,
            availability=evidence.availability,
            usable=usable,
        )