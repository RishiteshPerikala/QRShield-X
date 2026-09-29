"""Relationship assessment layer for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qrshield_x.core.enums import (
    Confidence,
    RelationshipID,
    RelationshipState,
    RiskContribution,
)
from qrshield_x.core.models import EvidenceRecord


@dataclass(frozen=True)
class RelationshipAssessmentResult:
    """Assessment result for one relationship."""

    relationship_id: RelationshipID
    state: RelationshipState
    confidence: Confidence
    supporting_evidence_ids: tuple[str, ...]


class RelationshipAssessor:
    """Assess relationship state from available evidence."""

    def assess(
        self,
        relationship_id: RelationshipID,
        evidence: Iterable[EvidenceRecord],
    ) -> RelationshipAssessmentResult:
        """
        Assess one relationship from its available evidence.

        The assessment is intentionally conservative:
        - no relevant evidence -> UNASSESSED
        - relevant evidence exists but is unavailable -> UNAVAILABLE
        - conflicting risk contributions -> CONFLICTING
        - relevant evidence with only neutral/unknown contribution
          -> INSUFFICIENT
        - consistent non-neutral evidence -> SUFFICIENT
        """
        relevant_evidence = [
            item
            for item in evidence
            if item.relationship_id == relationship_id
        ]

        if not relevant_evidence:
            return RelationshipAssessmentResult(
                relationship_id=relationship_id,
                state=RelationshipState.UNASSESSED,
                confidence=Confidence.LOW,
                supporting_evidence_ids=(),
            )

        available_evidence = [
            item
            for item in relevant_evidence
            if item.availability.value == "AVAILABLE"
        ]

        if not available_evidence:
            return RelationshipAssessmentResult(
                relationship_id=relationship_id,
                state=RelationshipState.UNAVAILABLE,
                confidence=Confidence.LOW,
                supporting_evidence_ids=tuple(
                    item.evidence_id
                    for item in relevant_evidence
                ),
            )

        contributions = {
            item.risk_contribution
            for item in available_evidence
        }

        if (
            RiskContribution.SUSPICIOUS in contributions
            and RiskContribution.HIGH_RISK in contributions
        ):
            state = RelationshipState.CONFLICTING
            confidence = Confidence.MEDIUM

        elif contributions <= {
            RiskContribution.NEUTRAL,
            RiskContribution.UNKNOWN,
        }:
            state = RelationshipState.INSUFFICIENT
            confidence = Confidence.LOW

        else:
            state = RelationshipState.SUFFICIENT
            confidence = self._derive_confidence(available_evidence)

        return RelationshipAssessmentResult(
            relationship_id=relationship_id,
            state=state,
            confidence=confidence,
            supporting_evidence_ids=tuple(
                item.evidence_id
                for item in available_evidence
            ),
        )

    @staticmethod
    def _derive_confidence(
        evidence: list[EvidenceRecord],
    ) -> Confidence:
        """Derive relationship confidence from evidence reliability."""
        reliabilities = {
            item.reliability.value
            for item in evidence
        }

        if "HIGH" in reliabilities and len(evidence) >= 2:
            return Confidence.HIGH

        if "HIGH" in reliabilities or "MEDIUM" in reliabilities:
            return Confidence.MEDIUM

        return Confidence.LOW