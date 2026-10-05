"""Explanation engine for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.core.models import CurrentCaseState
from qrshield_x.decision.decision_engine import DecisionResult


@dataclass(frozen=True)
class ExplanationResult:
    """Human-readable explanation of a decision."""

    decision: str
    confidence: str
    summary: str
    evidence_summary: str
    investigation_summary: str


class ExplanationEngine:
    """Generate a deterministic explanation for a decision."""

    def explain(
        self,
        state: CurrentCaseState,
        decision: DecisionResult,
    ) -> ExplanationResult:
        """Generate an explanation from the current state and decision."""

        resolved_relationships = sum(
            1
            for relationship_state in state.relationship_states.values()
            if relationship_state.value != "UNASSESSED"
        )

        unresolved_count = len(state.unresolved_relationships)

        return ExplanationResult(
            decision=decision.outcome.value,
            confidence=decision.confidence.value,
            summary=decision.reason,
            evidence_summary=(
                f"Current risk assessment: "
                f"{state.current_risk.value}. "
                f"Evidence sufficiency: "
                f"{state.evidence_sufficiency}. "
                f"Conflict status: "
                f"{state.conflict_status}."
            ),
            investigation_summary=(
                f"{resolved_relationships} relationship(s) assessed, "
                f"{unresolved_count} relationship(s) unresolved, and "
                f"{state.investigations_used} investigation(s) used."
            ),
        )