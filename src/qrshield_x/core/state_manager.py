"""Runtime state management for QRShield-X."""

from __future__ import annotations

from qrshield_x.core.enums import (
    Confidence,
    DecisionOutcome,
    RelationshipID,
    RelationshipState,
    RiskContribution,
)
from qrshield_x.core.models import CurrentCaseState


class StateManager:
    """Manage the derived runtime state of an investigation case."""

    def __init__(
        self,
        *,
        budget: float,
    ) -> None:
        """Initialize the case state manager."""
        if budget < 0:
            raise ValueError("Budget cannot be negative.")

        self._state = CurrentCaseState(
            current_risk=RiskContribution.UNKNOWN,
            current_confidence=Confidence.LOW,
            evidence_sufficiency="INSUFFICIENT",
            conflict_status="NONE",
            relationship_states={
                relationship_id: RelationshipState.UNASSESSED
                for relationship_id in RelationshipID
            },
            unresolved_relationships=list(RelationshipID),
            investigations_used=0,
            budget_remaining=budget,
            current_decision=None,
        )

    def get_state(self) -> CurrentCaseState:
        """Return the current runtime state."""
        return self._state.model_copy(deep=True)

    def update_relationship(
        self,
        relationship_id: RelationshipID,
        state: RelationshipState,
    ) -> None:
        """Update the current state of one relationship."""
        self._state.relationship_states[relationship_id] = state

        if state == RelationshipState.UNASSESSED:
            if relationship_id not in self._state.unresolved_relationships:
                self._state.unresolved_relationships.append(relationship_id)
        else:
            self._state.unresolved_relationships = [
                relationship
                for relationship in self._state.unresolved_relationships
                if relationship != relationship_id
            ]

    def update_risk(
        self,
        risk: RiskContribution,
        confidence: Confidence,
    ) -> None:
        """Update the current derived risk assessment."""
        self._state.current_risk = risk
        self._state.current_confidence = confidence

    def update_evidence_sufficiency(self, status: str) -> None:
        """Update the current evidence sufficiency status."""
        self._state.evidence_sufficiency = status

    def update_conflict_status(self, status: str) -> None:
        """Update the current conflict status."""
        self._state.conflict_status = status

    def record_investigation(self, cost: float) -> None:
        """
        Record resource usage by an investigation.

        Raises:
            ValueError: If cost is negative or exceeds the remaining budget.
        """
        if cost < 0:
            raise ValueError("Investigation cost cannot be negative.")

        if cost > self._state.budget_remaining:
            raise ValueError("Investigation cost exceeds remaining budget.")

        self._state.investigations_used += 1
        self._state.budget_remaining -= cost

    def set_decision(self, decision: DecisionOutcome) -> None:
        """Set the current derived decision."""
        self._state.current_decision = decision