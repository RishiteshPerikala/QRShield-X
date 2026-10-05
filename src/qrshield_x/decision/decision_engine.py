"""Decision engine for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.core.enums import (
    Confidence,
    DecisionOutcome,
    RelationshipState,
    RiskContribution,
)
from qrshield_x.core.models import CurrentCaseState


@dataclass(frozen=True)
class DecisionResult:
    """Result produced by the decision engine."""

    outcome: DecisionOutcome
    confidence: Confidence
    reason: str


class DecisionEngine:
    """Derive a decision outcome from the current investigation state."""

    def decide(
        self,
        state: CurrentCaseState,
    ) -> DecisionResult:
        """Produce a deterministic decision from the current state."""

        if state.conflict_status.upper() != "NONE":
            return DecisionResult(
                outcome=DecisionOutcome.INCONCLUSIVE_UNVERIFIED,
                confidence=state.current_confidence,
                reason=(
                    "The current investigation state contains conflicting "
                    "evidence."
                ),
            )

        if any(
            relationship_state == RelationshipState.CONFLICTING
            for relationship_state in state.relationship_states.values()
        ):
            return DecisionResult(
                outcome=DecisionOutcome.INCONCLUSIVE_UNVERIFIED,
                confidence=state.current_confidence,
                reason=(
                    "One or more investigation relationships are "
                    "conflicting."
                ),
            )

        if (
            state.current_risk == RiskContribution.HIGH_RISK
            and state.current_confidence == Confidence.HIGH
        ):
            return DecisionResult(
                outcome=DecisionOutcome.MALICIOUS_HIGH_RISK,
                confidence=state.current_confidence,
                reason=(
                    "High-risk evidence is supported with high confidence."
                ),
            )

        if (
            state.current_risk == RiskContribution.SUSPICIOUS
            and state.current_confidence
            in {Confidence.MEDIUM, Confidence.HIGH}
        ):
            return DecisionResult(
                outcome=DecisionOutcome.SUSPICIOUS,
                confidence=state.current_confidence,
                reason=(
                    "Suspicious evidence is supported with sufficient "
                    "confidence."
                ),
            )

        if (
            state.current_risk == RiskContribution.BENIGN_SUPPORTING
            and state.current_confidence
            in {Confidence.MEDIUM, Confidence.HIGH}
            and not state.unresolved_relationships
        ):
            return DecisionResult(
                outcome=DecisionOutcome.SAFE_LOW_RISK,
                confidence=state.current_confidence,
                reason=(
                    "Benign-supporting evidence is sufficiently confident "
                    "and no relationships remain unresolved."
                ),
            )

        return DecisionResult(
            outcome=DecisionOutcome.INCONCLUSIVE_UNVERIFIED,
            confidence=state.current_confidence,
            reason=(
                "The available evidence is insufficient for a confident "
                "final decision."
            ),
        )