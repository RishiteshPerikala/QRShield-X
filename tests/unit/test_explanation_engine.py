from qrshield_x.core.enums import (
    Confidence,
    DecisionOutcome,
    RelationshipID,
    RelationshipState,
    RiskContribution,
)
from qrshield_x.core.models import CurrentCaseState
from qrshield_x.decision.decision_engine import DecisionResult
from qrshield_x.decision.explanation_engine import ExplanationEngine


def build_state() -> CurrentCaseState:
    return CurrentCaseState(
        current_risk=RiskContribution.SUSPICIOUS,
        current_confidence=Confidence.MEDIUM,
        evidence_sufficiency="SUFFICIENT",
        conflict_status="NONE",
        relationship_states={
            RelationshipID.R1: RelationshipState.SUFFICIENT,
            RelationshipID.R2: RelationshipState.SUFFICIENT,
            RelationshipID.R3: RelationshipState.UNASSESSED,
            RelationshipID.R4: RelationshipState.UNASSESSED,
            RelationshipID.R5: RelationshipState.UNASSESSED,
        },
        unresolved_relationships=[
            RelationshipID.R3,
            RelationshipID.R4,
            RelationshipID.R5,
        ],
        investigations_used=2,
        budget_remaining=8.0,
        current_decision=DecisionOutcome.SUSPICIOUS,
    )


def test_explanation_contains_decision_and_confidence():
    engine = ExplanationEngine()

    decision = DecisionResult(
        outcome=DecisionOutcome.SUSPICIOUS,
        confidence=Confidence.MEDIUM,
        reason="Suspicious evidence is supported with sufficient confidence.",
    )

    result = engine.explain(
        build_state(),
        decision,
    )

    assert result.decision == "SUSPICIOUS"
    assert result.confidence == "MEDIUM"


def test_explanation_contains_evidence_summary():
    engine = ExplanationEngine()

    decision = DecisionResult(
        outcome=DecisionOutcome.SUSPICIOUS,
        confidence=Confidence.MEDIUM,
        reason="Suspicious evidence is supported with sufficient confidence.",
    )

    result = engine.explain(
        build_state(),
        decision,
    )

    assert "SUSPICIOUS" in result.evidence_summary
    assert "SUFFICIENT" in result.evidence_summary
    assert "NONE" in result.evidence_summary


def test_explanation_contains_investigation_summary():
    engine = ExplanationEngine()

    decision = DecisionResult(
        outcome=DecisionOutcome.SUSPICIOUS,
        confidence=Confidence.MEDIUM,
        reason="Suspicious evidence is supported with sufficient confidence.",
    )

    result = engine.explain(
        build_state(),
        decision,
    )

    assert "2 relationship(s) assessed" in result.investigation_summary
    assert "3 relationship(s) unresolved" in result.investigation_summary
    assert "2 investigation(s) used" in result.investigation_summary


def test_explanation_preserves_decision_reason():
    engine = ExplanationEngine()

    reason = "The available evidence is insufficient for a confident final decision."

    decision = DecisionResult(
        outcome=DecisionOutcome.INCONCLUSIVE_UNVERIFIED,
        confidence=Confidence.LOW,
        reason=reason,
    )

    result = engine.explain(
        build_state(),
        decision,
    )

    assert result.summary == reason