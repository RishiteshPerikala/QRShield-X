from qrshield_x.core.enums import (
    Confidence,
    DecisionOutcome,
    RelationshipID,
    RelationshipState,
    RiskContribution,
)
from qrshield_x.core.models import CurrentCaseState
from qrshield_x.decision.decision_engine import DecisionEngine


def build_state(
    *,
    risk: RiskContribution,
    confidence: Confidence,
    unresolved: list[RelationshipID] | None = None,
    conflict_status: str = "NONE",
) -> CurrentCaseState:
    return CurrentCaseState(
        current_risk=risk,
        current_confidence=confidence,
        evidence_sufficiency="SUFFICIENT",
        conflict_status=conflict_status,
        relationship_states={
            relationship_id: RelationshipState.SUFFICIENT
            for relationship_id in RelationshipID
        },
        unresolved_relationships=unresolved or [],
        investigations_used=1,
        budget_remaining=9.0,
        current_decision=None,
    )


def test_high_risk_high_confidence_is_malicious():
    engine = DecisionEngine()

    state = build_state(
        risk=RiskContribution.HIGH_RISK,
        confidence=Confidence.HIGH,
    )

    result = engine.decide(state)

    assert result.outcome == DecisionOutcome.MALICIOUS_HIGH_RISK
    assert result.confidence == Confidence.HIGH


def test_suspicious_medium_confidence_is_suspicious():
    engine = DecisionEngine()

    state = build_state(
        risk=RiskContribution.SUSPICIOUS,
        confidence=Confidence.MEDIUM,
    )

    result = engine.decide(state)

    assert result.outcome == DecisionOutcome.SUSPICIOUS
    assert result.confidence == Confidence.MEDIUM


def test_benign_high_confidence_with_no_unresolved_is_safe():
    engine = DecisionEngine()

    state = build_state(
        risk=RiskContribution.BENIGN_SUPPORTING,
        confidence=Confidence.HIGH,
    )

    result = engine.decide(state)

    assert result.outcome == DecisionOutcome.SAFE_LOW_RISK
    assert result.confidence == Confidence.HIGH


def test_benign_with_unresolved_relationship_is_inconclusive():
    engine = DecisionEngine()

    state = build_state(
        risk=RiskContribution.BENIGN_SUPPORTING,
        confidence=Confidence.HIGH,
        unresolved=[RelationshipID.R3],
    )

    result = engine.decide(state)

    assert result.outcome == DecisionOutcome.INCONCLUSIVE_UNVERIFIED


def test_conflicting_state_is_inconclusive():
    engine = DecisionEngine()

    state = build_state(
        risk=RiskContribution.HIGH_RISK,
        confidence=Confidence.HIGH,
        conflict_status="CONFLICTING",
    )

    result = engine.decide(state)

    assert result.outcome == DecisionOutcome.INCONCLUSIVE_UNVERIFIED


def test_low_confidence_high_risk_is_inconclusive():
    engine = DecisionEngine()

    state = build_state(
        risk=RiskContribution.HIGH_RISK,
        confidence=Confidence.LOW,
    )

    result = engine.decide(state)

    assert result.outcome == DecisionOutcome.INCONCLUSIVE_UNVERIFIED