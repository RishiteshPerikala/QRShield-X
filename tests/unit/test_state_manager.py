import pytest

from qrshield_x.core.enums import (
    Confidence,
    DecisionOutcome,
    RelationshipID,
    RelationshipState,
    RiskContribution,
)
from qrshield_x.core.state_manager import StateManager


def test_state_manager_initializes_relationships() -> None:
    manager = StateManager(budget=10.0)

    state = manager.get_state()

    assert len(state.relationship_states) == 5
    assert all(
        relationship_state == RelationshipState.UNASSESSED
        for relationship_state in state.relationship_states.values()
    )
    assert len(state.unresolved_relationships) == 5
    assert state.budget_remaining == 10.0


def test_relationship_state_can_be_updated() -> None:
    manager = StateManager(budget=10.0)

    manager.update_relationship(
        RelationshipID.R1,
        RelationshipState.SUFFICIENT,
    )

    state = manager.get_state()

    assert state.relationship_states[RelationshipID.R1] == (
        RelationshipState.SUFFICIENT
    )
    assert RelationshipID.R1 not in state.unresolved_relationships


def test_risk_can_be_updated() -> None:
    manager = StateManager(budget=10.0)

    manager.update_risk(
        RiskContribution.SUSPICIOUS,
        Confidence.HIGH,
    )

    state = manager.get_state()

    assert state.current_risk == RiskContribution.SUSPICIOUS
    assert state.current_confidence == Confidence.HIGH


def test_investigation_consumes_budget() -> None:
    manager = StateManager(budget=10.0)

    manager.record_investigation(cost=3.0)

    state = manager.get_state()

    assert state.investigations_used == 1
    assert state.budget_remaining == 7.0


def test_investigation_cannot_exceed_budget() -> None:
    manager = StateManager(budget=5.0)

    with pytest.raises(ValueError, match="exceeds remaining budget"):
        manager.record_investigation(cost=6.0)


def test_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        StateManager(budget=-1.0)


def test_decision_can_be_set() -> None:
    manager = StateManager(budget=10.0)

    manager.set_decision(DecisionOutcome.SUSPICIOUS)

    state = manager.get_state()

    assert state.current_decision == DecisionOutcome.SUSPICIOUS


def test_get_state_returns_independent_copy() -> None:
    manager = StateManager(budget=10.0)

    state = manager.get_state()
    state.budget_remaining = 0.0

    assert manager.get_state().budget_remaining == 10.0