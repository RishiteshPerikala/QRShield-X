"""Unit tests for action evaluation."""

from qrshield_x.controller.action_evaluation import ActionEvaluator
from qrshield_x.core.enums import RelationshipID, RelationshipState
from qrshield_x.investigations.framework import InvestigationAction


def make_action(cost: float = 2.0) -> InvestigationAction:
    """Create a test investigation action."""
    return InvestigationAction(
        action_id="A3_URL_ANALYSIS",
        relationship_id=RelationshipID.R1,
        cost=cost,
        description="Analyze decoded URL.",
    )


def test_high_sensitivity_produces_high_usefulness() -> None:
    evaluator = ActionEvaluator()

    result = evaluator.evaluate(
        make_action(),
        RelationshipState.CONFLICTING,
        budget_remaining=5.0,
    )

    assert result.sensitivity == "HIGH"
    assert result.usefulness == "HIGH"
    assert result.feasible is True


def test_medium_sensitivity_produces_medium_usefulness() -> None:
    evaluator = ActionEvaluator()

    result = evaluator.evaluate(
        make_action(),
        RelationshipState.INSUFFICIENT,
        budget_remaining=5.0,
    )

    assert result.sensitivity == "MEDIUM"
    assert result.usefulness == "MEDIUM"
    assert result.feasible is True


def test_low_sensitivity_produces_low_usefulness() -> None:
    evaluator = ActionEvaluator()

    result = evaluator.evaluate(
        make_action(),
        RelationshipState.SUFFICIENT,
        budget_remaining=5.0,
    )

    assert result.sensitivity == "LOW"
    assert result.usefulness == "LOW"
    assert result.feasible is True


def test_action_is_infeasible_when_cost_exceeds_budget() -> None:
    evaluator = ActionEvaluator()

    result = evaluator.evaluate(
        make_action(cost=10.0),
        RelationshipState.CONFLICTING,
        budget_remaining=5.0,
    )

    assert result.cost == 10.0
    assert result.feasible is False


def test_action_is_feasible_when_cost_equals_budget() -> None:
    evaluator = ActionEvaluator()

    result = evaluator.evaluate(
        make_action(cost=5.0),
        RelationshipState.CONFLICTING,
        budget_remaining=5.0,
    )

    assert result.feasible is True