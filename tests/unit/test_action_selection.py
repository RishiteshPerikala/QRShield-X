"""Unit tests for action selection."""

from qrshield_x.controller.action_evaluation import ActionEvaluationResult
from qrshield_x.controller.action_selection import ActionSelector
from qrshield_x.core.enums import RelationshipID, RelationshipState
from qrshield_x.investigations.framework import InvestigationAction


def make_evaluation(
    *,
    action_id: str,
    cost: float,
    usefulness: str,
    feasible: bool = True,
) -> ActionEvaluationResult:
    """Create a test action evaluation."""
    action = InvestigationAction(
        action_id=action_id,
        relationship_id=RelationshipID.R1,
        cost=cost,
        description="Test investigation.",
    )

    return ActionEvaluationResult(
        action=action,
        relationship_id=RelationshipID.R1,
        relationship_state=RelationshipState.CONFLICTING,
        sensitivity="HIGH",
        cost=cost,
        usefulness=usefulness,
        feasible=feasible,
    )


def test_selects_high_usefulness_action() -> None:
    selector = ActionSelector()

    evaluations = [
        make_evaluation(
            action_id="ACTION_LOW",
            cost=1.0,
            usefulness="LOW",
        ),
        make_evaluation(
            action_id="ACTION_HIGH",
            cost=5.0,
            usefulness="HIGH",
        ),
    ]

    result = selector.select(evaluations)

    assert result.selected_action is not None
    assert result.selected_action.action_id == "ACTION_HIGH"


def test_medium_usefulness_beats_low_usefulness() -> None:
    selector = ActionSelector()

    evaluations = [
        make_evaluation(
            action_id="ACTION_LOW",
            cost=1.0,
            usefulness="LOW",
        ),
        make_evaluation(
            action_id="ACTION_MEDIUM",
            cost=5.0,
            usefulness="MEDIUM",
        ),
    ]

    result = selector.select(evaluations)

    assert result.selected_action is not None
    assert result.selected_action.action_id == "ACTION_MEDIUM"


def test_infeasible_action_is_ignored() -> None:
    selector = ActionSelector()

    evaluations = [
        make_evaluation(
            action_id="ACTION_HIGH_INFEASIBLE",
            cost=1.0,
            usefulness="HIGH",
            feasible=False,
        ),
        make_evaluation(
            action_id="ACTION_MEDIUM",
            cost=2.0,
            usefulness="MEDIUM",
        ),
    ]

    result = selector.select(evaluations)

    assert result.selected_action is not None
    assert result.selected_action.action_id == "ACTION_MEDIUM"


def test_lower_cost_breaks_usefulness_tie() -> None:
    selector = ActionSelector()

    evaluations = [
        make_evaluation(
            action_id="ACTION_EXPENSIVE",
            cost=5.0,
            usefulness="HIGH",
        ),
        make_evaluation(
            action_id="ACTION_CHEAP",
            cost=2.0,
            usefulness="HIGH",
        ),
    ]

    result = selector.select(evaluations)

    assert result.selected_action is not None
    assert result.selected_action.action_id == "ACTION_CHEAP"


def test_action_id_breaks_complete_tie() -> None:
    selector = ActionSelector()

    evaluations = [
        make_evaluation(
            action_id="ACTION_B",
            cost=2.0,
            usefulness="HIGH",
        ),
        make_evaluation(
            action_id="ACTION_A",
            cost=2.0,
            usefulness="HIGH",
        ),
    ]

    result = selector.select(evaluations)

    assert result.selected_action is not None
    assert result.selected_action.action_id == "ACTION_A"


def test_no_feasible_action_returns_none() -> None:
    selector = ActionSelector()

    evaluations = [
        make_evaluation(
            action_id="ACTION_A",
            cost=5.0,
            usefulness="HIGH",
            feasible=False,
        ),
        make_evaluation(
            action_id="ACTION_B",
            cost=2.0,
            usefulness="MEDIUM",
            feasible=False,
        ),
    ]

    result = selector.select(evaluations)

    assert result.selected_action is None
    assert result.reason == "No feasible actions are available."


def test_empty_evaluations_returns_no_action() -> None:
    selector = ActionSelector()

    result = selector.select([])

    assert result.selected_action is None
    assert result.reason == "No evaluated actions were provided."