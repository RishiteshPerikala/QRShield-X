from qrshield_x.core.enums import RelationshipID
from qrshield_x.investigations.framework import InvestigationAction
from qrshield_x.safety.safety_layer import SafetyLayer


def build_action(
    *,
    action_id: str = "A5_REDIRECT",
    cost: float = 2.0,
) -> InvestigationAction:
    return InvestigationAction(
        action_id=action_id,
        relationship_id=RelationshipID.R1,
        cost=cost,
        description="Test investigation action.",
    )


def test_valid_action_is_allowed():
    layer = SafetyLayer()

    result = layer.check(
        build_action(),
        budget_remaining=5.0,
    )

    assert result.allowed is True
    assert result.action_id == "A5_REDIRECT"


def test_action_exceeding_budget_is_blocked():
    layer = SafetyLayer()

    result = layer.check(
        build_action(cost=6.0),
        budget_remaining=5.0,
    )

    assert result.allowed is False
    assert "exceeds remaining budget" in result.reason


def test_negative_budget_is_blocked():
    layer = SafetyLayer()

    result = layer.check(
        build_action(),
        budget_remaining=-1.0,
    )

    assert result.allowed is False
    assert "cannot be negative" in result.reason


def test_negative_action_cost_is_blocked():
    layer = SafetyLayer()

    result = layer.check(
        build_action(cost=-1.0),
        budget_remaining=5.0,
    )

    assert result.allowed is False
    assert "cannot be negative" in result.reason


def test_invalid_action_is_blocked():
    layer = SafetyLayer()

    result = layer.check(
        "not-an-action",
        budget_remaining=5.0,
    )

    assert result.allowed is False
    assert result.action_id == ""