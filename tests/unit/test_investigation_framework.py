import pytest

from qrshield_x.core.enums import RelationshipID
from qrshield_x.investigations.framework import (
    InvestigationAction,
    InvestigationFramework,
)


def create_action(
    action_id: str,
    relationship_id: RelationshipID = RelationshipID.R1,
    cost: float = 1.0,
) -> InvestigationAction:
    """Create an investigation action for testing."""
    return InvestigationAction(
        action_id=action_id,
        relationship_id=relationship_id,
        cost=cost,
        description="Test investigation action",
    )


def test_action_can_be_registered() -> None:
    framework = InvestigationFramework()
    action = create_action("A1")

    framework.register_action(action)

    assert framework.get_action("A1") == action


def test_actions_can_be_filtered_by_relationship() -> None:
    framework = InvestigationFramework()

    r1_action = create_action("A1", RelationshipID.R1)
    r2_action = create_action("A2", RelationshipID.R2)

    framework.register_action(r1_action)
    framework.register_action(r2_action)

    actions = framework.get_actions_for_relationship(RelationshipID.R1)

    assert actions == [r1_action]


def test_all_actions_can_be_retrieved() -> None:
    framework = InvestigationFramework()

    action_1 = create_action("A1")
    action_2 = create_action("A2", RelationshipID.R2)

    framework.register_action(action_1)
    framework.register_action(action_2)

    assert framework.get_all_actions() == [action_1, action_2]


def test_duplicate_action_id_is_rejected() -> None:
    framework = InvestigationFramework()

    framework.register_action(create_action("A1"))

    with pytest.raises(ValueError, match="already exists"):
        framework.register_action(create_action("A1"))


def test_negative_action_cost_is_rejected() -> None:
    framework = InvestigationFramework()

    with pytest.raises(ValueError, match="cannot be negative"):
        framework.register_action(create_action("A1", cost=-1.0))


def test_missing_action_returns_none() -> None:
    framework = InvestigationFramework()

    assert framework.get_action("UNKNOWN") is None