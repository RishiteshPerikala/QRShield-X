"""Unit tests for controller sensitivity."""

import pytest

from qrshield_x.controller.sensitivity import ControllerSensitivity
from qrshield_x.core.enums import RelationshipID, RelationshipState


@pytest.mark.parametrize(
    ("relationship_state", "expected_sensitivity"),
    [
        (RelationshipState.UNASSESSED, "HIGH"),
        (RelationshipState.UNAVAILABLE, "HIGH"),
        (RelationshipState.CONFLICTING, "HIGH"),
        (RelationshipState.INSUFFICIENT, "MEDIUM"),
        (RelationshipState.SUFFICIENT, "LOW"),
    ],
)
def test_sensitivity_mapping(
    relationship_state: RelationshipState,
    expected_sensitivity: str,
) -> None:
    controller = ControllerSensitivity()

    result = controller.evaluate(
        RelationshipID.R1,
        relationship_state,
    )

    assert result.relationship_id == RelationshipID.R1
    assert result.relationship_state == relationship_state
    assert result.sensitivity == expected_sensitivity
    