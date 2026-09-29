"""Investigation action evaluation for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.controller.sensitivity import ControllerSensitivity
from qrshield_x.core.enums import RelationshipID, RelationshipState
from qrshield_x.investigations.framework import InvestigationAction


@dataclass(frozen=True)
class ActionEvaluationResult:
    """Evaluation result for one investigation action."""

    action: InvestigationAction
    relationship_id: RelationshipID
    relationship_state: RelationshipState
    sensitivity: str
    cost: float
    usefulness: str
    feasible: bool


class ActionEvaluator:
    """Evaluate investigation actions before action selection."""

    def __init__(
        self,
        *,
        sensitivity: ControllerSensitivity | None = None,
    ) -> None:
        self._sensitivity = sensitivity or ControllerSensitivity()

    def evaluate(
        self,
        action: InvestigationAction,
        relationship_state: RelationshipState,
        *,
        budget_remaining: float,
    ) -> ActionEvaluationResult:
        """Evaluate one investigation action."""
        sensitivity_result = self._sensitivity.evaluate(
            action.relationship_id,
            relationship_state,
        )

        feasible = action.cost <= budget_remaining

        if sensitivity_result.sensitivity == "HIGH":
            usefulness = "HIGH"
        elif sensitivity_result.sensitivity == "MEDIUM":
            usefulness = "MEDIUM"
        else:
            usefulness = "LOW"

        return ActionEvaluationResult(
            action=action,
            relationship_id=action.relationship_id,
            relationship_state=relationship_state,
            sensitivity=sensitivity_result.sensitivity,
            cost=action.cost,
            usefulness=usefulness,
            feasible=feasible,
        )