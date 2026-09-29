"""Investigation action selection for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.controller.action_evaluation import ActionEvaluationResult
from qrshield_x.core.enums import RelationshipID
from qrshield_x.investigations.framework import InvestigationAction


@dataclass(frozen=True)
class ActionSelectionResult:
    """Selected investigation action."""

    relationship_id: RelationshipID
    selected_action: InvestigationAction | None
    reason: str


class ActionSelector:
    """Select the most suitable feasible investigation action."""

    _USEFULNESS_PRIORITY = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    def select(
        self,
        evaluations: list[ActionEvaluationResult],
    ) -> ActionSelectionResult:
        """Select one action from evaluated candidates."""
        if not evaluations:
            return ActionSelectionResult(
                relationship_id=RelationshipID.R1,
                selected_action=None,
                reason="No evaluated actions were provided.",
            )

        relationship_id = evaluations[0].relationship_id

        feasible = [
            evaluation
            for evaluation in evaluations
            if evaluation.feasible
        ]

        if not feasible:
            return ActionSelectionResult(
                relationship_id=relationship_id,
                selected_action=None,
                reason="No feasible actions are available.",
            )

        selected = min(
            feasible,
            key=lambda evaluation: (
                -self._USEFULNESS_PRIORITY[evaluation.usefulness],
                evaluation.cost,
                evaluation.action.action_id,
            ),
        )

        return ActionSelectionResult(
            relationship_id=selected.relationship_id,
            selected_action=selected.action,
            reason=(
                "Selected the highest-usefulness feasible action; "
                "ties were resolved by lower cost and action ID."
            ),
        )