"""Controller sensitivity evaluation for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.core.enums import RelationshipID, RelationshipState


@dataclass(frozen=True)
class SensitivityResult:
    """Sensitivity result for one relationship."""

    relationship_id: RelationshipID
    relationship_state: RelationshipState
    sensitivity: str


class ControllerSensitivity:
    """Determine how sensitive a relationship is to further investigation."""

    _SENSITIVITY_MAP = {
        RelationshipState.UNASSESSED: "HIGH",
        RelationshipState.UNAVAILABLE: "HIGH",
        RelationshipState.CONFLICTING: "HIGH",
        RelationshipState.INSUFFICIENT: "MEDIUM",
        RelationshipState.SUFFICIENT: "LOW",
    }

    def evaluate(
        self,
        relationship_id: RelationshipID,
        relationship_state: RelationshipState,
    ) -> SensitivityResult:
        """Evaluate investigation sensitivity for a relationship."""
        return SensitivityResult(
            relationship_id=relationship_id,
            relationship_state=relationship_state,
            sensitivity=self._SENSITIVITY_MAP[relationship_state],
        )