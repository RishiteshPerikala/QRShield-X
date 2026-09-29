"""Investigation candidate generation for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.controller.sensitivity import ControllerSensitivity
from qrshield_x.core.enums import RelationshipID, RelationshipState
from qrshield_x.investigations.framework import (
    InvestigationAction,
    InvestigationFramework,
)


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Candidate investigation actions for a relationship."""

    relationship_id: RelationshipID
    candidates: tuple[InvestigationAction, ...]


class CandidateGenerator:
    """Generate investigation candidates from relationship state."""

    def __init__(
        self,
        framework: InvestigationFramework,
        sensitivity: ControllerSensitivity | None = None,
    ) -> None:
        self._framework = framework
        self._sensitivity = sensitivity or ControllerSensitivity()

    def generate(
        self,
        relationship_id: RelationshipID,
        relationship_state: RelationshipState,
    ) -> CandidateGenerationResult:
        """Generate actions relevant to the current relationship state."""
        sensitivity_result = self._sensitivity.evaluate(
            relationship_id,
            relationship_state,
        )

        if sensitivity_result.sensitivity == "LOW":
            return CandidateGenerationResult(
                relationship_id=relationship_id,
                candidates=(),
            )

        actions = self._framework.get_actions_for_relationship(
            relationship_id
        )

        return CandidateGenerationResult(
            relationship_id=relationship_id,
            candidates=tuple(actions),
        )
    