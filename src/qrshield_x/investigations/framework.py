"""Investigation framework for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass, field

from qrshield_x.core.enums import RelationshipID


@dataclass(frozen=True)
class InvestigationAction:
    """Description of an investigation action."""

    action_id: str
    relationship_id: RelationshipID
    cost: float
    description: str = ""


@dataclass
class InvestigationResult:
    """Result produced by an investigation action."""

    action_id: str
    success: bool
    result: dict[str, object] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)


class InvestigationFramework:
    """Coordinate candidate investigation actions and their results."""

    def __init__(self) -> None:
        """Initialize an empty investigation framework."""
        self._actions: dict[str, InvestigationAction] = {}

    def register_action(self, action: InvestigationAction) -> None:
        """
        Register an investigation action.

        Raises:
            ValueError: If the action ID already exists or cost is negative.
        """
        if action.action_id in self._actions:
            raise ValueError(
                f"Investigation action '{action.action_id}' already exists."
            )

        if action.cost < 0:
            raise ValueError("Investigation action cost cannot be negative.")

        self._actions[action.action_id] = action

    def get_action(self, action_id: str) -> InvestigationAction | None:
        """Return a registered action by ID."""
        return self._actions.get(action_id)

    def get_actions_for_relationship(
        self,
        relationship_id: RelationshipID,
    ) -> list[InvestigationAction]:
        """Return actions associated with a relationship."""
        return [
            action
            for action in self._actions.values()
            if action.relationship_id == relationship_id
        ]

    def get_all_actions(self) -> list[InvestigationAction]:
        """Return all registered investigation actions."""
        return list(self._actions.values())