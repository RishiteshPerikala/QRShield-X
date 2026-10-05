"""Safety layer for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass

from qrshield_x.investigations.framework import InvestigationAction


@dataclass(frozen=True)
class SafetyCheckResult:
    """Result of a safety check for an investigation action."""

    allowed: bool
    action_id: str
    reason: str


class SafetyLayer:
    """Validate whether an investigation action may be executed."""

    def check(
        self,
        action: InvestigationAction,
        *,
        budget_remaining: float,
    ) -> SafetyCheckResult:
        """Check whether an investigation action is safe to execute."""

        if not isinstance(action, InvestigationAction):
            return SafetyCheckResult(
                allowed=False,
                action_id="",
                reason="Invalid investigation action.",
            )

        if budget_remaining < 0:
            return SafetyCheckResult(
                allowed=False,
                action_id=action.action_id,
                reason="Remaining budget cannot be negative.",
            )

        if action.cost < 0:
            return SafetyCheckResult(
                allowed=False,
                action_id=action.action_id,
                reason="Investigation action cost cannot be negative.",
            )

        if action.cost > budget_remaining:
            return SafetyCheckResult(
                allowed=False,
                action_id=action.action_id,
                reason="Investigation action exceeds remaining budget.",
            )

        return SafetyCheckResult(
            allowed=True,
            action_id=action.action_id,
            reason="Investigation action passed the safety checks.",
        )