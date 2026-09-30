"""Investigation loop orchestration for QRShield-X."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from qrshield_x.controller.action_evaluation import ActionEvaluator
from qrshield_x.controller.action_selection import ActionSelector
from qrshield_x.controller.candidate_generation import CandidateGenerator
from qrshield_x.core.enums import RelationshipID, RelationshipState
from qrshield_x.core.models import InvestigationRecord
from qrshield_x.core.state_manager import StateManager
from qrshield_x.investigations.framework import (
    InvestigationAction,
    InvestigationResult,
)


@dataclass(frozen=True)
class InvestigationLoopResult:
    """Result of one investigation-loop iteration."""

    executed: bool
    selected_action: InvestigationAction | None
    investigation_result: InvestigationResult | None
    investigation_record: InvestigationRecord | None
    reason: str


class InvestigationLoop:
    """Coordinate bounded investigation iterations."""

    def __init__(
        self,
        *,
        case_id: str,
        state_manager: StateManager,
        candidate_generator: CandidateGenerator,
        action_evaluator: ActionEvaluator,
        action_selector: ActionSelector,
        executor: Callable[
            [InvestigationAction],
            InvestigationResult,
        ],
    ) -> None:
        if not case_id.strip():
            raise ValueError("Case ID cannot be empty.")

        self._case_id = case_id
        self._state_manager = state_manager
        self._candidate_generator = candidate_generator
        self._action_evaluator = action_evaluator
        self._action_selector = action_selector
        self._executor = executor
        self._investigation_records: list[InvestigationRecord] = []

    def run_once(
        self,
        relationship_id: RelationshipID,
    ) -> InvestigationLoopResult:
        """Run one bounded investigation iteration."""

        state_before = self._state_manager.get_state()

        relationship_state_before = state_before.relationship_states[
            relationship_id
        ]

        candidate_result = self._candidate_generator.generate(
            relationship_id,
            relationship_state_before,
        )

        if not candidate_result.candidates:
            return InvestigationLoopResult(
                executed=False,
                selected_action=None,
                investigation_result=None,
                investigation_record=None,
                reason="No investigation candidates are available.",
            )

        evaluations = [
            self._action_evaluator.evaluate(
                action,
                relationship_state_before,
                budget_remaining=state_before.budget_remaining,
            )
            for action in candidate_result.candidates
        ]

        selection = self._action_selector.select(evaluations)

        if selection.selected_action is None:
            return InvestigationLoopResult(
                executed=False,
                selected_action=None,
                investigation_result=None,
                investigation_record=None,
                reason=selection.reason,
            )

        action = selection.selected_action

        investigation_result = self._executor(action)

        self._state_manager.record_investigation(
            action.cost
        )

        state_after = self._state_manager.get_state()

        selected_evaluation = next(
            evaluation
            for evaluation in evaluations
            if evaluation.action.action_id == action.action_id
        )

        investigation_id = (
            f"{self._case_id}-INV-{len(self._investigation_records) + 1}"
        )

        record = InvestigationRecord(
            investigation_id=investigation_id,
            case_id=self._case_id,
            action_id=action.action_id,
            triggering_relationship=relationship_id,
            relationship_state_before=relationship_state_before,
            decision_sensitivity=selected_evaluation.sensitivity,
            candidate_actions=[
                candidate.action_id
                for candidate in candidate_result.candidates
            ],
            selected_action=action.action_id,
            selection_reason=selection.reason,
            cost=action.cost,
            result=investigation_result.result,
            new_evidence_ids=investigation_result.evidence_ids,
            relationship_state_after=state_after.relationship_states[
                relationship_id
            ],
            timestamp=datetime.now(timezone.utc),
        )

        self._investigation_records.append(record)

        return InvestigationLoopResult(
            executed=True,
            selected_action=action,
            investigation_result=investigation_result,
            investigation_record=record,
            reason=selection.reason,
        )

    def get_investigation_records(self) -> list[InvestigationRecord]:
        """Return a copy of recorded investigation history."""
        return list(self._investigation_records)