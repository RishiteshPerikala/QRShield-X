import pytest
from qrshield_x.controller.action_evaluation import ActionEvaluator
from qrshield_x.controller.action_selection import ActionSelector
from qrshield_x.controller.candidate_generation import CandidateGenerator
from qrshield_x.controller.investigation_loop import InvestigationLoop
from qrshield_x.core.enums import RelationshipID, RelationshipState
from qrshield_x.core.state_manager import StateManager
from qrshield_x.investigations.framework import (
    InvestigationAction,
    InvestigationFramework,
    InvestigationResult,
)


def build_loop(
    *,
    budget: float = 10.0,
) -> tuple[InvestigationLoop, StateManager]:
    state_manager = StateManager(budget=budget)

    framework = InvestigationFramework()

    framework.register_action(
        InvestigationAction(
            action_id="A3_URL_ANALYSIS",
            relationship_id=RelationshipID.R1,
            cost=2.0,
            description="Analyze URL.",
        )
    )

    candidate_generator = CandidateGenerator(framework)
    action_evaluator = ActionEvaluator()
    action_selector = ActionSelector()

    def executor(
        action: InvestigationAction,
    ) -> InvestigationResult:
        return InvestigationResult(
            action_id=action.action_id,
            success=True,
            result={"executed": True},
            evidence_ids=["E1"],
        )

    loop = InvestigationLoop(
    case_id="CASE-001",
    state_manager=state_manager,
    candidate_generator=candidate_generator,
    action_evaluator=action_evaluator,
    action_selector=action_selector,
    executor=executor,
)

    return loop, state_manager


def test_investigation_loop_executes_selected_action() -> None:
    loop, state_manager = build_loop()

    result = loop.run_once(RelationshipID.R1)

    assert result.executed is True
    assert result.selected_action is not None
    assert result.selected_action.action_id == "A3_URL_ANALYSIS"
    assert result.investigation_result is not None
    assert result.investigation_result.success is True
    assert result.investigation_result.evidence_ids == ["E1"]

    assert result.investigation_record is not None

    record = result.investigation_record

    assert record.investigation_id == "CASE-001-INV-1"
    assert record.case_id == "CASE-001"
    assert record.action_id == "A3_URL_ANALYSIS"
    assert record.triggering_relationship == RelationshipID.R1
    assert record.selected_action == "A3_URL_ANALYSIS"
    assert record.candidate_actions == ["A3_URL_ANALYSIS"]
    assert record.new_evidence_ids == ["E1"]
    assert record.cost == 2.0
    assert record.result == {"executed": True}
  
    state = state_manager.get_state()

    assert state.investigations_used == 1
    assert state.budget_remaining == 8.0


def test_investigation_loop_does_not_execute_sufficient_relationship() -> None:
    loop, state_manager = build_loop()

    state_manager.update_relationship(
        RelationshipID.R1,
        RelationshipState.SUFFICIENT,
    )

    result = loop.run_once(RelationshipID.R1)

    assert result.executed is False
    assert result.selected_action is None
    assert result.investigation_result is None
    assert result.reason == "No investigation candidates are available."

    state = state_manager.get_state()

    assert state.investigations_used == 0
    assert state.budget_remaining == 10.0


def test_investigation_loop_respects_budget() -> None:
    loop, state_manager = build_loop(budget=1.0)

    result = loop.run_once(RelationshipID.R1)

    assert result.executed is False
    assert result.selected_action is None
    assert result.investigation_result is None
    assert result.reason == "No feasible actions are available."

    state = state_manager.get_state()

    assert state.investigations_used == 0
    assert state.budget_remaining == 1.0


def test_investigation_loop_handles_unassessed_relationship() -> None:
    loop, state_manager = build_loop()

    state = state_manager.get_state()

    assert (
        state.relationship_states[RelationshipID.R1]
        == RelationshipState.UNASSESSED
    )

    result = loop.run_once(RelationshipID.R1)

    assert result.executed is True

def test_investigation_records_are_preserved() -> None:
    loop, _ = build_loop()

    result = loop.run_once(RelationshipID.R1)

    assert result.investigation_record is not None

    records = loop.get_investigation_records()

    assert len(records) == 1
    assert records[0] == result.investigation_record


def test_investigation_loop_executor_failure_preserves_state() -> None:
    loop, state_manager = build_loop()

    def failing_executor(
        action: InvestigationAction,
    ) -> InvestigationResult:
        raise RuntimeError("Executor failed.")

    loop._executor = failing_executor

    with pytest.raises(RuntimeError, match="Executor failed"):
        loop.run_once(RelationshipID.R1)

    state = state_manager.get_state()

    assert state.investigations_used == 0
    assert state.budget_remaining == 10.0
    assert loop.get_investigation_records() == []