"""Unit tests for candidate generation."""

from qrshield_x.controller.candidate_generation import CandidateGenerator
from qrshield_x.core.enums import RelationshipID, RelationshipState
from qrshield_x.investigations.framework import (
    InvestigationAction,
    InvestigationFramework,
)


def make_framework() -> InvestigationFramework:
    """Create a framework with test investigation actions."""
    framework = InvestigationFramework()

    framework.register_action(
        InvestigationAction(
            action_id="A3_URL_ANALYSIS",
            relationship_id=RelationshipID.R1,
            cost=1.0,
            description="Analyze decoded URL.",
        )
    )

    framework.register_action(
        InvestigationAction(
            action_id="A4_DOMAIN_ANALYSIS",
            relationship_id=RelationshipID.R1,
            cost=2.0,
            description="Analyze destination domain.",
        )
    )

    framework.register_action(
        InvestigationAction(
            action_id="A6_IDENTITY_ANALYSIS",
            relationship_id=RelationshipID.R3,
            cost=3.0,
            description="Analyze domain identity.",
        )
    )

    return framework


def test_unassessed_relationship_generates_candidates() -> None:
    generator = CandidateGenerator(make_framework())

    result = generator.generate(
        RelationshipID.R1,
        RelationshipState.UNASSESSED,
    )

    assert len(result.candidates) == 2
    assert {
        action.action_id
        for action in result.candidates
    } == {
        "A3_URL_ANALYSIS",
        "A4_DOMAIN_ANALYSIS",
    }


def test_insufficient_relationship_generates_candidates() -> None:
    generator = CandidateGenerator(make_framework())

    result = generator.generate(
        RelationshipID.R1,
        RelationshipState.INSUFFICIENT,
    )

    assert len(result.candidates) == 2


def test_conflicting_relationship_generates_candidates() -> None:
    generator = CandidateGenerator(make_framework())

    result = generator.generate(
        RelationshipID.R1,
        RelationshipState.CONFLICTING,
    )

    assert len(result.candidates) == 2


def test_sufficient_relationship_generates_no_candidates() -> None:
    generator = CandidateGenerator(make_framework())

    result = generator.generate(
        RelationshipID.R1,
        RelationshipState.SUFFICIENT,
    )

    assert result.candidates == ()


def test_only_actions_for_requested_relationship_are_returned() -> None:
    generator = CandidateGenerator(make_framework())

    result = generator.generate(
        RelationshipID.R3,
        RelationshipState.UNASSESSED,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].action_id == "A6_IDENTITY_ANALYSIS"