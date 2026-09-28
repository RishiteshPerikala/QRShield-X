"""Pydantic data models for QRShield-X."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from qrshield_x.core.enums import (
    Availability,
    Confidence,
    DecisionOutcome,
    GroundTruthLabel,
    RelationshipID,
    RelationshipState as RelationshipStateEnum,
    Reliability,
    RiskContribution,
)


class QRShieldBaseModel(BaseModel):
    """Base model for QRShield-X Pydantic models."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )


class GroundTruth(QRShieldBaseModel):
    """Independent ground-truth information for a case."""

    label: GroundTruthLabel
    source: str
    status: str
    verification_method: str | None = None
    verification_timestamp: datetime | None = None
    confidence: Confidence


class Provenance(QRShieldBaseModel):
    """Origin and acquisition information for evidence."""

    source_dataset: str | None = None
    acquisition_method: str
    collection_timestamp: datetime
    source_reference: str | None = None
    availability: Availability
    failure_reason: str | None = None


class EvidenceRecord(QRShieldBaseModel):
    """A single piece of evidence produced during an investigation."""

    evidence_id: str
    case_id: str
    evidence_type: str
    source_action: str
    observation: dict[str, Any]
    relationship_id: RelationshipID | None = None
    risk_contribution: RiskContribution
    confidence: Confidence
    reliability: Reliability
    availability: Availability
    timestamp: datetime
    provenance_id: str | None = None


class RelationshipState(QRShieldBaseModel):
    """Current state of one investigation relationship."""

    relationship_id: RelationshipID
    state: RelationshipStateEnum
    confidence: Confidence
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    last_updated: datetime


class InvestigationRecord(QRShieldBaseModel):
    """Record of one controller investigation."""

    investigation_id: str
    case_id: str
    action_id: str
    triggering_relationship: RelationshipID
    relationship_state_before: RelationshipStateEnum
    decision_sensitivity: str
    candidate_actions: list[str] = Field(default_factory=list)
    selected_action: str
    selection_reason: str
    cost: float
    result: dict[str, Any]
    new_evidence_ids: list[str] = Field(default_factory=list)
    relationship_state_after: RelationshipStateEnum
    timestamp: datetime


class CurrentCaseState(QRShieldBaseModel):
    """Derived runtime state of the current investigation."""

    current_risk: RiskContribution
    current_confidence: Confidence
    evidence_sufficiency: str
    conflict_status: str
    relationship_states: dict[RelationshipID, RelationshipStateEnum]
    unresolved_relationships: list[RelationshipID] = Field(default_factory=list)
    investigations_used: int = 0
    budget_remaining: float
    current_decision: DecisionOutcome | None = None


class Case(QRShieldBaseModel):
    """Top-level representation of a QRShield-X investigation case."""

    case_id: str
    input: dict[str, Any]
    ground_truth: GroundTruth | None = None
    current_state: CurrentCaseState | None = None
    relationship_states: dict[
        RelationshipID, RelationshipStateEnum
    ] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    investigation_ids: list[str] = Field(default_factory=list)
    provenance: Provenance | None = None