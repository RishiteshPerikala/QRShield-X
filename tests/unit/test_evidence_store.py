from datetime import datetime, timezone

import pytest

from qrshield_x.core.enums import (
    Availability,
    Confidence,
    Reliability,
    RiskContribution,
)
from qrshield_x.core.evidence_store import EvidenceStore
from qrshield_x.core.models import EvidenceRecord


def create_evidence(evidence_id: str) -> EvidenceRecord:
    """Create a valid evidence record for testing."""
    return EvidenceRecord(
        evidence_id=evidence_id,
        case_id="C1",
        evidence_type="DECODED_URL",
        source_action="A1",
        observation={"url": "https://example.com"},
        risk_contribution=RiskContribution.NEUTRAL,
        confidence=Confidence.HIGH,
        reliability=Reliability.HIGH,
        availability=Availability.AVAILABLE,
        timestamp=datetime.now(timezone.utc),
    )


def test_append_and_retrieve_evidence() -> None:
    store = EvidenceStore()
    evidence = create_evidence("E1")

    store.append(evidence)

    assert store.get("E1") == evidence
    assert store.count() == 1


def test_store_returns_all_evidence() -> None:
    store = EvidenceStore()
    evidence_1 = create_evidence("E1")
    evidence_2 = create_evidence("E2")

    store.append(evidence_1)
    store.append(evidence_2)

    records = store.get_all()

    assert len(records) == 2
    assert evidence_1 in records
    assert evidence_2 in records


def test_duplicate_evidence_id_is_rejected() -> None:
    store = EvidenceStore()
    evidence = create_evidence("E1")

    store.append(evidence)

    with pytest.raises(ValueError, match="already exists"):
        store.append(evidence)


def test_missing_evidence_returns_none() -> None:
    store = EvidenceStore()

    assert store.get("UNKNOWN") is None