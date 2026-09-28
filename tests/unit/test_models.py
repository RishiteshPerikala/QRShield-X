from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from qrshield_x.core.enums import (
    Availability,
    Confidence,
    Reliability,
    RiskContribution,
)
from qrshield_x.core.models import EvidenceRecord


def test_evidence_record_accepts_valid_data() -> None:
    evidence = EvidenceRecord(
        evidence_id="E1",
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

    assert evidence.evidence_id == "E1"
    assert evidence.observation["url"] == "https://example.com"


def test_evidence_record_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="E1",
            case_id="C1",
            evidence_type="DECODED_URL",
            source_action="A1",
            observation={},
            risk_contribution=RiskContribution.NEUTRAL,
            confidence="INVALID",
            reliability=Reliability.HIGH,
            availability=Availability.AVAILABLE,
            timestamp=datetime.now(timezone.utc),
        )