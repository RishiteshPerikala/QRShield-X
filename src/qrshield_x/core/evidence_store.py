"""Append-only evidence storage for QRShield-X."""

from __future__ import annotations

from qrshield_x.core.models import EvidenceRecord


class EvidenceStore:
    """In-memory append-only store for evidence records."""

    def __init__(self) -> None:
        """Initialize an empty evidence store."""
        self._records: dict[str, EvidenceRecord] = {}

    def append(self, evidence: EvidenceRecord) -> None:
        """
        Append a new evidence record.

        Raises:
            ValueError: If the evidence ID already exists.
        """
        if evidence.evidence_id in self._records:
            raise ValueError(
                f"Evidence with ID '{evidence.evidence_id}' already exists."
            )

        self._records[evidence.evidence_id] = evidence

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        """Return an evidence record by ID, or None if it does not exist."""
        return self._records.get(evidence_id)

    def get_all(self) -> list[EvidenceRecord]:
        """Return all stored evidence records."""
        return list(self._records.values())

    def count(self) -> int:
        """Return the number of stored evidence records."""
        return len(self._records)