"""Core enumerations for QRShield-X."""

from enum import Enum


class RelationshipID(str, Enum):
    """Identifiers for defined relationships in the investigation framework."""

    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"


class RelationshipState(str, Enum):
    """Assessment state of a relationship."""

    UNASSESSED = "UNASSESSED"
    SUFFICIENT = "SUFFICIENT"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT = "INSUFFICIENT"
    UNAVAILABLE = "UNAVAILABLE"


class Availability(str, Enum):
    """Availability state of an evidence source or analysis."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class Confidence(str, Enum):
    """Confidence level associated with an assessment."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Reliability(str, Enum):
    """Reliability level associated with an evidence source."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskContribution(str, Enum):
    """Risk contribution provided by individual evidence."""

    BENIGN_SUPPORTING = "BENIGN_SUPPORTING"
    NEUTRAL = "NEUTRAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    UNKNOWN = "UNKNOWN"


class GroundTruthLabel(str, Enum):
    """Ground-truth label used for dataset and evaluation purposes."""

    BENIGN = "BENIGN"
    MALICIOUS = "MALICIOUS"
    UNKNOWN = "UNKNOWN"


class DecisionOutcome(str, Enum):
    """Final QRShield-X investigation outcome."""

    SAFE_LOW_RISK = "SAFE / LOW RISK"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS_HIGH_RISK = "MALICIOUS / HIGH RISK"
    INCONCLUSIVE_UNVERIFIED = "INCONCLUSIVE / UNVERIFIED"


class EvidenceType(str, Enum):
    """Evidence types produced by QRShield-X analyzers."""

    # A1 — QR Decode
    DECODED_URL = "DECODED_URL"
    DECODER_AGREEMENT = "DECODER_AGREEMENT"
    DECODE_FAILURE = "DECODE_FAILURE"

    # A2 — QR Structure
    QR_STRUCTURE = "QR_STRUCTURE"
    QR_DISTORTION = "QR_DISTORTION"
    QR_IMAGE_QUALITY = "QR_IMAGE_QUALITY"

    # A3 — URL Analysis
    URL_LEXICAL_FEATURES = "URL_LEXICAL_FEATURES"
    URL_OBFUSCATION = "URL_OBFUSCATION"
    URL_MODEL_OUTPUT = "URL_MODEL_OUTPUT"

    # A4 — Domain Analysis
    DOMAIN_CHARACTERISTICS = "DOMAIN_CHARACTERISTICS"
    DOMAIN_CONTEXT = "DOMAIN_CONTEXT"

    # A5 — Redirect Analysis
    REDIRECT_CHAIN = "REDIRECT_CHAIN"
    FINAL_DESTINATION = "FINAL_DESTINATION"
    REDIRECT_FAILURE = "REDIRECT_FAILURE"

    # A6 — Identity Analysis
    DOMAIN_IDENTITY_MATCH = "DOMAIN_IDENTITY_MATCH"
    DOMAIN_IDENTITY_MISMATCH = "DOMAIN_IDENTITY_MISMATCH"

    # A7.1 — HTML / DOM Analysis
    HTML_DOM_STRUCTURE = "HTML_DOM_STRUCTURE"

    # A7.2 — Form Analysis
    FORM_ANALYSIS = "FORM_ANALYSIS"
    CREDENTIAL_COLLECTION_INDICATOR = "CREDENTIAL_COLLECTION_INDICATOR"

    # A7.3 — Script / Link Analysis
    SCRIPT_ANALYSIS = "SCRIPT_ANALYSIS"
    LINK_ANALYSIS = "LINK_ANALYSIS"

    # A7.4 — Visual Brand Analysis
    VISUAL_BRAND_EVIDENCE = "VISUAL_BRAND_EVIDENCE"

    # A8 — Behavioral Analysis
    RUNTIME_NETWORK_ACTIVITY = "RUNTIME_NETWORK_ACTIVITY"
    BEHAVIORAL_ACTIVITY = "BEHAVIORAL_ACTIVITY"