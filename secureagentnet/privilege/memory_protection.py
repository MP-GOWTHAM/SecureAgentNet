"""Memory protection layer (methodology §2.7).

Guards persistent memory writes against injected content becoming a
standing instruction across sessions. Any content proposed for persistent
memory is checked against provenance trust and detector risk before being
committed, quarantined, or rejected — reuses the provenance tracker
(§2.1) and takes a risk score from the detector/adaptive risk engine as
inputs, rather than re-implementing its own scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from secureagentnet.provenance.tracker import ProvenanceTag


class MemoryWriteOutcome(str, Enum):
    COMMITTED = "committed"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


@dataclass
class MemoryWriteDecision:
    outcome: MemoryWriteOutcome
    reason: str


@dataclass
class MemoryWriteRequest:
    content: str
    tag: ProvenanceTag
    risk_score: float


class MemoryProtectionLayer:
    def __init__(
        self,
        reject_risk_threshold: float = 0.7,
        reject_trust_threshold: float = 0.15,
        quarantine_risk_threshold: float = 0.3,
        quarantine_trust_threshold: float = 0.5,
    ):
        """Two independent bars, either of which can escalate the outcome:
        risk (from the detector/adaptive risk engine) and trust (from
        provenance). A write is REJECTED if risk or trust alone is bad
        enough on its own; QUARANTINED (held for review, not committed) on
        a milder version of either signal; COMMITTED otherwise.
        """
        self.reject_risk_threshold = reject_risk_threshold
        self.reject_trust_threshold = reject_trust_threshold
        self.quarantine_risk_threshold = quarantine_risk_threshold
        self.quarantine_trust_threshold = quarantine_trust_threshold

    def evaluate(self, request: MemoryWriteRequest, trust_score: float) -> MemoryWriteDecision:
        if request.risk_score >= self.reject_risk_threshold:
            return MemoryWriteDecision(
                outcome=MemoryWriteOutcome.REJECTED,
                reason=f"risk_score {request.risk_score:.2f} >= reject threshold {self.reject_risk_threshold}",
            )
        if trust_score <= self.reject_trust_threshold:
            return MemoryWriteDecision(
                outcome=MemoryWriteOutcome.REJECTED,
                reason=f"source trust {trust_score:.2f} <= reject threshold {self.reject_trust_threshold}",
            )
        if request.risk_score >= self.quarantine_risk_threshold:
            return MemoryWriteDecision(
                outcome=MemoryWriteOutcome.QUARANTINED,
                reason=f"risk_score {request.risk_score:.2f} >= quarantine threshold {self.quarantine_risk_threshold}",
            )
        if trust_score <= self.quarantine_trust_threshold:
            return MemoryWriteDecision(
                outcome=MemoryWriteOutcome.QUARANTINED,
                reason=f"source trust {trust_score:.2f} <= quarantine threshold {self.quarantine_trust_threshold}",
            )
        return MemoryWriteDecision(outcome=MemoryWriteOutcome.COMMITTED, reason="low risk, trusted source")
