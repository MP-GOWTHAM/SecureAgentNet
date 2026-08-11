"""Prompt provenance tracker (methodology §2.1).

Tags every segment of an agent's context with where it came from, before
any semantic analysis runs, and maintains a trust score per source that
adapts from confirmed outcomes. Deliberately no ML model here — this stage
runs on every segment ahead of the (expensive) detector, so it's kept to a
lookup plus an exponential moving average update, exactly as specified.

Design notes:

- **Base trust is per source *type*** (a static config table — `user_input`,
  `tool:email`, `retrieval:web`, ...). **Adaptive trust is per source
  *identity*** (a specific sender/domain/URL) and starts at the type's base
  score, then drifts via EMA as outcomes are confirmed. This two-level
  design is what makes the worked example in the methodology doc work: two
  emails from the same `tool:email` source type end up with very different
  trust once one is confirmed malicious and the other repeatedly benign.

- **EMA update only fires on confirmed outcomes**, matching §3.3's
  "unconfirmed blocks do not feed the loop" rule — applied here too, since
  provenance trust is exactly the kind of signal that would be poisoned by
  updating on the pipeline's own unverified guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceType(str, Enum):
    USER_INPUT = "user_input"
    TOOL_CALENDAR = "tool:calendar"
    TOOL_EMAIL = "tool:email"
    TOOL_FILE = "tool:file"
    RETRIEVAL_WEB = "retrieval:web"
    TOOL_CODE = "tool:code"  # code_exec_agent's sandbox — code/data it reads or is given to run
    TOOL_SUPPORT = "tool:support"  # support_agent's ticket text — arbitrary external submitter, like email


# Base trust by source type — a static lookup, not learned. Values from the
# methodology doc's worked table for the original 5; TOOL_CODE/TOOL_SUPPORT
# added when wiring all 6 agent_env roles through the provenance tracker —
# TOOL_SUPPORT mirrors TOOL_EMAIL's 0.30 (both are arbitrary external
# submitter text), TOOL_CODE sits between TOOL_FILE and TOOL_EMAIL since
# sandboxed code is semi-structured but can embed arbitrary instructions.
DEFAULT_BASE_TRUST: dict[SourceType, float] = {
    SourceType.USER_INPUT: 0.95,
    SourceType.TOOL_CALENDAR: 0.70,
    SourceType.TOOL_EMAIL: 0.30,
    SourceType.TOOL_FILE: 0.50,
    SourceType.RETRIEVAL_WEB: 0.20,
    SourceType.TOOL_CODE: 0.40,
    SourceType.TOOL_SUPPORT: 0.30,
}


@dataclass
class ProvenanceTag:
    """Attached to a segment of agent context at ingestion time."""

    source_type: SourceType
    source_id: str  # e.g. sender address, domain, file path — the specific identity
    text: str


@dataclass
class TrustRecord:
    source_id: str
    source_type: SourceType
    trust: float
    n_updates: int = 0


class ProvenanceTracker:
    def __init__(self, base_trust: dict[SourceType, float] | None = None, decay: float = 0.8):
        self.base_trust = base_trust or dict(DEFAULT_BASE_TRUST)
        self.decay = decay
        self._records: dict[str, TrustRecord] = {}

    def _key(self, source_type: SourceType, source_id: str) -> str:
        return f"{source_type.value}:{source_id}"

    def trust_score(self, tag: ProvenanceTag) -> float:
        """Current trust for this specific source identity — adaptive
        history if any exists, else the base score for its source type.
        """
        key = self._key(tag.source_type, tag.source_id)
        record = self._records.get(key)
        if record is not None:
            return record.trust
        return self.base_trust.get(tag.source_type, 0.5)

    def confirm_outcome(self, tag: ProvenanceTag, outcome: float) -> float:
        """Update this source identity's trust from a *confirmed* outcome
        (1.0 = confirmed benign, 0.0 = confirmed malicious; intermediate
        values are allowed for partial-confidence confirmations).

        `outcome` must come from a verified label (benchmark ground truth
        or human review) — never from the pipeline's own unconfirmed
        decision, or trust becomes a feedback loop on its own guesses.
        """
        key = self._key(tag.source_type, tag.source_id)
        record = self._records.get(key)
        current = record.trust if record else self.base_trust.get(tag.source_type, 0.5)
        new_trust = current * self.decay + (1 - self.decay) * outcome
        self._records[key] = TrustRecord(
            source_id=tag.source_id,
            source_type=tag.source_type,
            trust=new_trust,
            n_updates=(record.n_updates + 1) if record else 1,
        )
        return new_trust

    def get_record(self, source_type: SourceType, source_id: str) -> TrustRecord | None:
        return self._records.get(self._key(source_type, source_id))
