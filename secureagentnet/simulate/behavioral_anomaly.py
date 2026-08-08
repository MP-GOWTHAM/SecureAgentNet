"""Behavioral anomaly detection (methodology §2.3).

Catches attacks that are linguistically unremarkable but produce anomalous
*action sequences* — a signal source independent of the text-level
detector. Per the methodology doc's scoping note, this module defines its
own baseline (typical tools a role's benign tasks call) rather than reusing
the injection datasets, and is deliberately simple: a per-role baseline
tool-set plus a deviation score, not a learned sequence model. A learned
version (e.g. an n-gram/HMM over actual session logs) would need real
session data this project doesn't have yet — that's the explicit
scope cut here, matching the doc's own caution to mark this as future work
rather than overclaim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Baseline: for each role, which tools a *typical benign task* calls.
# Reuses the same roles/tool vocabulary as privilege/policies and
# simulate/agent_env so the anomaly baseline and the ABAC scope agree on
# what "normal" means for a role.
ROLE_BASELINE_TOOLS: dict[str, set[str]] = {
    "email_agent": {"read_inbox", "search_email"},  # send_email is in-scope but not baseline-typical
    "file_agent": {"read_file", "list_files"},
    "research_agent": {"fetch_url", "web_search", "save_note"},
    "calendar_agent": {"list_events"},
    "code_exec_agent": {"run_code", "list_sandbox_files"},
    "support_agent": {"read_ticket", "reply_ticket"},
}


@dataclass
class ActionSequence:
    role: str
    tool_names: list[str] = field(default_factory=list)


@dataclass
class AnomalyResult:
    deviation_score: float  # 0-1, fraction of planned actions outside baseline
    novel_tools: list[str]  # tools called that aren't in this role's baseline
    is_anomalous: bool


class BehavioralAnomalyDetector:
    def __init__(self, baseline: dict[str, set[str]] | None = None, anomaly_threshold: float = 0.0):
        """`anomaly_threshold`: deviation_score strictly above this marks
        `is_anomalous=True`. Default 0.0 means *any* off-baseline tool call
        counts as anomalous — matching the doc's email-summarizer example,
        where a single `forward_email` call is itself the signal.
        """
        self.baseline = baseline or ROLE_BASELINE_TOOLS
        self.anomaly_threshold = anomaly_threshold

    def score(self, sequence: ActionSequence) -> AnomalyResult:
        baseline_tools = self.baseline.get(sequence.role, set())
        if not sequence.tool_names:
            return AnomalyResult(deviation_score=0.0, novel_tools=[], is_anomalous=False)

        novel = [t for t in sequence.tool_names if t not in baseline_tools]
        deviation = len(novel) / len(sequence.tool_names)
        return AnomalyResult(
            deviation_score=deviation,
            novel_tools=novel,
            is_anomalous=deviation > self.anomaly_threshold,
        )
