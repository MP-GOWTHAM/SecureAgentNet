"""Adaptive risk engine (methodology §2.4).

Fuses injection score, behavioral anomaly, source trust, privilege
deviation, and session history into one risk value — the cross-surface
correlation point. Phase 4's `FusionEngine` (correlation/fusion.py) already
does two-signal fusion (detector risk + privilege out-of-scope) with a
threshold rule; this module generalizes that to N signals with a
*weighted-sum* combiner by default, matching the doc's "may be a weighted
sum, a small learned model, or a rule-based combiner" — weighted sum is the
simplest of those three and the one that needs no additional training data,
so it's the sensible default; `combiner` is swappable for a learned model
later without touching call sites.

Signal polarity, so weights all point the same direction ("higher = more
risk" for every term):
- injection_score: already 0-1 risk, used as-is
- behavior_anomaly: already 0-1 deviation, used as-is
- source_trust: INVERTED (1 - trust) — high trust should *lower* risk
- privilege_deviation: 1.0 if out-of-scope else 0.0
- session_history: 0-1, caller-supplied (e.g. recent flag rate for this session)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class RiskSignals:
    injection_score: float
    behavior_anomaly: float = 0.0
    source_trust: float = 1.0  # 1.0 = fully trusted, default when provenance unknown
    privilege_out_of_scope: bool = False
    session_history_risk: float = 0.0


@dataclass
class RiskWeights:
    """Default weights give `injection` the largest single share (0.50,
    not the original 0.40) for a concrete, measured reason: `FusionEngine`
    applies the SAME absolute thresholds (block_risk_threshold=0.7, etc.)
    to both the raw 2-signal detector score and this weighted blend, so
    that risk_score means the same thing regardless of which path produced
    it (see fusion.py's docstring). At 0.40 injection weight, a maximally
    confident detector score (1.0) can only ever contribute 0.40 to the
    blend — block_risk_threshold=0.7 becomes structurally unreachable
    without multiple other signals simultaneously maxing out, which in
    practice (verified against the real qualifire eval) meant almost
    nothing ever blocked: Utility hit 1.000 and ASR nearly quintupled
    (0.089 -> 0.447) purely from threshold-scale mismatch, not from any
    actual improvement. Raising injection to 0.50 keeps every signal
    contributing (this is still fusion, not "ignore everything but the
    detector") while making the shared thresholds reachable again for the
    signal this whole project is centered on.

    A first attempt raised injection to 0.50 and re-ran the real qualifire
    comparison — it barely moved anything (ASR 0.089->0.447 before,
    0.089->0.449 after). The reason: at weight 0.50, injection PLUS
    behavior (0.20) tops out at exactly 0.70 for an unscoped attack (one
    whose tool call is technically in-scope, so the combo-block branch
    that needs `out_of_scope=True` never applies) — and `block_risk_threshold`
    is a strict `>` comparison, so a blend that can only ever *reach*
    0.70, never *exceed* it, structurally can never block through that
    branch no matter how confident the detector is. This is exactly the
    "in-scope but malicious content" scenario the project's C-ASR metric
    exists to measure — silently unblockable was the worst possible
    version of this bug.

    Fixed by giving injection alone enough weight (0.72) to strictly clear
    0.70 on its own, restoring the original "very high detector confidence
    blocks regardless of privilege scope" behavior the 2-signal path
    already had. `privilege`'s own weight here is intentionally small
    (0.03) — its real power isn't this blended term, it's the separate
    `out_of_scope AND risk > block_combo_risk_threshold` AND-gate inside
    `FusionEngine._decide()`; giving it a large weight here too would
    double-count the same signal through two paths.
    """

    injection: float = 0.72
    behavior: float = 0.14
    trust: float = 0.10
    privilege: float = 0.03
    session: float = 0.01

    def normalized(self) -> "RiskWeights":
        total = self.injection + self.behavior + self.trust + self.privilege + self.session
        if total == 0:
            return self
        return RiskWeights(
            injection=self.injection / total,
            behavior=self.behavior / total,
            trust=self.trust / total,
            privilege=self.privilege / total,
            session=self.session / total,
        )


def weighted_sum_combiner(signals: RiskSignals, weights: RiskWeights) -> float:
    w = weights.normalized()
    distrust = 1.0 - signals.source_trust
    privilege_term = 1.0 if signals.privilege_out_of_scope else 0.0
    risk = (
        w.injection * signals.injection_score
        + w.behavior * signals.behavior_anomaly
        + w.trust * distrust
        + w.privilege * privilege_term
        + w.session * signals.session_history_risk
    )
    return max(0.0, min(1.0, risk))


class AdaptiveRiskEngine:
    def __init__(
        self,
        weights: RiskWeights | None = None,
        combiner: Optional[Callable[[RiskSignals, RiskWeights], float]] = None,
    ):
        self.weights = weights or RiskWeights()
        self.combiner = combiner or weighted_sum_combiner

    def compute_risk(self, signals: RiskSignals) -> float:
        return self.combiner(signals, self.weights)
