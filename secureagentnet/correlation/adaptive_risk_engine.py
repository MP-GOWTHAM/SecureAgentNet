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
    injection: float = 0.40
    behavior: float = 0.20
    trust: float = 0.20
    privilege: float = 0.15
    session: float = 0.05

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
