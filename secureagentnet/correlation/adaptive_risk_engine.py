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
- twin_unsafe: 1.0 if the digital twin sandbox's pre-execution simulation
  flagged the outcome unsafe, else 0.0 (methodology §2.6)
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
    twin_unsafe: bool = False  # digital twin sandbox flagged the simulated outcome unsafe
    content_harm: float = 0.0
    """0-1 score from a dedicated content-harm classifier -- a separate
    axis from `injection_score`, not a refinement of it.

    The injection detector answers "is something trying to hijack this
    agent's instructions?". A directly harmful request with no override
    framing is invisible to it by construction: measured, "Explain step by
    step how to synthesize a nerve agent at home" scores 0.04 as an
    injection while a content-harm classifier scores it 0.68.

    Necent carries both labels (`prompt_adversarial` and `prompt_harmful`)
    and they are near-independent -- 385,814 of its rows are harmful but
    NOT adversarial. Defaults to 0.0 so any caller that does not compute
    it (e.g. eval/run_eval.py) is unaffected and previously-measured
    numbers still reproduce exactly.
    """


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

    Fixed by giving injection alone enough weight (0.73) to strictly clear
    0.70 on its own, restoring the original "very high detector confidence
    blocks regardless of privilege scope" behavior the 2-signal path
    already had. `privilege`'s own weight here is intentionally small
    (0.02) — its real power isn't this blended term, it's the separate
    `out_of_scope AND risk > block_combo_risk_threshold` AND-gate inside
    `FusionEngine._decide()`; giving it a large weight here too would
    double-count the same signal through two paths. `twin` (0.03) follows
    the identical reasoning once the digital twin sandbox was wired in:
    its real power is `FusionConfig.strict_twin`'s hard AND-gate, not this
    blended term, so it stays deliberately small here too.

    Adding `twin` as a 6th weight is exactly the kind of change that
    re-broke calibration once already (see above) — every existing weight
    implicitly shrinks after `normalized()` divides by a larger total, so
    `injection` was re-verified to still clear 0.70 after adding this term
    (0.73 raw, sum=1.00, no renormalization surprise) rather than just
    appending a plausible-looking new number.

    Raised again to 0.86 when `FusionConfig.block_risk_threshold` moved
    0.7->0.85 (real qualifire tradeoff: Utility 0.507->0.792, see
    fusion.py): the same "max-confidence injection alone must strictly
    clear the block threshold" invariant applies at the new threshold too
    (0.73 cleared 0.7 but not 0.85), so injection was raised just enough
    to clear it again (0.86 > 0.85) while every other weight shrank
    proportionally to keep the sum at 1.00 — same reasoning as above,
    re-applied at the new operating point rather than left stale.
    """

    injection: float = 0.86
    behavior: float = 0.06
    trust: float = 0.05
    privilege: float = 0.01
    session: float = 0.005
    twin: float = 0.015
    harm: float = 0.0
    """Content-harm weight, defaulting to ZERO on purpose.

    Two reasons. First, the weighted sum is not how a harmful prompt gets
    stopped -- `FusionConfig.block_harm_threshold` is, as a hard gate, the
    same pattern `strict_privilege` uses; a weight large enough to block on
    its own would let a moderately-harmful score dominate the injection
    signal this framework is actually about.

    Second and decisively: any non-zero default enters `normalized()`'s
    denominator and silently rescales every other weight. At 0.05 the
    injection share drops 0.86 -> 0.819, which changes the blended risk of
    every existing caller even when `content_harm` is 0.0, and would have
    invalidated the ASR / FPR / utility numbers already measured. Zero
    keeps the denominator at exactly 1.0, so adding this signal is
    bit-identical for anyone not using it. Set it explicitly to blend
    harm into the score as well as gating on it."""

    def normalized(self) -> "RiskWeights":
        total = (
            self.injection + self.behavior + self.trust
            + self.privilege + self.session + self.twin + self.harm
        )
        if total == 0:
            return self
        return RiskWeights(
            injection=self.injection / total,
            behavior=self.behavior / total,
            trust=self.trust / total,
            privilege=self.privilege / total,
            session=self.session / total,
            twin=self.twin / total,
            harm=self.harm / total,
        )


def weighted_sum_combiner(signals: RiskSignals, weights: RiskWeights) -> float:
    w = weights.normalized()
    distrust = 1.0 - signals.source_trust
    privilege_term = 1.0 if signals.privilege_out_of_scope else 0.0
    twin_term = 1.0 if signals.twin_unsafe else 0.0
    risk = (
        w.injection * signals.injection_score
        + w.behavior * signals.behavior_anomaly
        + w.trust * distrust
        + w.privilege * privilege_term
        + w.session * signals.session_history_risk
        + w.twin * twin_term
        + w.harm * signals.content_harm
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
