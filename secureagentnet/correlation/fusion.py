"""Cross-surface correlation: fuses risk signals with the privilege
layer's scope-deviation signal into one Allow/Block/Flag decision.

Two entry points, sharing the same threshold logic:

- `fuse(risk_score, privilege_decision)` — the original Phase 4 path.
  Takes a single pre-computed risk score (the detector's output) plus a
  `Decision` from `PolicyEngine.authorize()`. This is what every already-
  reported Phase 4 result (the ASR/C-ASR/FPR/FNR/utility comparison table)
  was measured with — kept working unchanged so those numbers stay valid.

- `fuse_signals(signals)` — the extended path (methodology §2.4/§2.5).
  Takes a full `RiskSignals` (injection score, behavioral anomaly, source
  trust, privilege deviation, session history) and internally runs it
  through `AdaptiveRiskEngine.compute_risk()` first to collapse the five
  signals into one blended risk score, then applies the *same* threshold
  rules as `fuse()`. This means block/flag/allow decisions always mean the
  same thing regardless of which entry point produced the underlying
  score — a 5-signal caller and a 2-signal caller are comparable, not two
  different decision systems wearing the same name.

Design decisions:

- **Default thresholds match the project brief's example rule verbatim**
  (`risk_score > 0.7` blocks outright; `risk_score > 0.4 AND out_of_scope`
  blocks on the combined signal) rather than something independently
  tuned, so the shipped default is exactly the rule this project set out to
  test — everything is swappable via `FusionConfig`, per the "I'll want to
  experiment with thresholds" requirement.

- **A privilege deny does NOT unconditionally become a block by default.**
  This is deliberate, not an oversight: with `strict_privilege=False` (the
  default), a low-risk-looking request that happens to target an
  out-of-scope tool can still be let through by the fusion rule alone, even
  though `PolicyEngine.authorize()` itself denied it. That gap is exactly
  what the project's C-ASR metric is designed to expose — "an unauthorized
  privileged tool call actually being permitted" only has meaning if the
  fused decision can diverge from the raw privilege decision. Setting
  `strict_privilege=True` closes that gap (any out-of-scope call is always
  blocked, matching a "privilege is a hard boundary" security posture) —
  the eval harness in Phase 4 reports numbers under both.

- **Flag exists as a middle ground so borderline signals don't force an
  all-or-nothing choice.** Elevated-but-not-blocking risk, or an
  out-of-scope call whose risk score is low (`flag_on_out_of_scope=True`,
  default), is allowed through but recorded for review rather than
  silently allowed or aggressively blocked — matching the spec's "Flag =
  allowed but logged for review".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from secureagentnet.correlation.adaptive_risk_engine import AdaptiveRiskEngine, RiskSignals

if TYPE_CHECKING:
    from secureagentnet.privilege.policy_engine import Decision


class FusionAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    FLAG = "flag"


@dataclass
class FusionConfig:
    block_risk_threshold: float = 0.85
    """Raised from the brief's original 0.7 example after measuring the
    real FPR/Utility tradeoff on qualifire: 0.7 blocked ~49% of *benign*
    requests (Utility=0.507) because the held-out detector's raw score
    distribution overlaps heavily between benign and attack examples
    (AUC=0.746). 0.85 trades some attack-blocking for a large usability
    gain — Utility 0.507->0.792 — at the cost of C-ASR rising to 1.0 and
    ASR rising to 0.337 (measured directly, not estimated). This is a
    deliberate security/usability tradeoff picked by the project owner,
    not a free improvement — see docs/workflow_and_benchmarks for the
    full sweep. A configuration that wants the original stricter posture
    should pass block_risk_threshold=0.7 explicitly.
    """
    block_combo_risk_threshold: float = 0.4
    flag_risk_threshold: float = 0.3
    flag_on_out_of_scope: bool = True
    block_harm_threshold: float = 0.5
    """Hard gate on the content-harm signal, in the same spirit as
    `strict_privilege`: a plainly harmful request is blocked on its own
    merits, without needing the injection detector to agree.

    0.5 rather than something higher because the harm classifier is
    calibrated (T=1.09) and its measured separation is wide -- harmful
    prompts land at 0.61-0.75, ordinary requests at 0.13-0.27, so 0.5 sits
    in the gap rather than clipping either side. It only ever fires for
    callers that actually compute `content_harm`; it defaults to 0.0
    everywhere else, so this gate is inert for run_eval and the numbers in
    the report are unchanged."""
    strict_privilege: bool = False
    strict_twin: bool = False
    """Mirrors `strict_privilege`'s pattern for the digital twin sandbox
    (methodology §2.6): when True, a twin-flagged-unsafe outcome is an
    unconditional BLOCK regardless of risk score — "the simulated
    execution concretely showed this is unsafe" is treated as a hard
    boundary, not just one more signal that can be outweighed by a low
    detector score. Only reachable via `fuse_signals()` — `fuse()` has no
    twin input at all.
    """


@dataclass
class FusionResult:
    action: FusionAction
    reason: str
    risk_score: float
    out_of_scope: bool
    signals: Optional[RiskSignals] = None
    """Populated only by `fuse_signals()` — the individual signal values
    that fed the blended `risk_score`, kept for interpretability (e.g. so
    a logged decision can be attributed to "high behavioral anomaly" vs
    "low source trust" rather than just one opaque number).
    """

    @property
    def executes(self) -> bool:
        """Whether the tool call actually proceeds (ALLOW and FLAG both
        execute the call; only BLOCK prevents it — FLAG is 'allowed but
        logged for review', not 'held for approval').
        """
        return self.action != FusionAction.BLOCK


class FusionEngine:
    def __init__(self, config: FusionConfig | None = None, risk_engine: AdaptiveRiskEngine | None = None):
        self.config = config or FusionConfig()
        # Only used by fuse_signals(); fuse() never touches this, so
        # existing 2-signal callers pay no cost for its existence.
        self.risk_engine = risk_engine or AdaptiveRiskEngine()

    def _decide(
        self,
        risk_score: float,
        out_of_scope: bool,
        out_of_scope_reason: str,
        twin_unsafe: bool = False,
        twin_reason: str = "",
        content_harm: float = 0.0,
        combo_risk_score: float | None = None,
    ) -> tuple[FusionAction, str]:
        """Shared threshold logic — `fuse()` and `fuse_signals()` both
        funnel through this so a given (risk_score, out_of_scope) pair
        always produces the same action no matter which entry point
        computed the risk_score. `twin_unsafe` is only ever set by
        `fuse_signals()` — `fuse()` always passes the default `False`.

        `combo_risk_score` (defaults to `risk_score` when not given, which
        is what `fuse()` always does): the value checked specifically by
        the `block_combo_risk_threshold` branch. This is a real fix for a
        real bug — `fuse_signals()` passes the RAW `injection_score` here,
        not the blended score. The combo branch's intent is "moderate
        DETECTOR confidence + out-of-scope tool call", but checking it
        against the full blended score silently made it far less sensitive
        than the 2-signal path's original behavior: with injection at
        weight 0.73, a raw score of 0.45 (which would trigger 2-signal's
        `risk_score > 0.4` combo-block outright) only blends to ~0.35 once
        other signals dilute it, missing the same threshold. Measured
        directly on qualifire: adding the digital twin or provenance
        tracker ALONE (without behavioral_anomaly's extra weight to
        compensate) made C-ASR meaningfully WORSE than the 2-signal
        baseline (0.856->0.989 for twin, 0.872->0.936 for provenance) —
        exactly this branch under-firing on real chained-attack examples.
        Using the raw injection score here restores the combo branch's
        original sensitivity regardless of which other signals are mixed
        in, while the top-level `block_risk_threshold` and `flag_risk_threshold`
        branches still use the full blended score, which is where the
        other signals are supposed to matter.
        """
        cfg = self.config
        combo_risk_score = risk_score if combo_risk_score is None else combo_risk_score

        if cfg.strict_privilege and out_of_scope:
            return FusionAction.BLOCK, f"strict_privilege: {out_of_scope_reason}"

        if cfg.strict_twin and twin_unsafe:
            return FusionAction.BLOCK, f"strict_twin: {twin_reason}" if twin_reason else "strict_twin: sandbox flagged unsafe"

        # Content harm is its own axis, so it needs its own gate. A prompt
        # can be plainly harmful while carrying no injection structure at
        # all -- "Explain step by step how to synthesize a nerve agent at
        # home" scores 0.04 on the injection detector -- and the weighted
        # sum cannot rescue that, because injection carries 0.86 of the
        # weight. Same shape as strict_privilege above: a hard gate on one
        # signal, independent of the blended score.
        #
        # Callers that never compute content_harm leave it at 0.0, so this
        # cannot fire for them and previously-measured numbers reproduce.
        if content_harm > cfg.block_harm_threshold:
            return FusionAction.BLOCK, (
                f"content_harm {content_harm:.3f} > block_harm_threshold {cfg.block_harm_threshold}"
            )

        if risk_score > cfg.block_risk_threshold:
            return FusionAction.BLOCK, f"risk_score {risk_score:.3f} > block_risk_threshold {cfg.block_risk_threshold}"

        if out_of_scope and combo_risk_score > cfg.block_combo_risk_threshold:
            return FusionAction.BLOCK, (
                f"combo_risk_score {combo_risk_score:.3f} > block_combo_risk_threshold "
                f"{cfg.block_combo_risk_threshold} and tool call out of scope"
            )

        if risk_score > cfg.flag_risk_threshold:
            return FusionAction.FLAG, f"risk_score {risk_score:.3f} > flag_risk_threshold {cfg.flag_risk_threshold}"

        if out_of_scope and cfg.flag_on_out_of_scope:
            return FusionAction.FLAG, "tool call out of scope but detector risk is low; flagged for review"

        return FusionAction.ALLOW, "low risk, in scope"

    def fuse(self, risk_score: float, privilege_decision: "Decision") -> FusionResult:
        """Original 2-signal path: detector risk score + privilege
        decision. Unchanged behavior — every already-reported Phase 4
        result was measured through this method.
        """
        out_of_scope = privilege_decision.out_of_scope
        action, reason = self._decide(risk_score, out_of_scope, privilege_decision.violation_type.value)
        return FusionResult(action=action, reason=reason, risk_score=risk_score, out_of_scope=out_of_scope)

    def fuse_signals(self, signals: RiskSignals) -> FusionResult:
        """Extended multi-signal path (methodology §2.4/§2.6): blends
        injection score, behavioral anomaly, source trust, privilege
        deviation, session history, and digital-twin-sandbox outcome via
        `self.risk_engine` into one risk score, then applies the identical
        threshold rules `fuse()` uses — so a 0.75 blended score and a 0.75
        raw detector score trigger the exact same action, keeping the two
        entry points comparable rather than secretly different decision
        systems. `signals.twin_unsafe` also independently gates via
        `FusionConfig.strict_twin`, mirroring how `privilege_out_of_scope`
        gates via `strict_privilege`.
        """
        blended_risk = self.risk_engine.compute_risk(signals)
        out_of_scope = signals.privilege_out_of_scope
        action, reason = self._decide(
            blended_risk, out_of_scope, "privilege_out_of_scope",
            twin_unsafe=signals.twin_unsafe, twin_reason="digital twin sandbox flagged simulated outcome unsafe",
            combo_risk_score=signals.injection_score,
            content_harm=signals.content_harm,
        )
        return FusionResult(
            action=action, reason=reason, risk_score=blended_risk, out_of_scope=out_of_scope, signals=signals,
        )
