"""Cross-surface correlation: fuses the detector's risk score with the
privilege layer's scope-deviation signal into one Allow/Block/Flag decision.

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from secureagentnet.privilege.policy_engine import Decision


class FusionAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    FLAG = "flag"


@dataclass
class FusionConfig:
    block_risk_threshold: float = 0.7
    block_combo_risk_threshold: float = 0.4
    flag_risk_threshold: float = 0.3
    flag_on_out_of_scope: bool = True
    strict_privilege: bool = False


@dataclass
class FusionResult:
    action: FusionAction
    reason: str
    risk_score: float
    out_of_scope: bool

    @property
    def executes(self) -> bool:
        """Whether the tool call actually proceeds (ALLOW and FLAG both
        execute the call; only BLOCK prevents it — FLAG is 'allowed but
        logged for review', not 'held for approval').
        """
        return self.action != FusionAction.BLOCK


class FusionEngine:
    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()

    def fuse(self, risk_score: float, privilege_decision: "Decision") -> FusionResult:
        cfg = self.config
        out_of_scope = privilege_decision.out_of_scope

        if cfg.strict_privilege and out_of_scope:
            return FusionResult(
                action=FusionAction.BLOCK,
                reason=f"strict_privilege: {privilege_decision.violation_type.value}",
                risk_score=risk_score,
                out_of_scope=out_of_scope,
            )

        if risk_score > cfg.block_risk_threshold:
            return FusionResult(
                action=FusionAction.BLOCK,
                reason=f"risk_score {risk_score:.3f} > block_risk_threshold {cfg.block_risk_threshold}",
                risk_score=risk_score,
                out_of_scope=out_of_scope,
            )

        if out_of_scope and risk_score > cfg.block_combo_risk_threshold:
            return FusionResult(
                action=FusionAction.BLOCK,
                reason=(
                    f"risk_score {risk_score:.3f} > block_combo_risk_threshold "
                    f"{cfg.block_combo_risk_threshold} and tool call out of scope"
                ),
                risk_score=risk_score,
                out_of_scope=out_of_scope,
            )

        if risk_score > cfg.flag_risk_threshold:
            return FusionResult(
                action=FusionAction.FLAG,
                reason=f"risk_score {risk_score:.3f} > flag_risk_threshold {cfg.flag_risk_threshold}",
                risk_score=risk_score,
                out_of_scope=out_of_scope,
            )

        if out_of_scope and cfg.flag_on_out_of_scope:
            return FusionResult(
                action=FusionAction.FLAG,
                reason="tool call out of scope but detector risk is low; flagged for review",
                risk_score=risk_score,
                out_of_scope=out_of_scope,
            )

        return FusionResult(
            action=FusionAction.ALLOW,
            reason="low risk, in scope",
            risk_score=risk_score,
            out_of_scope=out_of_scope,
        )
