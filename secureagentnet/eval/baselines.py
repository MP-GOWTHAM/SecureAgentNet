"""Single-signal baselines, derived from the same (risk_score,
privilege_decision) pair every `PipelineResult` already carries — so
comparing against the fused framework never requires a second detector
pass or a second privilege check, only a different decision function over
the same underlying signals.
"""

from __future__ import annotations

from secureagentnet.correlation.fusion import FusionAction
from secureagentnet.eval.metrics import DEFAULT_DETECTION_THRESHOLD
from secureagentnet.simulate.agent_env import PipelineResult


def framework_actions(results: list[PipelineResult]) -> list[FusionAction]:
    """The actual system under test: SecureAgentNet's fused decision,
    already computed per-example by `run_pipeline`/`FusionEngine`.
    """
    return [r.fusion_result.action for r in results]


def detection_only_actions(
    results: list[PipelineResult], threshold: float = DEFAULT_DETECTION_THRESHOLD
) -> list[FusionAction]:
    """Baseline: block purely on detector risk score, ignore privilege
    entirely. This is what a standalone injection classifier bolted onto
    an agent (no privilege layer at all) would do.
    """
    return [FusionAction.BLOCK if r.risk_score >= threshold else FusionAction.ALLOW for r in results]


def privilege_only_actions(results: list[PipelineResult]) -> list[FusionAction]:
    """Baseline: block purely on privilege scope, ignore detector risk
    entirely. This is what an ABAC layer with no content inspection at all
    would do — a hard, always-enforced boundary.
    """
    return [
        FusionAction.BLOCK if r.privilege_decision.out_of_scope else FusionAction.ALLOW
        for r in results
    ]
