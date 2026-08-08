from secureagentnet.correlation.fusion import FusionAction, FusionResult
from secureagentnet.eval.baselines import detection_only_actions, framework_actions, privilege_only_actions
from secureagentnet.privilege.policy_engine import Decision, ToolCallRequest, ViolationType
from secureagentnet.simulate.agent_env import PipelineResult


def _result(risk_score: float, out_of_scope: bool, framework_action: FusionAction) -> PipelineResult:
    decision = Decision(
        allowed=not out_of_scope,
        violation_type=ViolationType.RESOURCE_DENIED if out_of_scope else ViolationType.NONE,
        reason="test",
    )
    fusion_result = FusionResult(
        action=framework_action, reason="test", risk_score=risk_score, out_of_scope=out_of_scope
    )
    return PipelineResult(
        text="t", true_label=1, role="email_agent", violation_kind="test",
        request=ToolCallRequest(tool_name="x"), risk_score=risk_score,
        privilege_decision=decision, fusion_result=fusion_result,
    )


def test_framework_actions_reads_precomputed_fusion_result():
    results = [_result(0.9, False, FusionAction.BLOCK), _result(0.1, False, FusionAction.ALLOW)]
    assert framework_actions(results) == [FusionAction.BLOCK, FusionAction.ALLOW]


def test_detection_only_ignores_privilege_signal_entirely():
    # high risk + in scope -> blocked; low risk + out of scope -> allowed (privilege ignored)
    results = [_result(0.9, out_of_scope=False, framework_action=FusionAction.BLOCK),
               _result(0.1, out_of_scope=True, framework_action=FusionAction.FLAG)]
    actions = detection_only_actions(results, threshold=0.5)
    assert actions == [FusionAction.BLOCK, FusionAction.ALLOW]


def test_privilege_only_ignores_risk_score_entirely():
    # high risk + in scope -> allowed (risk ignored); low risk + out of scope -> blocked
    results = [_result(0.99, out_of_scope=False, framework_action=FusionAction.BLOCK),
               _result(0.01, out_of_scope=True, framework_action=FusionAction.FLAG)]
    actions = privilege_only_actions(results)
    assert actions == [FusionAction.ALLOW, FusionAction.BLOCK]


def test_privilege_only_never_flags_only_allow_or_block():
    results = [_result(0.5, out_of_scope=False, framework_action=FusionAction.FLAG)]
    actions = privilege_only_actions(results)
    assert actions[0] in (FusionAction.ALLOW, FusionAction.BLOCK)
