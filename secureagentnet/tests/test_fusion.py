from secureagentnet.correlation.fusion import FusionAction, FusionConfig, FusionEngine
from secureagentnet.privilege.policy_engine import Decision, ViolationType


def _decision(allowed: bool, violation_type=ViolationType.NONE) -> Decision:
    return Decision(allowed=allowed, violation_type=violation_type, reason="test")


def test_high_risk_alone_blocks_regardless_of_scope():
    engine = FusionEngine()
    result = engine.fuse(risk_score=0.9, privilege_decision=_decision(allowed=True))
    assert result.action == FusionAction.BLOCK


def test_medium_risk_with_out_of_scope_blocks():
    engine = FusionEngine()
    result = engine.fuse(
        risk_score=0.5, privilege_decision=_decision(allowed=False, violation_type=ViolationType.TOOL_NOT_PERMITTED)
    )
    assert result.action == FusionAction.BLOCK


def test_medium_risk_in_scope_does_not_block():
    """Same risk score as above, but the tool call is in scope — the combo
    rule needs both signals, not risk alone at this threshold.
    """
    engine = FusionEngine()
    result = engine.fuse(risk_score=0.5, privilege_decision=_decision(allowed=True))
    assert result.action != FusionAction.BLOCK


def test_low_risk_out_of_scope_is_flagged_not_blocked_by_default():
    """The gap C-ASR is designed to measure: a genuinely unauthorized tool
    call can still execute (as FLAG, not BLOCK) when risk looks low and
    strict_privilege is off.
    """
    engine = FusionEngine()
    result = engine.fuse(
        risk_score=0.1, privilege_decision=_decision(allowed=False, violation_type=ViolationType.RESOURCE_DENIED)
    )
    assert result.action == FusionAction.FLAG
    assert result.executes


def test_low_risk_out_of_scope_blocked_under_strict_privilege():
    engine = FusionEngine(FusionConfig(strict_privilege=True))
    result = engine.fuse(
        risk_score=0.1, privilege_decision=_decision(allowed=False, violation_type=ViolationType.RESOURCE_DENIED)
    )
    assert result.action == FusionAction.BLOCK
    assert not result.executes


def test_strict_privilege_does_not_affect_in_scope_calls():
    engine = FusionEngine(FusionConfig(strict_privilege=True))
    result = engine.fuse(risk_score=0.1, privilege_decision=_decision(allowed=True))
    assert result.action == FusionAction.ALLOW


def test_elevated_risk_below_block_threshold_is_flagged():
    engine = FusionEngine()
    result = engine.fuse(risk_score=0.35, privilege_decision=_decision(allowed=True))
    assert result.action == FusionAction.FLAG


def test_low_risk_in_scope_is_allowed():
    engine = FusionEngine()
    result = engine.fuse(risk_score=0.05, privilege_decision=_decision(allowed=True))
    assert result.action == FusionAction.ALLOW


def test_flag_on_out_of_scope_disabled_falls_through_to_allow():
    engine = FusionEngine(FusionConfig(flag_on_out_of_scope=False))
    result = engine.fuse(
        risk_score=0.05, privilege_decision=_decision(allowed=False, violation_type=ViolationType.RESOURCE_DENIED)
    )
    assert result.action == FusionAction.ALLOW


def test_fusion_result_executes_property():
    engine = FusionEngine()
    blocked = engine.fuse(risk_score=0.99, privilege_decision=_decision(allowed=True))
    allowed = engine.fuse(risk_score=0.01, privilege_decision=_decision(allowed=True))
    assert not blocked.executes
    assert allowed.executes


def test_thresholds_are_configurable():
    engine = FusionEngine(FusionConfig(block_risk_threshold=0.5, flag_risk_threshold=0.1))
    result = engine.fuse(risk_score=0.6, privilege_decision=_decision(allowed=True))
    assert result.action == FusionAction.BLOCK
