import pytest

from secureagentnet.correlation.adaptive_risk_engine import AdaptiveRiskEngine, RiskSignals, RiskWeights
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


# --- fuse_signals(): the 5-signal path -------------------------------------

def test_fuse_signals_max_injection_alone_can_still_clear_block_threshold():
    """Regression test for the real calibration bug found running this
    against qualifire: at the original default weights (injection=0.40),
    an in-scope attack (out_of_scope=False, so the combo-block branch
    never applies) with even a maximally confident detector score could
    never exceed block_risk_threshold=0.7 — every "in-scope but malicious
    content" attack was structurally unblockable, exactly the scenario
    C-ASR is meant to catch. injection weight must be high enough that
    injection_score=1.0 alone (everything else clean, in-scope) exceeds
    the default block_risk_threshold.
    """
    engine = FusionEngine()  # default config, default risk_engine weights
    signals = RiskSignals(
        injection_score=1.0, behavior_anomaly=0.0, source_trust=1.0, privilege_out_of_scope=False,
    )
    result = engine.fuse_signals(signals)
    assert result.action == FusionAction.BLOCK
    assert result.risk_score > engine.config.block_risk_threshold


def test_fuse_signals_blends_via_adaptive_risk_engine_by_default():
    engine = FusionEngine()
    signals = RiskSignals(injection_score=0.9, behavior_anomaly=0.0, source_trust=1.0, privilege_out_of_scope=False)
    result = engine.fuse_signals(signals)
    # blended risk should be well below the raw injection_score of 0.9,
    # since the other 4 signals are all "clean" and pull the weighted sum down
    assert result.risk_score < 0.9
    assert result.signals is signals


def test_fuse_signals_same_threshold_rules_as_fuse():
    """A blended risk score that lands above block_risk_threshold must
    trigger BLOCK via fuse_signals() exactly like an equal raw score would
    via fuse() — same decision system, not two different ones.
    """
    engine = FusionEngine(FusionConfig(block_risk_threshold=0.1))
    # injection_score alone (weight 0.4 of 1.0 normalized) still clears 0.1
    signals = RiskSignals(injection_score=1.0, behavior_anomaly=0, source_trust=1.0, privilege_out_of_scope=False)
    via_signals = engine.fuse_signals(signals)
    via_raw = engine.fuse(risk_score=via_signals.risk_score, privilege_decision=_decision(allowed=True))
    assert via_signals.action == via_raw.action == FusionAction.BLOCK


def test_fuse_signals_high_behavioral_anomaly_alone_can_trigger_flag_or_block():
    # threshold set below RiskWeights.behavior (0.14) so behavior_anomaly=1.0
    # alone (everything else clean) is enough to clear it
    engine = FusionEngine(FusionConfig(flag_risk_threshold=0.10))
    signals = RiskSignals(injection_score=0.0, behavior_anomaly=1.0, source_trust=1.0, privilege_out_of_scope=False)
    result = engine.fuse_signals(signals)
    assert result.action in (FusionAction.FLAG, FusionAction.BLOCK)


def test_fuse_signals_low_trust_alone_can_elevate_action():
    # threshold set below RiskWeights.trust (0.10) so full distrust alone
    # (everything else clean) is enough to clear it
    engine = FusionEngine(FusionConfig(flag_risk_threshold=0.08))
    trusted = engine.fuse_signals(RiskSignals(injection_score=0.0, source_trust=1.0, privilege_out_of_scope=False))
    untrusted = engine.fuse_signals(RiskSignals(injection_score=0.0, source_trust=0.0, privilege_out_of_scope=False))
    assert trusted.action == FusionAction.ALLOW
    assert untrusted.action != FusionAction.ALLOW


def test_fuse_signals_out_of_scope_still_respects_strict_privilege():
    engine = FusionEngine(FusionConfig(strict_privilege=True))
    signals = RiskSignals(injection_score=0.0, source_trust=1.0, privilege_out_of_scope=True)
    result = engine.fuse_signals(signals)
    assert result.action == FusionAction.BLOCK
    assert "strict_privilege" in result.reason


def test_fuse_signals_uses_custom_risk_engine_if_provided():
    custom = AdaptiveRiskEngine(weights=RiskWeights(injection=1, behavior=0, trust=0, privilege=0, session=0))
    engine = FusionEngine(risk_engine=custom)
    signals = RiskSignals(injection_score=0.42, behavior_anomaly=1.0, source_trust=0.0, privilege_out_of_scope=True)
    result = engine.fuse_signals(signals)
    # only injection_score should matter given the custom all-weight-on-injection engine
    assert result.risk_score == pytest.approx(0.42, abs=1e-6)


def test_fuse_and_fuse_signals_share_the_same_config():
    """Confirms fuse_signals() isn't a parallel, independently-configured
    decision path — changing config affects both entry points identically.
    """
    engine = FusionEngine(FusionConfig(block_risk_threshold=0.05))
    raw_result = engine.fuse(risk_score=0.5, privilege_decision=_decision(allowed=True))
    signal_result = engine.fuse_signals(RiskSignals(injection_score=0.5, source_trust=1.0, behavior_anomaly=0.0))
    assert raw_result.action == FusionAction.BLOCK
    # the blended score differs from 0.5 (other signals dilute it), so just
    # confirm the same low threshold makes both paths trigger easily
    assert signal_result.action in (FusionAction.BLOCK, FusionAction.FLAG)
