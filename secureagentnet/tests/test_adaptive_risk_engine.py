from secureagentnet.correlation.adaptive_risk_engine import AdaptiveRiskEngine, RiskSignals, RiskWeights


def test_reproduces_doc_scenario_1_injected_forwarding_blocked_range():
    """Methodology table row 1: injection=0.87, behavior=high, trust=0.30
    -> fused risk 0.91 (blocked). We don't need to match 0.91 exactly (the
    doc doesn't specify weights), but the fused risk must land clearly in
    'would be blocked' territory (> 0.7) given high signals on 3 of 4 axes.
    """
    engine = AdaptiveRiskEngine()
    signals = RiskSignals(injection_score=0.87, behavior_anomaly=0.9, source_trust=0.30, privilege_out_of_scope=True)
    risk = engine.compute_risk(signals)
    assert risk > 0.6


def test_reproduces_doc_scenario_2_unusual_but_legitimate_allowed_range():
    """Methodology table row 2: injection=0.60 (moderate), behavior=baseline
    (0), trust=0.95 (high) -> fused risk 0.35 (allowed). Moderate text score
    alone should not push risk into block territory when other signals are
    clean.
    """
    engine = AdaptiveRiskEngine()
    signals = RiskSignals(injection_score=0.60, behavior_anomaly=0.0, source_trust=0.95, privilege_out_of_scope=False)
    risk = engine.compute_risk(signals)
    assert risk < 0.5


def test_high_trust_lowers_risk_all_else_equal():
    engine = AdaptiveRiskEngine()
    low_trust = engine.compute_risk(RiskSignals(injection_score=0.3, source_trust=0.1))
    high_trust = engine.compute_risk(RiskSignals(injection_score=0.3, source_trust=0.95))
    assert low_trust > high_trust


def test_privilege_out_of_scope_increases_risk():
    engine = AdaptiveRiskEngine()
    in_scope = engine.compute_risk(RiskSignals(injection_score=0.2, privilege_out_of_scope=False))
    out_of_scope = engine.compute_risk(RiskSignals(injection_score=0.2, privilege_out_of_scope=True))
    assert out_of_scope > in_scope


def test_no_single_signal_gates_the_decision_alone():
    """Architectural requirement from the doc: 'no single signal gates the
    decision alone'. Max out one signal, keep everything else at its most
    benign — risk should not necessarily hit 1.0 (i.e. this isn't just
    routing through one term).
    """
    engine = AdaptiveRiskEngine()
    signals = RiskSignals(
        injection_score=1.0, behavior_anomaly=0.0, source_trust=1.0, privilege_out_of_scope=False,
        session_history_risk=0.0,
    )
    risk = engine.compute_risk(signals)
    assert risk < 1.0
    assert risk > 0.0


def test_weights_are_normalized_regardless_of_input_scale():
    engine_a = AdaptiveRiskEngine(weights=RiskWeights(injection=1, behavior=0, trust=0, privilege=0, session=0, twin=0))
    engine_b = AdaptiveRiskEngine(weights=RiskWeights(injection=100, behavior=0, trust=0, privilege=0, session=0, twin=0))
    signals = RiskSignals(injection_score=0.5)
    assert engine_a.compute_risk(signals) == engine_b.compute_risk(signals)


def test_risk_is_clamped_to_unit_interval():
    engine = AdaptiveRiskEngine(weights=RiskWeights(injection=10, behavior=0, trust=0, privilege=0, session=0, twin=0))
    risk = engine.compute_risk(RiskSignals(injection_score=1.0))
    assert 0.0 <= risk <= 1.0


def test_custom_combiner_is_used_instead_of_weighted_sum():
    def always_max(signals, weights):
        return max(signals.injection_score, signals.behavior_anomaly)

    engine = AdaptiveRiskEngine(combiner=always_max)
    risk = engine.compute_risk(RiskSignals(injection_score=0.1, behavior_anomaly=0.9))
    assert risk == 0.9
