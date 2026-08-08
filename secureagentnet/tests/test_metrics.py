"""Tests for eval/metrics.py using hand-built PipelineResults with known
ground truth, so every metric's expected value is computed by hand in the
test itself rather than trusted from the implementation.
"""

from secureagentnet.correlation.fusion import FusionAction, FusionResult
from secureagentnet.eval.metrics import compute_metrics
from secureagentnet.privilege.policy_engine import Decision, ToolCallRequest, ViolationType
from secureagentnet.simulate.agent_env import PipelineResult


def _result(true_label: int, risk_score: float, out_of_scope: bool, action: FusionAction) -> PipelineResult:
    decision = Decision(
        allowed=not out_of_scope,
        violation_type=ViolationType.RESOURCE_DENIED if out_of_scope else ViolationType.NONE,
        reason="test",
    )
    fusion_result = FusionResult(action=action, reason="test", risk_score=risk_score, out_of_scope=out_of_scope)
    return PipelineResult(
        text="t", true_label=true_label, role="email_agent", violation_kind="test",
        request=ToolCallRequest(tool_name="x"), risk_score=risk_score,
        privilege_decision=decision, fusion_result=fusion_result,
    )


def test_asr_counts_non_blocked_attacks_only():
    results = [
        _result(true_label=1, risk_score=0.9, out_of_scope=False, action=FusionAction.BLOCK),
        _result(true_label=1, risk_score=0.1, out_of_scope=False, action=FusionAction.ALLOW),
        _result(true_label=1, risk_score=0.3, out_of_scope=False, action=FusionAction.FLAG),
        _result(true_label=1, risk_score=0.9, out_of_scope=False, action=FusionAction.BLOCK),
    ]
    actions = [r.fusion_result.action for r in results]
    m = compute_metrics(results, actions)
    # 2 of 4 attacks executed (ALLOW + FLAG both count as executed)
    assert m.asr == 0.5
    assert m.n_attacks == 4


def test_c_asr_only_counts_evaded_and_out_of_scope_attacks():
    results = [
        # evaded detector (risk<0.5) AND out of scope AND executed -> counts toward c_asr numerator+denominator
        _result(true_label=1, risk_score=0.1, out_of_scope=True, action=FusionAction.FLAG),
        # evaded detector AND out of scope but BLOCKED -> in denominator, not numerator
        _result(true_label=1, risk_score=0.1, out_of_scope=True, action=FusionAction.BLOCK),
        # high risk (not evaded) -> excluded from c_asr denominator entirely
        _result(true_label=1, risk_score=0.9, out_of_scope=True, action=FusionAction.BLOCK),
        # in scope (not out_of_scope) -> excluded from c_asr denominator entirely
        _result(true_label=1, risk_score=0.1, out_of_scope=False, action=FusionAction.ALLOW),
    ]
    actions = [r.fusion_result.action for r in results]
    m = compute_metrics(results, actions, detection_threshold=0.5)
    assert m.n_chained_eligible == 2
    assert m.c_asr == 0.5  # 1 of 2 eligible executed


def test_c_asr_is_nan_when_no_eligible_examples():
    results = [_result(true_label=1, risk_score=0.9, out_of_scope=False, action=FusionAction.BLOCK)]
    actions = [r.fusion_result.action for r in results]
    m = compute_metrics(results, actions)
    assert m.c_asr != m.c_asr  # NaN != NaN


def test_fpr_fnr_use_raw_risk_score_not_fusion_action():
    results = [
        _result(true_label=0, risk_score=0.6, out_of_scope=False, action=FusionAction.ALLOW),  # FP: risk>=0.5 but benign, action irrelevant
        _result(true_label=0, risk_score=0.1, out_of_scope=False, action=FusionAction.BLOCK),  # not FP by risk, even though blocked
        _result(true_label=1, risk_score=0.2, out_of_scope=False, action=FusionAction.ALLOW),  # FN: risk<0.5 but attack
        _result(true_label=1, risk_score=0.8, out_of_scope=False, action=FusionAction.BLOCK),  # not FN
    ]
    actions = [r.fusion_result.action for r in results]
    m = compute_metrics(results, actions, detection_threshold=0.5)
    assert m.fpr == 0.5  # 1 of 2 benign
    assert m.fnr == 0.5  # 1 of 2 attacks


def test_utility_preservation_counts_non_blocked_benign_requests():
    results = [
        _result(true_label=0, risk_score=0.1, out_of_scope=False, action=FusionAction.ALLOW),
        _result(true_label=0, risk_score=0.1, out_of_scope=False, action=FusionAction.FLAG),
        _result(true_label=0, risk_score=0.9, out_of_scope=False, action=FusionAction.BLOCK),
    ]
    actions = [r.fusion_result.action for r in results]
    m = compute_metrics(results, actions)
    assert m.utility_preservation == 2 / 3


def test_metrics_as_dict_has_expected_keys():
    results = [_result(true_label=0, risk_score=0.1, out_of_scope=False, action=FusionAction.ALLOW)]
    actions = [r.fusion_result.action for r in results]
    m = compute_metrics(results, actions)
    d = m.as_dict()
    assert set(d.keys()) == {"ASR", "C-ASR", "FPR", "FNR", "Utility", "n_attacks", "n_benign", "n_chained_eligible"}
