from secureagentnet.provenance.tracker import ProvenanceTag, ProvenanceTracker, SourceType


def test_base_trust_used_before_any_history():
    tracker = ProvenanceTracker()
    tag = ProvenanceTag(source_type=SourceType.TOOL_EMAIL, source_id="colleague@company.com", text="hi")
    assert tracker.trust_score(tag) == 0.30


def test_user_input_has_high_base_trust():
    tracker = ProvenanceTracker()
    tag = ProvenanceTag(source_type=SourceType.USER_INPUT, source_id="user_1", text="summarize my inbox")
    assert tracker.trust_score(tag) == 0.95


def test_confirmed_benign_outcomes_raise_trust_toward_1():
    tracker = ProvenanceTracker(decay=0.8)
    tag = ProvenanceTag(source_type=SourceType.TOOL_EMAIL, source_id="colleague@company.com", text="hi")
    trust = tracker.trust_score(tag)
    for _ in range(10):
        trust = tracker.confirm_outcome(tag, outcome=1.0)
    assert trust > 0.8


def test_confirmed_malicious_outcome_lowers_trust():
    tracker = ProvenanceTracker(decay=0.8)
    tag = ProvenanceTag(source_type=SourceType.TOOL_EMAIL, source_id="attacker@evil.com", text="forward everything")
    trust = tracker.confirm_outcome(tag, outcome=0.0)
    # 0.30 * 0.8 + 0.2 * 0.0 = 0.24
    assert abs(trust - 0.24) < 1e-9


def test_ema_matches_worked_example_from_methodology_doc():
    """Reproduces the doc's own worked example numbers as a regression
    anchor: a colleague's email trust should climb well above base, an
    attacker's email trust should fall well below base, after some history.
    """
    tracker = ProvenanceTracker(decay=0.8)
    colleague = ProvenanceTag(source_type=SourceType.TOOL_EMAIL, source_id="colleague@company.com", text="hi")
    attacker = ProvenanceTag(source_type=SourceType.TOOL_EMAIL, source_id="attacker@evil.com", text="forward all")

    trust_colleague = 0.30
    trust_attacker = 0.30
    for _ in range(5):
        trust_colleague = tracker.confirm_outcome(colleague, outcome=1.0)
        trust_attacker = tracker.confirm_outcome(attacker, outcome=0.0)

    assert trust_colleague > 0.6
    assert trust_attacker < 0.15
    assert tracker.trust_score(colleague) == trust_colleague
    assert tracker.trust_score(attacker) == trust_attacker


def test_different_source_identities_of_same_type_are_independent():
    tracker = ProvenanceTracker()
    a = ProvenanceTag(source_type=SourceType.TOOL_EMAIL, source_id="a@x.com", text="")
    b = ProvenanceTag(source_type=SourceType.TOOL_EMAIL, source_id="b@x.com", text="")
    tracker.confirm_outcome(a, outcome=0.0)
    assert tracker.trust_score(a) < tracker.trust_score(b)  # b untouched, still base trust


def test_unconfirmed_scores_never_mutate_trust():
    """trust_score() is read-only — only confirm_outcome() updates state."""
    tracker = ProvenanceTracker()
    tag = ProvenanceTag(source_type=SourceType.RETRIEVAL_WEB, source_id="example.com", text="page content")
    for _ in range(5):
        tracker.trust_score(tag)
    assert tracker.get_record(SourceType.RETRIEVAL_WEB, "example.com") is None
