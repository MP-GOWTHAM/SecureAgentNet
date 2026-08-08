from secureagentnet.simulate.behavioral_anomaly import ActionSequence, BehavioralAnomalyDetector


def test_baseline_only_actions_are_not_anomalous():
    detector = BehavioralAnomalyDetector()
    seq = ActionSequence(role="email_agent", tool_names=["read_inbox", "search_email"])
    result = detector.score(seq)
    assert result.deviation_score == 0.0
    assert not result.is_anomalous
    assert result.novel_tools == []


def test_email_summarizer_forwarding_is_flagged():
    """Matches the doc's own worked example: an email-summarizer baseline
    is read-only, so a plan including forward_email deviates.
    """
    detector = BehavioralAnomalyDetector()
    seq = ActionSequence(role="email_agent", tool_names=["read_inbox", "forward_email"])
    result = detector.score(seq)
    assert result.is_anomalous
    assert "forward_email" in result.novel_tools
    assert result.deviation_score == 0.5


def test_fully_novel_sequence_scores_1():
    detector = BehavioralAnomalyDetector()
    seq = ActionSequence(role="file_agent", tool_names=["delete_file"])
    result = detector.score(seq)
    assert result.deviation_score == 1.0
    assert result.is_anomalous


def test_empty_sequence_is_not_anomalous():
    detector = BehavioralAnomalyDetector()
    seq = ActionSequence(role="email_agent", tool_names=[])
    result = detector.score(seq)
    assert result.deviation_score == 0.0
    assert not result.is_anomalous


def test_unknown_role_has_empty_baseline_so_everything_is_novel():
    detector = BehavioralAnomalyDetector()
    seq = ActionSequence(role="nonexistent_role", tool_names=["read_inbox"])
    result = detector.score(seq)
    assert result.deviation_score == 1.0


def test_custom_threshold_tolerates_some_deviation():
    detector = BehavioralAnomalyDetector(anomaly_threshold=0.5)
    seq = ActionSequence(role="email_agent", tool_names=["read_inbox", "forward_email"])
    result = detector.score(seq)
    assert result.deviation_score == 0.5
    assert not result.is_anomalous  # exactly at threshold, not above it
