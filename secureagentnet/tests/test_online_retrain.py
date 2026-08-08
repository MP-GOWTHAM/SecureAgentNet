"""Tests for the online-retraining orchestrator's versioning/rollback
logic, using a stubbed `retrain_fn` — no real fine-tuning run. Exercises
exactly what the methodology doc requires: buffering, threshold trigger,
promotion on improvement, rollback + flagging on regression.
"""

from secureagentnet.eval.online_retrain import OnlineRetrainingOrchestrator, RetrainRegistry


def test_track_b_not_triggered_below_buffer_threshold():
    orch = OnlineRetrainingOrchestrator(evasion_buffer_threshold=50)
    orch.add_evasions(["evasion"] * 10)
    assert not orch.should_trigger_track_b()


def test_track_b_triggered_at_buffer_threshold():
    orch = OnlineRetrainingOrchestrator(evasion_buffer_threshold=5)
    orch.add_evasions(["evasion"] * 5)
    assert orch.should_trigger_track_b()


def test_improved_f1_is_promoted():
    orch = OnlineRetrainingOrchestrator(
        evasion_buffer_threshold=5, regression_tolerance=0.02, initial_metrics={"f1": 0.90}
    )
    orch.add_evasions(["e"] * 5)

    def retrain_fn(batch, version):
        return {"f1": 0.93}

    outcome = orch.run_track_b(retrain_fn)
    assert outcome.promoted
    assert outcome.version == "v2"
    assert orch.registry.current_version == "v2"
    assert orch.registry.metrics_by_version["v2"]["f1"] == 0.93
    assert orch.evasion_buffer == []  # buffer cleared after promotion


def test_regression_beyond_tolerance_is_rejected_and_rolled_back():
    orch = OnlineRetrainingOrchestrator(
        evasion_buffer_threshold=5, regression_tolerance=0.02, initial_metrics={"f1": 0.90}
    )
    orch.add_evasions(["e"] * 5)

    def retrain_fn(batch, version):
        return {"f1": 0.80}  # regresses well beyond the 0.02 tolerance

    outcome = orch.run_track_b(retrain_fn)
    assert not outcome.promoted
    assert orch.registry.current_version == "v1"  # rolled back, stayed on v1
    assert "v2" not in orch.registry.versions
    assert len(orch.registry.rejected_batches) == 1
    assert orch.registry.rejected_batches[0]["version"] == "v2"


def test_regression_within_tolerance_is_still_promoted():
    orch = OnlineRetrainingOrchestrator(
        evasion_buffer_threshold=5, regression_tolerance=0.02, initial_metrics={"f1": 0.90}
    )
    orch.add_evasions(["e"] * 5)

    def retrain_fn(batch, version):
        return {"f1": 0.885}  # within 0.02 tolerance

    outcome = orch.run_track_b(retrain_fn)
    assert outcome.promoted
    assert orch.registry.current_version == "v2"


def test_versions_increment_sequentially_across_multiple_cycles():
    orch = OnlineRetrainingOrchestrator(evasion_buffer_threshold=3, initial_metrics={"f1": 0.80})

    def improving_retrain_fn(batch, version):
        return {"f1": 0.80 + 0.01 * int(version[1:])}

    orch.add_evasions(["e"] * 3)
    outcome1 = orch.run_track_b(improving_retrain_fn)
    orch.add_evasions(["e"] * 3)
    outcome2 = orch.run_track_b(improving_retrain_fn)

    assert outcome1.version == "v2"
    assert outcome2.version == "v3"
    assert orch.registry.versions == ["v1", "v2", "v3"]
    assert orch.registry.current_version == "v3"


def test_next_version_parses_numeric_suffix_not_list_length():
    """The second half of the real bug: starting a registry with a single
    entry ["v2"] must compute "v3" next, not "v2" again (which is what
    len(versions)+1 gave — silently colliding with and overwriting the
    starting version's own checkpoint).
    """
    registry = RetrainRegistry(versions=["v2"], current_version="v2")
    assert registry.next_version() == "v3"


def test_next_version_handles_a_full_chain():
    registry = RetrainRegistry(versions=["v1", "v2", "v3"], current_version="v3")
    assert registry.next_version() == "v4"


def test_starting_version_sets_both_registry_current_version_and_metrics_key():
    """The bug this guards against: constructing with starting_version="v2"
    must make the regression check actually compare against v2's real
    metrics, not silently fall back to an empty baseline (which would let
    every retrain "pass" regardless of quality).
    """
    orch = OnlineRetrainingOrchestrator(
        evasion_buffer_threshold=3, regression_tolerance=0.02,
        initial_metrics={"f1": 0.90}, starting_version="v2",
    )
    assert orch.registry.current_version == "v2"
    assert orch.registry.versions == ["v2"]
    assert orch.registry.metrics_by_version["v2"]["f1"] == 0.90


def test_regression_gate_actually_fires_when_resuming_from_a_later_version():
    """Reproduces the exact real-run bug: starting from v2 with a real
    baseline F1, a materially worse retrain must be rejected — not
    silently promoted because the baseline lookup missed.
    """
    orch = OnlineRetrainingOrchestrator(
        evasion_buffer_threshold=3, regression_tolerance=0.02,
        initial_metrics={"f1": 0.90}, starting_version="v2",
    )
    orch.add_evasions(["e"] * 3)

    def retrain_fn(batch, version):
        return {"f1": 0.50}  # would trivially "pass" against a wrongly-empty 0.0 baseline

    outcome = orch.run_track_b(retrain_fn)
    assert not outcome.promoted
    assert orch.registry.current_version == "v2"
    assert outcome.version == "v3"  # version numbering still correctly continues from v2


def test_starting_version_defaults_to_v1_for_backward_compatibility():
    orch = OnlineRetrainingOrchestrator(initial_metrics={"f1": 0.5})
    assert orch.registry.current_version == "v1"
    assert orch.registry.metrics_by_version["v1"]["f1"] == 0.5


def test_rejected_batch_does_not_prevent_future_retraining_attempts():
    orch = OnlineRetrainingOrchestrator(
        evasion_buffer_threshold=3, regression_tolerance=0.02, initial_metrics={"f1": 0.90}
    )
    orch.add_evasions(["bad"] * 3)
    orch.run_track_b(lambda batch, version: {"f1": 0.5})  # rejected
    assert orch.registry.current_version == "v1"

    orch.add_evasions(["good"] * 3)
    outcome = orch.run_track_b(lambda batch, version: {"f1": 0.95})
    assert outcome.promoted
    assert orch.registry.current_version == "v2"  # version numbering continues, not stuck
