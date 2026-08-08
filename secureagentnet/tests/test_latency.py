import time

from secureagentnet.eval.latency import compute_overhead_pct, measure_latency


def test_measure_latency_reports_expected_call_count():
    result = measure_latency("noop", lambda: None, n_calls=10, warmup=1)
    assert result.n_calls == 10
    assert result.label == "noop"


def test_measure_latency_percentiles_are_ordered():
    def fn():
        pass

    result = measure_latency("noop", fn, n_calls=20, warmup=2)
    assert result.min_ms <= result.p50_ms <= result.p95_ms <= result.p99_ms <= result.max_ms


def test_measure_latency_captures_a_real_sleep_duration():
    result = measure_latency("sleep_5ms", lambda: time.sleep(0.005), n_calls=5, warmup=1)
    assert result.mean_ms >= 4.0  # allow scheduler slack, should be close to 5ms


def test_compute_overhead_pct_zero_when_equal():
    assert compute_overhead_pct(defended_ms=10.0, undefended_ms=10.0) == 0.0


def test_compute_overhead_pct_positive_when_defended_slower():
    pct = compute_overhead_pct(defended_ms=15.0, undefended_ms=10.0)
    assert pct == 50.0


def test_compute_overhead_pct_infinite_when_undefended_is_zero():
    assert compute_overhead_pct(defended_ms=5.0, undefended_ms=0.0) == float("inf")
