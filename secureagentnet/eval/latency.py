"""Detection latency and runtime overhead (methodology §6.1).

Measures per-call wall-clock latency for each pipeline stage and the full
fused pipeline, then expresses overhead relative to an "undefended"
reference — a no-op function timed the same way, representing the fixed
cost of dispatching a tool call with zero checks. This is a real, honestly
measured baseline (actual Python function-call overhead, not a fabricated
zero), not a claim about what a production agent's own LLM call would cost
— this project has no real agent LLM in the loop to compare against, so
"overhead" here means specifically what SecureAgentNet's own stages add on
top of doing nothing, not on top of an end-to-end agent turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class LatencyResult:
    label: str
    n_calls: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict:
        return {
            "label": self.label, "n_calls": self.n_calls, "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms, "p95_ms": self.p95_ms, "p99_ms": self.p99_ms,
            "min_ms": self.min_ms, "max_ms": self.max_ms,
        }


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def measure_latency(label: str, fn: Callable[[], None], n_calls: int, warmup: int = 2) -> LatencyResult:
    """Times `fn()` (a zero-arg closure — callers pre-bind whatever input
    each call needs) `n_calls` times, after `warmup` untimed calls to avoid
    counting one-time costs (e.g. lazy CUDA/MPS kernel compilation) as
    per-call latency.
    """
    for _ in range(warmup):
        fn()

    samples_ms = []
    for _ in range(n_calls):
        start = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - start) * 1000)

    samples_ms.sort()
    return LatencyResult(
        label=label,
        n_calls=n_calls,
        mean_ms=sum(samples_ms) / len(samples_ms),
        p50_ms=_percentile(samples_ms, 50),
        p95_ms=_percentile(samples_ms, 95),
        p99_ms=_percentile(samples_ms, 99),
        min_ms=samples_ms[0],
        max_ms=samples_ms[-1],
    )


def compute_overhead_pct(defended_ms: float, undefended_ms: float) -> float:
    """% latency added relative to the undefended reference. If
    undefended_ms is (near) zero, this can be a very large number — that's
    an honest consequence of comparing against "do nothing", not a bug;
    report the absolute ms figures alongside this, not this number alone.
    """
    if undefended_ms <= 0:
        return float("inf")
    return (defended_ms - undefended_ms) / undefended_ms * 100
