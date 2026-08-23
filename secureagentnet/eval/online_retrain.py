"""Online adversarial retraining (methodology §5).

Connects the red-teaming loop to the live pipeline: a confirmed true
positive triggers red-team generation; every evasion immediately updates
Track A (calibration + memory index — already the per-example behavior
inside `red_team.run_red_team_round`); once the evasion buffer reaches a
threshold, Track B fires — full fine-tuning on the accumulated evasions
combined with the original training set, reusing `detector/train.py`'s
existing training loop rather than reimplementing it.

Versioning + regression gating (§5.5): every Track B run is evaluated
against the held-out qualifire benchmark before being promoted. If F1
degrades beyond `regression_tolerance`, the new version is rejected — the
current version stays live and the evasion batch is flagged for manual
review, per the doc's explicit "prevents the loop from improving on new
attacks while silently regressing on known ones" requirement.

`retrain_fn` and `eval_fn` are injected rather than hardcoded to
`detector/train.py` calls directly, so the orchestration logic (buffering,
versioning, rollback) is testable in isolation from an actual GPU
fine-tuning run — see `tests/test_online_retrain.py` for the stubbed
version, and `run_full_track_b_cycle()` below for how it wires to the real
training/eval code when actually invoked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrainOutcome:
    promoted: bool
    version: str
    reason: str
    metrics: dict


@dataclass
class RetrainRegistry:
    versions: list[str] = field(default_factory=lambda: ["v1"])
    current_version: str = "v1"
    metrics_by_version: dict[str, dict] = field(default_factory=dict)
    rejected_batches: list[dict] = field(default_factory=list)  # flagged for manual review

    def next_version(self) -> str:
        """Next version string, derived from the highest numeric suffix
        seen across `versions` — NOT `len(versions) + 1`. Length-based
        numbering silently breaks the moment a chain doesn't start at v1
        with exactly one entry (e.g. constructing with `starting_version="v2"`
        gives `versions=["v2"]`, length 1, which length-based numbering
        would compute back to "v2" again instead of "v3" — this is the
        second half of the real bug hit in practice: it wasn't just the
        regression-gate baseline lookup that broke, the version name
        collided too, silently overwriting the very checkpoint being
        compared against). Parsing the numeric suffix is correct regardless
        of how many versions are in the registry or what the starting
        version was named.
        """
        highest = 0
        for v in self.versions:
            if v.startswith("v") and v[1:].isdigit():
                highest = max(highest, int(v[1:]))
        return f"v{highest + 1}"


class OnlineRetrainingOrchestrator:
    def __init__(
        self,
        evasion_buffer_threshold: int = 50,
        regression_tolerance: float = 0.02,
        initial_metrics: Optional[dict] = None,
        starting_version: str = "v1",
    ):
        """`starting_version` names whatever checkpoint `initial_metrics`
        actually describes (default "v1", the very first detector). This
        used to be hardcoded to the literal string "v1" regardless of what
        version you were actually starting from — fine if you always begin
        a retraining chain from v1, but wrong the moment you resume
        retraining from a later checkpoint (e.g. bootstrapping a v3->v4
        cycle): the orchestrator's own registry would record the baseline
        under "v1" while `run_track_b`'s regression check looks up
        `metrics_by_version[current_version]`, so if you also set
        `registry.current_version` to anything other than "v1" after
        construction, the lookup misses entirely and the check silently
        compares against an empty baseline (effectively disabling the
        regression gate — this is exactly the bug that meant a real Track B
        run's promotion decision wasn't actually gated on anything). Now
        `starting_version` sets both `registry.versions` and
        `registry.current_version` up front, so the metrics key and the
        current-version pointer can never disagree.
        """
        self.evasion_buffer_threshold = evasion_buffer_threshold
        self.regression_tolerance = regression_tolerance
        self.evasion_buffer: list[str] = []
        self.registry = RetrainRegistry(versions=[starting_version], current_version=starting_version)
        if initial_metrics:
            self.registry.metrics_by_version[starting_version] = initial_metrics

    def add_evasions(self, evasions: list[str]) -> None:
        """Track A already happened per-evasion inside the red-team round
        (calibration.confirm_outcome + memory_index.add) — this only
        accumulates the buffer Track B fires from.
        """
        self.evasion_buffer.extend(evasions)

    def should_trigger_track_b(self) -> bool:
        return len(self.evasion_buffer) >= self.evasion_buffer_threshold

    def run_track_b(
        self,
        retrain_fn: Callable[[list[str], str], dict],
        current_metrics_key: str = "f1",
    ) -> RetrainOutcome:
        """`retrain_fn(evasion_batch, new_version) -> {"f1": ..., ...}` —
        performs the actual fine-tune + evaluate-on-holdout and returns the
        new version's metrics. Kept generic so tests can stub this without
        a real training run; `run_full_track_b_cycle` below shows the real
        wiring to `detector/train.py`.
        """
        new_version = self.registry.next_version()
        batch = list(self.evasion_buffer)
        logger.info("Track B triggered: %d evasions, retraining as %s", len(batch), new_version)

        new_metrics = retrain_fn(batch, new_version)
        baseline_metrics = self.registry.metrics_by_version.get(self.registry.current_version, {})
        baseline_score = baseline_metrics.get(current_metrics_key, 0.0)
        new_score = new_metrics.get(current_metrics_key, 0.0)

        if new_score < baseline_score - self.regression_tolerance:
            self.registry.rejected_batches.append({"version": new_version, "batch": batch, "metrics": new_metrics})
            self.evasion_buffer = []
            logger.warning(
                "Track B REJECTED %s: %s %.4f -> %.4f (regression > tolerance %.4f); rolled back to %s, "
                "batch flagged for manual review",
                new_version, current_metrics_key, baseline_score, new_score,
                self.regression_tolerance, self.registry.current_version,
            )
            return RetrainOutcome(
                promoted=False, version=new_version,
                reason=f"{current_metrics_key} regressed {baseline_score:.4f} -> {new_score:.4f}",
                metrics=new_metrics,
            )

        self.registry.versions.append(new_version)
        self.registry.metrics_by_version[new_version] = new_metrics
        self.registry.current_version = new_version
        self.evasion_buffer = []
        logger.info(
            "Track B PROMOTED %s: %s %.4f -> %.4f",
            new_version, current_metrics_key, baseline_score, new_score,
        )
        return RetrainOutcome(
            promoted=True, version=new_version,
            reason=f"{current_metrics_key} {baseline_score:.4f} -> {new_score:.4f}, within tolerance",
            metrics=new_metrics,
        )


def run_full_track_b_cycle(
    csv_path: str,
    evasion_batch: list[str],
    new_version: str,
    base_output_dir: str,
    epochs: int = 1,
) -> dict:
    """The real Track B wiring: combine the evasion batch with the
    original training data, fine-tune via `detector.train.train()`, return
    its held-out qualifire test metrics for the orchestrator's regression
    check. Not called from the test suite (needs a real GPU run + the
    consolidated CSV on disk) — exercised manually, same as Phase 2's
    initial training run.
    """
    import tempfile
    from pathlib import Path

    import pandas as pd

    from secureagentnet.detector import data_loader as dl
    from secureagentnet.detector.train import train as run_train

    splits = dl.build_splits_from_csv(csv_path)
    evasion_df = pd.DataFrame({
        "text": evasion_batch,
        "label": [1] * len(evasion_batch),
        "category": ["red_team_evasion"] * len(evasion_batch),
        "source": ["online_retrain"] * len(evasion_batch),
    })
    combined_train = pd.concat([splits["train"], evasion_df], ignore_index=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        combined_csv = Path(tmp_dir) / "combined.csv"
        # Reconstruct the consolidate.py-style schema build_splits_from_csv expects.
        combined = combined_train.copy()
        combined["attack_type"] = combined["category"]
        combined["source_dataset"] = combined["source"].map(
            {"neuralchemy": "hf_csv", "necent": "hf_csv3", "mindgard": "hf_csv4"}
        ).fillna("online_retrain")
        combined["split"] = "train"
        test_rows = splits["test"].copy()
        test_rows["attack_type"] = test_rows["category"]
        test_rows["source_dataset"] = "hf_csv2"
        test_rows["split"] = "test"
        # encoding pinned: on Windows pandas would otherwise write this
        # handoff CSV in the ANSI codepage while build_splits_from_csv reads
        # it back as UTF-8, mangling (or failing on) non-ASCII attack text.
        pd.concat([combined, test_rows], ignore_index=True).to_csv(combined_csv, index=False, encoding="utf-8")

        output_dir = Path(base_output_dir) / new_version
        result = run_train(csv_path=str(combined_csv), epochs=epochs, output_dir=output_dir)

    return {
        "f1": result["test_metrics"]["f1"],
        "auc": result["test_metrics"]["auc"],
        "accuracy": result["test_metrics"]["accuracy"],
        "n_evasions_added": len(evasion_batch),
    }
