"""Builds a consolidated CSV that pools ALL FOUR sources and splits them
80:20 train/test, with a raised Necent row cap.

!! METHODOLOGY WARNING !!
This deliberately breaks the qualifire holdout isolation that the rest of
the project relies on. `build_splits`/`build_splits_from_csv` normally route
every qualifire row to `test` and never let one reach `train` (see
data_loader.py's module docstring and test_data_loader.py's assertion). Here
qualifire rows are pooled with the other three sources and split like any
other row, so ~80% of the benchmark ends up in training.

Consequence: ASR / C-ASR / FPR / utility measured against the resulting
`test` split are optimistic and are NOT comparable to the README's published
table, nor to any run that uses the real holdout. Use this only for the
explicitly-requested pooled-split experiment.

Nothing in `secureagentnet/` is modified: the Necent cap is overridden on an
in-memory copy of the spec, and the output is fed to the existing
`--csv` code path.

Usage:
    python scripts/build_pooled_split_dataset.py --necent-max-rows 200000 --test-fraction 0.2
"""
import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from secureagentnet.detector import data_loader as dl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_pooled_split_dataset")

# Reverse of data_loader.CSV_SOURCE_MAP: the tag each source must carry so
# build_splits_from_csv maps it back to the same `source` name. hf_csv2 is
# the one tag that routes to `test`, so it marks the 20% test slice.
SOURCE_TO_CSV_TAG = {
    "neuralchemy": "hf_csv",
    "necent": "hf_csv3",
    "mindgard": "hf_csv4",
    "qualifire_holdout": "hf_csv2",
}
TEST_TAG = "hf_csv2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--necent-max-rows", type=int, default=200_000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "pooled_dataset.csv"))
    args = parser.parse_args()

    frames = []
    for spec in dl.DATASET_SPECS:
        # Override the Necent cap on a copy; DATASET_SPECS itself is untouched.
        if spec.name == "necent":
            spec = replace(spec, max_rows=args.necent_max_rows)
        logger.info("Loading %s (%s, max_rows=%s)...", spec.name, spec.hf_id, spec.max_rows)
        df = dl.load_and_normalize(spec, seed=args.seed)
        logger.info("  -> %d rows", len(df))
        if len(df):
            frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    logger.info("pooled rows before dedup: %d | by source: %s",
                len(pooled), pooled["source"].value_counts().to_dict())

    # Dedup here as well as downstream, so the 80:20 boundary is drawn over
    # distinct texts (otherwise a duplicate could straddle train and test).
    pooled = dl._dedup(pooled)
    logger.info("pooled rows after dedup: %d", len(pooled))

    # Stratified 80:20 by label, so both sides keep the pooled attack ratio.
    test_parts, train_parts = [], []
    for label in sorted(pooled["label"].unique()):
        subset = pooled[pooled["label"] == label].sample(frac=1.0, random_state=args.seed)
        n_test = int(len(subset) * args.test_fraction)
        test_parts.append(subset.iloc[:n_test])
        train_parts.append(subset.iloc[n_test:])
    test_df = pd.concat(test_parts, ignore_index=True)
    train_df = pd.concat(train_parts, ignore_index=True)

    # 20% slice is tagged hf_csv2 so the loader routes it to `test`; the 80%
    # keeps its true source tag so it lands in train/val.
    test_out = pd.DataFrame({
        "text": test_df["text"], "label": test_df["label"],
        "attack_type": test_df["category"].fillna("unknown"),
        "source_dataset": TEST_TAG, "split": "test",
    })
    train_out = pd.DataFrame({
        "text": train_df["text"], "label": train_df["label"],
        "attack_type": train_df["category"].fillna("unknown"),
        "source_dataset": train_df["source"].map(SOURCE_TO_CSV_TAG).fillna("hf_csv"),
        "split": "train",
    })
    # Any qualifire row that landed in the 80% must not carry hf_csv2, or the
    # loader would route it back to test. Fold it in under the neuralchemy
    # tag, which is the only other "train" tag whose label semantics are the
    # same plain 0/1.
    n_qualifire_in_train = int((train_df["source"] == "qualifire_holdout").sum())
    train_out.loc[train_out["source_dataset"] == TEST_TAG, "source_dataset"] = "hf_csv"
    logger.info("qualifire rows pooled INTO train (holdout isolation broken by design): %d",
                n_qualifire_in_train)

    out = pd.concat([train_out, test_out], ignore_index=True)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False, encoding="utf-8")

    logger.info("wrote %s | total=%d train=%d (%.1f%% positive) test=%d (%.1f%% positive)",
                dest, len(out), len(train_out), 100 * train_out["label"].mean(),
                len(test_out), 100 * test_out["label"].mean())


if __name__ == "__main__":
    main()
