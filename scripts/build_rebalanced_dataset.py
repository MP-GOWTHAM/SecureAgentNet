"""Writes a persona-rebalanced copy of the consolidated dataset.

`train_ensemble.py` applies the persona rebalance in-process, but
`detector/train.py` (the DistilBERT trainer) has no augmentation hooks --
and DistilBERT is the member that actually caps the deployed
false-positive rate.

Measured: the from-scratch ensemble alone improved to FPR 0.328 after
rebalancing, but `combined_max` stayed at 0.431, unchanged. The
combination rule is elementwise max, so its FPR is bounded below by the
worse member, and v3 sits at 0.405. Fixing one member cannot move the
deployed number; both have to be rebalanced.

Applying the same transform at the CSV level lets any trainer consume it
without code changes.

Only TRAIN rows are touched. Rows tagged hf_csv2 are the qualifire
holdout and are copied through untouched, so the evaluation set is
identical and results stay comparable.

Usage:
    python scripts/build_rebalanced_dataset.py
    python scripts/build_rebalanced_dataset.py --persona-attack-rate 0.535 --n-persona-benign 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from secureagentnet.detector.augment import (
    persona_framed_benign_augment,
    rebalance_persona_labels,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_rebalanced_dataset")

TEST_TAG = "hf_csv2"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "consolidated_dataset.csv"))
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "consolidated_rebalanced.csv"))
    p.add_argument("--persona-attack-rate", type=float, default=0.535)
    p.add_argument("--n-persona-benign", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    held_out = df[df["source_dataset"] == TEST_TAG]
    train = df[df["source_dataset"] != TEST_TAG].copy()
    logger.info("train rows %d | holdout rows %d (untouched)", len(train), len(held_out))

    train = rebalance_persona_labels(train, seed=a.seed, target_attack_rate=a.persona_attack_rate)
    if a.n_persona_benign:
        train = persona_framed_benign_augment(train, seed=a.seed, n_rows=a.n_persona_benign)

    # persona_framed_benign_augment tags generated rows source='augment';
    # the CSV path keys off source_dataset, so give them a train-routing tag.
    if "source_dataset" in train.columns:
        train["source_dataset"] = train["source_dataset"].fillna("hf_csv")
        train.loc[train["source_dataset"] == TEST_TAG, "source_dataset"] = "hf_csv"
    train["split"] = "train"

    out = pd.concat([train[df.columns], held_out], ignore_index=True)
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False, encoding="utf-8")

    n_test = int((out["source_dataset"] == TEST_TAG).sum())
    logger.info("wrote %s | total=%d train=%d holdout=%d (holdout unchanged: %s)",
                dest, len(out), len(out) - n_test, n_test, n_test == len(held_out))


if __name__ == "__main__":
    main()
