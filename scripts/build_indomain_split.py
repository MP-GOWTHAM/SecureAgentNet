"""Build an IN-DOMAIN corpus: qualifire split against itself.

Read this before using the numbers it produces.
-----------------------------------------------
Every other corpus in this repo holds qualifire out as an entire SOURCE,
so a model must generalise across a labelling convention it never saw.
That is the harder, more honest question, and the answer is ~73% accuracy
/ 0.83 AUC. The residual error is dominated by long roleplay prompts that
the training sources label as attacks and qualifire labels as benign.

This file answers a different question: how well does a detector work on
the distribution it is actually deployed against, when it has seen that
distribution's labelling convention. Numbers from it are IN-DOMAIN and
must never be reported as cross-source generalisation.

The mechanism is deliberately boring: qualifire rows are re-tagged so the
existing source-based splitter does the work. 80% become a training
source, 20% stay as the held-out source. The rows are disjoint, and the
loader's own cross-split leak filter still runs on top.

Optionally keeps the other sources too, but measured, that HURTS:
in-domain accuracy 0.907 with qualifire alone against 0.838 once the
other 80k rows are mixed in, because their convention disagrees.

Usage:
    python scripts/build_indomain_split.py
    python scripts/build_indomain_split.py --keep-other-sources
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_indomain")

HOLDOUT_SRC = "hf_csv2"      # qualifire, the held-out source
TRAIN_TAG = "hf_csv10"       # qualifire rows promoted to a training source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO_ROOT / "data" / "consolidated_v2.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "consolidated_indomain.csv"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--keep-other-sources", action="store_true",
                    help="also keep the non-qualifire sources; measured, this "
                         "lowers in-domain accuracy 0.907 -> 0.838")
    ap.add_argument("--add-other-attacks", action="store_true",
                    help="keep the other sources' ATTACK rows but drop their "
                         "benign rows. The convention conflict is on benign "
                         "labelling -- long roleplay the other sources call an "
                         "attack and qualifire calls benign -- so their attacks "
                         "are not in conflict and restore the canonical "
                         "coverage that qualifire-only training loses "
                         "(3/8 short attacks -> 8/8).")
    ap.add_argument("--other-attacks-max", type=int, default=1000,
                    help="cap on injected attack rows; 1000 is the measured knee")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.csv, encoding="utf-8").dropna(subset=["text", "label"])
    q = df[df["source_dataset"] == HOLDOUT_SRC].copy()
    other = df[df["source_dataset"] != HOLDOUT_SRC].copy()
    logger.info("qualifire rows %d, other sources %d", len(q), len(other))

    q_tr, q_te = train_test_split(q, test_size=a.test_frac, random_state=a.seed,
                                  stratify=q["label"])
    q_tr = q_tr.assign(source_dataset=TRAIN_TAG, split="train")
    q_te = q_te.assign(source_dataset=HOLDOUT_SRC, split="test")

    parts = [q_tr, q_te]
    if a.keep_other_sources:
        parts.insert(0, other)
    elif a.add_other_attacks:
        atk = other[other["label"] == 1]
        if a.other_attacks_max and len(atk) > a.other_attacks_max:
            # All 41k of them make the corpus 95% positive and cost AUC.
            # Swept with the TF-IDF model: coverage saturates at 8/8 short
            # attacks and 8/8 evasions by 1000 rows, and accuracy falls
            # monotonically past it (0.912 at 1000 -> 0.884 at 16000).
            atk = atk.sample(n=a.other_attacks_max, random_state=a.seed)
        logger.info("adding %d attack rows from the other sources "
                    "(their %d benign rows are the conflicting ones, dropped)",
                    len(atk), int((other["label"] == 0).sum()))
        parts.insert(0, atk)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(a.out, index=False, encoding="utf-8")

    logger.info("wrote %s: %d rows", a.out, len(out))
    logger.info("  in-domain train %d (%d attack / %d benign)", len(q_tr),
                int((q_tr["label"] == 1).sum()), int((q_tr["label"] == 0).sum()))
    logger.info("  held-out test   %d (%d attack / %d benign)", len(q_te),
                int((q_te["label"] == 1).sum()), int((q_te["label"] == 0).sum()))
    logger.info("NOTE: results from this corpus are IN-DOMAIN, not cross-source.")


if __name__ == "__main__":
    main()
