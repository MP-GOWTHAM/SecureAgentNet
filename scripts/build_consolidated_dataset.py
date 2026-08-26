"""Builds the canonical `consolidated_dataset.csv` that the scripts in this
directory expect via SECUREAGENTNET_CSV / DEFAULT_CSV.

This is the consolidate.py-style artifact referenced throughout the README:
all four HF sources merged into one CSV with columns
`text, label, attack_type, source_dataset, split`, where `source_dataset`
uses the hf_csv/hf_csv2/hf_csv3/hf_csv4 tags that data_loader.CSV_SOURCE_MAP
maps back to source names.

Holdout isolation is PRESERVED here: every qualifire row is tagged hf_csv2,
which build_splits_from_csv routes exclusively to `test` and never to
train/val. (Contrast scripts/build_pooled_split_dataset.py, which
deliberately breaks that for the pooled-split experiment.)

Usage:
    python scripts/build_consolidated_dataset.py
    python scripts/build_consolidated_dataset.py --necent-max-rows 30000
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
logger = logging.getLogger("build_consolidated_dataset")

SOURCE_TO_CSV_TAG = {
    "neuralchemy": "hf_csv",
    "qualifire_holdout": "hf_csv2",
    "necent": "hf_csv3",
    "mindgard": "hf_csv4",
    "smooth3": "hf_csv5",
    "jayavibhav": "hf_csv6",
    "imoxto": "hf_csv7",
}


def main():
    parser = argparse.ArgumentParser()
    # 30k matches DATASET_SPECS' default cap, i.e. the distribution the
    # shipped detector versions were trained on.
    parser.add_argument("--necent-max-rows", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "consolidated_dataset.csv"))
    args = parser.parse_args()

    frames = []
    for spec in dl.DATASET_SPECS:
        if spec.name == "necent":
            spec = replace(spec, max_rows=args.necent_max_rows)
        logger.info("Loading %s (%s, max_rows=%s)...", spec.name, spec.hf_id, spec.max_rows)
        df = dl.load_and_normalize(spec, seed=args.seed)
        logger.info("  -> %d rows", len(df))
        if len(df):
            frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "text": merged["text"],
        "label": merged["label"],
        "attack_type": merged["category"].fillna("unknown"),
        "source_dataset": merged["source"].map(SOURCE_TO_CSV_TAG).fillna("hf_csv"),
    })
    out["split"] = out["source_dataset"].map(lambda t: "test" if t == "hf_csv2" else "train")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False, encoding="utf-8")

    counts = out["source_dataset"].value_counts().to_dict()
    logger.info("wrote %s | total=%d | by tag: %s | test rows (hf_csv2, held out): %d",
                dest, len(out), counts, int((out["source_dataset"] == "hf_csv2").sum()))


if __name__ == "__main__":
    main()
