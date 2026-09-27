"""Retrain one model on the stratified 80:20 split, using the project's own
training code unchanged.

Both trainers fetch data through `data_loader.build_splits_from_csv`; it is
replaced here with a function that returns the prebuilt split, so the
recipe (tokenizer training, Stage A/B/C, best-val-AUC checkpointing) is
exactly the project's -- only the rows differ.

usage: train_strat_split.py {ensemble|distilbert|tfidf} SPLIT_DIR OUT_DIR
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd

from secureagentnet.detector import data_loader as dl

which, split_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_strat")

SPLITS = {n: pd.read_parquet(split_dir / f"{n}.parquet") for n in ("train", "val", "test")}
log.info("split: train=%d val=%d test=%d", *(len(SPLITS[n]) for n in ("train", "val", "test")))


def _prebuilt(*_a, **_k):
    return {k: v.copy() for k, v in SPLITS.items()}


dl.build_splits_from_csv = _prebuilt   # both trainers call dl.build_splits_from_csv
t0 = time.time()

if which == "ensemble":
    from secureagentnet.detector import train_ensemble as te
    # The architecture the diagrams describe: DPCNN byte branch + BiLSTM +
    # transformer, linear stacking head. Everything else is the default.
    metrics = te.train(csv_path="prebuilt", output_dir=out_dir, epochs=3,
                       char_branch="dpcnn", meta_kind="linear")

elif which == "distilbert":
    from secureagentnet.detector import train as tr
    metrics = tr.train(csv_path="prebuilt", output_dir=out_dir, epochs=2, batch_size=32)

elif which == "tfidf":
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    # Identical to tfidf_id93's recipe.
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=300_000,
                        min_df=2, sublinear_tf=True, lowercase=False),
        LogisticRegression(C=4.0, max_iter=2000),
    )
    tr_df = SPLITS["train"]
    log.info("fitting TF-IDF + LR on %d rows...", len(tr_df))
    pipe.fit(tr_df["text"].tolist(), tr_df["label"].to_numpy())
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    metrics = {"fit_rows": len(tr_df)}
else:
    raise SystemExit(f"unknown model {which}")

elapsed = time.time() - t0
log.info("DONE %s in %.1f min", which, elapsed / 60)
(out_dir / "strat80_train_info.json").write_text(
    json.dumps({"model": which, "minutes": round(elapsed / 60, 1),
                "metrics": metrics}, indent=2, default=str), encoding="utf-8")
