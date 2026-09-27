"""Build a stratified 80:20 split of the full 331k combined corpus.

Unlike every other protocol in this repo (which holds out a whole SOURCE),
this splits rows at random, stratified on label x source, so the test set
has the same mix of sources and the same attack rate as training.

Order matters:
  1. Same preprocessing as build_splits_from_csv (template strip, labels,
     source names, necent cap).
  2. Dedup the WHOLE corpus on the normalised text key BEFORE splitting.
     A random split without this puts the same prompt on both sides --
     the corpus mirrors viral jailbreaks across sources, and 9.1% of
     Smooth-3 is verbatim qualifire. Keys whose copies disagree on the
     label are dropped entirely: there is no right answer to test against.
  3. 80:20 stratified split -> train_pool / test.
  4. 10% of train_pool (stratified the same way) -> val, used for the
     meta-learner, temperature, and every threshold. Test is never used
     for tuning anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd
from sklearn.model_selection import train_test_split

from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.text_normalize import has_chat_template, strip_chat_template

CSV = REPO / "data" / "consolidated_v4.csv"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

df = pd.read_csv(CSV, usecols=["text", "label", "attack_type", "source_dataset", "split"],
                 encoding="utf-8")
n_raw = len(df)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
n_tmpl = int(df["text"].map(has_chat_template).sum())
df["text"] = df["text"].map(strip_chat_template)
df = df[df["text"].str.strip().astype(bool)]
df["label"] = df["label"].astype(int)
df["category"] = df["attack_type"].fillna("unknown")
df["source"] = df["source_dataset"].map(dl.CSV_SOURCE_MAP).fillna(df["source_dataset"])

# necent cap, as the loader applies it (a no-op here: v4 already holds 30k)
necent = df["source"] == "necent"
if necent.sum() > 30_000:
    frac = 30_000 / necent.sum()
    parts = [g.sample(frac=frac, random_state=SEED) for _, g in df[necent].groupby("label")]
    df = pd.concat([df[~necent], *parts], ignore_index=True)
n_pre = len(df)

# --- dedup the whole corpus on the normalised key ------------------------
df["_key"] = df["text"].map(dl._text_key)
label_sets = df.groupby("_key")["label"].nunique()
conflict_keys = set(label_sets[label_sets > 1].index)
n_conflict_rows = int(df["_key"].isin(conflict_keys).sum())
df = df[~df["_key"].isin(conflict_keys)]
n_before_dedup = len(df)
df = df.drop_duplicates(subset="_key").reset_index(drop=True)
n_dupes = n_before_dedup - len(df)

# --- 80:20 stratified on label x source -----------------------------------
strat = df["label"].astype(str) + "|" + df["source"]
train_pool, test = train_test_split(df, test_size=0.20, stratify=strat, random_state=SEED)
strat_tp = train_pool["label"].astype(str) + "|" + train_pool["source"]
train, val = train_test_split(train_pool, test_size=0.10, stratify=strat_tp, random_state=SEED)

# sanity: no text key on two sides
kt, kv, ks = set(train["_key"]), set(val["_key"]), set(test["_key"])
assert not (kt & ks) and not (kv & ks) and not (kt & kv), "key overlap across splits"

cols = [*dl.SCHEMA_COLUMNS]
for name, part in (("train", train), ("val", val), ("test", test)):
    part[cols].reset_index(drop=True).to_parquet(OUT / f"{name}.parquet", index=False)

def summ(d):
    return {"rows": int(len(d)), "attack": int(d["label"].sum()),
            "benign": int((d["label"] == 0).sum()),
            "attack_rate": round(float(d["label"].mean()), 4)}

stats = {
    "csv": str(CSV), "seed": SEED,
    "raw_rows": int(n_raw), "chat_template_rows_stripped": n_tmpl,
    "rows_after_preprocessing": int(n_pre),
    "rows_dropped_conflicting_labels": n_conflict_rows,
    "conflicting_keys": len(conflict_keys),
    "duplicate_rows_removed": int(n_dupes),
    "unique_rows": int(len(df)),
    "train": summ(train), "val": summ(val), "test": summ(test),
    "test_by_source": {s: summ(g) for s, g in test.groupby("source")},
}
(OUT / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in stats.items() if k != "test_by_source"}, indent=2))
print("\ntest by source:")
for s, v in stats["test_by_source"].items():
    print(f"  {s:<20} {v['rows']:>7} rows  {v['attack']:>6} attack  rate {v['attack_rate']:.3f}")
