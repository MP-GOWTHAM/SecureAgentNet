"""Confusion matrices for every model on the stratified 20% test split.

Seven detectors, all scored on the same 55,600 held-out rows:
  DPCNN branch, BiLSTM branch, Transformer branch  (sigmoid of raw branch logit)
  Ensemble member -- the meta-learner's fused, temperature-scaled output
  DistilBERT, TF-IDF
  Combined -- mean(ensemble, DistilBERT, TF-IDF)

Every threshold is tuned on the validation split (accuracy-maximising,
same grid as every earlier comparison). The test split tunes nothing.

usage: eval_strat_split.py SPLIT_DIR OUT_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device

MODELS = REPO / "secureagentnet" / "data" / "models"
split_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "strat80"
CACHE = out.with_suffix(".npz")
IMX = ("imoxto", "imoxto_twin")

va = pd.read_parquet(split_dir / "val.parquet")
te = pd.read_parquet(split_dir / "test.parquet")
yv, yt = va["label"].to_numpy(), te["label"].to_numpy()
print(f"val n={len(yv)}  test n={len(yt)} ({int(yt.sum())} attack / {int((yt == 0).sum())} benign)")
device = pick_device()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def score_ensemble(texts, bs=128):
    m = InjectionRiskModel.load(str(MODELS / f"{PREFIX}_ensemble"), map_location=str(device)).to(device)
    m.eval()
    tk = load_tokenizer(m.config.model_name)
    L, R = [], []
    for i in range(0, len(texts), bs):
        enc = tk(texts[i:i + bs], padding=True, truncation=True,
                 max_length=m.config.max_length, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        logits, pooled, feats = m.branch_logits(ids, mask)
        fused = m.fuse(logits, feats, pooled)
        R.append(torch.sigmoid(fused / m.log_temperature.exp()).float().cpu().numpy())
        L.append(logits.float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(L), np.concatenate(R)


@torch.no_grad()
def score_distilbert(texts, bs=128):
    m = InjectionRiskModel.load(str(MODELS / f"{PREFIX}_distilbert"), map_location=str(device)).to(device)
    m.eval()
    tk = load_tokenizer(m.config.model_name)
    R = []
    for i in range(0, len(texts), bs):
        enc = tk(texts[i:i + bs], padding=True, truncation=True,
                 max_length=m.config.max_length, return_tensors="pt")
        R.append(m.risk_score(enc["input_ids"].to(device),
                              enc["attention_mask"].to(device)).float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(R)


def score_tfidf(texts):
    pipe = joblib.load(MODELS / f"{PREFIX}_tfidf" / "model.joblib")
    return pipe.predict_proba(texts)[:, 1].astype("float32")


if CACHE.exists():
    S = dict(np.load(CACHE))
    print("loaded cached scores")
else:
    S = {}
    for tag, frame in (("val", va), ("test", te)):
        texts = frame["text"].tolist()
        print(f"scoring {tag}: ensemble...", flush=True)
        S[f"{tag}_branch"], S[f"{tag}_fused"] = score_ensemble(texts)
        print(f"scoring {tag}: distilbert...", flush=True)
        S[f"{tag}_dbert"] = score_distilbert(texts)
        print(f"scoring {tag}: tfidf...", flush=True)
        S[f"{tag}_tfidf"] = score_tfidf(texts)
    np.savez(CACHE, **S)


def scores(tag):
    b = S[f"{tag}_branch"]
    f, d, t = S[f"{tag}_fused"], S[f"{tag}_dbert"], S[f"{tag}_tfidf"]
    return {
        "dpcnn": sigmoid(b[:, 0]),
        "bilstm": sigmoid(b[:, 1]),
        "transformer": sigmoid(b[:, 2]),
        "ensemble_fused": f,
        "distilbert": d,
        "tfidf": t,
        "combined": (f + d + t) / 3.0,
    }


LABELS = {
    "dpcnn": "DPCNN branch",
    "bilstm": "BiLSTM + attention branch",
    "transformer": "Transformer branch",
    "ensemble_fused": "Ensemble \u2014 meta-learner fused",
    "distilbert": "DistilBERT",
    "tfidf": "TF-IDF + logistic regression",
    "combined": "Combined mean(ensemble, DistilBERT, TF-IDF)",
}
GRID = np.unique(np.round(np.linspace(0.02, 0.98, 481), 4))
SV, ST = scores("val"), scores("test")

def cm(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"n": int(len(y)), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": float((tp + tn) / max(len(y), 1)), "precision": float(prec),
            "recall": float(rec), "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
            "FPR": float(fp / max(fp + tn, 1)), "FNR": float(fn / max(fn + tp, 1)),
            "AUC": float(roc_auc_score(y, s)) if len(set(y)) > 1 else None}


# The six non-imoxto sources' rows are identical before and after the imoxto
# fix, so this subset is the like-for-like comparison between the two runs.
six = ~te["source"].isin(IMX).to_numpy()

results = {}
for key in LABELS:
    sv, st = SV[key], ST[key]
    accs = np.array([((sv >= t) == (yv == 1)).mean() for t in GRID])
    thr = float(GRID[int(np.argmax(accs))])
    results[key] = {"label": LABELS[key], "threshold": thr, "val_accuracy": float(accs.max()),
                    **cm(yt, st, thr),
                    "six_source": cm(yt[six], st[six], thr),
                    "imoxto_slice": cm(yt[~six], st[~six], thr) if (~six).any() else None}

print(f"\n{'model':<46} {'thr':>6} {'acc':>7} {'AUC':>7} {'FPR':>7} {'FNR':>7}   TP     FP     TN     FN"
      f"   six-src acc  imoxto acc")
print("-" * 142)
for r in results.values():
    ia = f"{r['imoxto_slice']['accuracy']:.4f}" if r["imoxto_slice"] else "--"
    print(f"{r['label']:<46} {r['threshold']:>6.3f} {r['accuracy']:>7.4f} {r['AUC']:>7.4f} "
          f"{r['FPR']:>7.4f} {r['FNR']:>7.4f}  {r['TP']:>5}  {r['FP']:>5}  {r['TN']:>5}  {r['FN']:>5}"
          f"   {r['six_source']['accuracy']:>10.4f}  {ia:>10}")

# per-source accuracy for the two headline detectors
te = te.assign(_c=(ST["combined"] >= results["combined"]["threshold"]).astype(int),
               _f=(ST["ensemble_fused"] >= results["ensemble_fused"]["threshold"]).astype(int))
by_src = {}
for s, g in te.groupby("source"):
    by_src[s] = {"rows": int(len(g)),
                 "combined_acc": float((g["_c"] == g["label"]).mean()),
                 "ensemble_fused_acc": float((g["_f"] == g["label"]).mean())}
print("\nper-source accuracy:")
for s, v in by_src.items():
    print(f"  {s:<20} {v['rows']:>6}  combined {v['combined_acc']:.4f}  ensemble {v['ensemble_fused_acc']:.4f}")

out.write_text(json.dumps({"n_test": int(len(yt)), "n_attack": int(yt.sum()),
                           "n_benign": int((yt == 0).sum()), "n_val": int(len(yv)),
                           "models": results, "by_source": by_src}, indent=2),
               encoding="utf-8")
print(f"\nwrote {out}")
