"""Reduce false positives on qualifire without retraining.

  baseline      mean(ens, DistilBERT, TF-IDF) >= 0.426 (7-source val threshold)
  vote 2-of-3   flag only if at least 2 members say attack, each member at
                its own validation-tuned threshold
  vote 3-of-3   all three must agree
  raised thr    mean of 3, threshold re-tuned FOR qualifire
  vote + raised 2-of-3 agreement AND mean above the raised threshold

"Raised" thresholds are cross-fitted: qualifire is split into two stratified
halves; the threshold is tuned on one half and applied to the other, then
swapped. No row is scored with a threshold that saw it. Done separately for
all 5000 rows and for the 980 rows never in training.

Vote-only options use no qualifire tuning, so their 7-source cost is also
reported (cached test scores).
"""
from __future__ import annotations

import json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
logging.disable(logging.CRITICAL)

import joblib, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device

SP = Path(sys.argv[1])
MODELS = REPO / "secureagentnet" / "data" / "models"
fix = SP / "strat80_flipall"
TJ = json.loads((SP / "strat80fix_eval.json").read_text(encoding="utf-8"))["models"]
T_ENS, T_DB, T_TF, T_MEAN = (TJ[k]["threshold"] for k in ("ensemble_fused", "distilbert", "tfidf", "combined"))

q = dl.build_splits_from_csv(str(REPO / "data" / "consolidated_v2.csv"))["test"]
yq = q["label"].to_numpy(); texts = q["text"].tolist()
seen = set()
for n in ("train", "val"):
    seen |= set(pd.read_parquet(fix / f"{n}.parquet")["text"].map(dl._text_key))
unseen = ~q["text"].map(dl._text_key).isin(seen).to_numpy()

CACHE = SP / "qualifire_member_scores.npz"
if CACHE.exists():
    z = np.load(CACHE); QF, QD, QT = z["f"], z["d"], z["t"]
else:
    dev = pick_device()

    @torch.no_grad()
    def sc(name):
        m = InjectionRiskModel.load(str(MODELS / name), map_location=str(dev)).to(dev); m.eval()
        tk = load_tokenizer(m.config.model_name); R = []
        for i in range(0, len(texts), 64):
            e = tk(texts[i:i + 64], padding=True, truncation=True, max_length=m.config.max_length, return_tensors="pt")
            R.append(m.risk_score(e["input_ids"].to(dev), e["attention_mask"].to(dev)).float().cpu().numpy())
        return np.concatenate(R)

    QF, QD = sc("strat80fix_ensemble"), sc("strat80fix_distilbert")
    QT = joblib.load(MODELS / "strat80fix_tfidf" / "model.joblib").predict_proba(texts)[:, 1]
    np.savez(CACHE, f=QF, d=QD, t=QT)

GRID = np.unique(np.round(np.linspace(0.02, 0.98, 481), 4))


def votes(f, d, t):
    return (f >= T_ENS).astype(int) + (d >= T_DB).astype(int) + (t >= T_TF).astype(int)


def crossfit(y, mean, gate):
    """Predictions where each half's threshold is tuned on the other half.
    gate: extra boolean requirement (e.g. 2-of-3 agreement) applied in both
    tuning and scoring."""
    pred = np.zeros(len(y), bool); thrs = []
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for tune, score in skf.split(mean, y):
        accs = [(((mean[tune] >= g) & gate[tune]) == (y[tune] == 1)).mean() for g in GRID]
        t = float(GRID[int(np.argmax(accs))]); thrs.append(t)
        pred[score] = (mean[score] >= t) & gate[score]
    return pred, thrs


def cm(y, p):
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    tn = int((~p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"n": int(len(y)), "TP": tp, "FP": fp, "TN": tn, "FN": fn, "acc": (tp + tn) / len(y),
            "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9),
            "FPR": fp / max(fp + tn, 1), "FNR": fn / max(fn + tp, 1)}


res = {}
for subset, mask in (("all5000", np.ones(len(yq), bool)), ("unseen980", unseen)):
    y = yq[mask]; f, d, t = QF[mask], QD[mask], QT[mask]
    mean = (f + d + t) / 3; v = votes(f, d, t); yes = np.ones(len(y), bool)
    p_raise, thr_raise = crossfit(y, mean, yes)
    p_both, thr_both = crossfit(y, mean, v >= 2)
    res[subset] = {
        "baseline":      {**cm(y, mean >= T_MEAN), "rule": f"mean >= {T_MEAN:.3f}"},
        "vote 2-of-3":   {**cm(y, v >= 2), "rule": "at least 2 members flag"},
        "vote 3-of-3":   {**cm(y, v >= 3), "rule": "all 3 members flag"},
        "raised thr":    {**cm(y, p_raise), "rule": f"mean >= {thr_raise} (cross-fitted)"},
        "vote + raised": {**cm(y, p_both), "rule": f"2-of-3 AND mean >= {thr_both} (cross-fitted)"},
    }

# 7-source cost of the options that need no qualifire tuning (+ raised thr at its mean value)
C = dict(np.load(SP / "strat80fix_eval.npz"))
yt = pd.read_parquet(fix / "test.parquet")["label"].to_numpy()
f, d, t = C["test_fused"], C["test_dbert"], C["test_tfidf"]
mean, v = (f + d + t) / 3, votes(f, d, t)
tr_all = float(np.mean([float(x) for x in res["all5000"]["raised thr"]["rule"].split("[")[1].split("]")[0].split(",")]))
res["test7src"] = {
    "baseline": cm(yt, mean >= T_MEAN), "vote 2-of-3": cm(yt, v >= 2), "vote 3-of-3": cm(yt, v >= 3),
    "raised thr": {**cm(yt, mean >= tr_all), "rule": f"mean >= {tr_all:.3f} (qualifire-tuned, avg of folds)"},
}

for subset, rows in res.items():
    print(f"\n=== {subset} ===")
    for name, c in rows.items():
        print(f"  {name:<14} acc {c['acc']:.4f}  FPR {c['FPR']:.4f}  FNR {c['FNR']:.4f}   "
              f"TP {c['TP']:>5} FP {c['FP']:>5} TN {c['TN']:>5} FN {c['FN']:>5}   {c.get('rule', '')}")
(SP / "fpr_fix.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
