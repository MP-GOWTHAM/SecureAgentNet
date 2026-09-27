"""Assemble combined_strat80fix_t057 and verify it through the real load path.

mean(strat80fix_ensemble, strat80fix_distilbert, strat80fix_tfidf) with the
qualifire-tuned operating point 0.571 folded into the score, so the
standard 0.5 cut downstream IS the new threshold.

Verified by loading with InjectionRiskModel.load (the web app's path) and
scoring qualifire (all + unseen), the 7-source test, and the project probes.
"""
from __future__ import annotations

import json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
logging.disable(logging.CRITICAL)

import numpy as np, pandas as pd
from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.combined import CombinedRiskModel, CombinedRiskModelConfig
from secureagentnet.detector.model import InjectionRiskModel
from secureagentnet.detector.train import pick_device
from probe_short_attacks import FILLER, SHORT_ATTACKS, SHORT_BENIGN

SP = Path(sys.argv[1])
MODELS = REPO / "secureagentnet" / "data" / "models"
NAME = "combined_strat80fix_t057"
THRESHOLD = 0.571

cfg = CombinedRiskModelConfig(
    members=["strat80fix_ensemble", "strat80fix_distilbert", "strat80fix_tfidf"],
    mode="mean", decision_threshold=THRESHOLD)
dev = pick_device()
CombinedRiskModel(cfg, map_location=str(dev)).save(MODELS / NAME)
print(f"saved {NAME}: {json.loads((MODELS / NAME / 'config.json').read_text())}")

m = InjectionRiskModel.load(str(MODELS / NAME), map_location=str(dev)).to(dev)
m.eval()
print("loaded as", type(m).__name__)


def cm(y, s):
    p = s >= 0.5
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    tn = int((~p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    return {"n": int(len(y)), "acc": (tp + tn) / len(y), "FPR": fp / max(fp + tn, 1),
            "FNR": fn / max(fn + tp, 1), "TP": tp, "FP": fp, "TN": tn, "FN": fn}


q = dl.build_splits_from_csv(str(REPO / "data" / "consolidated_v2.csv"))["test"]
yq = q["label"].to_numpy()
fix = SP / "strat80_flipall"
seen = set()
for n in ("train", "val"):
    seen |= set(pd.read_parquet(fix / f"{n}.parquet")["text"].map(dl._text_key))
unseen = ~q["text"].map(dl._text_key).isin(seen).to_numpy()
sq = m.score_from_texts(q["text"].tolist(), batch_size=64).cpu().numpy()

te = pd.read_parquet(fix / "test.parquet")
st = m.score_from_texts(te["text"].tolist(), batch_size=64).cpu().numpy()

EV = json.loads((REPO / "secureagentnet/tests/fixtures/evasions.json").read_text(encoding="utf-8"))
pa = m.score_from_texts(SHORT_ATTACKS).cpu().numpy()
pb = m.score_from_texts(SHORT_BENIGN).cpu().numpy()
pe = m.score_from_texts(EV).cpu().numpy()
pd_ = m.score_from_texts([f"{t} {FILLER}" for t in SHORT_ATTACKS]).cpu().numpy()

res = {"qualifire_all": cm(yq, sq), "qualifire_unseen": cm(yq[unseen], sq[unseen]),
       "test_7src": cm(te["label"].to_numpy(), st),
       "probes": {"short_attacks": f"{int((pa >= 0.5).sum())}/{len(pa)}",
                  "short_benign_fp": f"{int((pb >= 0.5).sum())}/{len(pb)}",
                  "evasions": f"{int((pe >= 0.5).sum())}/{len(pe)}",
                  "dilution_gap": round(float(np.mean(pd_ - pa)), 3)}}
for k, c in res.items():
    if k == "probes":
        print(f"  probes: {c}")
    else:
        print(f"  {k:<17} acc {c['acc']:.4f}  FPR {c['FPR']:.4f}  FNR {c['FNR']:.4f}  "
              f"TP {c['TP']} FP {c['FP']} TN {c['TN']} FN {c['FN']}")
(SP / "apply_057.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
