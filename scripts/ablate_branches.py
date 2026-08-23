"""Per-branch ablation for the from-scratch ensemble.

The design's central claim is that each branch fixes a *specific* measured
failure -- char-CNN counters obfuscation, BiLSTM+attention counters filler
dilution, the transformer carries lexical load. Aggregate ensemble metrics
cannot support that claim. This measures each branch on its own.

No retraining is needed, and that is a property of the architecture rather
than a shortcut: `train_ensemble` optimises
`loss = mean(branch_loss(logit_i) for i in 0..2)`, and the three branches
share no parameters (only `embed_proj` and `meta` sit downstream of the
concat, and neither is in that loss). Gradients from branch i therefore
touch branch i and its head alone, so each branch is already trained as if
it were alone. Reading its logit back out is the ablation.

Reported per branch:

    AUC                 threshold-free, the headline number
    F1 / FPR / FNR      at 0.5 -- see the calibration caveat below
    short attacks       of the 8 canonical seed attacks
    dilution gap        score(attack + filler) - score(bare attack)

Calibration caveat: only the *fused* output gets temperature scaling
(Stage C). Individual branch logits are uncalibrated, so their 0.5-
threshold numbers are not directly comparable to the ensemble's. AUC and
the dilution gap are threshold-free and are the fair comparisons.

Usage:
    python scripts/ablate_branches.py
    python scripts/ablate_branches.py --model-dir secureagentnet/data/models/ensemble_v4_persona
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.ensemble import EnsembleInjectionRiskModel
from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device

from probe_short_attacks import FILLER, SHORT_ATTACKS, SHORT_BENIGN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ablate_branches")

BRANCH_NAMES = ["char-CNN", "BiLSTM+attn", "transformer"]
THRESHOLD = 0.5


@torch.no_grad()
def branch_and_fused_scores(model, tokenizer, texts, device, batch_size=32):
    """Returns (branch_scores (n,3), fused_scores (n,))."""
    model.eval()
    b_rows, fused = [], []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=model.config.max_length, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits, _, feats = model.branch_logits(ids, mask)
        b_rows.append(torch.sigmoid(logits).float().cpu().numpy())
        fused_logit = model.meta(torch.cat([logits, feats], dim=1)).squeeze(-1)
        fused.append(torch.sigmoid(fused_logit / model.log_temperature.exp()).float().cpu().numpy())
    return np.concatenate(b_rows), np.concatenate(fused)


def metrics(scores, labels):
    pred = (scores >= THRESHOLD).astype(int)
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tp = int(((pred == 1) & (labels == 1)).sum())
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=str(REPO_ROOT / "secureagentnet" / "data" / "models" / "ensemble_v4_persona"))
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "consolidated_dataset.csv"))
    p.add_argument("--out", default=str(REPO_ROOT / "secureagentnet" / "reports" / "branch_ablation.json"))
    a = p.parse_args()

    device = pick_device()
    model = InjectionRiskModel.load(a.model_dir, map_location=str(device)).to(device)
    if not isinstance(model, EnsembleInjectionRiskModel):
        raise SystemExit(f"{a.model_dir} is not an ensemble checkpoint (branch ablation needs one)")
    tokenizer = load_tokenizer(model.config.model_name)

    test = dl.build_splits_from_csv(a.csv)["test"].reset_index(drop=True)
    labels = test["label"].to_numpy()
    logger.info("scoring %d holdout rows...", len(test))
    b_scores, fused = branch_and_fused_scores(model, tokenizer, test["text"].tolist(), device)

    # Behavioural probes
    atk_b, atk_f = branch_and_fused_scores(model, tokenizer, SHORT_ATTACKS, device)
    ben_b, ben_f = branch_and_fused_scores(model, tokenizer, SHORT_BENIGN, device)
    pad_b, pad_f = branch_and_fused_scores(
        model, tokenizer, [f"{t} {FILLER}" for t in SHORT_ATTACKS], device)

    rows, results = [], {}
    for i, name in enumerate(BRANCH_NAMES):
        m = metrics(b_scores[:, i], labels)
        m["short_attacks_caught"] = int((atk_b[:, i] >= THRESHOLD).sum())
        m["short_benign_fp"] = int((ben_b[:, i] >= THRESHOLD).sum())
        m["dilution_gap"] = float(np.mean(pad_b[:, i] - atk_b[:, i]))
        results[name] = m
        rows.append((name, m))

    m = metrics(fused, labels)
    m["short_attacks_caught"] = int((atk_f >= THRESHOLD).sum())
    m["short_benign_fp"] = int((ben_f >= THRESHOLD).sum())
    m["dilution_gap"] = float(np.mean(pad_f - atk_f))
    results["full ensemble"] = m
    rows.append(("full ensemble", m))

    print(f"\nPer-branch ablation — {Path(a.model_dir).name}, holdout n={len(test)}")
    print(f"{'branch':<16}{'AUC':>8}{'F1':>8}{'FPR':>8}{'FNR':>8}{'short atk':>11}{'benign FP':>11}{'dil.gap':>10}")
    print("-" * 80)
    for name, m in rows:
        print(f"{name:<16}{m['auc']:>8.4f}{m['f1']:>8.4f}{m['fpr']:>8.4f}{m['fnr']:>8.4f}"
              f"{m['short_attacks_caught']:>8}/8{m['short_benign_fp']:>8}/4{m['dilution_gap']:>+10.3f}")

    print("\nAUC and dilution gap are threshold-free and directly comparable.")
    print("F1/FPR/FNR at 0.5 are not: only the fused output is temperature-calibrated.")

    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
