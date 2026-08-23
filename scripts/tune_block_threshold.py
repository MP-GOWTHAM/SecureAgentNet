"""Retunes `block_risk_threshold` for a calibrated detector.

Why this is needed: the shipped default of 0.85 was set against
DistilBERT, whose scores are not calibrated and sit near 1.0 for anything
it considers an attack. The phase-2 ensemble applies temperature scaling,
so genuine attacks land around 0.7-0.9 -- below the 0.85 cut. The two
models had near-identical FNR (0.099 vs 0.090) yet ASR of 0.171 vs 0.101,
and that gap is the threshold mismatch, not a detection failure.

Protocol -- the important part:

    the sweep runs on the VALIDATION split
    the chosen threshold is then measured ONCE on the holdout

Sweeping on the holdout and reporting the best number from that sweep
would be tuning on the test set. That is the same class of error as the
pooled 80:20 split (which turned a real 0.747 AUC into an apparent 0.997)
and it would silently inflate every number in the results table.

`block_combo_risk_threshold` is swept in lockstep at `thr - 0.45`,
preserving the shipped 0.85/0.40 offset so the combo path keeps its
relative sensitivity rather than being retuned implicitly.

Usage:
    python scripts/tune_block_threshold.py --model-dir secureagentnet/data/models/ensemble_v2_aug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from secureagentnet.correlation.fusion import FusionConfig, FusionEngine
from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device
from secureagentnet.eval.baselines import framework_actions
from secureagentnet.eval.metrics import compute_metrics
from secureagentnet.eval.run_eval import score_texts
from secureagentnet.privilege.policy_engine import POLICIES_DIR, PolicyEngine, issue_credential
from secureagentnet.simulate.agent_env import ROLES, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tune_block_threshold")

EVAL_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SHIPPED_BLOCK, SHIPPED_COMBO = 0.85, 0.40
COMBO_OFFSET = SHIPPED_BLOCK - SHIPPED_COMBO  # 0.45, preserved across the sweep


def evaluate_at(threshold, risk_scores, df, policy_engine, credential_by_role, strict_privilege):
    engine = FusionEngine(FusionConfig(
        block_risk_threshold=threshold,
        block_combo_risk_threshold=max(0.05, threshold - COMBO_OFFSET),
        strict_privilege=strict_privilege,
    ))
    results = [
        run_pipeline(
            text=row.text, true_label=int(row.label), index=i, risk_score=risk_scores[i],
            policy_engine=policy_engine, fusion_engine=engine,
            credential_by_role=credential_by_role, now=EVAL_NOW,
        )
        for i, row in enumerate(df.itertuples())
    ]
    m = compute_metrics(results, framework_actions(results))
    return {"threshold": threshold, "asr": m.asr, "c_asr": m.c_asr,
            "fpr": m.fpr, "fnr": m.fnr, "utility": m.utility_preservation}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "consolidated_dataset.csv"))
    p.add_argument("--strict-privilege", action="store_true")
    p.add_argument("--match-model", default=None,
                   help="reference checkpoint whose block rate at 0.85 should be matched "
                        "(recommended; metric-based selection does not survive the val/holdout shift)")
    p.add_argument("--min-utility", type=float, default=0.70,
                   help="floor for the fallback metric-based rule when --match-model is not given")
    p.add_argument("--out", default=None, help="write chosen config as JSON")
    a = p.parse_args()

    device = pick_device()
    model = InjectionRiskModel.load(a.model_dir, map_location=str(device)).to(device)
    tokenizer = load_tokenizer(model.config.model_name)

    splits = dl.build_splits_from_csv(a.csv)
    val_df, test_df = splits["val"], splits["test"]

    policy_engine = PolicyEngine.from_directory(POLICIES_DIR)
    credential_by_role = {r: issue_credential(r, ttl_seconds=3600, now=EVAL_NOW) for r in ROLES}

    # ---- sweep on validation ------------------------------------------
    logger.info("Scoring %d VALIDATION rows (tuning split)...", len(val_df))
    val_scores = score_texts(model, tokenizer, val_df["text"].tolist(), device, model.config.max_length)

    thresholds = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90
    sweep = [
        evaluate_at(t, val_scores, val_df, policy_engine, credential_by_role, a.strict_privilege)
        for t in thresholds
    ]

    print("\nVALIDATION sweep (tuning split -- selection happens here)")
    print("thr    ASR     C-ASR   FPR     FNR     Utility")
    print("-" * 50)
    for r in sweep:
        print(f"{r['threshold']:.2f}   {r['asr']:.3f}   {r['c_asr']:.3f}   "
              f"{r['fpr']:.3f}   {r['fnr']:.3f}   {r['utility']:.3f}")

    # ---- selection ----------------------------------------------------
    # Optimising ASR/utility on validation does not transfer: validation FPR
    # is ~0.04 against ~0.38 on the holdout, so any utility floor set here
    # is meaningless there, and the ASR-minimising choice collapses to the
    # lowest threshold in the sweep.
    #
    # Block-rate matching instead. The shipped 0.85 encodes an intended
    # *aggressiveness* -- what fraction of attacks the framework blocks
    # outright -- and that intent is what should carry over to a calibrated
    # model, not a metric value. So: measure the block rate the reference
    # model produces at 0.85 on validation attacks, then pick the threshold
    # at which this model blocks the same fraction. It touches only
    # validation scores, never a holdout metric.
    if a.match_model:
        ref = InjectionRiskModel.load(a.match_model, map_location=str(device)).to(device)
        ref_tok = load_tokenizer(ref.config.model_name)
        attacks = val_df[val_df["label"] == 1]["text"].tolist()
        ref_scores = score_texts(ref, ref_tok, attacks, device, ref.config.max_length)
        target_rate = sum(s >= SHIPPED_BLOCK for s in ref_scores) / len(ref_scores)

        own_scores = score_texts(model, tokenizer, attacks, device, model.config.max_length)
        chosen = min(
            thresholds,
            key=lambda t: abs(sum(s >= t for s in own_scores) / len(own_scores) - target_rate),
        )
        own_rate = sum(s >= chosen for s in own_scores) / len(own_scores)
        print(f"\nselection rule: match the reference model's block rate on validation attacks")
        print(f"  reference {Path(a.match_model).name} at {SHIPPED_BLOCK}: blocks {target_rate:.1%}")
        print(f"  this model at {chosen:.2f}: blocks {own_rate:.1%}")
    else:
        eligible = [r for r in sweep if r["utility"] >= a.min_utility] or sweep
        chosen = min(eligible, key=lambda r: (r["asr"], -r["utility"]))["threshold"]
        print(f"\nselection rule: min ASR subject to validation utility >= {a.min_utility}")

    print(f"chosen block_risk_threshold = {chosen:.2f} "
          f"(combo {max(0.05, chosen - COMBO_OFFSET):.2f}), was {SHIPPED_BLOCK}")

    # ---- confirm ONCE on the holdout ----------------------------------
    logger.info("Scoring %d HOLDOUT rows (confirmation only, no selection)...", len(test_df))
    test_scores = score_texts(model, tokenizer, test_df["text"].tolist(), device, model.config.max_length)

    before = evaluate_at(SHIPPED_BLOCK, test_scores, test_df, policy_engine, credential_by_role, a.strict_privilege)
    after = evaluate_at(chosen, test_scores, test_df, policy_engine, credential_by_role, a.strict_privilege)

    print("\nHOLDOUT (confirmation -- these two rows only, chosen on validation)")
    print("config              ASR     C-ASR   FPR     FNR     Utility")
    print("-" * 60)
    for label, r in (("shipped 0.85", before), (f"retuned {chosen:.2f}", after)):
        print(f"{label:<18}  {r['asr']:.3f}   {r['c_asr']:.3f}   "
              f"{r['fpr']:.3f}   {r['fnr']:.3f}   {r['utility']:.3f}")

    if a.out:
        payload = {
            "model_dir": a.model_dir,
            "block_risk_threshold": chosen,
            "block_combo_risk_threshold": max(0.05, chosen - COMBO_OFFSET),
            "selected_on": "validation",
            "selection_rule": f"min ASR subject to utility >= {a.min_utility}",
            "validation_sweep": sweep,
            "holdout_confirmation": {"shipped": before, "retuned": after},
        }
        Path(a.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
