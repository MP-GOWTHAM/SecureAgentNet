"""Tunes the `gated_max` confidence floor, to cut the false-positive cost
of combining detectors.

Plain max inherits close to the union of both members' false positives.
Measured: ensemble_v6_smooth3 alone is FPR 0.208, DistilBERT v3 alone is
0.405, and max(v6, v3) is 0.415 -- the combination throws away almost the
entire FPR advantage of its better member.

The asymmetry this exploits: v3's value is concentrated in its confident
predictions. It scores the canonical short attacks that v6 misses at
0.98-0.99, while its false positives sit across the middle of the range.
Gating its contribution at a high floor should keep the rescues and drop
the noise.

Protocol
    the gate is swept on VALIDATION scores
    coverage is enforced as a hard guardrail on the two behavioural probes
    the chosen gate is measured ONCE on the holdout

The guardrail matters more than the metric here. A gate that lowers FPR
but loses a canonical short attack or a real evasion is not a better
model, it is a differently-broken one -- so any gate failing 8/8 on
either probe is rejected outright rather than traded off.

Usage:
    python scripts/tune_gated_max.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch

from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device
from secureagentnet.eval.run_eval import score_texts

from probe_short_attacks import FILLER, SHORT_ATTACKS, SHORT_BENIGN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger("tune_gated_max")

MODELS = REPO_ROOT / "secureagentnet" / "data" / "models"
THRESHOLD = 0.5


def member_scores(model_dir: Path, texts, device):
    m = InjectionRiskModel.load(str(model_dir), map_location=str(device)).to(device)
    m.eval()
    tk = load_tokenizer(m.config.model_name)
    return np.array(score_texts(m, tk, texts, device, m.config.max_length))


def combine(primary, secondary, gate):
    return np.maximum(primary, np.where(secondary >= gate, secondary, 0.0))


def fpr_fnr(scores, labels):
    pred = scores >= THRESHOLD
    fp = int((pred & (labels == 0)).sum()); tn = int((~pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum()); tp = int((pred & (labels == 1)).sum())
    return fp / max(fp + tn, 1), fn / max(fn + tp, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--primary", default=str(MODELS / "ensemble_v6_smooth3"))
    p.add_argument("--secondary", default=str(MODELS / "v3"))
    p.add_argument("--csv", default=str(REPO_ROOT / "data" / "consolidated_v2.csv"))
    p.add_argument("--out", default=str(REPO_ROOT / "secureagentnet" / "reports" / "gated_max_tuning.json"))
    a = p.parse_args()

    device = pick_device()
    splits = dl.build_splits_from_csv(a.csv)
    val, test = splits["val"], splits["test"]
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()

    evasions = json.loads(
        (Path(tempfile.gettempdir()) / "secureagentnet_run" / "evasions.json").read_text(encoding="utf-8")
    )
    padded = [f"{t} {FILLER}" for t in SHORT_ATTACKS]

    logger.info("scoring members ...")
    sets = {}
    for name, path in (("primary", a.primary), ("secondary", a.secondary)):
        sets[name] = {
            "val": member_scores(Path(path), val["text"].tolist(), device),
            "test": member_scores(Path(path), test["text"].tolist(), device),
            "atk": member_scores(Path(path), SHORT_ATTACKS, device),
            "ben": member_scores(Path(path), SHORT_BENIGN, device),
            "pad": member_scores(Path(path), padded, device),
            "eva": member_scores(Path(path), evasions, device),
        }

    gates = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.01]
    rows = []
    print("\nVALIDATION sweep (gate=0.00 is plain max; gate>1 is primary alone)")
    print(f"{'gate':>6}{'val FPR':>10}{'val FNR':>10}{'short':>8}{'evasions':>10}{'benign FP':>11}{'dil.gap':>10}")
    print("-" * 65)
    for g in gates:
        v = combine(sets["primary"]["val"], sets["secondary"]["val"], g)
        fpr, fnr = fpr_fnr(v, yv)
        atk = combine(sets["primary"]["atk"], sets["secondary"]["atk"], g)
        ben = combine(sets["primary"]["ben"], sets["secondary"]["ben"], g)
        pad = combine(sets["primary"]["pad"], sets["secondary"]["pad"], g)
        eva = combine(sets["primary"]["eva"], sets["secondary"]["eva"], g)
        row = {
            "gate": g, "val_fpr": fpr, "val_fnr": fnr,
            "short": int((atk >= THRESHOLD).sum()), "benign_fp": int((ben >= THRESHOLD).sum()),
            "evasions": int((eva >= THRESHOLD).sum()), "dilution_gap": float(np.mean(pad - atk)),
        }
        rows.append(row)
        print(f"{g:>6.2f}{fpr:>10.4f}{fnr:>10.4f}{row['short']:>6}/8{row['evasions']:>8}/8"
              f"{row['benign_fp']:>9}/4{row['dilution_gap']:>+10.3f}")

    # Guardrail first, then minimise FPR. Coverage is not tradeable.
    eligible = [r for r in rows if r["short"] == len(SHORT_ATTACKS)
                and r["evasions"] == len(evasions) and r["benign_fp"] == 0]
    if not eligible:
        raise SystemExit("no gate preserved full coverage; plain max remains the safe choice")
    best = min(eligible, key=lambda r: r["val_fpr"])
    print(f"\nselection: lowest validation FPR among gates keeping 8/8 short attacks, "
          f"8/8 evasions and 0/4 benign false positives")
    print(f"chosen gate = {best['gate']:.2f}")

    print("\nHOLDOUT (confirmation only)")
    print(f"{'config':<24}{'FPR':>10}{'FNR':>10}")
    print("-" * 44)
    for label, g in (("plain max", 0.0), (f"gated_max {best['gate']:.2f}", best["gate"]),
                     ("primary alone", 1.01)):
        t = combine(sets["primary"]["test"], sets["secondary"]["test"], g)
        fpr, fnr = fpr_fnr(t, yt)
        print(f"{label:<24}{fpr:>10.4f}{fnr:>10.4f}")

    Path(a.out).write_text(json.dumps({"sweep": rows, "chosen": best}, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
