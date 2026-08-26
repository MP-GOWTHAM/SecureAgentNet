"""Regression check: does a model still catch the known real evasions?

CI-oriented sibling of compare_evasions_ensemble.py — one model, a
threshold, and an exit code, rather than a comparison table.

This exists because aggregate metrics do not detect the failure it guards
against. Twice in this project a change improved headline numbers while
silently destroying evasion coverage:

  the DistilBERT persona rebalance improved FPR 0.405 -> 0.382 and lost
  every evasion, 7/8 -> 0/8, mean score 0.741 -> 0.042

  adding the Smooth-3 corpus improved AUC 0.8278 -> 0.9168 and FPR
  0.363 -> 0.208 while dropping the standalone model to 5/8

Neither would have been caught by a test on AUC, F1 or FPR.

The evasion set is tracked at secureagentnet/tests/fixtures/evasions.json
so that CI has something to check against -- it was previously read only
from the scratch run directory, which no CI runner ever has, so the check
skipped on every run while appearing to pass.

Note that it is a *fixed historical* set: the deployed combination now
catches all 8 at minimum score 0.759, so this guards against regression
rather than measuring current robustness. Re-run find_evasions.py against
the current model to generate a harder set, then pass it with --evasions.

Usage:
    python scripts/check_evasions.py --model-dir <dir> --min-caught 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device

RUN_DIR = Path(os.environ.get(
    "SECUREAGENTNET_RUN_DIR", Path(tempfile.gettempdir()) / "secureagentnet_run"))
# Tracked, so CI has a set to check. A freshly generated one in the run
# directory wins when present, since that is the harder, current set.
FIXTURE = REPO_ROOT / "secureagentnet" / "tests" / "fixtures" / "evasions.json"
THRESHOLD = 0.5


def default_evasions() -> Path:
    generated = RUN_DIR / "evasions.json"
    return generated if generated.exists() else FIXTURE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--min-caught", type=int, default=None,
                   help="exit non-zero if fewer than this many evasions are caught")
    p.add_argument("--min-mean-score", type=float, default=None,
                   help="exit non-zero if the mean score falls below this (margin, not just count)")
    p.add_argument("--evasions", default=None)
    a = p.parse_args()

    asserting = a.min_caught is not None or a.min_mean_score is not None
    path = Path(a.evasions) if a.evasions else default_evasions()

    if not path.exists() or not (evasions := json.loads(path.read_text(encoding="utf-8"))):
        # Skipping silently while asserting would make this check useless
        # exactly when its input went missing -- the failure it is meant to
        # catch would sail through as a pass.
        if asserting:
            print(f"FAIL: assertions requested but no usable evasion set at {path}")
            raise SystemExit(2)
        print(f"SKIP: no usable evasion set at {path} (run scripts/find_evasions.py first)")
        raise SystemExit(0)

    print(f"evasion set: {path} ({len(evasions)} entries)")

    device = pick_device()
    model = InjectionRiskModel.load(a.model_dir, map_location=str(device)).to(device)
    model.eval()
    tokenizer = load_tokenizer(model.config.model_name)

    scores = []
    for text in evasions:
        enc = tokenizer([text], padding=True, truncation=True,
                        max_length=model.config.max_length, return_tensors="pt")
        with torch.no_grad():
            scores.append(float(model.risk_score(
                enc["input_ids"].to(device), enc["attention_mask"].to(device))[0]))

    caught = sum(s >= THRESHOLD for s in scores)
    mean = sum(scores) / len(scores)
    name = Path(a.model_dir).name
    print(f"\n{name}: {caught}/{len(scores)} evasions caught "
          f"(mean {mean:.3f}, min {min(scores):.3f})")
    for i, (s, t) in enumerate(zip(scores, evasions), 1):
        print(f"  {'OK  ' if s >= THRESHOLD else 'MISS'} {s:.4f}  {t[:64]!r}")

    failures = []
    if a.min_caught is not None and caught < a.min_caught:
        failures.append(f"caught {caught}/{len(scores)}, required >= {a.min_caught}")
    # Margin matters as well as the count: a model scoring every evasion at
    # 0.51 is one small shift away from missing all of them.
    if a.min_mean_score is not None and mean < a.min_mean_score:
        failures.append(f"mean score {mean:.3f} below required {a.min_mean_score}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    if asserting:
        print("\nPASS")


if __name__ == "__main__":
    main()
