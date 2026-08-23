"""Track B cycle v2 -> v3, using the real evasions found by
scripts/find_evasions.py (dilution-in-benign-filler attacks that evaded v2)."""
import json
import logging
import sys

import os
from pathlib import Path
import tempfile

# Repo root, resolved from this file rather than hardcoded: scripts/x.py -> up 1.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Dataset CSV is not in the repo; override with SECUREAGENTNET_CSV.
DEFAULT_CSV = REPO_ROOT / "data" / "consolidated_dataset.csv"
# Scratch dir for intermediate run artifacts (was /tmp on macOS).
RUN_DIR = Path(os.environ.get("SECUREAGENTNET_RUN_DIR", Path(tempfile.gettempdir()) / "secureagentnet_run"))
RUN_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_track_b_v3")

import torch

from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device
from secureagentnet.eval.online_retrain import OnlineRetrainingOrchestrator, run_full_track_b_cycle

CSV_PATH = os.environ.get("SECUREAGENTNET_CSV", str(DEFAULT_CSV))
BASE_OUTPUT_DIR = str(REPO_ROOT / "secureagentnet" / "data" / "models")
V2_MODEL_DIR = f"{BASE_OUTPUT_DIR}/v2"

with open(RUN_DIR / "evasions.json", encoding="utf-8") as f:
    evasions = json.load(f)
logger.info("Loaded %d real evasions found against v2", len(evasions))

logger.info("Computing v2 baseline metrics on qualifire holdout...")
device = pick_device()
model = InjectionRiskModel.load(V2_MODEL_DIR, map_location=str(device)).to(device)
tokenizer = load_tokenizer(model.config.model_name)
model.eval()

from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.train import InjectionTextDataset, make_collate_fn, evaluate as eval_fn
from torch.utils.data import DataLoader

splits = dl.build_splits_from_csv(CSV_PATH)
test_loader = DataLoader(
    InjectionTextDataset(splits["test"], tokenizer, model.config.max_length),
    batch_size=32, shuffle=False, collate_fn=make_collate_fn(tokenizer, model.config.max_length),
)
v2_metrics = eval_fn(model, test_loader, device)
logger.info("v2 baseline: f1=%.4f auc=%.4f acc=%.4f", v2_metrics["f1"], v2_metrics["auc"], v2_metrics["accuracy"])

# Check: does v2 actually still miss these evasions right now? (should be
# yes, since they were found against v2 itself in find_evasions.py)
def score(text):
    enc = tokenizer([text], padding=True, truncation=True, max_length=model.config.max_length, return_tensors="pt")
    with torch.no_grad():
        return float(model.risk_score(enc["input_ids"].to(device), enc["attention_mask"].to(device))[0])

logger.info("Confirming v2 misses these evasions before retraining:")
for e in evasions:
    s = score(e)
    logger.info("  score=%.4f %s: %s", s, "CAUGHT" if s >= 0.5 else "EVADED", e[:80])

orch = OnlineRetrainingOrchestrator(
    evasion_buffer_threshold=len(evasions),
    regression_tolerance=0.02,
    initial_metrics={"f1": v2_metrics["f1"], "auc": v2_metrics["auc"]},
    starting_version="v2",
)
orch.add_evasions(evasions)


def retrain_fn(batch, version):
    return run_full_track_b_cycle(
        csv_path=CSV_PATH, evasion_batch=batch, new_version=version,
        base_output_dir=BASE_OUTPUT_DIR, epochs=2,
    )


logger.info("Triggering Track B v2 -> v3 with %d real evasions...", len(evasions))
outcome = orch.run_track_b(retrain_fn, current_metrics_key="f1")

logger.info("=" * 60)
logger.info("TRACK B OUTCOME: promoted=%s version=%s reason=%s", outcome.promoted, outcome.version, outcome.reason)
logger.info("v2 metrics: %s", v2_metrics)
logger.info("%s metrics: %s", outcome.version, outcome.metrics)

# Re-check the same evasions against the new model to see if retraining
# actually fixed the dilution weakness.
if outcome.promoted:
    logger.info("Re-checking original evasions against %s...", outcome.version)
    new_model = InjectionRiskModel.load(f"{BASE_OUTPUT_DIR}/{outcome.version}", map_location=str(device)).to(device)
    new_model.eval()

    def new_score(text):
        enc = tokenizer([text], padding=True, truncation=True, max_length=new_model.config.max_length, return_tensors="pt")
        with torch.no_grad():
            return float(new_model.risk_score(enc["input_ids"].to(device), enc["attention_mask"].to(device))[0])

    n_now_caught = 0
    for ev in evasions:
        s = new_score(ev)
        caught = s >= 0.5
        n_now_caught += int(caught)
        logger.info("  %s score=%.4f %s: %s", outcome.version, s, "CAUGHT" if caught else "STILL EVADES", ev[:80])
    logger.info("%d / %d original evasions now caught by %s", n_now_caught, len(evasions), outcome.version)

logger.info("=" * 60)
