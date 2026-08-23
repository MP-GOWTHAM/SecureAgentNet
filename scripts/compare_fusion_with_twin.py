"""Real comparison: 2-signal fusion vs digital-twin-augmented fusion vs
full multi-signal (behavioral + twin) fusion, on the qualifire test set,
using the trained v3 detector.

A FRESH DigitalTwinSandbox is built per example, not shared across the
whole run: each qualifire row is an independent, unrelated example, not
turns within one continuing agent session. An earlier version of this
script shared one sandbox across all 5000 examples, which let
CodeExecTwin's cumulative-CPU-seconds and SupportTwin's cumulative-refund
caps trip almost immediately from unrelated examples piling onto the same
counters, then flag nearly everything for the rest of the run — a
methodology bug, not a real finding about the defense (confirmed by
inspecting the state directly: cumulative CPU hit 6250s against a 60s cap,
cumulative refunds on one order hit $16,660 against a $100 cap). Per-
example sandboxes are the correct scope for this dataset; a real
multi-turn agent session would legitimately want one shared sandbox across
that session's own turns.
"""
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
logger = logging.getLogger("compare_fusion_with_twin")

import torch
from datetime import datetime, timezone

from secureagentnet.correlation.fusion import FusionEngine
from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device
from secureagentnet.eval.baselines import framework_actions
from secureagentnet.eval.metrics import compute_metrics
from secureagentnet.privilege.policy_engine import POLICIES_DIR, PolicyEngine, issue_credential
from secureagentnet.simulate.agent_env import ROLES, run_pipeline
from secureagentnet.simulate.behavioral_anomaly import BehavioralAnomalyDetector
from secureagentnet.simulate.digital_twin import DigitalTwinSandbox

CSV_PATH = os.environ.get("SECUREAGENTNET_CSV", str(DEFAULT_CSV))
V3_DIR = str(REPO_ROOT / "secureagentnet" / "data" / "models" / "v3")
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

device = pick_device()
model = InjectionRiskModel.load(V3_DIR, map_location=str(device)).to(device)
tokenizer = load_tokenizer(model.config.model_name)
model.eval()


def score_fn(text):
    enc = tokenizer([text], padding=True, truncation=True, max_length=model.config.max_length, return_tensors="pt")
    with torch.no_grad():
        return float(model.risk_score(enc["input_ids"].to(device), enc["attention_mask"].to(device))[0])


splits = dl.build_splits_from_csv(CSV_PATH)
test_df = splits["test"]
logger.info("Scoring %d qualifire examples...", len(test_df))
risk_scores = [score_fn(t) for t in test_df["text"].tolist()]

policy_engine = PolicyEngine.from_directory(POLICIES_DIR)
credential_by_role = {role: issue_credential(role, ttl_seconds=3600, now=NOW) for role in ROLES}
behavioral_detector = BehavioralAnomalyDetector()

configs = [
    ("2-signal (original)", False, False),
    ("+digital twin", False, True),
    ("+behavioral +digital twin (full multi-signal)", True, True),
]

for label, use_behavioral, use_twin in configs:
    fusion_engine = FusionEngine()
    results = [
        run_pipeline(
            text=row.text, true_label=int(row.label), index=idx, risk_score=risk_scores[idx],
            policy_engine=policy_engine, fusion_engine=fusion_engine,
            credential_by_role=credential_by_role, now=NOW,
            behavioral_detector=behavioral_detector if use_behavioral else None,
            # fresh sandbox per example -- see module docstring for why
            digital_twin=DigitalTwinSandbox() if use_twin else None,
        )
        for idx, row in enumerate(test_df.itertuples())
    ]
    actions = framework_actions(results)
    m = compute_metrics(results, actions)
    n_twin_unsafe = sum(1 for r in results if r.sandbox_result is not None and not r.sandbox_result.safe)
    logger.info(
        "%s: ASR=%.4f C-ASR=%.4f FPR=%.4f FNR=%.4f Utility=%.4f (n_twin_unsafe=%d)",
        label, m.asr, m.c_asr, m.fpr, m.fnr, m.utility_preservation, n_twin_unsafe,
    )
