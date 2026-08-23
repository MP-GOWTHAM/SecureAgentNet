"""Real comparison across all six wiring configurations, on the qualifire
test set, using the trained v3 detector: 2-signal baseline, each extra
signal added independently, and the full stack together. Fresh
DigitalTwinSandbox per example (see compare_fusion_with_twin.py for why).
Also reports memory-write outcomes for the one reachable save_note
scenario, since that's a separate decision axis from ASR/C-ASR/Utility.
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
logger = logging.getLogger("compare_fusion_full")

import torch
from datetime import datetime, timezone

from secureagentnet.correlation.fusion import FusionEngine
from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device
from secureagentnet.eval.baselines import framework_actions
from secureagentnet.eval.metrics import compute_metrics
from secureagentnet.privilege.memory_protection import MemoryProtectionLayer, MemoryWriteOutcome
from secureagentnet.privilege.policy_engine import POLICIES_DIR, PolicyEngine, issue_credential
from secureagentnet.provenance.tracker import ProvenanceTracker
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
memory_layer = MemoryProtectionLayer()

configs = [
    ("2-signal (original)", False, False, False),
    ("+provenance only", False, False, True),
    ("+behavioral +twin (prior best)", True, True, False),
    ("+behavioral +twin +provenance (full stack)", True, True, True),
]

for label, use_behavioral, use_twin, use_provenance in configs:
    fusion_engine = FusionEngine()
    provenance_tracker = ProvenanceTracker() if use_provenance else None
    results = [
        run_pipeline(
            text=row.text, true_label=int(row.label), index=idx, risk_score=risk_scores[idx],
            policy_engine=policy_engine, fusion_engine=fusion_engine,
            credential_by_role=credential_by_role, now=NOW,
            behavioral_detector=behavioral_detector if use_behavioral else None,
            digital_twin=DigitalTwinSandbox() if use_twin else None,
            provenance_tracker=provenance_tracker,
            memory_protection=memory_layer,
        )
        for idx, row in enumerate(test_df.itertuples())
    ]
    actions = framework_actions(results)
    m = compute_metrics(results, actions)

    mem_outcomes = [r.memory_write_decision.outcome for r in results if r.memory_write_decision is not None]
    mem_summary = {o.value: mem_outcomes.count(o) for o in MemoryWriteOutcome} if mem_outcomes else {}

    logger.info(
        "%s: ASR=%.4f C-ASR=%.4f FPR=%.4f FNR=%.4f Utility=%.4f | memory_writes n=%d %s",
        label, m.asr, m.c_asr, m.fpr, m.fnr, m.utility_preservation, len(mem_outcomes), mem_summary,
    )
