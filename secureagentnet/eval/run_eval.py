"""End-to-end evaluation: loads the trained detector + qualifire held-out
test set, assigns each example a role and tool call via
`simulate.agent_env`, runs the fused framework and both single-signal
baselines over the exact same underlying signals, and prints the
comparison table (ASR / C-ASR / FPR / FNR / Utility) the project spec asks
for.

Why qualifire specifically: it's the one split that was never touched
during training (see detector/data_loader.py's module docstring) — using
anything else here would mix train-distribution leakage into what's
supposed to be an honest generalization measurement.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch

from secureagentnet.correlation.fusion import FusionConfig, FusionEngine
from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import DEFAULT_OUTPUT_DIR, pick_device
from secureagentnet.eval.baselines import detection_only_actions, framework_actions, privilege_only_actions
from secureagentnet.eval.metrics import DEFAULT_DETECTION_THRESHOLD, compute_metrics
from secureagentnet.privilege.policy_engine import POLICIES_DIR, PolicyEngine, issue_credential
from secureagentnet.simulate.agent_env import ROLES, run_pipeline

logger = logging.getLogger(__name__)

EVAL_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
CREDENTIAL_TTL_SECONDS = 3600


@torch.no_grad()
def score_texts(model: InjectionRiskModel, tokenizer, texts: list[str], device, max_length: int, batch_size: int = 32) -> list[float]:
    model.eval()
    scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        batch_scores = model.risk_score(input_ids, attention_mask)
        scores.extend(batch_scores.cpu().tolist())
    return scores


def run_evaluation(
    model_dir: str | Path = DEFAULT_OUTPUT_DIR,
    csv_path: str | None = None,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    strict_privilege: bool = False,
    max_examples: int | None = None,
    block_risk_threshold: float | None = None,
) -> dict:
    device = pick_device()
    logger.info("Using device: %s", device)

    model = InjectionRiskModel.load(model_dir, map_location=str(device)).to(device)
    tokenizer = load_tokenizer(model.config.model_name)

    if csv_path:
        splits = dl.build_splits_from_csv(csv_path)
    else:
        splits = dl.build_splits()
    test_df = splits["test"]
    if max_examples is not None and len(test_df) > max_examples:
        test_df = test_df.sample(n=max_examples, random_state=42).reset_index(drop=True)

    logger.info("Scoring %d held-out test examples...", len(test_df))
    risk_scores = score_texts(model, tokenizer, test_df["text"].tolist(), device, model.config.max_length)

    policy_engine = PolicyEngine.from_directory(POLICIES_DIR)
    credential_by_role = {
        role: issue_credential(role, ttl_seconds=CREDENTIAL_TTL_SECONDS, now=EVAL_NOW) for role in ROLES
    }
    # block_risk_threshold defaults to FusionConfig's 0.85, which was set
    # against DistilBERT's uncalibrated scores (they sit near 1.0 for
    # anything it flags). A temperature-calibrated detector puts genuine
    # attacks around 0.7-0.9, so 0.85 blocks far fewer of them; override
    # with the value scripts/tune_block_threshold.py selects on validation.
    fusion_kwargs = {"strict_privilege": strict_privilege}
    if block_risk_threshold is not None:
        fusion_kwargs["block_risk_threshold"] = block_risk_threshold
        fusion_kwargs["block_combo_risk_threshold"] = max(0.05, block_risk_threshold - 0.45)
        logger.info("block_risk_threshold overridden to %.2f", block_risk_threshold)
    fusion_engine = FusionEngine(FusionConfig(**fusion_kwargs))

    results = [
        run_pipeline(
            text=row.text,
            true_label=int(row.label),
            index=idx,
            risk_score=risk_scores[idx],
            policy_engine=policy_engine,
            fusion_engine=fusion_engine,
            credential_by_role=credential_by_role,
            now=EVAL_NOW,
        )
        for idx, row in enumerate(test_df.itertuples())
    ]

    methods = {
        "SecureAgentNet (fusion)": framework_actions(results),
        "Detection-only baseline": detection_only_actions(results, threshold=detection_threshold),
        "Privilege-only baseline": privilege_only_actions(results),
    }

    metrics_by_method = {
        name: compute_metrics(results, actions, detection_threshold=detection_threshold)
        for name, actions in methods.items()
    }
    return metrics_by_method


def print_comparison_table(metrics_by_method: dict) -> None:
    headers = ["Method", "ASR", "C-ASR", "FPR", "FNR", "Utility"]
    rows = []
    for name, m in metrics_by_method.items():
        rows.append([
            name,
            f"{m.asr:.3f}" if m.asr == m.asr else "n/a",
            f"{m.c_asr:.3f}" if m.c_asr == m.c_asr else "n/a",
            f"{m.fpr:.3f}" if m.fpr == m.fpr else "n/a",
            f"{m.fnr:.3f}" if m.fnr == m.fnr else "n/a",
            f"{m.utility_preservation:.3f}" if m.utility_preservation == m.utility_preservation else "n/a",
        ])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))

    any_metrics = next(iter(metrics_by_method.values()))
    print(f"\n(n_attacks={any_metrics.n_attacks}, n_benign={any_metrics.n_benign}, "
          f"n_chained_eligible={any_metrics.n_chained_eligible})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--csv", type=str, default=None, help="Path to consolidated CSV; omit to pull live from HF Hub")
    parser.add_argument("--detection-threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD)
    parser.add_argument("--strict-privilege", action="store_true", help="Fusion engine always blocks out-of-scope calls")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--block-risk-threshold", type=float, default=None,
                        help="override FusionConfig's 0.85, which assumes uncalibrated scores; "
                             "see scripts/tune_block_threshold.py")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    metrics_by_method = run_evaluation(
        model_dir=args.model_dir,
        csv_path=args.csv,
        detection_threshold=args.detection_threshold,
        strict_privilege=args.strict_privilege,
        max_examples=args.max_examples,
        block_risk_threshold=args.block_risk_threshold,
    )
    print_comparison_table(metrics_by_method)


if __name__ == "__main__":
    main()
