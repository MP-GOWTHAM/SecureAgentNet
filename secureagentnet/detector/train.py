"""Trains InjectionRiskModel on the merged dataset and checkpoints the best
epoch by validation ROC-AUC.

Why ROC-AUC for checkpoint selection rather than val loss or accuracy: this
detector feeds a fusion layer (Phase 4) that will threshold the risk score
at a tunable cutoff, not a fixed 0.5. AUC measures ranking quality across
all thresholds, so it's the metric that actually matches how the score gets
used downstream — a model with worse val loss but better-separated risk
scores is more useful here than one that's merely well-calibrated at 0.5.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from secureagentnet.detector import data_loader as dl
from secureagentnet.detector.model import (
    DEFAULT_MODEL_NAME,
    InjectionRiskModel,
    InjectionRiskModelConfig,
    load_tokenizer,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "injection_detector"


class InjectionTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.float32)
        return enc

    return collate


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: InjectionRiskModel, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        total_loss += loss_fn(logits, labels).item() * len(labels)
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    preds = (scores >= threshold).astype(int)

    # roc_auc_score needs both classes present; guard the degenerate case
    # (e.g. a tiny smoke-test split) rather than letting it raise mid-training.
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")

    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
    }


def train(
    csv_path: str | None = None,
    necent_max_rows: int | None = 30_000,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2e-5,
    seed: int = 42,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_train_rows: int | None = None,
) -> dict:
    torch.manual_seed(seed)
    output_dir = Path(output_dir)

    if csv_path:
        splits = dl.build_splits_from_csv(csv_path, necent_max_rows=necent_max_rows, seed=seed)
    else:
        splits = dl.build_splits(seed=seed)

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    if max_train_rows is not None and len(train_df) > max_train_rows:
        # Smoke-test knob: cap train size for a fast end-to-end run without
        # touching the real hyperparameters, e.g. `--max-train-rows 500`.
        train_df = train_df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)

    tokenizer = load_tokenizer(model_name)
    collate_fn = make_collate_fn(tokenizer, max_length)

    train_loader = DataLoader(
        InjectionTextDataset(train_df, tokenizer, max_length),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        InjectionTextDataset(val_df, tokenizer, max_length),
        batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        InjectionTextDataset(test_df, tokenizer, max_length),
        batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn,
    )

    device = pick_device()
    logger.info("Using device: %s", device)

    model = InjectionRiskModel(InjectionRiskModelConfig(model_name=model_name, max_length=max_length)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_examples = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            n_examples += len(labels)
            if step % 100 == 0:
                logger.info("epoch %d step %d train_loss=%.4f", epoch, step, loss.item())

        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f val_auc=%.4f",
            epoch, running_loss / n_examples, val_metrics["loss"], val_metrics["accuracy"],
            val_metrics["f1"], val_metrics["auc"],
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            model.save(output_dir)
            logger.info("New best val AUC=%.4f (epoch %d) — checkpoint saved to %s", best_auc, epoch, output_dir)

    logger.info("Loading best checkpoint (epoch %d, val AUC=%.4f) for final test evaluation", best_epoch, best_auc)
    best_model = InjectionRiskModel.load(output_dir, map_location=str(device)).to(device)
    test_metrics = evaluate(best_model, test_loader, device)
    logger.info(
        "held-out test (qualifire): acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
        test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"],
        test_metrics["f1"], test_metrics["auc"],
    )

    return {"best_epoch": best_epoch, "best_val_auc": best_auc, "test_metrics": test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to consolidated CSV; omit to pull live from HF Hub")
    parser.add_argument("--necent-max-rows", type=int, default=30_000)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-train-rows", type=int, default=None, help="Cap train set size, for fast smoke tests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(
        csv_path=args.csv,
        necent_max_rows=args.necent_max_rows,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
        max_train_rows=args.max_train_rows,
    )


if __name__ == "__main__":
    main()
