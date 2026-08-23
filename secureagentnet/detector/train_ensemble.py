"""Training driver for the from-scratch ensemble detector.

Three stages, deliberately separated so the stacking head is never fitted
on data the branches were trained on:

    Stage A  train the three branches on `train` (each on its own BCE loss)
    Stage B  fit the stacking head on one half of `val`
    Stage C  fit temperature scaling on the other half of `val`

Stage B is the part that is easy to get wrong. Fitting the meta-learner on
in-sample branch predictions leaks and inflates the result in exactly the
way a pooled train/test split does -- the branches are near-perfect on
their own training data, so the meta-learner would learn to trust them far
more than it should. Holding out a slice of `val` is the cheap defence;
K-fold out-of-fold predictions are the stronger one and are the natural
upgrade from here (see `--help` note on --meta-fraction).

The qualifire holdout is never touched by any stage.

Usage:
    python -m secureagentnet.detector.train_ensemble --epochs 20
    python -m secureagentnet.detector.train_ensemble --csv data/consolidated_dataset.csv

KNOWN ISSUE (Windows): the process exits with 0xC0000409
(STATUS_STACK_BUFFER_OVERRUN) *after* a fully successful run. Every
artifact -- model.pt, config.json, the tokenizer and metrics.json -- is
written and flushed before it happens, and the saved checkpoint loads and
scores correctly (see tests/test_ensemble.py).

It is a native fault in DLL unload, not a training defect. `-X
faulthandler` puts the faulting frame inside `os._exit` itself, i.e. after
the interpreter is gone, among the 172 loaded extension modules -- torch,
faiss, tokenizers, scipy and sklearn each ship their own native runtime.
Every component was probed in isolation at production dimensions (char
view, stacking features, full forward, backward with optimizer and LR
schedule, save/load, the trained tokenizer over 2k real rows,
build_splits_from_csv) and each exits 0 alone; only the combination
faults. Python-level workarounds do not help: os._exit and
kernel32.TerminateProcess were both tried and both still fault, because
the stack-cookie check fires before either runs.

Practical consequence: **judge a run by metrics.json and the log tail, not
by the exit code.** Running under `python -X faulthandler` happens to exit
0, which is useful for CI but is a side effect rather than a fix.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import data_loader as dl
from .augment import (
    length_invariance_augment,
    oversample_long_benign,
    persona_framed_benign_augment,
    rebalance_persona_labels,
)
from .custom_tokenizer import (
    DEFAULT_VOCAB_SIZE,
    corpus_iterator,
    load_custom_tokenizer,
    token_byte_table,
    train_byte_level_bpe,
)
from .ensemble import EnsembleInjectionRiskModel, EnsembleRiskModelConfig
from .train import InjectionTextDataset, evaluate, make_collate_fn, pick_device

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "ensemble_v1"


@torch.no_grad()
def collect_branch_outputs(model, loader, device):
    """Run the branches over a loader and return (logits, feats, labels)."""
    model.eval()
    L, F, Y = [], [], []
    for batch in loader:
        logits, _, feats = model.branch_logits(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        L.append(logits.float().cpu())
        F.append(feats.float().cpu())
        Y.append(batch["labels"].float().cpu())
    return torch.cat(L), torch.cat(F), torch.cat(Y)


def fit_meta(model, logits, feats, labels, steps: int = 400, lr: float = 0.05) -> None:
    """Stage B: fit the 6-input stacking head. Plain BCE, no class weighting
    -- the head's job is to combine, and re-weighting here would bias the
    probabilities the temperature stage then tries to calibrate."""
    x = torch.cat([logits, feats], dim=1)
    head = nn.Linear(6, 1)
    head.load_state_dict(model.meta.state_dict())
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(head(x).squeeze(-1), labels)
        loss.backward()
        opt.step()
    model.meta.load_state_dict(head.state_dict())
    w = head.weight.detach().squeeze(0)
    logger.info(
        "Stage B: meta weights char=%.3f lstm=%.3f tf=%.3f | non_ascii=%.3f punct=%.3f len=%.3f (bias %.3f), loss=%.4f",
        *w.tolist(), head.bias.item(), loss.item(),
    )


def fit_temperature(model, logits, feats, labels, steps: int = 300, lr: float = 0.02) -> None:
    """Stage C: single-parameter temperature scaling.

    Attacks the calibration half of the DistilBERT model's precision
    problem: its scores rank well (recall 0.91) but the 0.5 cut is
    arbitrary because nothing ever calibrated them."""
    with torch.no_grad():
        fused = model.meta(torch.cat([logits, feats], dim=1)).squeeze(-1)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(fused / log_t.exp(), labels)
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.log_temperature.copy_(log_t.detach())
    logger.info("Stage C: temperature=%.4f (loss %.4f)", float(log_t.detach().exp()), loss.item())


def train(
    csv_path: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_length: int = 256,
    char_max_length: int = 1024,
    necent_max_rows: int = 30_000,
    meta_fraction: float = 0.5,
    limit_train: int | None = None,
    seed: int = 42,
    augment: bool = False,
    n_long_benign: int = 8000,
    n_diluted_attacks: int = 4000,
    oversample_long: bool = False,
    long_benign_fraction: float = 0.46,
    n_persona_benign: int = 0,
    rebalance_persona: bool = False,
    persona_attack_rate: float = 0.535,
) -> dict:
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = (
        dl.build_splits_from_csv(csv_path, necent_max_rows=necent_max_rows)
        if csv_path
        else dl.build_splits()
    )
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    if limit_train:
        train_df = train_df.sample(n=min(limit_train, len(train_df)), random_state=seed)
    logger.info("train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # Applied before the tokenizer is trained so its merges see the same
    # text distribution the branches will. Train split only -- val and the
    # holdout stay untouched, so metrics remain comparable.
    if augment:
        train_df = length_invariance_augment(
            train_df, seed=seed,
            n_long_benign=n_long_benign, n_diluted_attacks=n_diluted_attacks,
        )
    if oversample_long:
        train_df = oversample_long_benign(train_df, seed=seed, target_fraction=long_benign_fraction)
    if rebalance_persona:
        train_df = rebalance_persona_labels(
            train_df, seed=seed, target_attack_rate=persona_attack_rate)
    if n_persona_benign:
        train_df = persona_framed_benign_augment(train_df, seed=seed, n_rows=n_persona_benign)

    # --- tokenizer: trained on the TRAIN split only -----------------------
    tok_file = output_dir / "tokenizer.json"
    if tok_file.exists():
        tokenizer = load_custom_tokenizer(output_dir)
        # Loud on purpose. Reusing a tokenizer trained on a *different*
        # corpus silently changes results: a sweep over training-set sizes
        # that writes every point to one output dir will train the
        # tokenizer once on the first point and reuse it for the rest,
        # confounding the comparison. Observed directly -- a 3k-row point
        # scored 0.8159 on a reused 1k-row tokenizer against 0.8060 with
        # its own. Pass a separate --output-dir per run, or delete
        # tokenizer.json to force a retrain.
        logger.warning(
            "REUSING existing tokenizer from %s (vocab=%d) -- it was NOT retrained on "
            "this run's training data. Use a fresh --output-dir if that is not intended.",
            output_dir, len(tokenizer),
        )
    else:
        logger.info("training byte-level BPE on %d train texts...", len(train_df))
        tokenizer = train_byte_level_bpe(
            corpus_iterator(train_df["text"].tolist()),
            vocab_size=vocab_size,
            save_dir=output_dir,
        )

    device = pick_device()
    logger.info("Using device: %s", device)

    config = EnsembleRiskModelConfig(
        vocab_size=len(tokenizer),
        max_length=max_length,
        char_max_length=char_max_length,
        pad_token_id=tokenizer.pad_token_id or 0,
        model_name=str(output_dir),
    )
    model = EnsembleInjectionRiskModel(config)
    table, lengths = token_byte_table(tokenizer, config.max_token_bytes)
    model.set_token_table(table, lengths)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("ensemble parameters: %.2fM", n_params / 1e6)

    collate = make_collate_fn(tokenizer, max_length)
    mk = lambda df, shuffle: DataLoader(  # noqa: E731
        InjectionTextDataset(df, tokenizer, max_length),
        batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
    )
    train_loader, val_loader, test_loader = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    # Class weighting for the 33/67 imbalance in the merged corpus.
    n_pos = float(train_df["label"].sum())
    n_neg = float(len(train_df) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    logger.info("pos_weight=%.3f (pos=%d neg=%d)", float(pos_weight), int(n_pos), int(n_neg))

    branch_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(epochs * len(train_loader), 1), pct_start=0.1
    )

    # ---------------- Stage A: train the branches -------------------------
    best_auc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _, _ = model.branch_logits(input_ids, attention_mask)
            # Each branch is trained on its own objective. The stacking head
            # is NOT in this loss -- it is fitted in Stage B on held-out data.
            loss = sum(branch_loss(logits[:, i], labels) for i in range(3)) / 3.0

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if step % 200 == 0:
                logger.info("epoch %d step %d train_loss=%.4f", epoch, step, loss.item())

        with torch.no_grad():
            val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f val_auc=%.4f",
            epoch, running / max(len(train_loader), 1),
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1"], val_metrics["auc"],
        )
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info("New best val AUC=%.4f (epoch %d)", best_auc, epoch)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # ------------- Stages B and C: stacking, then calibration -------------
    # Two disjoint halves of val: fitting the head and calibrating it on the
    # same rows would let the temperature absorb the head's overfit.
    val_shuffled = val_df.sample(frac=1.0, random_state=seed)
    n_meta = int(len(val_shuffled) * meta_fraction)
    meta_df, calib_df = val_shuffled.iloc[:n_meta], val_shuffled.iloc[n_meta:]
    logger.info("Stage B/C split: meta=%d calibration=%d", len(meta_df), len(calib_df))

    ml, mf, my = collect_branch_outputs(model.cpu(), mk(meta_df, False), torch.device("cpu"))
    fit_meta(model, ml, mf, my)
    cl, cf, cy = collect_branch_outputs(model, mk(calib_df, False), torch.device("cpu"))
    fit_temperature(model, cl, cf, cy)
    model.to(device)

    # ---------------- final evaluation on the untouched holdout -----------
    with torch.no_grad():
        test_metrics = evaluate(model, test_loader, device)
    logger.info(
        "held-out test: acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
        test_metrics["accuracy"], test_metrics["precision"],
        test_metrics["recall"], test_metrics["f1"], test_metrics["auc"],
    )

    model.cpu().save(output_dir)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val_auc_best": best_auc, "test": test_metrics, "n_params": n_params}, f, indent=2)
    logger.info("saved ensemble to %s", output_dir)
    return test_metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--char-max-length", type=int, default=1024)
    p.add_argument("--necent-max-rows", type=int, default=30_000)
    p.add_argument("--meta-fraction", type=float, default=0.5,
                   help="fraction of val used to fit the stacking head; the rest calibrates")
    p.add_argument("--augment", action="store_true",
                   help="length-invariance augmentation: decorrelates text length from label, "
                        "which is a spurious cue in both the corpus and the benchmark")
    p.add_argument("--n-long-benign", type=int, default=8000)
    p.add_argument("--n-diluted-attacks", type=int, default=4000)
    p.add_argument("--oversample-long", action="store_true",
                   help="oversample REAL long benign rows; closes the representation gap "
                        "(benign >400 chars is 14%% of training but 46%% of the benchmark)")
    p.add_argument("--long-benign-fraction", type=float, default=0.46)
    p.add_argument("--n-persona-benign", type=int, default=0,
                   help="benign rows with persona/roleplay framing; targets the 60%% of false "
                        "positives caused by the train/benchmark disagreement over framing")
    p.add_argument("--rebalance-persona", action="store_true",
                   help="drop framing-only persona attack rows until their attack rate matches "
                        "the benchmark's convention (rows with real override language are kept)")
    p.add_argument("--persona-attack-rate", type=float, default=0.535,
                   help="target attack rate among persona-framed rows (benchmark's own 53.5%%)")
    p.add_argument("--limit-train", type=int, default=None, help="smoke-test on a subset")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(
        csv_path=a.csv, output_dir=a.output_dir, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, vocab_size=a.vocab_size, max_length=a.max_length,
        char_max_length=a.char_max_length, necent_max_rows=a.necent_max_rows,
        meta_fraction=a.meta_fraction, limit_train=a.limit_train, seed=a.seed,
        augment=a.augment, n_long_benign=a.n_long_benign, n_diluted_attacks=a.n_diluted_attacks,
        oversample_long=a.oversample_long, long_benign_fraction=a.long_benign_fraction,
        n_persona_benign=a.n_persona_benign,
        rebalance_persona=a.rebalance_persona, persona_attack_rate=a.persona_attack_rate,
    )


if __name__ == "__main__":
    main()
