"""Injection risk model: a transformer encoder + single-logit head.

Design decision: this is a custom nn.Module rather than
`AutoModelForSequenceClassification(num_labels=1)`. HF's built-in class picks
its loss function from `problem_type`/`num_labels` in a way that's easy to
misconfigure for this exact use case (num_labels=1 defaults to MSELoss/
regression, not what we want for a 0/1 label). Writing the head explicitly —
one linear layer to a single logit, trained with BCEWithLogitsLoss, sigmoid
applied only at inference to produce the risk score — makes the "continuous
0-1 risk score, not just a binary label" requirement fall directly out of
the architecture instead of needing a calibration step bolted on after.

Swapping the backbone later (e.g. from DistilBERT to a larger encoder) is a
one-line change to `model_name` since everything else is derived from the
backbone's own config (hidden size, tokenizer).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "distilbert-base-uncased"


@dataclass
class InjectionRiskModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 256
    dropout: float = 0.1


class InjectionRiskModel(nn.Module):
    """Encoder backbone + dropout + linear(hidden_size -> 1) risk head."""

    def __init__(self, config: InjectionRiskModelConfig):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config.model_name)
        hidden_size = AutoConfig.from_pretrained(config.model_name).hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(hidden_size, 1)

    def _pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """The dual-purpose representation (methodology §2.2): the same
        pooled vector feeds both the classification head (below) and, via
        `embed()`, the FAISS attack memory index (correlation/closed_loop.py)
        — one forward pass through the backbone serves both purposes, so
        the memory-similarity lookup adds no extra encoder cost.

        Mean-pooled over non-padded tokens rather than only the
        [CLS]/first-token embedding: injection text is often a short
        instruction buried inside a longer benign-looking tool output, so
        pooling across the whole sequence is less likely to miss it than
        relying on a single summary token.
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch,). Apply `torch.sigmoid` for the
        0-1 risk score — kept separate so training can use
        BCEWithLogitsLoss directly on logits (more numerically stable than
        computing sigmoid then BCE).
        """
        pooled = self._pool(input_ids, attention_mask)
        logits = self.head(self.dropout(pooled)).squeeze(-1)
        return logits

    @torch.no_grad()
    def risk_score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(input_ids, attention_mask))

    @torch.no_grad()
    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """The pooled representation, exposed standalone for
        `AttackMemoryIndex` — same vector `forward()` feeds into the risk
        head, just without the final linear layer applied.
        """
        self.eval()
        return self._pool(input_ids, attention_mask)

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, save_dir: str | Path, map_location: str | None = None):
        """Load a checkpoint from `save_dir`.

        Dispatches on the config's `kind` field so a phase-2 ensemble
        checkpoint loads through the same call site as a DistilBERT one.
        Checkpoints written before `kind` existed have no such key and are
        treated as DistilBERT, so old checkpoints keep loading unchanged.
        """
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", encoding="utf-8") as f:
            raw = json.load(f)
        if raw.get("kind") == "ensemble":
            from .ensemble import EnsembleInjectionRiskModel

            return EnsembleInjectionRiskModel.load(save_dir, map_location=map_location)
        if raw.get("kind") == "combined":
            from .combined import CombinedRiskModel

            return CombinedRiskModel.load(save_dir, map_location=map_location)
        config = InjectionRiskModelConfig(**raw)
        model = cls(config)
        state_dict = torch.load(save_dir / "model.pt", map_location=map_location)
        model.load_state_dict(state_dict)
        return model


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)
