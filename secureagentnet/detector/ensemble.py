"""Custom from-scratch ensemble detector (phase 2).

A drop-in replacement for `InjectionRiskModel` that uses no pretrained
weights. Three branches with deliberately different inductive biases are
combined by a stacking head:

    (1) char-CNN over a reconstructed byte view  -> counters obfuscation
    (2) BiLSTM with additive attention pooling   -> counters filler dilution
    (3) transformer encoder trained from scratch -> lexical / semantic load

Why these three, specifically: red-teaming the DistilBERT model produced
eight real evasions that cluster into two architectural causes. Mean
pooling averages a short injection across a long benign context (the
surviving evasion still scores 0.0721), and WordPiece shatters obfuscated
text such as "Thank. you. for." into unrecognizable subwords. An ensemble
of three *similar* models would inherit both weaknesses; the value here is
that the branches fail differently.

**No branch uses mean pooling.** Mean pooling is the identified root cause
of the dilution evasion, so it is eliminated by construction: branch (1)
uses max-over-time, branches (2) and (3) use masked additive attention.

Interface contract (identical to InjectionRiskModel, so nothing downstream
changes -- fusion, FAISS memory, red-team loop, eval harness, web app):

    forward(input_ids, attention_mask) -> logits, shape (batch,)
    risk_score(input_ids, attention_mask) -> sigmoid of the above
    embed(input_ids, attention_mask) -> pooled vector, dimension 768
    save(dir) / load(dir) -> config.json + model.pt

The 768 embed dimension is not negotiable: `AttackMemoryIndex(dim=768)` is
hardcoded, so a change in branch widths that altered the concatenated size
would silently corrupt the FAISS index rather than raise. The three branch
vectors are therefore each 256-d and concatenate to exactly 768.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .custom_tokenizer import CHAR_VOCAB_SIZE, PAD_BYTE_ID

BRANCH_DIM = 256
EMBED_DIM = 3 * BRANCH_DIM  # 768 -- must match AttackMemoryIndex(dim=768)

# ASCII punctuation, used for the `punctuation density` stacking feature.
# Period-separated obfuscation ("Thank. you. for.") spikes this.
_PUNCT_BYTES = sorted(bytes(b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))


@dataclass
class EnsembleRiskModelConfig:
    vocab_size: int = 16_000
    max_length: int = 256
    char_max_length: int = 1024
    max_token_bytes: int = 16
    d_bpe: int = 256
    d_char: int = 96
    char_filters: int = 128
    lstm_hidden: int = 128
    lstm_layers: int = 2
    n_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    # Path to the tokenizer directory. Named `model_name` so existing call
    # sites -- `load_tokenizer(model.config.model_name)` -- keep working.
    model_name: str = ""
    # Discriminator read by InjectionRiskModel.load() to dispatch here.
    kind: str = "ensemble"


class _AttentionPool(nn.Module):
    """Masked additive attention pooling.

    This is the direct answer to the dilution evasion. Mean pooling divides
    by the token count, so a ten-token injection inside a 250-token benign
    document contributes ~4% of the pooled vector and is averaged into
    noise. Attention can place most of its mass on those ten tokens
    instead, so the injected span survives pooling.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D)  mask: (B, T) with 1 for real tokens
        scores = self.score(torch.tanh(self.proj(h))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (h * weights).sum(dim=1)


class _CharCNNBranch(nn.Module):
    """Branch 1: convolutions over raw bytes, max-over-time pooled.

    Operates on the byte view rather than tokens, so obfuscation that
    splinters a word into many subword pieces -- or into single bytes --
    cannot hide the underlying character n-grams.
    """

    def __init__(self, cfg: EnsembleRiskModelConfig):
        super().__init__()
        self.emb = nn.Embedding(CHAR_VOCAB_SIZE, cfg.d_char, padding_idx=PAD_BYTE_ID)
        self.convs = nn.ModuleList(
            nn.Conv1d(cfg.d_char, cfg.char_filters, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        )
        self.proj = nn.Linear(3 * cfg.char_filters, BRANCH_DIM)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, char_ids: torch.Tensor, char_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(char_ids).transpose(1, 2)  # (B, d_char, C)
        neg_inf = torch.finfo(x.dtype).min
        pooled = []
        for conv in self.convs:
            f = torch.relu(conv(x))  # (B, filters, C)
            f = f.masked_fill(char_mask.unsqueeze(1) == 0, neg_inf)
            pooled.append(f.max(dim=2).values)
        h = torch.cat(pooled, dim=1)
        # A row that is entirely padding produces -inf from the masked max;
        # clamp it back to zero so it cannot poison the batch with NaNs.
        h = torch.nan_to_num(h, neginf=0.0)
        return self.proj(self.dropout(h))


class _BiLSTMBranch(nn.Module):
    """Branch 2: BiLSTM with attention pooling.

    Sequences are packed rather than run over raw padding. Masking the
    pooling alone is not enough here: the *backward* direction starts at
    the end of the sequence, so trailing pad steps propagate into the
    hidden states at real positions. Left unpacked, the same prompt scores
    differently depending on how long the other rows in its batch are --
    unacceptable in a detector whose output gates real requests.
    """

    def __init__(self, cfg: EnsembleRiskModelConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_bpe, padding_idx=cfg.pad_token_id)
        self.lstm = nn.LSTM(
            cfg.d_bpe,
            cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.pool = _AttentionPool(2 * cfg.lstm_hidden)
        self.proj = nn.Linear(2 * cfg.lstm_hidden, BRANCH_DIM)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        # pack_padded_sequence needs CPU lengths, and every length >= 1.
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=input_ids.size(1)
        )
        return self.proj(self.dropout(self.pool(h, attention_mask)))


class _TransformerBranch(nn.Module):
    """Branch 3: small transformer encoder, trained from scratch."""

    def __init__(self, cfg: EnsembleRiskModelConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_bpe, padding_idx=cfg.pad_token_id)
        self.pos = nn.Embedding(cfg.max_length, cfg.d_bpe)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_bpe,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and only
        # warns; disable it explicitly to keep the logs clean.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.pool = _AttentionPool(cfg.d_bpe)
        self.proj = nn.Linear(cfg.d_bpe, BRANCH_DIM)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.emb(input_ids) + self.pos(positions).unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        return self.proj(self.dropout(self.pool(h, attention_mask)))


class EnsembleInjectionRiskModel(nn.Module):
    """Three-branch ensemble with a stacking head. See module docstring."""

    def __init__(self, config: EnsembleRiskModelConfig):
        super().__init__()
        self.config = config

        # id -> raw bytes lookup, carried in the state dict so a checkpoint
        # is self-contained and load() does not need the tokenizer to
        # rebuild the char view.
        self.register_buffer(
            "token_bytes",
            torch.full((config.vocab_size, config.max_token_bytes), PAD_BYTE_ID, dtype=torch.int16),
        )
        self.register_buffer("token_lens", torch.zeros(config.vocab_size, dtype=torch.int16))
        self.register_buffer("punct_bytes", torch.tensor(_PUNCT_BYTES, dtype=torch.int16))

        self.char_branch = _CharCNNBranch(config)
        self.lstm_branch = _BiLSTMBranch(config)
        self.tf_branch = _TransformerBranch(config)

        # Per-branch heads: needed both to train each branch on its own
        # objective and to give the stacking head its three inputs.
        self.branch_heads = nn.ModuleList(nn.Linear(BRANCH_DIM, 1) for _ in range(3))
        self.embed_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

        # Stacking head over [logit1, logit2, logit3] + 3 handcrafted
        # features. Kept deliberately small -- a large meta-learner over
        # six inputs would overfit the fitting split.
        self.meta = nn.Linear(6, 1)
        nn.init.zeros_(self.meta.bias)
        with torch.no_grad():
            # Initialise as a plain mean of the branch logits so the model
            # is sensible before the meta head is fitted.
            self.meta.weight.copy_(torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0]]))

        # Temperature scaling, fitted on validation after training. T=1 is
        # a no-op, so training runs uncalibrated and inference is calibrated.
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ---------------------------------------------------------------- char view

    def set_token_table(self, table: list[list[int]], lengths: list[int]) -> None:
        self.token_bytes.copy_(torch.tensor(table, dtype=torch.int16))
        self.token_lens.copy_(torch.tensor(lengths, dtype=torch.int16))

    def _char_view(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Reconstruct a byte-level view from token ids, inside the model.

        This is what keeps the two-argument forward() contract while still
        giving branch 1 a genuine character-level input: the byte sequence
        is recovered from the ids rather than passed in as a third tensor.

        Token bytes are gathered, invalid positions are stably sorted to
        the end (a vectorised compaction), and the result is truncated to
        `char_max_length`.
        """
        B, T = input_ids.shape
        L = self.config.max_token_bytes
        C = self.config.char_max_length

        raw = self.token_bytes[input_ids].long()  # (B, T, L)
        lens = self.token_lens[input_ids].long()  # (B, T)
        positions = torch.arange(L, device=input_ids.device).view(1, 1, L)
        valid = positions < lens.unsqueeze(-1)
        valid &= attention_mask.bool().unsqueeze(-1)

        raw = raw.reshape(B, T * L)
        valid = valid.reshape(B, T * L)

        # Stable sort on the inverted mask pulls valid bytes to the front
        # while preserving their original order.
        order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)
        raw = raw.gather(1, order)[:, :C]
        keep = valid.gather(1, order)[:, :C]
        raw = raw.masked_fill(~keep, PAD_BYTE_ID)
        return raw, keep.long()

    def _stack_features(self, char_ids: torch.Tensor, char_mask: torch.Tensor) -> torch.Tensor:
        """Cheap surface statistics for the stacking head, computed from the
        byte view so no extra input tensor is required.

        Non-ASCII ratio catches homoglyph and fullwidth substitution;
        punctuation density catches period-separated obfuscation; length
        separates a terse injection from a long diluted one.
        """
        m = char_mask.float()
        n = m.sum(dim=1).clamp(min=1.0)
        non_ascii = (((char_ids >= 128) & (char_ids < PAD_BYTE_ID)).float() * m).sum(dim=1) / n
        is_punct = torch.isin(char_ids, self.punct_bytes.long())
        punct = (is_punct.float() * m).sum(dim=1) / n
        length = torch.log1p(n) / 10.0
        return torch.stack([non_ascii, punct, length], dim=1)

    # ---------------------------------------------------------------- forward

    def branch_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Returns (branch_logits (B,3), pooled (B,768), features (B,3))."""
        char_ids, char_mask = self._char_view(input_ids, attention_mask)

        h1 = self.char_branch(char_ids, char_mask)
        h2 = self.lstm_branch(input_ids, attention_mask)
        h3 = self.tf_branch(input_ids, attention_mask)

        logits = torch.cat(
            [head(h).squeeze(-1).unsqueeze(1) for head, h in zip(self.branch_heads, (h1, h2, h3))],
            dim=1,
        )
        pooled = self.embed_proj(torch.cat([h1, h2, h3], dim=1))
        return logits, pooled, self._stack_features(char_ids, char_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Fused logits, shape (batch,). Temperature is applied here, so it
        is a no-op during training (T=1) and calibrated at inference once
        `fit_temperature` has run."""
        logits, _, feats = self.branch_logits(input_ids, attention_mask)
        fused = self.meta(torch.cat([logits, feats], dim=1)).squeeze(-1)
        return fused / self.log_temperature.exp()

    @torch.no_grad()
    def risk_score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(input_ids, attention_mask))

    @torch.no_grad()
    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """768-d pooled representation for AttackMemoryIndex."""
        self.eval()
        _, pooled, _ = self.branch_logits(input_ids, attention_mask)
        return pooled

    # ---------------------------------------------------------------- io

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, save_dir: str | Path, map_location: str | None = None) -> "EnsembleInjectionRiskModel":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", encoding="utf-8") as f:
            raw = json.load(f)
        raw.pop("kind", None)
        config = EnsembleRiskModelConfig(**raw)
        # The tokenizer lives beside the weights; point config at wherever
        # the checkpoint actually is, so a moved directory still resolves.
        config.model_name = str(save_dir)
        model = cls(config)
        model.load_state_dict(torch.load(save_dir / "model.pt", map_location=map_location))
        return model
