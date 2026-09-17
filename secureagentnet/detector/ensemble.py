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
    # Architecture of branch 1 over the byte view.
    #   "multiwidth" -- parallel k=3/5/7 convs, max-over-time (the original)
    #   "dpcnn"      -- deep pyramid: residual blocks with stride-2 pooling
    # The default keeps checkpoints written before this field existed
    # loading unchanged, since the dataclass fills it in.
    char_branch: str = "multiwidth"
    dpcnn_blocks: int = 5
    # Stacking head.
    #   "linear" -- one fixed weight per branch, the same for every input
    #   "gated"  -- a small gating network picks per-input branch weights
    # Default keeps existing checkpoints loading, since their state_dict
    # holds a plain Linear(6, 1) under `meta`.
    meta_kind: str = "linear"
    gate_hidden: int = 16
    # Feed the 768-d pooled representation to the gate as well. Off by
    # default: Stage B fits the head on half of validation, which can be a
    # few hundred rows, and a 768-input gate overfits that badly.
    gate_uses_pooled: bool = False
    # Blend toward uniform: w = (1 - mix)/3 + mix * softmax(gate). With
    # identical branches and fitting data, pure gating (mix 1.0) lost AUC
    # in both protocols -- 0.8132 -> 0.8004 cross-source, 0.9401 -> 0.9185
    # in-domain -- because it routes almost one-hot (per-prompt std 0.42)
    # and gives up the variance reduction averaging buys when branches
    # correlate 0.84-0.86. 0.25 lets the gate tilt but not switch a branch
    # off, and was the only setting within noise of the linear head.
    gate_mix: float = 0.25
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
        # A row that is entirely padding pools to finfo.min, and nan_to_num
        # does NOT catch that -- finfo.min is finite, so the old guard here
        # was inert and the projection overflowed to NaN. Zero those rows
        # explicitly instead. Rows with any valid position are untouched.
        any_valid = (char_mask != 0).any(dim=1, keepdim=True)
        h = torch.where(any_valid, h, torch.zeros_like(h))
        return self.proj(self.dropout(h))


class _DPCNNCharBranch(nn.Module):
    """Branch 1, alternative: a deep pyramid CNN over the same byte view.

    The multi-width branch sees at most 7 bytes at once and then pools over
    time, which makes it a learned character n-gram detector -- order-blind
    beyond the filter width. Measured, that shows up as a +0.225 dilution
    gap and 0.783 recall on semantic jailbreaks, its two worst numbers.

    This keeps the byte input -- that is what carries homoglyph and
    zero-width evidence -- but grows the receptive field geometrically
    instead: each block halves the sequence, so `dpcnn_blocks` stride-2
    stages over 1024 bytes reach whole-sequence context. Compute halves
    with the length, so the depth is close to free.
    """

    def __init__(self, cfg: EnsembleRiskModelConfig):
        super().__init__()
        f = cfg.char_filters
        self.emb = nn.Embedding(CHAR_VOCAB_SIZE, cfg.d_char, padding_idx=PAD_BYTE_ID)
        # Region embedding: one conv to get from d_char into filter space.
        self.region = nn.Conv1d(cfg.d_char, f, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(f, f, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(f, f, kernel_size=3, padding=1),
            )
            for _ in range(cfg.dpcnn_blocks)
        )
        self.proj = nn.Linear(f, BRANCH_DIM)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, char_ids: torch.Tensor, char_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(char_ids).transpose(1, 2)          # (B, d_char, C)
        m = char_mask.unsqueeze(1).to(x.dtype)          # (B, 1, C)
        x = self.region(x * m)

        for i, block in enumerate(self.blocks):
            if i > 0:
                # Downsample first, so block 0 runs at full resolution.
                x = nn.functional.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                # The mask has to follow the pooling or padded positions
                # start counting as real ones further up the pyramid.
                m = nn.functional.max_pool1d(m, kernel_size=3, stride=2, padding=1)
            x = x + block(x)                            # pre-activation residual

        x = x.masked_fill(m == 0, torch.finfo(x.dtype).min)
        h = x.max(dim=2).values
        # Same guard as the other branch, and for the same reason: an
        # all-padding row pools to finfo.min, which is finite, so it has to
        # be zeroed explicitly rather than left to nan_to_num.
        any_valid = (char_mask != 0).any(dim=1, keepdim=True)
        h = torch.where(any_valid, h, torch.zeros_like(h))
        return self.proj(self.dropout(h))


class _GatedMeta(nn.Module):
    """Input-dependent stacking head: a mixture of the three branches.

    The linear head gives each branch one weight for every prompt. Measured,
    the branches do specialise, just not globally: BiLSTM wins every semantic
    family, the transformer wins word-substitution attacks, char-CNN wins
    character corruption. A single weight has to average over all of that.

    Here a small gating network reads the prompt's own evidence -- the three
    branch logits and the three surface features, optionally the pooled
    representation -- and emits a softmax over the branches, so a prompt
    heavy in non-ASCII characters can lean on the char branch while a long
    roleplay prompt leans on another:

        w(x)   = softmax(gate([logits, feats]) / tau)       per prompt
        fused  = scale * sum_i w_i(x) * logit_i  +  feat_proj(feats) + bias

    The gate's last layer is zero-initialised, so before Stage B fits it the
    weights are uniform and the head reduces to the mean of the branches --
    the same starting point as the linear head.
    """

    def __init__(self, cfg: "EnsembleRiskModelConfig"):
        super().__init__()
        in_dim = 6 + (EMBED_DIM if cfg.gate_uses_pooled else 0)
        self.uses_pooled = cfg.gate_uses_pooled
        self.mix = cfg.gate_mix
        self.norm = nn.LayerNorm(in_dim)
        self.hidden = nn.Linear(in_dim, cfg.gate_hidden)
        self.out = nn.Linear(cfg.gate_hidden, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # log(1) = 0: a softmax temperature of 1 to start.
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.feat_proj = nn.Linear(3, 1)
        nn.init.zeros_(self.feat_proj.weight)
        nn.init.zeros_(self.feat_proj.bias)

    def weights(self, logits, feats, pooled=None) -> torch.Tensor:
        """Per-prompt branch weights, shape (B, 3), rows sum to 1."""
        parts = [logits, feats]
        if self.uses_pooled:
            # Detached: the gate chooses between branches, it must not push
            # gradients back into representations it is judging.
            parts.append(pooled.detach())
        z = self.out(torch.nn.functional.gelu(self.hidden(self.norm(torch.cat(parts, 1)))))
        w = torch.softmax(z / self.log_tau.exp(), dim=1)
        return (1 - self.mix) / 3 + self.mix * w

    def forward(self, logits, feats, pooled=None) -> torch.Tensor:
        w = self.weights(logits, feats, pooled)
        mixed = (w * logits).sum(dim=1, keepdim=True)
        return self.scale * mixed + self.feat_proj(feats)


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

        # Only branch 1 is swappable; branches 2 and 3 are fixed, so an
        # A/B on the CNN changes one thing at a time.
        if config.char_branch == "dpcnn":
            self.char_branch = _DPCNNCharBranch(config)
        elif config.char_branch == "multiwidth":
            self.char_branch = _CharCNNBranch(config)
        else:
            raise ValueError(
                f"unknown char_branch {config.char_branch!r}; "
                "expected 'multiwidth' or 'dpcnn'"
            )
        self.lstm_branch = _BiLSTMBranch(config)
        self.tf_branch = _TransformerBranch(config)

        # Per-branch heads: needed both to train each branch on its own
        # objective and to give the stacking head its three inputs.
        self.branch_heads = nn.ModuleList(nn.Linear(BRANCH_DIM, 1) for _ in range(3))
        self.embed_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

        # Stacking head over [logit1, logit2, logit3] + 3 handcrafted
        # features. Kept deliberately small -- a large meta-learner over
        # six inputs would overfit the fitting split.
        if config.meta_kind == "gated":
            self.meta = _GatedMeta(config)
        elif config.meta_kind == "linear":
            self.meta = nn.Linear(6, 1)
            nn.init.zeros_(self.meta.bias)
            with torch.no_grad():
                # Initialise as a plain mean of the branch logits so the model
                # is sensible before the meta head is fitted.
                self.meta.weight.copy_(torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0]]))
        else:
            raise ValueError(f"unknown meta_kind {config.meta_kind!r}; "
                             "expected 'linear' or 'gated'")

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
        logits, pooled, feats = self.branch_logits(input_ids, attention_mask)
        return self.fuse(logits, feats, pooled) / self.log_temperature.exp()

    def fuse(self, logits: torch.Tensor, feats: torch.Tensor,
             pooled: torch.Tensor | None = None) -> torch.Tensor:
        """Stacking head, uncalibrated, shape (batch,). One call site for
        both head kinds so Stage B, Stage C and inference cannot drift."""
        if isinstance(self.meta, _GatedMeta):
            return self.meta(logits, feats, pooled).squeeze(-1)
        return self.meta(torch.cat([logits, feats], dim=1)).squeeze(-1)

    @torch.no_grad()
    def branch_weights(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """How much each branch counted for each prompt, shape (B, 3):
        [char, bilstm, transformer]. A linear head gives the same row for
        every prompt; a gated head gives a different one per prompt."""
        self.eval()
        logits, pooled, feats = self.branch_logits(input_ids, attention_mask)
        if isinstance(self.meta, _GatedMeta):
            return self.meta.weights(logits, feats, pooled)
        w = self.meta.weight[0, :3].abs()
        return (w / w.sum()).expand(logits.shape[0], 3)

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
