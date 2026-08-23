"""Byte-level BPE tokenizer trained from scratch on the project corpus.

Two decisions here are deliberate and both are load-bearing for the ensemble:

1. **No normalizer.** The tokenizer applies no NFKC, no lowercasing, no
   accent stripping. `distilbert-base-uncased` does all three, which erases
   exactly the homoglyph / fullwidth / zero-width evidence that marks an
   evasion attempt. An attack's disguise is itself evidence that an attack
   is happening, so the tokenizer must preserve it rather than clean it up.

2. **Byte level, with the full 256-byte initial alphabet.** Every possible
   input maps to a token sequence and `<unk>` can never be produced. Text
   the BPE merges have never seen — obfuscated, zero-width-spaced,
   homoglyph-substituted — degrades gracefully into single-byte tokens
   instead of collapsing to one unknown token. That degradation is what
   lets the char-CNN branch see through obfuscation.

The result is wrapped in `PreTrainedTokenizerFast`, so it is callable with
the same signature the rest of the project already uses
(`tokenizer(texts, padding=True, truncation=True, return_tensors="pt")`)
and `detector/train.py`'s dataset and collate function work against it
unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
DEFAULT_VOCAB_SIZE = 16_000

# One byte id past the real 0-255 range, used as the padding value in the
# char view so the char-CNN's embedding can have a real padding_idx.
PAD_BYTE_ID = 256
CHAR_VOCAB_SIZE = 257


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte<->unicode mapping, which `tokenizers`'
    ByteLevel pre-tokenizer uses internally.

    Reproduced here (rather than imported) because we need the *inverse*
    direction to recover raw bytes from token strings, and `tokenizers`
    does not expose it.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


def train_byte_level_bpe(
    texts: Iterable[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    save_dir: str | Path | None = None,
) -> PreTrainedTokenizerFast:
    """Train a byte-level BPE on `texts` and return an HF-compatible tokenizer."""
    backend = Tokenizer(models.BPE(unk_token=None))
    # No normalizer is assigned on purpose -- see module docstring.
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD_TOKEN],
        # Seeding the full byte alphabet is what guarantees no <unk> path.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(texts, trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=PAD_TOKEN,
        model_max_length=1_000_000,  # truncation is passed explicitly by callers
    )
    logger.info("trained byte-level BPE: vocab_size=%d", fast.vocab_size)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fast.save_pretrained(str(save_dir))
        logger.info("saved tokenizer to %s", save_dir)
    return fast


def token_byte_table(tokenizer: PreTrainedTokenizerFast, max_token_bytes: int = 16):
    """Build the id -> raw-bytes lookup the char view is reconstructed from.

    Returns `(table, lengths)` where `table` is a (vocab, max_token_bytes)
    list-of-lists of byte ids padded with PAD_BYTE_ID, and `lengths` gives
    each token's true byte count (0 for specials like <pad>).

    This is what lets the model expose a genuine character-level view while
    keeping the two-argument `forward(input_ids, attention_mask)` contract:
    the byte sequence is recovered from the token ids inside the model
    rather than being passed in as a third tensor.
    """
    decoder = unicode_to_bytes()
    vocab_size = len(tokenizer)
    table = [[PAD_BYTE_ID] * max_token_bytes for _ in range(vocab_size)]
    lengths = [0] * vocab_size

    id_to_token = {i: t for t, i in tokenizer.get_vocab().items()}
    for tid in range(vocab_size):
        tok = id_to_token.get(tid)
        if tok is None or tok == PAD_TOKEN:
            continue
        raw = [decoder[ch] for ch in tok if ch in decoder]
        raw = raw[:max_token_bytes]
        for j, b in enumerate(raw):
            table[tid][j] = b
        lengths[tid] = len(raw)
    return table, lengths


def load_custom_tokenizer(save_dir: str | Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(str(save_dir))


def corpus_iterator(texts: Sequence[str], batch_size: int = 1000):
    """Chunked iterator; `train_from_iterator` is much faster on batches."""
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]
