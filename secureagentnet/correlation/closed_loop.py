"""Closed-loop adaptation (methodology §3).

Two components, both updated only from *confirmed* outcomes (§3.3's rule,
enforced here by construction — neither class exposes an "update from a
guess" path, only `confirm_*` methods that take a verified label):

- `CalibrationLayer`: the base detector stays frozen; this adjusts the
  *decision threshold* applied to its score via an EMA, bounded so it can
  never drift to always-block or always-allow.
- `AttackMemoryIndex`: confirmed-malicious embeddings (the detector's own
  CLS/pooled vector — see `detector/model.py`'s forward pass, which already
  produces a pooled representation before the final linear head) go into a
  FAISS index. New instructions are checked for nearest-neighbor similarity
  against it — once one variant is caught, near-duplicates are flagged
  before the full detector pass, which is also the "Screen" step the
  red-teaming loop (§4.2) reuses.
"""

from __future__ import annotations

import os

# Must be set before faiss and torch have both loaded their native OpenMP
# runtimes in the same process, or importing/using both reliably segfaults
# on macOS (observed directly running this project's own test suite). Set
# here, at the top of the one module that imports faiss, so any entrypoint
# that pulls this module in gets the fix regardless of import order.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class CalibrationConfig:
    initial_threshold: float = 0.5
    decay: float = 0.9
    min_threshold: float = 0.2
    max_threshold: float = 0.8


class CalibrationLayer:
    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self.threshold = self.config.initial_threshold

    def is_flagged(self, risk_score: float) -> bool:
        return risk_score >= self.threshold

    def confirm_outcome(self, risk_score: float, is_malicious: bool) -> float:
        """Adjust the threshold from a *confirmed* label. A confirmed
        malicious example that scored *below* the current threshold (a
        missed evasion) should pull the threshold down so similar future
        scores get caught; a confirmed benign example that scored *above*
        threshold (a false positive) should push it up. The correction
        target is the score itself, not a fixed step — a near-miss barely
        moves the threshold, a wildly-missed evasion moves it more.
        """
        cfg = self.config
        if is_malicious:
            correction = risk_score if risk_score < self.threshold else self.threshold
        else:
            correction = risk_score if risk_score > self.threshold else self.threshold

        new_threshold = self.threshold * cfg.decay + (1 - cfg.decay) * correction
        self.threshold = max(cfg.min_threshold, min(cfg.max_threshold, new_threshold))
        return self.threshold


class AttackMemoryIndex:
    """FAISS flat index over confirmed-malicious embeddings (cosine
    similarity via inner product on L2-normalized vectors).
    """

    def __init__(self, dim: int, similarity_threshold: float = 0.9):
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self._index = faiss.IndexFlatIP(dim)
        self._texts: list[str] = []

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        vecs = vecs.astype("float32")
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vecs / norms

    def add(self, embedding: np.ndarray, text: str) -> None:
        vec = self._normalize(embedding.reshape(1, -1))
        self._index.add(vec)
        self._texts.append(text)

    def query(self, embedding: np.ndarray) -> tuple[float, str | None]:
        """Returns (best_similarity, matched_text). best_similarity is 0.0
        (and matched_text None) if the index is empty.
        """
        if self._index.ntotal == 0:
            return 0.0, None
        vec = self._normalize(embedding.reshape(1, -1))
        similarities, indices = self._index.search(vec, k=1)
        best_sim = float(similarities[0][0])
        best_idx = int(indices[0][0])
        return best_sim, self._texts[best_idx] if best_idx >= 0 else None

    def is_known_variant(self, embedding: np.ndarray) -> bool:
        similarity, _ = self.query(embedding)
        return similarity >= self.similarity_threshold

    def __len__(self) -> int:
        return self._index.ntotal
