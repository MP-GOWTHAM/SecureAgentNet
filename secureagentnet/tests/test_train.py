"""Tests for the training pipeline's data plumbing and metrics, isolated
from the dataset loader (no HF network calls) and from a full model forward
pass (evaluate() is exercised against a stub model returning fixed logits).
"""

import pandas as pd
import torch

from secureagentnet.detector.train import InjectionTextDataset, evaluate, make_collate_fn


class _StubModel:
    """Returns a caller-supplied logit per row, keyed by input_ids' first
    token id, so evaluate()'s metric computation can be tested without
    running a real transformer forward pass.
    """

    def __init__(self, logit_by_first_token: dict):
        self.logit_by_first_token = logit_by_first_token

    def eval(self):
        pass

    def __call__(self, input_ids, attention_mask):
        return torch.tensor([self.logit_by_first_token[int(ids[0])] for ids in input_ids])


class _StubTokenizer:
    """Maps each distinct text to a unique single-token id sequence, so
    tests can control exactly what logit the stub model returns per row.
    """

    def __init__(self):
        self._ids = {}
        self._next_id = 1

    def __call__(self, texts, padding=True, truncation=True, max_length=32, return_tensors="pt"):
        ids = []
        for t in texts:
            if t not in self._ids:
                self._ids[t] = self._next_id
                self._next_id += 1
            ids.append([self._ids[t]])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 1, dtype=torch.long),
        }


def test_collate_fn_produces_float_labels_and_padded_tensors():
    tokenizer = _StubTokenizer()
    collate = make_collate_fn(tokenizer, max_length=32)
    batch = [("attack text", 1.0), ("benign text", 0.0)]

    out = collate(batch)
    assert out["labels"].dtype == torch.float32
    assert list(out["labels"]) == [1.0, 0.0]
    assert out["input_ids"].shape[0] == 2


def test_injection_text_dataset_exposes_text_and_label_pairs():
    df = pd.DataFrame({"text": ["a", "b"], "label": [1, 0], "category": ["x", "y"], "source": ["s", "s"]})
    ds = InjectionTextDataset(df, tokenizer=None, max_length=32)

    assert len(ds) == 2
    assert ds[0] == ("a", 1.0)
    assert ds[1] == ("b", 0.0)


def test_evaluate_computes_expected_metrics_on_perfectly_separable_scores():
    tokenizer = _StubTokenizer()
    collate = make_collate_fn(tokenizer, max_length=32)

    # 2 benign scored low, 2 attack scored high -> perfect separation at 0.5
    texts = ["benign_1", "benign_2", "attack_1", "attack_2"]
    labels = [0.0, 0.0, 1.0, 1.0]
    batch = list(zip(texts, labels))
    encoded = collate(batch)
    id_to_logit = {int(encoded["input_ids"][i][0]): (-5.0 if labels[i] == 0 else 5.0) for i in range(4)}
    model = _StubModel(id_to_logit)

    class _OneBatchLoader:
        def __iter__(self):
            yield encoded

    metrics = evaluate(model, _OneBatchLoader(), device=torch.device("cpu"))

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc"] == 1.0
