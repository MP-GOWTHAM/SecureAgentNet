"""Tests for the 2026 dataset additions: the Smooth-3 normalizer and
`DatasetSpec.balance_labels`.

Both exist because of measured failures, and the tests pin the behaviour
that made them necessary rather than just the happy path:

  Smooth-3's label is a multi-label LIST (`['JAILBREAK',
  'INSTRUCTION_OVERRIDE']`), which `_normalize_label` cannot read -- it
  handles scalars only, so without a custom normalizer every row would be
  dropped as unrecognised.

  `balance_labels` overrides the stratified row cap. The default cap
  faithfully preserves a source's own class ratio, which is right for
  mindgard (100% attack is real information) and wrong for imoxto: taken
  at its native 24.8% positive it pushed corpus pos_weight 1.19 -> 1.61
  and cost 0.033 held-out AUC.
"""

import pandas as pd
import pytest

from secureagentnet.detector.data_loader import (
    CSV_SOURCE_MAP,
    DATASET_SPECS,
    DatasetSpec,
    _normalize_smooth3,
    load_and_normalize,
)


# ------------------------------------------------------- smooth3 normalizer


def _smooth3_frame():
    return pd.DataFrame({
        "text": ["benign one", "override attack", "roleplay attack", "exfil attack", "benign two"],
        "labels": [
            ["BENIGN"],
            ["JAILBREAK", "INSTRUCTION_OVERRIDE"],
            ["JAILBREAK", "ROLE_HIJACK"],
            ["JAILBREAK", "ROLE_HIJACK", "DATA_EXFILTRATION"],
            ["BENIGN"],
        ],
    })


def test_list_labels_collapse_to_binary():
    out = _normalize_smooth3(_smooth3_frame())
    assert out["label"].tolist() == [0, 1, 1, 1, 0]


def test_attack_taxonomy_is_preserved_in_category():
    """The multi-label detail is the dataset's value -- collapsing to a
    binary label must not throw it away."""
    out = _normalize_smooth3(_smooth3_frame())
    assert out.loc[3, "category"] == "JAILBREAK|ROLE_HIJACK|DATA_EXFILTRATION"
    assert out.loc[0, "category"] == "BENIGN"


def test_scalar_labels_still_work():
    """Defensive: upstream could flatten the column to a plain string."""
    df = pd.DataFrame({"text": ["a", "b"], "labels": ["BENIGN", "JAILBREAK"]})
    assert _normalize_smooth3(df)["label"].tolist() == [0, 1]


def test_normalizer_is_case_insensitive():
    df = pd.DataFrame({"text": ["a"], "labels": [["jailbreak", "role_hijack"]]})
    assert _normalize_smooth3(df)["label"].tolist() == [1]


# ---------------------------------------------------------- balance_labels


class _FakeDataset:
    """Stands in for a HF dataset so these tests need no network."""

    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df.copy()


@pytest.fixture
def skewed(monkeypatch):
    """1,000 rows at 20% positive -- imoxto's shape, exaggerated."""
    df = pd.DataFrame({
        "text": [f"row {i}" for i in range(1000)],
        "label": [1] * 200 + [0] * 800,
    })
    monkeypatch.setattr(
        "secureagentnet.detector.data_loader._load_hf_dataset",
        lambda spec, token=None: _FakeDataset(df),
    )
    return df


def test_balance_labels_draws_the_requested_ratio(skewed):
    spec = DatasetSpec(name="t", hf_id="x", role="train", max_rows=200, balance_labels=0.5)
    out = load_and_normalize(spec)
    assert len(out) == 200
    assert out["label"].mean() == pytest.approx(0.5)


def test_default_cap_preserves_the_source_ratio(skewed):
    """Without balance_labels the stratified cap must keep the 20% skew --
    that is the correct default for a source like mindgard."""
    spec = DatasetSpec(name="t", hf_id="x", role="train", max_rows=200)
    out = load_and_normalize(spec)
    assert out["label"].mean() == pytest.approx(0.2, abs=0.02)


def test_balance_labels_cannot_invent_rows_it_does_not_have(skewed):
    """Only 200 positives exist; asking for 500 must clamp, not duplicate --
    oversampling by duplication was measured to cause memorisation."""
    spec = DatasetSpec(name="t", hf_id="x", role="train", max_rows=1000, balance_labels=0.5)
    out = load_and_normalize(spec)
    assert out["label"].sum() == 200
    assert len(out) == 1000 - (500 - 200)


def test_balance_labels_is_inert_without_max_rows(skewed):
    spec = DatasetSpec(name="t", hf_id="x", role="train", balance_labels=0.5)
    out = load_and_normalize(spec)
    assert len(out) == 1000
    assert out["label"].mean() == pytest.approx(0.2)


# ------------------------------------------------------------ registration


def test_new_sources_are_registered():
    names = {s.name for s in DATASET_SPECS}
    assert {"smooth3", "jayavibhav", "imoxto"} <= names


def test_every_spec_has_a_csv_tag():
    """build_splits_from_csv routes by tag; a spec without one silently
    vanishes from any CSV-backed run."""
    assert {s.name for s in DATASET_SPECS} <= set(CSV_SOURCE_MAP.values())


def test_only_qualifire_is_held_out():
    assert [s.name for s in DATASET_SPECS if s.role == "test"] == ["qualifire_holdout"]


def test_imoxto_is_capped_and_balanced():
    """Both settings are load-bearing: uncapped it is 74% of the corpus, and
    at native skew it costs AUC."""
    spec = next(s for s in DATASET_SPECS if s.name == "imoxto")
    assert spec.max_rows == 120_000
    assert spec.balance_labels == 0.5
