"""Tests for the multi-turn corpus builder's pure logic.

The shape matching is load-bearing. Before it existed, turn count and
character length alone separated the new rows' labels at AUC 0.740 --
a detector trained on that learns to count turns rather than to read the
escalation, which is the same failure as the length shortcut the project
already fights.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "build_multiturn_dataset", REPO_ROOT / "scripts" / "build_multiturn_dataset.py")
bm = importlib.util.module_from_spec(_spec)
sys.modules["build_multiturn_dataset"] = bm
_spec.loader.exec_module(bm)


def test_render_keeps_user_turns_in_order():
    out = bm.render(["first question", "second question"])
    assert out == "User: first question\nUser: second question"


def test_render_drops_empty_and_none_turns():
    """The tom-gibbs export writes the literal string 'None' for absent
    assistant content; it must not become a turn."""
    assert bm.render(["real", "", "   ", "None", "none"]) == "User: real"


def test_parse_conv_extracts_only_user_content():
    """Assistant text would put model output -- including refusals, which
    correlate with the attack label -- into the training text."""
    raw = ("[{'role': 'user', 'content': 'ask one'}, "
           "{'role': 'assistant', 'content': 'answer'}, "
           "{'role': 'user', 'content': 'ask two'}]")
    assert bm._parse_conv(raw) == ["ask one", "ask two"]


def test_parse_conv_survives_malformed_input():
    assert bm._parse_conv("not a literal") == []
    assert bm._parse_conv("{'not': 'a list'}") == []


def _conv(n_turns, turn_chars):
    return "\n".join(f"User: {'x' * turn_chars}" for _ in range(n_turns))


def _frame(specs, label):
    return pd.DataFrame({
        "text": [_conv(t, c) for t, c in specs],
        "label": label, "attack_type": "multi_turn",
        "original_attack_type": None, "original_text": None,
        "source_dataset": "hf_csv9", "split": "train",
    }, columns=bm.SCHEMA)


def test_match_shape_equalises_turn_counts():
    attack = _frame([(11, 50)] * 40 + [(9, 50)] * 10, 1)
    benign = _frame([(9, 50)] * 40 + [(11, 50)] * 10, 0)
    a, b = bm.match_shape(attack, benign, seed=0)
    ta = sorted(a["text"].str.count(r"(?m)^User:"))
    tb = sorted(b["text"].str.count(r"(?m)^User:"))
    assert ta == tb, "turn-count histograms must match exactly"


def test_match_shape_destroys_shape_only_separability():
    """The end-to-end property that matters: after matching, turn count
    and length must not predict the label better than chance."""
    # 80/20 skew in opposite directions: strongly separable by shape, but
    # with enough overlap that matching has something to work with.
    attack = _frame([(11, 90)] * 80 + [(9, 40)] * 20, 1)
    benign = _frame([(9, 40)] * 80 + [(11, 90)] * 20, 0)

    def shape_auc(a, b):
        df = pd.concat([a, b], ignore_index=True)
        X = pd.DataFrame({
            "turns": df["text"].str.count(r"(?m)^User:"),
            "chars": df["text"].str.len(),
        }).to_numpy(float)
        y = df["label"].to_numpy()
        return cross_val_score(LogisticRegression(max_iter=1000), X, y,
                               cv=4, scoring="roc_auc").mean()

    before = shape_auc(attack, benign)
    a, b = bm.match_shape(attack, benign, seed=0)
    after = shape_auc(a, b)
    assert before > 0.65, f"fixture should be separable, got {before:.3f}"
    assert after < 0.60, f"matching should remove the cue, got {after:.3f}"


def test_match_shape_returns_equal_counts_per_label():
    attack = _frame([(5, 100)] * 30, 1)
    benign = _frame([(5, 100)] * 80, 0)
    a, b = bm.match_shape(attack, benign, seed=0)
    assert len(a) == len(b) == 30


def test_match_shape_falls_back_when_no_bucket_overlaps():
    """Disjoint shapes cannot be matched; returning the inputs unchanged is
    better than returning nothing, and the caller logs a warning."""
    attack = _frame([(3, 10)] * 5, 1)
    benign = _frame([(40, 900)] * 5, 0)
    a, b = bm.match_shape(attack, benign, seed=0)
    assert len(a) == 5 and len(b) == 5


def test_schema_matches_the_consolidated_csv_contract():
    assert bm.SCHEMA == ["text", "label", "attack_type", "original_attack_type",
                         "original_text", "source_dataset", "split"]


@pytest.mark.parametrize("src", ["hf_csv8", "hf_csv9"])
def test_new_sources_are_registered_as_training_sources(src):
    from secureagentnet.detector import data_loader as dl
    assert src in dl.CSV_SOURCE_MAP
    assert src not in dl.CSV_TEST_SOURCES, "must not displace the qualifire holdout"
