"""Add multi-turn and crescendo attacks to the corpus.

Why this script exists
----------------------
The corpus had 21 rows across `multi_turn` and `crescendo`, and they were
stored in the wrong shape: each TURN was its own row, labelled attack.

    "Let's start simple - what are some common household chemicals?"   label=1
    "Which of those should never be mixed together for safety?"        label=1

In isolation those are ordinary questions. A crescendo attack is defined
by the escalation across turns -- no single turn is harmful -- so storing
turns independently destroys the only signal that identifies it, and
teaches the detector that innocuous questions are attacks. Adding more
rows in that shape would make the problem worse, not better.

Here a conversation is one row, so the trajectory is visible.

Sources (both MIT licensed)
---------------------------
SafeMTData/SafeMTData, Attack_600
    600 genuine crescendo conversations, 5 user turns each: a benign
    opening that walks toward a harmful goal. `plain_query` holds the
    goal the escalation is steering at.

tom-gibbs/multi-turn_jailbreak_attack_datasets
    A cipher-substitution multi-turn family. It ships harmful AND
    completely-benign conversations that share an identical template,
    which is exactly what is needed: without benign multi-turn examples
    the model would simply learn "long conversation means attack", the
    same category of shortcut as the per-turn labelling above.

Design decisions worth knowing
------------------------------
Only USER turns are kept. The detector judges a prompt, not a
transcript, and importing assistant responses would put model output --
including refusals, which correlate with the attack label -- into the
training text.

Turns are joined with an explicit "User:" prefix per line, matching the
turn marker already present in the corpus (hf_csv3 uses "User: ...").

Usage:
    python scripts/build_multiturn_dataset.py            # writes the fragment
    python scripts/build_multiturn_dataset.py --merge    # + merges into a corpus
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_multiturn")

SCHEMA = ["text", "label", "attack_type", "original_attack_type",
          "original_text", "source_dataset", "split"]

SAFEMT = "SafeMTData/SafeMTData"
GIBBS = "tom-gibbs/multi-turn_jailbreak_attack_datasets"


def render(turns: list[str]) -> str:
    """One conversation -> one text. User turns only, one per line."""
    lines = [f"User: {str(t).strip()}" for t in turns if str(t).strip()
             and str(t).strip().lower() != "none"]
    return "\n".join(lines)


def load_crescendo() -> pd.DataFrame:
    p = hf_hub_download(SAFEMT, "SafeMTData/Attack_600.json", repo_type="dataset")
    records = json.loads(Path(p).read_text(encoding="utf-8"))
    rows = []
    for r in records:
        turns = r.get("multi_turn_queries") or []
        text = render(turns)
        if len(turns) < 2 or not text:
            continue
        rows.append({
            "text": text, "label": 1,
            "attack_type": "crescendo",
            "original_attack_type": r.get("category") or "crescendo",
            # The harmful goal the escalation steers at -- kept for audit,
            # never used as model input.
            "original_text": r.get("plain_query"),
            "source_dataset": "hf_csv8", "split": "train",
        })
    logger.info("crescendo: %d conversations from %s", len(rows), SAFEMT)
    return pd.DataFrame(rows, columns=SCHEMA)


ULTRACHAT = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_FILE = "data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet"


def load_benign_natural(n: int, turns: int, seed: int) -> pd.DataFrame:
    """Benign counterweight for the crescendo family, matched on shape.

    The crescendo conversations are uniformly 5 natural-language user
    turns. Without benign conversations of the same shape, "5 turns of
    ordinary questions" becomes the attack signal -- the model would flag
    any multi-turn chat. UltraChat (MIT) supplies real benign dialogue in
    the same register; each is truncated to the same turn count.
    """
    p = hf_hub_download(ULTRACHAT, ULTRACHAT_FILE, repo_type="dataset")
    df = pd.read_parquet(p, columns=["messages"])
    rows = []
    for msgs in df["messages"]:
        user = [m["content"] for m in msgs if m.get("role") == "user"]
        if len(user) < turns:
            continue
        text = render(user[:turns])
        if text:
            rows.append(text)
        if len(rows) >= n:
            break
    logger.info("benign natural: %d conversations from %s", len(rows), ULTRACHAT)
    return pd.DataFrame({
        "text": rows, "label": 0,
        "attack_type": "multi_turn_benign",
        "original_attack_type": "benign",
        "original_text": None,
        "source_dataset": "hf_csv8", "split": "train",
    }, columns=SCHEMA)


def match_shape(attack: pd.DataFrame, benign: pd.DataFrame, seed: int,
                n_bins: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equalise the (turn count, length) histogram across the two labels.

    Measured before this existed: turn count and character length alone
    separated the tom-gibbs labels at AUC 0.740 (harmful median 11 turns
    against benign 9), and the crescendo family at similar strength via
    length (517 chars against 717). A detector trained on that learns to
    count turns and characters, which is the same failure as the length
    shortcut the project already fights.

    Both sides are sampled down to the per-bucket minimum, so the joint
    distribution matches by construction rather than by hope.
    """
    def keyed(df):
        t = df["text"].str.count(r"(?m)^User:")
        n = df["text"].str.len()
        return df.assign(_t=t, _n=n)

    a, b = keyed(attack), keyed(benign)
    # Bin length on the pooled quantiles so both sides share edges.
    edges = pd.concat([a["_n"], b["_n"]]).quantile(
        [i / n_bins for i in range(1, n_bins)]).unique()
    a["_b"] = pd.cut(a["_n"], [-1, *edges, float("inf")], labels=False)
    b["_b"] = pd.cut(b["_n"], [-1, *edges, float("inf")], labels=False)

    keep_a, keep_b = [], []
    for key in sorted(set(zip(a["_t"], a["_b"])) & set(zip(b["_t"], b["_b"]))):
        ga = a[(a["_t"] == key[0]) & (a["_b"] == key[1])]
        gb = b[(b["_t"] == key[0]) & (b["_b"] == key[1])]
        k = min(len(ga), len(gb))
        keep_a.append(ga.sample(n=k, random_state=seed))
        keep_b.append(gb.sample(n=k, random_state=seed))
    if not keep_a:
        logger.warning("no overlapping shape buckets; leaving both sides unmatched")
        return attack, benign
    drop = ["_t", "_n", "_b"]
    ra = pd.concat(keep_a, ignore_index=True).drop(columns=drop)
    rb = pd.concat(keep_b, ignore_index=True).drop(columns=drop)
    logger.info("shape-matched: %d attack / %d benign (from %d / %d)",
                len(ra), len(rb), len(attack), len(benign))
    return ra, rb


def _parse_conv(raw: str) -> list[str]:
    """The field is a Python-literal list of {'role','content'} dicts."""
    try:
        conv = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(conv, list):
        return []
    return [m.get("content", "") for m in conv
            if isinstance(m, dict) and m.get("role") == "user"]


def load_multiturn(n_per_label: int, seed: int) -> pd.DataFrame:
    frames = {}
    for fname, label in (("Harmful Dataset.csv", 1),
                         ("Completely-Benign Dataset.csv", 0)):
        p = hf_hub_download(GIBBS, fname, repo_type="dataset")
        df = pd.read_csv(p)
        df = df[df["Multi-turn conversation"].notna()]
        texts = df["Multi-turn conversation"].map(lambda r: render(_parse_conv(r)))
        keep = texts.str.strip().astype(bool)
        out = pd.DataFrame({
            "text": texts[keep], "label": label,
            "attack_type": "multi_turn",
            "original_attack_type": "cipher_substitution",
            "original_text": df.loc[keep, "Goal"] if "Goal" in df.columns else None,
            "source_dataset": "hf_csv9", "split": "train",
        })
        logger.info("multi_turn label=%d: %d conversations available", label, len(out))
        frames[label] = out[SCHEMA]

    # Equalise turn counts first, then cap. Capping first would re-skew the
    # histogram the matching just fixed.
    atk, ben = match_shape(frames[1], frames[0], seed)
    if len(atk) > n_per_label:
        atk = atk.sample(n=n_per_label, random_state=seed)
        ben = ben.sample(n=min(n_per_label, len(ben)), random_state=seed)
    return pd.concat([atk, ben], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "multiturn_rows.csv"))
    ap.add_argument("--n-per-label", type=int, default=1200,
                    help="cap per label for the tom-gibbs family, kept equal "
                         "so conversation shape does not predict the label")
    ap.add_argument("--merge", nargs="?", const=str(REPO_ROOT / "data" / "consolidated_v2.csv"),
                    default=None, help="also merge into this corpus")
    ap.add_argument("--merged-out", default=str(REPO_ROOT / "data" / "consolidated_mt.csv"))
    ap.add_argument("--eval-out", default=str(REPO_ROOT / "data" / "multiturn_eval.csv"))
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction withheld from the corpus for honest evaluation")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cres = load_crescendo()
    turns = int(cres["text"].str.count(r"(?m)^User:").median())
    # Pull a surplus so shape matching has room to select from.
    benign_nat = load_benign_natural(len(cres) * 6, turns, a.seed)
    cres, benign_nat = match_shape(cres, benign_nat, a.seed)
    new = pd.concat([cres, benign_nat, load_multiturn(a.n_per_label, a.seed)],
                    ignore_index=True)

    # A slice is withheld from the corpus entirely. The qualifire holdout
    # contains no multi-turn rows, so without this there is nothing to
    # measure the addition against except data the model trained on.
    eval_idx = (new.groupby(["source_dataset", "label"])
                .sample(frac=a.holdout_frac, random_state=a.seed).index)
    eval_rows = new.loc[eval_idx]
    new = new.drop(index=eval_idx)
    eval_rows.to_csv(a.eval_out, index=False, encoding="utf-8")
    logger.info("held out %s: %d rows (%d attack / %d benign) -- never trained on",
                a.eval_out, len(eval_rows), int((eval_rows["label"] == 1).sum()),
                int((eval_rows["label"] == 0).sum()))

    new.to_csv(a.out, index=False, encoding="utf-8")
    logger.info("wrote %s: %d rows (%d attack / %d benign)",
                a.out, len(new), int((new["label"] == 1).sum()),
                int((new["label"] == 0).sum()))

    if a.merge:
        base = pd.read_csv(a.merge, encoding="utf-8")
        for c in SCHEMA:
            if c not in base.columns:
                base[c] = None
        merged = pd.concat([base[SCHEMA], new], ignore_index=True)
        merged.to_csv(a.merged_out, index=False, encoding="utf-8")
        logger.info("merged %s + %d new rows -> %s (%d rows)",
                    Path(a.merge).name, len(new), a.merged_out, len(merged))


if __name__ == "__main__":
    main()
