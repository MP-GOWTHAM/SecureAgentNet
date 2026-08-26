"""Merges the public prompt-injection datasets into unified train/val/test splits.

Design notes (why this file looks the way it does):

- Each source dataset uses different column names and label conventions
  (int 0/1, bool, or string labels like "injection"/"jailbreak"). Rather than
  writing one-off parsing per dataset inline in the merge step, every source
  is normalized through `DatasetSpec` into one schema:
      {text: str, label: int (1=adversarial, 0=benign), category: str, source: str}
  This keeps the merge/split logic in `build_splits` agnostic to source quirks.

- `qualifire/prompt-injections-benchmark` (now hosted as
  `rogue-security/prompt-injections-benchmark` after an upstream rename) is
  loaded into its OWN split and is never concatenated into the training pool.
  This is a hard requirement from the project spec: it exists to measure
  generalization to a benchmark the model never saw, so leaking it into
  train/val would invalidate every downstream ASR/FPR number.

- `Necent/llm-jailbreak-prompt-injection-dataset` and
  `Mindgard/evaded-prompt-injection-and-jailbreak-samples` are gated on the
  Hub (license click-through). `datasets.load_dataset` will raise a clear
  auth error if the caller hasn't run `huggingface-cli login` / accepted the
  license / exported HF_TOKEN — we catch that and surface an actionable
  message instead of a raw traceback, and continue with whatever sources
  did load rather than hard-failing the whole pipeline.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
SCHEMA_COLUMNS = ["text", "label", "category", "source"]

# Values (lowercased) that mean "this row is an attack" when a dataset
# stores its label as a string rather than an int/bool.
POSITIVE_STRING_VALUES = {
    "injection", "malicious", "attack", "jailbreak", "adversarial",
    "prompt_injection", "prompt injection", "true", "1", "unsafe",
}
NEGATIVE_STRING_VALUES = {
    "benign", "safe", "clean", "false", "0", "legitimate",
}


@dataclass
class DatasetSpec:
    """Declarative description of how to load + normalize one HF dataset."""

    name: str
    hf_id: str
    role: str  # "train" or "test"
    config: str | None = None
    split: str = "train"
    text_field_candidates: tuple[str, ...] = ("text", "prompt", "instruction", "input")
    label_field_candidates: tuple[str, ...] = ("label", "is_injection", "is_malicious", "target")
    category_field_candidates: tuple[str, ...] = ("category", "attack_type", "type")
    requires_auth: bool = False
    balance_labels: float | None = None
    """Target positive fraction when `max_rows` caps this source, e.g. 0.5.

    None (the default) keeps the stratified behaviour that preserves the
    source's own attack/benign ratio. Set it only for a large source whose
    skew would otherwise propagate into the merged corpus."""
    fallback_hf_ids: tuple[str, ...] = field(default_factory=tuple)
    # Row cap applied *before* dedup/splitting, sampled with `seed`. Necent
    # alone is ~1.17M rows post-explosion (InjecAgent/ToolEmu/BIPIA etc all
    # merged); left uncapped it would outweigh every other source ~250:1 and
    # both dominate training and blow up iteration time on a laptop. Capping
    # keeps the merge balanced across sources; raise/remove this once doing
    # a final large-scale training run rather than iterating.
    max_rows: int | None = None
    # Escape hatch for sources whose schema doesn't fit "one text column +
    # one label column" (e.g. Mindgard, which pairs an original attack with
    # its obfuscated variant rather than labeling rows benign/malicious).
    custom_normalizer: Callable[[pd.DataFrame], pd.DataFrame] | None = None


def _normalize_mindgard(df: pd.DataFrame) -> pd.DataFrame:
    """Mindgard has no label column: every row is an (original, evaded-variant)
    pair of attack text, both adversarial. We unpivot both columns into
    label=1 rows so the detector sees pre- and post-obfuscation phrasing of
    the same underlying attack (the dataset's whole point is evasion
    robustness), tagging which side each row came from via `category`.
    """
    parts = []
    for col, tag in (("original_sample", "original"), ("modified_sample", "evaded")):
        if col not in df.columns:
            continue
        part = pd.DataFrame()
        part["text"] = df[col].astype(str)
        part["label"] = 1
        part["category"] = df.get("attack_name", tag).astype(str) + f"::{tag}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["text", "label", "category"])


def _normalize_smooth3(df: pd.DataFrame) -> pd.DataFrame:
    """Smooth-3's label is a multi-label LIST, not a scalar: `['BENIGN']` or
    `['JAILBREAK', 'INSTRUCTION_OVERRIDE']`, sometimes with ROLE_HIJACK or
    DATA_EXFILTRATION alongside. `_normalize_label` handles scalars only, so
    the collapse to a binary label happens here, keeping the attack taxonomy
    in `category` where it stays inspectable.

    Added (2026) to fix a specific, measured weakness in the existing
    mixture rather than for volume. Two properties it has that the other
    three sources do not:

      length balance   median benign 307 chars vs attack 357. The existing
                       corpus runs benign 129 vs attack 436, which is the
                       spurious cue behind the length shortcut (see the
                       architecture doc, section 5.2) -- and the qualifire
                       benchmark shares that artifact, so a model exploiting
                       it is rewarded by held-out AUC while failing on short
                       real injections. This source carries almost no length
                       signal, so it cannot reinforce the shortcut.

      class balance    48% benign / 52% attack. Mindgard is 100% attack and
                       Necent is 27% attack, so the merged corpus has never
                       had a source that was close to balanced on its own.

    Its attacks are also predominantly dilution-style -- an override buried
    in innocuous prose about papermaking or fishing -- which is the exact
    family that defeated every DistilBERT version in the Track B cycle.
    """
    out = pd.DataFrame()
    out["text"] = df["text"].astype(str)
    labels = df["labels"]
    out["label"] = labels.map(
        lambda v: int(any("JAILBREAK" in str(x).upper() for x in v))
        if isinstance(v, (list, tuple))
        else int("JAILBREAK" in str(v).upper())
    )
    out["category"] = labels.map(
        lambda v: "|".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
    )
    return out


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        name="neuralchemy",
        hf_id="neuralchemy/Prompt-injection-dataset",
        role="train",
        config="core",
        split="train",
    ),
    DatasetSpec(
        name="necent",
        hf_id="Necent/llm-jailbreak-prompt-injection-dataset",
        role="train",
        split="train",
        text_field_candidates=("prompt", "text", "instruction", "input"),
        label_field_candidates=("prompt_adversarial", "label", "is_injection"),
        category_field_candidates=("prompt_type", "attack_technique", "category"),
        max_rows=30_000,
    ),
    DatasetSpec(
        name="mindgard",
        hf_id="Mindgard/evaded-prompt-injection-and-jailbreak-samples",
        role="train",
        split="train",
        custom_normalizer=_normalize_mindgard,
    ),
    DatasetSpec(
        name="smooth3",
        hf_id="Smooth-3/llm-prompt-injection-attacks",
        role="train",
        split="train",
        custom_normalizer=_normalize_smooth3,
    ),
    DatasetSpec(
        # 261,738 rows upstream, 48.8% positive, median length attack 380 /
        # benign 309 -- as length-neutral as smooth3, and stylistically the
        # same family, so it is capped rather than taken whole: the two
        # together would otherwise be ~half the corpus in one voice.
        #
        # 17.8% of it already appears in the corpus; dedup in
        # `_splits_from_merged` removes those, so the cap is applied to the
        # raw pull and the effective contribution is lower.
        #
        # NOT included: jayavibhav/prompt-injection-safety, which overlaps
        # the existing corpus 90.2% -- a repackaging, not a new source.
        name="jayavibhav",
        hf_id="jayavibhav/prompt-injection",
        role="train",
        split="train",
        max_rows=100_000,
    ),
    DatasetSpec(
        # 535,105 rows upstream with ZERO overlap against every other
        # source -- genuinely independent provenance (HackAPrompt-style
        # competition submissions rather than generated variants), which is
        # what the measured evidence says actually helps: one new source
        # moved held-out AUC 0.8278 -> 0.9168, while 1.1M extra Necent rows
        # cost 0.078.
        #
        # Capped at 120k for the same reason Necent is capped at 30k.
        # Uncapped it would be 74% of the corpus, and the one time a single
        # source reached 97.8% the model regressed to that source's own
        # ceiling (AUC 0.7501). Its label field is `labels`, not `label`.
        name="imoxto",
        hf_id="imoxto/prompt_injection_cleaned_dataset-v2",
        role="train",
        split="train",
        label_field_candidates=("labels", "label", "is_injection"),
        max_rows=120_000,
        # Upstream is 24.8% positive. Taken at that ratio it dragged the
        # corpus from 1.19 to 1.61 pos_weight and cost 0.033 held-out AUC;
        # drawn 50/50 it contributes its provenance without its skew.
        balance_labels=0.5,
    ),
    DatasetSpec(
        name="qualifire_holdout",
        hf_id="qualifire/prompt-injections-benchmark",
        role="test",
        split="test",
        # Upstream renamed the repo; datasets.load_dataset follows HF's
        # redirect for the old id automatically, but we keep the new id
        # as an explicit fallback in case the redirect is ever removed.
        fallback_hf_ids=("rogue-security/prompt-injections-benchmark",),
    ),
]


def _normalize_label(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in POSITIVE_STRING_VALUES:
            return 1
        if v in NEGATIVE_STRING_VALUES:
            return 0
    return None


def _pick_field(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _load_hf_dataset(spec: DatasetSpec, hf_token: str | None):
    from datasets import load_dataset

    ids_to_try = (spec.hf_id, *spec.fallback_hf_ids)
    last_err = None
    for hf_id in ids_to_try:
        try:
            kwargs = {"split": spec.split}
            if spec.config:
                kwargs["name"] = spec.config
            if hf_token:
                kwargs["token"] = hf_token
            return load_dataset(hf_id, **kwargs)
        except Exception as e:  # noqa: BLE001 - want to try all fallbacks then report
            last_err = e
            continue
    raise last_err


def load_and_normalize(spec: DatasetSpec, hf_token: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Load one dataset from the Hub and coerce it to SCHEMA_COLUMNS."""
    try:
        ds = _load_hf_dataset(spec, hf_token)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if spec.requires_auth or "gated" in msg or "authentication" in msg or "401" in msg:
            logger.warning(
                "Skipping %s (%s): requires HF authentication + license acceptance. "
                "Run `huggingface-cli login` or set HF_TOKEN, then accept the dataset's "
                "license at https://huggingface.co/datasets/%s. Continuing without it.",
                spec.name, spec.hf_id, spec.hf_id,
            )
            return pd.DataFrame(columns=SCHEMA_COLUMNS)
        logger.warning("Skipping %s (%s): %s", spec.name, spec.hf_id, e)
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = ds.to_pandas()

    if spec.custom_normalizer is not None:
        out = spec.custom_normalizer(df)
        out["source"] = spec.name
    else:
        columns = list(df.columns)
        text_col = _pick_field(columns, spec.text_field_candidates)
        label_col = _pick_field(columns, spec.label_field_candidates)
        category_col = _pick_field(columns, spec.category_field_candidates)

        if text_col is None or label_col is None:
            raise ValueError(
                f"{spec.name}: could not find text/label columns among {columns}. "
                f"Tried text candidates {spec.text_field_candidates} and "
                f"label candidates {spec.label_field_candidates}."
            )

        out = pd.DataFrame()
        out["text"] = df[text_col].astype(str)
        out["label"] = df[label_col].map(_normalize_label)
        out["category"] = df[category_col].astype(str) if category_col else "unknown"
        out["source"] = spec.name

        n_before = len(out)
        out = out.dropna(subset=["label"])
        out["label"] = out["label"].astype(int)
        n_dropped = n_before - len(out)
        if n_dropped:
            logger.warning("%s: dropped %d rows with unrecognized label values", spec.name, n_dropped)

    if spec.balance_labels is not None and spec.max_rows is not None:
        # Opt-in override of the stratified cap below: draw a fixed
        # attack/benign ratio instead of preserving the source's own.
        #
        # Added because the stratified cap faithfully carried imoxto's
        # 24.8%-positive skew into the corpus, which pushed pos_weight
        # 1.19 -> 1.61 and measurably cost held-out AUC (0.9168 -> 0.8835)
        # when that source was added. Preserving a source's ratio is the
        # right default; it is the wrong choice for a large, heavily
        # skewed source being mixed with balanced ones.
        n_pos = int(spec.max_rows * spec.balance_labels)
        n_neg = spec.max_rows - n_pos
        pos, neg = out[out["label"] == 1], out[out["label"] == 0]
        take_pos, take_neg = min(len(pos), n_pos), min(len(neg), n_neg)
        out = pd.concat(
            [pos.sample(n=take_pos, random_state=seed), neg.sample(n=take_neg, random_state=seed)],
            ignore_index=True,
        )
        logger.info(
            "%s: balanced sample to %d rows (%.1f%% positive, requested %.0f%%)",
            spec.name, len(out), 100 * out["label"].mean(), 100 * spec.balance_labels,
        )
    elif spec.max_rows is not None and len(out) > spec.max_rows:
        # Stratified sample so the cap doesn't accidentally skew the
        # attack/benign ratio of a source that's mostly one class.
        frac = spec.max_rows / len(out)
        sampled_parts = [g.sample(frac=frac, random_state=seed) for _, g in out.groupby("label")]
        out = pd.concat(sampled_parts, ignore_index=True)
        logger.info("%s: sampled down to %d rows (max_rows=%d)", spec.name, len(out), spec.max_rows)

    return out[SCHEMA_COLUMNS]


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate text, keeping the first occurrence.

    Attack datasets scraped from overlapping sources (e.g. jailbreak
    collections that all mirror the same viral "DAN" prompt) contain a lot
    of literal duplicates; leaving them in lets a handful of memorized
    strings dominate both train and reported metrics.
    """
    df = df.copy()
    df["_hash"] = df["text"].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    df = df.drop_duplicates(subset="_hash").drop(columns="_hash")
    return df.reset_index(drop=True)


def _text_key(text: str) -> str:
    """Normalised identity used for cross-split leak checks.

    Deliberately stronger than `_dedup`'s exact hash: case- and
    whitespace-insensitive, so a trivially reformatted copy of a holdout
    row still cannot slip into training.
    """
    return hashlib.sha256(" ".join(str(text).lower().split()).encode("utf-8")).hexdigest()


def _splits_from_merged(
    merged: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Dedup + stratified train/val split, given a merged frame with a
    `split_role` column ("train" or "test"). Shared by both the live
    Hub-loading path and the local-CSV path so the leak-prevention and
    stratification logic only has to be correct in one place.
    """
    train_pool = _dedup(merged[merged["split_role"] == "train"].drop(columns="split_role"))
    test = _dedup(merged[merged["split_role"] == "test"].drop(columns="split_role"))

    # Holding out a whole source only isolates it if no *other* source
    # re-contains its texts. hf_csv5 (Smooth-3) did: 9.1% of it was
    # verbatim qualifire, covering 81.6% of the 5000-row benchmark. The
    # damage scales with model capacity -- a DistilBERT trained on it
    # scored FPR 0.069 over the full holdout but 0.199 over the part it
    # had genuinely not seen, so the leak read as a breakthrough.
    #
    # Rows are dropped from TRAIN, never from the holdout, so the
    # benchmark stays comparable with previously published numbers.
    test_keys = set(test["text"].map(_text_key))
    leaked = train_pool["text"].map(_text_key).isin(test_keys)
    if leaked.any():
        logger.warning(
            "holdout leak: dropped %d training rows (%.1f%% of train) whose text "
            "also appears in the holdout source", int(leaked.sum()), 100 * leaked.mean())
        train_pool = train_pool[~leaked].reset_index(drop=True)
        if train_pool.empty:
            raise ValueError(
                "every training row duplicates the holdout source; there is "
                "nothing left to train on. Check that the training sources are "
                "not copies of the benchmark."
            )

    # Stratified train/val split by label so both splits keep the same
    # attack/benign ratio as the pooled training data.
    train_parts, val_parts = [], []
    for label in sorted(train_pool["label"].unique()):
        subset = train_pool[train_pool["label"] == label].sample(frac=1.0, random_state=seed)
        n_val = max(1, int(len(subset) * val_fraction)) if len(subset) > 1 else 0
        val_parts.append(subset.iloc[:n_val])
        train_parts.append(subset.iloc[n_val:])

    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "Splits built: train=%d (%.1f%% positive) val=%d (%.1f%% positive) test=%d (%.1f%% positive)",
        len(train), 100 * train["label"].mean() if len(train) else 0,
        len(val), 100 * val["label"].mean() if len(val) else 0,
        len(test), 100 * test["label"].mean() if len(test) else 0,
    )
    return {"train": train, "val": val, "test": test}


def build_splits(
    hf_token: str | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return {"train": df, "val": df, "test": df}, pulling all four sources
    live from the Hub.

    `test` is built exclusively from the qualifire/rogue-security benchmark
    and must never be merged into train/val (see module docstring).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "splits.parquet"

    if use_cache and cache_file.exists():
        logger.info("Loading cached splits from %s", cache_file)
        merged = pd.read_parquet(cache_file)
    else:
        frames = []
        for spec in DATASET_SPECS:
            logger.info("Loading %s (%s)...", spec.name, spec.hf_id)
            df = load_and_normalize(spec, hf_token=hf_token, seed=seed)
            df["split_role"] = spec.role
            frames.append(df.assign(split_role=spec.role))
            logger.info("  -> %d rows", len(df))
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=[*SCHEMA_COLUMNS, "split_role"]
        )
        if use_cache:
            merged.to_parquet(cache_file, index=False)

    return _splits_from_merged(merged, val_fraction=val_fraction, seed=seed)


# Maps consolidate.py's source_dataset tags back to the same source names
# DATASET_SPECS uses, so downstream code (metrics, error messages, the
# `source` column) doesn't need to know which loading path produced a row.
CSV_SOURCE_MAP = {
    "hf_csv": "neuralchemy",
    "hf_csv2": "qualifire_holdout",
    "hf_csv3": "necent",
    "hf_csv4": "mindgard",
    "hf_csv5": "smooth3",
    "hf_csv6": "jayavibhav",
    "hf_csv7": "imoxto",
}
# Only hf_csv2 (qualifire) is the held-out benchmark; everything else is
# pooled into train/val regardless of the CSV's own train/validation split
# (that split is wildly uneven across sources - e.g. only neuralchemy has
# a validation slice - so we re-derive train/val ourselves via
# `_splits_from_merged` for a consistent, source-balanced val set).
CSV_TEST_SOURCES = {"hf_csv2"}


def build_splits_from_csv(
    csv_path: str | Path,
    necent_max_rows: int | None = 30_000,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build {"train", "val", "test"} from a locally pre-consolidated CSV
    (e.g. produced by a `consolidate.py`-style script merging the same four
    HF sources), instead of hitting the Hub.

    Expects columns: text, label, attack_type, source_dataset, split.
    `label` is used as-is (0/1) — if your CSV encodes Necent's label from
    `is_dangerous` (a general harmful-content signal, not injection-specific)
    that broader semantics carries through unchanged; re-derive labels
    upstream in your consolidation script if you want injection-only
    positives instead.
    """
    csv_path = Path(csv_path)
    usecols = ["text", "label", "attack_type", "source_dataset", "split"]
    df = pd.read_csv(csv_path, usecols=usecols, encoding="utf-8")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["category"] = df["attack_type"].fillna("unknown")
    df["source"] = df["source_dataset"].map(CSV_SOURCE_MAP).fillna(df["source_dataset"])
    df["split_role"] = df["source_dataset"].map(
        lambda s: "test" if s in CSV_TEST_SOURCES else "train"
    )

    if necent_max_rows is not None:
        necent_mask = df["source"] == "necent"
        necent_df = df[necent_mask]
        if len(necent_df) > necent_max_rows:
            frac = necent_max_rows / len(necent_df)
            sampled = [g.sample(frac=frac, random_state=seed) for _, g in necent_df.groupby("label")]
            df = pd.concat([df[~necent_mask], *sampled], ignore_index=True)
            logger.info("necent: sampled down to %d rows (necent_max_rows=%d)", sum(len(s) for s in sampled), necent_max_rows)

    merged = df[[*SCHEMA_COLUMNS, "split_role"]]
    return _splits_from_merged(merged, val_fraction=val_fraction, seed=seed)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to a pre-consolidated CSV instead of the live Hub loader")
    parser.add_argument("--necent-max-rows", type=int, default=30_000)
    args = parser.parse_args()

    if args.csv:
        splits = build_splits_from_csv(args.csv, necent_max_rows=args.necent_max_rows)
    else:
        splits = build_splits(use_cache=False)

    for name, df in splits.items():
        print(f"{name}: {len(df)} rows, columns={list(df.columns)}")
        print(df.head(3))
