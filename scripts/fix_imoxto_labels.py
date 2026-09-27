"""Build contrastive imoxto fixes on top of the stratified split.

imoxto (imoxto/prompt_injection_cleaned_dataset-v2) is HackAPrompt: each
row is a competition level's prompt with a competitor's input inserted, and
`label` records whether the attack SUCCEEDED -- failed attacks are labelled
benign, which is wrong for a detector of injection attempts.

Every row is serialised exactly as

    PREFIX + f(INPUT) + MIDDLE + "\\n\\n\\n" + INPUT

where PREFIX/MIDDLE are the level's fixed text, INPUT is the competitor's
text repeated raw at the end, and f is the level's own transform (identity,
or the slash level's \\c\\h\\a\\r rendering). Rows are parsed EXACTLY on that
structure -- no prefix/suffix guessing -- and grouped by their (PREFIX, f,
MIDDLE) signature; only signatures shared by >= MIN_ROWS train rows count as
a level. A mis-parse produces a rare signature and is excluded.

Modes:
  keep1  keep label-1 rows (successful attacks: certainly injections), drop
         label-0 rows (benign label wrong, true label uncertain -- some are
         bare URLs), and add a benign TWIN for every kept attack.
  flip   every parsed imoxto row -> attack; a TWIN for each row that was 0.

TWIN = PREFIX + f(H) + MIDDLE + "\\n\\n\\n" + H, with H a harmless text of the
same length from the SAME split's non-imoxto benign rows. Level text, level
transform and the repeat artifact are then identical on both sides; only the
content of the input differs. Unparseable imoxto rows are dropped: an attack
without a twin would leave its template on one side only.

usage: fix_imoxto_labels.py MODE SPLIT_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

mode, split_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
assert mode in ("flip", "flipall", "keep1")   # flipall: every attempt -> attack, a twin for EVERY one
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
S = {n: pd.read_parquet(split_dir / f"{n}.parquet") for n in ("train", "val", "test")}

SEP = "\n\n\n"
BS = chr(92)
MIN_ROWS = 50


def slash(s):
    return "".join(BS + c for c in s)


TRANSFORMS = {"id": lambda s: s, "slash": slash}


def parse(text):
    """-> (prefix, fname, input, middle) or None."""
    if SEP not in text:
        return None
    inp = text.rsplit(SEP, 1)[1]
    if not inp.strip():
        return None
    head = text[: len(text) - len(inp) - len(SEP)]
    for fname, f in TRANSFORMS.items():
        shown = f(inp)
        i = head.find(shown)
        if i < 0:
            continue
        prefix, middle = head[:i], head[i + len(shown):]
        if prefix + shown + middle + SEP + inp == text:
            return prefix, fname, inp, middle
    return None


# ---- levels = frequent signatures among TRAIN imoxto rows -----------------
tr_imx = S["train"].loc[S["train"]["source"] == "imoxto", "text"]
sigs = Counter()
for t in tr_imx:
    p = parse(t)
    if p:
        sigs[(p[0], p[1], p[3])] += 1
LEVELS = {s for s, n in sigs.items() if n >= MIN_ROWS}
print(f"{len(LEVELS)} levels (signatures with >= {MIN_ROWS} train rows); "
      f"they cover {sum(sigs[s] for s in LEVELS)} of {len(tr_imx)} train imoxto rows")


def level_parse(text):
    p = parse(text)
    if p and (p[0], p[1], p[3]) in LEVELS:
        return p
    return None


def harmless_of_length(pool, n):
    t = pool[rng.integers(len(pool))]
    if len(t) <= n:
        return t
    cut = t[:n]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > n * 0.6 else cut


def make_twin(parsed, pool):
    prefix, fname, inp, middle = parsed
    h = harmless_of_length(pool, max(len(inp), 20))
    return prefix + TRANSFORMS[fname](h) + middle + SEP + h


stats = {"mode": mode, "levels": len(LEVELS)}
for name, d in S.items():
    d = d.copy()
    imx = d["source"] == "imoxto"
    pool = d[(~imx) & (d["label"] == 0)]["text"]
    pool = pool[pool.str.len().between(20, 2000)].tolist()

    parsed = d["text"].where(imx).map(lambda t: level_parse(t) if isinstance(t, str) else None)
    ok = parsed.notna()
    n_unparsed = int((imx & ~ok).sum())
    keep = ~imx | ok
    if mode == "keep1":
        n_dropped0 = int((imx & ok & (d["label"] == 0)).sum())
        keep &= ~(imx & (d["label"] == 0))
        twin_from = imx & ok & (d["label"] == 1)
    elif mode == "flipall":
        n_dropped0 = 0
        twin_from = imx & ok
    else:
        n_dropped0 = 0
        twin_from = imx & ok & (d["label"] == 0)

    twins = [make_twin(p, pool) for p in parsed[twin_from]]
    d = d[keep].copy()
    d.loc[d["source"] == "imoxto", "label"] = 1
    tw = pd.DataFrame({"text": twins, "label": 0,
                       "category": "benign_template_twin", "source": "imoxto_twin"})
    d = pd.concat([d, tw], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    d.to_parquet(out_dir / f"{name}.parquet", index=False)
    n_att = int((d["source"] == "imoxto").sum())
    stats[name] = {"rows": int(len(d)), "attack_rate": round(float(d["label"].mean()), 4),
                   "imoxto_attacks": n_att, "twins": len(twins),
                   "unparsed_imoxto_dropped": n_unparsed, "label0_dropped": n_dropped0}
    print(f"{name}: {len(d)} rows, attack rate {d['label'].mean():.3f} | imoxto attacks {n_att}, "
          f"twins {len(twins)}, unparsed dropped {n_unparsed}, label-0 dropped {n_dropped0}")

# one pair to eyeball
for t in tr_imx:
    p = level_parse(t)
    if p and 25 < len(p[2]) < 90 and p[3]:
        print("\nATTACK:", repr(t[-260:]))
        print("TWIN:  ", repr(make_twin(p, pool)[-260:]))
        break
(out_dir / "contrast_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
