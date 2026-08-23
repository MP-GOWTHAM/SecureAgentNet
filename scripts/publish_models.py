"""Publishes the checkpoints a working install needs to the Hugging Face Hub.

They cannot live in git. Five of the DistilBERT checkpoints are 253 MB
each, against GitHub's 100 MB hard per-file limit -- a push containing one
is rejected outright, not merely discouraged. Git LFS would work but
GitHub's free LFS tier (1 GB storage, 1 GB/month bandwidth) is smaller
than this project's 1.6 GB of checkpoints.

Only three are needed to run the app, and they are the three published:

    ensemble_v4_persona    49 MB   from-scratch ensemble (injection)
    harm_detector          49 MB   content-harm classifier
    v3                    253 MB   DistilBERT, the other combined_max member

`combined_max` itself is only a config referencing the two members, so it
is recreated locally by the bootstrap script rather than uploaded.

Run this yourself -- it needs a Hugging Face token with WRITE scope, and
tokens should not be handled by anyone but you:

    hf auth login          # a write token, not the read one used for datasets
    python scripts/publish_models.py --repo-id <your-username>/secureagentnet-models

Add --private to publish to a private repo (the bootstrap script then
needs an authenticated reader).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("publish_models")

MODELS_DIR = REPO_ROOT / "secureagentnet" / "data" / "models"

# name -> (what it is, needed to run the app?)
PUBLISH = {
    "ensemble_v4_persona": "from-scratch ensemble detector (injection axis)",
    "harm_detector": "content-harm classifier (harm axis)",
    "v3": "DistilBERT after the Track B cycle (second combined_max member)",
    # Not needed by the default runtime configuration, but the README
    # offers it as the "maximum utility" option (FPR 0.328 against
    # combined_max's 0.431), and a documented option that a fresh clone
    # cannot obtain is not really an option.
    "ensemble_v5_fpr": "persona-rebalanced ensemble (lower FPR, misses one short attack)",
}

CARD = """---
library_name: pytorch
tags: [prompt-injection, security, secureagentnet]
---

# SecureAgentNet detector checkpoints

Checkpoints for [SecureAgentNet](https://github.com/MP-GOWTHAM/SecureAgentNet).
They live here rather than in git because five of the project's checkpoints
are 253 MB each, against GitHub's 100 MB per-file hard limit.

| Directory | What it is | Held-out AUC |
|---|---|---|
| `ensemble_v4_persona` | From-scratch ensemble: char-CNN + BiLSTM-attention + scratch transformer, 12.3M params | 0.8278 |
| `v3` | DistilBERT after the Track B online-retraining cycle, 66M params | 0.7875 |
| `harm_detector` | Content-harm classifier (a separate axis from injection) | 0.9028 |
| `ensemble_v5_fpr` | Persona-rebalanced ensemble — lower FPR (0.328 vs 0.363), misses one canonical short attack | 0.8237 |

The recommended runtime configuration is `combined_max` — the elementwise
max of `ensemble_v4_persona` and `v3`. The two fail in opposite directions
(dilution gap +0.142 vs −0.303), and the combination is the only one
measured with no known blind spot: 8/8 canonical short attacks and 8/8 real
evasions, FNR 0.035. It is a config referencing the two members, so it is
recreated locally rather than stored here.

Download with `scripts/bootstrap_windows.ps1` in the repo, or manually:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="REPO_ID", local_dir="secureagentnet/data/models")
```

See the repository's `docs/SecureAgentNet_Detector_Architecture.docx` for
the full evaluation, including the configurations that failed.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. mpgowtham/secureagentnet-models")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded and exit")
    a = p.parse_args()

    missing = [n for n in PUBLISH if not (MODELS_DIR / n / "config.json").exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {', '.join(missing)} — train them first")

    total = 0
    for name, desc in PUBLISH.items():
        size = sum(f.stat().st_size for f in (MODELS_DIR / name).rglob("*") if f.is_file())
        total += size
        logger.info("%-22s %6.0f MB  %s", name, size / 1e6, desc)
    logger.info("total upload: %.0f MB", total / 1e6)

    if a.dry_run:
        logger.info("dry run — nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
        logger.info("authenticated as %s", who.get("name"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"not authenticated ({exc}). Run `hf auth login` with a WRITE-scope token first."
        ) from exc

    api.create_repo(repo_id=a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    logger.info("repo ready: %s (private=%s)", a.repo_id, a.private)

    card = MODELS_DIR / "README.md"
    card.write_text(CARD.replace("REPO_ID", a.repo_id), encoding="utf-8")

    for name in PUBLISH:
        logger.info("uploading %s ...", name)
        api.upload_folder(
            repo_id=a.repo_id, repo_type="model",
            folder_path=str(MODELS_DIR / name), path_in_repo=name,
        )
    api.upload_file(
        repo_id=a.repo_id, repo_type="model",
        path_or_fileobj=str(card), path_in_repo="README.md",
    )
    card.unlink(missing_ok=True)

    logger.info("done — https://huggingface.co/%s", a.repo_id)
    logger.info("now set MODELS_REPO in scripts/bootstrap_windows.ps1 to %s", a.repo_id)


if __name__ == "__main__":
    main()
