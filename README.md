# SecureAgentNet

Runtime defense framework for tool-integrated LLM agents that fuses **prompt
injection detection** with **privilege governance**, so that an injection
which partially evades the detector still has to clear a separate
tool-scope check before it can do anything.

## Why

Existing defenses treat these as separate problems: a semantic detector
doesn't know what the agent is still allowed to do if it misses an attack,
and a privilege/ABAC system doesn't know whether the instruction behind a
tool call was injected. SecureAgentNet correlates both signals and reports a
**Chained Attack Success Rate (C-ASR)**: the fraction of injections that
*both* evade the detector *and* result in an out-of-scope tool call actually
being permitted — the failure mode neither defense catches alone.

## Status

- [x] **Phase 1** — repo scaffold + dataset loader
- [x] **Phase 2** — semantic injection detector (DistilBERT, train/eval)
- [x] **Phase 3** — privilege governance (ABAC policy engine)
- [x] **Phase 4** — correlation/fusion layer + C-ASR evaluation harness

## Layout

```
secureagentnet/
├── detector/           # injection classifier: model, training, data loading
├── privilege/          # ABAC policy engine + per-role policy configs
│   └── policies/        # email_agent.yaml, file_agent.yaml, research_agent.yaml
├── correlation/         # fuses detector score + privilege deviation → decision
├── eval/               # ASR / C-ASR / FPR / FNR / utility metrics + baselines
├── simulate/           # mock tool-calling agent for chained-attack scenarios
├── configs/            # run configs
└── tests/
```

## Privilege governance (Phase 3)

`secureagentnet/privilege/policy_engine.py` is a deny-by-default ABAC
engine: each agent role gets a YAML policy under `privilege/policies/`
listing exactly which tools it may call and, per tool, which resources
(recipients, paths, URLs) it may target via glob patterns. A tool call must
also present a `ScopedCredential` — a simulated short-lived, role-bound
token (no real crypto; see the module docstring for why that's an
intentional scope cut) — that expires like a real STS-issued token would.

```python
from secureagentnet.privilege.policy_engine import PolicyEngine, ToolCallRequest, issue_credential

engine = PolicyEngine.from_directory()  # loads privilege/policies/*.yaml
cred = issue_credential("email_agent", ttl_seconds=300)

engine.authorize(cred, ToolCallRequest(tool_name="send_email", resource="alice@secureagentnet-corp.com"))
# Decision(allowed=True, violation_type=NONE, ...)

engine.authorize(cred, ToolCallRequest(tool_name="delete_file", resource="/workspace/report.docx"))
# Decision(allowed=False, violation_type=TOOL_NOT_PERMITTED, ...)
```

`Decision.out_of_scope` is the signal Phase 4's fusion layer will combine
with the detector's risk score. Six roles ship as examples, each scoped to
only the tools/resources that role's job actually requires, and each
demonstrating a different kind of constraint:

| Role | Tools | What it blocks |
|---|---|---|
| `email_agent` | send (domain-restricted), read/search inbox | exfiltration to external domains, mass-BCC |
| `file_agent` | read/write/list, confined to `/workspace/**` | workspace escape, no delete at all |
| `research_agent` | fetch http(s), web search, save notes | `file://`/non-http schemes from a fetched page's injected instructions |
| `calendar_agent` | create/list events, cancel own events | mass-invite spam, cancelling someone else's event |
| `code_exec_agent` | run code in `/sandbox/**`, capped runtime | sandbox escape, unbounded/resource-exhaustion loops, no network tool at all |
| `support_agent` | read/reply tickets, refund ≤ $50 | large refunds triggered by injected ticket text, no email/file access |

See `secureagentnet/tests/test_policy_engine.py` for the hand-written
attack scenario behind each row.

Also included: **generalized ABAC conditions** beyond the resource glob
(`ToolPermission.conditions`, e.g. capping `send_email` to 5 recipients
per call — checked against `ToolCallRequest.params`, fails closed if the
attribute is missing), a **`CredentialStore`** for early revocation
(`store.revoke(token_id)` denies a still-unexpired credential on its next
call), and an **`AuditLog`** that records every `authorize()` decision
in-memory and optionally as append-only JSONL (`AuditLog(path=...)`) for a
full decision trail, not just aggregate metrics.

## Correlation / fusion + evaluation (Phase 4)

`secureagentnet/correlation/fusion.py`'s `FusionEngine` combines a
detector risk score with `Decision.out_of_scope` into one
Allow/Block/Flag decision. Default thresholds match the project brief's
example rule verbatim (`risk_score > 0.7` blocks outright;
`risk_score > 0.4 AND out_of_scope` blocks on the combined signal); a
`FusionConfig(strict_privilege=True)` makes any out-of-scope call an
unconditional block regardless of risk score — see the module docstring
for why that knob matters for C-ASR specifically.

`secureagentnet/simulate/agent_env.py` bridges the (text-only) detector
test set and the (tool-call-only) privilege layer: it deterministically
assigns each test example a role and a plausible tool call — for attacks,
one of several violation types per role, always including at least one
**in-scope** attack (a tool call privilege alone would allow, so only the
detector has any chance of catching it — the case that makes C-ASR
non-trivial).

`secureagentnet/eval/run_eval.py` scores the qualifire held-out test set
end-to-end and prints a comparison table against two single-signal
baselines (`eval/baselines.py`): detection-only (ignores privilege
entirely) and privilege-only (ignores detector risk entirely). Metrics
(`eval/metrics.py`): **ASR** (fraction of attacks whose tool call executes
— FLAG counts as executed, only BLOCK prevents it), **C-ASR** (among
attacks that both evaded the raw detector *and* targeted a genuinely
out-of-scope tool, the fraction that still executed under this method),
detector **FPR/FNR**, and **utility preservation** (fraction of benign
requests that still execute).

```bash
python -m secureagentnet.eval.run_eval --csv /path/to/consolidated_dataset.csv
python -m secureagentnet.eval.run_eval --csv /path/to/consolidated_dataset.csv --strict-privilege
```

**Original run (upstream, pre-port).** These are the author's macOS results
for the original DistilBERT detector, kept for continuity. They were *not*
produced by this Windows port — see the table below it for measurements
taken here.

| Method | ASR | C-ASR | FPR | FNR | Utility |
|---|---|---|---|---|---|
| SecureAgentNet (default thresholds) | 0.093 | 0.914 | 0.661 | 0.084 | 0.389 |
| SecureAgentNet (`strict_privilege=True`) | **0.050** | **0.000** | 0.661 | 0.084 | 0.389 |
| Detection-only baseline | 0.084 | 1.000 | 0.661 | 0.084 | 0.339 |
| Privilege-only baseline | 0.427 | 0.000 | 0.661 | 0.084 | 1.000 |

Two things this table is actually showing: (1) with default (lenient)
thresholds, the fusion engine's C-ASR (91.4%) is barely better than
detection-only (100%) — a genuinely unauthorized call with a low risk
score mostly gets through as FLAG rather than BLOCK, which is the exact
gap `strict_privilege` closes to 0% while simultaneously *improving*
overall ASR past both baselines (0.050, better than either single-signal
defense alone) at no utility cost. (2) The detector's poor precision on
qualifire (discussed under Phase 2) shows up directly here as a 66% FPR,
dragging utility preservation down to ~39% under any method that uses the
detector at all — privilege-only hits 100% utility simply because it
never blocks benign requests by construction. That's the honest tradeoff
this framework makes as currently trained, not a bug in the eval harness.

**Measured on this port**, same 5,000-row qualifire holdout, same harness.
The detector was retrained here and a from-scratch ensemble added, so these
supersede the table above for anything built from this repo:

| Configuration | ASR | C-ASR | FPR | FNR | Utility |
|---|---|---|---|---|---|
| `combined_max` + `strict_privilege` **(recommended)** | **0.047** | **0.000** | 0.431 | **0.035** | 0.617 |
| `combined_max` (default thresholds) | 0.065 | 0.814 | 0.431 | 0.035 | 0.617 |
| `ensemble_v5_fpr` @ 0.50 | 0.065 | 0.067 | **0.328** | 0.163 | **0.672** |
| DistilBERT v1 @ 0.85 (this port's baseline) | 0.101 | 0.957 | 0.439 | 0.090 | 0.579 |

Two differences from the upstream table are worth naming. **FPR is 0.43
here rather than 0.66** — better, but still high, and roughly 60% of it is
not a model error at all: the training corpora label roleplay framing as an
attack 84.7% of the time while the benchmark does so only 53.5%, so the
detector is penalised for learning the convention it was trained on. FPR on
plain (non-persona) benign text is 0.188. **Utility is correspondingly
higher** at 0.617–0.672 against 0.389.

The `strict_privilege` finding replicates exactly: C-ASR drops to 0.000
while ASR simultaneously beats both single-signal baselines. That is the
paper's central claim and it holds on this port.

Full results for every configuration measured — including the interventions
that failed — are in `docs/SecureAgentNet_Detector_Architecture.docx` §8.

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| `neuralchemy/Prompt-injection-dataset` | train | clean `text`/`label`/`category` schema, 29 attack categories |
| `Necent/llm-jailbreak-prompt-injection-dataset` | train | ~1.17M rows aggregating InjecAgent/ToolEmu/BIPIA/etc; sampled to 30k stratified rows for iteration speed (see `DatasetSpec.max_rows` in `detector/data_loader.py`) |
| `Mindgard/evaded-prompt-injection-and-jailbreak-samples` | train | (original, obfuscated-variant) pairs, no label column — both sides are unpivoted as positive (label=1) since the point of the dataset is evasion-robustness |
| `Smooth-3/llm-prompt-injection-attacks` | train | 49,500 rows, 52% attack, multi-label list collapsed to binary. **Nearly length-neutral** (median benign 307 / attack 357), which is why it was added — see below |
| `jayavibhav/prompt-injection` | train | 261,738 upstream, capped to 100k. 48.8% attack, same length profile as Smooth-3. 17.8% overlaps existing rows and is removed by dedup |
| `imoxto/prompt_injection_cleaned_dataset-v2` | train | 535,105 upstream, capped to 120k drawn **50/50** via `DatasetSpec.balance_labels` (upstream is 24.8% attack, and that skew measurably cost AUC) |
| `qualifire/prompt-injections-benchmark` (now `rogue-security/prompt-injections-benchmark`) | **test only** | 5,000 rows, held out entirely — never merged into train/val |

Not included: `jayavibhav/prompt-injection-safety` overlaps the existing
corpus **90.2%** — a repackaging rather than a new source.

**Corpus size is a deliberate choice, not a limit.** Four independent runs
measured it against the held-out benchmark:

| Corpus | Rows | Held-out AUC |
|---|---|---|
| Necent uncapped (97.8% one source) | 1,207,449 | 0.7501 |
| 7 sources, imoxto at native skew | 331,517 | 0.8835 |
| 7 sources, imoxto rebalanced 50/50 | 331,517 | 0.8929 |
| **5 sources, balanced** | **111,517** | **0.9168** |

More rows did not help; **source balance did**. Adding one length-neutral
source moved AUC 0.8278 → 0.9168 and FPR 0.363 → 0.208, while adding 1.1M
extra Necent rows cost 0.078 AUC. That is why `Necent` is capped at 30k and
`imoxto` is both capped and rebalanced.

`secureagentnet/detector/data_loader.py` normalizes all of them into one
schema (`text`, `label`, `category`, `source`), dedups by exact text hash,
and produces a stratified `{train, val, test}` split. The **qualifire
holdout is architecturally isolated**: `build_splits` only ever draws `test`
from `role="test"` specs, and `secureagentnet/tests/test_data_loader.py`
directly asserts no holdout rows leak into `train`/`val`.

`Necent` and `Mindgard` are marked `requires_auth=True` as a precaution —
if the Hub ever gates them behind a license click-through, `load_and_normalize`
catches the auth error, logs an actionable message (`huggingface-cli login`
/ `HF_TOKEN` + accept the license URL), and continues with whatever sources
did load rather than crashing the whole build.

### Regenerate the splits

```bash
# live from the Hub (slow, needs network + possibly HF auth for gated sources)
python -m secureagentnet.detector.data_loader

# from a locally pre-consolidated CSV (fast, e.g. one produced by your own
# consolidate.py merging the same four sources) — columns expected:
# text, label, attack_type, source_dataset, split
python -m secureagentnet.detector.data_loader --csv /path/to/consolidated_dataset.csv
```

Splits are cached to `secureagentnet/data/cache/splits.parquet` on the Hub
path; delete that file (or pass `use_cache=False`) to re-pull from the Hub.
The `--csv` path always reads fresh (it's already local and fast).

Note the CSV path preserves whatever label semantics your CSV encodes. If
your consolidation script derives Necent's label from `is_dangerous` (any
harmful content) rather than an injection-specific field, that broader
scope carries into the detector as-is — the loader only re-derives train/val
splits and the Necent sampling cap (`--necent-max-rows`, default 30k), not
the label itself. `qualifire`/`hf_csv2` rows are always routed to the
held-out `test` split regardless of what the CSV's own `split` column says
for other sources.

## Extended architecture (methodology doc)

Ten additional subsystems from the extended methodology doc, each mapped
to its own module and tested independently:

| §  | Subsystem | Module |
|---|---|---|
| 2.1 | Prompt provenance tracker | `provenance/tracker.py` — per-source-type base trust + per-identity EMA-adaptive trust |
| 2.3 | Behavioral anomaly detection | `simulate/behavioral_anomaly.py` — per-role baseline tool-set, deviation score |
| 2.4 | Adaptive risk engine | `correlation/adaptive_risk_engine.py` — N-signal weighted-sum fusion (swappable combiner) |
| 2.5 | Dynamic privilege governance | `privilege/dynamic_governance.py` — risk-reactive scope tightening + cascading revocation |
| 2.6 | Digital twin sandbox | `simulate/digital_twin.py` — stateful mock Inbox/Filesystem/Calendar backends |
| 2.7 | Memory protection layer | `privilege/memory_protection.py` — commit/quarantine/reject on risk + trust |
| 2.8 | Tamper-evident audit logs | `privilege/policy_engine.py`'s `AuditLog` — SHA-256 hash-chained entries, `verify_chain()` |
| 3 | Closed-loop adaptation | `correlation/closed_loop.py` — `CalibrationLayer` (EMA threshold) + `AttackMemoryIndex` (FAISS) |
| 4 | Adversarial red-teaming | `eval/red_team.py` — pluggable `AttackGenerator` (LLM or rule-based), generate→screen→detect→classify→update→feedback loop |
| 5 | Online retraining | `eval/online_retrain.py` — Track A/B split, versioning, regression-gated promotion/rollback |

**Red-team generator**: `LLMAttackGenerator` calls an OpenAI-compatible
endpoint configured via `TOKENROUTER_BASE_URL`/`TOKENROUTER_API_KEY`/
`TOKENROUTER_MODEL` in a git-ignored `.env` (never hardcoded/logged);
`RuleBasedAttackGenerator` is a deterministic, network-free fallback used
in tests. A real 3-round live run against the trained detector (using the
LLM generator) caught 100% of the variants that completed within the
endpoint's response time — 0% evasion rate — with graceful fallback to the
rule-based path on the endpoint's own timeouts.

**Known environment issue**: importing `faiss` and `torch` in the same
process reliably segfaults on macOS (OpenMP runtime conflict) —
`correlation/closed_loop.py` sets `KMP_DUPLICATE_LIB_OK`/`OMP_NUM_THREADS`
at import time as a fix; `tests/conftest.py` sets the same as a backstop.
The same duplicate-OpenMP hazard exists on Windows (torch ships
`libiomp5md.dll`, the `faiss-cpu` wheel links its own), so those two
environment variables stay set unconditionally rather than macOS-gated.

## §6 evaluation deliverables

`secureagentnet/eval/latency.py` measures real per-call latency (mean/p50/
p95/p99) for the detector alone, privilege check alone, and the full fused
pipeline, against an "undefended" no-op reference — see
`secureagentnet/reports/latency.json` for the measured numbers.

| Stage | macOS / MPS (upstream) | Windows / RTX 5070 (this port) |
|---|---|---|
| Framework-fused (mean) | 12.8 ms | **3.29 ms** |
| Detection-only (mean) | 20.8 ms | 3.90 ms |
| Privilege-only (mean) | 0.0012 ms | 0.0014 ms |

Both at 50 calls/stage against v3. The privilege check is negligible either
way — pure Python and Pydantic validation; the detector's forward pass is
what dominates end-to-end latency.
`overhead_pct_framework_vs_undefended` is reported but is astronomically
large by construction (undefended is ~0.00004 ms) — quote the absolute
millisecond figures, not that percentage.

Three figures in `secureagentnet/reports/figures/`, generated by
`scripts/generate_report_artifacts.py` against the real trained
checkpoints:
- **Figure 1** — evasion rate across 8 red-team rounds (v3, mean across 8
  seed attacks): converges to ~0 after round 0, showing calibration +
  memory index closing the gap within a single run.
- **Figure 2** — evasion rate v1 vs v3 under the identical red-team
  protocol. Upstream reported **6.5% → 1.3%**, a ~5× reduction from the
  Track B cycle. **This did not reproduce on the port**: measured here it
  inverts, v1 0.000 → v3 0.014. The reason is visible in the logs — the
  basic rule-based generator scores 0% against v1, so there is no headroom
  to improve, while the 8 real evasions were found against v2 with the
  *strengthened* generator. Fine-tuning on that narrow set shifts v3's
  boundary and it misses ~1.4% of basic variants v1 caught. On the
  targeted weakness v3 is clearly better (7 of 8 original evasions now
  caught, holdout F1 0.6941 → 0.7153), so this is a narrow-fine-tuning
  trade-off rather than a regression in the cycle itself. The regression
  gate did not catch it because it checks F1 only, not red-team robustness.
- **Figure 3** — utility vs ASR frontier across `block_risk_threshold`
  0.3–0.9 on the full qualifire test set. Upstream: ASR 4.8%→13.0%,
  utility 31.6%→56.8%. Measured here the shape replicates cleanly —
  ASR 8.1%→13.6%, utility 57.8%→63.5% — the same monotonic tunable
  surface, shifted up in utility because this port's detector has a lower
  FPR. Per-request latency (~3.3 ms here) is roughly constant across
  thresholds, since the detector forward pass dominates and does not
  depend on where the threshold sits.

`eval/red_team.py`'s `StoppingCondition` implements §5.6's three stopping
modes (`fixed_rounds`, `evasion_rate_threshold` with a configurable
consecutive-rounds requirement, `eval_window`), with a hard `max_rounds`
safety cap under every mode.

`simulate/digital_twin.py` covers all six roles' tools (an earlier version
of this README claimed this before `research_agent`'s tools actually had a
twin — that gap is now closed via `WebTwin`), including three **stateful
cross-call checks** the privilege layer's per-call ABAC conditions
genuinely can't express: `CodeExecTwin` catches cumulative CPU time across
a session exceeding a cap even when each individual `timeout_seconds` is
small; `SupportTwin` catches cumulative refunds on one order exceeding a
cap even when each individual `issue_refund` call is within the per-call
limit; `WebTwin` catches fetching too many *distinct* domains in one
session (a crawl/exfil pattern) even when every individual `fetch_url`
call matches the role's `["https://*"]` resource pattern fine.

## GUI

`secureagentnet/webapp/` is a local Flask app: enter a prompt, pick an
agent role/tool, and see the real fused decision (Allow/Flag/Block) with
the full signal breakdown — including a plain-language reason when it's
blocked, e.g. *"Blocked before reaching the agent: risk_score 0.963 >
block_risk_threshold 0.7"*. From there:

- **Run Red-Team Loop on this Prompt** — runs a live red-team round
  (`RuleBasedAttackGenerator`) against that exact prompt, using the
  server's real, running `CalibrationLayer`/`AttackMemoryIndex` — any
  evasions found actually adjust the live threshold and get added to
  memory, visible in the status bar (`calibration threshold=...`,
  `memory index size=...`).
- **Unlearn this Red-Team Session** — reverts exactly what that one
  red-team run changed: the calibration threshold snaps back to its
  pre-session value (`CalibrationLayer.restore()`, not another EMA step)
  and every memory entry that session added is removed
  (`AttackMemoryIndex.remove_texts()`), rebuilding the FAISS index. This
  is scoped per red-team session, not a full system reset — other
  sessions' additions are untouched.

```bash
python -m secureagentnet.webapp.app
# open http://127.0.0.1:5050
```

Requires a trained checkpoint at `secureagentnet/data/models/v3/` (or set
`SECUREAGENTNET_MODEL_DIR` to point elsewhere).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest secureagentnet/tests
```

## Windows Installation

The project runs natively on Windows 10/11 — no WSL, Docker, or Unix shell
required. `pyproject.toml` requires Python **3.11+**; the commands below use
3.13, which is what this port was verified against.

### Quick start (one command)

```powershell
git clone https://github.com/MP-GOWTHAM/SecureAgentNet.git
cd SecureAgentNet
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_windows.ps1
```

Checkpoints are pulled from
[mpgowtham/secureagentnet-models](https://huggingface.co/mpgowtham/secureagentnet-models)
(public, 367 MB) — no Hugging Face login needed for this path.

Creates the venv, installs dependencies, installs a CUDA build of torch,
downloads the trained checkpoints, writes the `combined_max` config, and
runs the test suite. Flags: `-SkipCuda` (no NVIDIA GPU), `-SkipModels`
(environment only), `-CudaIndex` (see the GPU table below).

**A clone alone is not runnable**, and that is deliberate. Three things are
excluded from git:

| Excluded | Size | Why |
|---|---|---|
| `secureagentnet/data/models/` | 1.6 GB | Five checkpoints are 253 MB each; GitHub hard-rejects files over 100 MB |
| `data/` | 191 MB | Regenerable from Hugging Face by the `scripts/build_*_dataset.py` builders |
| `.venv/` | ~5 GB | Local environment |

The web app loads a checkpoint at startup, so it will not run until one is
installed. Two ways to get one:

**Download** (default, a few minutes) — the bootstrap script fetches them
from `mpgowtham/secureagentnet-models`. Three checkpoints are published:

| Checkpoint | Size | Held-out AUC |
|---|---|---|
| `ensemble_v4_persona` | 49 MB | 0.8278 |
| `v3` (DistilBERT, post-Track B) | 253 MB | 0.7875 |
| `harm_detector` | 49 MB | 0.9028 |

`combined_max` — the recommended runtime configuration — is a config
referencing the first two, written locally by the bootstrap script rather
than downloaded, so the weights are never duplicated.

To publish your own copy instead (needs a **write**-scope token):

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe scripts\publish_models.py --repo-id <user>/secureagentnet-models
```

then pass `-ModelsRepo <user>/secureagentnet-models` to the bootstrap script.

**Or train from scratch** (~45 min on an RTX 5070, needs a Hugging Face
login and acceptance of three dataset licences) — see
*Training from scratch* below.

### PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

If `Activate.ps1` is blocked by execution policy, either run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current
session, or use the CMD activation script below.

### Command Prompt

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

### Scripted setup

`scripts/setup_windows.ps1` and `scripts/setup_windows.bat` do all of the
above (venv → pip upgrade → install → test run) in one step:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

### Windows dependency notes

`requirements-windows.txt` is `requirements.txt` plus four packages the code
imports but the original file never listed: **`faiss-cpu`** (the Windows
wheel name for `faiss`, used by `correlation/closed_loop.py`; there is no
Windows GPU wheel on PyPI), **`matplotlib`**, **`python-dotenv`**, and
**`pyarrow`** (for the `.parquet` split cache). No packages were removed.

`torch` installs the **CPU** build by default on Windows — `pip install
torch` gives you `2.x+cpu` and `torch.cuda.is_available()` returns `False`
even with a working NVIDIA driver. For GPU training you must install from
PyTorch's CUDA index explicitly:

```powershell
pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-windows.txt
```

Pick the CUDA index to match your GPU's compute capability:

| GPU generation | Compute capability | Index URL |
|---|---|---|
| Blackwell (RTX 50-series, e.g. 5070) | `sm_120` | `.../whl/cu128` or newer — **cu124 and older will not work** |
| Ada / Ampere (RTX 40/30-series) | `sm_89` / `sm_86` | `.../whl/cu124` |

Verify with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Two notes from doing this on an RTX 5070:

- Installing the CUDA wheel may **downgrade** `torch` (the cu128 index
  lagged the default index by two minor versions). This is expected.
- It also pulls a newer `fsspec` than `datasets` allows, printing a
  dependency-conflict warning. Re-pin afterwards, then confirm the
  environment is clean:

  ```powershell
  pip install "fsspec[http]<=2026.6.0"
  pip check
  ```

`pick_device()` returns `cuda` when available and falls back to `cpu` on
Windows (MPS is macOS-only and is now gated behind `platform.system()`).

## Running everything

Every command below assumes the venv is active (`.\.venv\Scripts\Activate.ps1`)
or is prefixed with `.\.venv\Scripts\python.exe`. Nothing here needs a
Hugging Face login except the dataset builders and model publishing.

| I want to… | Command |
|---|---|
| Set up from a fresh clone | `powershell -ExecutionPolicy Bypass -File scripts\bootstrap_windows.ps1` |
| Start the web app | `python -m secureagentnet.webapp.app` |
| Run the tests | `pytest secureagentnet/tests` |
| Evaluate a model | `python -m secureagentnet.eval.run_eval --model-dir <dir> --strict-privilege` |
| Check the short-attack blind spot | `python scripts\probe_short_attacks.py` |
| Measure each ensemble branch | `python scripts\ablate_branches.py` |
| Compare combination rules | `python scripts\combine_detectors.py` |
| Re-tune the block threshold | `python scripts\tune_block_threshold.py --model-dir <dir> --match-model <ref>` |
| Regenerate report figures | `python scripts\generate_report_artifacts.py` |

### 1. Start the web application

```powershell
$env:SECUREAGENTNET_MODEL_DIR = "$PWD\secureagentnet\data\models\combined_max"
python -m secureagentnet.webapp.app
```

Then open <http://127.0.0.1:5050>. Omit the env var to use the default
(`secureagentnet\data\models\v3`, resolved **relative to the repo**).
`combined_max` is the recommended configuration — see *Which model to run*.

### 2. Tests

```powershell
pytest secureagentnet/tests
pytest secureagentnet/tests -q -k ensemble
```

### 3. Evaluation

The headline table (ASR / C-ASR / FPR / FNR / utility) on the held-out
qualifire benchmark:

```powershell
python -m secureagentnet.eval.run_eval --csv data\consolidated_dataset.csv --model-dir secureagentnet\data\models\combined_max --strict-privilege
```

Useful flags: `--strict-privilege` (hard-blocks out-of-scope calls, drives
C-ASR to 0), `--block-risk-threshold 0.50` (for a temperature-calibrated
detector — the 0.85 default assumes DistilBERT's uncalibrated scores),
`--max-examples 500` (quick smoke run).

### 4. Behavioural probes

Aggregate metrics miss both of the failure modes this project actually hit,
so these two run in seconds and are worth putting in CI:

```powershell
python scripts\probe_short_attacks.py --verbose
python scripts\compare_evasions_ensemble.py
```

The first reports how many of the 8 canonical short attacks each model
catches plus a *dilution gap* (near zero means the model reads the attack,
not the text length). The second scores the 8 real evasions found by
red-teaming against every checkpoint.

### 5. Analysis

```powershell
python scripts\ablate_branches.py
python scripts\combine_detectors.py
python scripts\tune_block_threshold.py --model-dir secureagentnet\data\models\ensemble_v4_persona --match-model secureagentnet\data\models\v3
```

Results are written to `secureagentnet\reports\*.json`.

### 6. Red-team and the Track B retraining cycle

```powershell
python scripts\run_track_b.py
python scripts\find_evasions.py
python scripts\run_track_b_v3.py
```

`find_evasions.py` respects `SECUREAGENTNET_MODEL_DIR` (which checkpoint to
attack) and `SECUREAGENTNET_RUN_DIR` (where `evasions.json` is written —
override it or you will overwrite the reference set the v3 cycle consumes).

The web UI exposes the same loop: submit a prompt that is blocked or
flagged, click **Run red-team loop**, then **Unlearn** to revert the
calibration threshold and memory-index changes it makes.

### 7. Report artifacts

```powershell
python scripts\generate_report_artifacts.py
```

Writes `secureagentnet\reports\figures\figure{1,2,3}*.png`, `latency.json`
and `frontier.json`. Needs the `injection_detector` and `v3` checkpoints.

### 8. Datasets

Needs a Hugging Face login and acceptance of the three gated licences
(see **Datasets** above).

```powershell
python scripts\build_consolidated_dataset.py
python scripts\build_harm_dataset.py --n-per-class 30000 --n-mix-benign 12000
python scripts\build_rebalanced_dataset.py
python -m secureagentnet.detector.data_loader
```

### 9. Publishing checkpoints

```powershell
.\.venv\Scripts\hf.exe auth login
python scripts\publish_models.py --repo-id <user>/secureagentnet-models --dry-run
python scripts\publish_models.py --repo-id <user>/secureagentnet-models
```

### Which model to run

| Goal | `SECUREAGENTNET_MODEL_DIR` | Trade-off |
|---|---|---|
| **Maximum security** (recommended) | `combined_max` | FNR 0.035, 8/8 short attacks, 8/8 evasions; FPR 0.431 |
| Maximum utility | `ensemble_v5_fpr` | FPR 0.328, utility 0.672; misses 1 short attack |
| Best single model | `ensemble_v4_persona` | AUC 0.8278; misses 1 short attack |
| Comparable with prior work | `v3` | Misses the dilution evasion |

Set `SECUREAGENTNET_HARM_MODEL_DIR` to point the content-harm classifier
elsewhere; it defaults to `secureagentnet\data\models\harm_detector` and is
skipped silently if absent.

Full measurements for every configuration, including the ones that failed,
are in `docs\SecureAgentNet_Detector_Architecture.docx` §8.

### Scripts under `scripts/`

The one-off analysis scripts no longer hardcode absolute macOS paths. They
resolve the repo root from their own location and read two optional
environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SECUREAGENTNET_CSV` | `<repo>\data\consolidated_dataset.csv` | consolidated dataset CSV |
| `SECUREAGENTNET_RUN_DIR` | `%TEMP%\secureagentnet_run` | scratch dir for intermediate artifacts (was `/tmp`) |

```powershell
$env:SECUREAGENTNET_CSV = "C:\path\to\consolidated_dataset.csv"
python scripts\generate_report_artifacts.py
```

### Training from scratch

Needed only if you are not downloading checkpoints. Requires a Hugging Face
login (`hf auth login`) and acceptance of the three gated dataset licences
listed under **Datasets**.

```powershell
.\.venv\Scripts\python.exe scripts\build_consolidated_dataset.py
.\.venv\Scripts\python.exe -m secureagentnet.detector.train_ensemble --csv data\consolidated_dataset.csv --epochs 1 --augment --n-persona-benign 6000 --output-dir secureagentnet\data\models\ensemble_v4_persona
.\.venv\Scripts\python.exe -m secureagentnet.detector.train --csv data\consolidated_dataset.csv --epochs 3 --output-dir secureagentnet\data\models\v3
.\.venv\Scripts\python.exe scripts\build_harm_dataset.py --n-per-class 30000 --n-mix-benign 12000
.\.venv\Scripts\python.exe -m secureagentnet.detector.train_ensemble --csv data\harm_dataset.csv --epochs 1 --n-persona-benign 6000 --output-dir secureagentnet\data\models\harm_detector
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_windows.ps1 -SkipModels -SkipCuda
```

The last line writes `combined_max` from the two members and verifies the
suite. **One epoch is intentional** — more epochs lower held-out AUC
(0.809 at 1 epoch, 0.770 at 20) and no in-distribution signal detects it;
see §5.1 of `docs/SecureAgentNet_Detector_Architecture.docx`.

Judge a training run by `metrics.json` and the log tail, not the exit code:
on Windows the process exits `0xC0000409` after a fully successful run (a
native DLL-unload fault after all artifacts are flushed).

`pick_device()` picks `cuda` when available and falls back to `cpu` on
Windows (MPS is macOS-only and is now gated behind `platform.system()`).
