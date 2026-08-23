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

Actual run against the trained checkpoint (5,000 qualifire examples):

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

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| `neuralchemy/Prompt-injection-dataset` | train | clean `text`/`label`/`category` schema, 29 attack categories |
| `Necent/llm-jailbreak-prompt-injection-dataset` | train | ~1.17M rows aggregating InjecAgent/ToolEmu/BIPIA/etc; sampled to 30k stratified rows for iteration speed (see `DatasetSpec.max_rows` in `detector/data_loader.py`) |
| `Mindgard/evaded-prompt-injection-and-jailbreak-samples` | train | (original, obfuscated-variant) pairs, no label column — both sides are unpivoted as positive (label=1) since the point of the dataset is evasion-robustness |
| `qualifire/prompt-injections-benchmark` (now `rogue-security/prompt-injections-benchmark`) | **test only** | 5,000 rows, held out entirely — never merged into train/val |

`secureagentnet/detector/data_loader.py` normalizes all four into one
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
`secureagentnet/reports/latency.json` for the measured numbers (v3, MPS,
50 calls/stage): framework-fused mean **12.8ms** (p50 10.1ms), detection-only
mean 20.8ms, privilege-only mean **0.0012ms** (negligible — pure Python/
Pydantic checks). `overhead_pct_framework_vs_undefended` is reported but
is astronomically large by construction (undefended is ~0.00004ms) —
report the absolute ms figures, not that percentage alone.

Three figures in `secureagentnet/reports/figures/`, generated by
`scripts/generate_report_artifacts.py` against the real trained
checkpoints:
- **Figure 1** — evasion rate across 8 red-team rounds (v3, mean across 8
  seed attacks): converges to ~0 after round 0, showing calibration +
  memory index closing the gap within a single run.
- **Figure 2** — evasion rate v1 vs v3 under the identical red-team
  protocol: **6.5% → 1.3%**, a real ~5x reduction from the Track B
  retraining cycle on genuine evasions. v2 isn't shown — its checkpoint
  was overwritten mid-run by the versioning bug (now fixed; see
  `eval/online_retrain.py`'s `RetrainRegistry.next_version()`).
- **Figure 3** — utility vs ASR frontier across `block_risk_threshold`
  0.3–0.9 on the full qualifire test set: ASR 4.8%→13.0% as threshold
  rises, utility 31.6%→56.8% — the tunable design surface the doc asks
  for, with per-request latency (~12.8ms) noted as roughly constant across
  thresholds since the detector forward pass dominates and doesn't depend
  on where the threshold is set.

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

### Run the tests

```powershell
pytest secureagentnet/tests
```

### Start the web application

```powershell
python -m secureagentnet.webapp.app
```

Then open <http://127.0.0.1:5050>. The model directory now defaults to
`secureagentnet\data\models\v3` **relative to the repo**, so no path editing
is needed; `SECUREAGENTNET_MODEL_DIR` still overrides it.

### Run the evaluation

```powershell
python -m secureagentnet.eval.run_eval --csv C:\path\to\consolidated_dataset.csv
python -m secureagentnet.eval.run_eval --csv C:\path\to\consolidated_dataset.csv --strict-privilege
```

### Regenerate the splits

```powershell
python -m secureagentnet.detector.data_loader
python -m secureagentnet.detector.data_loader --csv C:\path\to\consolidated_dataset.csv
```

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

`pick_device()` picks `cuda` when available and falls back to `cpu` on
Windows (MPS is macOS-only and is now gated behind `platform.system()`).
