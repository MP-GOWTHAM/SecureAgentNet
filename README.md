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

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest secureagentnet/tests
```
