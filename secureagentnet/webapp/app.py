"""Local GUI for SecureAgentNet: enter a prompt, submit it, done. Role
detection, the fused decision, red-teaming that same prompt, and
unlearning the red-team session's effect afterward all happen server-side
in one request (`Pipeline.submit`) — no separate controls in the GUI for
any of that.

Run with: python -m secureagentnet.webapp.app
Then open http://127.0.0.1:5050
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Repo root: .../secureagentnet/webapp/app.py -> up 3.
REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass  # TOKENROUTER_* just won't be set -- LLM red-teaming falls back to rule-based

from flask import Flask, jsonify, request, send_from_directory

import torch

from secureagentnet.correlation.adaptive_risk_engine import RiskSignals
from secureagentnet.correlation.closed_loop import AttackMemoryIndex, CalibrationConfig, CalibrationLayer
from secureagentnet.correlation.fusion import FusionAction, FusionEngine
from secureagentnet.detector.model import InjectionRiskModel, load_tokenizer
from secureagentnet.detector.train import pick_device
from secureagentnet.eval.red_team import LLMAttackGenerator, RuleBasedAttackGenerator, StoppingCondition, run_red_team_loop
from secureagentnet.privilege.memory_protection import MemoryProtectionLayer, MemoryWriteRequest
from secureagentnet.privilege.policy_engine import POLICIES_DIR, PolicyEngine, ToolCallRequest, issue_credential
from secureagentnet.simulate.agent_env import MEMORY_WRITE_TOOLS, ROLE_SOURCE_TYPE, ROLES, _build_provenance_tag
from secureagentnet.simulate.behavioral_anomaly import ActionSequence, BehavioralAnomalyDetector
# DigitalTwinSandbox is deliberately not imported: the twin is no longer
# part of the webapp's fused decision. simulate/digital_twin.py and its
# tests remain, and eval/run_eval.py never used the twin either, so this
# only removes it from the live product surface.
from secureagentnet.provenance.tracker import ProvenanceTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("webapp")

MODEL_DIR = os.environ.get(
    "SECUREAGENTNET_MODEL_DIR",
    # combined_gated_v7: max(ensemble_v6_smooth3, v3) with the secondary
    # gated at 0.95. Catches 8/8 canonical short attacks and 8/8 real
    # evasions -- the coverage plain max gives -- at FPR 0.368 rather than
    # 0.415 and utility 0.654 rather than 0.621.
    #
    # Falls back to v3 if that config is absent, so a checkout without the
    # downloaded checkpoints still starts against whatever is present.
    str(REPO_ROOT / "secureagentnet" / "data" / "models" / "combined_gated_v7"),
)
if not (Path(MODEL_DIR) / "config.json").exists():
    _fallback = REPO_ROOT / "secureagentnet" / "data" / "models" / "v3"
    if (_fallback / "config.json").exists():
        MODEL_DIR = str(_fallback)
HARM_MODEL_DIR = Path(os.environ.get(
    "SECUREAGENTNET_HARM_MODEL_DIR",
    # v3 is the two-source classifier: wider margin (harmful mean 0.769 vs
    # 0.652) and no false fires on benign roleplay, which matters because
    # block_harm_threshold is a hard block rather than a flag.
    str(REPO_ROOT / "secureagentnet" / "data" / "models" / "harm_detector_v3"),
))
if not (HARM_MODEL_DIR / "config.json").exists():
    _harm_fallback = REPO_ROOT / "secureagentnet" / "data" / "models" / "harm_detector"
    if (_harm_fallback / "config.json").exists():
        HARM_MODEL_DIR = _harm_fallback
CREDENTIAL_TTL_SECONDS = 24 * 3600

# Sensible default (tool_name, resource) per role — the tool call that
# role's own prompt gets evaluated against once auto-detection (below)
# picks a role, since the GUI no longer asks the user to specify one.
ROLE_DEFAULT_ACTION = {
    "email_agent": {"tool_name": "send_email", "resource": "alice@secureagentnet-corp.com", "params": {"recipient_count": 1}},
    "file_agent": {"tool_name": "write_file", "resource": "/workspace/notes/summary.md", "params": {}},
    "research_agent": {"tool_name": "save_note", "resource": "/workspace/notes/research.md", "params": {}},
    "calendar_agent": {"tool_name": "create_event", "resource": None, "params": {"attendee_count": 2}},
    "code_exec_agent": {"tool_name": "run_code", "resource": "/sandbox/script.py", "params": {"timeout_seconds": 5}},
    "support_agent": {"tool_name": "issue_refund", "resource": "order_1", "params": {"amount_usd": 20}},
}

# Keyword -> role, used to auto-detect which agent role a prompt is most
# plausibly directed at, since the GUI collects only the prompt text now.
# Deliberately simple keyword matching, not a classifier — this project
# already has one real ML model (the injection detector); guessing intent
# from keywords is an honest, inspectable heuristic for a demo GUI, not a
# second model pretending to understand the prompt.
#
# Audited once already for two distinct failure modes, both real bugs
# found by testing actual prompts against the running GUI, not caught by
# reasoning about the list in the abstract:
#
# 1. SUBSTRING COLLISIONS — plain `kw in text` matches a keyword occurring
#    *inside* an unrelated word: "mail" inside "blackmail", "script"
#    inside "description"/"subscription", "order" inside
#    "murder"/"disorder"/"border", "note" inside "denote". Fixed by
#    anchoring every match to a LEFT word boundary (`\bkeyword`, not
#    `\bkeyword\b`) — left-only, not both sides, because several keywords
#    are deliberate stems meant to match inflected forms ("infiltrat" ->
#    "infiltrate"/"infiltration", "hack" -> "hacker"/"hacking"/"hacked").
#    A left boundary blocks "blackmail" (no boundary between "black" and
#    "mail") while still matching "mailbox" (boundary at the true start)
#    and "infiltrating" (suffix after the stem is fine).
#
# 2. GENERIC/IDIOMATIC WHOLE-WORD FALSE POSITIVES — some keywords are
#    just too common as ordinary English regardless of boundaries, and
#    happen to be common jailbreak-framing words specifically: "write"
#    fires on "write a story/tutorial/guide about X" (one of the most
#    common jailbreak framings there is, nothing to do with file
#    operations), "forward" fires on "going forward"/"look forward to",
#    "event" fires on "in the event of X", "run" is an extremely generic
#    verb, "order" fires on "in order to". No regex fixes a word being
#    legitimately, commonly used with an unrelated meaning — these were
#    removed outright rather than pattern-matched around.
ROLE_KEYWORDS = {
    "email_agent": ["email", "inbox", "mail"],
    "file_agent": ["file", "workspace", "delete", "document"],
    "research_agent": ["search", "web", "url", "fetch", "research", "note", "browse"],
    "calendar_agent": ["calendar", "meeting", "schedule", "invite"],
    "code_exec_agent": [
        "code", "script", "execute", "sandbox", "python",
        # Attack/roleplay-framed prompts ("develop a hacking strategy",
        # "infiltrate a database") don't mention any tool by name but are
        # squarely about compromising a system -- code_exec_agent is the
        # tightest-scoped, least-trusted role of the six (see
        # privilege/policies/code_exec_agent.yaml and simulate/digital_twin.py's
        # own "highest-value target" framing), so routing this kind of
        # prompt there is the safer default than accidentally landing on a
        # more permissive role that happens to match no keywords either.
        "hack", "hacker", "hacking", "infiltrat", "exploit", "malware",
        "breach", "database", "steal", "backdoor", "vulnerability",
    ],
    "support_agent": ["refund", "ticket", "support", "customer"],
}
# Used only when NO keyword matches anything above -- kept distinct from
# code_exec_agent's attack-keyword list so the UI can tell "we recognized
# this as attack-shaped and routed it to the tightest role" apart from
# "we genuinely don't know, this is a guess" (see detect_role's return).
DEFAULT_ROLE = "email_agent"


def detect_role(prompt: str) -> tuple[str, bool]:
    """Returns (role, matched) — `matched` is False only when nothing in
    ROLE_KEYWORDS hit and DEFAULT_ROLE was used as a last resort, so
    callers (and the GUI) can distinguish a real keyword-based detection
    from an honest "couldn't tell" fallback rather than presenting both
    with the same false confidence.

    Matches are left-word-boundary-anchored (see ROLE_KEYWORDS' comment
    for why left-only, not `\\b...\\b`).
    """
    lowered = prompt.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(kw), lowered) for kw in keywords):
            return role, True
    return DEFAULT_ROLE, False

app = Flask(__name__, static_folder=None)


class Pipeline:
    """Process-lifetime state — one running instance of the fused
    pipeline, matching how a real deployed agent would hold one detector,
    one calibration layer, one memory index, etc. across many requests
    rather than rebuilding them per call.
    """

    def __init__(self):
        logger.info("Loading detector from %s ...", MODEL_DIR)
        self.device = pick_device()
        self.model = InjectionRiskModel.load(MODEL_DIR, map_location=str(self.device)).to(self.device)
        self.tokenizer = load_tokenizer(self.model.config.model_name)
        self.model.eval()

        self.policy_engine = PolicyEngine.from_directory(POLICIES_DIR)
        self.credential_by_role = {
            role: issue_credential(role, ttl_seconds=CREDENTIAL_TTL_SECONDS) for role in ROLES
        }
        self.fusion_engine = FusionEngine()
        self.calibration = CalibrationLayer(CalibrationConfig(initial_threshold=0.5))
        self.memory_index = AttackMemoryIndex(dim=768)
        self.provenance_tracker = ProvenanceTracker()
        self.memory_protection = MemoryProtectionLayer()
        self.behavioral_detector = BehavioralAnomalyDetector()
        self.redteam_sessions: dict[str, dict] = {}

        # Content-harm classifier: a second, independent axis. The
        # injection detector cannot see a plainly harmful request that
        # carries no instruction-override framing (measured: "synthesize a
        # nerve agent at home" scores 0.04 as an injection, 0.68 as harm).
        # Optional -- if the checkpoint is absent the signal stays 0.0 and
        # every decision is exactly what it was before.
        self.harm_model = None
        self.harm_tokenizer = None
        if HARM_MODEL_DIR.exists() and (HARM_MODEL_DIR / "config.json").exists():
            try:
                self.harm_model = InjectionRiskModel.load(
                    str(HARM_MODEL_DIR), map_location=str(self.device)).to(self.device)
                self.harm_model.eval()
                self.harm_tokenizer = load_tokenizer(self.harm_model.config.model_name)
                logger.info("content-harm classifier loaded from %s", HARM_MODEL_DIR)
            except Exception as exc:  # noqa: BLE001 - never let this break startup
                logger.warning("content-harm classifier unavailable (%s); signal stays 0.0", exc)
        else:
            logger.info("no content-harm checkpoint at %s; signal stays 0.0", HARM_MODEL_DIR)
        logger.info("Pipeline ready on device %s.", self.device)

    def harm_score(self, text: str) -> float:
        """Content-harm probability, or 0.0 when no classifier is loaded."""
        if self.harm_model is None:
            return 0.0
        enc = self.harm_tokenizer(
            [text], padding=True, truncation=True,
            max_length=self.harm_model.config.max_length, return_tensors="pt",
        )
        with torch.no_grad():
            return float(self.harm_model.risk_score(
                enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))[0])

    def score(self, text: str) -> float:
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=self.model.config.max_length, return_tensors="pt")
        with torch.no_grad():
            return float(self.model.risk_score(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))[0])

    def embed(self, text: str):
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=self.model.config.max_length, return_tensors="pt")
        with torch.no_grad():
            vec = self.model.embed(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))[0]
        return vec.cpu().numpy()

    def analyze_text_only(self, prompt: str) -> dict:
        """Classify the text when no role keyword matched.

        Previously an unmatched prompt was forced through DEFAULT_ROLE
        ("email_agent"), which fabricated a tool call nobody asked for:
        `send_email` to a fixed address. That invented action then failed
        the digital twin ("recipient does not match any known contact")
        and produced a behavioural anomaly of 1.0 — noise generated
        entirely by the guess, on prompts as innocuous as "Summarize the
        quarterly sales report".

        With no role there is no agent action to govern, so only the text
        can be judged. Privilege, digital twin, provenance and behavioural
        analysis are skipped rather than fed a placeholder, and the fused
        decision runs on the detector signal alone. RiskSignals' own
        defaults are the neutral "nothing else is known" values
        (source_trust 1.0, twin_unsafe False), so nothing is invented here.
        """
        risk_score = self.score(prompt)
        harm = self.harm_score(prompt)
        signals = RiskSignals(injection_score=risk_score, content_harm=harm)
        fusion_result = self.fusion_engine.fuse_signals(signals)

        if fusion_result.action == FusionAction.BLOCK:
            conflict_message = f"Blocked on prompt content: {fusion_result.reason}"
        elif fusion_result.action == FusionAction.FLAG:
            conflict_message = f"Flagged on prompt content: {fusion_result.reason}"
        else:
            conflict_message = "No injection detected in the prompt."

        return {
            "action": fusion_result.action.value,
            "proceeds": fusion_result.action != FusionAction.BLOCK,
            "conflict_message": conflict_message,
            "reason": fusion_result.reason,
            "blended_risk_score": fusion_result.risk_score,
            "raw_injection_score": risk_score,
            # Flags the reduced analysis so the UI can say "not evaluated"
            # instead of rendering defaults as if they were measurements.
            "governance_evaluated": False,
            "signals": {
                "injection_score": signals.injection_score,
                "content_harm": harm if self.harm_model is not None else None,
                "behavior_anomaly": None,
                "source_trust": None,
                "privilege_out_of_scope": None,
            },
            "request": None,
            "privilege_decision": None,
            "memory_write_decision": None,
        }

    def analyze(self, prompt: str, role: str, tool_name: str, resource: str | None, params: dict) -> dict:
        now = datetime.now(timezone.utc)
        request_obj = ToolCallRequest(tool_name=tool_name, resource=resource, params=params or {})
        credential = self.credential_by_role[role]

        risk_score = self.score(prompt)
        privilege_decision = self.policy_engine.authorize(credential, request_obj, now=now)

        tag = _build_provenance_tag(role, request_obj, prompt)
        source_trust = self.provenance_tracker.trust_score(tag)

        anomaly = self.behavioral_detector.score(ActionSequence(role=role, tool_names=[tool_name]))

        # Digital-twin sandbox intentionally not part of the fused signal
        # here. The evaluation harness (eval/run_eval.py) never passed the
        # twin either -- every ASR/C-ASR/utility number in the report was
        # measured without it -- so leaving the twin out of the webapp's
        # fusion aligns the live product with what was actually measured.
        # `twin_unsafe` defaults to False on RiskSignals, so the twin's
        # weight in adaptive_risk_engine (0.015) contributes zero.
        signals = RiskSignals(
            injection_score=risk_score,
            behavior_anomaly=anomaly.deviation_score,
            source_trust=source_trust,
            privilege_out_of_scope=privilege_decision.out_of_scope,
            session_history_risk=0.0,
            content_harm=self.harm_score(prompt),
        )
        fusion_result = self.fusion_engine.fuse_signals(signals)

        memory_write_decision = None
        if tool_name in MEMORY_WRITE_TOOLS:
            memory_write_decision = self.memory_protection.evaluate(
                MemoryWriteRequest(content=prompt, tag=tag, risk_score=risk_score), trust_score=source_trust,
            )

        proceeds = fusion_result.action != FusionAction.BLOCK
        if fusion_result.action == FusionAction.BLOCK:
            conflict_message = f"Blocked before reaching the agent: {fusion_result.reason}"
        elif fusion_result.action == FusionAction.FLAG:
            conflict_message = f"Allowed, but flagged for review: {fusion_result.reason}"
        else:
            conflict_message = "No conflict — proceeding to agent."

        return {
            "action": fusion_result.action.value,
            "proceeds": proceeds,
            "conflict_message": conflict_message,
            "reason": fusion_result.reason,
            "blended_risk_score": fusion_result.risk_score,
            "raw_injection_score": risk_score,
            "governance_evaluated": True,
            "signals": {
                "injection_score": signals.injection_score,
                "content_harm": signals.content_harm if self.harm_model is not None else None,
                "behavior_anomaly": signals.behavior_anomaly,
                "source_trust": signals.source_trust,
                "privilege_out_of_scope": signals.privilege_out_of_scope,
            },
            "privilege_decision": {
                "allowed": privilege_decision.allowed,
                "violation_type": privilege_decision.violation_type.value,
                "reason": privilege_decision.reason,
            },
            "memory_write_decision": (
                {"outcome": memory_write_decision.outcome.value, "reason": memory_write_decision.reason}
                if memory_write_decision else None
            ),
            "request": {"tool_name": tool_name, "resource": resource, "role": role},
        }

    def _make_generator(self):
        """Prefers the real LLM generator (TOKENROUTER_* from .env) since
        it produces genuinely novel paraphrases/obfuscations a fixed rule
        set can't -- falls back to RuleBasedAttackGenerator (deterministic,
        no network) if credentials aren't configured or construction fails,
        rather than the request erroring out.
        """
        try:
            return LLMAttackGenerator(), "llm"
        except ValueError as e:
            logger.warning("LLM generator unavailable (%s); falling back to rule-based.", e)
            return RuleBasedAttackGenerator(seed=int.from_bytes(os.urandom(2), "big")), "rule_based"

    def run_redteam(self, prompt: str, rounds: int, variants_per_round: int) -> dict:
        threshold_before = self.calibration.snapshot()
        generator, generator_kind = self._make_generator()

        results = run_red_team_loop(
            original_attack=prompt,
            generator=generator,
            score_fn=self.score,
            embed_fn=self.embed,
            memory_index=self.memory_index,
            calibration=self.calibration,
            stopping_condition=StoppingCondition(mode="fixed_rounds", max_rounds=rounds),
            n_variants_per_round=variants_per_round,
        )

        added_texts = [e for r in results for e in r.evasions]
        session_id = str(uuid.uuid4())
        self.redteam_sessions[session_id] = {
            "threshold_before": threshold_before,
            "added_texts": added_texts,
        }

        return {
            "redteam_session_id": session_id,
            "generator": generator_kind,
            "threshold_before": threshold_before,
            "threshold_after": self.calibration.threshold,
            "rounds": [
                {
                    "round_index": r.round_index,
                    "n_generated": r.n_generated,
                    "n_screened_out": r.n_screened_out,
                    "n_caught": r.n_caught,
                    "n_evaded": r.n_evaded,
                    "evasion_rate": r.evasion_rate,
                    "evasions": r.evasions,
                }
                for r in results
            ],
            "total_evasions": len(added_texts),
        }

    def unlearn(self, session_id: str) -> dict:
        session = self.redteam_sessions.pop(session_id, None)
        if session is None:
            return {"error": f"unknown redteam_session_id: {session_id}"}, 404

        self.calibration.restore(session["threshold_before"])
        n_removed = self.memory_index.remove_texts(set(session["added_texts"]))
        return {
            "threshold_restored_to": self.calibration.threshold,
            "n_removed": n_removed,
            "memory_index_size": len(self.memory_index),
        }

    def submit(self, prompt: str, rounds: int = 3, variants_per_round: int = 8) -> dict:
        """One call, one prompt in, everything else automatic: detect the
        role, run the full fused decision, red-team that exact prompt
        against the live detector, then immediately unlearn that
        red-team session so it leaves no lasting drift on shared state —
        this endpoint is meant to answer "is this prompt safe, and how
        robust is that verdict" for one prompt at a time, not to
        accumulate calibration/memory changes across unrelated callers'
        prompts the way a real multi-turn agent session would.
        """
        role, role_matched = detect_role(prompt)
        if role_matched:
            action_defaults = ROLE_DEFAULT_ACTION[role]
            analysis = self.analyze(
                prompt, role, action_defaults["tool_name"], action_defaults["resource"],
                dict(action_defaults["params"]),
            )
        else:
            # No keyword matched: judge the text, do not invent an agent
            # action to govern. See analyze_text_only().
            analysis = self.analyze_text_only(prompt)

        # Only red-team something the pipeline actually objected to.
        # run_red_team_round treats its input as an attack by construction:
        # any low-scoring variant is recorded via
        # confirm_outcome(is_malicious=True) and written into the memory
        # index. Running that over a benign prompt therefore labels benign
        # text malicious and drags the calibration threshold down. This
        # path auto-unlearns so the damage was transient, but it would
        # persist if unlearn ever failed -- and an ALLOW at low risk has
        # nothing worth probing anyway. Mirrors the rule the GUI already
        # applies before offering its red-team button.
        if analysis["action"] == FusionAction.ALLOW.value:
            redteam = None
            unlearn_result = None
        else:
            redteam = self.run_redteam(prompt, rounds, variants_per_round)
            unlearn_result = self.unlearn(redteam["redteam_session_id"])

        return {
            # None, not DEFAULT_ROLE, when nothing matched -- reporting a
            # role here while analyze_text_only() governed nothing would
            # contradict /api/analyze and re-introduce the "silently became
            # email_agent" confusion this path exists to avoid.
            "detected_role": role if role_matched else None,
            "role_matched": role_matched,
            "analysis": analysis,
            # null when the verdict was ALLOW and no red-team ran, so a
            # caller can tell "probed, found nothing" apart from "not
            # probed" -- both would otherwise look like zero evasions.
            "redteam": None if redteam is None else {
                "total_evasions": redteam["total_evasions"],
                "rounds": redteam["rounds"],
                "threshold_before": redteam["threshold_before"],
                "threshold_peak": redteam["threshold_after"],
            },
            "unlearn": unlearn_result,
        }


pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global pipeline
    if pipeline is None:
        pipeline = Pipeline()
    return pipeline


@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).resolve().parent / "static"), "index.html")


@app.route("/api/roles")
def api_roles():
    return jsonify({"roles": ROLES, "defaults": ROLE_DEFAULT_ACTION})


@app.route("/api/submit", methods=["POST"])
def api_submit():
    """The only endpoint the GUI calls now: prompt in, everything else
    (role detection, the fused decision, red-teaming, unlearning)
    automatic. /api/analyze, /api/redteam, /api/unlearn stay available
    below as separate endpoints for anyone driving the pipeline step by
    step directly (e.g. the comparison scripts, or a future UI that wants
    manual control back) — /api/submit is a thin orchestration on top of
    them, not a replacement.
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    rounds = int(data.get("rounds", 3))
    variants_per_round = int(data.get("variants_per_round", 8))
    result = get_pipeline().submit(prompt, rounds, variants_per_round)
    return jsonify(result)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Auto-detect only when the caller didn't specify a role explicitly --
    # same detect_role() the /api/submit path uses, so a bare {"prompt": ...}
    # call here doesn't fall back to a silently-wrong ROLES[0] default the
    # way this endpoint used to (the exact bug class fixed elsewhere in
    # this file for the /api/submit path).
    role_matched = True
    if "role" in data:
        role = data["role"]
        if role not in ROLES:
            return jsonify({"error": f"unknown role: {role}"}), 400
    else:
        role, role_matched = detect_role(prompt)

    # Nothing matched and the caller named no role: there is no agent
    # action to govern, so classify the text instead of inventing a
    # send_email to a fixed address. See Pipeline.analyze_text_only().
    if not role_matched:
        result = get_pipeline().analyze_text_only(prompt)
        result["detected_role"] = None
        result["role_matched"] = False
        return jsonify(result)

    action_defaults = ROLE_DEFAULT_ACTION[role]
    tool_name = data.get("tool_name") or action_defaults["tool_name"]
    resource = data.get("resource", action_defaults["resource"])
    params = data.get("params") or dict(action_defaults["params"])

    result = get_pipeline().analyze(prompt, role, tool_name, resource, params)
    result["detected_role"] = role
    result["role_matched"] = role_matched
    return jsonify(result)


@app.route("/api/redteam", methods=["POST"])
def api_redteam():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    rounds = int(data.get("rounds", 3))
    variants_per_round = int(data.get("variants_per_round", 8))
    result = get_pipeline().run_redteam(prompt, rounds, variants_per_round)
    return jsonify(result)


@app.route("/api/unlearn", methods=["POST"])
def api_unlearn():
    data = request.get_json(force=True)
    session_id = data.get("redteam_session_id")
    if not session_id:
        return jsonify({"error": "redteam_session_id is required"}), 400
    result = get_pipeline().unlearn(session_id)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@app.route("/api/status")
def api_status():
    p = get_pipeline()
    cfg = p.fusion_engine.config

    # Which checkpoint is live. There are now several on disk (DistilBERT
    # v1/v2/v3, the from-scratch ensembles, and the combined wrapper) and
    # they behave very differently, so "which model am I looking at" is
    # the first thing anyone needs from this screen.
    model_name = Path(MODEL_DIR).name
    model_kind = getattr(p.model.config, "kind", "distilbert")
    members = list(getattr(p.model.config, "members", []) or [])

    return jsonify({
        "device": str(p.device),
        "calibration_threshold": p.calibration.threshold,
        "memory_index_size": len(p.memory_index),
        "active_redteam_sessions": len(p.redteam_sessions),
        "model": {
            "name": model_name,
            "kind": model_kind,
            "members": [Path(m).name for m in members],
        },
        # The UI draws these as markers on the risk meters, so a config
        # change is reflected in the display instead of being hardcoded
        # in two places that can drift apart.
        "thresholds": {
            "block": cfg.block_risk_threshold,
            "flag": cfg.flag_risk_threshold,
            "block_combo": cfg.block_combo_risk_threshold,
        },
    })


def main():
    get_pipeline()  # load eagerly so the first request isn't slow
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
