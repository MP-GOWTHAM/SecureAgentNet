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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
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
from secureagentnet.simulate.digital_twin import DigitalTwinSandbox
from secureagentnet.provenance.tracker import ProvenanceTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("webapp")

MODEL_DIR = os.environ.get(
    "SECUREAGENTNET_MODEL_DIR",
    "/Users/gowtham/Desktop/secureagentnet/secureagentnet/data/models/v3",
)
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
        self.digital_twin = DigitalTwinSandbox()
        self.redteam_sessions: dict[str, dict] = {}
        logger.info("Pipeline ready on device %s.", self.device)

    def score(self, text: str) -> float:
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=self.model.config.max_length, return_tensors="pt")
        with torch.no_grad():
            return float(self.model.risk_score(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))[0])

    def embed(self, text: str):
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=self.model.config.max_length, return_tensors="pt")
        with torch.no_grad():
            vec = self.model.embed(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))[0]
        return vec.cpu().numpy()

    def analyze(self, prompt: str, role: str, tool_name: str, resource: str | None, params: dict) -> dict:
        now = datetime.now(timezone.utc)
        request_obj = ToolCallRequest(tool_name=tool_name, resource=resource, params=params or {})
        credential = self.credential_by_role[role]

        risk_score = self.score(prompt)
        privilege_decision = self.policy_engine.authorize(credential, request_obj, now=now)
        sandbox_result = self.digital_twin.run(request_obj)

        tag = _build_provenance_tag(role, request_obj, prompt)
        source_trust = self.provenance_tracker.trust_score(tag)

        anomaly = self.behavioral_detector.score(ActionSequence(role=role, tool_names=[tool_name]))

        signals = RiskSignals(
            injection_score=risk_score,
            behavior_anomaly=anomaly.deviation_score,
            source_trust=source_trust,
            privilege_out_of_scope=privilege_decision.out_of_scope,
            session_history_risk=0.0,
            twin_unsafe=not sandbox_result.safe,
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
            "signals": {
                "injection_score": signals.injection_score,
                "behavior_anomaly": signals.behavior_anomaly,
                "source_trust": signals.source_trust,
                "privilege_out_of_scope": signals.privilege_out_of_scope,
                "twin_unsafe": signals.twin_unsafe,
            },
            "privilege_decision": {
                "allowed": privilege_decision.allowed,
                "violation_type": privilege_decision.violation_type.value,
                "reason": privilege_decision.reason,
            },
            "twin_result": {
                "safe": sandbox_result.safe,
                "notes": sandbox_result.safety_notes,
                "outcome": sandbox_result.twin_outcome,
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
        action_defaults = ROLE_DEFAULT_ACTION[role]
        analysis = self.analyze(
            prompt, role, action_defaults["tool_name"], action_defaults["resource"], dict(action_defaults["params"]),
        )

        redteam = self.run_redteam(prompt, rounds, variants_per_round)
        unlearn_result = self.unlearn(redteam["redteam_session_id"])

        return {
            "detected_role": role,
            "role_matched": role_matched,
            "analysis": analysis,
            "redteam": {
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
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), "index.html")


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
    return jsonify({
        "device": str(p.device),
        "calibration_threshold": p.calibration.threshold,
        "memory_index_size": len(p.memory_index),
        "active_redteam_sessions": len(p.redteam_sessions),
    })


def main():
    get_pipeline()  # load eagerly so the first request isn't slow
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
