"""Standalone tests for the ABAC policy engine, independent of the detector.

Scenarios are hand-written mock agent situations mirroring the project's
threat model: an injected instruction trying to get an over-scoped agent to
do something outside its role (exfiltrate via email, escape the workspace,
delete files, replay an expired credential).
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from secureagentnet.privilege.policy_engine import (
    POLICIES_DIR,
    AttributeCondition,
    AuditLog,
    ConditionOp,
    CredentialStore,
    PolicyEngine,
    RolePolicy,
    ScopedCredential,
    ToolCallRequest,
    ToolPermission,
    ViolationType,
    issue_credential,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    return PolicyEngine.from_directory(POLICIES_DIR)


def _credential(role: str, ttl_seconds: int = 300, issued_at=NOW) -> ScopedCredential:
    return issue_credential(role, ttl_seconds=ttl_seconds, now=issued_at)


# --- loading real policy files -----------------------------------------

def test_loads_all_shipped_role_policies(engine):
    assert set(engine.policies.keys()) == {
        "email_agent", "file_agent", "research_agent",
        "calendar_agent", "code_exec_agent", "support_agent",
    }


def test_duplicate_role_across_files_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("role: dup\ntools: []\n")
    (tmp_path / "b.yaml").write_text("role: dup\ntools: []\n")
    with pytest.raises(ValueError, match="Duplicate role policy"):
        PolicyEngine.from_directory(tmp_path)


# --- email agent: the spec's canonical example --------------------------

def test_email_agent_may_send_to_internal_domain(engine):
    cred = _credential("email_agent")
    request = ToolCallRequest(
        tool_name="send_email", resource="alice@secureagentnet-corp.com", params={"recipient_count": 1}
    )
    decision = engine.authorize(cred, request, now=NOW)
    assert decision.allowed
    assert decision.violation_type == ViolationType.NONE


def test_email_agent_may_not_exfiltrate_to_external_domain(engine):
    """The chained-attack scenario: an injected instruction tries to get
    the email agent to forward data to an attacker-controlled address.
    Tool is permitted, but the resource (recipient) isn't.
    """
    cred = _credential("email_agent")
    request = ToolCallRequest(
        tool_name="send_email", resource="attacker@evil.com", params={"recipient_count": 1}
    )
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.RESOURCE_DENIED
    assert decision.out_of_scope


def test_email_agent_may_not_bulk_send_even_to_internal_domain(engine):
    """ABAC condition beyond the resource glob: an injected 'BCC all these
    addresses' instruction is denied by recipient_count even though every
    address is in-domain.
    """
    cred = _credential("email_agent")
    request = ToolCallRequest(
        tool_name="send_email", resource="alice@secureagentnet-corp.com", params={"recipient_count": 200}
    )
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.CONDITION_DENIED


def test_email_agent_send_denied_when_recipient_count_param_missing(engine):
    """Conditions fail closed: an omitted attribute the policy requires is
    treated as not satisfying the condition, not as "no constraint".
    """
    cred = _credential("email_agent")
    request = ToolCallRequest(tool_name="send_email", resource="alice@secureagentnet-corp.com")
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.CONDITION_DENIED


def test_email_agent_may_not_delete_files(engine):
    """Matches the project spec's example verbatim: 'email agent may call
    send_email but not delete_file'.
    """
    cred = _credential("email_agent")
    request = ToolCallRequest(tool_name="delete_file", resource="/workspace/report.docx")
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.TOOL_NOT_PERMITTED


# --- file agent: workspace containment -----------------------------------

def test_file_agent_may_write_inside_workspace(engine):
    cred = _credential("file_agent")
    request = ToolCallRequest(tool_name="write_file", resource="/workspace/notes/draft.md")
    decision = engine.authorize(cred, request, now=NOW)
    assert decision.allowed


def test_file_agent_may_not_write_outside_workspace(engine):
    """Injected instruction tries to escape the sandboxed workspace dir."""
    cred = _credential("file_agent")
    request = ToolCallRequest(tool_name="write_file", resource="/etc/passwd")
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.RESOURCE_DENIED


def test_file_agent_has_no_delete_permission_at_all(engine):
    cred = _credential("file_agent")
    request = ToolCallRequest(tool_name="delete_file", resource="/workspace/report.docx")
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.TOOL_NOT_PERMITTED


# --- research agent: web-facing role -------------------------------------

def test_research_agent_may_fetch_https_urls(engine):
    cred = _credential("research_agent")
    request = ToolCallRequest(tool_name="fetch_url", resource="https://example.com/article")
    decision = engine.authorize(cred, request, now=NOW)
    assert decision.allowed


def test_research_agent_may_not_fetch_file_scheme(engine):
    """A page it fetched might contain an injected instruction to read a
    local file via file:// — resource pattern rejects non-http(s) schemes.
    """
    cred = _credential("research_agent")
    request = ToolCallRequest(tool_name="fetch_url", resource="file:///etc/passwd")
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.RESOURCE_DENIED


# --- calendar agent: mass-invite / cross-event tampering ------------------

def test_calendar_agent_may_create_small_event(engine):
    cred = _credential("calendar_agent")
    request = ToolCallRequest(tool_name="create_event", params={"attendee_count": 3})
    decision = engine.authorize(cred, request, now=NOW)
    assert decision.allowed


def test_calendar_agent_may_not_mass_invite(engine):
    """Injected 'invite the whole company' instruction — tool is permitted,
    the attendee_count condition isn't.
    """
    cred = _credential("calendar_agent")
    request = ToolCallRequest(tool_name="create_event", params={"attendee_count": 500})
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.CONDITION_DENIED


def test_calendar_agent_may_cancel_only_self_created_events(engine):
    cred = _credential("calendar_agent")
    allowed = engine.authorize(
        cred, ToolCallRequest(tool_name="cancel_event", resource="self-created:evt_123"), now=NOW
    )
    denied = engine.authorize(
        cred, ToolCallRequest(tool_name="cancel_event", resource="someone-elses:evt_999"), now=NOW
    )
    assert allowed.allowed
    assert not denied.allowed
    assert denied.violation_type == ViolationType.RESOURCE_DENIED


# --- code exec agent: highest-value target, tightest scope -----------------

def test_code_exec_agent_may_run_short_sandboxed_code(engine):
    cred = _credential("code_exec_agent")
    request = ToolCallRequest(
        tool_name="run_code", resource="/sandbox/script.py", params={"timeout_seconds": 5}
    )
    decision = engine.authorize(cred, request, now=NOW)
    assert decision.allowed


def test_code_exec_agent_may_not_escape_sandbox(engine):
    cred = _credential("code_exec_agent")
    request = ToolCallRequest(
        tool_name="run_code", resource="/etc/cron.d/evil", params={"timeout_seconds": 5}
    )
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.RESOURCE_DENIED


def test_code_exec_agent_may_not_run_unbounded_loops(engine):
    """Injected instruction tries to run something that never times out
    (resource exhaustion / cryptomining pattern)."""
    cred = _credential("code_exec_agent")
    request = ToolCallRequest(
        tool_name="run_code", resource="/sandbox/script.py", params={"timeout_seconds": 999999}
    )
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.CONDITION_DENIED


def test_code_exec_agent_has_no_network_tool_at_all(engine):
    """There's no fetch_url/http tool in this role's policy at all — an
    injected 'curl this URL' instruction has no matching permission,
    regardless of resource or conditions.
    """
    cred = _credential("code_exec_agent")
    request = ToolCallRequest(tool_name="fetch_url", resource="https://attacker.com/exfil")
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.TOOL_NOT_PERMITTED


# --- support agent: monetary-limit condition --------------------------------

def test_support_agent_may_issue_small_refund(engine):
    cred = _credential("support_agent")
    request = ToolCallRequest(tool_name="issue_refund", resource="order_42", params={"amount_usd": 25})
    decision = engine.authorize(cred, request, now=NOW)
    assert decision.allowed


def test_support_agent_may_not_issue_large_refund(engine):
    """Injected 'the customer says refund $5000' instruction embedded in a
    ticket body — tool is permitted, the amount condition isn't.
    """
    cred = _credential("support_agent")
    request = ToolCallRequest(tool_name="issue_refund", resource="order_42", params={"amount_usd": 5000})
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.CONDITION_DENIED


def test_support_agent_cannot_send_email_or_touch_files(engine):
    """Support agent's ticket replies stay inside the ticketing system —
    no send_email/file tools are in scope at all.
    """
    cred = _credential("support_agent")
    for tool_name in ("send_email", "write_file", "run_code"):
        decision = engine.authorize(cred, ToolCallRequest(tool_name=tool_name), now=NOW)
        assert not decision.allowed
        assert decision.violation_type == ViolationType.TOOL_NOT_PERMITTED


# --- credential lifecycle -------------------------------------------------

def test_expired_credential_denies_an_otherwise_valid_request(engine):
    cred = _credential("email_agent", ttl_seconds=60, issued_at=NOW)
    request = ToolCallRequest(tool_name="send_email", resource="alice@secureagentnet-corp.com")
    later = NOW + timedelta(seconds=61)
    decision = engine.authorize(cred, request, now=later)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.CREDENTIAL_EXPIRED


def test_credential_still_valid_just_before_expiry(engine):
    cred = _credential("email_agent", ttl_seconds=60, issued_at=NOW)
    request = ToolCallRequest(
        tool_name="send_email", resource="alice@secureagentnet-corp.com", params={"recipient_count": 1}
    )
    just_before = NOW + timedelta(seconds=59)
    decision = engine.authorize(cred, request, now=just_before)
    assert decision.allowed


def test_unknown_role_is_denied(engine):
    cred = _credential("nonexistent_agent")
    request = ToolCallRequest(tool_name="send_email", resource="alice@secureagentnet-corp.com")
    decision = engine.authorize(cred, request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.UNKNOWN_ROLE


def test_issue_credential_sets_expected_expiry():
    cred = issue_credential("email_agent", ttl_seconds=120, now=NOW)
    assert cred.role == "email_agent"
    assert cred.expires_at == NOW + timedelta(seconds=120)
    assert not cred.is_expired(now=NOW + timedelta(seconds=119))
    assert cred.is_expired(now=NOW + timedelta(seconds=121))


# --- ToolPermission resource matching (unit-level) ------------------------

def test_tool_permission_wildcard_matches_any_resource():
    perm = ToolPermission(name="web_search")
    assert perm.matches_resource("anything at all")
    assert perm.matches_resource(None)


def test_tool_permission_glob_pattern_matching():
    perm = ToolPermission(name="write_file", resource_patterns=["/workspace/**"])
    assert perm.matches_resource("/workspace/sub/dir/file.txt")
    assert not perm.matches_resource("/etc/passwd")


def test_role_policy_get_tool_permission_returns_none_for_unlisted_tool():
    policy = RolePolicy(role="x", tools=[ToolPermission(name="send_email")])
    assert policy.get_tool_permission("delete_file") is None
    assert policy.get_tool_permission("send_email") is not None


# --- AttributeCondition (unit-level) --------------------------------------

@pytest.mark.parametrize(
    "op,value,actual,expected",
    [
        (ConditionOp.EQ, "prod", "prod", True),
        (ConditionOp.EQ, "prod", "staging", False),
        (ConditionOp.NE, "prod", "staging", True),
        (ConditionOp.IN, ["a", "b"], "a", True),
        (ConditionOp.IN, ["a", "b"], "c", False),
        (ConditionOp.NOT_IN, ["a", "b"], "c", True),
        (ConditionOp.LT, 5, 3, True),
        (ConditionOp.LT, 5, 5, False),
        (ConditionOp.LTE, 5, 5, True),
        (ConditionOp.GT, 5, 6, True),
        (ConditionOp.GTE, 5, 5, True),
    ],
)
def test_attribute_condition_operators(op, value, actual, expected):
    cond = AttributeCondition(field="x", op=op, value=value)
    assert cond.evaluate({"x": actual}) == expected


def test_attribute_condition_fails_closed_when_field_missing():
    cond = AttributeCondition(field="recipient_count", op=ConditionOp.LTE, value=5)
    assert cond.evaluate({}) is False


# --- CredentialStore + revocation ------------------------------------------

def test_credential_store_issued_credential_is_not_revoked_by_default():
    store = CredentialStore()
    cred = store.issue("email_agent", ttl_seconds=300, now=NOW)
    assert not store.is_revoked(cred.token_id)
    assert store.get(cred.token_id) == cred


def test_revoked_credential_is_denied_even_if_unexpired(engine):
    store = CredentialStore()
    engine_with_store = PolicyEngine(engine.policies, credential_store=store)
    cred = store.issue("email_agent", ttl_seconds=300, now=NOW)
    request = ToolCallRequest(
        tool_name="send_email", resource="alice@secureagentnet-corp.com", params={"recipient_count": 1}
    )

    before_revoke = engine_with_store.authorize(cred, request, now=NOW)
    assert before_revoke.allowed

    store.revoke(cred.token_id)
    after_revoke = engine_with_store.authorize(cred, request, now=NOW)
    assert not after_revoke.allowed
    assert after_revoke.violation_type == ViolationType.CREDENTIAL_REVOKED


def test_engine_without_credential_store_ignores_revocation_concept(engine):
    """A credential from an unrelated store can't be "revoked" against an
    engine that has no store attached — revocation is opt-in per engine.
    """
    store = CredentialStore()
    cred = store.issue("email_agent", ttl_seconds=300, now=NOW)
    store.revoke(cred.token_id)
    request = ToolCallRequest(
        tool_name="send_email", resource="alice@secureagentnet-corp.com", params={"recipient_count": 1}
    )
    decision = engine.authorize(cred, request, now=NOW)
    assert decision.allowed


# --- AuditLog ---------------------------------------------------------------

def test_audit_log_records_both_allowed_and_denied_decisions(engine):
    audit = AuditLog()
    engine_with_audit = PolicyEngine(engine.policies, audit_log=audit)
    cred = _credential("email_agent")

    engine_with_audit.authorize(
        cred,
        ToolCallRequest(tool_name="send_email", resource="alice@secureagentnet-corp.com", params={"recipient_count": 1}),
        now=NOW,
    )
    engine_with_audit.authorize(
        cred, ToolCallRequest(tool_name="delete_file", resource="/workspace/x"), now=NOW
    )

    assert len(audit.entries) == 2
    assert audit.entries[0].allowed is True
    assert audit.entries[1].allowed is False
    assert audit.entries[1].violation_type == ViolationType.TOOL_NOT_PERMITTED
    assert audit.entries[1].role == "email_agent"


def test_audit_log_chain_is_intact_after_normal_recording(engine):
    audit = AuditLog()
    engine_with_audit = PolicyEngine(engine.policies, audit_log=audit)
    cred = _credential("email_agent")
    for _ in range(3):
        engine_with_audit.authorize(cred, ToolCallRequest(tool_name="read_inbox"), now=NOW)
    intact, broken_at = audit.verify_chain()
    assert intact
    assert broken_at is None


def test_audit_log_detects_tampered_entry():
    audit = AuditLog()
    engine_with_audit = PolicyEngine(PolicyEngine.from_directory(POLICIES_DIR).policies, audit_log=audit)
    cred = _credential("email_agent")
    for _ in range(3):
        engine_with_audit.authorize(cred, ToolCallRequest(tool_name="read_inbox"), now=NOW)

    # tamper with a past entry in place (simulating someone editing the log)
    audit.entries[1] = audit.entries[1].model_copy(update={"allowed": not audit.entries[1].allowed})

    intact, broken_at = audit.verify_chain()
    assert not intact
    assert broken_at == 1


def test_audit_log_persists_to_jsonl_file(tmp_path, engine):
    log_path = tmp_path / "audit" / "decisions.jsonl"
    audit = AuditLog(path=log_path)
    engine_with_audit = PolicyEngine(engine.policies, audit_log=audit)
    cred = _credential("file_agent")

    engine_with_audit.authorize(cred, ToolCallRequest(tool_name="delete_file", resource="/workspace/x"), now=NOW)

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["allowed"] is False
    assert entry["violation_type"] == "tool_not_permitted"
