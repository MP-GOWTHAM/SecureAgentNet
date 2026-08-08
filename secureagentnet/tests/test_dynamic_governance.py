from datetime import datetime, timezone

import pytest

from secureagentnet.privilege.dynamic_governance import DynamicPrivilegeGovernor
from secureagentnet.privilege.policy_engine import (
    POLICIES_DIR,
    CredentialStore,
    PolicyEngine,
    ToolCallRequest,
    ViolationType,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def store():
    return CredentialStore()


@pytest.fixture
def policy_engine(store):
    # Must share the same CredentialStore as the governor, or authorize()
    # has no way to see revocations the governor triggers.
    return PolicyEngine.from_directory(POLICIES_DIR, credential_store=store)


@pytest.fixture
def governor(policy_engine, store):
    return DynamicPrivilegeGovernor(policy_engine, store, tighten_threshold=0.5, revoke_threshold=0.8)


def test_low_risk_uses_normal_policy_scope(governor, store):
    cred = store.issue("email_agent", ttl_seconds=3600, now=NOW)
    request = ToolCallRequest(tool_name="send_email", resource="alice@secureagentnet-corp.com", params={"recipient_count": 1})
    decision = governor.evaluate(risk_score=0.1, credential=cred, request=request, now=NOW)
    assert decision.allowed  # send_email is in normal scope, low risk doesn't tighten anything


def test_elevated_risk_tightens_scope_below_revoke_threshold(governor, store):
    cred = store.issue("email_agent", ttl_seconds=3600, now=NOW)
    request = ToolCallRequest(tool_name="send_email", resource="alice@secureagentnet-corp.com", params={"recipient_count": 1})
    decision = governor.evaluate(risk_score=0.6, credential=cred, request=request, now=NOW)
    assert not decision.allowed  # send_email not in tightened allowlist for email_agent
    assert not store.is_revoked(cred.token_id)  # tightening != revocation


def test_tightened_scope_still_allows_allowlisted_tools(governor, store):
    cred = store.issue("email_agent", ttl_seconds=3600, now=NOW)
    request = ToolCallRequest(tool_name="read_inbox")
    decision = governor.evaluate(risk_score=0.6, credential=cred, request=request, now=NOW)
    assert decision.allowed


def test_high_risk_revokes_credential_outright(governor, store):
    cred = store.issue("email_agent", ttl_seconds=3600, now=NOW)
    request = ToolCallRequest(tool_name="read_inbox")
    decision = governor.evaluate(risk_score=0.9, credential=cred, request=request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.CREDENTIAL_REVOKED
    assert store.is_revoked(cred.token_id)


def test_revocation_cascades_to_derived_credentials(governor, store, policy_engine):
    parent = store.issue("research_agent", ttl_seconds=3600, now=NOW)
    child = store.derive(parent.token_id, "research_agent", ttl_seconds=3600, now=NOW)

    request = ToolCallRequest(tool_name="web_search")
    governor.evaluate(risk_score=0.9, credential=parent, request=request, now=NOW)

    assert store.is_revoked(parent.token_id)
    assert store.is_revoked(child.token_id)
    # child is now denied too, even though the governor was never called
    # with it directly — cascade revocation, checked via the shared store.
    child_decision = policy_engine.authorize(child, request, now=NOW)
    assert not child_decision.allowed
    assert child_decision.violation_type == ViolationType.CREDENTIAL_REVOKED


def test_construction_fails_if_policy_engine_uses_a_different_store(policy_engine):
    other_store = CredentialStore()
    with pytest.raises(ValueError, match="same credential_store"):
        DynamicPrivilegeGovernor(policy_engine, other_store)


def test_static_policy_denial_is_not_overridden_by_low_risk(governor, store):
    """Dynamic governance only ever narrows scope, never widens it past
    what the static policy already denies.
    """
    cred = store.issue("email_agent", ttl_seconds=3600, now=NOW)
    request = ToolCallRequest(tool_name="delete_file")  # never permitted for email_agent at all
    decision = governor.evaluate(risk_score=0.0, credential=cred, request=request, now=NOW)
    assert not decision.allowed
    assert decision.violation_type == ViolationType.TOOL_NOT_PERMITTED
