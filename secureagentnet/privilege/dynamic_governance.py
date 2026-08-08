"""Dynamic privilege governance (methodology §2.5).

Wraps `PolicyEngine` + `CredentialStore` with risk-reactive scope control:
a credential's effective permissions can shrink or disappear mid-session as
the adaptive risk engine's score rises, on top of the static ABAC policy
that never changes. Two thresholds, two responses:

- `tighten_threshold`: above this, only tools on the role's *tightened
  allowlist* (a stricter subset than its full policy) are permitted —
  scope shrinks without revoking the credential outright.
- `revoke_threshold`: above this, the credential (and everything
  cascade-derived from it, via `CredentialStore.revoke_cascade`) is
  revoked outright.

This sits in front of `PolicyEngine.authorize()`, not instead of it — the
static policy is still the hard outer boundary; this only ever makes
things *more* restrictive than the policy allows, never less.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from secureagentnet.privilege.policy_engine import (
    CredentialStore,
    Decision,
    PolicyEngine,
    ScopedCredential,
    ToolCallRequest,
    ViolationType,
)

# Default tightened allowlist per role: what stays permitted once risk
# crosses tighten_threshold. Deliberately narrower than each role's full
# policy — e.g. email_agent keeps read-only inbox access but loses
# send_email until risk subsides or the credential is revoked outright.
DEFAULT_TIGHTENED_ALLOWLIST: dict[str, set[str]] = {
    "email_agent": {"read_inbox", "search_email"},
    "file_agent": {"read_file", "list_files"},
    "research_agent": {"web_search"},
    "calendar_agent": {"list_events"},
    "code_exec_agent": set(),  # highest-value target — tightened means no execution at all
    "support_agent": {"read_ticket"},
}


class DynamicPrivilegeGovernor:
    def __init__(
        self,
        policy_engine: PolicyEngine,
        credential_store: CredentialStore,
        tighten_threshold: float = 0.5,
        revoke_threshold: float = 0.8,
        tightened_allowlist: Optional[dict[str, set[str]]] = None,
    ):
        if policy_engine.credential_store is not credential_store:
            # Caught in testing the hard way: if authorize() checks a
            # *different* CredentialStore than the one this governor
            # revokes into, revocation silently has no effect — the
            # decision that follows a revoke_cascade() call still comes
            # back allowed.
            raise ValueError(
                "policy_engine must be constructed with the same credential_store instance "
                "passed here (PolicyEngine.from_directory(..., credential_store=credential_store)), "
                "or revocations this governor triggers will never be checked by authorize()."
            )
        self.policy_engine = policy_engine
        self.credential_store = credential_store
        self.tighten_threshold = tighten_threshold
        self.revoke_threshold = revoke_threshold
        self.tightened_allowlist = tightened_allowlist or DEFAULT_TIGHTENED_ALLOWLIST

    def evaluate(
        self,
        risk_score: float,
        credential: ScopedCredential,
        request: ToolCallRequest,
        now: Optional[datetime] = None,
    ) -> Decision:
        if risk_score >= self.revoke_threshold:
            self.credential_store.revoke_cascade(credential.token_id)

        decision = self.policy_engine.authorize(credential, request, now=now)
        if not decision.allowed:
            return decision  # static policy (or the revoke above) already denies

        if risk_score >= self.tighten_threshold:
            allowed_tools = self.tightened_allowlist.get(credential.role, set())
            if request.tool_name not in allowed_tools:
                return Decision(
                    allowed=False,
                    violation_type=ViolationType.TOOL_NOT_PERMITTED,
                    reason=(
                        f"scope tightened due to elevated risk ({risk_score:.2f} >= "
                        f"{self.tighten_threshold}); '{request.tool_name}' not in tightened allowlist"
                    ),
                )

        return decision
