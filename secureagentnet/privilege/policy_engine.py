"""ABAC-style privilege governance for tool-integrated agents.

Design decisions:

- **Policies are declarative YAML, not if/else code.** Each role gets one
  YAML file under `policies/` listing exactly which tools it may call and,
  per tool, which resources (recipients, paths, URLs — whatever the tool's
  object is) it may act on. Adding a new agent role or tightening an
  existing one is a config edit, not a code change — the whole point of an
  ABAC layer is that scope decisions live in data the security team can
  audit, not scattered through the agent's control flow.

- **Deny by default.** A tool not listed for a role is denied; a resource
  pattern that doesn't match is denied. This is the standard ABAC/least-
  privilege posture and matters specifically for this project: the fusion
  layer (Phase 4) treats "requested tool out of scope" as a signal that
  compounds with detector risk, so an accidentally-permissive policy
  silently defeats the whole cross-surface correlation idea.

- **Short-lived scoped credentials are simulated, not cryptographic.** A
  `ScopedCredential` is a role + expiry + random token id — no signing, no
  real secret material. That's an intentional scope cut for this project
  (real credential issuance is an infra problem, not a detection-research
  one) but the *shape* of the check (every tool call must present a
  non-expired, role-bound credential) mirrors how real short-lived-token
  systems (e.g. STS-issued scoped tokens) are actually checked.

- **Decisions report *why*, not just allow/deny.** `violation_type` lets
  the correlation engine (and this project's C-ASR metric) distinguish
  "tool never in scope for this role" from "credential expired" from
  "resource didn't match" — useful both for debugging policies and for
  reporting which privilege failure mode chained attacks actually exploit.

- **Resource globs are one attribute dimension; `conditions` covers the
  rest.** `resource_patterns` handles "which object" (recipient, path,
  URL). Real ABAC also needs to gate on arbitrary call-time attributes —
  "no more than 5 recipients per send", "only during business hours" — so
  `ToolPermission.conditions` adds field/operator/value checks evaluated
  against `ToolCallRequest.params`, without hardcoding those checks as
  Python.

- **Credentials are revocable, not just expiring.** `CredentialStore`
  tracks every credential `issue()`s and lets an operator `revoke()` one
  before its natural expiry (e.g. on detecting the session is compromised)
  — `PolicyEngine.authorize()` checks revocation when a store is attached,
  independent of the expiry check.

- **Every decision can be audited.** `AuditLog` records each `authorize()`
  call (who, what, when, allowed/denied, why) to an in-memory list and
  optionally an append-only JSONL file, so the eval harness (and a human
  reviewing what a "Flag" decision actually did) has a paper trail rather
  than only the final aggregate metrics.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

POLICIES_DIR = Path(__file__).resolve().parent / "policies"


class ViolationType(str, Enum):
    NONE = "none"
    UNKNOWN_ROLE = "unknown_role"
    CREDENTIAL_EXPIRED = "credential_expired"
    CREDENTIAL_REVOKED = "credential_revoked"
    CREDENTIAL_ROLE_MISMATCH = "credential_role_mismatch"
    TOOL_NOT_PERMITTED = "tool_not_permitted"
    RESOURCE_DENIED = "resource_denied"
    CONDITION_DENIED = "condition_denied"


class ConditionOp(str, Enum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"


_OPS = {
    ConditionOp.EQ: lambda a, v: a == v,
    ConditionOp.NE: lambda a, v: a != v,
    ConditionOp.IN: lambda a, v: a in v,
    ConditionOp.NOT_IN: lambda a, v: a not in v,
    ConditionOp.LT: lambda a, v: a < v,
    ConditionOp.LTE: lambda a, v: a <= v,
    ConditionOp.GT: lambda a, v: a > v,
    ConditionOp.GTE: lambda a, v: a >= v,
}


class AttributeCondition(BaseModel):
    """One ABAC condition on a call-time attribute, e.g.
    `{field: recipient_count, op: lte, value: 5}` checked against
    `ToolCallRequest.params["recipient_count"]`.

    A condition whose field is absent from `params` fails closed (denied) —
    a policy that requires an attribute the request didn't supply should
    not be treated as satisfied.
    """

    field: str
    op: ConditionOp
    value: Any

    def evaluate(self, params: dict) -> bool:
        if self.field not in params:
            return False
        return _OPS[self.op](params[self.field], self.value)


class ToolPermission(BaseModel):
    """One tool a role may call, and which resources/attributes it may
    target.

    `resource_patterns` are fnmatch-style globs (`*@company.com`,
    `/workspace/**`, `https://internal.company.com/*`) matched against the
    tool call's `resource` field. `["*"]` (the default) means "any resource
    this tool can address" — i.e. no ABAC resource constraint beyond the
    tool-level allow, for tools where that's appropriate (e.g. read-only
    listing operations).

    `conditions` are additional attribute checks against the call's
    `params` (see `AttributeCondition`); all must pass.
    """

    name: str
    resource_patterns: list[str] = Field(default_factory=lambda: ["*"])
    conditions: list[AttributeCondition] = Field(default_factory=list)

    def matches_resource(self, resource: Optional[str]) -> bool:
        if resource is None:
            # A tool call with no resource attribute (e.g. a parameterless
            # tool) is fine as long as the tool itself is permitted.
            return True
        return any(fnmatch.fnmatch(resource, pattern) for pattern in self.resource_patterns)

    def matches_conditions(self, params: dict) -> bool:
        return all(c.evaluate(params) for c in self.conditions)


class RolePolicy(BaseModel):
    role: str
    description: str = ""
    tools: list[ToolPermission] = Field(default_factory=list)

    def get_tool_permission(self, tool_name: str) -> Optional[ToolPermission]:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None


class ToolCallRequest(BaseModel):
    """A tool call an agent is attempting, as the privilege layer sees it."""

    tool_name: str
    resource: Optional[str] = None
    params: dict = Field(default_factory=dict)
    task_id: Optional[str] = None


class Decision(BaseModel):
    allowed: bool
    violation_type: ViolationType
    reason: str

    @property
    def out_of_scope(self) -> bool:
        """Convenience alias matching the correlation engine's vocabulary
        (Phase 4 fuses this with detector risk as `requested_tool_out_of_scope`).
        """
        return not self.allowed


class ScopedCredential(BaseModel):
    """A simulated short-lived, role-scoped permission token.

    No real cryptography — `token_id` is just a random identifier for
    logging/audit correlation, not a verifiable secret. See module
    docstring for why that's an intentional scope cut here.
    """

    token_id: str
    role: str
    issued_at: datetime
    expires_at: datetime
    derived_from: Optional[str] = None  # parent credential's token_id, if any

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.expires_at


def issue_credential(
    role: str, ttl_seconds: int, now: Optional[datetime] = None, derived_from: Optional[str] = None
) -> ScopedCredential:
    now = now or datetime.now(timezone.utc)
    return ScopedCredential(
        token_id=str(uuid.uuid4()),
        role=role,
        issued_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
        derived_from=derived_from,
    )


class CredentialStore:
    """Tracks issued credentials, supports early revocation, and — for
    dynamic privilege governance (methodology §2.5) — cascading revocation:
    revoking a credential also revokes every credential *derived from* it,
    transitively. This is the blast-radius containment feature: if a
    research_agent's credential is flagged mid-session and it had spawned a
    scoped sub-credential for a delegated task, that sub-credential doesn't
    stay live just because it wasn't directly flagged.

    Deliberately in-memory only (a dict + a set + an adjacency list) — this
    mirrors the rest of the privilege layer's "simulate the shape of the
    real system, skip the real infra" scope cut (see module docstring). A
    production version would back this with a real store (DB/cache) shared
    across agent processes; that's out of scope for this project.
    """

    def __init__(self):
        self._credentials: dict[str, ScopedCredential] = {}
        self._revoked: set[str] = set()
        self._children: dict[str, list[str]] = {}  # parent token_id -> [child token_id, ...]

    def issue(self, role: str, ttl_seconds: int, now: Optional[datetime] = None) -> ScopedCredential:
        cred = issue_credential(role, ttl_seconds, now=now)
        self._credentials[cred.token_id] = cred
        return cred

    def derive(
        self, parent_token_id: str, role: str, ttl_seconds: int, now: Optional[datetime] = None
    ) -> ScopedCredential:
        """Issue a new credential scoped as a child of `parent_token_id`,
        for delegated sub-tasks. `role` may equal the parent's role or be
        narrower — this method only tracks lineage for cascade revocation,
        it doesn't itself enforce that a derived role is a subset of the
        parent's (that's the policy layer's job via normal `authorize()`
        calls against whatever role the derived credential carries).
        """
        cred = issue_credential(role, ttl_seconds, now=now, derived_from=parent_token_id)
        self._credentials[cred.token_id] = cred
        self._children.setdefault(parent_token_id, []).append(cred.token_id)
        return cred

    def revoke(self, token_id: str) -> None:
        self._revoked.add(token_id)

    def revoke_cascade(self, token_id: str) -> list[str]:
        """Revoke `token_id` and every credential transitively derived from
        it. Returns the full list of token_ids revoked (including
        `token_id` itself).
        """
        revoked: list[str] = []
        stack = [token_id]
        while stack:
            current = stack.pop()
            if current in self._revoked:
                continue
            self._revoked.add(current)
            revoked.append(current)
            stack.extend(self._children.get(current, []))
        return revoked

    def is_revoked(self, token_id: str) -> bool:
        return token_id in self._revoked

    def get(self, token_id: str) -> Optional[ScopedCredential]:
        return self._credentials.get(token_id)


class AuditLogEntry(BaseModel):
    timestamp: datetime
    token_id: str
    role: str
    tool_name: str
    resource: Optional[str]
    task_id: Optional[str]
    allowed: bool
    violation_type: ViolationType
    reason: str


GENESIS_HASH = "0" * 64


class AuditLog:
    """Records every `authorize()` decision. Kept in-memory (`entries`)
    always; also appended as JSONL to `path` if given, so the eval harness
    can replay/inspect a run's full decision trail rather than only its
    aggregate metrics.

    Tamper-evident (methodology §2.8): every entry is hash-chained —
    `entry_hash = sha256(prev_hash + entry_json)` — so altering or deleting
    any past entry changes every hash after it. `verify_chain()` recomputes
    the chain from `entries` and reports the first break, if any. This
    catches *silent* edits to the in-memory/on-disk log after the fact; it
    does not itself prevent someone with write access from truncating the
    log file and starting a new chain — genuine tamper-*proofing* would
    need an external anchor (e.g. periodically publishing the latest hash
    somewhere the log's own writer can't rewrite), which is out of scope
    here.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self.entries: list[AuditLogEntry] = []
        self.hashes: list[str] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def last_hash(self) -> str:
        return self.hashes[-1] if self.hashes else GENESIS_HASH

    def _compute_hash(self, prev_hash: str, entry: AuditLogEntry) -> str:
        payload = f"{prev_hash}{entry.model_dump_json()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def record(self, entry: AuditLogEntry) -> str:
        prev_hash = self.last_hash
        entry_hash = self._compute_hash(prev_hash, entry)
        self.entries.append(entry)
        self.hashes.append(entry_hash)
        if self.path:
            # encoding/newline pinned explicitly: Windows would otherwise use
            # the ANSI codepage (UnicodeEncodeError on non-ASCII prompt text)
            # and translate "\n" to "\r\n", corrupting the JSONL framing.
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                record = {"prev_hash": prev_hash, "hash": entry_hash, **json.loads(entry.model_dump_json())}
                f.write(json.dumps(record) + "\n")
        return entry_hash

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        """Recomputes the hash chain from `entries` and compares against
        `hashes`. Returns (is_intact, first_broken_index) — index is None
        if the chain is intact.
        """
        prev_hash = GENESIS_HASH
        for i, (entry, stored_hash) in enumerate(zip(self.entries, self.hashes)):
            recomputed = self._compute_hash(prev_hash, entry)
            if recomputed != stored_hash:
                return False, i
            prev_hash = stored_hash
        return True, None


class PolicyEngine:
    def __init__(
        self,
        policies: dict[str, RolePolicy],
        credential_store: Optional[CredentialStore] = None,
        audit_log: Optional[AuditLog] = None,
    ):
        self.policies = policies
        self.credential_store = credential_store
        self.audit_log = audit_log

    @classmethod
    def from_directory(
        cls,
        directory: str | Path = POLICIES_DIR,
        credential_store: Optional[CredentialStore] = None,
        audit_log: Optional[AuditLog] = None,
    ) -> "PolicyEngine":
        directory = Path(directory)
        policies: dict[str, RolePolicy] = {}
        for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            policy = RolePolicy.model_validate(raw)
            if policy.role in policies:
                raise ValueError(f"Duplicate role policy '{policy.role}' in {path}")
            policies[policy.role] = policy
        return cls(policies, credential_store=credential_store, audit_log=audit_log)

    def authorize(
        self,
        credential: ScopedCredential,
        request: ToolCallRequest,
        now: Optional[datetime] = None,
    ) -> Decision:
        eval_time = now or datetime.now(timezone.utc)
        decision = self._decide(credential, request, eval_time)
        if self.audit_log is not None:
            self.audit_log.record(
                AuditLogEntry(
                    timestamp=eval_time,
                    token_id=credential.token_id,
                    role=credential.role,
                    tool_name=request.tool_name,
                    resource=request.resource,
                    task_id=request.task_id,
                    allowed=decision.allowed,
                    violation_type=decision.violation_type,
                    reason=decision.reason,
                )
            )
        return decision

    def _decide(self, credential: ScopedCredential, request: ToolCallRequest, eval_time: datetime) -> Decision:
        if credential.is_expired(eval_time):
            return Decision(
                allowed=False,
                violation_type=ViolationType.CREDENTIAL_EXPIRED,
                reason=f"credential {credential.token_id} expired at {credential.expires_at.isoformat()}",
            )

        if self.credential_store is not None and self.credential_store.is_revoked(credential.token_id):
            return Decision(
                allowed=False,
                violation_type=ViolationType.CREDENTIAL_REVOKED,
                reason=f"credential {credential.token_id} was revoked",
            )

        policy = self.policies.get(credential.role)
        if policy is None:
            return Decision(
                allowed=False,
                violation_type=ViolationType.UNKNOWN_ROLE,
                reason=f"no policy defined for role '{credential.role}'",
            )

        tool_perm = policy.get_tool_permission(request.tool_name)
        if tool_perm is None:
            return Decision(
                allowed=False,
                violation_type=ViolationType.TOOL_NOT_PERMITTED,
                reason=f"role '{credential.role}' has no permission to call '{request.tool_name}'",
            )

        if not tool_perm.matches_resource(request.resource):
            return Decision(
                allowed=False,
                violation_type=ViolationType.RESOURCE_DENIED,
                reason=(
                    f"role '{credential.role}' may call '{request.tool_name}' but resource "
                    f"'{request.resource}' does not match allowed patterns {tool_perm.resource_patterns}"
                ),
            )

        if not tool_perm.matches_conditions(request.params):
            return Decision(
                allowed=False,
                violation_type=ViolationType.CONDITION_DENIED,
                reason=(
                    f"role '{credential.role}' may call '{request.tool_name}' on this resource but "
                    f"request params {request.params} fail required conditions {tool_perm.conditions}"
                ),
            )

        return Decision(allowed=True, violation_type=ViolationType.NONE, reason="permitted")
