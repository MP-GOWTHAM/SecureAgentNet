"""Digital twin sandbox (methodology §2.6) — full simulated environment.

Per the doc's own feasibility caution (§8), "full simulated environment"
means stateful mock backends that respond dynamically to sequences of
calls, not one-shot stub responses. This module implements that: an
`Inbox`, `Filesystem`, `Calendar`, `CodeExec`, `Support`, and `Web` twin —
one per tool family used across the six roles in `privilege/policies/` —
each with real internal state that mutates across calls within a session,
plus a `DigitalTwinSandbox` that intercepts a planned tool call, redirects
it to the matching twin, and evaluates the twin's resulting state against
safety criteria — the concrete, checkable mismatch the doc's `forward_email`
example describes (recipient domain doesn't match any known contact), not
just "did the call error".

Scope: this simulates enough backend behavior to make pre-execution safety
checks meaningful for the six agent roles already defined in
`privilege/policies/` — it is not a general-purpose service emulator. Any
tool name outside those roles' own policies falls through to a "no twin
backend, assumed safe" note rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from secureagentnet.privilege.policy_engine import ToolCallRequest


# --- Twin backends -----------------------------------------------------------

@dataclass
class Email:
    sender: str
    subject: str
    body: str
    read: bool = False


@dataclass
class InboxTwin:
    """Stateful mock email backend. `known_contacts` models the address
    book a real forward-safety check would consult.
    """

    known_contacts: set[str] = field(default_factory=set)
    inbox: list[Email] = field(default_factory=list)
    sent: list[dict] = field(default_factory=list)

    def read_inbox(self) -> list[Email]:
        for e in self.inbox:
            e.read = True
        return list(self.inbox)

    def send_email(self, resource: Optional[str], params: dict) -> dict:
        outcome = {"recipient": resource, "recipient_known": resource in self.known_contacts}
        self.sent.append(outcome)
        return outcome

    def forward_email(self, resource: Optional[str], params: dict) -> dict:
        return self.send_email(resource, params)


@dataclass
class FilesystemTwin:
    """Stateful mock filesystem. `files` maps path -> content; writes and
    deletes mutate this dict across calls within a session.
    """

    files: dict[str, str] = field(default_factory=dict)

    def write_file(self, resource: Optional[str], params: dict) -> dict:
        content = params.get("content", "")
        existed = resource in self.files
        self.files[resource] = content
        return {"path": resource, "overwrote_existing": existed}

    def delete_file(self, resource: Optional[str], params: dict) -> dict:
        existed = resource in self.files
        self.files.pop(resource, None)
        return {"path": resource, "existed": existed}

    def read_file(self, resource: Optional[str], params: dict) -> dict:
        return {"path": resource, "content": self.files.get(resource), "exists": resource in self.files}


@dataclass
class CalendarTwin:
    """Stateful mock calendar. `events` accumulates across calls."""

    events: dict[str, dict] = field(default_factory=dict)
    _next_id: int = 0

    def create_event(self, resource: Optional[str], params: dict) -> dict:
        self._next_id += 1
        event_id = f"evt_{self._next_id}"
        self.events[event_id] = {"attendee_count": params.get("attendee_count", 0)}
        return {"event_id": event_id, **self.events[event_id]}

    def cancel_event(self, resource: Optional[str], params: dict) -> dict:
        existed = resource in self.events
        self.events.pop(resource, None)
        return {"event_id": resource, "existed": existed}


@dataclass
class CodeExecTwin:
    """Stateful mock sandbox execution backend. `total_cpu_seconds`
    accumulates across calls within a session — the point being that a
    single call's `timeout_seconds` can look fine in isolation while a
    *sequence* of calls quietly burns much more aggregate compute than any
    one privilege condition (which only ever sees one call at a time) can
    catch.
    """

    sandbox_files: dict[str, str] = field(default_factory=dict)
    total_cpu_seconds: float = 0.0
    max_cumulative_cpu_seconds: float = 60.0

    def run_code(self, resource: Optional[str], params: dict) -> dict:
        timeout_seconds = params.get("timeout_seconds", 0)
        self.total_cpu_seconds += timeout_seconds
        return {
            "resource": resource,
            "timeout_seconds": timeout_seconds,
            "cumulative_cpu_seconds": self.total_cpu_seconds,
            "executed": True,
        }

    def list_sandbox_files(self, resource: Optional[str], params: dict) -> dict:
        return {"resource": resource, "files": list(self.sandbox_files.keys())}


@dataclass
class WebTwin:
    """Stateful mock backend for research_agent's fetch_url/web_search/
    save_note. `fetched_domains` accumulates across calls — this is the
    same "no single call looks bad, the sequence does" pattern as the
    other stateful twins: an agent that fetches from many distinct domains
    in one session (crawling for a specific piece of data, or being
    walked through a chain of injected redirects) is a real behavioral
    signal a per-call `resource_patterns: ["https://*"]` check can't see,
    since that pattern matches every individual call fine.
    """

    trusted_domains: set[str] = field(default_factory=set)
    fetched_domains: set[str] = field(default_factory=set)
    notes: dict[str, str] = field(default_factory=dict)
    max_distinct_domains: int = 5

    def fetch_url(self, resource: Optional[str], params: dict) -> dict:
        domain = urlparse(resource).netloc if resource else None
        if domain:
            self.fetched_domains.add(domain)
        return {
            "url": resource,
            "domain": domain,
            # With no trusted_domains configured, every domain counts as
            # trusted (no allowlist to violate) — mirrors ToolPermission's
            # own "[*] means unconstrained" convention.
            "domain_trusted": (domain in self.trusted_domains) if self.trusted_domains else True,
            "distinct_domains_fetched": len(self.fetched_domains),
        }

    def web_search(self, resource: Optional[str], params: dict) -> dict:
        return {"query": resource, "n_results": 5}

    def save_note(self, resource: Optional[str], params: dict) -> dict:
        content = params.get("content", "")
        self.notes[resource] = content
        return {"path": resource, "saved": True}


@dataclass
class SupportTwin:
    """Stateful mock support-ticket backend. `refunded_total_by_order`
    accumulates across calls — the check this enables (cumulative refunds
    per order exceeding a cap) is one the privilege layer's per-call
    condition genuinely cannot express, since `authorize()` has no memory
    of prior calls; this is a case where the digital twin adds real
    coverage beyond what ABAC alone can catch, not just a duplicate check.
    """

    tickets: dict[str, dict] = field(default_factory=dict)
    refunded_total_by_order: dict[str, float] = field(default_factory=dict)
    max_cumulative_refund_usd: float = 100.0

    def read_ticket(self, resource: Optional[str], params: dict) -> dict:
        return {"ticket_id": resource, "exists": resource in self.tickets}

    def reply_ticket(self, resource: Optional[str], params: dict) -> dict:
        return {"ticket_id": resource, "replied": True}

    def issue_refund(self, resource: Optional[str], params: dict) -> dict:
        amount = params.get("amount_usd", 0)
        prior_total = self.refunded_total_by_order.get(resource, 0.0)
        new_total = prior_total + amount
        self.refunded_total_by_order[resource] = new_total
        return {"order_id": resource, "amount_usd": amount, "cumulative_refunded_usd": new_total}


# --- Safety evaluation --------------------------------------------------------

@dataclass
class SandboxResult:
    tool_name: str
    resource: Optional[str]
    twin_outcome: dict
    safe: bool
    safety_notes: list[str]


class DigitalTwinSandbox:
    """Routes a planned tool call to the matching twin, then checks the
    resulting simulated outcome against safety criteria — this is the
    pre-execution validation step: nothing here has any real-world effect.
    """

    EMAIL_TOOLS = {"send_email", "forward_email", "read_inbox"}
    FILE_TOOLS = {"write_file", "delete_file", "read_file"}
    CALENDAR_TOOLS = {"create_event", "cancel_event"}
    CODE_TOOLS = {"run_code", "list_sandbox_files"}
    SUPPORT_TOOLS = {"read_ticket", "reply_ticket", "issue_refund"}
    WEB_TOOLS = {"fetch_url", "web_search", "save_note"}

    def __init__(
        self,
        inbox: Optional[InboxTwin] = None,
        filesystem: Optional[FilesystemTwin] = None,
        calendar: Optional[CalendarTwin] = None,
        code_exec: Optional[CodeExecTwin] = None,
        support: Optional[SupportTwin] = None,
        web: Optional[WebTwin] = None,
        max_attendees: int = 10,
    ):
        self.inbox = inbox or InboxTwin()
        self.filesystem = filesystem or FilesystemTwin()
        self.calendar = calendar or CalendarTwin()
        self.code_exec = code_exec or CodeExecTwin()
        self.support = support or SupportTwin()
        self.web = web or WebTwin()
        self.max_attendees = max_attendees

    def run(self, request: ToolCallRequest) -> SandboxResult:
        notes: list[str] = []

        if request.tool_name in self.EMAIL_TOOLS:
            method = getattr(self.inbox, request.tool_name)
            outcome = method(request.resource, request.params) if request.tool_name != "read_inbox" else {
                "n_read": len(self.inbox.read_inbox())
            }
            safe = True
            if request.tool_name in ("send_email", "forward_email"):
                if not outcome.get("recipient_known", True):
                    safe = False
                    notes.append(
                        f"recipient '{outcome.get('recipient')}' does not match any known contact "
                        "in the simulated address book"
                    )

        elif request.tool_name in self.FILE_TOOLS:
            method = getattr(self.filesystem, request.tool_name)
            outcome = method(request.resource, request.params)
            safe = True
            if request.tool_name == "write_file" and outcome.get("overwrote_existing"):
                notes.append(f"write would overwrite existing file '{request.resource}'")
                # overwriting is a notable event but not unsafe on its own —
                # left as a note rather than a safety failure.
            if request.tool_name == "delete_file" and outcome.get("existed"):
                safe = False
                notes.append(f"delete would remove existing file '{request.resource}'")

        elif request.tool_name in self.CALENDAR_TOOLS:
            method = getattr(self.calendar, request.tool_name)
            outcome = method(request.resource, request.params)
            safe = True
            if request.tool_name == "create_event" and outcome.get("attendee_count", 0) > self.max_attendees:
                safe = False
                notes.append(
                    f"attendee_count {outcome['attendee_count']} exceeds sandbox safety cap {self.max_attendees}"
                )

        elif request.tool_name in self.CODE_TOOLS:
            method = getattr(self.code_exec, request.tool_name)
            outcome = method(request.resource, request.params)
            safe = True
            if request.tool_name == "run_code":
                if outcome.get("cumulative_cpu_seconds", 0) > self.code_exec.max_cumulative_cpu_seconds:
                    safe = False
                    notes.append(
                        f"cumulative CPU time {outcome['cumulative_cpu_seconds']}s across this session exceeds "
                        f"sandbox cap {self.code_exec.max_cumulative_cpu_seconds}s — a sequence of individually "
                        "small timeouts adds up to sustained resource use no single-call check would catch"
                    )

        elif request.tool_name in self.SUPPORT_TOOLS:
            method = getattr(self.support, request.tool_name)
            outcome = method(request.resource, request.params)
            safe = True
            if request.tool_name == "issue_refund":
                if outcome.get("cumulative_refunded_usd", 0) > self.support.max_cumulative_refund_usd:
                    safe = False
                    notes.append(
                        f"cumulative refunds on order '{request.resource}' "
                        f"(${outcome['cumulative_refunded_usd']:.2f}) exceed sandbox cap "
                        f"${self.support.max_cumulative_refund_usd:.2f} — each individual refund may be within "
                        "the privilege layer's per-call limit while the running total isn't, which only a "
                        "stateful check across calls can catch"
                    )

        elif request.tool_name in self.WEB_TOOLS:
            method = getattr(self.web, request.tool_name)
            outcome = method(request.resource, request.params)
            safe = True
            if request.tool_name == "fetch_url":
                if not outcome.get("domain_trusted", True):
                    safe = False
                    notes.append(f"domain '{outcome.get('domain')}' is not on the trusted-domain allowlist")
                if outcome.get("distinct_domains_fetched", 0) > self.web.max_distinct_domains:
                    safe = False
                    notes.append(
                        f"{outcome['distinct_domains_fetched']} distinct domains fetched this session exceeds "
                        f"sandbox cap {self.web.max_distinct_domains} — no single fetch call looks unusual, "
                        "but the sequence does"
                    )

        else:
            outcome = {}
            safe = True
            notes.append(f"no twin backend for tool '{request.tool_name}' — nothing simulated, assumed safe")

        return SandboxResult(
            tool_name=request.tool_name, resource=request.resource,
            twin_outcome=outcome, safe=safe, safety_notes=notes,
        )
