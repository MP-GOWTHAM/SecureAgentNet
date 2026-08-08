"""Digital twin sandbox (methodology §2.6) — full simulated environment.

Per the doc's own feasibility caution (§8), "full simulated environment"
means stateful mock backends that respond dynamically to sequences of
calls, not one-shot stub responses. This module implements that: an
`Inbox`, `Filesystem`, and `Calendar` twin, each with real internal state
that mutates across calls within a session, plus a `DigitalTwinSandbox`
that intercepts a planned tool call, redirects it to the matching twin,
and evaluates the twin's resulting state against safety criteria — the
concrete, checkable mismatch the doc's `forward_email` example describes
(recipient domain doesn't match any known contact), not just "did the call
error".

Scope: this simulates enough backend behavior to make pre-execution safety
checks meaningful for the six agent roles already defined in
`privilege/policies/` — it is not a general-purpose service emulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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

    def __init__(
        self,
        inbox: Optional[InboxTwin] = None,
        filesystem: Optional[FilesystemTwin] = None,
        calendar: Optional[CalendarTwin] = None,
        max_attendees: int = 10,
    ):
        self.inbox = inbox or InboxTwin()
        self.filesystem = filesystem or FilesystemTwin()
        self.calendar = calendar or CalendarTwin()
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

        else:
            outcome = {}
            safe = True
            notes.append(f"no twin backend for tool '{request.tool_name}' — nothing simulated, assumed safe")

        return SandboxResult(
            tool_name=request.tool_name, resource=request.resource,
            twin_outcome=outcome, safe=safe, safety_notes=notes,
        )
