from secureagentnet.privilege.policy_engine import ToolCallRequest
from secureagentnet.simulate.digital_twin import CalendarTwin, DigitalTwinSandbox, FilesystemTwin, InboxTwin


def test_forward_to_known_contact_is_safe():
    """Matches the doc's own worked example, positive case: recipient
    matches a known contact.
    """
    inbox = InboxTwin(known_contacts={"manager@secureagentnet-corp.com"})
    sandbox = DigitalTwinSandbox(inbox=inbox)
    result = sandbox.run(ToolCallRequest(tool_name="forward_email", resource="manager@secureagentnet-corp.com"))
    assert result.safe
    assert result.twin_outcome["recipient_known"]


def test_forward_to_unknown_recipient_is_flagged_unsafe():
    """The doc's own worked example: forwarding to a domain that matches
    no known contact and isn't who the instruction named.
    """
    inbox = InboxTwin(known_contacts={"manager@secureagentnet-corp.com"})
    sandbox = DigitalTwinSandbox(inbox=inbox)
    result = sandbox.run(ToolCallRequest(tool_name="forward_email", resource="attacker@evil.com"))
    assert not result.safe
    assert any("known contact" in note for note in result.safety_notes)


def test_state_persists_across_calls_within_a_session():
    """The 'full simulated environment' requirement: a write followed by a
    read reflects the earlier write — state, not stateless stubs.
    """
    fs = FilesystemTwin()
    sandbox = DigitalTwinSandbox(filesystem=fs)
    sandbox.run(ToolCallRequest(tool_name="write_file", resource="/workspace/notes.md", params={"content": "hello"}))
    result = sandbox.run(ToolCallRequest(tool_name="read_file", resource="/workspace/notes.md"))
    assert result.twin_outcome["content"] == "hello"
    assert result.twin_outcome["exists"]


def test_deleting_existing_file_is_flagged_unsafe():
    fs = FilesystemTwin(files={"/workspace/report.docx": "important"})
    sandbox = DigitalTwinSandbox(filesystem=fs)
    result = sandbox.run(ToolCallRequest(tool_name="delete_file", resource="/workspace/report.docx"))
    assert not result.safe
    assert "/workspace/report.docx" not in fs.files  # twin state actually mutated


def test_deleting_nonexistent_file_is_safe_noop():
    fs = FilesystemTwin()
    sandbox = DigitalTwinSandbox(filesystem=fs)
    result = sandbox.run(ToolCallRequest(tool_name="delete_file", resource="/workspace/ghost.txt"))
    assert result.safe


def test_calendar_mass_invite_flagged_unsafe():
    sandbox = DigitalTwinSandbox(calendar=CalendarTwin(), max_attendees=10)
    result = sandbox.run(ToolCallRequest(tool_name="create_event", params={"attendee_count": 500}))
    assert not result.safe


def test_calendar_event_ids_increment_across_calls():
    calendar = CalendarTwin()
    sandbox = DigitalTwinSandbox(calendar=calendar)
    r1 = sandbox.run(ToolCallRequest(tool_name="create_event", params={"attendee_count": 2}))
    r2 = sandbox.run(ToolCallRequest(tool_name="create_event", params={"attendee_count": 2}))
    assert r1.twin_outcome["event_id"] != r2.twin_outcome["event_id"]
    assert len(calendar.events) == 2


def test_unrecognized_tool_defaults_to_safe_with_a_note():
    sandbox = DigitalTwinSandbox()
    result = sandbox.run(ToolCallRequest(tool_name="run_code", resource="/sandbox/x"))
    assert result.safe
    assert "no twin backend" in result.safety_notes[0]
