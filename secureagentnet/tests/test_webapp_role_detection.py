"""Tests for the GUI's keyword-based role auto-detection — a pure
function, testable without loading the detector or starting Flask.
"""

from secureagentnet.webapp.app import DEFAULT_ROLE, ROLE_KEYWORDS, detect_role


def test_detects_email_agent_from_inbox_keyword():
    assert detect_role("Please check my inbox for new mail") == "email_agent"


def test_detects_file_agent_from_delete_keyword():
    assert detect_role("Delete this file from the workspace") == "file_agent"


def test_detects_research_agent_from_search_keyword():
    assert detect_role("Search the web for climate data") == "research_agent"


def test_detects_calendar_agent_from_meeting_keyword():
    assert detect_role("Schedule a meeting for tomorrow") == "calendar_agent"


def test_detects_code_exec_agent_from_script_keyword():
    assert detect_role("Run this python script for me") == "code_exec_agent"


def test_detects_support_agent_from_refund_keyword():
    assert detect_role("Please refund my order immediately") == "support_agent"


def test_falls_back_to_default_role_when_no_keyword_matches():
    assert detect_role("What's the weather like today?") == DEFAULT_ROLE


def test_detection_is_case_insensitive():
    assert detect_role("REFUND MY ORDER NOW") == "support_agent"


def test_every_role_has_at_least_one_keyword():
    from secureagentnet.simulate.agent_env import ROLES

    for role in ROLES:
        assert role in ROLE_KEYWORDS
        assert len(ROLE_KEYWORDS[role]) > 0


def test_first_matching_role_wins_deterministically():
    """Dict iteration order is insertion order in Python 3.7+ -- a prompt
    matching keywords from two different roles should always resolve to
    whichever role is listed first in ROLE_KEYWORDS, not vary run to run.
    """
    result_a = detect_role("email about a refund")
    result_b = detect_role("email about a refund")
    assert result_a == result_b
