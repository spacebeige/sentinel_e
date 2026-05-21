import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.browser_runtime.groq_router import GroqBrowserRouter
from agents.browser_runtime.memory_manager import BrowserMemoryManager
from agents.browser_runtime.permissions import BrowserPermissionPolicy


def test_permission_policy_allows_safe_scroll():
    decision = BrowserPermissionPolicy().evaluate({"type": "scroll", "direction": "down"})

    assert decision.allowed is True
    assert decision.requires_confirmation is False


def test_permission_policy_requires_confirmation_for_submit_like_click():
    page_state = {
        "buttons": [
            {"selector": "#send", "text": "Send email"},
        ],
        "inputs": [],
    }

    decision = BrowserPermissionPolicy().evaluate({"type": "click", "selector": "#send"}, page_state)

    assert decision.allowed is False
    assert decision.requires_confirmation is True


def test_memory_manager_records_compact_selector_memory(tmp_path):
    path = tmp_path / "browser_memory.json"
    memory = BrowserMemoryManager(path=str(path))

    memory.record_selector("https://example.com/page", "#search", "fill", True)
    memory.record_selector("https://example.com/page", "#missing", "click", False, note="not found")

    snapshot = memory.snapshot()
    assert "https://example.com/page" in snapshot["successful_selectors"]
    assert "https://example.com/page" in snapshot["failed_selectors"]


def test_groq_json_extraction_handles_fenced_json():
    content = '```json\n{"action":{"type":"extract"},"done":false}\n```'

    parsed = GroqBrowserRouter._extract_json(content)

    assert parsed == {"action": {"type": "extract"}, "done": False}
