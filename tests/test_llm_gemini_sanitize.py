"""Tests for Gemini message history sanitization."""
import json

from services.llm_gemini import sanitize_messages_for_gemini, trim_messages_for_llm


def test_merges_consecutive_assistant_text():
    msgs = [
        {"role": "assistant", "content": "ठीक है।"},
        {"role": "assistant", "content": "क्या आप उनसे appointment लेना चाहेंगी?"},
        {"role": "user", "content": "हां"},
    ]
    out = sanitize_messages_for_gemini(msgs)
    assert len(out) == 2
    assert "ठीक है" in out[0]["content"]
    assert "appointment" in out[0]["content"]


def test_drops_orphan_tool_calls_without_result():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "user", "content": "hello"},
    ]
    out = sanitize_messages_for_gemini(msgs)
    assert out == [{"role": "user", "content": "hello"}]


def test_keeps_tool_call_pair():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "check_doctor_slots", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "check_doctor_slots", "content": json.dumps({"success": True})},
        {"role": "user", "content": "nine am"},
    ]
    out = sanitize_messages_for_gemini(msgs)
    assert len(out) == 3
    assert out[0].get("tool_calls")
    assert out[1]["role"] == "tool"


def test_trim_preserves_tool_pair_at_boundary():
    rest = []
    for i in range(14):
        rest.append({"role": "user", "content": f"u{i}"})
        rest.append({"role": "assistant", "content": f"a{i}"})
    rest.extend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "{}"},
        {"role": "user", "content": "final"},
    ])
    trimmed = trim_messages_for_llm(rest, max_non_system=4)
    assert trimmed[-1]["content"] == "final"
    tool_idx = next(
        (i for i, m in enumerate(trimmed) if m.get("role") == "assistant" and m.get("tool_calls")),
        None,
    )
    if tool_idx is not None:
        assert trimmed[tool_idx + 1]["role"] == "tool"
