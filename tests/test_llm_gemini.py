"""Tests for Gemini 2.5 Pro LLM client — message conversion and streaming."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Message conversion tests (no network needed) ──────────────────────────────

def test_system_message_extracted_as_instruction():
    from services.llm_gemini import _convert_messages
    msgs = [{"role": "system", "content": "You are Priya."}]
    system, contents = _convert_messages(msgs)
    assert system == "You are Priya."
    assert contents == []


def test_user_message_becomes_user_content():
    from services.llm_gemini import _convert_messages
    msgs = [{"role": "user", "content": "Hello"}]
    _, contents = _convert_messages(msgs)
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello"


def test_assistant_message_becomes_model_content():
    from services.llm_gemini import _convert_messages
    msgs = [{"role": "assistant", "content": "Hi there!"}]
    _, contents = _convert_messages(msgs)
    assert contents[0].role == "model"
    assert contents[0].parts[0].text == "Hi there!"


def test_tool_result_becomes_function_response():
    from services.llm_gemini import _convert_messages
    msgs = [{
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "check_doctor_slots",
        "content": json.dumps({"slots": ["9am", "11am"]}),
    }]
    _, contents = _convert_messages(msgs)
    assert contents[0].role == "user"
    part = contents[0].parts[0]
    assert part.function_response.name == "check_doctor_slots"
    assert part.function_response.response == {"slots": ["9am", "11am"]}


def test_assistant_with_tool_calls_becomes_function_call():
    from services.llm_gemini import _convert_messages
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "list_doctors", "arguments": '{"specialty": "cardiology"}'},
        }],
    }]
    _, contents = _convert_messages(msgs)
    assert contents[0].role == "model"
    assert contents[0].parts[0].function_call.name == "list_doctors"
    assert contents[0].parts[0].function_call.args == {"specialty": "cardiology"}


def test_tool_schema_conversion():
    from services.llm_gemini import _convert_tools
    openai_tools = [{
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment",
            "parameters": {
                "type": "object",
                "properties": {"doctor_name": {"type": "string"}},
                "required": ["doctor_name"],
            },
        }
    }]
    gemini_tools = _convert_tools(openai_tools)
    assert len(gemini_tools) == 1
    decl = gemini_tools[0].function_declarations[0]
    assert decl.name == "book_appointment"
    assert decl.description == "Book an appointment"


def test_empty_tool_list_returns_empty():
    from services.llm_gemini import _convert_tools
    assert _convert_tools([]) == []
    assert _convert_tools(None) == []


def test_trim_messages_default_keeps_20_non_system():
    from services.llm_gemini import trim_messages_for_llm
    system = {"role": "system", "content": "You are Priya."}
    # Build 22 alternating user/assistant messages
    rest = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(22)
    ]
    result = trim_messages_for_llm([system] + rest)
    non_system = [m for m in result if m["role"] != "system"]
    assert len(non_system) == 20


def test_trim_messages_explicit_12_still_works():
    from services.llm_gemini import trim_messages_for_llm
    system = {"role": "system", "content": "sys"}
    rest = [{"role": "user", "content": f"m{i}"} for i in range(15)]
    result = trim_messages_for_llm([system] + rest, max_non_system=12)
    non_system = [m for m in result if m["role"] != "system"]
    assert len(non_system) == 12
