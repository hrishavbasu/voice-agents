"""Tests for session management (pipeline/session.py)."""

import pytest

import memory.store as store_module
from memory.store import InMemoryStore


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    """Each test gets an isolated in-memory store."""
    instance = InMemoryStore()
    monkeypatch.setattr(store_module, "_store_instance", instance)
    yield instance
    store_module._store_instance = None


# Import after monkeypatching so the module picks up the patched store
from pipeline.session import (
    create_session,
    get_session,
    update_session,
    append_message,
    increment_retry,
    get_transcript,
    end_session,
)


@pytest.mark.asyncio
async def test_create_session_returns_dict():
    session = await create_session("call-001", "+919999999999")
    assert session["call_id"] == "call-001"
    assert session["caller_phone"] == "+919999999999"
    assert session["flow_state"] == "greeting"
    assert session["messages"] == []
    assert session["retry_count"] == 0


@pytest.mark.asyncio
async def test_create_session_is_persisted():
    await create_session("call-002", "+91111")
    fetched = await get_session("call-002")
    assert fetched is not None
    assert fetched["call_id"] == "call-002"


@pytest.mark.asyncio
async def test_get_session_missing_returns_none():
    result = await get_session("no-such-call")
    assert result is None


@pytest.mark.asyncio
async def test_update_session_merges_fields():
    await create_session("call-003", "+91222")
    await update_session("call-003", {"flow_state": "scheduling", "crm_contact_id": "CRM-1"})
    session = await get_session("call-003")
    assert session["flow_state"] == "scheduling"
    assert session["crm_contact_id"] == "CRM-1"
    # Original fields preserved
    assert session["caller_phone"] == "+91222"


@pytest.mark.asyncio
async def test_append_message_adds_to_history():
    await create_session("call-004", "+91333")
    await append_message("call-004", "user", "Hello, I need a doctor.")
    session = await get_session("call-004")
    assert len(session["messages"]) == 1
    assert session["messages"][0] == {"role": "user", "content": "Hello, I need a doctor."}


@pytest.mark.asyncio
async def test_append_message_with_extra_fields():
    await create_session("call-005", "+91444")
    await append_message(
        "call-005",
        "assistant",
        None,
        extra={"tool_calls": [{"id": "tc1", "function": {"name": "book_appointment"}}]},
    )
    session = await get_session("call-005")
    msg = session["messages"][0]
    assert msg["role"] == "assistant"
    assert "tool_calls" in msg


@pytest.mark.asyncio
async def test_append_message_updates_transcript_lines():
    await create_session("call-006", "+91555")
    await append_message("call-006", "user", "Hi there")
    await append_message("call-006", "assistant", "Hello, how can I help?")
    transcript = await get_transcript("call-006")
    assert "[USER] Hi there" in transcript
    assert "[ASSISTANT] Hello, how can I help?" in transcript


@pytest.mark.asyncio
async def test_increment_retry_starts_at_one():
    await create_session("call-007", "+91666")
    count = await increment_retry("call-007")
    assert count == 1


@pytest.mark.asyncio
async def test_increment_retry_accumulates():
    await create_session("call-008", "+91777")
    await increment_retry("call-008")
    count = await increment_retry("call-008")
    assert count == 2


@pytest.mark.asyncio
async def test_get_transcript_empty_for_new_session():
    await create_session("call-009", "+91888")
    transcript = await get_transcript("call-009")
    assert transcript == ""


@pytest.mark.asyncio
async def test_get_transcript_missing_call_returns_empty():
    result = await get_transcript("ghost-call")
    assert result == ""


@pytest.mark.asyncio
async def test_end_session_sets_flow_state():
    await create_session("call-010", "+91999")
    await end_session("call-010")
    session = await get_session("call-010")
    assert session["flow_state"] == "ended"
    assert "end_time" in session
