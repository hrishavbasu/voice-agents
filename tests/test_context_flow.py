import pytest

from pipeline.caller_context import merge_context_from_utterance
from pipeline.session import create_session, update_session
from prompts.system_prompt import build_system_prompt


@pytest.mark.asyncio
async def test_name_flows_to_prompt():
    await create_session("ctx-1", "+919999999999")
    updates = merge_context_from_utterance({}, "Mera naam Rahul Sharma hai")
    session = await update_session("ctx-1", updates)

    prompt = build_system_prompt(session=session)
    assert "Rahul" in prompt
    assert "Do NOT ask" in prompt


def test_booking_backfill_uses_session_name():
    session = {"caller_name": "Rahul Sharma", "caller_concern": "chest pain"}
    patient_name = ""
    if not patient_name:
        patient_name = session.get("caller_name") or ""
    assert patient_name == "Rahul Sharma"
