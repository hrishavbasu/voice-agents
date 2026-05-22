import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.voice_pipeline import VoicePipeline


@pytest.mark.asyncio
async def test_first_oos_declines_without_escalate():
    telephony = MagicMock()
    telephony.clear_playback_buffer = AsyncMock()
    p = VoicePipeline("oos-1", "+910000000001", telephony)
    p._running = True
    p._speak = AsyncMock(return_value=1.0)

    session = {
        "out_of_scope_strikes": 0,
        "last_oos_category": None,
        "caller_language": "hinglish",
    }
    with patch("pipeline.voice_pipeline.append_message", AsyncMock()):
        with patch("pipeline.voice_pipeline.update_session", AsyncMock(return_value=session)):
            with patch("tools.escalation.escalate_to_human", AsyncMock()) as esc:
                handled = await p._handle_out_of_scope("I need a refund on my bill", session)

    assert handled is True
    assert p._speak.called
    assert not esc.called


@pytest.mark.asyncio
async def test_repeat_oos_escalates():
    telephony = MagicMock()
    p = VoicePipeline("oos-2", "+910000000002", telephony)
    p._running = True
    p._speak = AsyncMock(return_value=1.0)

    session = {
        "out_of_scope_strikes": 1,
        "last_oos_category": "billing",
        "caller_language": "english",
        "crm_contact_id": None,
    }
    with patch("pipeline.voice_pipeline.get_transcript", AsyncMock(return_value="")):
        with patch("pipeline.voice_pipeline.append_message", AsyncMock()):
            with patch("pipeline.voice_pipeline.update_session", AsyncMock(return_value=session)):
                with patch(
                    "tools.escalation.escalate_to_human",
                    AsyncMock(return_value={"success": True}),
                ) as esc:
                    handled = await p._handle_out_of_scope("my bill is wrong again", session)

    assert handled is True
    assert esc.called


@pytest.mark.asyncio
async def test_insist_escalates_on_first_oos():
    telephony = MagicMock()
    p = VoicePipeline("oos-3", "+910000000003", telephony)
    p._running = True
    p._speak = AsyncMock(return_value=1.0)

    session = {
        "out_of_scope_strikes": 0,
        "last_oos_category": None,
        "caller_language": "english",
    }
    with patch("pipeline.voice_pipeline.get_transcript", AsyncMock(return_value="")):
        with patch("pipeline.voice_pipeline.append_message", AsyncMock()):
            with patch(
                "tools.escalation.escalate_to_human",
                AsyncMock(return_value={"success": True}),
            ) as esc:
                handled = await p._handle_out_of_scope(
                    "refund my bill, connect me to a human agent",
                    session,
                )

    assert handled is True
    assert esc.called
