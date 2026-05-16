"""Emergency path must advise 108 and transfer to human."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_pipeline():
    from pipeline.voice_pipeline import VoicePipeline

    telephony = MagicMock()
    telephony.clear_playback_buffer = AsyncMock()
    telephony.transfer = AsyncMock()

    pipeline = VoicePipeline("call-emg-1", "+919876543210", telephony)
    pipeline._running = True
    pipeline._speak = AsyncMock(return_value=3.0)
    return pipeline


@pytest.mark.asyncio
async def test_emergency_chest_pain_triggers_human_transfer(mock_pipeline):
    with patch("pipeline.voice_pipeline.get_transcript", AsyncMock(return_value="caller: chest pain")):
        with patch("pipeline.voice_pipeline.get_session", AsyncMock(return_value={})):
            with patch("tools.escalation.escalate_to_human", AsyncMock(return_value={"success": True})) as esc:
                fired = await mock_pipeline._check_emergency("I have severe chest pain")

    assert fired is True
    esc.assert_called_once()
    kwargs = esc.call_args.kwargs
    assert kwargs["priority"] == "EMERGENCY"
    assert "EMERGENCY" in kwargs["reason"]
    assert kwargs["telephony_session"] is mock_pipeline._telephony
    mock_pipeline._speak.assert_called_once()
    assert mock_pipeline._running is False


@pytest.mark.asyncio
async def test_mild_symptoms_do_not_trigger_emergency(mock_pipeline):
    with patch("tools.escalation.escalate_to_human", AsyncMock()) as esc:
        fired = await mock_pipeline._check_emergency("I have mild chest discomfort, not severe.")

    assert fired is False
    esc.assert_not_called()


@pytest.mark.asyncio
async def test_hindi_emergency_phrase_triggers_transfer(mock_pipeline):
    with patch("pipeline.voice_pipeline.get_transcript", AsyncMock(return_value="")):
        with patch("pipeline.voice_pipeline.get_session", AsyncMock(return_value={})):
            with patch("tools.escalation.escalate_to_human", AsyncMock(return_value={"success": True})) as esc:
                fired = await mock_pipeline._check_emergency("मुझे छाती में दर्द हो रहा है")

    assert fired is True
    esc.assert_called_once()
