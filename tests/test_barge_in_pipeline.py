"""VoicePipeline barge-in: cancel, clear buffer, no ack when silent mode."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

class MockTelephony:
    def __init__(self):
        self.events: list[str] = []
        self.stream_ready = asyncio.Event()
        self.stream_ready.set()

    async def clear_playback_buffer(self):
        self.events.append("clear")

    async def send_silence(self, ms: int):
        self.events.append("silence")

    async def send_audio(self, chunk: bytes):
        self.events.append("audio")


@pytest.mark.asyncio
async def test_on_speech_started_clears_and_interrupts():
    from pipeline.voice_pipeline import VoicePipeline

    telephony = MockTelephony()
    pipeline = VoicePipeline("bi-1", "+911111111111", telephony)
    pipeline._running = True
    pipeline._greeting_finished = True
    pipeline._tts_playing = True
    pipeline._agent_in_turn = True
    pipeline._playback_until = time.monotonic() + 5
    pipeline._sentences_spoken_this_turn = 2
    pipeline._transcript_queue.put_nowait("नौ बजे")

    pipeline._tts = MagicMock()
    pipeline._tts.cancel = MagicMock()
    pipeline._interruption.reset()
    pipeline._cancel_prefetch = MagicMock()
    pipeline._speak = AsyncMock()

    with patch("pipeline.voice_pipeline.BARGE_IN_ACK_MODE", "silent"):
        await pipeline._on_speech_started()

    pipeline._tts.cancel.assert_called_once()
    assert "clear" in telephony.events
    assert pipeline._interruption.is_interrupted
    assert not pipeline._transcript_queue.empty()
    pipeline._speak.assert_not_called()


@pytest.mark.asyncio
async def test_barge_in_stops_slow_speak_within_400ms():
    from pipeline.interruption import InterruptionController
    from services.tts import TTSService

    telephony = MockTelephony()
    interruption = InterruptionController()
    tts = TTSService()

    async def slow_synth(text, language_code="hi-IN", pitch_override=None):
        for _ in range(40):
            if tts.is_cancelled:
                return
            await asyncio.sleep(0.02)
            yield b"\xff" * 160

    tts.synthesize = slow_synth  # type: ignore

    async def run():
        tts.reset()
        async for chunk in tts.synthesize("long"):
            if interruption.is_interrupted or tts.is_cancelled:
                break
            await telephony.send_audio(chunk)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.12)
    t0 = time.monotonic()
    tts.cancel()
    await telephony.clear_playback_buffer()
    await interruption.trigger_immediate()
    await asyncio.wait_for(task, timeout=2.0)
    assert time.monotonic() - t0 < 0.5
    assert "clear" in telephony.events
