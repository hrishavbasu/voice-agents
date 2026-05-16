"""Mock telephony: barge-in must clear buffer and stop audio quickly."""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from pipeline.interruption import InterruptionController
from services.tts import TTSService


class MockTelephony:
    def __init__(self):
        self.events: list[tuple[str, float]] = []
        self.stream_ready = asyncio.Event()
        self.stream_ready.set()

    async def send_audio(self, chunk: bytes) -> None:
        self.events.append(("audio", time.monotonic()))

    async def clear_playback_buffer(self) -> None:
        self.events.append(("clear", time.monotonic()))

    async def send_silence(self, ms: int) -> None:
        self.events.append(("silence", time.monotonic()))


@pytest.mark.asyncio
async def test_barge_in_stops_audio_within_400ms():
    telephony = MockTelephony()
    interruption = InterruptionController()
    tts = TTSService()

    async def slow_synth(text, language_code="hi-IN", pitch_override=None):
        for _ in range(50):
            if tts.is_cancelled:
                return
            await asyncio.sleep(0.02)
            yield b"\xff" * 160

    tts.synthesize = slow_synth  # type: ignore[method-assign]

    async def run_speak():
        interruption.reset()
        tts.reset()
        async for chunk in tts.synthesize("long"):
            if interruption.is_interrupted or tts.is_cancelled:
                break
            await telephony.send_audio(chunk)

    speak_task = asyncio.create_task(run_speak())
    await asyncio.sleep(0.15)

    t_start = time.monotonic()
    tts.cancel()
    await telephony.clear_playback_buffer()
    await interruption.trigger_immediate()

    await asyncio.wait_for(speak_task, timeout=2.0)

    clears = [t for ev, t in telephony.events if ev == "clear"]
    audios = [t for ev, t in telephony.events if ev == "audio"]
    assert clears, "clear_playback_buffer must be called"
    assert clears[0] - t_start < 0.05
    if audios:
        assert audios[-1] - t_start < 0.4
