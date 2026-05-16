import asyncio
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_prefetch_returns_audio_bytes():
    with patch(
        "pipeline.tts_prefetch.sarvam_synthesize_to_bytes",
        AsyncMock(return_value=b"\xff" * 320),
    ):
        from pipeline.tts_prefetch import TtsPrefetchSlot

        slot = TtsPrefetchSlot()
        slot.start("hello", "hi-IN", None)
        await asyncio.sleep(0.05)
        assert slot.take() == b"\xff" * 320


@pytest.mark.asyncio
async def test_cancel_discards_prefetch():
    with patch(
        "pipeline.tts_prefetch.sarvam_synthesize_to_bytes",
        AsyncMock(side_effect=asyncio.sleep(1)),
    ):
        from pipeline.tts_prefetch import TtsPrefetchSlot

        slot = TtsPrefetchSlot()
        slot.start("hello", "hi-IN", None)
        slot.cancel()
        await asyncio.sleep(0.05)
        assert slot.take() is None
