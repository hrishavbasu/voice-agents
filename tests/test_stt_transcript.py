"""
Tests for DeepgramSTT utterance-accumulation and overlap logic.

All Deepgram SDK calls are mocked — no network access required.
"""

import asyncio
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the submodule is in sys.modules so patch() can resolve it
import services.stt  # noqa: F401


# ── Helpers to build fake Deepgram result objects ─────────────────────────────

def _make_word(text: str, speaker: int = 0):
    w = MagicMock()
    w.speaker = speaker
    w.word = text
    return w


def _make_result(
    transcript: str,
    is_final: bool,
    speech_final: bool,
    confidence: float = 0.95,
    words: list | None = None,
):
    alt = MagicMock()
    alt.transcript = transcript
    alt.confidence = confidence
    alt.words = words or [_make_word(t) for t in transcript.split()]

    channel = MagicMock()
    channel.alternatives = [alt]

    result = MagicMock()
    result.channel = channel
    result.is_final = is_final
    result.speech_final = speech_final
    return result


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def stt(monkeypatch):
    """Return a DeepgramSTT with its Deepgram client fully mocked."""
    with patch("services.stt.DeepgramClient") as MockClient:
        mock_ws = AsyncMock()
        mock_ws.start = AsyncMock(return_value=True)
        mock_ws.on = MagicMock()
        mock_ws.keep_alive = AsyncMock()
        mock_ws.finish = AsyncMock()

        mock_listen = MagicMock()
        mock_listen.asyncwebsocket.v.return_value = mock_ws

        MockClient.return_value.listen = mock_listen

        from services.stt import DeepgramSTT

        received: list[tuple[str, bool]] = []

        async def on_transcript(text, is_final):
            received.append((text, is_final))

        instance = DeepgramSTT(on_transcript=on_transcript)
        # Expose received for assertions
        instance._test_received = received
        yield instance


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_speech_final_yields_utterance(stt):
    result = _make_result("I need an appointment", is_final=True, speech_final=True)
    await stt._handle_transcript(None, result)
    assert len(stt._test_received) == 1
    assert "appointment" in stt._test_received[0][0]


@pytest.mark.asyncio
async def test_non_final_fragment_is_accumulated(stt):
    fragment = _make_result("I need", is_final=True, speech_final=False)
    await stt._handle_transcript(None, fragment)
    assert len(stt._test_received) == 0
    assert stt._utterance_parts == ["I need"]


@pytest.mark.asyncio
async def test_final_revision_replaces_accumulated_parts(stt):
    """speech_final with high word overlap → use only the final transcript."""
    stt._utterance_parts = ["I need an appointment"]
    revised = _make_result(
        "I need an appointment.",
        is_final=True,
        speech_final=True,
    )
    await stt._handle_transcript(None, revised)
    assert len(stt._test_received) == 1
    assert stt._test_received[0][0] == "I need an appointment."


@pytest.mark.asyncio
async def test_speech_final_flushes_accumulated_when_empty_transcript(stt):
    stt._utterance_parts = ["Part one", "part two"]
    empty = _make_result("", is_final=True, speech_final=True)
    await stt._handle_transcript(None, empty)
    assert len(stt._test_received) == 1
    assert "Part one" in stt._test_received[0][0]


@pytest.mark.asyncio
async def test_single_word_utterance_discarded(stt):
    short = _make_result("Hi", is_final=True, speech_final=True)
    await stt._handle_transcript(None, short)
    assert len(stt._test_received) == 0


@pytest.mark.asyncio
async def test_low_confidence_fragment_discarded(stt):
    low_conf = _make_result(
        "mumbai appointment", is_final=True, speech_final=False, confidence=0.5
    )
    await stt._handle_transcript(None, low_conf)
    assert stt._utterance_parts == []


@pytest.mark.asyncio
async def test_non_primary_speaker_ignored(stt):
    # Lock primary speaker to 0 first
    stt._primary_speaker = 0
    intruder_words = [_make_word("noise", speaker=1)]
    intruder = _make_result(
        "noise noise", is_final=True, speech_final=True, words=intruder_words
    )
    await stt._handle_transcript(None, intruder)
    assert len(stt._test_received) == 0


@pytest.mark.asyncio
async def test_primary_speaker_locked_on_first_speech_final(stt):
    assert stt._primary_speaker is None
    words = [_make_word("hello", speaker=2), _make_word("world", speaker=2)]
    result = _make_result("hello world", is_final=True, speech_final=True, words=words)
    await stt._handle_transcript(None, result)
    assert stt._primary_speaker == 2


@pytest.mark.asyncio
async def test_utterance_parts_cleared_after_speech_final(stt):
    fragment = _make_result("some text", is_final=True, speech_final=False)
    await stt._handle_transcript(None, fragment)
    assert stt._utterance_parts

    final = _make_result("some text complete", is_final=True, speech_final=True)
    await stt._handle_transcript(None, final)
    assert stt._utterance_parts == []


@pytest.mark.asyncio
async def test_handle_speech_started_clears_parts(stt):
    stt._utterance_parts = ["stale", "data"]
    await stt._handle_speech_started(None, None)
    assert stt._utterance_parts == []
