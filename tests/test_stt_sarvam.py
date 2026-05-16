"""Tests for Sarvam STT WebSocket client."""
import pytest


@pytest.fixture
def make_stt():
    """Factory: return a SarvamSTT with mock callbacks and tracking lists."""
    from services.stt_sarvam import SarvamSTT

    def _make(on_transcript=None, on_speech_started=None):
        transcripts = []
        speech_starts = []

        async def _on_transcript(text, is_final):
            transcripts.append((text, is_final))

        async def _on_speech_started():
            speech_starts.append(True)

        stt = SarvamSTT(
            on_transcript=on_transcript or _on_transcript,
            on_speech_started=on_speech_started or _on_speech_started,
        )
        stt._transcripts = transcripts
        stt._speech_starts = speech_starts
        return stt

    return _make


@pytest.mark.asyncio
async def test_speech_started_fires_callback(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "speech_started"})
    assert len(stt._speech_starts) == 1


@pytest.mark.asyncio
async def test_speech_started_clears_utterance_parts(make_stt):
    stt = make_stt()
    stt._utterance_parts = ["partial", "text"]
    await stt._handle_event({"type": "speech_started"})
    assert stt._utterance_parts == []


@pytest.mark.asyncio
async def test_final_transcript_fires_callback(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "transcript", "transcript": "hello world", "is_final": True})
    assert len(stt._transcripts) == 1
    assert stt._transcripts[0][0] == "hello world"
    assert stt._transcripts[0][1] is True


@pytest.mark.asyncio
async def test_non_final_transcript_accumulates(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "transcript", "transcript": "hello", "is_final": False})
    assert stt._utterance_parts == ["hello"]
    assert len(stt._transcripts) == 0


@pytest.mark.asyncio
async def test_final_transcript_appends_accumulated_parts(make_stt):
    stt = make_stt()
    stt._utterance_parts = ["I want to book"]
    await stt._handle_event({"type": "transcript", "transcript": "an appointment", "is_final": True})
    text = stt._transcripts[0][0]
    assert "I want to book" in text
    assert "an appointment" in text


@pytest.mark.asyncio
async def test_single_word_utterance_discarded(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "transcript", "transcript": "हाँ", "is_final": True})
    assert len(stt._transcripts) == 0


@pytest.mark.asyncio
async def test_empty_transcript_ignored(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "transcript", "transcript": "", "is_final": True})
    assert len(stt._transcripts) == 0


@pytest.mark.asyncio
async def test_unknown_event_type_ignored(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "heartbeat"})
    assert len(stt._speech_starts) == 0
    assert len(stt._transcripts) == 0
