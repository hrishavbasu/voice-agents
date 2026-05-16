"""Tests for Sarvam STT WebSocket client."""
import pytest

from services.stt_sarvam import SarvamSTT, build_sarvam_stt_ws_url, _LEGACY_SUBSCRIBE_PATH


@pytest.fixture
def make_stt():
    """Factory: return a SarvamSTT with mock callbacks and tracking lists."""
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


def test_build_url_uses_speech_to_text_ws():
    url = build_sarvam_stt_ws_url()
    assert "/speech-to-text/ws" in url
    assert _LEGACY_SUBSCRIBE_PATH not in url
    assert "sample_rate=16000" in url
    assert "language-code=hi-IN" in url


def test_build_url_replaces_legacy_subscribe(monkeypatch):
    monkeypatch.setenv(
        "SARVAM_STT_URL",
        "wss://api.sarvam.ai/speech-to-text-translate/subscribe",
    )
    url = build_sarvam_stt_ws_url()
    assert "/speech-to-text/ws" in url


@pytest.mark.asyncio
async def test_start_speech_fires_callback(make_stt):
    stt = make_stt()
    await stt._handle_message({
        "type": "events",
        "data": {"signal_type": "START_SPEECH"},
    })
    assert len(stt._speech_starts) == 1


@pytest.mark.asyncio
async def test_end_speech_flushes_transcript(make_stt):
    stt = make_stt()
    await stt._handle_message({
        "type": "data",
        "data": {"transcript": "hello there"},
    })
    await stt._handle_message({
        "type": "events",
        "data": {"signal_type": "END_SPEECH"},
    })
    assert len(stt._transcripts) == 1
    assert "hello there" in stt._transcripts[0][0]


@pytest.mark.asyncio
async def test_end_before_data_still_flushes(make_stt):
    """Sarvam often sends END_SPEECH before the type=data transcript."""
    stt = make_stt()
    await stt._handle_message({
        "type": "events",
        "data": {"signal_type": "START_SPEECH"},
    })
    await stt._handle_message({
        "type": "events",
        "data": {"signal_type": "END_SPEECH"},
    })
    assert len(stt._transcripts) == 0
    await stt._handle_message({
        "type": "data",
        "data": {"transcript": "मुझे appointment चाहिए"},
    })
    assert len(stt._transcripts) == 1
    assert "appointment" in stt._transcripts[0][0]


@pytest.mark.asyncio
async def test_data_segments_accumulate_until_end(make_stt):
    stt = make_stt()
    await stt._handle_message({"type": "data", "data": {"transcript": "I want"}})
    await stt._handle_message({"type": "data", "data": {"transcript": "an appointment"}})
    await stt._handle_message({"type": "events", "data": {"signal_type": "END_SPEECH"}})
    text = stt._transcripts[0][0]
    assert "I want" in text
    assert "appointment" in text


@pytest.mark.asyncio
async def test_single_word_utterance_accepted(make_stt):
    stt = make_stt()
    await stt._handle_message({"type": "data", "data": {"transcript": "हाँ"}})
    await stt._handle_message({"type": "events", "data": {"signal_type": "END_SPEECH"}})
    assert len(stt._transcripts) == 1


@pytest.mark.asyncio
async def test_legacy_transcript_final(make_stt):
    stt = make_stt()
    await stt._handle_message({
        "type": "transcript",
        "transcript": "hello world",
        "is_final": True,
    })
    assert stt._transcripts[0][0] == "hello world"
