"""Tests for Sarvam TTS HTTP-blocking barge-in fix and prosody params."""
import asyncio
import base64
import io
import wave
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_wav_b64(n_frames: int = 2000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * n_frames)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(autouse=True)
def clear_cache():
    from services import tts_sarvam
    tts_sarvam._audio_cache.clear()
    tts_sarvam._http_client = None
    yield
    tts_sarvam._audio_cache.clear()
    tts_sarvam._http_client = None


@pytest.mark.asyncio
async def test_pre_cancelled_skips_http_request():
    """If cancelled_flag is True before the call, no HTTP request is made."""
    mock_client = AsyncMock()

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [True]  # already cancelled
            chunks = []
            async for chunk in sarvam_synthesize("test", cancelled):
                chunks.append(chunk)

    assert chunks == []
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_during_http_fetch_stops_generation():
    """Cancel flag set while HTTP request is in flight stops synthesis."""
    wav_b64 = _make_wav_b64(n_frames=44100)

    async def slow_post(*args, **kwargs):
        await asyncio.sleep(0.2)  # simulate network latency
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"audios": [wav_b64]}
        return resp

    mock_client = AsyncMock()
    mock_client.post = slow_post
    cancelled = [False]

    async def cancel_after_50ms():
        await asyncio.sleep(0.05)
        cancelled[0] = True

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            chunks = []
            await asyncio.gather(
                cancel_after_50ms(),
                _collect(sarvam_synthesize("long text", cancelled), chunks),
            )

    assert chunks == [], "Should yield no chunks when cancelled mid-fetch"


async def _collect(gen, out: list):
    async for item in gen:
        out.append(item)


@pytest.mark.asyncio
async def test_prosody_params_sent_in_payload():
    """pace, pitch, loudness appear in the Sarvam API request payload."""
    wav_b64 = _make_wav_b64()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [wav_b64]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            import services.tts_sarvam as mod
            with patch.object(mod, "SARVAM_TTS_PACE", 0.9), \
                 patch.object(mod, "SARVAM_TTS_PITCH", 0.05), \
                 patch.object(mod, "SARVAM_TTS_LOUDNESS", 1.5):
                from services.tts_sarvam import sarvam_synthesize
                cancelled = [False]
                async for _ in sarvam_synthesize("test", cancelled, pitch_override=0.05):
                    pass

    payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args.args[1]
    assert payload["pace"] == 0.9
    assert payload["pitch"] == 0.05
    assert payload["loudness"] == 1.5
