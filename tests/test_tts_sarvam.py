"""Tests for Sarvam Bulbul TTS provider."""
import asyncio
import audioop
import base64
import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_wav_b64(sample_rate: int = 22050, n_frames: int = 2000) -> str:
    """Create a minimal valid WAV file encoded as base64."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(autouse=True)
def reset_sarvam_cache():
    """Clear the LRU cache between tests."""
    from services import tts_sarvam
    tts_sarvam._audio_cache.clear()
    tts_sarvam._http_client = None
    yield
    tts_sarvam._audio_cache.clear()
    tts_sarvam._http_client = None


@pytest.mark.asyncio
async def test_synthesize_returns_ulaw_bytes():
    """sarvam_synthesize yields non-empty bytes for a simple Hindi text."""
    wav_b64 = _make_wav_b64()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [wav_b64]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [False]
            chunks = []
            async for chunk in sarvam_synthesize("नमस्ते", cancelled, language_code="hi-IN"):
                chunks.append(chunk)

    assert chunks, "Expected at least one audio chunk"
    assert all(isinstance(c, bytes) for c in chunks)


@pytest.mark.asyncio
async def test_synthesize_uses_explicit_language_code():
    """language_code parameter is passed to Sarvam API payload, not inferred from text."""
    wav_b64 = _make_wav_b64()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [wav_b64]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [False]
            async for _ in sarvam_synthesize("hello", cancelled, language_code="en-IN"):
                pass

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert payload["target_language_code"] == "en-IN"


@pytest.mark.asyncio
async def test_cancel_flag_stops_streaming():
    """Setting cancelled[0] = True mid-stream stops chunk generation."""
    wav_b64 = _make_wav_b64(n_frames=44100)  # ~2s of audio, many chunks
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [wav_b64]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [False]
            chunks = []
            async for chunk in sarvam_synthesize("test", cancelled, language_code="hi-IN"):
                chunks.append(chunk)
                cancelled[0] = True  # cancel after first chunk

    assert len(chunks) == 1, "Expected exactly 1 chunk before cancel stopped iteration"


@pytest.mark.asyncio
async def test_missing_api_key_raises():
    """RuntimeError raised when SARVAM_API_KEY is empty."""
    import importlib
    import services.tts_sarvam as mod
    with patch.dict("os.environ", {"SARVAM_API_KEY": ""}):
        with patch.object(mod, "SARVAM_API_KEY", ""):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [False]
            with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):
                async for _ in sarvam_synthesize("test", cancelled):
                    pass


def test_pcm16_16k_to_ulaw_8k_produces_half_length():
    """Downsampled μ-law output is half the sample count of 16kHz input."""
    from services.tts_sarvam import _pcm16_16k_to_ulaw_8k
    # 1600 samples @ 16kHz = 100ms PCM16 → 800 samples @ 8kHz μ-law
    pcm = b"\x00\x00" * 1600
    ulaw = _pcm16_16k_to_ulaw_8k(pcm)
    # μ-law is 1 byte/sample → expect ~800 bytes (allow ±2 for resampling rounding)
    assert abs(len(ulaw) - 800) <= 2
