"""
Sarvam Bulbul TTS provider.

Synthesizes speech via Sarvam's /text-to-speech REST API and normalizes
the response to raw μ-law 8kHz bytes (Twilio's native G.711 format).

Output format: ulaw_8000 — raw G.711 μ-law, 8kHz, mono, 1 byte/sample.

Usage:
    cancelled = [False]
    async for chunk in sarvam_synthesize("नमस्ते", cancelled, language_code="hi-IN"):
        await telephony.send_audio(chunk)

    # Cancel mid-stream (barge-in):
    cancelled[0] = True
"""

import asyncio
import audioop
import base64
import io
import logging
import os
import wave
from collections import OrderedDict
from typing import AsyncIterator, Optional

import httpx

from config.base_config import (
    SARVAM_API_KEY,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
    SARVAM_TTS_PACE,
    SARVAM_TTS_PITCH,
    SARVAM_TTS_LOUDNESS,
)

logger = logging.getLogger(__name__)

_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_AUDIO_CACHE_MAX_ITEMS = 128

# LRU cache: key = (text, language_code, model, speaker) → raw μ-law bytes
_audio_cache: "OrderedDict[tuple[str, str, str, str], bytes]" = OrderedDict()
_http_client: Optional[httpx.AsyncClient] = None


def _cache_get(key: tuple) -> Optional[bytes]:
    data = _audio_cache.get(key)
    if data is not None:
        _audio_cache.move_to_end(key)
    return data


def _cache_set(key: tuple, audio_bytes: bytes) -> None:
    _audio_cache[key] = audio_bytes
    _audio_cache.move_to_end(key)
    while len(_audio_cache) > _AUDIO_CACHE_MAX_ITEMS:
        _audio_cache.popitem(last=False)


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=45,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client


def _decode_first_wav(audio_b64_list: list) -> bytes:
    if not audio_b64_list:
        raise ValueError("Sarvam TTS returned no audio")
    return base64.b64decode(audio_b64_list[0])


def _wav_to_raw_pcm16_mono_16k(wav_bytes: bytes) -> bytes:
    """Convert WAV (any sample rate, mono/stereo) → raw PCM16 mono 16kHz."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if channels == 2:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
    if sampwidth != 2:
        frames = audioop.lin2lin(frames, sampwidth, 2)
        sampwidth = 2
    if framerate != 16000:
        frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 16000, None)
    return frames


def _pcm16_16k_to_ulaw_8k(pcm16: bytes) -> bytes:
    """Downsample PCM16 16kHz → μ-law 8kHz (Twilio native format)."""
    downsampled, _ = audioop.ratecv(pcm16, 2, 1, 16000, 8000, None)
    return audioop.lin2ulaw(downsampled, 2)


async def sarvam_synthesize(
    text: str,
    cancelled_flag: list,
    language_code: str = "hi-IN",
    pitch_override: Optional[float] = None,
) -> AsyncIterator[bytes]:
    """
    Synthesize speech with Sarvam Bulbul and yield raw μ-law 8kHz chunks.

    Args:
        text: Text to synthesize (Hindi, English, or Hinglish).
        cancelled_flag: Single-element list [False]; set to [True] to stop mid-stream.
        language_code: "hi-IN" for Hindi/Hinglish, "en-IN" for Indian English.
        pitch_override: Optional pitch value (−1.0–1.0). Pass 0.05 for questions.
            Defaults to SARVAM_TTS_PITCH config value when None.
    """
    api_key = (os.getenv("SARVAM_API_KEY") or SARVAM_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not configured. Add it to .env: SARVAM_API_KEY=your_key"
        )

    _pitch = pitch_override if pitch_override is not None else SARVAM_TTS_PITCH
    cache_key = (text, language_code, SARVAM_TTS_MODEL, SARVAM_TTS_SPEAKER, _pitch)
    audio_bytes = _cache_get(cache_key)

    if audio_bytes is None:
        # Check cancel flag before starting the network request — handles the
        # common case where barge-in fires in the gap between two sentences.
        if cancelled_flag[0]:
            return
        payload = {
            "text": text,
            "target_language_code": language_code,
            "model": SARVAM_TTS_MODEL,
            "speaker": SARVAM_TTS_SPEAKER,
            "speech_sample_rate": 8000,
            "pace": SARVAM_TTS_PACE,
            "pitch": _pitch,
            "loudness": SARVAM_TTS_LOUDNESS,
        }
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }

        client = await _get_http_client()

        # Wrap the HTTP request in a task so we can cancel it if barge-in fires
        # during the network round-trip (Sarvam returns the full WAV at once —
        # there is no streaming, so this is the only cancellation window).
        fetch_task = asyncio.create_task(
            client.post(_SARVAM_TTS_URL, headers=headers, json=payload)
        )
        while not fetch_task.done():
            if cancelled_flag[0]:
                fetch_task.cancel()
                return
            await asyncio.sleep(0.05)
        response = await fetch_task

        # Some accounts use Bearer auth — retry once if subscription key rejected
        if response.status_code == 403:
            retry_headers = {**headers, "Authorization": f"Bearer {api_key}"}
            response = await client.post(_SARVAM_TTS_URL, headers=retry_headers, json=payload)

        if response.status_code >= 400:
            logger.error(
                "Sarvam TTS HTTP %s: %s", response.status_code, response.text[:300]
            )
            response.raise_for_status()

        try:
            data = response.json()
        except Exception as exc:
            logger.error("Sarvam TTS: failed to parse JSON response: %s", exc)
            raise

        audios = data.get("audios")
        if not audios or not isinstance(audios, list):
            raise ValueError(
                f"Sarvam TTS: unexpected response structure — 'audios' field missing or empty. "
                f"Response keys: {list(data.keys())}"
            )

        try:
            wav_bytes = _decode_first_wav(audios)
            pcm16 = _wav_to_raw_pcm16_mono_16k(wav_bytes)
            audio_bytes = _pcm16_16k_to_ulaw_8k(pcm16)
        except Exception as exc:
            logger.error("Sarvam TTS: audio conversion failed: %s", exc)
            raise

        _cache_set(cache_key, audio_bytes)

    chunk_size = 1024
    for i in range(0, len(audio_bytes), chunk_size):
        if cancelled_flag[0]:
            return
        yield audio_bytes[i: i + chunk_size]
