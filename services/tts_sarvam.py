"""
Sarvam Bulbul TTS provider.

Sarvam returns base64-encoded WAV audio. This module normalizes it to raw bytes
expected by the app:
  - Browser:  pcm_16000 (raw PCM16 mono 16kHz)
  - Telephony: ulaw_8000 (raw G.711 mu-law 8kHz)
"""

import audioop
import base64
import io
import logging
import os
import wave
from collections import OrderedDict
from typing import AsyncIterator

import httpx

from config.base_config import SARVAM_API_KEY, SARVAM_TTS_MODEL, SARVAM_TTS_SPEAKER

logger = logging.getLogger(__name__)

_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_AUDIO_CACHE_MAX_ITEMS = 128
_audio_cache: "OrderedDict[tuple[str, str, str, str, str], bytes]" = OrderedDict()
_http_client: httpx.AsyncClient | None = None


def _cache_get(key: tuple[str, str, str, str, str]) -> bytes | None:
    data = _audio_cache.get(key)
    if data is not None:
        _audio_cache.move_to_end(key)
    return data


def _cache_set(key: tuple[str, str, str, str, str], audio_bytes: bytes) -> None:
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


def _language_code_for_text(text: str) -> str:
    """Infer target language for Bulbul normalization."""
    if any("\u0900" <= ch <= "\u097f" for ch in text):
        return "hi-IN"
    return "en-IN"


def _decode_first_wav(audio_b64_list: list[str]) -> bytes:
    if not audio_b64_list:
        raise ValueError("Sarvam TTS returned no audio")
    return base64.b64decode(audio_b64_list[0])


def _wav_to_raw_pcm16_mono_16k(wav_bytes: bytes) -> bytes:
    """Convert WAV to raw PCM16 mono 16kHz."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if channels == 2:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
        channels = 1
    if channels != 1:
        raise ValueError(f"Unsupported channel count from Sarvam WAV: {channels}")
    if sampwidth != 2:
        # Normalize to 16-bit PCM expected by downstream paths.
        frames = audioop.lin2lin(frames, sampwidth, 2)
        sampwidth = 2
    if framerate != 16000:
        frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 16000, None)
    return frames


def _pcm16_16k_to_ulaw_8k(pcm16: bytes) -> bytes:
    downsampled, _ = audioop.ratecv(pcm16, 2, 1, 16000, 8000, None)
    return audioop.lin2ulaw(downsampled, 2)


async def sarvam_synthesize(
    text: str,
    cancelled_flag: list,
    output_format: str = "ulaw_8000",
) -> AsyncIterator[bytes]:
    """
    Synthesize speech with Sarvam Bulbul and yield normalized raw audio chunks.
    """
    # Resolve key dynamically so runtime env updates are honored without import-time staleness.
    sarvam_key = (os.getenv("SARVAM_API_KEY") or SARVAM_API_KEY or "").strip()
    if not sarvam_key:
        raise RuntimeError("SARVAM_API_KEY is not configured")

    sample_rate = 16000 if output_format == "pcm_16000" else 8000
    payload = {
        "text": text,
        "target_language_code": _language_code_for_text(text),
        "model": SARVAM_TTS_MODEL,
        "speaker": SARVAM_TTS_SPEAKER,
        "speech_sample_rate": sample_rate,
    }
    headers = {
        "api-subscription-key": sarvam_key,
        "Authorization": f"Bearer {sarvam_key}",
        "Content-Type": "application/json",
    }

    lang = payload["target_language_code"]
    cache_key = (text, output_format, lang, SARVAM_TTS_MODEL, SARVAM_TTS_SPEAKER)
    audio_bytes = _cache_get(cache_key)
    if audio_bytes is None:
        client = await _get_http_client()
        response = await client.post(_SARVAM_TTS_URL, headers=headers, json=payload)
        # Some accounts intermittently reject one auth style; retry once with api-key header only.
        if response.status_code == 403:
            retry_headers = {
                "api-subscription-key": sarvam_key,
                "Content-Type": "application/json",
            }
            response = await client.post(_SARVAM_TTS_URL, headers=retry_headers, json=payload)
        if response.status_code >= 400:
            logger.error(
                "Sarvam TTS HTTP %s: %s",
                response.status_code,
                response.text[:300],
            )
            response.raise_for_status()
        data = response.json()

        wav_bytes = _decode_first_wav(data.get("audios", []))
        pcm16 = _wav_to_raw_pcm16_mono_16k(wav_bytes)
        if output_format == "pcm_16000":
            audio_bytes = pcm16
        else:
            audio_bytes = _pcm16_16k_to_ulaw_8k(pcm16)
        _cache_set(cache_key, audio_bytes)

    chunk_size = 1024
    for i in range(0, len(audio_bytes), chunk_size):
        if cancelled_flag[0]:
            return
        yield audio_bytes[i : i + chunk_size]
