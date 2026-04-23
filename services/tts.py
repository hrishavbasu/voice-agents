"""
TTS service — supports ElevenLabs (Indian voice, primary), Deepgram Aura
(prototype/free), and Cartesia Sonic (production).

Selected by TTS_PROVIDER env var:
  "elevenlabs"  — Indian English voice via ElevenLabs (recommended)
  "deepgram"    — Deepgram Aura free tier
  "cartesia"    — Cartesia Sonic production

Usage:
    tts = TTSService()
    async for audio_chunk in tts.synthesize("Hello there!"):
        await telephony.send_audio(audio_chunk)

    # Cancel mid-stream (barge-in):
    tts.cancel()
"""

import asyncio
import logging
from typing import AsyncIterator, Optional

import httpx

from config.base_config import (
    DEEPGRAM_API_KEY,
    CARTESIA_API_KEY,
    TTS_PROVIDER,
    TTS_VOICE_ID,
    CARTESIA_VOICE_ID,
    TTS_SAMPLE_RATE,
    TTS_ENCODING,
    AZURE_TTS_VOICE,
)

logger = logging.getLogger(__name__)

FILLER_TEXTS = [
    "Let me check that for you.",
    "One moment please.",
    "Sure, I can help with that.",
    "Give me just a second.",
]


class TTSService:
    """Streaming TTS with cancellation support for barge-in."""

    def __init__(self) -> None:
        self._cancelled = False
        self._provider = TTS_PROVIDER.lower()
        # Shared mutable flag passed into elevenlabs_synthesize for mid-stream cancel
        self._cancel_flag: list = [False]

    def cancel(self) -> None:
        """Signal that the current synthesis stream should be abandoned."""
        self._cancelled = True
        self._cancel_flag[0] = True
        logger.debug("TTS: cancel requested")

    def reset(self) -> None:
        """Clear the cancel flag before starting a new utterance."""
        self._cancelled = False
        self._cancel_flag = [False]

    async def synthesize(self, text: str, voice_id: str | None = None) -> AsyncIterator[bytes]:
        """Stream audio chunks for *text*.  Stops early if cancel() was called.

        ElevenLabs / Azure failures automatically fall back to Deepgram
        so the caller always hears audio.
        """
        self.reset()
        if self._provider == "elevenlabs":
            try:
                async for chunk in self._elevenlabs(text, voice_id):
                    if self._cancelled:
                        return
                    yield chunk
            except Exception as exc:
                logger.warning(
                    "ElevenLabs failed (%s) — falling back to Deepgram TTS", exc
                )
                async for chunk in self._deepgram(text):
                    if self._cancelled:
                        return
                    yield chunk
        elif self._provider == "azure":
            try:
                async for chunk in self._azure(text):
                    if self._cancelled:
                        return
                    yield chunk
            except Exception as exc:
                logger.warning(
                    "Azure TTS failed (%s) — falling back to Deepgram TTS", exc
                )
                async for chunk in self._deepgram(text):
                    if self._cancelled:
                        return
                    yield chunk
        elif self._provider == "cartesia":
            async for chunk in self._cartesia(text):
                if self._cancelled:
                    return
                yield chunk
        else:
            async for chunk in self._deepgram(text):
                if self._cancelled:
                    return
                yield chunk

    # ── Azure TTS (Indian English — en-IN-NeerjaNeural) ──────────────────────

    async def _azure(self, text: str) -> AsyncIterator[bytes]:
        from services.tts_azure import azure_synthesize
        async for chunk in azure_synthesize(text, self._cancel_flag):
            yield chunk

    # ── ElevenLabs (Indian English — primary) ─────────────────────────────────

    async def _elevenlabs(self, text: str, voice_id: str | None = None) -> AsyncIterator[bytes]:
        from services.tts_elevenlabs import elevenlabs_synthesize
        async for chunk in elevenlabs_synthesize(text, self._cancel_flag, voice_id=voice_id):
            yield chunk

    # ── Deepgram Aura (REST streaming, free tier) ─────────────────────────────

    async def _deepgram(self, text: str) -> AsyncIterator[bytes]:
        url = "https://api.deepgram.com/v1/speak"
        params = {
            "model": TTS_VOICE_ID or "aura-asteria-en",
            "encoding": TTS_ENCODING,
            "sample_rate": TTS_SAMPLE_RATE,
            "container": "none",
        }
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream(
                    "POST", url, headers=headers, params=params, json=payload
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=1024):
                        if self._cancelled:
                            return
                        if chunk:
                            yield chunk
        except httpx.HTTPStatusError as exc:
            logger.error("Deepgram TTS HTTP error %s: %s", exc.response.status_code, exc)
        except Exception as exc:
            logger.error("Deepgram TTS error: %s", exc)

    # ── Cartesia Sonic (WebSocket streaming, production) ──────────────────────

    async def _cartesia(self, text: str) -> AsyncIterator[bytes]:
        """
        Cartesia streaming TTS via their REST bytes endpoint.
        Switch to their WebSocket for lowest latency in production.
        """
        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "X-API-Key": CARTESIA_API_KEY,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": "sonic-english",
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_mulaw",
                "sample_rate": TTS_SAMPLE_RATE,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=1024):
                        if self._cancelled:
                            return
                        if chunk:
                            yield chunk
        except httpx.HTTPStatusError as exc:
            logger.error("Cartesia TTS HTTP error %s: %s", exc.response.status_code, exc)
        except Exception as exc:
            logger.error("Cartesia TTS error: %s", exc)
