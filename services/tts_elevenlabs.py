"""
ElevenLabs TTS provider — Indian English / Hindi voice for Apollo clinic agent.

Uses ElevenLabs streaming API with ulaw_8000 output (native Twilio format).
No audio conversion needed.

Recommended voices:
  Indian English: Alekhya  (m28sDRnudtExG3WLAufB) — warm, professional, soothing
  Hindi:          Mahi     (OwA6IqdLakQOd19pSLOn) — voice-bot optimised, grounded

Set in .env:
  ELEVENLABS_VOICE_ID=m28sDRnudtExG3WLAufB       (Indian English, default)
  ELEVENLABS_VOICE_ID_HINDI=OwA6IqdLakQOd19pSLOn  (Hindi, optional)
"""

import logging
from typing import AsyncIterator

import httpx

from config.base_config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
from services.pronunciation_dict import get_pronunciation_dict_locators

logger = logging.getLogger(__name__)

_BASE = "https://api.elevenlabs.io/v1"


async def elevenlabs_synthesize(
    text: str,
    cancelled_flag: list,
    voice_id: str | None = None,
    output_format: str = "ulaw_8000",
) -> AsyncIterator[bytes]:
    """
    Stream audio from ElevenLabs.

    Args:
        text:           Text to synthesise.
        cancelled_flag: Mutable list [False]; flip to [True] to abort mid-stream.
        voice_id:       Override voice (defaults to ELEVENLABS_VOICE_ID env var).
        output_format:  ElevenLabs output format.
                        "ulaw_8000"   — Twilio μ-law 8kHz (default, telephony)
                        "pcm_16000"   — raw PCM16 16kHz (browser AudioContext)
                        "mp3_44100_64"— MP3 44.1kHz 64kbps (browser Audio element)

    Yields:
        Raw audio bytes in the requested format.
    """
    vid = voice_id or ELEVENLABS_VOICE_ID or "m28sDRnudtExG3WLAufB"
    url = f"{_BASE}/text-to-speech/{vid}/stream"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    params = {"output_format": output_format}
    payload = {
        "text": text,
        # eleven_multilingual_v2 — required for AI-generated voices in Hindi; turbo_v2_5 can
        # produce near-silent audio when used with generated (non-cloned) voices.
        "model_id": "eleven_multilingual_v2",
        # Tuned for natural conversation — lower stability = more expressive,
        # style > 0 adds warmth, similarity_boost preserves voice character.
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.82,
            "style": 0.35,
            "use_speaker_boost": True,
        },
    }

    # Attach pronunciation dictionary if configured — overrides default G2P for
    # proper nouns like "Navi" that Tripti mispronounces in Latin script.
    locators = get_pronunciation_dict_locators()
    if locators:
        payload["pronunciation_dictionary_locators"] = locators

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST", url, headers=headers, params=params, json=payload
            ) as response:
                # Read status before streaming — avoids ResponseNotRead on error
                if response.status_code >= 400:
                    await response.aread()
                    body = response.text[:300]
                    logger.error(
                        "ElevenLabs TTS HTTP %s — check plan/voice ID. Body: %s",
                        response.status_code,
                        body,
                    )
                    raise httpx.HTTPStatusError(
                        f"ElevenLabs {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if cancelled_flag[0]:
                        return
                    if chunk:
                        yield chunk
    except httpx.HTTPStatusError:
        raise  # let TTSService catch this and fall back to Deepgram
    except Exception as exc:
        logger.error("ElevenLabs TTS error: %s", exc)
        raise  # propagate so fallback triggers
