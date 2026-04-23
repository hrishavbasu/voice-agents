"""
Azure Cognitive Services TTS — Indian English female voice.

Voice: en-IN-NeerjaNeural (warm, professional Indian English)
       hi-IN-SwaraNeural  (Hindi, optional)

Free tier: 0.5M chars/month (Neural TTS)
Output:    audio-8khz-8bit-mono-mulaw  — native Twilio format, no conversion needed

Set in .env:
    AZURE_TTS_KEY=<your-key>
    AZURE_TTS_REGION=eastus          # or centralindia, southeastasia, etc.
    AZURE_TTS_VOICE=en-IN-NeerjaNeural
"""

import logging
from typing import AsyncIterator

import httpx

from config.base_config import AZURE_TTS_KEY, AZURE_TTS_REGION, AZURE_TTS_VOICE

logger = logging.getLogger(__name__)


def _ssml(text: str, voice: str) -> str:
    """
    Build SSML with natural prosody settings to avoid robotic tone.

    - style="customerservice": warm, helpful register (supported by SwaraNeural)
    - rate="0.95": slightly slower than default — more natural on phone
    - pitch="+1st": tiny lift keeps voice from sounding flat/monotone
    - styledegree="1.5": lean into the style without overdoing it
    """
    lang = voice[:5]  # "hi-IN"
    text = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )
    return (
        "<speak version='1.0' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='https://www.w3.org/2001/mstts' "
        f"xml:lang='{lang}'>"
        f"<voice name='{voice}'>"
        "<mstts:express-as style='customerservice' styledegree='1.5'>"
        "<prosody rate='0.95' pitch='+1st'>"
        f"{text}"
        "</prosody>"
        "</mstts:express-as>"
        "</voice>"
        "</speak>"
    )


async def azure_synthesize(
    text: str,
    cancelled_flag: list,
    voice: str | None = None,
) -> AsyncIterator[bytes]:
    """
    Stream μ-law 8kHz audio from Azure TTS.

    hi-IN-SwaraNeural handles both Hindi (Devanagari) and English/Hinglish
    natively — no voice switching needed.

    Args:
        text:           Text to synthesise.
        cancelled_flag: Mutable list [False]; flip to [True] to abort mid-stream.
        voice:          Override voice name (defaults to AZURE_TTS_VOICE env var).

    Yields:
        Raw μ-law 8kHz audio bytes compatible with Twilio Media Streams.
    """
    voice_name = voice or AZURE_TTS_VOICE or "hi-IN-SwaraNeural"
    url = (
        f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com"
        "/cognitiveservices/v1"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TTS_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "raw-8khz-8bit-mono-mulaw",
        "User-Agent": "CustomerSupportAI/1.0",
    }
    ssml = _ssml(text, voice_name)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            async with client.stream(
                "POST", url, headers=headers, content=ssml.encode()
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    logger.error(
                        "Azure TTS HTTP %s. Body: %s",
                        response.status_code,
                        response.text[:300],
                    )
                    raise httpx.HTTPStatusError(
                        f"Azure TTS {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if cancelled_flag[0]:
                        return
                    if chunk:
                        yield chunk
    except httpx.HTTPStatusError:
        raise
    except Exception as exc:
        logger.error("Azure TTS error: %s", exc)
        raise
