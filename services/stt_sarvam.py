"""
Sarvam STT WebSocket client.

Same interface as DeepgramSTT — connect(), send_audio(), close().
Barge-in fires via on_speech_started() callback when Sarvam emits
a speech_started event.
"""
import asyncio
import json
import logging
import os
from typing import Callable, Awaitable, Optional

import websockets

from config.base_config import (
    SARVAM_API_KEY,
    SARVAM_STT_URL,
    SARVAM_STT_INTERRUPT_MIN_FRAMES,
    SARVAM_STT_MIN_SPEECH_FRAMES,
    SARVAM_STT_VOLUME_THRESHOLD,
    SARVAM_STT_HIGH_VAD,
    SARVAM_STT_NEGATIVE_FRAMES_COUNT,
    SARVAM_STT_NEGATIVE_FRAMES_WINDOW,
)

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, bool], Awaitable[None]]
SpeechStartedCallback = Callable[[], Awaitable[None]]


class SarvamSTT:
    """Streaming STT via Sarvam WebSocket with native Indian-acoustic VAD barge-in."""

    def __init__(
        self,
        on_transcript: TranscriptCallback,
        on_speech_started: Optional[SpeechStartedCallback] = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._ws = None
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._utterance_parts: list[str] = []

    async def connect(self) -> None:
        """Open WebSocket connection and send barge-in config."""
        api_key = (os.getenv("SARVAM_API_KEY") or SARVAM_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured")

        self._ws = await websockets.connect(
            SARVAM_STT_URL,
            additional_headers={"api-subscription-key": api_key},
        )

        config_msg = {
            "type": "config",
            "interrupt_min_speech_frames": SARVAM_STT_INTERRUPT_MIN_FRAMES,
            "min_speech_frames": SARVAM_STT_MIN_SPEECH_FRAMES,
            "start_speech_volume_threshold": SARVAM_STT_VOLUME_THRESHOLD,
            "high_vad_sensitivity": SARVAM_STT_HIGH_VAD,
            "negative_frames_count": SARVAM_STT_NEGATIVE_FRAMES_COUNT,
            "negative_frames_window": SARVAM_STT_NEGATIVE_FRAMES_WINDOW,
        }
        await self._ws.send(json.dumps(config_msg))

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info(
            "Sarvam STT connected (interrupt_min_frames=%d, vad_high=%s)",
            SARVAM_STT_INTERRUPT_MIN_FRAMES,
            SARVAM_STT_HIGH_VAD,
        )

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Forward a raw μ-law 8kHz audio chunk to Sarvam."""
        if self._ws:
            try:
                await self._ws.send(audio_chunk)
            except Exception as exc:
                logger.debug("Sarvam STT send_audio error: %s", exc)

    async def close(self) -> None:
        """Gracefully shut down the WebSocket and background tasks."""
        tasks = []
        if self._receive_task:
            self._receive_task.cancel()
            tasks.append(self._receive_task)
        if self._keepalive_task:
            self._keepalive_task.cancel()
            tasks.append(self._keepalive_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Sarvam STT connection closed")

    async def _receive_loop(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue  # Sarvam sends JSON text frames only
                try:
                    data = json.loads(message)
                    await self._handle_event(data)
                except json.JSONDecodeError as exc:
                    logger.warning("Sarvam STT parse error: %s", exc)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Sarvam STT receive_loop ended: %s", exc)
        finally:
            self._ws = None  # allows _keepalive_loop to exit cleanly

    async def _handle_event(self, data: dict) -> None:
        event_type = data.get("type", "")

        if event_type == "speech_started":
            logger.debug("Sarvam STT: speech_started — barge-in trigger")
            self._utterance_parts = []
            if self._on_speech_started:
                await self._on_speech_started()

        elif event_type == "transcript":
            transcript = (data.get("transcript") or "").strip()
            is_final = bool(data.get("is_final", False))

            if not transcript:
                return

            if is_final:
                if self._utterance_parts:
                    full = " ".join(self._utterance_parts) + " " + transcript
                else:
                    full = transcript
                self._utterance_parts = []

                if len(full.split()) < 2:
                    logger.debug("Sarvam STT: discarding short utterance: %r", full)
                    return

                logger.info("Sarvam STT utterance complete: %s", full)
                await self._on_transcript(full, True)
            else:
                self._utterance_parts.append(transcript)
                logger.debug("Sarvam STT interim: %s", transcript)

    async def _keepalive_loop(self) -> None:
        try:
            while self._ws:
                await asyncio.sleep(8)
                if self._ws:
                    await self._ws.ping()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Sarvam STT keepalive stopped: %s", exc)
