"""
Sarvam STT WebSocket client (saaras:v3).

Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe/ws

Twilio μ-law 8 kHz → PCM → small WAV chunks sent as JSON messages (required by API).
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import json
import logging
import os
import struct
from typing import Callable, Awaitable, Optional
from urllib.parse import urlencode

import websockets

from config.base_config import (
    SARVAM_API_KEY,
    SARVAM_STT_HIGH_VAD,
    SARVAM_STT_INTERRUPT_MIN_FRAMES,
    SARVAM_STT_MIN_SPEECH_FRAMES,
    SARVAM_STT_NEGATIVE_FRAMES_COUNT,
    SARVAM_STT_NEGATIVE_FRAMES_WINDOW,
    SARVAM_STT_VOLUME_THRESHOLD,
)

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, bool], Awaitable[None]]
SpeechStartedCallback = Callable[[], Awaitable[None]]

_LEGACY_SUBSCRIBE_PATH = "/speech-to-text-translate/subscribe"
_TWILIO_RATE = 8000
# 16 kHz is preferred by Sarvam; Twilio μ-law is 8 kHz and upsampled before send.
_SAMPLE_RATE = int(os.getenv("SARVAM_STT_SAMPLE_RATE", "16000"))
# ~250 ms of mono s16le at the connection sample rate
_SEND_BUFFER_BYTES = int(_SAMPLE_RATE * 0.25 * 2)
_MIN_UTTERANCE_WORDS = 1


def _pcm_to_wav(pcm: bytes, sample_rate: int = _SAMPLE_RATE) -> bytes:
    channels = 1
    bits = 16
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )
    return header + pcm


def build_sarvam_stt_ws_url() -> str:
    custom = os.getenv("SARVAM_STT_URL", "").strip()
    if custom and _LEGACY_SUBSCRIBE_PATH not in custom:
        return custom
    if custom and _LEGACY_SUBSCRIBE_PATH in custom:
        logger.warning("Deprecated Sarvam /subscribe URL — using /speech-to-text/ws")

    path = os.getenv("SARVAM_STT_WS_PATH", "/speech-to-text/ws")
    params = {
        "language-code": os.getenv("SARVAM_STT_LANGUAGE", "hi-IN"),
        "model": os.getenv("SARVAM_STT_MODEL", "saaras:v3"),
        "mode": os.getenv("SARVAM_STT_MODE", "codemix"),
        "sample_rate": str(_SAMPLE_RATE),
        "high_vad_sensitivity": "true" if SARVAM_STT_HIGH_VAD else "false",
        "vad_signals": "true",
        "min_speech_frames": str(SARVAM_STT_MIN_SPEECH_FRAMES),
        "interrupt_min_speech_frames": str(SARVAM_STT_INTERRUPT_MIN_FRAMES),
        "negative_frames_count": str(SARVAM_STT_NEGATIVE_FRAMES_COUNT),
        "negative_frames_window": str(SARVAM_STT_NEGATIVE_FRAMES_WINDOW),
        "start_speech_volume_threshold": str(SARVAM_STT_VOLUME_THRESHOLD),
    }
    return f"wss://api.sarvam.ai{path}?{urlencode(params)}"


class SarvamSTT:
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
        self._speech_ended = False  # END_SPEECH often arrives before type=data transcript
        self._pcm_buffer = bytearray()
        self._ratecv_state = None
        self._ready = asyncio.Event()
        self._chunks_sent = 0
        self.had_error = False

    async def connect(self) -> None:
        api_key = (os.getenv("SARVAM_API_KEY") or SARVAM_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured")

        url = build_sarvam_stt_ws_url()
        self._ws = await websockets.connect(
            url,
            additional_headers={"Api-Subscription-Key": api_key},
        )
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        self._ready.set()  # Sarvam does not send a handshake; don't block the greeting
        logger.info("Sarvam STT connected: %s", url.split("?")[0])

    async def send_audio(self, audio_chunk: bytes) -> None:
        if not self._ws or not audio_chunk or self.had_error:
            return
        try:
            pcm = audioop.ulaw2lin(audio_chunk, 2)
            if _SAMPLE_RATE != _TWILIO_RATE:
                pcm, self._ratecv_state = audioop.ratecv(
                    pcm, 2, 1, _TWILIO_RATE, _SAMPLE_RATE, self._ratecv_state
                )
            self._pcm_buffer.extend(pcm)
            if len(self._pcm_buffer) < _SEND_BUFFER_BYTES:
                return
            await self._flush_pcm_buffer()
        except Exception as exc:
            logger.warning("Sarvam STT send_audio: %s", exc)

    async def _flush_pcm_buffer(self) -> None:
        if not self._ws or not self._pcm_buffer:
            return
        pcm = bytes(self._pcm_buffer)
        self._pcm_buffer.clear()
        wav = _pcm_to_wav(pcm, _SAMPLE_RATE)
        msg = json.dumps({
            "audio": {
                "data": base64.b64encode(wav).decode("ascii"),
                "sample_rate": str(_SAMPLE_RATE),
                "encoding": "audio/wav",
            }
        })
        await self._ws.send(msg)
        self._chunks_sent += 1
        if self._chunks_sent == 1:
            logger.info("Sarvam STT: first audio chunk sent (%s Hz)", _SAMPLE_RATE)

    async def close(self) -> None:
        if self._pcm_buffer:
            try:
                await self._flush_pcm_buffer()
            except Exception:
                pass
        for task in (self._receive_task, self._keepalive_task):
            if task:
                task.cancel()
        if self._receive_task or self._keepalive_task:
            await asyncio.gather(
                *(t for t in (self._receive_task, self._keepalive_task) if t),
                return_exceptions=True,
            )
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "flush"}))
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._utterance_parts = []
        logger.info("Sarvam STT connection closed")

    async def _receive_loop(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue
                data = json.loads(message)
                if not self._ready.is_set():
                    self._ready.set()
                await self._handle_message(data)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Sarvam STT receive_loop ended: %s", exc)
        finally:
            self._ws = None

    async def _handle_message(self, data: dict) -> None:
        msg_type = data.get("type", "")
        payload = data.get("data")
        if not isinstance(payload, dict):
            payload = {}

        if msg_type == "error":
            self.had_error = True
            err = payload.get("message") or payload.get("error") or data
            logger.error("Sarvam STT error: %s", err)
            return

        if msg_type == "events":
            signal = (
                payload.get("signal_type")
                or payload.get("event_type")
            )
            if signal == "START_SPEECH":
                self._speech_ended = False
                if self._utterance_parts:
                    await self._flush_utterance()
                logger.info("Sarvam STT: START_SPEECH")
                if self._on_speech_started:
                    await self._on_speech_started()
            elif signal == "END_SPEECH":
                self._speech_ended = True
                logger.info("Sarvam STT: END_SPEECH")
                await self._flush_utterance()
            return

        if msg_type == "data":
            transcript = (
                payload.get("transcript")
                or payload.get("text")
                or ""
            ).strip()
            if transcript:
                self._utterance_parts.append(transcript)
                logger.info("Sarvam STT transcript chunk: %s", transcript)
            if self._speech_ended:
                await self._flush_utterance()
                self._speech_ended = False
            return

        if msg_type == "transcript":
            await self._handle_legacy_transcript(data)

    async def _handle_legacy_transcript(self, data: dict) -> None:
        transcript = (data.get("transcript") or "").strip()
        if data.get("is_final"):
            await self._emit_utterance(transcript)
        elif transcript:
            self._utterance_parts.append(transcript)

    async def _flush_utterance(self) -> None:
        if not self._utterance_parts:
            return
        full = " ".join(self._utterance_parts).strip()
        self._utterance_parts = []
        await self._emit_utterance(full)

    async def _emit_utterance(self, full: str) -> None:
        if not full or len(full.split()) < _MIN_UTTERANCE_WORDS:
            return
        logger.info("Sarvam STT utterance complete: %s", full)
        await self._on_transcript(full, True)

    async def _keepalive_loop(self) -> None:
        try:
            while self._ws:
                await asyncio.sleep(8)
                if self._ws:
                    await self._ws.ping()
        except asyncio.CancelledError:
            pass
