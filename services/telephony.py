"""
Telephony client — abstracts Twilio (prototype) and Telnyx (production).

Responsibilities:
  1. Receive raw μ-law 8kHz audio from the carrier WebSocket
  2. Forward audio to the STT pipeline
  3. Send TTS audio chunks back to the carrier
  4. Execute call-control actions: hold, transfer (SIP / PSTN)

Set TELEPHONY_PROVIDER=twilio or telnyx in your .env.
"""

import asyncio
import base64
import json
import logging
from typing import Callable, Awaitable, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)

AudioCallback = Callable[[bytes], Awaitable[None]]


class TelephonySession:
    """
    Wraps a single carrier WebSocket connection for one call.

    The carrier (Twilio / Telnyx) sends JSON frames over the WebSocket.
    We decode the audio payload and forward it; we encode TTS audio and
    send it back in the same frame format.
    """

    def __init__(
        self,
        websocket: WebSocket,
        on_audio: AudioCallback,
        provider: str = "twilio",
    ) -> None:
        self._ws = websocket
        self._on_audio = on_audio
        self._provider = provider.lower()
        self._stream_sid: Optional[str] = None
        self._call_sid: Optional[str] = None
        self._active = True
        self.stream_ready = asyncio.Event()  # fired when "start" frame received

    @property
    def call_sid(self) -> Optional[str]:
        return self._call_sid

    @property
    def stream_sid(self) -> Optional[str]:
        return self._stream_sid

    async def receive_loop(self) -> None:
        """
        Continuously read frames from the carrier WebSocket and forward
        audio to the STT pipeline until the call ends.
        """
        try:
            while self._active:
                raw = await self._ws.receive_text()
                await self._dispatch(raw)
        except Exception as exc:
            logger.info("TelephonySession receive_loop ended: %s", exc)
        finally:
            self._active = False

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send a μ-law audio chunk back to the carrier (plays to caller)."""
        if not self._active:
            return
        try:
            if self._provider == "twilio":
                await self._twilio_send(audio_chunk)
            else:
                await self._telnyx_send(audio_chunk)
        except Exception as exc:
            logger.warning("send_audio error: %s", exc)

    async def send_silence(self, duration_ms: int = 100) -> None:
        """Send μ-law silence to avoid dead air during barge-in cancellation."""
        # μ-law silence is 0xFF (not 0x00 — zeros are loud noise in G.711)
        silence = bytes([0xFF] * int(duration_ms * 8))  # 8 bytes/ms at 8kHz
        await self.send_audio(silence)

    async def clear_playback_buffer(self) -> None:
        """Best-effort carrier buffer clear for immediate barge-in."""
        if not self._active:
            return
        try:
            if self._provider == "twilio" and self._stream_sid:
                await self._ws.send_text(json.dumps({
                    "event": "clear",
                    "streamSid": self._stream_sid,
                }))
        except Exception as exc:
            logger.debug("clear_playback_buffer ignored: %s", exc)

    async def transfer(self, destination: str) -> None:
        """
        Transfer the call to a PSTN number or SIP URI.
        Twilio: REST API TwiML redirect.
        Telnyx: Call Control transfer action.
        """
        logger.info("Transferring call %s → %s", self._call_sid, destination)
        if self._provider == "twilio":
            await self._twilio_transfer(destination)
        else:
            await self._telnyx_transfer(destination)
        self._active = False

    async def hangup(self) -> None:
        """End the call from our side."""
        logger.info("Hanging up call %s", self._call_sid)
        if self._provider == "twilio":
            await self._twilio_hangup()
        else:
            await self._telnyx_hangup()
        self._active = False

    # ── Internal dispatch ─────────────────────────────────────────────────────

    async def _dispatch(self, raw: str) -> None:
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            return

        event = frame.get("event") or frame.get("type", "")

        if event == "start":
            self._stream_sid = (
                frame.get("streamSid")
                or frame.get("stream_id", "unknown")
            )
            self._call_sid = (
                (frame.get("start") or {}).get("callSid")
                or frame.get("call_control_id", "unknown")
            )
            logger.info(
                "Call started: call_sid=%s stream_sid=%s",
                self._call_sid,
                self._stream_sid,
            )
            self.stream_ready.set()  # unblock pipeline greeting

        elif event == "media":
            # Twilio: frame["media"]["payload"] — base64 μ-law
            # Telnyx:  frame["payload"]["audio"] — base64 μ-law
            payload = (
                (frame.get("media") or {}).get("payload")
                or (frame.get("payload") or {}).get("audio", "")
            )
            if payload:
                audio = base64.b64decode(payload)
                await self._on_audio(audio)

        elif event in ("stop", "call.hangup"):
            logger.info("Call ended: %s", self._call_sid)
            self._active = False

    # ── Twilio helpers ────────────────────────────────────────────────────────

    async def _twilio_send(self, audio: bytes) -> None:
        payload = base64.b64encode(audio).decode()
        frame = {
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {"payload": payload},
        }
        await self._ws.send_text(json.dumps(frame))

    async def _twilio_transfer(self, destination: str) -> None:
        """
        Transfer via Twilio REST API — redirects the live call to a TwiML
        endpoint that dials the destination number.
        """
        import httpx
        from config.base_config import (
            TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, PUBLIC_URL
        )

        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            logger.error("Twilio credentials missing — cannot transfer call")
            return

        # The redirect URL returns <Dial> TwiML with the destination number
        from urllib.parse import quote
        redirect_url = f"{PUBLIC_URL}/transfer-twiml?to={quote(destination)}"

        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{TWILIO_ACCOUNT_SID}/Calls/{self._call_sid}.json"
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                    data={"Url": redirect_url, "Method": "POST"},
                )
                if resp.status_code in (200, 201):
                    logger.info("Twilio transfer initiated → %s", destination)
                else:
                    logger.error(
                        "Twilio transfer failed: %s %s", resp.status_code, resp.text
                    )
        except Exception as exc:
            logger.error("Twilio transfer error: %s", exc)

    async def _twilio_hangup(self) -> None:
        frame = {
            "event": "mark",
            "streamSid": self._stream_sid,
            "mark": {"name": "hangup"},
        }
        await self._ws.send_text(json.dumps(frame))

    # ── Telnyx helpers ────────────────────────────────────────────────────────

    async def _telnyx_send(self, audio: bytes) -> None:
        payload = base64.b64encode(audio).decode()
        frame = {
            "event": "media",
            "stream_id": self._stream_sid,
            "payload": {"audio": payload},
        }
        await self._ws.send_text(json.dumps(frame))

    async def _telnyx_transfer(self, destination: str) -> None:
        import httpx
        from config.base_config import TELNYX_API_KEY
        url = f"https://api.telnyx.com/v2/calls/{self._call_sid}/actions/transfer"
        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                headers={"Authorization": f"Bearer {TELNYX_API_KEY}"},
                json={"to": destination},
            )

    async def _telnyx_hangup(self) -> None:
        import httpx
        from config.base_config import TELNYX_API_KEY
        url = f"https://api.telnyx.com/v2/calls/{self._call_sid}/actions/hangup"
        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                headers={"Authorization": f"Bearer {TELNYX_API_KEY}"},
                json={},
            )
