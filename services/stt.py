"""
Deepgram streaming STT wrapper.

Usage:
    stt = DeepgramSTT(on_transcript=my_callback, on_speech_started=barge_in_cb)
    await stt.connect()
    await stt.send_audio(pcm_chunk)
    await stt.close()

The on_transcript callback receives (transcript: str, is_final: bool).
The on_speech_started callback receives no arguments — triggers barge-in.
"""

import asyncio
import logging
from typing import Callable, Awaitable, Optional

from deepgram import (
    DeepgramClient,
    LiveOptions,
    LiveTranscriptionEvents,
)

from config.base_config import (
    DEEPGRAM_API_KEY,
    STT_MODEL,
    STT_LANGUAGE,
    STT_SMART_FORMAT,
    STT_INTERIM_RESULTS,
    STT_ENDPOINTING_MS,
    STT_UTTERANCE_END_MS,
    STT_MIN_WORDS,
    STT_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, bool], Awaitable[None]]
SpeechStartedCallback = Callable[[], Awaitable[None]]


class DeepgramSTT:
    """Streaming STT via Deepgram Live API (Nova-3)."""

    def __init__(
        self,
        on_transcript: TranscriptCallback,
        on_speech_started: Optional[SpeechStartedCallback] = None,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._loop = asyncio.get_event_loop()
        self._encoding = encoding
        self._sample_rate = sample_rate

        # Keepalive option in client config can cause 400 handshake failures on
        # some Deepgram accounts. Use default client and explicit app-level
        # keep_alive() loop below.
        self._client = DeepgramClient(DEEPGRAM_API_KEY)
        self._connection = None

        # Accumulate is_final chunks until speech_final — prevents triggering
        # the LLM on mid-utterance fragments (the root cause of parallel responses)
        self._utterance_parts: list[str] = []
        self._primary_speaker: Optional[int] = None  # locked on first speech_final

    async def connect(self) -> None:
        """Open the WebSocket connection and register event handlers."""
        self._connection = self._client.listen.asyncwebsocket.v("1")

        self._connection.on(
            LiveTranscriptionEvents.Transcript, self._handle_transcript
        )
        self._connection.on(
            LiveTranscriptionEvents.SpeechStarted, self._handle_speech_started
        )
        self._connection.on(
            LiveTranscriptionEvents.Error, self._handle_error
        )

        # Some Deepgram accounts reject specific realtime option combinations.
        # Try progressively simpler profiles before failing the call.
        attempts = [
            # Fastest/most reliable profile first for Twilio real-time calls.
            {
                "label": "minimal-auto-lang",
                "kwargs": {
                    "model": "nova-2",
                    "encoding": self._encoding,
                    "sample_rate": self._sample_rate,
                    "channels": 1,
                },
            },
            {
                "label": "minimal-hi",
                "kwargs": {
                    "model": "nova-2",
                    "language": "hi",
                    "encoding": self._encoding,
                    "sample_rate": self._sample_rate,
                    "channels": 1,
                },
            },
            {
                "label": "full",
                "kwargs": {
                    "model": STT_MODEL,
                    "language": STT_LANGUAGE,
                    "smart_format": STT_SMART_FORMAT,
                    "interim_results": STT_INTERIM_RESULTS,
                    "endpointing": STT_ENDPOINTING_MS,
                    "utterance_end_ms": str(STT_UTTERANCE_END_MS),
                    "encoding": self._encoding,
                    "sample_rate": self._sample_rate,
                    "channels": 1,
                    "diarize": True,
                },
            },
            {
                "label": "no-diarize",
                "kwargs": {
                    "model": STT_MODEL,
                    "language": STT_LANGUAGE,
                    "smart_format": STT_SMART_FORMAT,
                    "interim_results": STT_INTERIM_RESULTS,
                    "endpointing": STT_ENDPOINTING_MS,
                    "utterance_end_ms": str(STT_UTTERANCE_END_MS),
                    "encoding": self._encoding,
                    "sample_rate": self._sample_rate,
                    "channels": 1,
                    "diarize": False,
                },
            },
        ]
        started = False
        last_attempt = None
        for cfg in attempts:
            last_attempt = cfg
            live_opts = LiveOptions(**cfg["kwargs"])
            logger.info(
                "Deepgram STT connect attempt profile=%s kwargs=%s",
                cfg["label"], cfg["kwargs"]
            )
            started = await self._connection.start(live_opts)
            if started:
                break
            logger.warning(
                "Deepgram STT connect attempt failed (profile=%s)",
                cfg["label"]
            )

        if not started:
            raise RuntimeError(f"Failed to open Deepgram WebSocket (last_attempt={last_attempt})")
        logger.info(
            "Deepgram STT connected (profile=%s kwargs=%s)",
            last_attempt["label"], last_attempt["kwargs"]
        )

        # Explicit keepalive every 8s — prevents Deepgram 1011 timeout when
        # caller's mic is suppressed during TTS playback (echo cancellation).
        # keepalive:"true" in DeepgramClientOptions only handles WS-level pings,
        # not Deepgram's application-level audio-inactivity timer.
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Forward a raw audio chunk to Deepgram."""
        if self._connection:
            await self._connection.send(audio_chunk)

    async def close(self) -> None:
        """Gracefully close the STT connection."""
        if hasattr(self, "_keepalive_task"):
            self._keepalive_task.cancel()
        if self._connection:
            await self._connection.finish()
            self._connection = None
            logger.info("Deepgram STT connection closed")

    async def _keepalive_loop(self) -> None:
        """Send application-level KeepAlive to Deepgram every 8s.

        Deepgram closes the connection (1011) if no audio OR KeepAlive message
        is received within its inactivity window (~10-12s). This fires even when
        keepalive:"true" is set in DeepgramClientOptions because that option only
        handles WebSocket-level pings, not Deepgram's audio-inactivity timer.
        """
        try:
            while self._connection:
                await asyncio.sleep(8)
                if self._connection:
                    await self._connection.keep_alive()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("STT keepalive stopped: %s", exc)

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _handle_transcript(self, _client, result, **_kwargs) -> None:
        try:
            alt = result.channel.alternatives[0]
            transcript: str = alt.transcript.strip()
            is_final: bool = result.is_final
            speech_final: bool = getattr(result, "speech_final", False)
            confidence: float = getattr(alt, "confidence", 1.0) or 1.0

            # ── Speaker filter ────────────────────────────────────────────────
            # Diarization assigns a speaker ID to each word. Derive the dominant
            # speaker for this result and compare to the locked primary speaker.
            words = getattr(alt, "words", None)
            if words:
                speaker_ids = [getattr(w, "speaker", None) for w in words
                               if getattr(w, "speaker", None) is not None]
                if speaker_ids:
                    dominant = max(set(speaker_ids), key=speaker_ids.count)
                    if self._primary_speaker is None and speech_final:
                        self._primary_speaker = dominant
                        logger.info("STT: primary speaker locked → %d", dominant)
                    elif self._primary_speaker is not None and dominant != self._primary_speaker:
                        logger.debug(
                            "STT: ignoring speaker %d (primary=%d)",
                            dominant, self._primary_speaker,
                        )
                        return

            if speech_final:
                # speech_final is Deepgram's signal that the utterance is complete.
                #
                # For SHORT utterances Deepgram emits the same sentence TWICE:
                #   Event 1: is_final=True, speech_final=False  → "Hi Priya can you help"
                #   Event 2: is_final=True, speech_final=True   → "Hi, Priya. Can you help?"
                #
                # Naïvely appending both produces "Hi Priya can you help Hi, Priya. Can you help?"
                # Fix: when speech_final fires, check word overlap between the new transcript
                # and accumulated parts.  High overlap → speech_final is a corrected revision
                # of the same sentence, use it alone.  Low overlap → it's a second segment of
                # a long utterance, prepend what we have.

                if transcript and confidence >= STT_CONFIDENCE_THRESHOLD:
                    if self._utterance_parts:
                        acc_words = set(" ".join(self._utterance_parts).lower().split())
                        new_words = set(transcript.lower().split())
                        overlap = len(acc_words & new_words) / max(len(acc_words), 1)
                        if overlap >= 0.6:
                            # speech_final is a polished revision — prefer it alone
                            full_utterance = transcript
                        else:
                            # Genuinely new segment (long multi-part utterance)
                            full_utterance = " ".join(self._utterance_parts) + " " + transcript
                    else:
                        full_utterance = transcript
                elif self._utterance_parts:
                    # speech_final fired with empty transcript — flush accumulated parts
                    full_utterance = " ".join(self._utterance_parts)
                else:
                    return

                self._utterance_parts = []

                # Keep short utterances configurable for faster barge-in responses.
                if len(full_utterance.split()) < STT_MIN_WORDS:
                    logger.debug("STT: discarding short utterance (min_words=%d): %r", STT_MIN_WORDS, full_utterance)
                    return

                logger.info("STT utterance complete: %s", full_utterance)
                await self._on_transcript(full_utterance, True)
                return

            # Non-speech_final is_final: accumulate as an intermediate segment
            if is_final and transcript:
                if confidence < STT_CONFIDENCE_THRESHOLD:
                    logger.debug("STT fragment discarded (confidence=%.2f < %.2f): %s",
                                 confidence, STT_CONFIDENCE_THRESHOLD, transcript)
                    return
                self._utterance_parts.append(transcript)
                logger.debug("STT fragment [conf=%.2f]: %s", confidence, transcript)

        except (AttributeError, IndexError) as exc:
            logger.warning("STT transcript parse error: %s", exc)

    async def _handle_speech_started(self, _client, _result, **_kwargs) -> None:
        logger.debug("STT: SpeechStarted — barge-in trigger")
        self._utterance_parts = []  # discard any partial fragments from previous turn
        if self._on_speech_started:
            await self._on_speech_started()

    async def _handle_error(self, _client, error, **_kwargs) -> None:
        logger.error("Deepgram STT error: %s", error)
