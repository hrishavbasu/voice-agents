"""
Barge-in / interruption handling.

When Deepgram fires SpeechStarted while the TTS is playing:
  1. Set a flag so the TTS async generator stops yielding.
  2. Send a short silence burst to the carrier to avoid clicking.
  3. Flush any queued audio from the send buffer.

The pipeline owns an InterruptionController instance shared between the
STT callback and the TTS send loop.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class InterruptionController:
    """
    Thread-safe (asyncio) controller for barge-in interruption.

    Usage:
        ctrl = InterruptionController()

        # In STT speech-started callback:
        await ctrl.trigger()

        # In TTS send loop (check before each chunk):
        if ctrl.is_interrupted:
            break

        # Reset before starting a new TTS utterance:
        ctrl.reset()
    """

    # Debounce window in seconds — short noise bursts under this threshold
    # don't count as a real barge-in, preventing the bot from cutting itself
    # off on waiting-room noise, coughs, or brief audio artefacts.
    _DEBOUNCE_SECS = 0.15

    def __init__(self) -> None:
        self._interrupted = False
        self._debouncing = False
        self._event = asyncio.Event()
        self._reset_gen = 0  # incremented by reset() to cancel in-flight debounces

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted

    async def trigger(self) -> None:
        """Called when speech is detected during TTS playback.

        A 150 ms debounce prevents a single noise spike from killing the
        current TTS sentence.  If reset() is called within the debounce
        window (e.g. TTS finished naturally), the interrupt is discarded.
        """
        if self._interrupted or self._debouncing:
            return
        self._debouncing = True
        gen = self._reset_gen
        await asyncio.sleep(self._DEBOUNCE_SECS)
        self._debouncing = False
        # Discard if reset() was called during the sleep
        if self._reset_gen != gen:
            return
        if not self._interrupted:
            self._interrupted = True
            self._event.set()
            logger.debug("Barge-in confirmed after debounce")

    def reset(self) -> None:
        """Call this before starting each new TTS utterance."""
        self._interrupted = False
        self._debouncing = False
        self._reset_gen += 1
        self._event.clear()

    async def wait_for_interruption(self) -> None:
        """Await until a barge-in is triggered (used in tests / monitoring)."""
        await self._event.wait()
