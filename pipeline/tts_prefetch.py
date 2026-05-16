"""One-slot background Sarvam TTS prefetch for the next sentence."""
import asyncio
import logging
from typing import Optional

from services.tts_sarvam import sarvam_synthesize_to_bytes

logger = logging.getLogger(__name__)


class TtsPrefetchSlot:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._text: Optional[str] = None
        self._result: Optional[bytes] = None

    def start(self, text: str, language_code: str, pitch_override: Optional[float]) -> None:
        self.cancel()
        self._text = text
        self._result = None

        async def _run():
            try:
                self._result = await sarvam_synthesize_to_bytes(text, language_code, pitch_override)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("Prefetch failed: %s", exc)

        self._task = asyncio.create_task(_run())

    def take(self) -> Optional[bytes]:
        if self._task and not self._task.done():
            return None
        data = self._result
        self._result = None
        self._text = None
        self._task = None
        return data

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self._result = None
        self._text = None
