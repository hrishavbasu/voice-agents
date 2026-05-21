"""
Voice pipeline — the core per-call orchestration loop.

One VoicePipeline instance is created per incoming call. It wires together:
  STT → LLM → TTS → Telephony

And handles:
  - Barge-in (interruption controller)
  - Tool execution (CRM, scheduling, escalation)
  - Session state (memory store)
  - Filler audio injection before tool calls
  - Retry counting and auto-escalation

Usage (called from main.py WebSocket handler):
    pipeline = VoicePipeline(call_id, caller_phone, telephony_session)
    await pipeline.start()
    # pipeline.on_audio(chunk) is called by telephony receive_loop
    await pipeline.shutdown()
"""

import asyncio
import logging
import random
import re
import time
from typing import Optional

from config.base_config import (
    SILENCE_TIMEOUT_SECS,
    SILENCE_HANGUP_SECS,
    ADAPTIVE_HOLD_SHORT_MS,
    ADAPTIVE_HOLD_NORMAL_MS,
    BARGE_IN_ACK_MODE,
    POST_BOOKING_WAIT_SECS,
)
from pipeline.tts_prefetch import TtsPrefetchSlot
from pipeline.caller_context import (
    merge_context_from_utterance,
    merge_context_from_crm,
)
from config.company_config import COMPANY_CONFIG
from pipeline.interruption import InterruptionController
from pipeline.session import (
    create_session,
    get_session,
    update_session,
    append_message,
    increment_retry,
    get_transcript,
    end_session,
)
from prompts.system_prompt import build_system_prompt
from services.stt import get_stt
from services.llm import stream_response
from services.tts import TTSService, FILLER_TEXTS
from tools.definitions import get_tool_schemas
from tools.crm import lookup_contact_by_phone, create_or_update_contact

logger = logging.getLogger(__name__)


def _detect_language(text: str) -> str:
    """
    Classify transcript as 'english', 'hindi', or 'hinglish'.

    Uses the ratio of Devanagari characters to total alphabetic characters:
      > 60%  → hindi
      5-60%  → hinglish  (code-switched)
      < 5%   → english
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return "hinglish"
    deva = sum(1 for c in alpha if "\u0900" <= c <= "\u097f")
    ratio = deva / len(alpha)
    if ratio > 0.60:
        return "hindi"
    if ratio > 0.05:
        return "hinglish"
    return "english"


# Language-aware filler phrases — matched to detected caller language so the
# bot doesn't suddenly switch to English mid Hindi/Hinglish conversation.
TOOL_FILLERS: dict[str, list[str]] = {
    "english": [
        "Let me check that for you.",
        "One moment.",
        "Give me just a second.",
        "I'll look into that now.",
        "Bear with me.",
        "Just a moment.",
        "Let me pull that up.",
        "I'll find out right away.",
    ],
    "hindi": [
        "एक पल के लिए रुकिए।",
        "अभी देख लेते हैं।",
        "बस एक सेकंड।",
        "चेक कर रहे हैं।",
        "थोड़ा सा इंतज़ार करें।",
        "हाँ, अभी पता करते हैं।",
    ],
    "hinglish": [
        "एक second, check कर रहे हैं।",
        "बस एक पल।",
        "देख लेते हैं।",
        "Just a second.",
        "Let me check करते हैं।",
        "हाँ, अभी देख लेते हैं।",
    ],
}
_BARGE_IN_ACK: dict = {
    "hindi":    ["haan, bataiye", "ji, haan?", "haan ji?"],
    "hinglish": ["haan, bataiye", "yes, bataiye?", "haan ji?"],
    "english":  ["yes, go ahead", "please go on"],
}


def collect_warm_cache_phrases() -> list[tuple[str, str]]:
    """Phrases to pre-synthesize at startup (fillers + barge-in acks)."""
    phrases: list[tuple[str, str]] = []
    for lang, fillers in TOOL_FILLERS.items():
        code = "hi-IN" if lang in ("hindi", "hinglish") else "en-IN"
        phrases.extend((f, code) for f in fillers)
    for lang, acks in _BARGE_IN_ACK.items():
        code = "hi-IN" if lang in ("hindi", "hinglish") else "en-IN"
        phrases.extend((a, code) for a in acks)
    return phrases


# Keywords that require an immediate hard emergency response before LLM.
# Language-inclusive: covers English + common Hindi/Hinglish equivalents.
# Matched with word boundaries via re — avoids substring false positives
# (e.g. "help me with booking" must NOT fire; "chest pain" must).
_EMERGENCY_KEYWORDS = frozenset({
    "emergency", "chest pain", "heart attack", "stroke",
    "bleeding", "unconscious", "can't breathe", "cant breathe",
    "not breathing", "seizure", "overdose", "ambulance",
    "severe pain", "help me now", "dying",
})

# Substring match (no \\b — Devanagari / multi-word Hindi phrases)
_EMERGENCY_PHRASES = (
    "छाती में दर्द",
    "सांस नहीं",
    "सांस लेना",
    "बेहोश",
    "खून बह",
    "इमरजेंसी",
    "बचाओ",
    "बहुत तकलीफ",
    "दिल का दौरा",
    "एम्बुलेंस",
    "bahut taklif",
    "saans nahi",
    "behosh",
    "imergency",
)

_DISCOURSE_RE = re.compile(
    r'\b(हाँ|ठीक है|actually|हाँ जी|अच्छा)(\s+)(?=[^\s,।])',
    re.IGNORECASE,
)


def _add_prosody_markers(text: str) -> str:
    """Add natural pause cues before sending text to TTS."""
    text = text.replace("—", ", ")
    text = text.replace("...", ",")
    text = _DISCOURSE_RE.sub(lambda m: m.group(1) + "," + m.group(2), text)
    return text


def _strip_markdown(text: str) -> str:
    """Remove Gemini markdown artefacts that TTS reads literally."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+:$', '', text)
    return text.strip()


class VoicePipeline:
    """Orchestrates a single inbound call end-to-end."""

    def __init__(
        self,
        call_id: str,
        caller_phone: str,
        telephony_session,  # services.telephony.TelephonySession
    ) -> None:
        self.call_id = call_id
        self.caller_phone = caller_phone
        self._telephony = telephony_session

        self._tts = TTSService()
        self._interruption = InterruptionController()
        self._stt: Optional = None

        self._tool_schemas = get_tool_schemas()
        self._crm_contact: Optional[dict] = None
        self._caller_language: str = "hinglish"  # updated on first clear detection
        self._tts_playing = False   # True when a TTS chunk is actively streaming
        self._agent_in_turn = False  # True for the entire agent response turn (multi-sentence)
        self._playback_until = 0.0
        self._prefetch = TtsPrefetchSlot()

        # Queue of final STT transcripts waiting to be processed
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._last_activity_at: float = 0.0
        self._user_turn_count = 0
        self._greeting_finished = False
        self._stt_retry_prompted = False
        self._bg_tasks: set[asyncio.Task] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise session, STT connection, greet caller, start loops."""
        self._running = True

        # Create session
        await create_session(self.call_id, self.caller_phone)

        # Async CRM lookup (don't block greeting)
        self._spawn(self._async_crm_lookup())

        # Connect STT
        self._stt = get_stt(
            on_transcript=self._on_transcript,
            on_speech_started=self._on_speech_started,
        )
        await self._stt.connect()

        # Wait for Twilio "start" frame confirming stream is ready
        try:
            await asyncio.wait_for(self._telephony.stream_ready.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Stream ready timeout — proceeding anyway")

        # Twilio passes caller_phone in stream customParameters (not always in WS URL)
        tel_caller = getattr(self._telephony, "caller_phone", None)
        if tel_caller and tel_caller != "unknown":
            self.caller_phone = tel_caller
            await update_session(self.call_id, {"caller_phone": tel_caller})

        # Play greeting (silence watch stays off until greeting finishes)
        greeting = await self._build_greeting()
        await self._speak(greeting)
        self._greeting_finished = True
        self._last_activity_at = time.monotonic()

        # Start the LLM processing loop
        self._spawn(self._llm_loop())
        self._spawn(self._silence_watch_loop())

    async def shutdown(self) -> None:
        """Cleanly close STT connection and end session."""
        self._running = False
        if self._bg_tasks:
            for task in list(self._bg_tasks):
                task.cancel()
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)
            self._bg_tasks.clear()
        if self._stt:
            await self._stt.close()
        await end_session(self.call_id)
        logger.info("Pipeline shutdown for call_id=%s", self.call_id)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # ── Audio ingestion (called by telephony receive_loop) ────────────────────

    async def on_audio(self, chunk: bytes) -> None:
        """Forward raw audio from the carrier to STT; track caller activity."""
        if not self._running or not chunk:
            return
        # Inbound speech while agent is not talking → reset silence timer
        if self._greeting_finished and not self._tts_playing and not self._agent_in_turn:
            try:
                pcm = audioop.ulaw2lin(chunk, 2)
                if audioop.rms(pcm, 2) > 180:
                    self._last_activity_at = time.monotonic()
            except Exception:
                pass
        if self._stt:
            await self._stt.send_audio(chunk)

    # ── STT callbacks ─────────────────────────────────────────────────────────

    async def _on_transcript(self, text: str, is_final: bool) -> None:
        self._last_activity_at = time.monotonic()
        if is_final and text.strip():
            self._user_turn_count += 1
            logger.info("[CALLER] %s", text)
            # Persist caller text immediately so call logs always show user data
            await append_message(self.call_id, "user", text)
            session = await get_session(self.call_id) or {}
            ctx_updates = merge_context_from_utterance(session, text)
            if ctx_updates:
                await self._apply_session_context(
                    session, ctx_updates, source="utterance", detail=text
                )
            await self._transcript_queue.put(text)

    async def _on_speech_started(self) -> None:
        """Barge-in when caller speaks over the agent; always treat as activity."""
        self._last_activity_at = time.monotonic()
        if not self._greeting_finished:
            return

        agent_audible = (
            self._tts_playing
            or self._agent_in_turn
            or (time.monotonic() < self._playback_until)
        )
        if not agent_audible:
            return

        sentences_spoken = getattr(self, "_sentences_spoken_this_turn", 0)
        logger.info("Barge-in: caller speaking over agent playback")
        self._tts.cancel()
        await self._telephony.clear_playback_buffer()
        await self._interruption.trigger_immediate()
        # Do not drain _transcript_queue — the caller's words are needed for the next turn.
        # Slightly longer tail silence helps flush any residual buffered playback.
        await self._telephony.send_silence(140)
        self._cancel_prefetch()

        if not self._should_play_barge_in_ack(sentences_spoken):
            return
        if not await self._interruption.debounce_ack_gate():
            return
        lang = self._caller_language if self._caller_language in _BARGE_IN_ACK else "hinglish"
        ack = random.choice(_BARGE_IN_ACK[lang])
        self._interruption.reset()
        await self._speak(ack)

    def _should_play_barge_in_ack(self, sentences_spoken: int) -> bool:
        if sentences_spoken < 1:
            return False
        mode = BARGE_IN_ACK_MODE
        if mode == "silent":
            return False
        if mode == "always":
            return True
        return self._transcript_queue.empty()

    def _cancel_prefetch(self) -> None:
        self._prefetch.cancel()

    def _lang_code(self) -> str:
        return "hi-IN" if self._caller_language in ("hindi", "hinglish") else "en-IN"

    def _schedule_prefetch(self, text: str, pitch_override: float = None) -> None:
        self._prefetch.start(text, self._lang_code(), pitch_override)

    # ── LLM processing loop ───────────────────────────────────────────────────

    async def _llm_loop(self) -> None:
        """
        Wait for final transcripts from the STT queue, send to LLM,
        stream TTS sentences, execute any tool calls.
        """
        while self._running:
            try:
                user_text = await asyncio.wait_for(
                    self._transcript_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            # Adaptive hold: short utterances (< 3 words) get 400ms to absorb
            # Hindi filler tokens ("haan", "okay", "ji") that precede the real
            # request. Normal utterances still get 150ms.
            _hold_ms = ADAPTIVE_HOLD_SHORT_MS if len(user_text.split()) < 3 else ADAPTIVE_HOLD_NORMAL_MS
            await asyncio.sleep(_hold_ms / 1000)
            while not self._transcript_queue.empty():
                extra = self._transcript_queue.get_nowait()
                user_text = user_text + " " + extra
            # Caller often starts with short fragments ("मैम मेरे ...").
            # Wait a bit longer once before responding to avoid robotic "ji?"
            if self._needs_extra_short_hold(user_text):
                await asyncio.sleep(0.45)
                while not self._transcript_queue.empty():
                    extra = self._transcript_queue.get_nowait()
                    user_text = user_text + " " + extra

            # Suppress isolated single-word utterances — almost always partial speech or echo
            if len(user_text.split()) == 1:
                await asyncio.sleep(0.6)
                while not self._transcript_queue.empty():
                    user_text = user_text + " " + self._transcript_queue.get_nowait()
                if len(user_text.split()) == 1:
                    logger.info("Suppressing single-word utterance: %r", user_text)
                    continue

            # User turn already in session/transcript from _on_transcript
            session = await get_session(self.call_id) or {}

            # Hard emergency intercept — fires BEFORE LLM, deterministic.
            # LLM-based detection is not reliable enough for healthcare.
            if await self._check_emergency(user_text):
                continue

            session = await get_session(self.call_id) or session
            if await self._handle_out_of_scope(user_text, session):
                continue

            # Detect caller language and persist — first clear detection wins;
            # subsequent turns can shift hinglish→english/hindi but not overwrite
            # a firm detection unless the ratio is unambiguous (>60%).
            detected = _detect_language(user_text)
            session = await get_session(self.call_id) or session
            current_lang = session.get("caller_language")

            # Language lock: once identified as Hindi/Hinglish, never downgrade to
            # English on a single utterance — callers mix in English words constantly.
            # "But ma'am," has no Devanagari → naively detected as "english",
            # but the caller is Hinglish.  Only allow english→hindi/hinglish or
            # first-time detection to set the language; never hindi/hinglish→english.
            _allow_update = (
                not current_lang  # first detection
                or (current_lang == "english" and detected != "english")  # upgrade
                or (current_lang == "hinglish" and detected == "hindi")   # refine
            )
            if _allow_update and current_lang != detected:
                await update_session(self.call_id, {"caller_language": detected})
                current_lang = detected
                self._caller_language = detected

            messages = session.get("messages", []) if session else []

            # Prepend system prompt (not stored in session to save space)
            system = build_system_prompt(
                caller_phone=self.caller_phone,
                crm_contact=self._crm_contact,
                caller_language=current_lang,
                session=session,
            )
            from services.llm_gemini import trim_messages_for_llm

            full_messages = trim_messages_for_llm(
                [{"role": "system", "content": system}] + messages,
                max_non_system=20,
            )

            # Stream LLM response
            assistant_text_parts = []
            self._agent_in_turn = True
            self._sentences_spoken_this_turn = 0
            self._interruption.reset()
            self._tts.reset()
            self._cancel_prefetch()
            correction_text: str | None = None
            prev_sentence: str | None = None

            async for item in stream_response(full_messages, tools=self._tool_schemas):
                if self._interruption.is_interrupted:
                    logger.info("Barge-in: stopping LLM response playback")
                    break

                # Mid-generation correction: caller spoke while LLM was streaming.
                # Cancel TTS, note the correction, break — outer loop will reprocess
                # with user_text + correction so the agent never acts on stale input.
                if not self._transcript_queue.empty():
                    correction_text = self._transcript_queue.get_nowait()
                    while not self._transcript_queue.empty():
                        correction_text += " " + self._transcript_queue.get_nowait()
                    logger.info("Mid-generation correction: %r — reprocessing", correction_text)
                    self._tts.cancel()
                    break

                if isinstance(item, str):
                    item = self._postprocess_agent_text(item, session)
                    if prev_sentence is not None:
                        assistant_text_parts.append(prev_sentence)
                        self._sentences_spoken_this_turn += 1
                        prefetched = self._prefetch.take()
                        await self._speak(
                            prev_sentence,
                            prefetched_audio=prefetched,
                            record_transcript=False,
                        )
                        if self._interruption.is_interrupted:
                            break
                    pitch = 0.05 if item.rstrip().endswith("?") else None
                    self._schedule_prefetch(item, pitch)
                    prev_sentence = item

                elif isinstance(item, dict) and item.get("type") == "tool_call":
                    if prev_sentence:
                        assistant_text_parts.append(prev_sentence)
                        self._sentences_spoken_this_turn += 1
                        prefetched = self._prefetch.take()
                        await self._speak(
                            prev_sentence,
                            prefetched_audio=prefetched,
                            record_transcript=False,
                        )
                        prev_sentence = None
                        self._cancel_prefetch()
                    await self._handle_tool_call(item)

            if prev_sentence and not self._interruption.is_interrupted:
                assistant_text_parts.append(prev_sentence)
                self._sentences_spoken_this_turn += 1
                prefetched = self._prefetch.take()
                await self._speak(
                    prev_sentence,
                    prefetched_audio=prefetched,
                    record_transcript=False,
                )

            self._agent_in_turn = False
            self._cancel_prefetch()
            # Record full assistant turn
            if assistant_text_parts:
                full_response = " ".join(assistant_text_parts)
                await append_message(self.call_id, "assistant", full_response)

            # Re-queue correction so the next loop iteration processes it with
            # full context (including what the agent just said before being cut off).
            if correction_text:
                await self._transcript_queue.put(correction_text)

    # ── TTS speaking ──────────────────────────────────────────────────────────

    @staticmethod
    def _postprocess_agent_text(text: str, session: dict) -> str:
        """Session-aware rewrites to avoid redundant robotic prompts."""
        out = text
        # Guard fires when a SPECIFIC DOCTOR is already selected. At that point
        # asking "what's your concern / why do you want a doctor" is redundant —
        # the caller has already committed to a doctor. Only requiring specialty
        # (not doctor) is not enough: the agent may legitimately ask about the
        # concern when specialty is known but no specific doctor is chosen yet
        # (to narrow down which doctor within the specialty). That caused a
        # 5-repetition loop in live calls when specialty-only context was set.
        has_doctor_ctx = bool((session or {}).get("preferred_doctor"))
        if has_doctor_ctx and re.search(r"(परेशानी|किसलिए)", out):
            return "ठीक है, हम appointment आगे बढ़ाते हैं।"
        if (session or {}).get("caller_name") and re.search(r"अपना\s+नाम\s+बता", out):
            return "धन्यवाद, आपका नाम मेरे पास है।"
        return out

    @staticmethod
    def _needs_extra_short_hold(text: str) -> bool:
        """
        Return True for likely trailing fragments where waiting briefly improves
        turn cohesion and reduces accidental interjections.
        """
        t = (text or "").strip().lower()
        if not t:
            return False
        words = t.split()
        if len(words) >= 3:
            return False
        # Typical fragment/openers observed in live calls.
        fragment_tokens = {
            "मैम", "ma'am", "madam", "mam", "मेरे", "मेरा", "मैं", "जी", "hmm", "हम्म",
            "जो", "तो", "और",
        }
        return any(w in fragment_tokens for w in words)

    @staticmethod
    def _normalize_for_tts(text: str) -> str:
        """Pre-process text so TTS engines don't read symbols or digits unnaturally."""
        # ₹1,700 → "rupees 1700" (comma stripped so TTS reads as integer)
        # Keep as English fallback; LLM should ideally produce word-form already.
        text = re.sub(r"₹\s*([\d,]+)", lambda m: "rupees " + m.group(1).replace(",", ""), text)
        # Bare ₹ without number
        text = text.replace("₹", "rupees")
        # Strip thousand-separator commas from standalone numbers so "1,700" → "1700"
        text = re.sub(r"\b(\d{1,3}),(\d{3})\b", r"\1\2", text)
        # Strip parenthetical English time annotations — LLM writes "साढ़े तीन बजे (3:30 PM)"
        # for human clarity but TTS reads the parenthetical aloud. The Hindi already says it.
        text = re.sub(r"\(\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*\)", "", text)
        # Expand Devanagari abbreviations that TTS reads unnaturally
        text = text.replace("डॉ.", "डॉक्टर")
        # Devanagari substitutions for Latin-script proper nouns Tripti mispronounces
        for original, replacement in COMPANY_CONFIG.get("tts_substitutions", {}).items():
            text = text.replace(original, replacement)
        # Avoid robotic honorific address in mixed-script output.
        text = re.sub(r"\b(?:Ma'am|Madam|mam)\b", "जी", text, flags=re.IGNORECASE)
        text = _strip_markdown(text)
        text = _add_prosody_markers(text)
        text = re.sub(
            r"क्या\s+आप\s+उनसे\s+appointment\s+लेना\s+ठीक\s+रहेगा\?",
            "क्या हम उनके साथ appointment रखें?",
            text,
        )
        text = re.sub(
            r"क्या\s+आप\s+अपना\s+नाम\s+बता\s+सक(?:ती|ते)\s+हैं\?",
            "कृपया अपना नाम बता दें?",
            text,
        )
        text = re.sub(
            r"क्या\s+आप\s+उनके\s+लिए\s+कोई\s+और\s+समय\s+देखना\s+चाह(?:ेंगी|ेंगे|ें)\?",
            "क्या उनके लिए कोई और समय देखें?",
            text,
        )
        text = re.sub(
            r"appointment\s+देखना\s+चाह(?:ेंगी|ेंगे|ें)\?",
            "appointment देखना चाहें?",
            text,
        )
        # Keep caller-directed phrasing gender-neutral.
        text = re.sub(r"आना\s+चाह(?:ेंगी|ेंगे|ें)\??", "आना ठीक रहेगा?", text)
        text = re.sub(r"लेना\s+चाह(?:ेंगी|ेंगे|ें)\??", "लेना ठीक रहेगा?", text)
        text = re.sub(r"करना\s+चाह(?:ेंगी|ेंगे|ें)\??", "करना ठीक रहेगा?", text)
        text = re.sub(r"आप\s+से\s+appointment\s+लेना\s+ठीक\s+रहेगा\?", "appointment लेना ठीक रहेगा?", text)
        return text

    def _should_stop_speaking(self) -> bool:
        return self._interruption.is_interrupted or self._tts.is_cancelled

    async def _speak(
        self,
        text: str,
        pitch_override: float = None,
        prefetched_audio: bytes | None = None,
        *,
        record_transcript: bool = True,
    ) -> float:
        """Synthesise text and stream audio to the carrier.

        Returns approximate playback duration in seconds (bytes / 8000 for μ-law 8kHz).
        Does NOT reset the interruption controller — callers own that reset
        so a barge-in during a filler phrase is not silently swallowed.
        """
        text = self._normalize_for_tts(text)
        logger.info("[AGENT] %s", text)
        self._tts.reset()
        self._tts_playing = True
        total_bytes = 0
        # Buffer to 160-byte (20 ms) boundaries — Twilio's G.711 packet size.
        # Sending sub-frame chunks causes decoder glitches that sound like crackling.
        _FRAME = 160
        buf = bytearray()
        try:
            lang_code = "hi-IN" if self._caller_language in ("hindi", "hinglish") else "en-IN"
            _pitch_bump = pitch_override if pitch_override is not None else (0.05 if text.rstrip().endswith("?") else None)

            async def _audio_chunks():
                if prefetched_audio is not None:
                    for i in range(0, len(prefetched_audio), 1024):
                        yield prefetched_audio[i : i + 1024]
                else:
                    async for chunk in self._tts.synthesize(
                        text, language_code=lang_code, pitch_override=_pitch_bump
                    ):
                        yield chunk

            async for chunk in _audio_chunks():
                if not self._running:
                    break
                if self._should_stop_speaking():
                    break
                buf.extend(chunk)
                while len(buf) >= _FRAME:
                    if not self._running:
                        break
                    if self._should_stop_speaking():
                        break
                    frame = bytes(buf[:_FRAME])
                    buf = buf[_FRAME:]
                    total_bytes += len(frame)
                    await self._telephony.send_audio(frame)
                    # Pace sending to real-time (20 ms per 20 ms frame) so that
                    # Twilio's play buffer never gets more than ~1 frame ahead of
                    # actual playback.  Without this sleep, the entire sentence is
                    # queued instantly and a barge-in clear arrives too late to
                    # interrupt the current sentence.
                    await asyncio.sleep(0.018)
                if self._should_stop_speaking():
                    break
            # Flush remainder padded with μ-law silence (0xFF)
            if buf and self._running and not self._should_stop_speaking():
                padded = bytes(buf) + bytes([0xFF] * (_FRAME - len(buf) % _FRAME))
                total_bytes += len(buf)
                await self._telephony.send_audio(padded)
        finally:
            self._tts_playing = False
            if self._greeting_finished:
                self._last_activity_at = time.monotonic()
        duration = total_bytes / 8000
        self._playback_until = time.monotonic() + duration + 0.6
        if record_transcript and self._running and text.strip():
            await append_message(self.call_id, "assistant", text)
        return duration

    async def _play_filler(self) -> None:
        """Play a random filler phrase in the caller's language.

        Avoids repeating the same phrase twice in a row.
        """
        lang = self._caller_language if self._caller_language in TOOL_FILLERS else "hinglish"
        pool = TOOL_FILLERS[lang]
        candidates = [f for f in pool if f != getattr(self, "_last_filler", None)]
        filler = random.choice(candidates or pool)
        self._last_filler = filler
        await self._speak(filler)

    # ── Tool execution ────────────────────────────────────────────────────────

    async def _handle_tool_call(self, tool_call: dict) -> None:
        import json as _json

        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})
        tool_id = tool_call.get("id", f"call_{name}")
        logger.info("Executing tool: %s(%s)", name, args)

        # NOTE: assistant+tool_calls is saved AFTER the tool result is known,
        # inside _speak_tool_result. Saving it here (before the tool runs) causes
        # orphaned tool_calls messages if barge-in interrupts before the result.

        await self._play_filler()

        result: dict = {}

        try:
            if name == "check_doctor_slots":
                from tools.scheduling import check_doctor_slots
                doc = args.get("doctor_name", "")
                pref_date = args.get("preferred_date")
                slot_updates = {}
                if doc:
                    slot_updates["preferred_doctor"] = doc
                if pref_date:
                    slot_updates["preferred_date"] = pref_date
                if slot_updates:
                    session = await get_session(self.call_id) or {}
                    session = await self._apply_session_context(
                        session, slot_updates, source="tool", detail=name
                    )
                result = check_doctor_slots(
                    doctor_name=doc,
                    preferred_date=pref_date,
                )

            elif name == "list_doctors":
                from tools.scheduling import list_doctors
                specialty = args.get("specialty")
                if specialty:
                    session = await get_session(self.call_id) or {}
                    await self._apply_session_context(
                        session,
                        {"preferred_specialty": specialty},
                        source="tool",
                        detail=name,
                    )
                result = list_doctors(specialty=specialty)

            elif name == "book_appointment":
                from tools.scheduling import book_appointment
                session = await get_session(self.call_id) or {}
                patient_name = (args.get("patient_name") or "").strip()
                if not patient_name or patient_name.lower() in {
                    "unknown", "caller", "patient", "user", "na",
                }:
                    patient_name = session.get("caller_name") or ""
                concern = (args.get("concern") or "").strip() or session.get("caller_concern") or ""
                doctor_name = args.get("doctor_name") or session.get("preferred_doctor")
                specialty = args.get("specialty") or session.get("preferred_specialty")
                preferred_date = args.get("preferred_date") or session.get("preferred_date")
                preferred_time = args.get("preferred_time") or session.get("preferred_time")
                result = book_appointment(
                    caller_phone=self.caller_phone,
                    patient_name=patient_name,
                    concern=concern,
                    doctor_name=doctor_name,
                    specialty=specialty,
                    preferred_date=preferred_date,
                    preferred_time=preferred_time,
                )
                book_ctx = {}
                if patient_name:
                    book_ctx["caller_name"] = patient_name
                if concern:
                    book_ctx["caller_concern"] = concern
                if doctor_name:
                    book_ctx["preferred_doctor"] = doctor_name
                if specialty:
                    book_ctx["preferred_specialty"] = specialty
                if preferred_date:
                    book_ctx["preferred_date"] = preferred_date
                if preferred_time:
                    book_ctx["preferred_time"] = preferred_time
                if book_ctx:
                    session = await get_session(self.call_id) or {}
                    await self._apply_session_context(
                        session, book_ctx, source="tool", detail=name
                    )

            elif name == "escalate_to_human":
                session = await get_session(self.call_id)
                crm_id = session.get("crm_contact_id") if session else None
                transcript = await get_transcript(self.call_id)
                from tools.escalation import escalate_to_human
                result = await escalate_to_human(
                    call_id=self.call_id,
                    caller_phone=self.caller_phone,
                    transcript=transcript,
                    reason=args.get("reason", "Customer requested human agent"),
                    telephony_session=self._telephony,
                    crm_contact_id=crm_id,
                )
                if result.get("success"):
                    await self._speak(
                        "I'm transferring you to one of our specialists now. "
                        "They'll have all the context from our conversation."
                    )
                    self._running = False
                    return

        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            result = {"success": False, "reason": str(exc)}

        # Check escalation threshold on failure
        if not result.get("success"):
            retry = await increment_retry(self.call_id)
            max_retry = COMPANY_CONFIG.get("max_retry_before_escalate", 2)
            if retry >= max_retry:
                await self._speak(
                    "I'm having difficulty resolving this. "
                    "Let me connect you with a human agent who can help."
                )
                from tools.escalation import escalate_to_human
                transcript = await get_transcript(self.call_id)
                session = await get_session(self.call_id)
                crm_id = session.get("crm_contact_id") if session else None
                await escalate_to_human(
                    call_id=self.call_id,
                    caller_phone=self.caller_phone,
                    transcript=transcript,
                    reason="Exceeded retry limit",
                    telephony_session=self._telephony,
                    crm_contact_id=crm_id,
                )
                self._running = False
                return

        # Reset barge-in state before speaking the tool result — a patient
        # who interrupted during the filler has already been noted; we still
        # want to speak the result so they hear what was found.
        self._interruption.reset()

        # Feed tool result back to LLM using the proper tool-role format so it
        # continues the conversation naturally (no separate summarize call needed).
        await self._speak_tool_result(name, tool_id, args, result)

        # Soft-close successful booking calls to avoid open-ended loops.
        if name == "book_appointment" and result.get("success"):
            await self._soft_close_after_booking()

    async def _speak_tool_result(
        self, tool_name: str, tool_call_id: str, tool_args: dict, result: dict
    ) -> None:
        """
        Append the tool result in OpenAI's proper format (role: tool) and let
        the LLM continue speaking naturally.  This avoids the double-injection
        and fake-user-message corruption that caused the booking loop.
        """
        import json as _json

        session = await get_session(self.call_id)
        messages = session.get("messages", []) if session else []
        caller_language = (session or {}).get("caller_language")
        system = build_system_prompt(
            caller_phone=self.caller_phone,
            crm_contact=self._crm_contact,
            caller_language=caller_language,
            session=session,
        )

        result_content = _json.dumps(result)

        # Build the assistant+tool_calls message explicitly so we can include it
        # in full_messages AND persist it. Fetching messages BEFORE appends means
        # we must manually append both messages to the in-memory list we pass to LLM.
        assistant_tool_call_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": _json.dumps(tool_args),
                    },
                }
            ],
        }
        tool_result_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result_content,
        }

        # Persist both messages to session (for future turns)
        await append_message(
            self.call_id,
            "assistant",
            "",
            extra={
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": _json.dumps(tool_args),
                        },
                    }
                ]
            },
        )
        await append_message(
            self.call_id,
            "tool",
            result_content,
            extra={"tool_call_id": tool_call_id, "name": tool_name},
        )

        # full_messages must include assistant+tool_calls BEFORE tool result —
        # messages was fetched before the appends above, so add them explicitly.
        from services.llm_gemini import trim_messages_for_llm

        full_messages = trim_messages_for_llm(
            [{"role": "system", "content": system}]
            + messages
            + [assistant_tool_call_msg, tool_result_msg],
            max_non_system=20,
        )

        response_parts = []
        async for item in stream_response(full_messages, tools=self._tool_schemas):
            if self._interruption.is_interrupted:
                break
            if isinstance(item, str):
                response_parts.append(item)
                await self._speak(item, record_transcript=False)
            elif isinstance(item, dict) and item.get("type") == "tool_call":
                # LLM chained another tool call (e.g. check_slots → book)
                await self._handle_tool_call(item)

        if response_parts:
            await append_message(
                self.call_id, "assistant", " ".join(response_parts)
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _apply_session_context(
        self,
        session: dict,
        updates: dict,
        *,
        source: str,
        detail: str,
    ) -> dict:
        from pipeline.caller_context import apply_context_updates, record_context_event

        field_updates, events = apply_context_updates(
            session, updates, source=source, detail=detail
        )
        if not field_updates:
            return session
        merged_events = record_context_event(session, events)
        payload = {**field_updates, "context_events": merged_events}
        return await update_session(self.call_id, payload)

    async def _handle_out_of_scope(self, user_text: str, session: dict) -> bool:
        """Return True if this turn was fully handled (no LLM)."""
        from pipeline.scope_guard import (
            caller_insists_on_human,
            decline_message,
            detect_out_of_scope,
            should_escalate_oos,
            transfer_message,
        )

        category = detect_out_of_scope(user_text)
        if not category:
            return False

        insists = caller_insists_on_human(user_text)
        strikes = int(session.get("out_of_scope_strikes") or 0)
        last_cat = session.get("last_oos_category")
        lang = session.get("caller_language") or self._caller_language

        if should_escalate_oos(
            category=category,
            strikes=strikes,
            last_category=last_cat,
            insists=insists,
        ):
            msg = transfer_message(lang)
            self._interruption.reset()
            await self._speak(msg)
            transcript = await get_transcript(self.call_id)
            crm_id = session.get("crm_contact_id")
            from tools.escalation import escalate_to_human

            await escalate_to_human(
                call_id=self.call_id,
                caller_phone=self.caller_phone,
                transcript=transcript,
                reason=f"OOS repeat/insist: {category}",
                telephony_session=self._telephony,
                crm_contact_id=crm_id,
            )
            await append_message(self.call_id, "assistant", msg)
            self._running = False
            return True

        msg = decline_message(category, lang)
        self._interruption.reset()
        await self._speak(msg)
        await append_message(self.call_id, "assistant", msg)
        await update_session(
            self.call_id,
            {
                "out_of_scope_strikes": strikes + 1,
                "last_oos_category": category,
            },
        )
        return True

    async def _check_emergency(self, text: str) -> bool:
        """Pre-LLM keyword scan for medical emergencies.

        Returns True and triggers immediate 108 prompt + escalation if any
        emergency keyword is detected.  This path is deterministic and does
        NOT route through the LLM, so it fires even under high LLM latency.
        """
        lower = text.lower()
        # Do not treat mild/ routine symptoms as emergency
        if re.search(r"\b(mild|slight|minor|thoda)\b", lower):
            matched_en = False
        else:
            matched_en = any(
                re.search(r"\b" + re.escape(kw) + r"\b", lower)
                for kw in _EMERGENCY_KEYWORDS
            )
        matched_hi = any(phrase in text for phrase in _EMERGENCY_PHRASES)
        if not matched_en and not matched_hi:
            return False

        logger.warning("EMERGENCY detected in: %r — 108 advisory + human transfer", text)

        self._interruption.reset()
        self._tts.cancel()
        await self._telephony.clear_playback_buffer()
        duration = await self._speak(
            "This sounds urgent. If it is life-threatening, please call one zero eight now. "
            "I am transferring you to our hospital team right away. "
            "यह जरूरी लग रहा है — अगर हालत गंभीर है तो अभी एक शून्य आठ पर कॉल करें। "
            "अभी हमारी टीम से जोड़ रहे हैं।"
        )
        # Wait for Twilio to finish playing the buffered audio before transferring.
        # _speak() returns as soon as the last byte is written to the WS buffer —
        # the transfer must not fire until the caller has actually heard the message.
        await asyncio.sleep(duration + 0.5)

        # Escalate with EMERGENCY priority flag
        transcript = await get_transcript(self.call_id)
        session = await get_session(self.call_id)
        crm_id = session.get("crm_contact_id") if session else None
        from tools.escalation import escalate_to_human
        await escalate_to_human(
            call_id=self.call_id,
            caller_phone=self.caller_phone,
            transcript=transcript,
            reason="EMERGENCY: keyword detected in caller speech",
            telephony_session=self._telephony,
            crm_contact_id=crm_id,
            priority="EMERGENCY",
        )
        self._running = False
        return True

    async def _soft_close_after_booking(self) -> None:
        if not self._running:
            return
        lang = self._caller_language
        if lang == "english":
            confirm_msg = (
                "Your appointment is confirmed. "
                "You'll receive an SMS confirmation shortly."
            )
            followup_msg = "Is there anything else I can help you with?"
            bye_msg = "Alright, take care. Goodbye!"
        elif lang == "hindi":
            confirm_msg = (
                "आपकी अपॉइंटमेंट कन्फर्म हो गई है। "
                "आपको SMS पर confirmation आ जाएगी।"
            )
            followup_msg = "कुछ और पूछना है?"
            bye_msg = "ठीक है, ध्यान रखिए। अलविदा।"
        else:
            confirm_msg = (
                "Aapki appointment confirm ho gayi hai. "
                "Aapko SMS par confirmation aa jaayegi."
            )
            followup_msg = "Kuch aur poochhna hai?"
            bye_msg = "Theek hai, dhyan rakhiye. Goodbye!"

        self._interruption.reset()
        await self._speak(confirm_msg)
        if not self._running:
            return

        await self._speak(followup_msg)

        # Wait up to POST_BOOKING_WAIT_SECS for a caller response.
        # If they respond, put it back in the queue so _llm_loop handles it naturally.
        self._agent_in_turn = False
        self._last_activity_at = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._transcript_queue.get(),
                timeout=POST_BOOKING_WAIT_SECS,
            )
            await self._transcript_queue.put(response)
            return  # caller responded — do not hang up
        except asyncio.TimeoutError:
            pass

        # No response — warm goodbye then hang up.
        self._interruption.reset()
        duration = await self._speak(bye_msg)
        await asyncio.sleep(duration + 0.3)
        await self._telephony.hangup()
        self._running = False

    async def _silence_watch_loop(self) -> None:
        """Prompt 'are you still there?' after inactivity, then hang up."""
        while self._running:
            await asyncio.sleep(1.0)
            if (
                not self._greeting_finished
                or self._tts_playing
                or self._agent_in_turn
                or not self._running
            ):
                continue

            stt_broken = (
                getattr(self._stt, "had_error", False) and self._user_turn_count == 0
            )
            idle_limit = SILENCE_TIMEOUT_SECS * 2 if stt_broken else SILENCE_TIMEOUT_SECS
            idle = time.monotonic() - self._last_activity_at
            if idle < idle_limit:
                continue

            if stt_broken and not getattr(self, "_stt_retry_prompted", False):
                self._stt_retry_prompted = True
                logger.warning("Silence with STT errors — prompting caller to repeat")
                self._last_activity_at = time.monotonic()
                await self._speak(
                    "Sorry, I could not hear you clearly. Please say that again."
                )
                continue

            logger.info("Silence timeout after %.0fs — prompting caller", idle)
            self._last_activity_at = time.monotonic()
            lang = self._caller_language
            if lang == "hindi":
                prompt = "क्या आप वहाँ हैं?"
            elif lang == "english":
                prompt = "Are you still there?"
            else:
                prompt = "क्या आप वहाँ हैं?"
            self._interruption.reset()
            await self._speak(prompt)

            # Wait for hangup window
            await asyncio.sleep(SILENCE_HANGUP_SECS)
            if not self._running:
                return
            idle2 = time.monotonic() - self._last_activity_at
            if idle2 < SILENCE_HANGUP_SECS:
                continue  # caller responded — back to normal watch

            logger.info("Caller still silent — hanging up")
            if lang == "hindi":
                bye = "ठीक है, आपसे बात करके अच्छा लगा। अलविदा।"
            elif lang == "english":
                bye = "Alright, have a good day. Goodbye."
            else:
                bye = "ठीक है, have a good day. Goodbye."
            self._interruption.reset()
            duration = await self._speak(bye)
            await asyncio.sleep(duration + 0.3)
            await self._telephony.hangup()
            self._running = False

    async def _build_greeting(self) -> str:
        cfg = COMPANY_CONFIG
        company = cfg.get("company_name", "our hospital")
        agent = cfg.get("agent_name", "Priya")
        session = await get_session(self.call_id) or {}
        name = session.get("caller_name")
        if not name and self._crm_contact:
            props = self._crm_contact.get("properties", {})
            name = f"{props.get('firstname', '')} {props.get('lastname', '')}".strip()
        if name:
            return (
                f"नमस्ते {name}, {company} में आपका स्वागत है। "
                f"मैं {agent} हूँ। "
                f"बताइए, आपकी क्या मदद कर सकते हैं?"
            )
        return (
            f"नमस्ते, {company} में आपका स्वागत है। "
            f"मैं {agent} हूँ। "
            f"बताइए, आपकी क्या मदद कर सकते हैं?"
        )

    async def _async_crm_lookup(self) -> None:
        """Non-blocking CRM lookup — resolves in background after call starts."""
        try:
            contact = lookup_contact_by_phone(self.caller_phone)
            if contact:
                self._crm_contact = contact
                session = await get_session(self.call_id) or {}
                crm_ctx = merge_context_from_crm(session, contact)
                if crm_ctx:
                    session = await self._apply_session_context(
                        session, crm_ctx, source="crm", detail="hubspot"
                    )
                await update_session(self.call_id, {"crm_contact_id": contact["id"]})
                logger.info(
                    "CRM contact resolved: %s → %s",
                    self.caller_phone,
                    contact["id"],
                )
            else:
                # Create a stub contact so we can log the call later
                contact_id = create_or_update_contact(self.caller_phone)
                if contact_id:
                    await update_session(
                        self.call_id, {"crm_contact_id": contact_id}
                    )
        except Exception as exc:
            logger.warning("Async CRM lookup failed: %s", exc)
