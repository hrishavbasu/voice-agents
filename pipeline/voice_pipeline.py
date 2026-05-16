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
        "मैं अभी देखती हूँ।",
        "बस एक सेकंड।",
        "मैं चेक कर रही हूँ।",
        "थोड़ा सा इंतज़ार करें।",
        "हाँ, मैं अभी पता करती हूँ।",
    ],
    "hinglish": [
        "एक second, मैं check करती हूँ।",
        "बस एक पल।",
        "मैं देखती हूँ।",
        "Just a second.",
        "Let me check करती हूँ।",
        "हाँ, मैं अभी देख लेती हूँ।",
    ],
}
_BARGE_IN_ACK: dict = {
    "hindi":    ["haan, bataiye", "ji, haan?", "haan ji?"],
    "hinglish": ["haan, bataiye", "yes, bataiye?", "haan ji?"],
    "english":  ["yes, go ahead", "please go on"],
}
# Keywords that require an immediate hard emergency response before LLM.
# Language-inclusive: covers English + common Hindi/Hinglish equivalents.
# Matched with word boundaries via re — avoids substring false positives
# (e.g. "help me with booking" must NOT fire; "chest pain" must).
_EMERGENCY_KEYWORDS = frozenset({
    "emergency", "chest pain", "heart attack", "stroke",
    "bleeding", "unconscious", "can't breathe", "cant breathe",
    "not breathing", "seizure", "overdose", "bahut bura", "bachao",
    "ambulance",
})

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

        # Queue of final STT transcripts waiting to be processed
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._last_activity_at: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise session, STT connection, greet caller, start loops."""
        self._running = True

        # Create session
        await create_session(self.call_id, self.caller_phone)

        # Async CRM lookup (don't block greeting)
        asyncio.create_task(self._async_crm_lookup())

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

        # Play greeting
        greeting = self._build_greeting()
        await self._speak(greeting)
        await append_message(self.call_id, "assistant", greeting)

        # Start the LLM processing loop
        asyncio.create_task(self._llm_loop())
        self._last_activity_at = time.monotonic()
        asyncio.create_task(self._silence_watch_loop())

    async def shutdown(self) -> None:
        """Cleanly close STT connection and end session."""
        self._running = False
        if self._stt:
            await self._stt.close()
        await end_session(self.call_id)
        logger.info("Pipeline shutdown for call_id=%s", self.call_id)

    # ── Audio ingestion (called by telephony receive_loop) ────────────────────

    async def on_audio(self, chunk: bytes) -> None:
        """Forward raw audio from the carrier to Deepgram STT."""
        if self._stt and self._running:
            await self._stt.send_audio(chunk)

    # ── STT callbacks ─────────────────────────────────────────────────────────

    async def _on_transcript(self, text: str, is_final: bool) -> None:
        self._last_activity_at = time.monotonic()
        if is_final and text.strip():
            logger.info("[CALLER] %s", text)
            await self._transcript_queue.put(text)

    async def _on_speech_started(self) -> None:
        """Barge-in: cancel TTS only if agent is currently speaking."""
        speaking_or_buffered = (
            self._tts_playing
            or self._agent_in_turn
            or (time.monotonic() < self._playback_until)
        )
        if not speaking_or_buffered:
            return
        sentences_spoken = getattr(self, "_sentences_spoken_this_turn", 0)
        self._tts.cancel()
        await self._interruption.trigger()
        await self._telephony.clear_playback_buffer()
        # Discard transcripts queued during TTS playback (stale/echo fragments)
        while not self._transcript_queue.empty():
            self._transcript_queue.get_nowait()
        # Tail silence clears handset jitter after a hard barge-in cut.
        await self._telephony.send_silence(80)
        # Acknowledge the interruption if agent had already spoken ≥ 1 sentence —
        # avoids acknowledging coughs or noise that fire before any speech.
        if sentences_spoken >= 1:
            lang = self._caller_language if self._caller_language in _BARGE_IN_ACK else "hinglish"
            ack = random.choice(_BARGE_IN_ACK[lang])
            self._interruption.reset()
            await self._speak(ack)

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

            await append_message(self.call_id, "user", user_text)

            # Hard emergency intercept — fires BEFORE LLM, deterministic.
            # LLM-based detection is not reliable enough for healthcare.
            if await self._check_emergency(user_text):
                continue

            # Detect caller language and persist — first clear detection wins;
            # subsequent turns can shift hinglish→english/hindi but not overwrite
            # a firm detection unless the ratio is unambiguous (>60%).
            detected = _detect_language(user_text)
            session = await get_session(self.call_id)
            current_lang = (session or {}).get("caller_language")

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
            )
            # Keep last 10 messages only — reduces tokens per request by ~40%
            full_messages = [{"role": "system", "content": system}] + messages[-10:]

            # Stream LLM response
            assistant_text_parts = []
            self._agent_in_turn = True
            self._sentences_spoken_this_turn = 0
            self._interruption.reset()
            self._tts.reset()
            correction_text: str | None = None

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
                    # Sentence — speak it and record it
                    assistant_text_parts.append(item)
                    self._sentences_spoken_this_turn += 1
                    await self._speak(item)

                elif isinstance(item, dict) and item.get("type") == "tool_call":
                    await self._handle_tool_call(item)

            self._agent_in_turn = False
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
    def _normalize_for_tts(text: str) -> str:
        """Pre-process text so TTS engines don't read symbols or digits unnaturally."""
        # ₹1,700 → "rupees 1700" (comma stripped so TTS reads as integer)
        # Keep as English fallback; LLM should ideally produce word-form already.
        text = re.sub(r"₹\s*([\d,]+)", lambda m: "rupees " + m.group(1).replace(",", ""), text)
        # Bare ₹ without number
        text = text.replace("₹", "rupees")
        # Strip thousand-separator commas from standalone numbers so "1,700" → "1700"
        text = re.sub(r"\b(\d{1,3}),(\d{3})\b", r"\1\2", text)
        # Expand Devanagari abbreviations that TTS reads unnaturally
        text = text.replace("डॉ.", "डॉक्टर")
        # Devanagari substitutions for Latin-script proper nouns Tripti mispronounces
        for original, replacement in COMPANY_CONFIG.get("tts_substitutions", {}).items():
            text = text.replace(original, replacement)
        text = _strip_markdown(text)
        text = _add_prosody_markers(text)
        return text

    async def _speak(self, text: str, pitch_override: float = None) -> float:
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
            async for chunk in self._tts.synthesize(text, language_code=lang_code, pitch_override=_pitch_bump):
                if self._interruption.is_interrupted:
                    break
                buf.extend(chunk)
                while len(buf) >= _FRAME:
                    if self._interruption.is_interrupted:
                        break
                    frame = bytes(buf[:_FRAME])
                    buf = buf[_FRAME:]
                    total_bytes += len(frame)
                    await self._telephony.send_audio(frame)
                if self._interruption.is_interrupted:
                    break
            # Flush remainder padded with μ-law silence (0xFF)
            if buf and not self._interruption.is_interrupted:
                padded = bytes(buf) + bytes([0xFF] * (_FRAME - len(buf) % _FRAME))
                total_bytes += len(buf)
                await self._telephony.send_audio(padded)
        finally:
            self._tts_playing = False
            self._last_activity_at = time.monotonic()
        duration = total_bytes / 8000
        self._playback_until = time.monotonic() + duration + 0.25
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
                result = check_doctor_slots(
                    doctor_name=args.get("doctor_name", ""),
                    preferred_date=args.get("preferred_date"),
                )

            elif name == "list_doctors":
                from tools.scheduling import list_doctors
                result = list_doctors(specialty=args.get("specialty"))

            elif name == "book_appointment":
                from tools.scheduling import book_appointment
                result = book_appointment(
                    caller_phone=self.caller_phone,
                    patient_name=args.get("patient_name", ""),
                    concern=args.get("concern", ""),
                    doctor_name=args.get("doctor_name"),
                    specialty=args.get("specialty"),
                    preferred_date=args.get("preferred_date"),
                    preferred_time=args.get("preferred_time"),
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
        full_messages = (
            [{"role": "system", "content": system}]
            + messages[-10:]
            + [assistant_tool_call_msg, tool_result_msg]
        )

        response_parts = []
        async for item in stream_response(full_messages, tools=self._tool_schemas):
            if self._interruption.is_interrupted:
                break
            if isinstance(item, str):
                response_parts.append(item)
                await self._speak(item)
            elif isinstance(item, dict) and item.get("type") == "tool_call":
                # LLM chained another tool call (e.g. check_slots → book)
                await self._handle_tool_call(item)

        if response_parts:
            await append_message(
                self.call_id, "assistant", " ".join(response_parts)
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _check_emergency(self, text: str) -> bool:
        """Pre-LLM keyword scan for medical emergencies.

        Returns True and triggers immediate 108 prompt + escalation if any
        emergency keyword is detected.  This path is deterministic and does
        NOT route through the LLM, so it fires even under high LLM latency.
        """
        lower = text.lower()
        matched = any(
            re.search(r"\b" + re.escape(kw) + r"\b", lower)
            for kw in _EMERGENCY_KEYWORDS
        )
        if not matched:
            return False

        logger.warning("EMERGENCY keyword detected in: %r — bypassing LLM", text)

        # Speak 108 advisory immediately — no confirmation step
        self._interruption.reset()
        duration = await self._speak(
            "If this is a medical emergency please call one zero eight right now. "
            "I'm connecting you to our team immediately. "
            "अगर यह कोई मेडिकल इमरजेंसी है, तो कृपया अभी एक शून्य आठ पर कॉल करें।"
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
            msg = "Your appointment is confirmed. Thank you for calling Apollo Hospitals. Have a good day."
        elif lang == "hindi":
            msg = "आपकी अपॉइंटमेंट कन्फर्म हो गई है। Apollo Hospitals पर कॉल करने के लिए धन्यवाद। आपका दिन शुभ हो।"
        else:
            msg = "Aapki appointment confirm ho gayi hai. Apollo Hospitals ko call karne ke liye dhanyavaad."
        self._interruption.reset()
        duration = await self._speak(msg)
        await asyncio.sleep(duration + 0.2)
        await self._telephony.hangup()
        self._running = False

    async def _silence_watch_loop(self) -> None:
        """Prompt 'are you still there?' after inactivity, then hang up."""
        while self._running:
            await asyncio.sleep(1.0)
            if self._tts_playing or self._agent_in_turn or not self._running:
                continue
            idle = time.monotonic() - self._last_activity_at
            if idle < SILENCE_TIMEOUT_SECS:
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

    def _build_greeting(self) -> str:
        cfg = COMPANY_CONFIG
        company = cfg.get("company_name", "our hospital")
        agent = cfg.get("agent_name", "Priya")
        return (
            f"नमस्ते, {company} में आपका स्वागत है। "
            f"मैं {agent} हूँ। "
            f"How may I assist you today?"
        )

    async def _async_crm_lookup(self) -> None:
        """Non-blocking CRM lookup — resolves in background after call starts."""
        try:
            contact = lookup_contact_by_phone(self.caller_phone)
            if contact:
                self._crm_contact = contact
                await update_session(
                    self.call_id,
                    {"crm_contact_id": contact["id"]},
                )
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
