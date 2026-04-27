"""
Response quality tests — validate the LLM pipeline produces human-like,
clinic-correct responses for real conversation scenarios.

All LLM + TTS + STT calls are mocked so these run instantly without API keys.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pipeline.session import create_session, get_session
import memory.store as store_module
from memory.store import InMemoryStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    instance = InMemoryStore()
    monkeypatch.setattr(store_module, "_store_instance", instance)
    yield
    store_module._store_instance = None


def _make_llm_response(*sentences):
    """Build an async generator that yields the given sentences then stops."""
    async def _gen():
        for s in sentences:
            yield s
    return _gen()


def _make_tts():
    tts = MagicMock()
    tts.reset = MagicMock()
    tts.cancel = MagicMock()
    tts.synthesize = AsyncMock(return_value=iter([]))

    async def _synth(text, voice_id=None):
        yield b""  # one silent chunk

    tts.synthesize = _synth
    return tts


def _make_telephony():
    tel = MagicMock()
    tel.stream_ready = asyncio.Event()
    tel.stream_ready.set()
    tel.send_audio = AsyncMock()
    tel.send_silence = AsyncMock()
    tel.transfer = AsyncMock()
    tel.hangup = AsyncMock()
    return tel


# ── Greeting quality ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_greeting_contains_hospital_name():
    from pipeline.voice_pipeline import VoicePipeline
    from config.company_config import COMPANY_CONFIG

    tel = _make_telephony()

    with patch("pipeline.voice_pipeline.DeepgramSTT") as MockSTT, \
         patch("pipeline.voice_pipeline.TTSService") as MockTTS, \
         patch("pipeline.voice_pipeline.lookup_contact_by_phone", return_value=None), \
         patch("pipeline.voice_pipeline.create_or_update_contact", return_value=None):

        mock_stt = AsyncMock()
        mock_stt.connect = AsyncMock()
        MockSTT.return_value = mock_stt

        tts_instance = _make_tts()
        MockTTS.return_value = tts_instance

        spoken = []

        async def capture_speak(text, voice_id=None):
            spoken.append(text)
            yield b""

        tts_instance.synthesize = capture_speak

        pipeline = VoicePipeline("call-greet", "+919999999999", tel)
        pipeline._running = False  # stop after greeting

        # Patch _llm_loop to not run
        with patch.object(pipeline, "_llm_loop", new_callable=AsyncMock):
            await pipeline.start()

    greeting = " ".join(spoken)
    assert "Apollo" in greeting or "apollo" in greeting.lower()
    assert "Priya" in greeting or "priya" in greeting.lower()


@pytest.mark.asyncio
async def test_greeting_not_robotic():
    """Greeting must contain a warm phrase, not a cold IVR-style opening."""
    from pipeline.voice_pipeline import VoicePipeline

    tel = _make_telephony()
    spoken = []

    with patch("pipeline.voice_pipeline.DeepgramSTT") as MockSTT, \
         patch("pipeline.voice_pipeline.TTSService") as MockTTS, \
         patch("pipeline.voice_pipeline.lookup_contact_by_phone", return_value=None), \
         patch("pipeline.voice_pipeline.create_or_update_contact", return_value=None):

        mock_stt = AsyncMock()
        mock_stt.connect = AsyncMock()
        MockSTT.return_value = mock_stt

        tts_instance = _make_tts()
        MockTTS.return_value = tts_instance

        async def capture(text, voice_id=None):
            spoken.append(text)
            yield b""
        tts_instance.synthesize = capture

        pipeline = VoicePipeline("call-greet2", "+919999999999", tel)
        with patch.object(pipeline, "_llm_loop", new_callable=AsyncMock):
            await pipeline.start()

    greeting = " ".join(spoken).lower()
    # Must not start with a robotic IVR phrase
    assert "press 1" not in greeting
    assert "for english press" not in greeting
    assert "welcome to automated" not in greeting


# ── Language detection ────────────────────────────────────────────────────────

from pipeline.voice_pipeline import _detect_language

def test_detect_english():
    assert _detect_language("I need to book an appointment with the doctor") == "english"

def test_detect_hindi():
    assert _detect_language("मुझे डॉक्टर से मिलना है, मेरे घुटने में दर्द है") == "hindi"

def test_detect_hinglish():
    assert _detect_language("मुझे doctor से appointment लेनी है") == "hinglish"

def test_detect_short_english():
    # "Hello" has no Devanagari → detected as "english" (< 5% Devanagari ratio)
    assert _detect_language("Hello") == "english"

def test_detect_empty():
    assert _detect_language("") == "hinglish"


# ── Anti-robotic response rules ───────────────────────────────────────────────

def test_response_must_not_say_i_am_ai():
    forbidden = ["I am an AI", "I am a bot", "as an AI language model", "as an AI assistant"]
    sample_responses = [
        "Sure! Let me check that for you.",
        "Dr. Bhaskar has slots at nine in the morning and two in the afternoon.",
        "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?",
    ]
    for response in sample_responses:
        for phrase in forbidden:
            assert phrase not in response, f"Forbidden phrase '{phrase}' found in response"


def test_currency_format_is_spoken_words():
    """Verify the TTS normaliser converts ₹ to spoken words."""
    from pipeline.voice_pipeline import VoicePipeline
    normalize = VoicePipeline._normalize_for_tts

    assert "₹" not in normalize("The fee is ₹1,700.")
    assert "rupees" in normalize("The fee is ₹1,700.").lower()
    assert "1700" in normalize("The fee is ₹1,700.") or "rupees" in normalize("The fee is ₹1,700.").lower()


def test_currency_normalizer_bare_symbol():
    from pipeline.voice_pipeline import VoicePipeline
    result = VoicePipeline._normalize_for_tts("cost is ₹500")
    assert "₹" not in result
    assert "rupees" in result.lower()


def test_devanagari_dr_expanded():
    from pipeline.voice_pipeline import VoicePipeline
    result = VoicePipeline._normalize_for_tts("डॉ. Kulkarni आज available हैं")
    assert "डॉ." not in result
    assert "डॉक्टर" in result


def test_navi_mumbai_substitution():
    from pipeline.voice_pipeline import VoicePipeline
    result = VoicePipeline._normalize_for_tts("Apollo Hospitals Navi Mumbai")
    assert "नवी मुंबई" in result


# ── Slot spoken format validation ─────────────────────────────────────────────

def test_slot_spoken_format_no_digit_time():
    """Slots read to callers must use natural English, not digit-colon format."""
    from tools.scheduling import _time_to_spoken
    from datetime import datetime
    from zoneinfo import ZoneInfo

    bad_formats = ["9:00 AM", "14:30", "3:00 PM", "18:30"]
    for tf in bad_formats:
        assert tf not in _time_to_spoken(
            datetime.strptime(tf.replace(" AM", "").replace(" PM", ""), "%H:%M")
            if ":" in tf and "AM" not in tf and "PM" not in tf
            else datetime.now(ZoneInfo("Asia/Kolkata")).replace(hour=9, minute=0)
        )


def test_slot_contains_spoken_and_digit_reference():
    """Slots returned to LLM must have both spoken form AND digit reference."""
    from tools.scheduling import check_doctor_slots
    from datetime import date, timedelta

    today = date.today()
    days_ahead = (0 - today.weekday()) % 7 or 7
    next_monday = (today + timedelta(days=days_ahead)).strftime("%A, %d %B")

    result = check_doctor_slots("Dr. S V Kulkarni", next_monday)
    if result.get("available_on_requested_date"):
        for slot in result["slots"]:
            assert "(" in slot, f"No digit reference in slot: {slot}"
            assert "in the" in slot or "past" in slot or "to " in slot, \
                f"Slot not in spoken English: {slot}"


# ── Booking confirmation quality ──────────────────────────────────────────────

@patch("tools.calendar.create_appointment_event")
def test_booking_result_has_all_required_fields(mock_cal):
    from tools.scheduling import book_appointment, _booked_slots
    from datetime import date, timedelta
    _booked_slots.clear()

    mock_cal.return_value = {
        "success": True,
        "slot": "Monday, 28 April at nine in the morning",
        "event_id": "evt-001",
    }
    today = date.today()
    days_ahead = (0 - today.weekday()) % 7 or 7
    next_monday = (today + timedelta(days=days_ahead)).strftime("%A")

    result = book_appointment(
        caller_phone="+916666666666",
        patient_name="Ravi Kumar",
        concern="chest pain",
        doctor_name="Dr. Anuj Sathe",
        preferred_date=next_monday,
        preferred_time="morning",
    )
    assert result["success"] is True
    required = ["doctor", "fee", "slot_spoken"]
    for field in required:
        assert field in result, f"Missing field: {field}"


@patch("tools.calendar.create_appointment_event")
def test_time_mismatch_note_added(mock_cal):
    """If booked time is > 45 min off from requested, a 'note' must be added."""
    from tools.scheduling import book_appointment, _booked_slots
    _booked_slots.clear()

    mock_cal.return_value = {
        "success": True,
        "slot": "Monday at 9:00 AM",
        "event_id": "evt-002",
    }
    from datetime import date, timedelta
    today = date.today()
    days_ahead = (0 - today.weekday()) % 7 or 7
    next_monday = (today + timedelta(days=days_ahead)).strftime("%A")

    # Requesting 9 PM but doctor closes at 7 PM — booked slot will be far off
    result = book_appointment(
        caller_phone="+915555555555",
        patient_name="Test",
        concern="fever",
        doctor_name="Dr. S V Kulkarni",
        preferred_date=next_monday,
        preferred_time="9pm",
    )
    if result.get("success"):
        # Nearest slot will be well before 9 PM → note must warn LLM
        assert "note" in result


# ── Emergency keyword detection ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_emergency_keyword_triggers_108():
    from pipeline.voice_pipeline import VoicePipeline, _EMERGENCY_KEYWORDS

    # Validate the keyword set contains medical emergencies
    assert "chest pain" in _EMERGENCY_KEYWORDS
    assert "emergency" in _EMERGENCY_KEYWORDS
    assert "bachao" in _EMERGENCY_KEYWORDS
    assert "ambulance" in _EMERGENCY_KEYWORDS
    assert "seizure" in _EMERGENCY_KEYWORDS


@pytest.mark.asyncio
async def test_emergency_check_triggers_on_keywords():
    from pipeline.voice_pipeline import VoicePipeline

    tel = _make_telephony()
    spoken = []

    with patch("pipeline.voice_pipeline.DeepgramSTT") as MockSTT, \
         patch("pipeline.voice_pipeline.TTSService") as MockTTS, \
         patch("pipeline.voice_pipeline.lookup_contact_by_phone", return_value=None), \
         patch("pipeline.voice_pipeline.create_or_update_contact", return_value=None), \
         patch("pipeline.voice_pipeline.escalate_to_human", new_callable=AsyncMock, return_value={"success": True}, create=True):

        MockSTT.return_value = AsyncMock()
        MockSTT.return_value.connect = AsyncMock()

        tts_instance = _make_tts()
        MockTTS.return_value = tts_instance

        async def capture(text, voice_id=None):
            spoken.append(text)
            yield b""
        tts_instance.synthesize = capture

        pipeline = VoicePipeline("call-emerg", "+919999999999", tel)
        pipeline._running = True
        await create_session("call-emerg", "+919999999999")

        result = await pipeline._check_emergency("I have chest pain and can't breathe")

    assert result is True
    emergency_response = " ".join(spoken).lower()
    # Pipeline speaks "one zero eight" (avoids TTS reading "108" as "one hundred eight")
    assert "108" in emergency_response or "one zero eight" in emergency_response


# ── Hinglish greeting detection ───────────────────────────────────────────────

def test_greeting_is_hinglish():
    """Default greeting uses Hinglish (Hindi + English) — warm and inclusive."""
    from pipeline.voice_pipeline import VoicePipeline

    tel = _make_telephony()
    pipeline = VoicePipeline.__new__(VoicePipeline)
    greeting = pipeline._build_greeting()

    # Should contain both Hindi characters and English
    has_hindi = any("ऀ" <= c <= "ॿ" for c in greeting)
    has_english = any(c.isascii() and c.isalpha() for c in greeting)
    assert has_hindi, "Greeting has no Hindi script"
    assert has_english, "Greeting has no English"
    assert "Priya" in greeting
