"""Tests for per-call caller context (name memory, prompt injection)."""
from pipeline.caller_context import (
    apply_context_updates,
    extract_caller_name,
    extract_concern,
    format_context_for_prompt,
    merge_context_from_crm,
    merge_context_from_utterance,
    record_context_event,
)
from prompts.system_prompt import build_system_prompt


def test_extract_name_hinglish_intro():
    assert extract_caller_name("Mera naam Rahul Sharma hai") == "Rahul Sharma"


def test_extract_name_english():
    assert extract_caller_name("My name is Priya Patel") == "Priya Patel"


def test_extract_name_devanagari():
    assert extract_caller_name("मेरा नाम अमित कुमार है") == "अमित कुमार"


def test_extract_name_bol_raha_hoon_not_full_phrase():
    assert extract_caller_name("हां प्रिया मैं जदी बात कर रहा हूं।") == "जदी"


def test_does_not_capture_greeting_as_name():
    assert extract_caller_name("नमस्कार Ma'am.") is None


def test_does_not_capture_uncertain_repeated_intro_name():
    assert extract_caller_name("मैं मैं जरी बात कर रहा हूँ") is None


def test_does_not_capture_agent_name():
    assert extract_caller_name("Priya bol rahi hoon") is None


def test_merge_utterance_only_once():
    session = {"caller_name": "Rahul"}
    assert merge_context_from_utterance(session, "Mera naam Amit hai") == {}


def test_merge_concern():
    updates = merge_context_from_utterance({}, "I am coming for chest pain")
    assert updates.get("caller_concern") == "chest pain"


def test_merge_crm_skips_if_name_known():
    session = {"caller_name": "Rahul"}
    crm = {"properties": {"firstname": "Amit", "lastname": "Kumar"}}
    assert merge_context_from_crm(session, crm) == {}


def test_merge_crm_fills_name():
    crm = {"properties": {"firstname": "Amit", "lastname": "Kumar"}}
    assert merge_context_from_crm({}, crm) == {"caller_name": "Amit Kumar"}


def test_format_context_forbids_reask():
    block = format_context_for_prompt({"caller_name": "Rahul Sharma"})
    assert "Rahul Sharma" in block
    assert "Do NOT ask" in block
    assert "booking" in block.lower()


def test_record_context_event_appends_on_change():
    session = {"context_events": [], "caller_name": None}
    updates, events = apply_context_updates(
        session,
        {"caller_name": "Rahul"},
        source="utterance",
        detail="Mera naam Rahul hai",
    )
    assert updates == {"caller_name": "Rahul"}
    assert len(events) == 1
    assert events[0]["field"] == "caller_name"
    assert events[0]["value"] == "Rahul"
    assert events[0]["source"] == "utterance"


def test_record_context_event_skips_unchanged():
    session = {"context_events": [], "caller_name": "Rahul"}
    updates, events = apply_context_updates(
        session,
        {"caller_name": "Rahul"},
        source="utterance",
        detail="again",
    )
    assert updates == {}
    assert events == []


def test_context_events_cap_at_20():
    session = {
        "context_events": [
            {"ts": 0, "field": "x", "value": "v", "source": "t", "detail": "d"}
        ]
        * 20,
        "caller_name": None,
    }
    _, events = apply_context_updates(
        session, {"caller_name": "A"}, source="utterance", detail="x"
    )
    merged = record_context_event(session, events)
    assert len(merged) == 20
    assert merged[-1]["field"] == "caller_name"


def test_system_prompt_includes_session_context():
    prompt = build_system_prompt(
        caller_phone="+91999",
        session={"caller_name": "Rahul", "caller_concern": "chest pain"},
    )
    assert "Rahul" in prompt
    assert "chest pain" in prompt
    assert "Do NOT ask" in prompt


def test_thik_not_captured_as_name():
    # STT often transcribes "ठीक है" as Roman "thik hai" — must not become caller name
    assert extract_caller_name("thik hai") is None


def test_theek_hai_not_captured_as_name():
    assert extract_caller_name("theek hai") is None


def test_hai_alone_not_captured_as_name():
    assert extract_caller_name("hai") is None


def test_hain_not_captured_as_name():
    assert extract_caller_name("hain") is None


def test_thik_chalega_not_captured_as_name():
    assert extract_caller_name("thik chalega") is None
