"""
Tests for build_system_prompt — verifies the LLM receives correct
instructions for human-like, clinically-accurate responses.
"""

import pytest
from prompts.system_prompt import build_system_prompt


@pytest.fixture
def prompt():
    return build_system_prompt(caller_phone="+919999999999")


def test_agent_name_in_prompt(prompt):
    assert "Priya" in prompt


def test_company_name_in_prompt(prompt):
    assert "Apollo" in prompt


def test_no_ai_disclosure_rule(prompt):
    assert "I am an AI" in prompt or "never say" in prompt.lower() or "never" in prompt.lower()
    # Specifically the anti-rule must be present
    assert "I am an AI" in prompt


def test_language_mirror_rule(prompt):
    assert "Mirror" in prompt or "mirror" in prompt


def test_hindi_devanagari_rule(prompt):
    # Must instruct Devanagari for Hindi words
    assert "Devanagari" in prompt or "DEVANAGARI" in prompt


def test_no_bullet_points_rule(prompt):
    assert "bullet" in prompt.lower() or "markdown" in prompt.lower()


def test_two_to_three_sentences_guideline(prompt):
    assert "2-3 sentences" in prompt or "two" in prompt.lower()


def test_booking_flow_8_steps_present(prompt):
    # Booking flow must have all 8 steps
    assert "full name" in prompt.lower() or "patient" in prompt.lower()
    assert "concern" in prompt.lower() or "symptoms" in prompt.lower()
    assert "check_doctor_slots" in prompt
    assert "book_appointment" in prompt


def test_tool_call_before_availability_statement(prompt):
    assert "check_doctor_slots" in prompt
    # Must warn against hallucinating availability
    assert "hallucination" in prompt.lower() or "zero knowledge" in prompt.lower() or "nothing without" in prompt.lower()


def test_emergency_handling_in_prompt(prompt):
    assert "108" in prompt
    assert "emergency" in prompt.lower()


def test_currency_spoken_words_rule(prompt):
    # Must forbid ₹ digit format in speech
    assert "₹" in prompt
    assert "rupee" in prompt.lower() or "rupees" in prompt.lower()


def test_time_spoken_words_rule(prompt):
    # Must forbid digit-colon time format in speech
    assert "9:00 AM" in prompt or "digit" in prompt.lower()
    assert "morning" in prompt.lower()


def test_escalation_rule_present(prompt):
    assert "escalate" in prompt.lower() or "escalation" in prompt.lower()


def test_no_robotic_phrases_rule(prompt):
    assert "robotic" in prompt.lower() or "IVR" in prompt or "anti-robotic" in prompt.lower()


def test_doctor_name_tool_argument_rule(prompt):
    # Must instruct passing English doctor names to tools
    assert "TOOL ARGUMENT" in prompt or "tool argument" in prompt.lower()


def test_caller_phone_injected(prompt):
    assert "+919999999999" in prompt


def test_crm_contact_name_injected():
    crm = {"properties": {"firstname": "Rahul", "lastname": "Sharma"}}
    p = build_system_prompt(caller_phone="+91111", crm_contact=crm)
    assert "Rahul Sharma" in p


def test_language_directive_english():
    p = build_system_prompt(caller_language="english")
    assert "ENGLISH" in p
    assert "Do NOT use Hindi" in p


def test_language_directive_hindi():
    p = build_system_prompt(caller_language="hindi")
    assert "HINDI" in p
    assert "DEVANAGARI" in p


def test_language_directive_hinglish():
    p = build_system_prompt(caller_language="hinglish")
    assert "HINGLISH" in p


def test_out_of_scope_escalation_list(prompt):
    assert "Insurance" in prompt or "insurance" in prompt.lower()
    assert "prescription" in prompt.lower() or "Prescription" in prompt
    assert "medical advice" in prompt.lower()
