"""Regression tests derived from real call logs for human-like quality."""

from pipeline.caller_context import extract_caller_name
from pipeline.voice_pipeline import VoicePipeline


def test_log_regression_name_not_captured_from_greeting_honorific():
    # From CA98...: "नमस्कार Ma'am." was incorrectly saved as caller_name.
    assert extract_caller_name("नमस्कार Ma'am.") is None


def test_log_regression_maam_removed_from_spoken_lines():
    # From CA98...: "आपको किस दिन appointment चाहिए Ma'am?"
    out = VoicePipeline._normalize_for_tts("आपको किस दिन appointment चाहिए Ma'am?")
    assert "Ma'am" not in out
    assert "Madam" not in out
    assert "मैम" not in out


def test_log_regression_awkward_neutral_phrase_is_humanized():
    # From recent output after neutralization patch:
    # "क्या आप उनसे appointment लेना ठीक रहेगा?" sounds synthetic.
    out = VoicePipeline._normalize_for_tts("क्या आप उनसे appointment लेना ठीक रहेगा?")
    assert out == "क्या हम उनके साथ appointment रखें?"


def test_log_regression_no_gendered_second_person_suffixes():
    # Ensure caller-directed suffixes from old turns are removed.
    out = VoicePipeline._normalize_for_tts("क्या आप सोमवार को आना चाहेंगे?")
    assert "चाहेंगे" not in out
    assert "चाहेंगी" not in out
