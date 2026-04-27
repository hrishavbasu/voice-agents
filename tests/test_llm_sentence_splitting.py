"""Tests for LLM sentence-splitting regex (services/llm.py)."""

import re
import pytest

# Import the compiled regex directly — no network calls triggered by this import
# because the module-level client construction uses the fake keys from conftest.
from services.llm import _SENTENCE_RE


def split(text: str) -> list[str]:
    """Helper that mirrors what stream_response does with the buffer."""
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def test_splits_on_period_space():
    result = split("Hello there. How can I help you?")
    assert result == ["Hello there.", "How can I help you?"]


def test_splits_on_exclamation():
    result = split("Great! Let me check that for you.")
    assert result == ["Great!", "Let me check that for you."]


def test_splits_on_question_mark():
    result = split("Is that right? Let me confirm.")
    assert result == ["Is that right?", "Let me confirm."]


def test_multiple_sentences():
    result = split("One. Two. Three.")
    assert result == ["One.", "Two.", "Three."]


def test_no_split_after_dr():
    result = split("Please see Dr. Anuj Sathe for your appointment.")
    assert len(result) == 1


def test_no_split_after_mr():
    result = split("Mr. Sharma will meet you shortly.")
    assert len(result) == 1


def test_no_split_after_mrs():
    result = split("Mrs. Patel called earlier. She will call back.")
    assert result[0].startswith("Mrs. Patel called earlier.")


def test_no_split_after_ms():
    result = split("Ms. Rao is available at 3pm. Please confirm.")
    assert result[0].startswith("Ms. Rao is available")


def test_no_split_inside_word():
    result = split("The cost is Rs.500 only.")
    # Should remain as a single sentence (no space after the period)
    assert len(result) == 1


def test_hindi_devanagari_sentence():
    # Devanagari uses । as sentence terminator; our regex only splits on .!?
    # so a single Hindi sentence stays whole
    result = split("नमस्ते। मैं आपकी कैसे मदद कर सकती हूँ।")
    assert len(result) == 1


def test_empty_string():
    assert split("") == []


def test_single_sentence_no_split():
    result = split("Hello how are you")
    assert result == ["Hello how are you"]


def test_trailing_space_handled():
    result = split("First sentence. ")
    assert result == ["First sentence."]
