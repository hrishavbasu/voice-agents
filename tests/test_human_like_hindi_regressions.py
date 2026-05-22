"""Human-like Hindi conversation regressions from recent live calls."""

from pipeline.voice_pipeline import VoicePipeline
from services.llm_gemini import _clause_split


def test_name_prompt_is_gender_neutral():
    out = VoicePipeline._normalize_for_tts("क्या आप अपना नाम बता सकती हैं?")
    assert out == "कृपया अपना नाम बता दें?"


def test_slot_followup_avoids_gendered_ending():
    out = VoicePipeline._normalize_for_tts("क्या आप उनके लिए कोई और समय देखना चाहेंगी?")
    assert out == "क्या उनके लिए कोई और समय देखें?"


def test_hindi_sentence_not_over_split_on_commas():
    sentence = "जयदीप, रात के दस बजे का appointment नहीं मिल पाएगा, लेकिन हम कोई और समय देख सकते हैं।"
    parts = _clause_split(sentence)
    assert parts == [sentence]


def test_redundant_concern_question_removed_when_specialty_known():
    session = {"preferred_specialty": "General Physician", "preferred_doctor": "Dr. S V Kulkarni"}
    out = VoicePipeline._postprocess_agent_text(
        "और आपकी परेशानी क्या है? किसलिए आप डॉक्टर से मिलना चाहते हैं?",
        session,
    )
    assert out == "ठीक है, हम appointment आगे बढ़ाते हैं।"


def test_humanize_generic_prompt_fragment():
    out = VoicePipeline._normalize_for_tts("जी, बताइए?")
    assert out == "जी, बताइए?"


def test_redundant_concern_prompt_without_doctor_keyword_removed():
    session = {"preferred_doctor": "Dr. S V Kulkarni"}
    out = VoicePipeline._postprocess_agent_text(
        "और आपको क्या परेशानी है, बताएँगे?",
        session,
    )
    assert out == "ठीक है, हम appointment आगे बढ़ाते हैं।"


def test_no_malformed_qahege_suffix_after_neutralization():
    out = VoicePipeline._normalize_for_tts("क्या आप किसी और समय या किसी और दिन आना चाहेंगे?")
    assert "गे?" not in out.replace("ठीक रहेगा?", "")
    assert out == "क्या आप किसी और समय या किसी और दिन आना ठीक रहेगा?"


def test_generic_appointment_see_time_question_is_neutralized():
    out = VoicePipeline._normalize_for_tts(
        "क्या आप किसी और समय या दिन के लिए appointment देखना चाहेंगी?"
    )
    assert out == "क्या आप किसी और समय या दिन के लिए appointment देखना चाहें?"
