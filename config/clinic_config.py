"""
Apollo Hospitals Navi Mumbai — clinic voice agent configuration.

Imported by config/company_config.py so all existing module imports
(from config.company_config import COMPANY_CONFIG) continue to work unchanged.
"""

COMPANY_CONFIG: dict = {
    # ── Identity ──────────────────────────────────────────────────────────────
    "company_name": "Apollo Hospitals Navi Mumbai",
    # TTS pronunciation entries — IPA phonemes for Latin-script proper nouns.
    # Used by ElevenLabs Pronunciation Dictionary (created at startup).
    "tts_pronunciation_entries": {
        "Navi": "nʌvi",
    },
    # Devanagari substitutions — applied BEFORE sending text to TTS.
    # Tripti pronounces Devanagari/Marathi natively; Latin script triggers long-ā.
    # More reliable than IPA dictionary for proper nouns in Indian names.
    "tts_substitutions": {
        "Navi Mumbai": "नवी मुंबई",
    },
    "support_phone": "+916366530173",
    "agent_name": "Priya",
    "persona": (
        "warm, professional, and caring — like a trained clinic receptionist. "
        "You speak naturally in English, Hindi, or Hinglish based on caller preference. "
        "You use short spoken sentences, maximum two per turn. "
        "You vary acknowledgments — never repeat the same filler. "
        "You never say 'I am an AI' or 'I am a bot'. "
        "If unsure, say 'Let me check that for you.' "
        "You are empathetic — callers may be unwell or anxious. "
        "Greet with warmth and make the caller feel heard."
    ),

    # ── Escalation ────────────────────────────────────────────────────────────
    "escalation_phone": "+919622883922",   # Your number for human escalation
    "max_retry_before_escalate": 2,

    # ── CRM ───────────────────────────────────────────────────────────────────
    "crm_type": "hubspot",
    "crm_credentials": {},

    # ── Tools enabled ─────────────────────────────────────────────────────────
    "tools_enabled": [
        "check_doctor_slots",
        "list_doctors",
        "book_appointment",
        "escalate_to_human",
    ],

    # ── Knowledge base ────────────────────────────────────────────────────────
    "troubleshooting_kb": "kb/apollo_clinic_kb.md",

    # ── Voice / TTS overrides (leave blank — use ELEVENLABS_VOICE_ID env var) ─
    "tts_voice_id": "",

    # ── Business hours (IST — Asia/Kolkata) ──────────────────────────────────
    "timezone": "Asia/Kolkata",
    "business_hours": {
        "monday":    {"open": "09:00", "close": "19:00"},
        "tuesday":   {"open": "09:00", "close": "19:00"},
        "wednesday": {"open": "09:00", "close": "19:00"},
        "thursday":  {"open": "09:00", "close": "19:00"},
        "friday":    {"open": "09:00", "close": "19:00"},
        "saturday":  {"open": "09:00", "close": "17:00"},
        "sunday":    None,
    },

    # ── Appointment slot duration (minutes) ───────────────────────────────────
    "appointment_slot_duration_minutes": 30,

    # ── Doctors ───────────────────────────────────────────────────────────────
    # Each entry drives both the KB and the list_doctors tool response.
    "doctors": [
        {
            "name": "Dr. S V Kulkarni",
            "specialty": "Internal Medicine",
            "experience_years": 46,
            "fee_inr": 1700,
            "available_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
        },
        {
            "name": "Dr. Aabha Nagral",
            "specialty": "Gastroenterology",
            "experience_years": 39,
            "fee_inr": 2000,
            "available_days": ["monday", "wednesday", "friday"],
        },
        {
            "name": "Dr. Shantesh D. Kaushik",
            "specialty": "Cardiothoracic & Vascular Surgery",
            "experience_years": 39,
            "fee_inr": 2300,
            "available_days": ["tuesday", "thursday", "saturday"],
        },
        {
            "name": "Dr. Atul Bhaskar",
            "specialty": "Orthopedic Surgery",
            "experience_years": 34,
            "fee_inr": 2500,
            "available_days": ["monday", "tuesday", "thursday", "friday"],
        },
        {
            "name": "Dr. Atul Seth",
            "specialty": "Ophthalmology",
            "experience_years": 30,
            "fee_inr": 2000,
            "available_days": ["monday", "wednesday", "friday"],
        },
        {
            "name": "Dr. Naresh Biyani",
            "specialty": "Neurosurgery",
            "qualification": "MBBS, MCh",
            "experience_years": 28,
            "fee_inr": 3000,
            "available_days": ["tuesday", "thursday"],
        },
        {
            "name": "Dr. Jyoti Bajpai",
            "specialty": "Medical Oncology",
            "experience_years": 27,
            "fee_inr": 2300,
            "available_days": ["monday", "wednesday", "friday"],
        },
        {
            "name": "Dr. Anuj Sathe",
            "specialty": "Cardiology",
            "qualification": "MBBS, MD, DM",
            "experience_years": 17,
            "fee_inr": 2000,
            "available_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
        },
        {
            "name": "Dr. Nilesh Doctor",
            "specialty": "Surgical Gastroenterology",
            "qualification": "MBBS, MS, DNB",
            "experience_years": 32,
            "fee_inr": 1500,
            "available_days": ["monday", "thursday", "saturday"],
        },
        {
            "name": "Dr. Bhushan Chavan",
            "specialty": "Pediatric Cardiology",
            "qualification": "MBBS, DNB, Fellowship",
            "experience_years": 17,
            "fee_inr": 1800,
            "available_days": ["tuesday", "friday"],
        },
    ],
}
