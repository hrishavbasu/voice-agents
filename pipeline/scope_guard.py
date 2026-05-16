"""Out-of-scope topic detection and decline/escalate messaging."""
from __future__ import annotations

import re
from typing import Optional

_IN_SCOPE_APPOINTMENT_HINTS = (
    "consultation fee",
    "doctor fee",
    "appointment fee",
    "kitna charge",
    "booking fee",
    "book appointment",
    "appointment book",
    "slot available",
    "doctor appointment",
    "अपॉइंटमेंट",
    "डॉक्टर से मिलना",
)

_CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    ("billing", [
        r"\bbill\b",
        r"\binvoice\b",
        r"\brefund\b",
        r"\bpayment dispute\b",
        r"\bbilling\b",
        r"बिल",
        r"भुगतान",
        r"रिफंड",
    ]),
    ("insurance", [
        r"\binsurance\b",
        r"\bclaim\b",
        r"\bcoverage\b",
        r"\bempanel",
        r"\btpa\b",
        r"बीमा",
        r"क्लेम",
    ]),
    ("lab", [
        r"\blab report\b",
        r"\btest result\b",
        r"\bblood report\b",
        r"\bpathology\b",
        r"लैब",
        r"रिपोर्ट",
        r"टेस्ट रिजल्ट",
    ]),
    ("pharmacy", [
        r"\bpharmacy\b",
        r"\bprescription\b",
        r"\bmedicine stock\b",
        r"\bdosage\b",
        r"दवा",
        r"फार्मेसी",
        r"नुस्खा",
    ]),
    ("medical_advice", [
        r"\bdiagnos",
        r"what medicine should",
        r"\bself.?medicate\b",
        r"क्या दवा",
        r"डायग्नोस",
    ]),
    ("general_oos", [
        r"\bward\b",
        r"\bparking\b",
        r"\bjob opening\b",
        r"\bvisiting hours for patient\b",
        r"वार्ड",
        r"पार्किंग",
    ]),
]

_INSIST_PATTERNS = [
    r"\bhuman\b",
    r"\bagent\b",
    r"\bmanager\b",
    r"\btransfer\b",
    r"\bconnect me\b",
    r"\bspeak to someone\b",
    r"किसी से बात",
    r"इंसान",
    r"मैनेजर",
    r"ट्रांसफर",
]

DECLINE_MESSAGES = {
    "english": (
        "I can only help with doctor appointments on this line. "
        "I can't handle {topic} here. Would you like to book an appointment?"
    ),
    "hindi": (
        "मैं यहाँ सिर्फ डॉक्टर अपॉइंटमेंट में मदद कर सकती हूँ। "
        "{topic} के लिए अभी मैं सहायता नहीं कर सकती। क्या आप अपॉइंटमेंट बुक करना चाहेंगे?"
    ),
    "hinglish": (
        "Main yahan sirf doctor appointments mein help kar sakti hoon. "
        "{topic} ke liye abhi main assist nahi kar sakti. "
        "Kya aap appointment book karna chahenge?"
    ),
}

TRANSFER_MESSAGES = {
    "english": "I'll connect you to our team now.",
    "hindi": "मैं आपको अभी हमारी टीम से जोड़ रही हूँ।",
    "hinglish": "Main aapko abhi hamari team se connect kar rahi hoon.",
}

_TOPIC_LABELS = {
    "billing": "billing",
    "insurance": "insurance",
    "lab": "lab reports",
    "pharmacy": "pharmacy",
    "medical_advice": "medical advice",
    "general_oos": "that request",
}


def _in_scope_appointment_context(lower: str) -> bool:
    return any(h in lower for h in _IN_SCOPE_APPOINTMENT_HINTS)


def _search_pattern(pat: str, text: str, lower: str) -> bool:
    target = lower if pat.isascii() else text
    return re.search(pat, target, re.IGNORECASE) is not None


def detect_out_of_scope(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    lower = text.lower()
    if _in_scope_appointment_context(lower):
        return None
    for category, patterns in _CATEGORY_PATTERNS:
        for pat in patterns:
            if _search_pattern(pat, text, lower):
                return category
    return None


def caller_insists_on_human(text: str) -> bool:
    lower = (text or "").lower()
    for pat in _INSIST_PATTERNS:
        if _search_pattern(pat, text, lower):
            return True
    return False


def should_escalate_oos(
    *,
    category: str,
    strikes: int,
    last_category: Optional[str],
    insists: bool,
) -> bool:
    if insists:
        return True
    if strikes >= 1 and last_category == category:
        return True
    if strikes >= 2:
        return True
    return False


def _lang_key(caller_language: Optional[str]) -> str:
    if caller_language in DECLINE_MESSAGES:
        return caller_language
    return "hinglish"


def decline_message(category: str, caller_language: Optional[str]) -> str:
    topic = _TOPIC_LABELS.get(category, "that")
    template = DECLINE_MESSAGES[_lang_key(caller_language)]
    return template.format(topic=topic)


def transfer_message(caller_language: Optional[str]) -> str:
    return TRANSFER_MESSAGES[_lang_key(caller_language)]
