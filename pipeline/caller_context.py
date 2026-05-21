"""
Extract and persist per-call caller context (name, concern, preferences).

Keeps the LLM from re-asking for information the caller already gave.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

_CONTEXT_EVENTS_CAP = 20

# Agent / system names — never treat as caller name
_BLOCKED_NAMES = frozenset({
    "priya", "apollo", "hospitals", "hospital", "doctor", "dr", "appointment",
    "namaste", "hello", "hi", "yes", "no", "haan", "ji", "okay", "thanks",
    "namaskar", "ma'am", "madam", "mam", "मैम", "मेम",
    "cardiologist", "cardiology", "internal", "medicine", "gastro", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "kal", "parso", "aaj", "tomorrow", "today",
    # Hindi affirmations / acknowledgements (Devanagari) — commonly misread as names
    "हाँ", "हां", "नहीं",
    "ठीक", "बिल्कुल", "ज़रूर",
    "चलेगा", "चलेगी", "चलेगे",
    "अच्छा", "बढ़िया", "बहुत",
    "शुक्रिया", "धन्यवाद",
    "दीजिए", "करें", "कीजिए", "बताइए", "बताओ",
    "समझ", "अलविदा",
    # Roman equivalents missing from original set
    "bilkul", "zaroor", "theek", "thik", "nahi", "sure", "acha", "achha", "karo", "kijiye",
    "hai", "hain", "tha", "thi", "the",
})

_NAME_INTRO_PATTERNS = [
    re.compile(
        r"(?:my name is|i am|i'm|i'?m|this is|call me|name is)\s+"
        r"([A-Za-z][A-Za-z\s.'-]{1,48})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:mera naam|mera name|naam hai|naam)\s+"
        r"([A-Za-z\u0900-\u097f][A-Za-z\u0900-\u097f\s.'-]{1,48}?)"
        r"(?:\s+hai|\s+hu|\s+हूँ|\s+हूं|,|\.|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"मैं\s+([\u0900-\u097fA-Za-z][\u0900-\u097fA-Za-z.'-]{0,30}?)"
        r"\s+(?:हू[ँं]|बात\s+कर|बोल)",
        re.IGNORECASE,
    ),
    re.compile(r"मेरा\s+नाम\s+([\u0900-\u097fA-Za-z][\u0900-\u097fA-Za-z\s.'-]{1,48})\s+है"),
    re.compile(r"(?:i'm|i am)\s+([\u0900-\u097f][\u0900-\u097f\s]{1,30})", re.IGNORECASE),
]

# Matches pure affirmation utterances that are never caller names
_SHORT_ACK_RE = re.compile(
    r"^(हाँ|हां|नहीं|ठीक|बिल्कुल|ज़रूर|चलेगा|चलेगी|बढ़िया|अच्छा|शुक्रिया|धन्यवाद)"
    r"(\s+(है|हैं|चलेगा|चलेगी|जी|sir|सर))?$",
    re.IGNORECASE,
)

_CONCERN_PATTERNS = [
    re.compile(
        r"(?:coming for|reason is|problem is|symptoms?|suffering from|"
        r"checkup for|appointment for|consultation for)\s+(.{3,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:आया हूँ|आई हूँ|दिक्कत|समस्या|तकलीफ|चेकअप|इलाज)\s+(.{3,80})",
        re.IGNORECASE,
    ),
]


def _clean_name(raw: str) -> Optional[str]:
    name = re.sub(r"\s+", " ", raw).strip(" .,!?")
    if not name or len(name) < 2:
        return None
    # Drop trailing filler words
    name = re.sub(
        r"\s+(hai|hu|hoon|हूँ|हूं|here|speaking|bol raha|bol rahi|बात कर.*).*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    name = re.sub(r"\s+बात\s+कर.*$", "", name, flags=re.IGNORECASE).strip()
    tokens = name.split()
    if not tokens:
        return None
    lowered = {t.lower() for t in tokens}
    if lowered & {"ma'am", "madam", "mam"}:
        return None
    if "मैम" in name or "मेम" in name:
        return None
    # Reject if every token is blocked
    if all(t.lower() in _BLOCKED_NAMES for t in tokens):
        return None
    if tokens[0].lower() in ("dr", "doctor", "डॉक्टर", "डॉ"):
        return None
    if len(tokens) > 4:
        return None
    return name.title() if name.isascii() else name


def extract_caller_name(text: str) -> Optional[str]:
    """Best-effort name from a single caller utterance."""
    text = text.strip()
    if not text:
        return None
    if "मैं मैं" in text:
        # STT stutter/opening duplicate often yields unstable first name token.
        return None

    for pat in _NAME_INTRO_PATTERNS:
        m = pat.search(text)
        if m:
            cleaned = _clean_name(m.group(1))
            if cleaned:
                return cleaned

    # Reject pure affirmation utterances before short-form check
    if _SHORT_ACK_RE.match(text.strip()):
        return None

    # Short standalone introduction: "Rahul Sharma" / "मैं राहुल"
    words = text.split()
    if 1 <= len(words) <= 3 and len(text) < 40:
        lower = text.lower()
        if any(
            kw in lower
            for kw in (
                "appointment", "doctor", "slot", "kal", "parso", "cardio",
                "pain", "book", "time", "baje", "बजे",
                "namaskar", "namaste", "ma'am", "madam", "मैम", "मेम",
            )
        ):
            return None
        if re.match(r"^[\u0900-\u097fA-Za-z][\u0900-\u097fA-Za-z\s.'-]+$", text):
            cleaned = _clean_name(text)
            if cleaned and not any(t.lower() in _BLOCKED_NAMES for t in cleaned.split()):
                return cleaned

    return None


def extract_concern(text: str) -> Optional[str]:
    text = text.strip()
    for pat in _CONCERN_PATTERNS:
        m = pat.search(text)
        if m:
            concern = m.group(1).strip(" .,!?")
            if len(concern) >= 3:
                return concern[:120]
    return None


def record_context_event(session: dict, new_events: list[dict]) -> list[dict]:
    """Merge new_events into session list, cap length, return final list."""
    events = list(session.get("context_events") or [])
    events.extend(new_events)
    if len(events) > _CONTEXT_EVENTS_CAP:
        events = events[-_CONTEXT_EVENTS_CAP:]
    return events


def apply_context_updates(
    session: dict,
    updates: dict,
    *,
    source: str,
    detail: str,
) -> tuple[dict, list[dict]]:
    """Return (field_updates, events) only for keys whose values actually change."""
    field_updates: dict = {}
    new_events: list[dict] = []
    now = time.time()
    detail = (detail or "")[:200]

    for field, value in updates.items():
        if value is None:
            continue
        str_val = str(value).strip()
        if not str_val:
            continue
        if session.get(field) == str_val:
            continue
        field_updates[field] = str_val
        new_events.append({
            "ts": now,
            "field": field,
            "value": str_val,
            "source": source,
            "detail": detail,
        })
        logger.info(
            "context_update field=%s value=%r source=%s",
            field,
            str_val,
            source,
        )
    return field_updates, new_events


def merge_context_from_utterance(
    existing: dict,
    utterance: str,
) -> dict:
    """Return session field updates inferred from this caller utterance."""
    raw: dict = {}
    if not existing.get("caller_name"):
        name = extract_caller_name(utterance)
        if name:
            raw["caller_name"] = name
    if not existing.get("caller_concern"):
        concern = extract_concern(utterance)
        if concern:
            raw["caller_concern"] = concern
    updates, _ = apply_context_updates(
        existing, raw, source="utterance", detail=utterance
    )
    return updates


def merge_context_from_crm(existing: dict, crm_contact: Optional[dict]) -> dict:
    if existing.get("caller_name") or not crm_contact:
        return {}
    props = crm_contact.get("properties", {})
    first = (props.get("firstname") or "").strip()
    last = (props.get("lastname") or "").strip()
    name = f"{first} {last}".strip()
    if not name:
        return {}
    updates, _ = apply_context_updates(
        existing, {"caller_name": name}, source="crm", detail="hubspot"
    )
    return updates


def format_context_for_prompt(session: Optional[dict]) -> str:
    """Human-readable block injected into the system prompt."""
    if not session:
        return ""

    lines = ["\n## Caller context — USE THIS; do not re-ask (CRITICAL)"]
    has_any = False

    name = session.get("caller_name")
    if name:
        has_any = True
        lines.append(
            f"- **Patient name:** {name} — ALREADY KNOWN. "
            "Do NOT ask \"आपका नाम क्या है?\" or \"What is your name?\" again during the call. "
            "Address them by name when natural. "
            "ONLY at booking time (right before book_appointment), confirm once: "
            f"\"{name} — booking ke liye naam sahi hai?\" / \"Just confirming the name for the appointment — {name}?\""
        )

    concern = session.get("caller_concern")
    if concern:
        has_any = True
        lines.append(
            f"- **Reason for visit:** {concern} — already shared; do not ask again unless unclear."
        )

    doctor = session.get("preferred_doctor")
    if doctor:
        has_any = True
        lines.append(
            f"- **Preferred doctor:** {doctor} — remember this; don't ask which doctor unless they want to change."
        )

    pref_date = session.get("preferred_date")
    if pref_date:
        has_any = True
        lines.append(f"- **Preferred date discussed:** {pref_date}")

    pref_time = session.get("preferred_time")
    if pref_time:
        has_any = True
        lines.append(f"- **Preferred time discussed:** {pref_time}")

    specialty = session.get("preferred_specialty")
    if specialty:
        has_any = True
        lines.append(f"- **Specialty discussed:** {specialty}")

    if not has_any:
        return ""

    lines.append(
        "- Carry forward ALL of the above across turns. "
        "Acknowledge what they already said before asking the next missing detail only."
    )
    return "\n".join(lines)
