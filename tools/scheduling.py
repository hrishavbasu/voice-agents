"""
Clinic scheduling tool — book appointments and list doctors.

Fixes applied:
  - Relative date parsing: "tomorrow", "Monday", "next week" → actual date objects
  - Full-day slot generation for target date (not just 5 from now)
  - abs() time distance so nearest slot to requested time wins
  - Doctor unavailability communicated back to LLM
  - Duplicate booking prevention: old Calendar event deleted on rebook
"""

import logging
import re
from datetime import datetime, timedelta, date
from typing import Optional
from zoneinfo import ZoneInfo

from config.company_config import COMPANY_CONFIG

logger = logging.getLogger(__name__)

# ── Time-to-spoken conversion ─────────────────────────────────────────────────
# Returns English natural-language time strings that the LLM reads verbatim.
# Prevents the LLM from hallucinating incorrect Hindi time names when converting
# raw digit strings like "5:00 PM" — a known failure mode observed in production.

_HOUR_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}

_HINDI_WEEKDAYS = {
    "सोमवार": "monday",
    "मंगलवार": "tuesday",
    "बुधवार": "wednesday",
    "गुरुवार": "thursday",
    "बृहस्पतिवार": "thursday",
    "शुक्रवार": "friday",
    "शनिवार": "saturday",
    "रविवार": "sunday",
}

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _normalize_input(text: str) -> str:
    return text.translate(_DEVANAGARI_DIGITS).strip().lower()


def _fee_spoken_en(value: Optional[int]) -> str:
    if value is None:
        return "fee will be shared by the hospital"
    common = {
        1500: "one thousand five hundred rupees",
        1700: "one thousand seven hundred rupees",
        1800: "one thousand eight hundred rupees",
        2000: "two thousand rupees",
        2300: "two thousand three hundred rupees",
        2500: "two thousand five hundred rupees",
        3000: "three thousand rupees",
    }
    return common.get(value, f"{value} rupees")


def _fee_spoken_hi(value: Optional[int]) -> str:
    if value is None:
        return "फीस अस्पताल टीम बताएगी"
    common = {
        1500: "एक हज़ार पांच सौ रुपये",
        1700: "एक हज़ार सात सौ रुपये",
        1800: "एक हज़ार आठ सौ रुपये",
        2000: "दो हज़ार रुपये",
        2300: "दो हज़ार तीन सौ रुपये",
        2500: "दो हज़ार पांच सौ रुपये",
        3000: "तीन हज़ार रुपये",
    }
    return common.get(value, f"{value} रुपये")


def _time_to_spoken(dt: datetime) -> str:
    """Convert a datetime to a natural English spoken time string.

    Examples:
        09:00 → "nine in the morning"
        14:30 → "half past two in the afternoon"
        17:00 → "five in the evening"
        18:30 → "half past six in the evening"
    """
    h, m = dt.hour, dt.minute
    h12 = h % 12 or 12

    if h < 12:
        period = "morning"
    elif h < 17:
        period = "afternoon"
    elif h < 21:
        period = "evening"
    else:
        period = "night"

    hour_w = _HOUR_WORDS[h12]

    if m == 0:
        return f"{hour_w} in the {period}"
    if m == 30:
        return f"half past {hour_w} in the {period}"
    if m == 15:
        return f"quarter past {hour_w} in the {period}"
    if m == 45:
        next_h12 = (h12 % 12) + 1
        return f"quarter to {_HOUR_WORDS[next_h12]} in the {period}"
    return f"{hour_w} {m:02d} in the {period}"


# In-memory store: patient_key → {"slot_iso": ..., "event_id": ..., "doctor": ...}
# Used to cancel old calendar events on rebook
_patient_bookings: dict[str, dict] = {}

# Slot-level lock: slot_iso → True (prevents double-booking in same process)
_booked_slots: set[str] = set()


def _tz() -> ZoneInfo:
    return ZoneInfo(COMPANY_CONFIG.get("timezone", "Asia/Kolkata"))


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_preferred_date(preferred_date: str) -> Optional[date]:
    """
    Convert caller's natural date expression to a date object.

    Handles: "tomorrow", "day after tomorrow", weekday names ("Monday"),
    "next Monday", absolute dates ("20 April", "April 20", "20th").
    Returns None if unparseable.
    """
    today = datetime.now(_tz()).date()
    pd = _normalize_input(preferred_date)
    for hi, en in _HINDI_WEEKDAYS.items():
        pd = pd.replace(hi, en)

    # Relative
    if pd in ("today", "aaj", "आज"):
        return today
    if pd in ("tomorrow", "kal", "kl", "कल"):
        return today + timedelta(days=1)
    if "day after" in pd or "parso" in pd or "परसों" in pd:
        return today + timedelta(days=2)
    if "next week" in pd or "अगले हफ्ते" in pd:
        return today + timedelta(days=7)

    # Weekday name: find next occurrence
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for i, name in enumerate(weekdays):
        if name in pd or name[:3] in pd:
            days_ahead = (i - today.weekday()) % 7
            if "next " in pd or "अगले" in pd:
                days_ahead = days_ahead or 7
            if days_ahead == 0:
                days_ahead = 7  # "Monday" means *next* Monday if today is Monday
            return today + timedelta(days=days_ahead)

    # Absolute: "20 april", "april 20", "20th april", "20/4", "20-04"
    months = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
        "january":1,"february":2,"march":3,"april":4,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    }
    # "20 april" or "april 20"
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)", pd)
    if m:
        day_n, mon_s = int(m.group(1)), m.group(2)[:3]
        mon_n = months.get(mon_s)
        if mon_n:
            year = today.year if (mon_n, day_n) >= (today.month, today.day) else today.year + 1
            try:
                return date(year, mon_n, day_n)
            except ValueError:
                pass
    m = re.search(r"([a-z]+)\s+(\d{1,2})", pd)
    if m:
        mon_s, day_n = m.group(1)[:3], int(m.group(2))
        mon_n = months.get(mon_s)
        if mon_n:
            year = today.year if (mon_n, day_n) >= (today.month, today.day) else today.year + 1
            try:
                return date(year, mon_n, day_n)
            except ValueError:
                pass

    return None


def _parse_preferred_hour(preferred_time: str) -> Optional[float]:
    """
    Parse caller's preferred time to a 24h float (e.g. 18.5 for 6:30 PM).
    Returns float so _pick_best_slot can distinguish 6:00 from 6:30.
    """
    pt = _normalize_input(preferred_time)
    pt = (
        pt.replace("सुबह", "morning")
        .replace("दोपहर", "afternoon")
        .replace("शाम", "evening")
        .replace("रात", "night")
    )

    _WORD_HOURS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "one o'clock": 1, "two o'clock": 2,
        "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5,
        "cheh": 6, "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
        "gyarah": 11, "baarah": 12,
    }
    _WORD_MINUTES = {
        "thirty": 30, "fifteen": 15, "forty five": 45, "forty-five": 45,
        "half past": 30, "quarter past": 15, "quarter to": -15,
    }

    word_hour: Optional[int] = None
    word_minute: int = 0

    m_half = re.search(r"(half past|quarter past|quarter to)\s+(\w+)", pt)
    if m_half:
        adj = _WORD_MINUTES.get(m_half.group(1), 0)
        h = _WORD_HOURS.get(m_half.group(2))
        if h is not None:
            word_hour = h
            word_minute = adj

    if word_hour is None:
        for hw, hv in _WORD_HOURS.items():
            if re.search(r"\b" + hw + r"\b", pt):
                word_hour = hv
                for mw, mv in _WORD_MINUTES.items():
                    if mw in pt:
                        word_minute = mv
                        break
                break

    is_morning   = any(w in pt for w in ["morning", "subah", "savere"])
    is_afternoon = any(w in pt for w in ["afternoon", "dopahar", "dupahr"])
    is_evening   = any(w in pt for w in ["evening", "shaam", "sham"])
    is_night     = any(w in pt for w in ["night", "raat"])

    if word_hour is not None:
        if is_morning:
            if word_hour == 12:
                word_hour = 0
        elif is_afternoon or is_evening or is_night:
            if word_hour != 12:
                word_hour += 12
        else:
            if word_hour <= 8:
                word_hour += 12
        return word_hour + word_minute / 60.0

    if is_morning:
        return 10.0
    if is_afternoon:
        return 14.0
    if is_evening or is_night:
        return 17.0

    # ── Digit-based parsing (preserves minutes) ───────────────────────────────
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", pt)
    if not m:
        return None

    hour = int(m.group(1))
    minutes = int(m.group(2)) if m.group(2) else 0
    ampm = m.group(3)

    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif ampm is None and hour <= 8:
        hour += 12

    return hour + minutes / 60.0


# ── Slot generation ───────────────────────────────────────────────────────────

def _slots_for_date(doctor: dict, target_date: date) -> list[datetime]:
    """
    Return ALL available 30-min slots for a doctor on a specific date.
    Returns empty list if doctor not available that day.
    """
    duration = COMPANY_CONFIG.get("appointment_slot_duration_minutes", 30)
    biz_hours = COMPANY_CONFIG.get("business_hours", {})
    available_days = set(doctor.get("available_days", []))
    tz = _tz()

    day_name = target_date.strftime("%A").lower()

    if day_name not in available_days:
        return []

    hours = biz_hours.get(day_name)
    if not hours:
        return []

    open_h, open_m = map(int, hours["open"].split(":"))
    close_h, close_m = map(int, hours["close"].split(":"))

    now = datetime.now(tz)
    slots = []
    candidate = datetime(
        target_date.year, target_date.month, target_date.day,
        open_h, open_m, 0, tzinfo=tz
    )
    close_dt = datetime(
        target_date.year, target_date.month, target_date.day,
        close_h, close_m, 0, tzinfo=tz
    )

    while candidate + timedelta(minutes=duration) <= close_dt:
        # Skip slots already passed today
        if candidate > now:
            key = candidate.isoformat()
            if key not in _booked_slots:
                slots.append(candidate)
        candidate += timedelta(minutes=duration)

    return slots


def _next_available_slots_from(doctor: dict, from_date: date, n: int = 20) -> list[datetime]:
    """Return up to N available slots starting from from_date, across multiple days."""
    slots = []
    search_date = from_date
    max_days = 30

    for _ in range(max_days):
        if len(slots) >= n:
            break
        day_slots = _slots_for_date(doctor, search_date)
        slots.extend(day_slots)
        search_date += timedelta(days=1)

    return slots[:n]


def _pick_best_slot(
    slots: list[datetime],
    preferred_time: Optional[str],
) -> datetime:
    """
    From a list of slots (all on the same date or nearby), pick the one
    closest to preferred_time. Uses abs() so both directions are considered.
    """
    if not preferred_time:
        return slots[0]

    target_hour = _parse_preferred_hour(preferred_time)
    if target_hour is None:
        return slots[0]

    def distance(slot: datetime) -> float:
        # Prefer exact or first slot at/after target; penalise before slightly less
        diff = slot.hour + slot.minute / 60 - target_hour
        return abs(diff) + (0.01 if diff < 0 else 0)

    return min(slots, key=distance)


# ── Slot availability check ───────────────────────────────────────────────────

def check_doctor_slots(
    doctor_name: str,
    preferred_date: Optional[str] = None,
) -> dict:
    """
    Return available time slots for a doctor on a given date (or next available day).
    Use this BEFORE calling book_appointment to show the caller what times are open.

    Returns:
        {
            "success": True,
            "doctor": "Dr. Atul Bhaskar",
            "date": "Monday, 20 April",
            "slots": ["9:00 AM", "9:30 AM", ..., "4:00 PM", "4:30 PM"],
            "slot_count": N,
        }
    or if doctor not available on that date:
        {
            "success": True,
            "available_on_requested_date": False,
            "doctor": "...",
            "requested_date": "Saturday, 18 April",
            "next_available_date": "Monday, 20 April",
            "slots_on_next_date": ["9:00 AM", ...],
            "available_days": "Monday, Tuesday, Thursday, Friday",
        }
    """
    tz = _tz()
    today = datetime.now(tz).date()

    doctor = _find_doctor(doctor_name, None)
    if not doctor:
        return {"success": False, "reason": f"No doctor found matching '{doctor_name}'."}

    target_date = _parse_preferred_date(preferred_date) if preferred_date else today + timedelta(days=1)
    if target_date is None:
        target_date = today + timedelta(days=1)

    available_days = set(doctor.get("available_days", []))
    available_days_display = ", ".join(d.capitalize() for d in doctor.get("available_days", []))

    day_name = target_date.strftime("%A").lower()

    if day_name not in available_days:
        # Find next available day
        next_date = target_date + timedelta(days=1)
        for _ in range(14):
            if next_date.strftime("%A").lower() in available_days:
                break
            next_date += timedelta(days=1)

        next_slots = _slots_for_date(doctor, next_date)
        # Return a concise set of times for voice (max 6 spaced across the day)
        next_times = _sample_slots_for_voice(next_slots)

        return {
            "success": True,
            "available_on_requested_date": False,
            "doctor": doctor["name"],
            "specialty": doctor["specialty"],
            "requested_date": target_date.strftime("%A, %d %B"),
            "next_available_date": next_date.strftime("%A, %d %B"),
            "slots_on_next_date": next_times,
            "slots_on_next_date_objects": [_slot_payload(s) for s in next_slots[:6]],
            "available_days": available_days_display,
            "fee_inr": doctor.get("fee_inr"),
            "fee_spoken_en": _fee_spoken_en(doctor.get("fee_inr")),
            "fee_spoken_hi": _fee_spoken_hi(doctor.get("fee_inr")),
        }

    slots = _slots_for_date(doctor, target_date)
    if not slots:
        return {
            "success": False,
            "reason": f"{doctor['name']} has no open slots on {target_date.strftime('%A, %d %B')} "
                      f"(possibly fully booked). Try another date.",
        }

    return {
        "success": True,
        "available_on_requested_date": True,
        "doctor": doctor["name"],
        "specialty": doctor["specialty"],
        "date": target_date.strftime("%A, %d %B"),
        "slots": _sample_slots_for_voice(slots),
        "slot_objects": [_slot_payload(s) for s in slots[:6]],
        "slot_count": len(slots),
        "fee_inr": doctor.get("fee_inr"),
        "fee_spoken_en": _fee_spoken_en(doctor.get("fee_inr")),
        "fee_spoken_hi": _fee_spoken_hi(doctor.get("fee_inr")),
    }


def _sample_slots_for_voice(slots: list[datetime]) -> list[str]:
    """Pick up to 6 representative slots spread across the day.

    Returns strings in the format "spoken form (digit time)" so the LLM
    reads the spoken form verbatim and cannot hallucinate an incorrect
    time name.  Example: "five in the evening (5:00 PM)".

    The digit time in parentheses is the booking reference — the LLM
    passes it as preferred_time to book_appointment.
    """
    if not slots:
        return []

    if len(slots) <= 6:
        selected = slots
    else:
        indices = [0, len(slots)//5, 2*len(slots)//5, 3*len(slots)//5,
                   4*len(slots)//5, len(slots)-1]
        seen_keys: set[str] = set()
        selected = []
        for i in indices:
            key = slots[i].strftime("%-I:%M %p")
            if key not in seen_keys:
                seen_keys.add(key)
                selected.append(slots[i])

    return [f"{_time_to_spoken(s)} ({s.strftime('%-I:%M %p')})" for s in selected]


def _slot_payload(slot: datetime) -> dict:
    digit = slot.strftime("%-I:%M %p")
    spoken_en = _time_to_spoken(slot)
    spoken_hi = (
        spoken_en.replace("in the morning", "सुबह")
        .replace("in the afternoon", "दोपहर")
        .replace("in the evening", "शाम")
        .replace("in the night", "रात")
        .replace("half past", "साढ़े")
        .replace("quarter past", "सवा")
        .replace("quarter to", "पौने")
    )
    return {
        "digit_time": digit,
        "spoken_en": spoken_en,
        "spoken_hi": spoken_hi,
        "spoken_hinglish": spoken_hi,
    }


def _find_alternative_doctors(specialty: str, preferred_date: Optional[str]) -> list[dict]:
    """
    Find other doctors of the same specialty who have slots on the preferred date.
    Returns list of dicts with name, available slots, fee.
    """
    tz = _tz()
    today = datetime.now(tz).date()
    target_date = _parse_preferred_date(preferred_date) if preferred_date else today + timedelta(days=1)
    if target_date is None:
        target_date = today + timedelta(days=1)

    doctors = COMPANY_CONFIG.get("doctors", [])
    alternatives = []

    for d in doctors:
        if specialty.lower() not in d.get("specialty", "").lower():
            continue
        day_name = target_date.strftime("%A").lower()
        if day_name in set(d.get("available_days", [])):
            slots = _slots_for_date(d, target_date)
            if slots:
                alternatives.append({
                    "name": d["name"],
                    "specialty": d["specialty"],
                    "fee": f"₹{d.get('fee_inr', '?')}",
                    "fee_inr": d.get("fee_inr"),
                    "fee_spoken_en": _fee_spoken_en(d.get("fee_inr")),
                    "fee_spoken_hi": _fee_spoken_hi(d.get("fee_inr")),
                    "available_slots_today": _sample_slots_for_voice(slots),
                    "available_slot_objects": [_slot_payload(s) for s in slots[:6]],
                })

    return alternatives


# ── Doctor helpers ────────────────────────────────────────────────────────────

def list_doctors(specialty: Optional[str] = None) -> dict:
    doctors = COMPANY_CONFIG.get("doctors", [])

    if specialty:
        kw = specialty.lower()
        aliases = {
            "heart": "cardio", "cardiac": "cardio",
            "cardiology": "cardio", "कार्डियोलॉजी": "cardio", "हृदय": "cardio",
            "bone": "orthop", "knee": "orthop", "joint": "orthop", "leg": "orthop",
            "orthopedic": "orthop", "ऑर्थोपेडिक": "orthop", "हड्डी": "orthop",
            "eye": "ophthalm", "vision": "ophthalm",
            "नेत्र": "ophthalm", "आंख": "ophthalm",
            "stomach": "gastro", "digestive": "gastro",
            "gastro": "gastro", "गैस्ट्रो": "gastro", "पेट": "gastro",
            "brain": "neuro", "nerve": "neuro",
            "neuro": "neuro", "न्यूरो": "neuro", "दिमाग": "neuro",
            "cancer": "oncol", "tumor": "oncol",
            "oncology": "oncol", "ऑन्कोलॉजी": "oncol",
            "child": "pediatr", "kids": "pediatr",
            "pediatric": "pediatr", "बाल": "pediatr",
            "general": "internal", "fever": "internal",
            "physician": "internal", "जनरल": "internal",
        }
        for word, mapped in aliases.items():
            if word in kw:
                kw = mapped
                break
        doctors = [d for d in doctors if kw in d.get("specialty", "").lower()]

    formatted = []
    for d in doctors:
        days = ", ".join(day.capitalize() for day in d.get("available_days", []))
        entry = {
            "name": d["name"],
            "specialty": d["specialty"],
            "experience": f"{d.get('experience_years', '?')} years",
            "fee": f"₹{d.get('fee_inr', '?')}",
            "fee_inr": d.get("fee_inr"),
            "fee_spoken_en": _fee_spoken_en(d.get("fee_inr")),
            "fee_spoken_hi": _fee_spoken_hi(d.get("fee_inr")),
            "available_days": days,
        }
        if d.get("qualification"):
            entry["qualification"] = d["qualification"]
        formatted.append(entry)

    return {"success": True, "doctors": formatted, "count": len(formatted)}


_SPECIALTY_ALIASES: dict[str, str] = {
    "heart": "cardio", "cardiac": "cardio",
    "cardiology": "cardio", "कार्डियोलॉजी": "cardio", "हृदय": "cardio",
    "bone": "orthop", "knee": "orthop", "joint": "orthop", "leg": "orthop",
    "orthopedic": "orthop", "ऑर्थोपेडिक": "orthop", "हड्डी": "orthop",
    "eye": "ophthalm", "vision": "ophthalm",
    "नेत्र": "ophthalm", "आंख": "ophthalm",
    "stomach": "gastro", "digestive": "gastro",
    "gastro": "gastro", "गैस्ट्रो": "gastro", "पेट": "gastro",
    "brain": "neuro", "nerve": "neuro",
    "neuro": "neuro", "न्यूरो": "neuro", "दिमाग": "neuro",
    "cancer": "oncol", "tumor": "oncol",
    "oncology": "oncol", "ऑन्कोलॉजी": "oncol",
    "child": "pediatr", "kids": "pediatr",
    "pediatric": "pediatr", "बाल": "pediatr",
    "general": "internal", "physician": "internal", "fever": "internal",
    "जनरल": "internal",
}


def _find_doctor(doctor_name: Optional[str], specialty: Optional[str]) -> Optional[dict]:
    doctors = COMPANY_CONFIG.get("doctors", [])
    if doctor_name:
        name_lower = doctor_name.lower()
        for d in doctors:
            if name_lower in d["name"].lower():
                return d
    if specialty:
        spec_lower = specialty.lower()
        for word, mapped in _SPECIALTY_ALIASES.items():
            if word in spec_lower:
                spec_lower = mapped
                break
        for d in doctors:
            if spec_lower in d["specialty"].lower():
                return d
    return None


# ── Main booking function ─────────────────────────────────────────────────────

def book_appointment(
    caller_phone: str,
    patient_name: str,
    concern: str,
    doctor_name: Optional[str] = None,
    specialty: Optional[str] = None,
    preferred_date: Optional[str] = None,
    preferred_time: Optional[str] = None,
) -> dict:
    """
    Book a clinic appointment.

    1. Resolves doctor by name or specialty.
    2. Parses preferred_date into a real date object.
    3. Checks if doctor is available on that date.
    4. Generates all slots for that day, picks nearest to preferred_time.
    5. Cancels any previous booking for this patient+doctor before creating new one.
    6. Creates Google Calendar event.

    Returns:
        {"success": True, "slot": "...", "doctor": "...", "fee": "..."}
        {"success": False, "reason": "..."}  — includes helpful info for LLM
    """
    tz = _tz()
    today = datetime.now(tz).date()

    # ── Resolve doctor ────────────────────────────────────────────────────────
    doctor = _find_doctor(doctor_name, specialty)
    if not doctor:
        doctors = COMPANY_CONFIG.get("doctors", [])
        if not doctors:
            return {"success": False, "reason": "No doctors configured."}
        doctor = doctors[0]

    available_days_display = ", ".join(d.capitalize() for d in doctor.get("available_days", []))

    # ── Parse target date ─────────────────────────────────────────────────────
    target_date: Optional[date] = None
    if preferred_date:
        target_date = _parse_preferred_date(preferred_date)

    if target_date is None:
        # No date preference — use next available day for this doctor
        target_date = today

    # ── Check doctor availability on target date ───────────────────────────────
    day_name = target_date.strftime("%A").lower()
    if day_name not in set(doctor.get("available_days", [])):
        # Find next day the doctor IS available
        next_available = target_date + timedelta(days=1)
        for _ in range(14):
            if next_available.strftime("%A").lower() in set(doctor.get("available_days", [])):
                break
            next_available += timedelta(days=1)

        next_slots = _sample_slots_for_voice(_slots_for_date(doctor, next_available))

        # Also check for alternative doctors available on the requested date
        alternatives = _find_alternative_doctors(doctor["specialty"], preferred_date)

        return {
            "success": False,
            "doctor_unavailable_on_requested_date": True,
            "reason": (
                f"{doctor['name']} is not available on {target_date.strftime('%A, %d %B')} "
                f"(available: {available_days_display}). "
                f"Next available: {next_available.strftime('%A, %d %B')} with slots: {', '.join(next_slots)}. "
                f"Ask patient: book on {next_available.strftime('%A')},"
                f" or switch doctor?"
            ),
            "next_available_date": next_available.strftime("%A, %d %B"),
            "next_available_slots": next_slots,
            "alternative_doctors_on_requested_date": alternatives,
        }

    # ── Generate all slots for target date ────────────────────────────────────
    day_slots = _slots_for_date(doctor, target_date)

    if not day_slots:
        # Try next available day
        fallback_slots = _next_available_slots_from(doctor, target_date + timedelta(days=1), n=20)
        if not fallback_slots:
            return {
                "success": False,
                "reason": f"{doctor['name']} has no slots available in the next 30 days.",
            }
        # Use next available day's slots
        day_slots = [s for s in fallback_slots if s.date() == fallback_slots[0].date()]
        target_date = fallback_slots[0].date()

    # ── Pick best time slot ───────────────────────────────────────────────────
    chosen = _pick_best_slot(day_slots, preferred_time)

    # ── Cancel ALL existing bookings for this caller phone number ────────────
    # Keying by phone (not patient name) prevents double-bookings when the
    # caller changes doctor mid-conversation, or gives a slightly different
    # name spelling.  All prior bookings for this phone are cancelled first.
    import os as _os
    prefix = f"{caller_phone}::"
    stale_keys = [k for k in _patient_bookings if k.startswith(prefix)]
    for stale_key in stale_keys:
        old = _patient_bookings.pop(stale_key)
        old_event_id = old.get("event_id")
        if old_event_id and old_event_id != "dry-run":
            try:
                from tools.calendar import _get_service
                svc = _get_service()
                if svc:
                    calendar_id = _os.getenv("GOOGLE_CALENDAR_ID", "primary")
                    svc.events().delete(calendarId=calendar_id, eventId=old_event_id).execute()
                    logger.info("Deleted old calendar event %s on rebook for %s",
                                old_event_id, caller_phone)
            except Exception as exc:
                logger.warning("Could not delete old event %s: %s", old_event_id, exc)
        _booked_slots.discard(old.get("slot_iso", ""))
    patient_key = f"{caller_phone}::{doctor['name'].lower()}"

    # ── Create Google Calendar event ──────────────────────────────────────────
    from tools.calendar import create_appointment_event
    result = create_appointment_event(
        patient_name=patient_name,
        patient_phone=caller_phone,
        doctor_name=doctor["name"],
        specialty=doctor["specialty"],
        concern=concern,
        start_dt=chosen,
        duration_minutes=COMPANY_CONFIG.get("appointment_slot_duration_minutes", 30),
        fee_inr=doctor.get("fee_inr"),
    )

    if result.get("success"):
        slot_iso = chosen.isoformat()
        _booked_slots.add(slot_iso)
        _patient_bookings[patient_key] = {
            "slot_iso": slot_iso,
            "event_id": result.get("event_id"),
            "doctor": doctor["name"],
        }

        booked_spoken = _time_to_spoken(chosen)
        booked_digit = chosen.strftime("%-I:%M %p")

        success_result: dict = {
            "success": True,
            "slot": result["slot"],
            "slot_spoken": f"{booked_spoken} ({booked_digit})",
            "slot_object": _slot_payload(chosen),
            "doctor": doctor["name"],
            "specialty": doctor["specialty"],
            "fee": f"₹{doctor.get('fee_inr', '?')}",
            "fee_inr": doctor.get("fee_inr"),
            "fee_spoken_en": _fee_spoken_en(doctor.get("fee_inr")),
            "fee_spoken_hi": _fee_spoken_hi(doctor.get("fee_inr")),
            "event_id": result.get("event_id"),
        }

        # Warn LLM if the booked time is significantly different from what was asked.
        # Without this, the LLM sees a mismatch and falsely reports the slot as unavailable.
        if preferred_time:
            pref_hour = _parse_preferred_hour(preferred_time)
            if pref_hour is not None:
                diff = abs(chosen.hour + chosen.minute / 60 - pref_hour)
                if diff > 0.25:  # more than 15 min off
                    success_result["note"] = (
                        f"Booked at {booked_spoken} ({booked_digit}) — nearest available slot. "
                        f"Requested time '{preferred_time}' had no exact match. "
                        "Tell the caller the actual booked time and confirm they accept it. "
                        "Do NOT say the slot was unavailable — it was booked successfully."
                    )

        return success_result

    return result
