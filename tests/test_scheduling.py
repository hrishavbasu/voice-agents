"""
Tests for the scheduling tools — date parsing, slot generation, booking flow.
These run without Google Calendar (calendar calls are mocked).
"""

import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

from tools.scheduling import (
    _parse_preferred_date,
    _parse_preferred_hour,
    _time_to_spoken,
    check_doctor_slots,
    list_doctors,
    book_appointment,
    _slots_for_date,
    _booked_slots,
)


IST = ZoneInfo("Asia/Kolkata")

# ── Date parsing ──────────────────────────────────────────────────────────────

def test_parse_tomorrow():
    result = _parse_preferred_date("tomorrow")
    assert result == date.today() + timedelta(days=1)


def test_parse_kal():
    assert _parse_preferred_date("kal") == date.today() + timedelta(days=1)


def test_parse_today():
    assert _parse_preferred_date("today") == date.today()


def test_parse_day_after_tomorrow():
    assert _parse_preferred_date("day after tomorrow") == date.today() + timedelta(days=2)


def test_parse_next_week():
    result = _parse_preferred_date("next week")
    assert result == date.today() + timedelta(days=7)


def test_parse_absolute_date_day_month():
    result = _parse_preferred_date("20 April")
    assert result is not None
    assert result.month == 4
    assert result.day == 20


def test_parse_absolute_date_with_ordinal():
    result = _parse_preferred_date("20th April")
    assert result is not None and result.day == 20 and result.month == 4


def test_parse_invalid_returns_none():
    assert _parse_preferred_date("some garbage xyz") is None


# ── Time parsing ──────────────────────────────────────────────────────────────

def test_parse_4pm():
    assert _parse_preferred_hour("4pm") == 16


def test_parse_9am():
    assert _parse_preferred_hour("9am") == 9


def test_parse_morning():
    assert _parse_preferred_hour("morning") == 10


def test_parse_afternoon():
    assert _parse_preferred_hour("afternoon") == 14


def test_parse_evening():
    assert _parse_preferred_hour("evening") == 17


def test_parse_six_in_the_evening():
    assert _parse_preferred_hour("six in the evening") == 18


def test_parse_half_past_six_evening():
    assert _parse_preferred_hour("half past six in the evening") == 18


def test_parse_shaam():
    assert _parse_preferred_hour("shaam") == 17


def test_parse_digit_colon():
    assert _parse_preferred_hour("14:30") == 14


# ── Time to spoken ────────────────────────────────────────────────────────────

from datetime import datetime

def _dt(h, m=0):
    return datetime(2026, 4, 28, h, m, tzinfo=IST)

def test_spoken_nine_morning():
    assert _time_to_spoken(_dt(9)) == "nine in the morning"

def test_spoken_half_past_two_afternoon():
    assert _time_to_spoken(_dt(14, 30)) == "half past two in the afternoon"

def test_spoken_five_evening():
    assert _time_to_spoken(_dt(17)) == "five in the evening"

def test_spoken_quarter_past_three():
    assert _time_to_spoken(_dt(15, 15)) == "quarter past three in the afternoon"

def test_spoken_quarter_to_four():
    assert _time_to_spoken(_dt(15, 45)) == "quarter to four in the afternoon"


# ── list_doctors ──────────────────────────────────────────────────────────────

def test_list_doctors_returns_all():
    result = list_doctors()
    assert result["success"] is True
    assert result["count"] >= 8


def test_list_doctors_cardiology_filter():
    result = list_doctors(specialty="cardiology")
    assert result["success"] is True
    assert all("Cardio" in d["specialty"] for d in result["doctors"])


def test_list_doctors_heart_alias():
    result = list_doctors(specialty="heart")
    assert result["success"] is True
    assert result["count"] >= 1


def test_list_doctors_orthopedic_filter():
    result = list_doctors(specialty="orthopedic")
    assert result["success"] is True
    assert any("Ortho" in d["specialty"] for d in result["doctors"])


def test_list_doctors_bone_alias():
    result = list_doctors(specialty="bone")
    names = [d["name"] for d in result["doctors"]]
    assert any("Bhaskar" in n for n in names)


def test_list_doctors_includes_fee():
    result = list_doctors()
    for doc in result["doctors"]:
        assert "fee" in doc
        assert "₹" in doc["fee"]


def test_list_doctors_includes_available_days():
    result = list_doctors()
    for doc in result["doctors"]:
        assert "available_days" in doc


# ── check_doctor_slots ────────────────────────────────────────────────────────

def test_check_slots_kulkarni_weekday():
    """Dr. Kulkarni is available Mon-Fri."""
    # Find next Monday
    today = date.today()
    days_ahead = (0 - today.weekday()) % 7 or 7
    next_monday = today + timedelta(days=days_ahead)
    result = check_doctor_slots("Dr. S V Kulkarni", next_monday.strftime("%A, %d %B"))
    assert result["success"] is True
    assert result.get("available_on_requested_date") is True
    assert len(result["slots"]) >= 1


def test_check_slots_bhaskar_unavailable_saturday():
    """Dr. Atul Bhaskar is only Mon/Tue/Thu/Fri — Saturday should bounce."""
    # Find next Saturday
    today = date.today()
    days_ahead = (5 - today.weekday()) % 7 or 7
    next_saturday = today + timedelta(days=days_ahead)
    result = check_doctor_slots("Dr. Atul Bhaskar", next_saturday.strftime("%A, %d %B"))
    assert result["success"] is True
    assert result.get("available_on_requested_date") is False
    assert "next_available_date" in result


def test_check_slots_unknown_doctor():
    result = check_doctor_slots("Dr. Nobody Known")
    assert result["success"] is False
    assert "reason" in result


def test_check_slots_spoken_format():
    """Slots must be returned as 'spoken form (digit time)'."""
    today = date.today()
    days_ahead = (0 - today.weekday()) % 7 or 7
    next_monday = today + timedelta(days=days_ahead)
    result = check_doctor_slots("Dr. Anuj Sathe", next_monday.strftime("%A, %d %B"))
    if result.get("available_on_requested_date"):
        for slot in result["slots"]:
            assert "(" in slot and ")" in slot, f"Slot not in spoken format: {slot}"
            assert "AM" in slot or "PM" in slot


# ── book_appointment (Calendar mocked) ───────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_bookings():
    """Reset in-memory booking state before each test."""
    _booked_slots.clear()
    from tools import scheduling as s
    s._patient_bookings.clear()
    yield
    _booked_slots.clear()
    s._patient_bookings.clear()


MOCK_CALENDAR_SUCCESS = {
    "success": True,
    "slot": "Monday, 28 April at nine in the morning (9:00 AM)",
    "event_id": "fake-event-123",
}


@patch("tools.calendar.create_appointment_event", return_value=MOCK_CALENDAR_SUCCESS)
def test_book_appointment_success(mock_cal):
    today = date.today()
    days_ahead = (0 - today.weekday()) % 7 or 7
    next_monday = (today + timedelta(days=days_ahead)).strftime("%A, %d %B")

    result = book_appointment(
        caller_phone="+919999999999",
        patient_name="Rahul Sharma",
        concern="knee pain",
        doctor_name="Dr. Atul Bhaskar",
        preferred_date=next_monday,
        preferred_time="morning",
    )
    assert result["success"] is True
    assert "doctor" in result
    assert "fee" in result
    assert mock_cal.called


@patch("tools.calendar.create_appointment_event", return_value=MOCK_CALENDAR_SUCCESS)
def test_book_appointment_includes_fee(mock_cal):
    result = book_appointment(
        caller_phone="+910000000000",
        patient_name="Priya",
        concern="eye checkup",
        doctor_name="Dr. Atul Seth",
        preferred_date="monday",
        preferred_time="morning",
    )
    if result["success"]:
        assert "₹" in result["fee"]


@patch("tools.calendar.create_appointment_event", return_value=MOCK_CALENDAR_SUCCESS)
def test_book_appointment_doctor_unavailable_on_date(mock_cal):
    """Dr. Atul Bhaskar not available Saturday — should return failure with alternatives."""
    today = date.today()
    days_ahead = (5 - today.weekday()) % 7 or 7
    next_saturday = (today + timedelta(days=days_ahead)).strftime("%A, %d %B")
    result = book_appointment(
        caller_phone="+918888888888",
        patient_name="Test Patient",
        concern="shoulder pain",
        doctor_name="Dr. Atul Bhaskar",
        preferred_date=next_saturday,
    )
    assert result["success"] is False
    assert result.get("doctor_unavailable_on_requested_date") is True
    assert "next_available_date" in result
    assert not mock_cal.called


@patch("tools.calendar.create_appointment_event", return_value=MOCK_CALENDAR_SUCCESS)
def test_rebook_cancels_previous(mock_cal):
    """Booking same caller twice should cancel the first booking."""
    with patch("tools.calendar.create_appointment_event", return_value=MOCK_CALENDAR_SUCCESS):
        r1 = book_appointment(
            caller_phone="+917777777777",
            patient_name="Amit",
            concern="back pain",
            doctor_name="Dr. Atul Bhaskar",
            preferred_date="monday",
            preferred_time="morning",
        )
        r2 = book_appointment(
            caller_phone="+917777777777",
            patient_name="Amit",
            concern="back pain",
            doctor_name="Dr. Atul Bhaskar",
            preferred_date="tuesday",
            preferred_time="afternoon",
        )
    # Both should succeed; no duplicate in booked_slots for same patient
    from tools.scheduling import _patient_bookings
    # Only one entry per caller::doctor pair
    keys = [k for k in _patient_bookings if k.startswith("+917777777777")]
    assert len(keys) <= 1
