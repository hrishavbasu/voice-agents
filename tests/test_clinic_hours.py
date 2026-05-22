"""Clinic books appointments 9 AM – 9 PM only."""
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from tools.scheduling import (
    _slots_for_date,
    _preferred_time_allowed,
    book_appointment,
    parse_preferred_date,
)

TZ = ZoneInfo("Asia/Kolkata")
FIXED_NOW = datetime(2026, 5, 18, 8, 0, 0, tzinfo=TZ)  # Monday


@pytest.fixture
def monday():
    with patch("tools.scheduling.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_NOW
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        yield date(2026, 5, 18)


DOCTOR = {
    "name": "Dr. Test",
    "specialty": "Internal Medicine",
    "available_days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"],
}


def test_slots_start_at_9am(monday):
    slots = _slots_for_date(DOCTOR, monday)
    assert slots, "expected slots on Monday"
    assert slots[0].hour == 9 and slots[0].minute == 0
    assert all(s.hour >= 9 for s in slots)


def test_slots_end_by_9pm(monday):
    slots = _slots_for_date(DOCTOR, monday)
    last = slots[-1]
    assert last.hour == 20 and last.minute == 30


def test_rejects_8am_request(monday):
    ok, reason = _preferred_time_allowed("8 AM", monday)
    assert ok is False
    assert "9 AM" in reason


def test_rejects_10pm_request(monday):
    ok, reason = _preferred_time_allowed("10 PM", monday)
    assert ok is False


def test_accepts_6pm_request(monday):
    ok, _ = _preferred_time_allowed("6 PM", monday)
    assert ok is True


def test_book_rejects_outside_hours(monday):
    with patch("tools.scheduling._find_doctor", return_value=DOCTOR):
        result = book_appointment(
                caller_phone="+911234567890",
                patient_name="Test Patient",
                concern="checkup",
                doctor_name="Dr. Test",
                preferred_date="kal",
                preferred_time="10 PM",
            )
    assert result["success"] is False
    assert "9 AM" in result["reason"]
