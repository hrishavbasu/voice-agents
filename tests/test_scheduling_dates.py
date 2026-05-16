"""Tests for Hindi/English relative date parsing (kal, parso, aaj, etc.)."""
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from tools.scheduling import parse_preferred_date, _normalize_date_phrase

TZ = ZoneInfo("Asia/Kolkata")
FIXED_NOW = datetime(2026, 5, 16, 10, 0, 0, tzinfo=TZ)  # Saturday


@pytest.fixture
def fixed_today():
    with patch("tools.scheduling.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_NOW
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        yield date(2026, 5, 16)


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("kal", date(2026, 5, 17)),
        ("Kal ko", date(2026, 5, 17)),
        ("tomorrow", date(2026, 5, 17)),
        ("parso", date(2026, 5, 18)),
        ("parson", date(2026, 5, 18)),
        ("parso subah", date(2026, 5, 18)),
        ("day after tomorrow", date(2026, 5, 18)),
        ("aaj", date(2026, 5, 16)),
        ("today", date(2026, 5, 16)),
        ("कल", date(2026, 5, 17)),
        ("परसों", date(2026, 5, 18)),
        ("आज", date(2026, 5, 16)),
        ("Monday", date(2026, 5, 18)),  # next Monday from Sat May 16
        ("सोमवार", date(2026, 5, 18)),
        ("सोमवार, 18 मई", date(2026, 5, 18)),
        ("20 May", date(2026, 5, 20)),
    ],
)
def test_parse_preferred_date(phrase, expected, fixed_today):
    assert parse_preferred_date(phrase) == expected


def test_normalize_strips_hinglish_noise():
    assert _normalize_date_phrase("Kal ko subah") == "kal"


def test_unknown_date_returns_none(fixed_today):
    assert parse_preferred_date("someday maybe") is None
