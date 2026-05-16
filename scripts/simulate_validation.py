#!/usr/bin/env python3
"""Validate barge-in logic, emergency→human, and date parsing (no Twilio)."""
from __future__ import annotations

import asyncio
import sys
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def section(title: str) -> None:
    print(f"\n── {title} ──")


async def run_barge_in() -> tuple[int, int]:
    from pipeline.voice_pipeline import VoicePipeline
    import time

    passed = failed = 0
    telephony = MagicMock()
    telephony.clear_playback_buffer = AsyncMock()
    telephony.send_silence = AsyncMock()

    p = VoicePipeline("sim-bi", "+910000000000", telephony)
    p._running = True
    p._tts_playing = True
    p._agent_in_turn = True
    p._playback_until = time.monotonic() + 10
    p._sentences_spoken_this_turn = 1
    p._tts = MagicMock()
    p._tts.cancel = MagicMock()
    p._cancel_prefetch = MagicMock()
    p._speak = AsyncMock()

    with patch("pipeline.voice_pipeline.BARGE_IN_ACK_MODE", "silent"):
        await p._on_speech_started()

    ok = (
        p._tts.cancel.called
        and telephony.clear_playback_buffer.called
        and p._interruption.is_interrupted
        and not p._speak.called
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] speech_started → cancel + clear + no ack")
    return (1, 0) if ok else (0, 1)


async def run_emergency() -> tuple[int, int]:
    from pipeline.voice_pipeline import VoicePipeline

    telephony = MagicMock()
    telephony.clear_playback_buffer = AsyncMock()
    p = VoicePipeline("sim-em", "+910000000000", telephony)
    p._running = True
    p._speak = AsyncMock(return_value=2.0)

    cases = [
        ("severe chest pain", True),
        ("mild chest discomfort", False),
        ("मुझे सांस नहीं आ रही", True),
        ("kal appointment chahiye", False),
    ]
    passed = failed = 0
    for text, should_fire in cases:
        p._running = True
        with patch("pipeline.voice_pipeline.get_transcript", AsyncMock(return_value="")):
            with patch("pipeline.voice_pipeline.get_session", AsyncMock(return_value={})):
                with patch("tools.escalation.escalate_to_human", AsyncMock(return_value={"success": True})) as esc:
                    fired = await p._check_emergency(text)
        ok = fired == should_fire
        print(f"  [{'PASS' if ok else 'FAIL'}] {text[:40]!r} → emergency={fired} (expect {should_fire})")
        if ok:
            passed += 1
        else:
            failed += 1
        if should_fire and ok:
            assert esc.called
    return passed, failed


def run_clinic_hours() -> tuple[int, int]:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from tools.scheduling import _slots_for_date, _preferred_time_allowed

    tz = ZoneInfo("Asia/Kolkata")
    fixed = datetime(2026, 5, 18, 7, 0, 0, tzinfo=tz)  # before 9 AM open
    doctor = {
        "name": "Dr. Test",
        "specialty": "Internal",
        "available_days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"],
    }
    passed = failed = 0
    with patch("tools.scheduling.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        slots = _slots_for_date(doctor, fixed.date())
        ok1 = slots and slots[0].hour == 9 and slots[-1].hour == 20
        print(f"  [{'PASS' if ok1 else 'FAIL'}] slots 9:00 AM – 8:30 PM (last start)")
        passed += ok1
        failed += not ok1
        ok2, _ = _preferred_time_allowed("10 PM", fixed.date())
        print(f"  [{'PASS' if not ok2 else 'FAIL'}] rejects 10 PM")
        passed += not ok2
        failed += ok2
    return passed, failed


def run_call_logs() -> tuple[int, int]:
    import tempfile
    from pathlib import Path

    from pipeline.call_logs import persist_call_log

    with tempfile.TemporaryDirectory() as tmp:
        import pipeline.call_logs as cl

        old_dir = cl.CALL_LOGS_DIR
        old_en = cl.CALL_LOGS_ENABLED
        cl.CALL_LOGS_DIR = tmp
        cl.CALL_LOGS_ENABLED = True
        try:
            path = persist_call_log({
                "call_id": "SIM_VALIDATE",
                "caller_phone": "+910000000000",
                "start_time": 1.0,
                "end_time": 2.0,
                "transcript_lines": ["[USER] test"],
                "flow_state": "ended",
            })
            ok = path is not None and Path(tmp, "SIM_VALIDATE.json").exists()
        finally:
            cl.CALL_LOGS_DIR = old_dir
            cl.CALL_LOGS_ENABLED = old_en
    print(f"  [{'PASS' if ok else 'FAIL'}] per-call json+log write")
    return (1, 0) if ok else (0, 1)


def run_context_trace() -> tuple[int, int]:
    from pipeline.caller_context import (
        apply_context_updates,
        merge_context_from_utterance,
        record_context_event,
    )

    session = {"context_events": [], "caller_name": None}
    u = merge_context_from_utterance(session, "Mera naam Rahul hai")
    _, ev = apply_context_updates(session, u, source="utterance", detail="test")
    events = record_context_event(session, ev)
    ok = u.get("caller_name") == "Rahul" and len(events) == 1
    print(f"  [{'PASS' if ok else 'FAIL'}] context event recorded for name")
    return (1, 0) if ok else (0, 1)


def run_oos_guard() -> tuple[int, int]:
    from pipeline.scope_guard import detect_out_of_scope, should_escalate_oos

    passed = failed = 0
    cases = [
        ("billing detect", detect_out_of_scope("refund my bill") == "billing"),
        ("appointment ok", detect_out_of_scope("book Dr. Sharma kal") is None),
        (
            "repeat escalate",
            should_escalate_oos(
                category="billing",
                strikes=1,
                last_category="billing",
                insists=False,
            ),
        ),
    ]
    for label, ok in cases:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


def run_dates() -> tuple[int, int]:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from tools.scheduling import parse_preferred_date

    tz = ZoneInfo("Asia/Kolkata")
    fixed = datetime(2026, 5, 16, 10, 0, 0, tzinfo=tz)
    cases = [
        ("kal", date(2026, 5, 17)),
        ("parso", date(2026, 5, 18)),
        ("parson subah", date(2026, 5, 18)),
        ("कल", date(2026, 5, 17)),
        ("परसों", date(2026, 5, 18)),
        ("tomorrow", date(2026, 5, 17)),
    ]
    passed = failed = 0
    with patch("tools.scheduling.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        for phrase, expected in cases:
            got = parse_preferred_date(phrase)
            ok = got == expected
            print(f"  [{'PASS' if ok else 'FAIL'}] {phrase!r} → {got} (expect {expected})")
            if ok:
                passed += 1
            else:
                failed += 1
    return passed, failed


async def main() -> int:
    print("=" * 60)
    print("Validation: barge-in · emergency · context · OOS · dates")
    print("=" * 60)

    p, f = await run_barge_in()
    section("Emergency → human transfer")
    pe, fe = await run_emergency()
    p += pe
    f += fe
    section("Clinic hours 9 AM – 9 PM")
    ph, fh = run_clinic_hours()
    p += ph
    f += fh
    section("Date parsing (kal / parso / …)")
    pd, fd = run_dates()
    p += pd
    f += fd
    section("Per-call local logs")
    pcl, fcl = run_call_logs()
    p += pcl
    f += fcl
    section("Context trace")
    pct, fct = run_context_trace()
    p += pct
    f += fct
    section("Out-of-scope guard")
    po, fo = run_oos_guard()
    p += po
    f += fo

    print("\n" + "=" * 60)
    print(f"Total: {p} passed, {f} failed")
    print("=" * 60)
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
