"""Tests for per-call local log files."""
import json
from pathlib import Path

import pytest

from pipeline.call_logs import (
    build_call_log_record,
    format_call_log_text,
    persist_call_log,
    _safe_call_id,
)


def test_safe_call_id_sanitizes():
    assert _safe_call_id("CA123abc") == "CA123abc"
    assert "/" not in _safe_call_id("CA/evil")


def test_build_call_log_record():
    record = build_call_log_record({
        "call_id": "CAtest",
        "caller_phone": "+91999",
        "start_time": 1000.0,
        "end_time": 1060.0,
        "transcript_lines": ["[USER] hello", "[ASSISTANT] namaste"],
        "caller_name": "Rahul",
        "context_events": [{"field": "caller_name", "value": "Rahul", "source": "utterance", "detail": "x"}],
    })
    assert record["duration_seconds"] == 60.0
    assert "hello" in record["transcript"]
    assert record["caller_name"] == "Rahul"


def test_persist_call_log_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.call_logs.CALL_LOGS_ENABLED", True)
    monkeypatch.setattr("pipeline.call_logs.CALL_LOGS_DIR", str(tmp_path))

    path = persist_call_log({
        "call_id": "CAfiletest",
        "caller_phone": "+91111",
        "start_time": 1.0,
        "end_time": 2.0,
        "transcript_lines": ["[USER] hi"],
        "flow_state": "ended",
    })

    assert path is not None
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["call_id"] == "CAfiletest"
    log_path = tmp_path / "CAfiletest.log"
    assert log_path.exists()
    assert "[USER] hi" in log_path.read_text(encoding="utf-8")


def test_persist_disabled(monkeypatch):
    monkeypatch.setattr("pipeline.call_logs.CALL_LOGS_ENABLED", False)
    assert persist_call_log({"call_id": "x"}) is None


@pytest.mark.asyncio
async def test_end_session_writes_log(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.call_logs.CALL_LOGS_ENABLED", True)
    monkeypatch.setattr("pipeline.call_logs.CALL_LOGS_DIR", str(tmp_path))

    from pipeline.session import create_session, append_message, end_session

    await create_session("CAendtest", "+92222")
    await append_message("CAendtest", "user", "Mera naam Rahul hai")
    await append_message("CAendtest", "assistant", "Namaste Rahul")
    await end_session("CAendtest")

    json_path = tmp_path / "CAendtest.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "Rahul" in data["transcript"]


@pytest.mark.asyncio
async def test_empty_assistant_messages_not_added_to_transcript_lines():
    from pipeline.session import create_session, append_message, get_session

    await create_session("CAemptymsg", "+91111")
    await append_message("CAemptymsg", "assistant", "", extra={"tool_calls": [{"id": "x"}]})
    await append_message("CAemptymsg", "user", "hello")
    session = await get_session("CAemptymsg")
    lines = session.get("transcript_lines", [])
    assert lines == ["[USER] hello"]
