"""Tests for call log HTTP endpoints."""
import json

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_logs_status(client):
    r = client.get("/logs")
    assert r.status_code == 200
    data = r.json()
    assert "enabled" in data
    assert "endpoints" in data


def test_logs_list_and_get(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.call_logs.CALL_LOGS_ENABLED", True)
    monkeypatch.setattr("pipeline.call_logs.CALL_LOGS_DIR", str(tmp_path))

    from pipeline.call_logs import persist_call_log

    persist_call_log({
        "call_id": "CAapi01",
        "caller_phone": "+91999",
        "start_time": 1.0,
        "end_time": 10.0,
        "transcript_lines": ["[USER] hello", "[ASSISTANT] hi"],
        "caller_name": "Test",
        "flow_state": "ended",
    })

    c = TestClient(app)
    listing = c.get("/logs/calls?limit=10")
    assert listing.status_code == 200
    body = listing.json()
    assert body["enabled"] is True
    assert any(x["call_id"] == "CAapi01" for x in body["calls"])

    detail = c.get("/logs/calls/CAapi01")
    assert detail.status_code == 200
    assert detail.json()["caller_name"] == "Test"
    assert detail.json()["source"] == "file"

    transcript = c.get("/logs/calls/CAapi01/transcript")
    assert transcript.status_code == 200
    assert "[USER] hello" in transcript.text


def test_logs_not_found(client):
    r = client.get("/logs/calls/CA_does_not_exist_xyz")
    assert r.status_code == 404
