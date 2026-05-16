"""Tests for config GET/POST endpoints."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    """TestClient with user_settings.json pointing to a temp dir."""
    import config.user_settings as us_mod
    tmp_settings = tmp_path / "user_settings.json"

    with patch.object(us_mod, "_PATH", tmp_settings):
        from main import app
        return TestClient(app)


def test_get_config_returns_empty_dict_when_no_file(client):
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json() == {}


def test_post_config_saves_settings(client, tmp_path):
    import config.user_settings as us_mod
    tmp_settings = tmp_path / "user_settings.json"

    with patch.object(us_mod, "_PATH", tmp_settings):
        payload = {"stt_provider": "sarvam", "sarvam_tts_pace": 0.9}
        response = client.post("/config", json=payload)
        assert response.status_code == 200
        assert response.json() == {"status": "saved"}
        saved = json.loads(tmp_settings.read_text())
        assert saved["stt_provider"] == "sarvam"
        assert saved["sarvam_tts_pace"] == 0.9


def test_get_config_returns_saved_settings(client, tmp_path):
    import config.user_settings as us_mod
    tmp_settings = tmp_path / "user_settings.json"
    tmp_settings.write_text(json.dumps({"llm_provider": "gemini"}))

    with patch.object(us_mod, "_PATH", tmp_settings):
        response = client.get("/config")
        assert response.json()["llm_provider"] == "gemini"


def test_config_ui_returns_html(client):
    response = client.get("/config/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()
