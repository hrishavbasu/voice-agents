"""Unit tests for build_system_prompt() empathy level behaviour."""
import sys
import pytest
from unittest.mock import patch


def test_empathy_level_default_is_3(monkeypatch):
    """Default empathy level is 3 when no config or env var is set."""
    monkeypatch.delenv("EMPATHY_LEVEL", raising=False)

    # Remove any cached config modules so reimport reads a clean state
    for key in list(sys.modules):
        if key.startswith("config.base_config") or key.startswith("config.user_settings"):
            del sys.modules[key]

    # load_user_settings() runs at import time — patch it to return empty dict
    with patch("config.user_settings.load_user_settings", return_value={}):
        import config.base_config as bc
        assert bc.EMPATHY_LEVEL == 3


from prompts.system_prompt import build_system_prompt
from unittest.mock import patch as _patch


def _build(empathy_level: int) -> str:
    """Build a system prompt with a patched EMPATHY_LEVEL."""
    import prompts.system_prompt as spm
    with _patch.object(spm, "EMPATHY_LEVEL", empathy_level):
        return build_system_prompt()


def test_level_1_persona_is_efficient():
    prompt = _build(1)
    assert "efficient, precise, and solution-focused" in prompt


def test_level_3_persona_is_warm():
    prompt = _build(3)
    assert "warm, professional, and caring" in prompt


def test_level_5_persona_is_deeply_empathetic():
    prompt = _build(5)
    assert "deeply empathetic and emotionally present" in prompt


def test_level_1_frustrated_caller_acknowledge_once():
    prompt = _build(1)
    assert "Acknowledge once" in prompt


def test_level_5_frustrated_caller_proactive_handoff():
    prompt = _build(5)
    assert "first signal of distress" in prompt


def test_level_3_frustrated_caller_after_2_turns():
    prompt = _build(3)
    assert "2 more turns" in prompt or "2 more frustrated" in prompt
