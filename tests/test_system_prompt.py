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
