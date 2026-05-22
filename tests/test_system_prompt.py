"""Unit tests for build_system_prompt() empathy level behaviour."""
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_empathy_level_default_is_3():
    """Default empathy level is 3 when no config or env var is set."""
    import config.user_settings as us_mod
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False
    with patch.object(us_mod, "_PATH", mock_path):
        for mod in list(sys.modules.keys()):
            if mod.startswith("config.base_config"):
                del sys.modules[mod]
        import config.base_config as bc
        assert bc.EMPATHY_LEVEL == 3
