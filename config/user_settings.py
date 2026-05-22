"""Load and persist runtime config that overrides .env values."""
import json
import logging
from pathlib import Path

_PATH = Path(__file__).parent / "user_settings.json"
_log = logging.getLogger(__name__)


def load_user_settings() -> dict:
    if _PATH.exists():
        try:
            return json.loads(_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("user_settings.json is unreadable, using defaults: %s", exc)
            return {}
    return {}


def save_user_settings(settings: dict) -> None:
    try:
        _PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        _log.error("Could not save user_settings.json: %s", exc)
        raise
