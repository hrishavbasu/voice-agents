"""Load and persist runtime config that overrides .env values."""
import json
from pathlib import Path

_PATH = Path(__file__).parent / "user_settings.json"


def load_user_settings() -> dict:
    if _PATH.exists():
        try:
            return json.loads(_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_user_settings(settings: dict) -> None:
    _PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
