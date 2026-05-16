"""
Persist per-call transcripts and session metadata to local files.

Files written on session end (one call_id per pair):
  logs/calls/{call_id}.json  — structured session snapshot
  logs/calls/{call_id}.log   — plain-text transcript
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.base_config import CALL_LOGS_DIR, CALL_LOGS_ENABLED

logger = logging.getLogger(__name__)

_UNSAFE_CALL_ID = re.compile(r"[^\w.\-]+")


def _safe_call_id(call_id: str) -> str:
    """Filesystem-safe call id (Twilio SID + fallback)."""
    cleaned = _UNSAFE_CALL_ID.sub("_", (call_id or "unknown").strip())
    return cleaned[:128] or "unknown"


def _logs_root() -> Path:
    root = Path(CALL_LOGS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_call_log_record(session: dict) -> dict[str, Any]:
    """Serializable snapshot for JSON log (excludes huge tool payloads if needed)."""
    start = session.get("start_time")
    end = session.get("end_time") or time.time()
    duration = round(float(end) - float(start), 2) if start else None

    transcript_lines = session.get("transcript_lines") or []
    return {
        "call_id": session.get("call_id"),
        "caller_phone": session.get("caller_phone"),
        "flow_state": session.get("flow_state"),
        "start_time": start,
        "end_time": end,
        "duration_seconds": duration,
        "caller_language": session.get("caller_language"),
        "caller_name": session.get("caller_name"),
        "caller_concern": session.get("caller_concern"),
        "preferred_doctor": session.get("preferred_doctor"),
        "preferred_date": session.get("preferred_date"),
        "preferred_time": session.get("preferred_time"),
        "preferred_specialty": session.get("preferred_specialty"),
        "crm_contact_id": session.get("crm_contact_id"),
        "out_of_scope_strikes": session.get("out_of_scope_strikes"),
        "last_oos_category": session.get("last_oos_category"),
        "context_events": session.get("context_events") or [],
        "transcript_lines": transcript_lines,
        "transcript": "\n".join(transcript_lines),
        "message_count": len(session.get("messages") or []),
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }


def format_call_log_text(record: dict[str, Any]) -> str:
    """Human-readable call log body."""
    lines = [
        f"Call ID: {record.get('call_id')}",
        f"Caller: {record.get('caller_phone')}",
        f"Duration: {record.get('duration_seconds')}s",
        f"Flow: {record.get('flow_state')}",
        f"Language: {record.get('caller_language')}",
        f"Patient: {record.get('caller_name')}",
        f"Concern: {record.get('caller_concern')}",
        f"Doctor: {record.get('preferred_doctor')} | Date: {record.get('preferred_date')} | Time: {record.get('preferred_time')}",
        f"OOS strikes: {record.get('out_of_scope_strikes')} ({record.get('last_oos_category')})",
        f"Logged at: {record.get('logged_at')}",
        "",
        "── Transcript ──",
        record.get("transcript") or "(empty)",
        "",
    ]
    events = record.get("context_events") or []
    if events:
        lines.append("── Context events ──")
        for ev in events:
            lines.append(
                f"  [{ev.get('source')}] {ev.get('field')}={ev.get('value')!r} — {ev.get('detail', '')[:80]}"
            )
        lines.append("")
    return "\n".join(lines)


def list_call_logs(*, limit: int = 50) -> list[dict[str, Any]]:
    """Summaries of saved call logs, newest first."""
    if not CALL_LOGS_ENABLED:
        return []
    root = _logs_root()
    if not root.exists():
        return []
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    summaries: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable log %s: %s", path, exc)
            continue
        summaries.append({
            "call_id": data.get("call_id") or path.stem,
            "caller_phone": data.get("caller_phone"),
            "caller_name": data.get("caller_name"),
            "flow_state": data.get("flow_state"),
            "duration_seconds": data.get("duration_seconds"),
            "logged_at": data.get("logged_at"),
            "transcript_preview": (data.get("transcript") or "")[:120],
            "has_log_file": (root / f"{path.stem}.log").exists(),
        })
    return summaries


def load_call_log_json(call_id: str) -> Optional[dict[str, Any]]:
    """Load structured log for call_id, or None if missing."""
    path = _logs_root() / f"{_safe_call_id(call_id)}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read call log %s: %s", path, exc)
        return None


def load_call_log_text(call_id: str) -> Optional[str]:
    """Load plain-text log for call_id."""
    path = _logs_root() / f"{_safe_call_id(call_id)}.log"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to read call log text %s: %s", path, exc)
        return None


def call_logs_status() -> dict[str, Any]:
    """Logging subsystem status for /logs."""
    root = _logs_root()
    count = len(list(root.glob("*.json"))) if root.exists() else 0
    return {
        "enabled": CALL_LOGS_ENABLED,
        "directory": str(root.resolve()),
        "saved_call_count": count,
    }


def persist_call_log(session: Optional[dict]) -> Optional[Path]:
    """
    Write JSON + text logs for this call. Returns json path or None if disabled/empty.
    """
    if not CALL_LOGS_ENABLED or not session:
        return None

    call_id = session.get("call_id") or "unknown"
    safe_id = _safe_call_id(str(call_id))
    root = _logs_root()
    json_path = root / f"{safe_id}.json"
    log_path = root / f"{safe_id}.log"

    record = build_call_log_record(session)
    text = format_call_log_text(record)

    try:
        json_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_path.write_text(text, encoding="utf-8")
        logger.info("Call log saved: %s", json_path)
        return json_path
    except OSError as exc:
        logger.error("Failed to write call log for %s: %s", call_id, exc)
        return None
