"""
Call session management.

A session holds all per-call state:
  - Conversation history (messages list for LLM)
  - Flow state (greeting / troubleshooting / scheduling / escalated)
  - CRM contact ID (resolved async after call starts)
  - Retry counter (for escalation threshold)
  - Call metadata (call_sid, caller_phone, start_time)

Sessions are backed by the memory store (in-memory or Redis).
"""

import asyncio
import logging
import time
from typing import Optional

from memory.store import get_store

logger = logging.getLogger(__name__)

FLOW_STATES = {
    "greeting",
    "troubleshooting",
    "scheduling",
    "crm_lookup",
    "escalating",
    "escalated",
    "ended",
}


async def create_session(call_id: str, caller_phone: str) -> dict:
    """Initialise a new session and persist it."""
    session = {
        "call_id": call_id,
        "caller_phone": caller_phone,
        "messages": [],           # LLM message history
        "flow_state": "greeting",
        "crm_contact_id": None,
        "caller_name": None,
        "caller_concern": None,
        "preferred_doctor": None,
        "preferred_date": None,
        "preferred_time": None,
        "preferred_specialty": None,
        "context_events": [],
        "out_of_scope_strikes": 0,
        "last_oos_category": None,
        "retry_count": 0,
        "start_time": time.time(),
        "transcript_lines": [],   # raw caller/agent lines for post-call log
    }
    store = get_store()
    await store.set(call_id, session)
    logger.info("Session created for call_id=%s caller=%s", call_id, caller_phone)
    return session


async def get_session(call_id: str) -> Optional[dict]:
    store = get_store()
    return await store.get(call_id)


async def update_session(call_id: str, updates: dict) -> dict:
    """Merge *updates* into the existing session and persist."""
    store = get_store()
    session = await store.get(call_id) or {}
    session.update(updates)
    await store.set(call_id, session)
    return session


async def append_message(
    call_id: str,
    role: str,
    content: str,
    *,
    extra: Optional[dict] = None,
) -> None:
    """
    Append a message to the LLM conversation history.

    Pass *extra* to merge additional OpenAI fields into the message dict
    (e.g. tool_calls for assistant messages, tool_call_id for tool messages).
    """
    store = get_store()
    session = await store.get(call_id) or {}
    messages: list = session.get("messages", [])
    msg: dict = {"role": role, "content": content}
    if extra:
        msg.update(extra)
    messages.append(msg)
    session["messages"] = messages

    # Keep transcript readable: skip empty assistant/tool rows used only
    # for tool-call bookkeeping.
    if content.strip():
        lines: list = session.get("transcript_lines", [])
        lines.append(f"[{role.upper()}] {content}")
        session["transcript_lines"] = lines

    await store.set(call_id, session)


async def increment_retry(call_id: str) -> int:
    """Increment retry counter and return new value."""
    store = get_store()
    session = await store.get(call_id) or {}
    count = session.get("retry_count", 0) + 1
    session["retry_count"] = count
    await store.set(call_id, session)
    return count


async def get_transcript(call_id: str) -> str:
    """Return the full transcript as a single string."""
    session = await get_session(call_id)
    if not session:
        return ""
    lines = session.get("transcript_lines", [])
    return "\n".join(lines)


async def end_session(call_id: str) -> None:
    """Mark session as ended, persist local call log, keep session in store for TTL."""
    session = await update_session(
        call_id, {"flow_state": "ended", "end_time": time.time()}
    )
    from pipeline.call_logs import persist_call_log

    persist_call_log(session)
    logger.info("Session ended for call_id=%s", call_id)
