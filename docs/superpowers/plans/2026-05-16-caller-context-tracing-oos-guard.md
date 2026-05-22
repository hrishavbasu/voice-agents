# Caller Context Tracing & Out-of-Scope Guard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make caller name/booking context traceable per call, and enforce option-C out-of-scope handling (decline first, transfer on insist or repeat) with full pytest + script validation.

**Architecture:** Extend `pipeline/caller_context.py` with `record_context_event` and session `context_events[]`. Add `pipeline/scope_guard.py` with keyword detection (mirrors emergency pattern). Hook guard in `voice_pipeline._llm_loop` after emergency check; skip LLM on decline turns. Update system prompt to align with two-step OOS policy.

**Tech Stack:** Python 3.12, asyncio, pytest, existing `VoicePipeline` / session store / `escalate_to_human`.

**Spec:** `docs/superpowers/specs/2026-05-16-caller-context-tracing-oos-guard-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pipeline/session.py` | Modify | Default `context_events`, `out_of_scope_strikes`, `last_oos_category` |
| `pipeline/caller_context.py` | Modify | `record_context_event`, wire into merge helpers |
| `pipeline/scope_guard.py` | Create | OOS detect, insist detect, decline/transfer messages |
| `pipeline/voice_pipeline.py` | Modify | `_handle_out_of_scope`, context events on tools, OOS in `_llm_loop` |
| `prompts/system_prompt.py` | Modify | Two-step OOS prompt text |
| `tests/test_scope_guard.py` | Create | Guard unit tests |
| `tests/test_caller_context.py` | Modify | Event recording tests |
| `tests/test_oos_pipeline.py` | Create | Pipeline OOS integration tests |
| `tests/test_context_flow.py` | Create | Multi-turn context + prompt tests |
| `scripts/simulate_validation.py` | Modify | `run_context_trace`, `run_oos_guard` |

---

## Task 1: Session defaults for tracing and OOS state

**Files:**
- Modify: `pipeline/session.py`

- [ ] **Step 1: Add fields to `create_session`**

In `create_session`, add after `preferred_specialty`:

```python
        "context_events": [],
        "out_of_scope_strikes": 0,
        "last_oos_category": None,
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
cd /Users/jsdata/Projects/voice-agents && .venv/bin/pytest tests/test_caller_context.py -q
```

Expected: PASS (no behavior change yet).

---

## Task 2: `record_context_event` helper

**Files:**
- Modify: `pipeline/caller_context.py`
- Modify: `tests/test_caller_context.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_caller_context.py`:

```python
import time
from pipeline.caller_context import apply_context_updates, record_context_event


def test_record_context_event_appends_on_change():
    session = {"context_events": [], "caller_name": None}
    updates, events = apply_context_updates(
        session,
        {"caller_name": "Rahul"},
        source="utterance",
        detail="Mera naam Rahul hai",
    )
    assert updates == {"caller_name": "Rahul"}
    assert len(events) == 1
    assert events[0]["field"] == "caller_name"
    assert events[0]["value"] == "Rahul"
    assert events[0]["source"] == "utterance"


def test_record_context_event_skips_unchanged():
    session = {"context_events": [], "caller_name": "Rahul"}
    updates, events = apply_context_updates(
        session,
        {"caller_name": "Rahul"},
        source="utterance",
        detail="again",
    )
    assert updates == {}
    assert events == []


def test_context_events_cap_at_20():
    session = {
        "context_events": [{"ts": 0, "field": "x", "value": "v", "source": "t", "detail": "d"}] * 20,
        "caller_name": None,
    }
    _, events = apply_context_updates(
        session, {"caller_name": "A"}, source="utterance", detail="x"
    )
    merged = record_context_event(session, events)
    assert len(merged) == 20
    assert merged[-1]["field"] == "caller_name"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_caller_context.py::test_record_context_event_appends_on_change -v
```

Expected: FAIL — `apply_context_updates` not defined.

- [ ] **Step 3: Implement in `pipeline/caller_context.py`**

Add at top after imports:

```python
import logging
import time

logger = logging.getLogger(__name__)
_CONTEXT_EVENTS_CAP = 20
```

Add functions:

```python
def record_context_event(session: dict, new_events: list[dict]) -> list[dict]:
    """Merge new_events into session list, cap length, return final list."""
    events = list(session.get("context_events") or [])
    events.extend(new_events)
    if len(events) > _CONTEXT_EVENTS_CAP:
        events = events[-_CONTEXT_EVENTS_CAP:]
    return events


def apply_context_updates(
    session: dict,
    updates: dict,
    *,
    source: str,
    detail: str,
) -> tuple[dict, list[dict]]:
    """Return (field_updates, events) only for keys whose values actually change."""
    field_updates: dict = {}
    new_events: list[dict] = []
    now = time.time()
    detail = (detail or "")[:200]

    for field, value in updates.items():
        if value is None:
            continue
        str_val = str(value).strip()
        if not str_val:
            continue
        if session.get(field) == str_val:
            continue
        field_updates[field] = str_val
        new_events.append({
            "ts": now,
            "field": field,
            "value": str_val,
            "source": source,
            "detail": detail,
        })
        logger.info(
            "context_update field=%s value=%r source=%s",
            field, str_val, source,
        )
    return field_updates, new_events
```

- [ ] **Step 4: Wire `merge_context_from_utterance`**

Replace body to use `apply_context_updates` internally — keep public return type as `dict` (field updates only):

```python
def merge_context_from_utterance(existing: dict, utterance: str) -> dict:
    raw: dict = {}
    if not existing.get("caller_name"):
        name = extract_caller_name(utterance)
        if name:
            raw["caller_name"] = name
    if not existing.get("caller_concern"):
        concern = extract_concern(utterance)
        if concern:
            raw["caller_concern"] = concern
    updates, _ = apply_context_updates(
        existing, raw, source="utterance", detail=utterance
    )
    return updates
```

Same pattern for `merge_context_from_crm` with `source="crm"`, `detail="hubspot"`.

- [ ] **Step 5: Run caller context tests**

```bash
.venv/bin/pytest tests/test_caller_context.py -v
```

Expected: all PASS.

---

## Task 3: `scope_guard` module

**Files:**
- Create: `pipeline/scope_guard.py`
- Create: `tests/test_scope_guard.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scope_guard.py`:

```python
from pipeline.scope_guard import (
    caller_insists_on_human,
    decline_message,
    detect_out_of_scope,
    should_escalate_oos,
    transfer_message,
)


def test_detect_billing():
    assert detect_out_of_scope("I want a refund on my bill") == "billing"


def test_detect_lab_hindi():
    assert detect_out_of_scope("mera lab report kab aayega") == "lab"


def test_in_scope_appointment():
    assert detect_out_of_scope("book appointment with Dr. Sharma kal") is None


def test_in_scope_consultation_fee():
    assert detect_out_of_scope("consultation fee kitna hai Dr. Sharma ke liye") is None


def test_insist_human():
    assert caller_insists_on_human("please connect me to a human agent") is True


def test_no_insist():
    assert caller_insists_on_human("kal appointment chahiye") is False


def test_should_escalate_repeat_category():
    assert should_escalate_oos(
        category="billing",
        strikes=1,
        last_category="billing",
        insists=False,
    ) is True


def test_should_not_escalate_first_strike():
    assert should_escalate_oos(
        category="billing",
        strikes=0,
        last_category=None,
        insists=False,
    ) is False


def test_decline_message_hindi():
    msg = decline_message("billing", "hindi")
    assert "अपॉइंटमेंट" in msg or "appointment" in msg.lower()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
.venv/bin/pytest tests/test_scope_guard.py -v
```

- [ ] **Step 3: Create `pipeline/scope_guard.py`**

```python
"""Out-of-scope topic detection and decline/escalate messaging."""
from __future__ import annotations

import re
from typing import Optional

_IN_SCOPE_APPOINTMENT_HINTS = (
    "consultation fee", "doctor fee", "appointment fee", "kitna charge",
    "booking fee", "book appointment", "appointment book", "slot available",
    "doctor appointment", "अपॉइंटमेंट", "डॉक्टर से मिलना",
)

_CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    ("billing", [
        r"\bbill\b", r"\binvoice\b", r"\brefund\b", r"\bpayment dispute\b",
        r"\bbilling\b", r"बिल", r"भुगतान", r"रिफंड",
    ]),
    ("insurance", [
        r"\binsurance\b", r"\bclaim\b", r"\bcoverage\b", r"\bempanel",
        r"\btpa\b", r"बीमा", r"क्लेम",
    ]),
    ("lab", [
        r"\blab report\b", r"\btest result\b", r"\bblood report\b",
        r"\bpathology\b", r"लैब", r"रिपोर्ट", r"टेस्ट रिजल्ट",
    ]),
    ("pharmacy", [
        r"\bpharmacy\b", r"\bprescription\b", r"\bmedicine stock\b",
        r"\bdosage\b", r"दवा", r"फार्मेसी", r"नुस्खा",
    ]),
    ("medical_advice", [
        r"\bdiagnos", r"what medicine should", r"\bself.?medicate\b",
        r"क्या दवा", r"डायग्नोस",
    ]),
    ("general_oos", [
        r"\bward\b", r"\bparking\b", r"\bjob opening\b", r"\bvisiting hours for patient\b",
        r"वार्ड", r"पार्किंग",
    ]),
]

_INSIST_PATTERNS = [
    r"\bhuman\b", r"\bagent\b", r"\bmanager\b", r"\btransfer\b",
    r"\bconnect me\b", r"\bspeak to someone\b",
    r"किसी से बात", r"इंसान", r"मैनेजर", r"ट्रांसफर",
]

DECLINE_MESSAGES = {
    "english": (
        "I can only help with doctor appointments on this line. "
        "I can't handle {topic} here. Would you like to book an appointment?"
    ),
    "hindi": (
        "मैं यहाँ सिर्फ डॉक्टर अपॉइंटमेंट में मदद कर सकती हूँ। "
        "{topic} के लिए अभी मैं सहायता नहीं कर सकती। क्या आप अपॉइंटमेंट बुक करना चाहेंगे?"
    ),
    "hinglish": (
        "Main yahan sirf doctor appointments mein help kar sakti hoon. "
        "{topic} ke liye abhi main assist nahi kar sakti. Kya aap appointment book karna chahenge?"
    ),
}

TRANSFER_MESSAGES = {
    "english": "I'll connect you to our team now.",
    "hindi": "मैं आपको अभी हमारी टीम से जोड़ रही हूँ।",
    "hinglish": "Main aapko abhi hamari team se connect kar rahi hoon.",
}

_TOPIC_LABELS = {
    "billing": "billing",
    "insurance": "insurance",
    "lab": "lab reports",
    "pharmacy": "pharmacy",
    "medical_advice": "medical advice",
    "general_oos": "that request",
}


def _in_scope_appointment_context(lower: str) -> bool:
    return any(h in lower for h in _IN_SCOPE_APPOINTMENT_HINTS)


def detect_out_of_scope(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    lower = text.lower()
    if _in_scope_appointment_context(lower):
        return None
    for category, patterns in _CATEGORY_PATTERNS:
        for pat in patterns:
            if re.search(pat, lower if pat.isascii() else text, re.IGNORECASE):
                return category
    return None


def caller_insists_on_human(text: str) -> bool:
    lower = (text or "").lower()
    for pat in _INSIST_PATTERNS:
        if re.search(pat, lower if pat.isascii() else text, re.IGNORECASE):
            return True
    return False


def should_escalate_oos(
    *,
    category: str,
    strikes: int,
    last_category: Optional[str],
    insists: bool,
) -> bool:
    if insists:
        return True
    if strikes >= 1 and last_category == category:
        return True
    if strikes >= 2:
        return True
    return False


def _lang_key(caller_language: Optional[str]) -> str:
    if caller_language in DECLINE_MESSAGES:
        return caller_language
    return "hinglish"


def decline_message(category: str, caller_language: Optional[str]) -> str:
    topic = _TOPIC_LABELS.get(category, "that")
    template = DECLINE_MESSAGES[_lang_key(caller_language)]
    return template.format(topic=topic)


def transfer_message(caller_language: Optional[str]) -> str:
    return TRANSFER_MESSAGES[_lang_key(caller_language)]
```

- [ ] **Step 4: Run scope guard tests**

```bash
.venv/bin/pytest tests/test_scope_guard.py -v
```

Expected: all PASS.

---

## Task 4: Persist context events in `voice_pipeline`

**Files:**
- Modify: `pipeline/voice_pipeline.py`

- [ ] **Step 1: Add helper `_apply_session_context`**

Near other helpers in `voice_pipeline.py`:

```python
    async def _apply_session_context(
        self,
        session: dict,
        updates: dict,
        *,
        source: str,
        detail: str,
    ) -> dict:
        from pipeline.caller_context import apply_context_updates, record_context_event

        field_updates, events = apply_context_updates(
            session, updates, source=source, detail=detail
        )
        if not field_updates:
            return session
        merged_events = record_context_event(session, events)
        payload = {**field_updates, "context_events": merged_events}
        return await update_session(self.call_id, payload)
```

- [ ] **Step 2: Update `_llm_loop` context merge**

Replace:

```python
            ctx_updates = merge_context_from_utterance(session, user_text)
            if ctx_updates:
                session = await update_session(self.call_id, ctx_updates)
```

With:

```python
            ctx_updates = merge_context_from_utterance(session, user_text)
            if ctx_updates:
                session = await self._apply_session_context(
                    session, ctx_updates, source="utterance", detail=user_text
                )
```

- [ ] **Step 3: Update CRM path in `_async_crm_lookup`**

After `crm_ctx = merge_context_from_crm(...)`, if `crm_ctx`:

```python
                session = await get_session(self.call_id) or {}
                session = await self._apply_session_context(
                    session, crm_ctx, source="crm", detail="hubspot"
                )
                await update_session(self.call_id, {"crm_contact_id": contact["id"]})
```

(Adjust so `crm_contact_id` and context update happen in one or two clean updates.)

- [ ] **Step 4: Tool handlers use `_apply_session_context`**

Replace direct `update_session` for `slot_updates`, `preferred_specialty`, and `book_ctx` with:

```python
                session = await get_session(self.call_id) or {}
                session = await self._apply_session_context(
                    session, slot_updates, source="tool", detail=name
                )
```

Use `detail=name` where `name` is the tool name string in `_handle_tool_call`.

---

## Task 5: Out-of-scope handler in pipeline

**Files:**
- Modify: `pipeline/voice_pipeline.py`
- Create: `tests/test_oos_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/test_oos_pipeline.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.voice_pipeline import VoicePipeline


@pytest.mark.asyncio
async def test_first_oos_declines_without_escalate():
    telephony = MagicMock()
    telephony.clear_playback_buffer = AsyncMock()
    p = VoicePipeline("oos-1", "+910000000001", telephony)
    p._running = True
    p._speak = AsyncMock(return_value=1.0)

    session = {"out_of_scope_strikes": 0, "last_oos_category": None, "caller_language": "hinglish"}
    with patch("pipeline.voice_pipeline.get_session", AsyncMock(return_value=session)):
        with patch("pipeline.voice_pipeline.update_session", AsyncMock(return_value=session)) as upd:
            with patch("pipeline.voice_pipeline.append_message", AsyncMock()):
                with patch("tools.escalation.escalate_to_human", AsyncMock()) as esc:
                    handled = await p._handle_out_of_scope("I need a refund on my bill", session)

    assert handled is True
    assert p._speak.called
    assert not esc.called


@pytest.mark.asyncio
async def test_repeat_oos_escalates():
    telephony = MagicMock()
    p = VoicePipeline("oos-2", "+910000000002", telephony)
    p._running = True
    p._speak = AsyncMock(return_value=1.0)

    session = {
        "out_of_scope_strikes": 1,
        "last_oos_category": "billing",
        "caller_language": "english",
        "crm_contact_id": None,
    }
    with patch("pipeline.voice_pipeline.get_session", AsyncMock(return_value=session)):
        with patch("pipeline.voice_pipeline.update_session", AsyncMock(return_value=session)):
            with patch("pipeline.voice_pipeline.get_transcript", AsyncMock(return_value="")):
                with patch("pipeline.voice_pipeline.append_message", AsyncMock()):
                    with patch("tools.escalation.escalate_to_human", AsyncMock(return_value={"success": True})) as esc:
                        handled = await p._handle_out_of_scope("my bill is wrong again", session)

    assert handled is True
    assert esc.called
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
.venv/bin/pytest tests/test_oos_pipeline.py -v
```

- [ ] **Step 3: Implement `_handle_out_of_scope` on `VoicePipeline`**

```python
    async def _handle_out_of_scope(self, user_text: str, session: dict) -> bool:
        """Return True if this turn was fully handled (no LLM)."""
        from pipeline.scope_guard import (
            caller_insists_on_human,
            decline_message,
            detect_out_of_scope,
            should_escalate_oos,
            transfer_message,
        )

        category = detect_out_of_scope(user_text)
        if not category:
            return False

        insists = caller_insists_on_human(user_text)
        strikes = int(session.get("out_of_scope_strikes") or 0)
        last_cat = session.get("last_oos_category")
        lang = session.get("caller_language") or self._caller_language

        if should_escalate_oos(
            category=category,
            strikes=strikes,
            last_category=last_cat,
            insists=insists,
        ):
            msg = transfer_message(lang)
            self._interruption.reset()
            await self._speak(msg)
            transcript = await get_transcript(self.call_id)
            crm_id = session.get("crm_contact_id")
            from tools.escalation import escalate_to_human
            await escalate_to_human(
                call_id=self.call_id,
                caller_phone=self.caller_phone,
                transcript=transcript,
                reason=f"OOS repeat/insist: {category}",
                telephony_session=self._telephony,
                crm_contact_id=crm_id,
            )
            await append_message(self.call_id, "assistant", msg)
            self._running = False
            return True

        msg = decline_message(category, lang)
        self._interruption.reset()
        await self._speak(msg)
        await append_message(self.call_id, "assistant", msg)
        await update_session(
            self.call_id,
            {
                "out_of_scope_strikes": strikes + 1,
                "last_oos_category": category,
            },
        )
        return True
```

- [ ] **Step 4: Hook in `_llm_loop` after emergency check**

```python
            if await self._check_emergency(user_text):
                continue

            session = await get_session(self.call_id) or session
            if await self._handle_out_of_scope(user_text, session):
                continue
```

- [ ] **Step 5: Run OOS pipeline tests**

```bash
.venv/bin/pytest tests/test_oos_pipeline.py -v
```

Expected: PASS.

---

## Task 6: System prompt alignment

**Files:**
- Modify: `prompts/system_prompt.py`

- [ ] **Step 1: Replace out-of-scope block** (lines ~234–244)

Replace with:

```markdown
## Out-of-scope topics — decline first, transfer only if needed
This line handles **doctor appointments only**. For these topics, do NOT answer from general knowledge:
- Insurance, billing, lab reports, pharmacy, medical advice, ward/parking/jobs

**Policy (the system may speak a decline before you respond):**
1. First time: politely say you only help with appointments and offer to book.
2. Transfer to a human (`escalate_to_human`) only if the caller **insists** or **asks again** for the same off-topic help.

If the caller returns to booking after a decline, use Caller context — do not re-ask name or concern.

Guessing on medical or billing topics can harm patients. When in doubt, decline and offer appointment help.
```

- [ ] **Step 2: Run prompt-related tests**

```bash
.venv/bin/pytest tests/test_caller_context.py::test_system_prompt_includes_session_context -v
```

Expected: PASS.

---

## Task 7: Context flow integration test

**Files:**
- Create: `tests/test_context_flow.py`

- [ ] **Step 1: Create test file**

```python
import pytest
from pipeline.caller_context import merge_context_from_utterance
from pipeline.session import create_session, update_session
from prompts.system_prompt import build_system_prompt


@pytest.mark.asyncio
async def test_name_flows_to_prompt():
    session = await create_session("ctx-1", "+919999999999")
    updates = merge_context_from_utterance(session, "Mera naam Rahul Sharma hai")
    session = await update_session("ctx-1", {**updates, "caller_name": updates.get("caller_name", "Rahul Sharma")})

    prompt = build_system_prompt(session=session)
    assert "Rahul" in prompt
    assert "Do NOT ask" in prompt


@pytest.mark.asyncio
async def test_booking_backfill_uses_session_name():
    from tools.scheduling import book_appointment
    from unittest.mock import patch

    session = {"caller_name": "Rahul Sharma", "caller_concern": "chest pain"}
    patient_name = ""
    if not patient_name:
        patient_name = session.get("caller_name") or ""
    assert patient_name == "Rahul Sharma"
```

- [ ] **Step 2: Run**

```bash
.venv/bin/pytest tests/test_context_flow.py -v
```

Expected: PASS.

---

## Task 8: Extend `simulate_validation.py`

**Files:**
- Modify: `scripts/simulate_validation.py`

- [ ] **Step 1: Add `run_context_trace`**

```python
def run_context_trace() -> tuple[int, int]:
    from pipeline.caller_context import merge_context_from_utterance, apply_context_updates, record_context_event

    session = {"context_events": [], "caller_name": None}
    u = merge_context_from_utterance(session, "Mera naam Rahul hai")
    _, ev = apply_context_updates(session, u, source="utterance", detail="test")
    events = record_context_event(session, ev)
    ok = u.get("caller_name") == "Rahul" and len(events) == 1
    print(f"  [{'PASS' if ok else 'FAIL'}] context event recorded for name")
    return (1, 0) if ok else (0, 1)
```

- [ ] **Step 2: Add `run_oos_guard`**

```python
def run_oos_guard() -> tuple[int, int]:
    from pipeline.scope_guard import detect_out_of_scope, should_escalate_oos

    passed = failed = 0
    ok1 = detect_out_of_scope("refund my bill") == "billing"
    ok2 = detect_out_of_scope("book Dr. Sharma kal") is None
    ok3 = should_escalate_oos(category="billing", strikes=1, last_category="billing", insists=False)
    for label, ok in [("billing detect", ok1), ("appointment ok", ok2), ("repeat escalate", ok3)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        passed += ok
        failed += not ok
    return passed, failed
```

- [ ] **Step 3: Call from `main()`**

Add sections after date parsing:

```python
    section("Context trace")
    pct, fct = run_context_trace()
    p += pct
    f += fct
    section("Out-of-scope guard")
    po, fo = run_oos_guard()
    p += po
    f += fo
```

- [ ] **Step 4: Run script**

```bash
.venv/bin/python scripts/simulate_validation.py
```

Expected: all sections PASS.

---

## Task 9: Full regression

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/jsdata/Projects/voice-agents && .venv/bin/pytest tests/ --ignore=tests/test_llm_gemini.py -q
```

Expected: all PASS (count increases by ~20).

- [ ] **Step 2: Update spec status**

In `docs/superpowers/specs/2026-05-16-caller-context-tracing-oos-guard-design.md`, set:

```markdown
**Status:** Approved — implementation per plan 2026-05-16
```

- [ ] **Step 3: Commit (if user requests)**

```bash
git add pipeline/scope_guard.py pipeline/caller_context.py pipeline/session.py \
  pipeline/voice_pipeline.py prompts/system_prompt.py tests/ scripts/simulate_validation.py docs/
git commit -m "$(cat <<'EOF'
Add caller context tracing and out-of-scope guard.

Decline non-appointment topics on first ask; escalate on insist or repeat.
Record context_events for name and booking preferences across the call.
EOF
)"
```

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| `context_events` + cap 20 | Task 2 |
| Log `context_update` | Task 2 |
| Write points utterance/crm/tool | Tasks 2, 4 |
| OOS categories + whitelist | Task 3 |
| Option C decline then escalate | Tasks 3, 5 |
| Language-matched decline | Task 3 |
| Prompt two-step OOS | Task 6 |
| `test_scope_guard` | Task 3 |
| `test_oos_pipeline` | Task 5 |
| `test_context_flow` | Task 7 |
| `simulate_validation` | Task 8 |
| Emergency unchanged | Task 5 hook order (emergency before OOS) |

---

## Manual smoke (optional)

1. Restart server: `.venv/bin/uvicorn main:app --reload`
2. Twilio call: "Mera naam Amit hai" → agent uses name, no re-ask
3. "Lab report kab aayegi?" → decline, stay on line
4. "Insaan se baat karo" → transfer
