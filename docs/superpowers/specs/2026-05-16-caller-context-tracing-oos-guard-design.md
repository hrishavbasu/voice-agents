# Caller Context Tracing & Out-of-Scope Guard — Design Spec

**Date:** 2026-05-16  
**Status:** Approved — implementation per plan `docs/superpowers/plans/2026-05-16-caller-context-tracing-oos-guard.md`  
**Branch context:** Sarvam voice agent (`voice-agents`)

## Summary

Two complementary improvements for Apollo Hospitals voice agent calls:

1. **Traceable caller context** — auditable record of when name, concern, and booking preferences are captured and how they flow into the LLM and tools.
2. **Out-of-scope guard (option C)** — on non-appointment topics, politely decline first; transfer to a human only if the caller insists or repeats the same off-topic request.

Emergency handling remains unchanged (immediate 108 advisory + transfer).

---

## Goals

| Goal | Success metric |
|------|----------------|
| Name/context persistence is observable | `context_events` on session; grep-friendly logs |
| Name not re-asked mid-call | Covered by existing `caller_context` + tests |
| OOS first ask → decline, stay on line | Deterministic TTS; no `escalate_to_human` |
| OOS repeat or insist → transfer | `escalate_to_human` called |
| E2E validation without Twilio | Extended `simulate_validation.py` + pytest |

## Non-goals

- Full intent classification model / pre-LLM router for every turn
- Post-call CRM write-back for context events
- UI dashboard for traces

---

## Section 1: Caller Context Tracing

### Current flow (baseline)

```
STT final → append_message(user)
         → merge_context_from_utterance(session, text) → update_session
         → build_system_prompt(session) → LLM
         → tools update preferred_* / book_appointment backfill
```

CRM async path: `merge_context_from_crm` seeds `caller_name` when HubSpot has a record.

### New: `context_events` on session

Append-only list (cap **20** entries per call):

```python
{
  "ts": float,           # time.time()
  "field": str,          # e.g. "caller_name"
  "value": str,
  "source": str,         # "utterance" | "crm" | "tool"
  "detail": str,         # truncated utterance or tool name
}
```

### Write points

| Location | Fields | Source |
|----------|--------|--------|
| `merge_context_from_utterance` | `caller_name`, `caller_concern` | `utterance` |
| `merge_context_from_crm` | `caller_name` | `crm` |
| `check_doctor_slots` handler | `preferred_doctor`, `preferred_date` | `tool` |
| `list_doctors` handler | `preferred_specialty` | `tool` |
| `book_appointment` handler | all booking fields | `tool` |

Helper: `record_context_event(session, field, value, source, detail)` in `pipeline/caller_context.py`.

### Read points (unchanged behavior, documented)

- `format_context_for_prompt(session)` → LLM system prompt block
- `_build_greeting()` → personalized greeting when name known
- `book_appointment` tool args → backfill from session when LLM omits fields

### Logging

On each event:

```
INFO context_update call_id=%s field=%s value=%r source=%s
```

### API surface

- `record_context_event(existing_session, updates: dict, source, detail) -> dict`  
  Merges field updates **and** appends events only for keys that actually changed.

---

## Section 2: Out-of-Scope Guard (Option C)

### In scope

- Doctor appointment booking, slot checks, doctor listing
- Clinic information present in the knowledge base (hours, fees, specialties)
- Emergency keywords → existing `_check_emergency` path (no change)

### Out of scope (decline path)

| Category | Example phrases |
|----------|-----------------|
| `billing` | bill, invoice, payment dispute, refund |
| `insurance` | claim, coverage, empanelment, TPA |
| `lab` | lab report, test results, blood report |
| `pharmacy` | medicine stock, prescription refill, dosage |
| `medical_advice` | diagnose, what medicine should I take |
| `general_oos` | ward location, parking, jobs, unrelated services |

Lists live in `pipeline/scope_guard.py` (English keywords + Hindi phrases). Use word-boundary matching where possible to reduce false positives.

### Whitelist (do NOT flag)

Phrases that mention payment/fee **in an appointment context**, e.g.:

- "consultation fee", "doctor fee", "kitna charge", "appointment ke liye payment"

Implemented as `_IN_SCOPE_APPOINTMENT_HINTS` checked before OOS classification.

### Session fields

```python
out_of_scope_strikes: int = 0
last_oos_category: str | None = None
```

### Detection API

```python
def detect_out_of_scope(text: str) -> Optional[str]:
    """Return category label or None."""

def caller_insists_on_human(text: str) -> bool:
    """Transfer intent: human, agent, manager, किसी से बात, etc."""
```

### Pipeline integration (`voice_pipeline._llm_loop`)

Order after `merge_context_from_utterance` and **after** `_check_emergency`:

```
if category := detect_out_of_scope(user_text):
    if caller_insists_on_human(user_text) or (
        session.out_of_scope_strikes >= 1
        and session.last_oos_category == category
    ):
        → speak brief transfer line → escalate_to_human(reason=f"OOS:{category}")
        → stop LLM for this turn (or end call per escalation behavior)
    else:
        → speak decline message (language-aware)
        → increment strikes, set last_oos_category
        → append assistant message to history
        → continue (do NOT call LLM this turn)
```

**First decline:** no `escalate_to_human`.  
**Escalate when:** insist keywords **OR** same category on a subsequent OOS utterance (`strikes >= 1` before increment on repeat — implement as: second detection of same category triggers escalate).

Clarified strike logic:

1. First OOS → `strikes=1`, `last_oos_category=cat`, decline spoken.
2. Second OOS same category **or** insist on first/second → escalate.
3. Different OOS category on second turn → decline again with updated category (strikes=2); third → escalate (optional: escalate on any second insist). **Spec choice:** escalate on (insist) OR (same category twice) OR (`strikes >= 2`).

### Decline messages (match `caller_language`)

| Language | Message (abbreviated) |
|----------|---------------------|
| `english` | "I can only help with doctor appointments here. I can't handle billing on this line. Would you like to book an appointment?" |
| `hindi` | Devanagari equivalent |
| `hinglish` / default | Natural Hinglish mix |

Messages stored in `pipeline/scope_guard.py` as `DECLINE_MESSAGES[lang]`.

Transfer line before escalate (short): "I'll connect you to our team now."

### Prompt changes (`prompts/system_prompt.py`)

Replace current out-of-scope block:

- **Remove:** "immediately call escalate_to_human"
- **Add:** Two-step policy aligned with guard; LLM should not contradict a decline already spoken this turn
- **Add:** After OOS decline, if caller returns to booking, resume from Caller context — do not re-ask name/concern

---

## Section 3: Natural Conversation

- Caller context block rules unchanged (name lock, booking-time confirm only).
- OOS decline is a **terminal action for that turn** (no LLM) to avoid hallucinated billing answers.
- If guard misses a subtle OOS ask, LLM prompt still lists topics and says to decline politely (backup, not primary).

---

## Section 4: Testing Plan

### Unit: `tests/test_scope_guard.py`

- Detects billing, lab, pharmacy, insurance utterances (EN + HI samples).
- Does not flag: "book appointment with Dr. Sharma", "consultation fee kitna hai".
- `caller_insists_on_human` true/false cases.

### Unit: `tests/test_caller_context.py` (extend)

- `record_context_event` appends events, caps at 20.
- Merge from utterance creates event with `source=utterance`.

### Pipeline: `tests/test_oos_pipeline.py`

Mock `VoicePipeline._speak`, `escalate_to_human`:

1. First billing utterance → decline spoken, escalate not called, `strikes==1`.
2. Second billing utterance → escalate called.
3. First billing + "connect me to a person" → escalate on first.

### Context flow: `tests/test_context_flow.py`

Multi-step session simulation:

1. User: "Mera naam Rahul hai" → session has name + event.
2. `build_system_prompt(session)` contains "Do NOT ask" + Rahul.
3. Mock book with empty `patient_name` → uses "Rahul".

### Script: `scripts/simulate_validation.py`

Add sections:

- `run_context_trace()` — merge + events
- `run_oos_guard()` — first decline / second escalate

### Manual (optional)

Twilio: lab report question → decline; insist → transfer.

---

## Section 5: Error Handling

| Case | Behavior |
|------|----------|
| False positive (fee during booking) | Whitelist appointment-fee phrases |
| `escalation_phone` missing | Decline still works; insist path logs error, speaks apology, no crash |
| `context_events` overflow | Drop oldest when len > 20 |
| Emergency + OOS same utterance | Emergency wins (runs first) |

---

## File changes (implementation preview)

| File | Change |
|------|--------|
| `pipeline/caller_context.py` | `record_context_event`, wire events in merge helpers |
| `pipeline/scope_guard.py` | **New** — detect, insist, decline messages |
| `pipeline/session.py` | Default `context_events`, `out_of_scope_strikes`, `last_oos_category` |
| `pipeline/voice_pipeline.py` | OOS check in `_llm_loop`; record events on tool paths |
| `prompts/system_prompt.py` | OOS two-step wording |
| `tests/test_*.py` | New/extended tests |
| `scripts/simulate_validation.py` | OOS + context sections |

---

## Self-review checklist

- [x] No TBD placeholders
- [x] Consistent with option C (decline → then escalate)
- [x] Scope bounded to tracing + OOS + tests
- [x] Emergency path explicitly unchanged
- [x] Strike/escalate logic explicit

---

## Approval

- **Sections 1–5:** Approved by user (2026-05-16, message "1")
- **Decline language:** Match `caller_language` (recommended default; user did not override)
