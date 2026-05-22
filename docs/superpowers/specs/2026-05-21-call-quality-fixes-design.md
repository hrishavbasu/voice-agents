# Call Quality Fixes — Design Spec

**Date:** 2026-05-21
**Branch:** sarvam_tuned
**Approach:** B — Code + prompt engineering (no TTS parameter changes)
**Source:** Call evaluation sheet, calls 1–4 (19-May and 21-May-2026)

---

## Problem Summary

Four live calls were evaluated. Recurring issues:

| Issue | Calls affected |
|-------|----------------|
| Agent assumed "Thik hai" as caller name | 1 |
| No sense of today's date ("kal", "Saturday" guessed) | 3, 4 |
| Re-asked "why did you call?" mid-conversation | 3, 4 |
| Listed every appointment slot (long, unnatural) | 1 |
| No mention of where booking confirmation is sent | 1 |
| Repetitive acknowledgments ("हाँ ठीक है" loop) | 1, 4 |
| Irate caller not de-escalated | 4 |
| Abrupt disconnect after booking (Closing score: 0) | 1 |
| No proactive close on non-booking calls | 3, 4 |
| Occasionally slipped into pure Hindi (not Hinglish) | 1 |

**Out of scope for this plan:** Response latency, TTS pronunciation quality, between-chunk pauses (architectural Sarvam limits — separate initiative).

---

## Section 1: Code-layer fixes

### 1a. `pipeline/caller_context.py` — Blocked name filter

**Problem:** `_BLOCKED_NAMES` contains `"theek"` but not `"thik"` (common STT alternate spelling). When Sarvam STT transcribes `"ठीक है"` in Roman script as `"thik hai"`, both tokens pass the filter. `_clean_name("thik hai")` returns `"Thik Hai"`, which gets stored as `caller_name`.

**Fix:** Add to `_BLOCKED_NAMES`:
```python
"thik", "hai", "hain", "tha", "thi",
```
These are copula verbs and affirmation completions that should never be treated as name tokens.

### 1b. `services/llm_gemini.py` + `pipeline/voice_pipeline.py` — Message history window

**Problem:** `trim_messages_for_llm` is called with `max_non_system=12` in both `_llm_loop` and `_speak_tool_result`. A full booking flow (name → concern → doctor → date → slots → confirm → book) uses ~14–16 messages. Once the window overflows, the LLM loses the caller's stated concern and re-asks "why did you call?"

**Fix:** Raise `max_non_system` from `12` → `20` in both call sites.

### 1c. `prompts/system_prompt.py` — Current date injection

**Problem:** The system prompt contains no temporal grounding. "kal", "this Saturday", "next Monday" are unresolvable — the LLM guesses or hallucinates dates.

**Fix:** Import `datetime` and inject at the top of the prompt:
```python
from datetime import datetime
today_str = datetime.now().strftime("%A, %d %B %Y")
# Injected into prompt: f"Today's date is {today_str}."
```

---

## Section 2: System prompt changes

All changes are in `prompts/system_prompt.py` → `build_system_prompt()`.

### 2a. Acknowledgment variety — strengthen existing rule

**Problem:** Agent loops on `"हाँ ठीक है"` and `"हाँ चलेगा"` as acknowledgments because the current rule says "vary acknowledgments" without specifics.

**Fix:** Add an explicit banned-repeat rule and an approved rotation pool:
- Never use the same acknowledgment phrase twice in a row.
- Approved short acks: `"बिल्कुल"`, `"sure"`, `"got it"`, `"समझ गए"`, `"ठीक है"`, `"हाँ"`, `"ji"`. Any single phrase may appear at most once every 3 turns.

### 2b. Slot listing — group by window, not enumerate

**Problem:** Step 7 example shows the agent reading every slot individually. For 5+ slots this produces a long, robotic list.

**Fix:** Replace step 7 with:
> "If 4 or more slots are available, group them by time window: 'सुबह में दो slots हैं और शाम में तीन — कौन सा time बेहतर रहेगा?' Offer exact times only after the caller picks morning or evening. Never read more than 3 specific times in one turn."

### 2c. Booking confirmation channel

**Problem:** After booking, the agent confirms doctor/date/time/fee but never tells the caller how they will receive the confirmation. Caller has no idea if they'll get an SMS, a call, or nothing.

**Fix:** Add to step 9:
> "After confirming the booking details, always tell the caller: 'आपको SMS पर confirmation आ जाएगी.' (Hinglish/Hindi) or 'You'll receive an SMS confirmation.' (English)."

### 2d. Irate caller handling — new section

**Problem:** No instructions exist for emotional escalation. When Call 4's caller became irate, the agent continued the standard booking flow with no empathy or de-escalation attempt.

**Fix:** Add new section `## Handling frustrated or irate callers`:
- Trigger signals: raised voice, repeated complaints, words like `"yaar"`, `"kya hua"`, `"itni der"`, `"baat nahi sun rahe"`, `"bahut time ho gaya"`.
- Step 1: Acknowledge explicitly — `"Samajh mein aata hai, sorry for the inconvenience."` — before anything else.
- Step 2: Do NOT repeat the question that triggered the frustration.
- Step 3: Offer a concrete next step (e.g. alternate slot, different doctor, or escalation to human).
- Step 4: If frustration continues for 2 more turns, proactively offer `escalate_to_human`.

### 2e. Hinglish language purity

**Problem:** Call 1 noted "sometimes used pure Hindi terms" — the agent occasionally shifted to full Hindi sentences when the caller was Hinglish.

**Fix:** Strengthen the existing Hinglish directive:
> "HINGLISH STRICT: When caller language is Hinglish, every response must be a natural Hindi-English mix. Never respond with an entirely Hindi sentence. Hindi words use Devanagari script; English words stay in Roman. If you find yourself writing a full Hindi sentence, add at least one English word or phrase to it."

### 2f. Non-booking call closing

**Problem:** Calls that don't end in a booking have no proactive close. The agent falls silent and the silence-watch timeout fires with a generic "क्या आप वहाँ हैं?" which feels abrupt.

**Fix:** Add to the system prompt:
> "When you have handled the caller's request and there is nothing left to do (e.g. they declined to book, or their question was answered), do not leave silence. Offer a warm close: 'Theek hai, kuch aur poochhna hai?' Wait for a response. If they say no or stay silent, close warmly: 'Theek hai, dhyan rakhiye. Goodbye!' Do not use generic call-centre closings."

---

## Section 3: Closing experience

### 3a. `voice_pipeline.py` — `_soft_close_after_booking` two-step close

**Problem:** After a successful booking, the agent immediately speaks a goodbye and calls `telephony.hangup()`. The caller has no chance to ask a follow-up question. Call 1 scored 0 on Closing.

**Fix:** Replace the immediate hangup with a two-step sequence:
1. Speak booking confirmation + SMS notice.
2. Ask: `"Kuch aur poochhna hai?"` / `"कुछ और पूछना है?"` / `"Anything else I can help with?"`
3. Wait up to 8 seconds (add `POST_BOOKING_WAIT_SECS = 8` to `config/base_config.py`).
4a. If the caller responds → re-enter normal `_llm_loop` (set `_agent_in_turn = False`, let queue drain).
4b. If silence → speak warm goodbye then `telephony.hangup()`.

Warm goodbye variants:
- Hinglish: `"Theek hai, dhyan rakhiye. Goodbye!"`
- Hindi: `"ठीक है, ध्यान रखिए। अलविदा।"`
- English: `"Alright, take care. Goodbye!"`

---

## Files changed

| File | Change type |
|------|-------------|
| `pipeline/caller_context.py` | Add 5 tokens to `_BLOCKED_NAMES` |
| `services/llm_gemini.py` | `max_non_system` default 12 → 20 |
| `pipeline/voice_pipeline.py` | `max_non_system` call sites 12 → 20; rewrite `_soft_close_after_booking` |
| `prompts/system_prompt.py` | Inject date; 5 prompt section changes |
| `config/base_config.py` | Add `POST_BOOKING_WAIT_SECS = 8` |

---

## Success criteria

The next round of call evaluation should show:
- No instances of affirmation words stored as caller name
- Agent correctly interprets "kal", "Saturday" relative to today's date
- No re-asking of caller's stated concern within the same call
- Slot responses group by morning/evening before listing exact times
- Every booking ends with SMS confirmation mention + follow-up question
- Irate callers receive an explicit empathy acknowledgment before any redirect
- All calls end with a warm close, not silence or timeout
- Closing score ≥ 5 on next evaluation sheet
