# Sarvam Phase 2: Barge-in Latency & Human-like Pipeline

**Date:** 2026-05-16  
**Branch:** Sarvam  
**Status:** Approved — ready for implementation plan  
**Builds on:** `2026-05-16-sarvam-barge-in-gemini-config-ui-design.md` (Phase 1 complete)

---

## Problem Statement

Live Twilio calls exhibit two compounding issues (user-validated as **E: mix of A + D**):

1. **Slow barge-in stop (A):** Caller interrupts but the agent keeps talking for ~1–2 seconds before going quiet.
2. **Unnatural / slow first response (D):** Voice feels robotic; time from end of caller speech to first agent audio is too long.

Phase 1 delivered Sarvam STT/TTS, HTTP-cancel during Sarvam fetch, Gemini LLM, prosody preprocessing, config UI, and unit tests (21/21 barge-in/STT/prosody). Phase 2 addresses **ordering bugs, playback buffering, prefetch latency, and validation** — not a provider swap.

---

## Root Cause Analysis

### Barge-in latency (A)

| Layer | Issue |
|--------|--------|
| `_on_speech_started` order | `clear_playback_buffer()` runs **after** `interruption.trigger()`, which **sleeps 150ms** before setting `is_interrupted`. |
| `_speak` loop | Exits on `is_interrupted` only; does not check `TTSService.cancel()` — playback can continue during debounce. |
| Sarvam TTS | Batch HTTP per sentence; audio already sent to Twilio plays until `clear` propagates. |
| Sarvam STT VAD | `speech_started` fires after `interrupt_min_speech_frames` (default 3) — adds detection delay. |

### Human-like / first-audio latency (D)

| Layer | Issue |
|--------|--------|
| Serial pipeline | First clause waits for Gemini TTFT **plus** full Sarvam HTTP — no prefetch while playing. |
| Multi-sentence turns | Each sentence = another Sarvam round-trip. |
| Barge-in ack | Fixed phrases after every interrupt (≥1 sentence) feel robotic. |
| Config drift | `.env.example` out of sync with code defaults (`groq`/`meera` vs `gemini`/`pavithra`). |

### Inherent ceiling (not fixable in Phase 2)

Sarvam TTS has **no streaming API** — minimum per-clause latency is one HTTP round-trip (~300–800ms typical). Goal is to hide this via prefetch and shorter first clauses, not eliminate it.

---

## Approach: Phase 2 Hot Path + Prefetch (Approach C)

Combine immediate barge-in teardown (Approach A), latency/voice tuning (Approach B), and a mock telephony E2E test. No new providers; no Redis/calendar work in this phase.

---

## Section 1: Barge-in Hot Path

### 1.1 `InterruptionController` split

**File:** `pipeline/interruption.py`

Add two concepts:

- **`trigger_immediate()`** — sets `_interrupted = True` with no sleep. Used to stop `_speak` and LLM playback loops.
- **`trigger_ack_gate()`** (optional) — existing 150ms debounce, used **only** to decide whether to play a barge-in acknowledgment phrase (noise vs real speech).

`trigger()` becomes an alias for `trigger_immediate()` or is replaced; debounce moves out of the stop-playback path.

### 1.2 `_on_speech_started` reorder

**File:** `pipeline/voice_pipeline.py`

New sequence:

```
1. self._tts.cancel()
2. await self._telephony.clear_playback_buffer()
3. await self._interruption.trigger_immediate()
4. Discard stale transcripts in _transcript_queue
5. await self._telephony.send_silence(80)
6. If ack warranted (see 1.4): debounce → speak ack
```

### 1.3 `_speak` cancel awareness

**File:** `pipeline/voice_pipeline.py`

In the frame-send loop, break when **either**:

- `self._interruption.is_interrupted`, or
- `self._tts` cancel flag is set (expose read-only `is_cancelled` on `TTSService`).

Do not flush padded remainder frames if cancelled.

### 1.4 Barge-in acknowledgment policy

**New env / config key:** `BARGE_IN_ACK_MODE`

| Value | Behavior |
|--------|----------|
| `silent` | Never play ack after barge-in |
| `sometimes` (default) | Play ack only if ≥1 sentence spoken **and** no partial transcript already in queue |
| `always` | Current behavior (ack after ≥1 sentence) |

Ack phrases remain in `_BARGE_IN_ACK`; vary pool per language as today.

### 1.5 Sarvam STT VAD defaults (tunable via `/config/ui`)

**File:** `config/base_config.py`

Production trial defaults:

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| `SARVAM_STT_INTERRUPT_MIN_FRAMES` | 3 | **2** |

Keep `SARVAM_STT_HIGH_VAD=true`, `SARVAM_STT_VOLUME_THRESHOLD=-40` unless live testing shows false triggers.

---

## Section 2: Human-like Latency

### 2.1 Sentence prefetch

**File:** `pipeline/voice_pipeline.py` (or small helper `pipeline/tts_prefetch.py`)

While playing sentence/clause **N**:

- If LLM has already yielded sentence **N+1**, start `sarvam_synthesize` in a background `asyncio.Task` and store result in a one-slot prefetch buffer.
- On barge-in or turn end: cancel prefetch task and discard buffer.
- On speak **N+1**: use prefetched audio if ready; else await synthesis as today.

Constraints:

- Max **1** prefetched utterance (memory bound).
- Prefetch key includes text, language, speaker, pitch — same as TTS cache key.
- Respect `cancelled_flag` on prefetch task.

### 2.2 Startup warm-cache

**File:** `services/tts_sarvam.py` or `main.py` lifespan hook

At app startup (if `TTS_PROVIDER=sarvam` and API key present), fire-and-forget synthesis for:

- Default greeting (from pipeline or config)
- All `TOOL_FILLERS` strings (per language)
- All `_BARGE_IN_ACK` strings

Log failures; do not block server start.

### 2.3 First-clause length guard

**File:** `services/llm_gemini.py`

After `_clause_split`, if this is the **first** sentence of a turn and word count > **15**, split at first comma or yield first 15 words as clause 1 and defer remainder (same buffer logic as short-clause merge).

**File:** `prompts/system_prompt.py`

Add one line: first spoken reply should be ≤15 words before the first question.

### 2.4 Voice defaults & config UI

**Files:** `config/base_config.py`, `static/config.html`, `.env.example`

| Setting | Phase 2 default |
|---------|-----------------|
| `SARVAM_TTS_MODEL` | `bulbul:v3` |
| `SARVAM_TTS_SPEAKER` | `pavithra` |
| `SARVAM_TTS_PACE` | `0.92` |
| `SARVAM_TTS_LOUDNESS` | `1.3` |
| `LLM_PROVIDER` | `gemini` |
| `STT_PROVIDER` | `sarvam` |
| `TTS_ALLOW_FALLBACK` | `false` |

Align `.env.example` with these defaults (placeholders only for secrets).

---

## Section 3: Testing & Success Criteria

### 3.1 New E2E mock test

**File:** `tests/test_barge_in_e2e.py`

Mock `TelephonySession` that records:

- Timestamps of `send_audio` calls
- Timestamp of `clear_playback_buffer`

Simulate:

1. Pipeline starts `_speak` with pre-baked long audio (or mocked slow synthesize)
2. Fire `_on_speech_started` mid-playback
3. Assert: `clear` sent; no `send_audio` after `T_start + 400ms` (configurable tolerance)

### 3.2 Live call checklist (manual)

| # | Scenario | Pass |
|---|----------|------|
| 1 | Interrupt mid-sentence during booking flow | Agent stops immediately; hears caller |
| 2 | Interrupt during tool filler (“ek minute”) | Filler stops; agent responds to new intent |
| 3 | Interrupt during long slot list | Stops without repeating full list |

### 3.3 Regression

- Existing: `test_tts_sarvam_barge_in.py`, `test_stt_sarvam.py`, `test_prosody.py` — all pass
- Full suite: `pytest tests/` — target 40/40 (requires `google-genai` in `.venv`)

### 3.4 Success metrics

| Metric | Target |
|--------|--------|
| Mock E2E stop latency | < 400ms from `speech_started` to last audio frame |
| Live subjective barge-in | “Immediate” (user report) |
| Time to first audio after caller stops | < 2s p50 on typical turn |
| Voice quality | Indian receptionist tone; no markdown read aloud |

---

## Section 4: Files Changed

| File | Change |
|------|--------|
| `pipeline/interruption.py` | `trigger_immediate`, ack debounce split |
| `pipeline/voice_pipeline.py` | Reorder barge-in; cancel-aware `_speak`; ack mode; prefetch integration |
| `pipeline/tts_prefetch.py` | New (optional) — one-slot prefetch helper |
| `services/tts.py` | `is_cancelled` property |
| `services/tts_sarvam.py` | Warm-cache helper; ensure prefetch uses same cache |
| `services/llm_gemini.py` | First-clause length guard |
| `config/base_config.py` | New env vars; STT default 2 frames; TTS v3 defaults |
| `prompts/system_prompt.py` | Short first-reply instruction |
| `static/config.html` | Expose `BARGE_IN_ACK_MODE`, STT interrupt frames |
| `.env.example` | Sync with Sarvam/Gemini defaults |
| `tests/test_barge_in_e2e.py` | New |

**Unchanged:** telephony WebSocket protocol, tool implementations, session store, CRM/scheduling logic.

---

## Section 5: Out of Scope

- True streaming Sarvam TTS
- Reverting to Deepgram STT for barge-in
- Redis session store / multi-instance deployment
- Production calendar/CRM dry-run guards
- ElevenLabs/Deepgram TTS fallback (keep `TTS_ALLOW_FALLBACK=false` in prod)
- Sanitizing historical `.env` commits (separate hygiene task)

---

## Section 6: Rollout

1. Implement Phase 2 on branch `Sarvam`
2. Run `pytest tests/` + manual live checklist ×3
3. Tune VAD via `/config/ui` if false barge-ins on noisy lines
4. Deploy with `TTS_PROVIDER=sarvam`, `STT_PROVIDER=sarvam`, `LLM_PROVIDER=gemini`

---

## Approval

- **User:** approved 2026-05-16 (“ok” after Approach C design presentation)
- **Next step:** implementation plan via `writing-plans` skill (after user reviews this spec file)
