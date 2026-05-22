# Apollo Hospital Voice Agent — Design Spec
**Date:** 2026-04-29  
**Branch:** Sarvam  
**Scope:** Sarvam TTS integration + pipeline fixes for Indian call center production readiness

---

## 1. Overview

A real-time voice agent for Apollo Hospital's appointment booking line. Callers speak naturally in Hinglish (Hindi-English mix), English, or Hindi. The agent books doctor appointments, handles barge-in interruptions naturally, and ends calls gracefully.

**Stack:** Twilio (telephony) → Deepgram Nova-3 (STT) → Groq/Cerebras (LLM) → Sarvam (TTS)

---

## 2. Goals

- Sarvam replaces ElevenLabs as the primary TTS provider for all calls
- Default language: Hinglish (`hi-IN` with English mixing)
- Natural Indian phone etiquette: short utterance hold, barge-in acknowledgement, soft-close
- Fix 9 identified bugs without touching working Deepgram STT / LLM / Twilio code

---

## 3. Architecture

```
Incoming Call
    │
    ▼
Twilio WebSocket (/ws/stream/{call_id})
    │
    ├─► DeepgramSTT (Nova-3, hi-IN + en-IN, diarization)
    │       └─► transcript_queue
    │
    ├─► VoicePipeline._llm_loop()
    │       ├─► 400ms filler hold (< 3 word utterances)
    │       ├─► LLM stream (Groq → Cerebras → OpenRouter)
    │       ├─► Tool calls (scheduling, calendar, crm, escalation)
    │       ├─► Soft-close state (5s silence timer after booking)
    │       └─► _speak() → SarvamTTS
    │
    └─► InterruptionController
            ├─► 150ms debounce on SpeechStarted
            ├─► 120ms tail silence on barge-in
            └─► "Haan, bataiye" acknowledgement inject
```

---

## 4. Component Changes

### 4.1 Sarvam TTS (`services/tts_sarvam.py`)

**New file** — ported and fixed from `claude/thirsty-dijkstra-0dabe2`.

- API: `POST https://api.sarvam.ai/text-to-speech`
- Auth: `API-Subscription-Key: {SARVAM_API_KEY}` header
- Voice: `meera` (female, Hinglish-optimised) — configurable via `SARVAM_VOICE` env var
- Language routing:
  - `caller_language == "hindi"` or `"hinglish"` → `target_language_code: "hi-IN"`
  - `caller_language == "english"` → `target_language_code: "en-IN"`
- Output: PCM 8kHz → convert to μ-law for Twilio (via `audioop.lin2ulaw`)
- Streaming: Sarvam returns full audio per request; chunk into 160-byte frames
- Fallback chain: Sarvam → ElevenLabs → Azure
- Error handling: on `4xx`/`5xx` or timeout > 3s, raise `TTSProviderError` → caller catches and tries next provider

**Interface matches existing `tts.py` abstraction:**
```python
async def synthesize(self, text: str, language_code: str = "hi-IN") -> AsyncIterator[bytes]:
    ...
```

### 4.2 TTS Abstraction (`services/tts.py`)

- Add `"sarvam"` case to `get_tts_provider()` factory
- Set `TTS_PROVIDER=sarvam` as default
- Pass `language_code` derived from `session.caller_language` on every `synthesize()` call
- Fallback order: `sarvam → elevenlabs → azure`

### 4.3 `_speak()` in `voice_pipeline.py`

- Replace hardcoded `ELEVENLABS_VOICE_ID` parameter with `language_code=current_lang_code`
- Widen `_tts_playing = True` guard: set at start of full agent turn (before first sentence), clear only when `_llm_loop` returns to `transcript_queue.get()` wait state — eliminates the inter-sentence race window

### 4.4 Short Utterance Hold (`pipeline/voice_pipeline.py`)

Replace flat 150ms post-STT hold with adaptive hold:

```python
word_count = len(user_text.split())
hold_ms = 400 if word_count < 3 else 150
await asyncio.sleep(hold_ms / 1000)
# Drain any follow-on fragment that arrived during hold
while not self._transcript_queue.empty():
    user_text += " " + self._transcript_queue.get_nowait()
```

### 4.5 Interruption Fixes (`pipeline/interruption.py` + `voice_pipeline.py`)

**Tail silence:** increase `send_silence(50)` → `send_silence(120)` in `_on_speech_started()`

**Barge-in acknowledgement:** after confirmed interrupt (agent had spoken ≥ 1 sentence before cut off), prepend a short filler before processing:
```python
_BARGE_IN_ACK = {
    "hindi":   ["haan, bataiye", "ji, haan?", "haan ji?"],
    "hinglish": ["haan, bataiye", "yes, bataiye?", "haan ji?"],
    "english":  ["yes, go ahead", "please go on"],
}
```
Pick randomly, speak it, then process the caller's transcript. Skip if agent had spoken < 1 sentence (avoid acknowledging noise).

### 4.6 Soft-Close State (`pipeline/voice_pipeline.py`)

After `book_appointment` tool returns success, transition to `SOFT_CLOSE` state:

```python
async def _soft_close(self) -> None:
    await self._speak("Koi aur madad chahiye aapko?")
    try:
        user_text = await asyncio.wait_for(
            self._transcript_queue.get(), timeout=5.0
        )
        # Caller responded — continue normally
        await self._process_turn(user_text)
    except asyncio.TimeoutError:
        await self._speak(
            "Dhanyavaad! Apollo Hospital mein aapka swagat hai. Bye!"
        )
        await asyncio.sleep(0.5)   # let final audio drain
        await self._telephony.hangup()
        self._running = False
```

---

## 5. Environment Variables

Added to `.env.example` (with placeholder values — no real keys):

```env
# Sarvam TTS
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_VOICE=meera          # meera | arvind
TTS_PROVIDER=sarvam         # sarvam | elevenlabs | azure (fallback chain)
```

Scrub from `.env.example`:
- All real Twilio, Deepgram, ElevenLabs, OpenRouter, Google, HubSpot credentials → replace with `your_key_here`

---

## 6. Bug Fix Summary

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `services/tts.py` | ElevenLabs hardcoded as primary | Add Sarvam, set as default |
| 2 | `pipeline/voice_pipeline.py` | Flat 150ms hold misses filler utterances | Adaptive 400ms hold for < 3 words |
| 3 | `pipeline/voice_pipeline.py` | `_tts_playing` inter-sentence race | Widen guard to full agent turn |
| 4 | `pipeline/interruption.py` | 50ms tail silence too short for mobile | Increase to 120ms |
| 5 | `pipeline/voice_pipeline.py` | No barge-in acknowledgement | Inject "Haan, bataiye" after confirmed interrupt |
| 6 | `pipeline/voice_pipeline.py` | No soft-close after booking | Add `_soft_close()` with 5s timeout |
| 7 | `services/tts.py` | Hindi voice not switching | Pass `language_code` per turn to Sarvam |
| 8 | `tools/calendar.py` | Dry-run silently succeeds | Raise config error if service account missing in prod |
| 9 | `.env.example` | Real credentials committed | Scrub all keys → placeholders |

---

## 7. Out of Scope (Follow-up)

- **Redis slot locking** — `_booked_slots` is currently in-memory. Concurrent bookings across multiple processes can double-book the same slot. Fix requires Redis infrastructure (`SETNX` atomic lock per slot key). Flagged as separate infrastructure task.
- **Post-call CRM logging** — successful bookings not written back to HubSpot. Separate task.
- **Prosody-based emergency detection** — keyword matching only for now.
- **Call recording** — WAV storage for QA.

---

## 8. Testing Plan

1. **Unit:** `SarvamTTS.synthesize()` returns valid μ-law bytes for Hindi and English text
2. **Unit:** Adaptive hold — 3-word utterance gets 150ms, 2-word gets 400ms, follow-on fragment is merged
3. **Unit:** Soft-close timeout fires hangup after 5s silence; caller response resets it
4. **Integration:** End-to-end call via Twilio dev number — happy path booking in Hinglish
5. **Integration:** Barge-in mid-sentence — verify 120ms silence, acknowledgement phrase, correct re-processing
6. **Integration:** Sarvam API down → fallback to ElevenLabs transparent to caller
7. **Manual:** Listen test — Indian native speaker rates naturalness of Meera voice on booking confirmation phrases

---

## 9. Files Changed

| File | Change Type |
|------|-------------|
| `services/tts_sarvam.py` | New |
| `services/tts.py` | Modified — add Sarvam provider, language_code routing |
| `pipeline/voice_pipeline.py` | Modified — hold logic, _tts_playing guard, soft-close, barge-in ack |
| `pipeline/interruption.py` | Modified — tail silence 50→120ms |
| `tools/calendar.py` | Modified — config guard for dry-run |
| `.env.example` | Modified — add Sarvam vars, scrub real credentials |
