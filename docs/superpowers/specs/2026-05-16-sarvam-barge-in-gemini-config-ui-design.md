# Voice Agent: Sarvam STT Barge-in, Gemini LLM, Prosody & Config UI

**Date:** 2026-05-16  
**Branch:** Sarvam  
**Status:** Approved — ready for implementation

---

## Problem Statement

Three compounding issues make the current voice agent feel unnatural:

1. **Barge-in is broken during Sarvam TTS synthesis.** `tts_sarvam.py` makes a blocking `await client.post(...)` HTTP call. The `cancelled_flag` is only checked in the chunk-yielding loop *after* the full response arrives. If a caller interrupts during the ~500ms–2s Sarvam synthesis window, the flag is set but the pipeline is stuck waiting for the HTTP response — barge-in appears to fail.

2. **STT uses Deepgram, not Sarvam.** Barge-in detection fires via Deepgram's `SpeechStarted` event, which is not tuned for Indian acoustic environments (traffic noise, TV audio, code-mixed speech). Sarvam's STT WebSocket has native VAD parameters for this exact problem.

3. **No natural prosody.** Sarvam Bulbul (speaker: meera) sounds robotic. No pace, pitch, or loudness tuning. The sentence splitter only breaks at full stops, so long sentences go to TTS as one block — causing 1–2s of silence before the caller hears anything.

Additionally: the LLM (Groq/Llama) doesn't maintain a consistent Hindi/English persona as well as Gemini 2.5 Pro, and there is no UI to tune any of these settings without editing `.env`.

---

## Approach: Three Independent Service Swaps (Approach A)

Replace Deepgram STT with a Sarvam STT WebSocket client using the same `on_transcript` / `on_speech_started` interface. Fix the Sarvam TTS HTTP blocking bug. Add Gemini 2.5 Pro as a new LLM provider. Add prosody preprocessing. Add a config UI. All changes are self-contained — the pipeline orchestration, telephony, tools, and session logic are unchanged.

---

## What Changes / What Stays the Same

**Unchanged:**
- `pipeline/interruption.py` — InterruptionController, debounce logic
- `pipeline/voice_pipeline.py` — orchestration loop, `_speak()`, tool handling (except prosody additions)
- `services/telephony.py` — Twilio/Telnyx WebSocket handling
- All tool implementations (`tools/scheduling.py`, `tools/crm.py`, `tools/escalation.py`)
- `pipeline/session.py` — Redis/in-memory session management
- `prompts/system_prompt.py` — minor addition only (Gemini anti-markdown + prosody rhythm instruction)

**Changed or added:**

| File | Type | Change |
|---|---|---|
| `services/stt_sarvam.py` | New | Sarvam STT WebSocket client |
| `services/stt.py` | New | STT provider router (Sarvam or Deepgram) |
| `services/llm_gemini.py` | New | Gemini 2.5 Pro streaming client |
| `services/llm.py` | Modified | Route to Gemini when `LLM_PROVIDER=gemini` |
| `services/tts_sarvam.py` | Modified | Fix HTTP-blocking bug; add pace/pitch/loudness params |
| `pipeline/voice_pipeline.py` | Modified | Prosody processor, silence timeout, import from STT router |
| `config/base_config.py` | Modified | New env vars; load `user_settings.json` overrides |
| `config/user_settings.py` | New | JSON config loader/writer |
| `static/config.html` | New | Single-file admin config UI |
| `main.py` | Modified | Config API endpoints, serve static config page |
| `requirements.txt` | Modified | Add `google-genai` |

---

## Section 1: Sarvam STT WebSocket Client

### File: `services/stt_sarvam.py`

Replaces `DeepgramSTT` with an identical interface. The pipeline calls only `connect()`, `send_audio()`, `close()` — so `voice_pipeline.py` changes only its import.

**Interface (identical to DeepgramSTT):**
```python
class SarvamSTT:
    async def connect(self) -> None
    async def send_audio(self, audio_chunk: bytes) -> None
    async def close(self) -> None
```

**Connection:**
- WebSocket endpoint: check `https://docs.sarvam.ai/api-reference-docs/speech-to-text-translate/translate/ws` for the current WSS URL — store as `SARVAM_STT_URL` env var
- Auth: `api-subscription-key` header (same key as TTS)
- Audio format: raw μ-law 8kHz binary frames (exactly what Twilio delivers — no conversion needed)
- A background receive task reads events from the WebSocket

**Barge-in parameters (sent in connection handshake, all env-var overridable):**

| Parameter | Default | Rationale |
|---|---|---|
| `interrupt_min_speech_frames` | 3 | ~90ms — responsive without false triggers on coughs |
| `min_speech_frames` | 5 | ~150ms of speech needed to register a valid turn |
| `start_speech_volume_threshold` | -40 dB | Ignores TV audio, background hum, traffic |
| `high_vad_sensitivity` | true | Stricter VAD — reduces noise false-positives |
| `negative_frames_count` | 8 | Silence frames before turn ends |
| `negative_frames_window` | 20 | Window for silence detection |

**Event handling:**
- `speech_started` → calls `on_speech_started()` callback → triggers `InterruptionController.trigger()` (same path as Deepgram today)
- `transcript` with `is_final=true` → calls `on_transcript(text, True)` callback
- `transcript` with `is_final=false` → accumulated internally, not forwarded (Sarvam emits interim results; we wait for final to avoid triggering the LLM on fragments)

**Keepalive:** ping every 8s — prevents WebSocket timeout during TTS playback when caller mic is suppressed by echo cancellation.

**Speaker noise filtering:** Since Sarvam STT does not support diarization (unlike Deepgram Nova-3), the current speaker-lock logic (`_primary_speaker`) is removed. The `start_speech_volume_threshold` VAD parameter serves as the noise gate instead.

### File: `services/stt.py` (new router)

```python
STT_PROVIDER = os.getenv("STT_PROVIDER", "sarvam")

def get_stt(on_transcript, on_speech_started):
    if STT_PROVIDER == "deepgram":
        from services.stt_deepgram import DeepgramSTT
        return DeepgramSTT(on_transcript, on_speech_started)
    from services.stt_sarvam import SarvamSTT
    return SarvamSTT(on_transcript, on_speech_started)
```

`services/stt.py` (the current Deepgram file) is renamed to `services/stt_deepgram.py` — no logic changes. `voice_pipeline.py` imports `get_stt` from `services.stt` instead of `DeepgramSTT` from `services.stt`.

---

## Section 2: Sarvam TTS Fixes

### File: `services/tts_sarvam.py`

**Fix 1 — HTTP blocking bug:**

Wrap `client.post(...)` in a background asyncio task, polling `cancelled_flag[0]` every 50ms:

```python
# Check before even starting the request (most common case: barge-in between sentences)
if cancelled_flag[0]:
    return

fetch_task = asyncio.create_task(client.post(_SARVAM_TTS_URL, headers=headers, json=payload))
while not fetch_task.done():
    if cancelled_flag[0]:
        fetch_task.cancel()
        return
    await asyncio.sleep(0.05)
response = await fetch_task
```

**Fix 2 — Prosody parameters:**

Add `pace`, `pitch`, `loudness` to the API payload, sourced from config (env-var or `user_settings.json`):

```python
payload = {
    "text": text,
    "target_language_code": language_code,
    "model": SARVAM_TTS_MODEL,          # default: bulbul:v2
    "speaker": SARVAM_TTS_SPEAKER,       # default: pavithra
    "speech_sample_rate": 8000,
    "pace": SARVAM_TTS_PACE,             # default: 0.9
    "pitch": SARVAM_TTS_PITCH,           # default: 0.0 (+ 0.05 if text ends with ?)
    "loudness": SARVAM_TTS_LOUDNESS,     # default: 1.5
}
```

**Speaker change:** Default `SARVAM_TTS_SPEAKER` changes from `meera` → `pavithra`. Pavithra has a warmer, more conversational cadence. All speakers remain available via config.

---

## Section 3: Gemini 2.5 Pro LLM

### File: `services/llm_gemini.py`

Uses `google-genai` SDK. Exposes the same `stream_response(messages, tools)` async generator contract — yields `str` (sentences) and `dict` (tool calls). `voice_pipeline.py` needs no changes.

**Key settings:**
```python
model           = GEMINI_MODEL        # default: gemini-2.5-pro-preview-06-05 — verify current preview ID at aistudio.google.com
temperature     = LLM_TEMPERATURE     # same as current (0.5)
max_output_tokens = LLM_MAX_TOKENS    # same (200)
thinking_budget = 0                   # CRITICAL: disables chain-of-thought
                                      # Without this: 3-5s silent pause before every response
```

**Format conversions (run once at call setup, not per-request):**

*System prompt:* Passed via `system_instruction` parameter (not as a message in the conversation history). This keeps the context window clean.

*Message roles:* OpenAI `assistant` → Gemini `model`. OpenAI `tool` result messages → Gemini `user` messages with `function_response` parts.

*Tool schemas:* OpenAI `tools` list → Gemini `FunctionDeclaration` list. Fields (`name`, `description`, `parameters`) are identical — only the wrapper type differs. Conversion is a thin one-pass transform at the start of each `stream_response` call.

**Streaming:** `generate_content_stream()` yields chunks with `candidates[0].content.parts`. Text parts feed into the same `_SENTENCE_RE` sentence buffer used by the OpenAI path. Function call parts are collected and yielded as `{"type": "tool_call", ...}` dicts after text streaming completes.

**Routing in `services/llm.py`:**
```python
if LLM_PROVIDER == "gemini":
    from services.llm_gemini import stream_response
    # existing Groq/Cerebras/OpenRouter path unchanged for other providers
```

The existing fallback chain (Groq → Cerebras → OpenRouter) is fully preserved.

**New env vars:**
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-pro-preview-06-05
```

---

## Section 4: Prosody & Natural Speech

### Sentence splitter update (`services/llm.py`)

`_SENTENCE_RE` currently splits only at `.!?` and Devanagari dandas `।॥`. Extended to also split at:
- `, ` only when the text segment before that comma is ≥ 5 words — avoids splitting short phrases like "Hello, Priya" or "हाँ, sure" into separate TTS calls
- ` — ` em-dashes
- `: ` colons (when followed by a list)

Effect: *"Doctor Bhaskar has slots at nine, eleven, and three — which works?"* becomes two TTS calls. The first phrase plays ~800ms sooner.

### Prosody preprocessor (`pipeline/voice_pipeline.py`)

New `_add_prosody_markers(text: str) -> str` applied inside `_speak()` before `sarvam_synthesize`:

- Em-dash `—` → `, ` (Sarvam reads `—` as "dash" literally; comma gives the same natural pause)
- Discourse markers without trailing comma get one: `"हाँ "` → `"हाँ, "` / `"ठीक है "` → `"ठीक है, "` / `"actually "` → `"actually, "` / `"so "` → `"so, "` / `"well "` → `"well, "`
- Ellipsis `...` → `,` (Sarvam over-pauses on ellipsis)
- Text ending with `?` → slight pitch bump passed to `sarvam_synthesize` as `pitch=SARVAM_TTS_PITCH + 0.05`

### Anti-markdown post-processing (`pipeline/voice_pipeline.py` `_normalize_for_tts`)

Added to existing normaliser (runs before `_add_prosody_markers`):
- `**text**` → `text`
- `*text*` → `text`
- Leading `- ` or `• ` stripped from sentence fragments
- Trailing ` :` stripped (Gemini appends before lists)

### System prompt addition (`prompts/system_prompt.py`)

One paragraph appended to the "Critical voice rules" section:

> "Write with natural spoken rhythm. Use commas generously at clause boundaries — they become spoken pauses for the caller. End questions with `?`. Never use colons, dashes, asterisks, or bullet points — these are read literally by the voice engine. Vary sentence length: short confirmations (`हाँ, sure.`), medium explanations. Never pack more than one idea into one breath."

### Silence timeout (`pipeline/voice_pipeline.py`)

Background task launched alongside `_llm_loop`:

- Watches time since last transcript was received or agent spoke
- After **10s** of inactivity: speaks language-matched *"क्या आप वहाँ हैं?"* / *"Are you still there?"*
- After a further **8s** of inactivity: speaks a warm goodbye and calls `telephony.hangup()`
- Resets on every new transcript queued or every `_speak()` call
- Timeout durations configurable via `SILENCE_TIMEOUT_SECS` and `SILENCE_HANGUP_SECS` env vars / config UI

---

## Section 5: Config UI

### Architecture

**Backend (2 new endpoints in `main.py`):**

```
GET  /config     — reads config/user_settings.json, returns JSON
POST /config     — validates input, writes config/user_settings.json
GET  /config/ui  — serves static/config.html
```

`config/user_settings.json` is loaded in `config/base_config.py` after `load_dotenv()`. JSON values override `.env` values for all settings exposed in the UI. New calls pick up changes immediately (file is read at pipeline construction time). In-progress calls are unaffected.

**Config loader (`config/user_settings.py`):**
```python
def load_user_settings() -> dict:
    path = Path(__file__).parent / "user_settings.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}

def save_user_settings(settings: dict) -> None:
    path = Path(__file__).parent / "user_settings.json"
    path.write_text(json.dumps(settings, indent=2))
```

**`config/user_settings.json` is gitignored** — it contains API keys and tuning that vary per deployment.

### UI — `static/config.html`

Single self-contained HTML file. No build step, no external JS framework. Tailwind CDN excluded — plain CSS with a clean card layout.

**Five collapsible sections:**

**STT (Speech Recognition)**
- Provider: dropdown (Sarvam / Deepgram)
- Interrupt min speech frames: number input (default: 3)
- Min speech frames: number input (default: 5)
- Volume threshold: slider -60 to 0 dB (default: -40)
- High VAD sensitivity: toggle (default: on)
- Negative frames count: number input (default: 8)
- Negative frames window: number input (default: 20)

**TTS (Text to Speech)**
- Provider: dropdown (Sarvam / ElevenLabs / Azure / Deepgram)
- Speaker: dropdown (pavithra / meera / kalpana / arvind / amol / amartya)
- Model: text input (default: bulbul:v2 — verify availability; fall back to bulbul:v1 if not yet live)
- Pace: slider 0.5–2.0, step 0.05 (default: 0.9)
- Pitch: slider -1.0–1.0, step 0.05 (default: 0.0)
- Loudness: slider 0.5–3.0, step 0.1 (default: 1.5)

**LLM**
- Provider: dropdown (Gemini / Groq / OpenRouter / Cerebras)
- Model: text input (default: gemini-2.5-pro-preview-06-05)
- Temperature: slider 0.0–1.0, step 0.05 (default: 0.5)
- Max tokens: number input (default: 200)
- Thinking budget: number input (Gemini only, default: 0)

**Prosody & Timing**
- Silence prompt timeout: number input in seconds (default: 10)
- Silence hangup timeout: number input in seconds (default: 8)
- Adaptive hold — short utterances: number input in ms (default: 400)
- Adaptive hold — normal utterances: number input in ms (default: 150)

**Agent**
- Agent name: text input (default: Priya)
- Persona: textarea
- Default language: dropdown (Hinglish / Hindi / English)

**UX behaviour:**
- Page loads: `GET /config` populates all fields with current values
- Sliders show live numeric value as thumb moves
- Single **Save Settings** button at bottom — POSTs full settings object
- Inline success/error message below the button (no page reload)
- Sections are collapsed by default; click header to expand

---

## New Environment Variables

```bash
# STT
STT_PROVIDER=sarvam                     # sarvam | deepgram
SARVAM_STT_INTERRUPT_MIN_FRAMES=3
SARVAM_STT_MIN_SPEECH_FRAMES=5
SARVAM_STT_VOLUME_THRESHOLD=-40
SARVAM_STT_HIGH_VAD=true
SARVAM_STT_NEGATIVE_FRAMES_COUNT=8
SARVAM_STT_NEGATIVE_FRAMES_WINDOW=20

# TTS prosody
SARVAM_TTS_SPEAKER=pavithra             # was: meera
SARVAM_TTS_MODEL=bulbul:v2              # was: bulbul:v1
SARVAM_TTS_PACE=0.9
SARVAM_TTS_PITCH=0.0
SARVAM_TTS_LOUDNESS=1.5

# LLM
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-pro-preview-06-05

# Silence handling
SILENCE_TIMEOUT_SECS=10
SILENCE_HANGUP_SECS=8
```

All of the above are also configurable via the config UI and persist to `config/user_settings.json`.

---

## File Layout After Implementation

```
services/
  stt_sarvam.py        ← new
  stt_deepgram.py      ← renamed from stt.py (no logic change)
  stt.py               ← new router
  llm_gemini.py        ← new
  llm.py               ← modified (Gemini routing, sentence splitter)
  tts_sarvam.py        ← modified (HTTP fix, prosody params)
  tts.py               ← unchanged

pipeline/
  voice_pipeline.py    ← modified (prosody processor, silence timeout, STT router import)
  interruption.py      ← unchanged
  session.py           ← unchanged

config/
  base_config.py       ← modified (new vars, load user_settings)
  user_settings.py     ← new (load/save JSON)
  user_settings.json   ← gitignored, created on first save

static/
  config.html          ← new

main.py                ← modified (config endpoints)
requirements.txt       ← add google-genai
prompts/system_prompt.py ← modified (Gemini prose rhythm instruction)
```

---

## Testing Checklist

- [ ] Sarvam STT connects and fires `on_speech_started` when caller speaks
- [ ] Barge-in interrupts TTS during HTTP fetch (cancelled_flag polled correctly)
- [ ] Barge-in interrupts TTS during chunk playback (existing path)
- [ ] Gemini 2.5 Pro responds within 1s (thinking_budget=0 verified in logs)
- [ ] Gemini tool calls (check_doctor_slots, book_appointment) parse correctly
- [ ] Prosody processor does not corrupt Hindi/Devanagari text
- [ ] Sentence splitter produces shorter chunks; first audio plays sooner
- [ ] Silence timeout fires "are you still there?" after 10s of inactivity
- [ ] Silence hangup fires after further 8s, call ends cleanly
- [ ] Config UI loads current settings on page open
- [ ] Config UI saves settings; next call uses new values
- [ ] Sarvam speaker "pavithra" sounds more natural than "meera"
- [ ] Anti-markdown strip removes Gemini bold/bullet artifacts before TTS
