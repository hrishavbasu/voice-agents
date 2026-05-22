# Voice Agent: Sarvam STT, Gemini LLM, Prosody & Config UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Deepgram STT with Sarvam WebSocket STT (native Indian-acoustic barge-in), add Gemini 2.5 Pro as LLM, fix Sarvam TTS HTTP-blocking barge-in bug, add prosody preprocessing for natural speech, and add a web config UI.

**Architecture:** Each service swap (STT, TTS, LLM) uses the same interface contract as today — `voice_pipeline.py` is touched only for prosody and the STT import. Config persists to `config/user_settings.json` which `base_config.py` loads after `.env`, so UI changes apply immediately to new calls.

**Tech Stack:** Python 3.11+, FastAPI, `websockets>=12.0` (Sarvam STT client), `google-genai` (Gemini), `pytest-asyncio` (tests), vanilla HTML/JS (config UI).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `config/user_settings.py` | Create | Load/save `user_settings.json` |
| `config/user_settings.json` | Create (gitignored) | Persisted runtime config |
| `config/base_config.py` | Modify | New env vars + overlay user_settings.json |
| `services/stt_deepgram.py` | Rename from `stt.py` | Deepgram STT (unchanged logic) |
| `services/stt.py` | Create | STT provider router |
| `services/stt_sarvam.py` | Create | Sarvam STT WebSocket client |
| `services/tts_sarvam.py` | Modify | HTTP-blocking fix + pace/pitch/loudness |
| `services/llm_gemini.py` | Create | Gemini 2.5 Pro streaming client |
| `services/llm.py` | Modify | Gemini routing + comma-clause sentence splitter |
| `pipeline/voice_pipeline.py` | Modify | Prosody processor, anti-markdown, silence timeout, STT router import |
| `prompts/system_prompt.py` | Modify | Gemini prose rhythm instruction |
| `main.py` | Modify | Config API endpoints |
| `static/config.html` | Create | Admin config UI |
| `requirements.txt` | Modify | Add `websockets`, `google-genai` |
| `tests/test_stt_sarvam.py` | Create | SarvamSTT event handling tests |
| `tests/test_llm_gemini.py` | Create | Gemini message conversion + streaming tests |
| `tests/test_tts_sarvam_barge_in.py` | Create | TTS cancellation-during-HTTP-fetch tests |
| `tests/test_prosody.py` | Create | Prosody processor + anti-markdown tests |
| `tests/test_config_api.py` | Create | Config GET/POST endpoint tests |

---

## Task 1: Config Foundation

**Files:**
- Create: `config/user_settings.py`
- Modify: `config/base_config.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create `config/user_settings.py`**

```python
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
```

- [ ] **Step 2: Add `config/user_settings.json` to `.gitignore`**

Open `.gitignore` and add:
```
config/user_settings.json
```

- [ ] **Step 3: Add new env vars and user_settings overlay to `config/base_config.py`**

After the existing `load_dotenv()` line, add at the top of the file:
```python
from config.user_settings import load_user_settings as _load_user_settings
_US = _load_user_settings()

def _get(key: str, default):
    """Return user_settings value if present, else env var, else default."""
    if key in _US:
        return _US[key]
    return os.getenv(key, default)
```

Then add these new config vars (after the existing Sarvam TTS block):
```python
# ── STT provider ──────────────────────────────────────────────────────────────
STT_PROVIDER = _get("stt_provider", "sarvam")  # sarvam | deepgram

# Sarvam STT WebSocket — check https://docs.sarvam.ai for current WSS URL
SARVAM_STT_URL = os.getenv("SARVAM_STT_URL", "wss://api.sarvam.ai/speech-to-text-translate/subscribe")
SARVAM_STT_INTERRUPT_MIN_FRAMES = int(_get("stt_interrupt_min_frames", "3"))
SARVAM_STT_MIN_SPEECH_FRAMES    = int(_get("stt_min_speech_frames", "5"))
SARVAM_STT_VOLUME_THRESHOLD     = int(_get("stt_volume_threshold", "-40"))
SARVAM_STT_HIGH_VAD             = str(_get("stt_high_vad", "true")).lower() == "true"
SARVAM_STT_NEGATIVE_FRAMES_COUNT  = int(_get("stt_negative_frames_count", "8"))
SARVAM_STT_NEGATIVE_FRAMES_WINDOW = int(_get("stt_negative_frames_window", "20"))

# ── TTS prosody ───────────────────────────────────────────────────────────────
SARVAM_TTS_PACE     = float(_get("sarvam_tts_pace", "0.9"))
SARVAM_TTS_PITCH    = float(_get("sarvam_tts_pitch", "0.0"))
SARVAM_TTS_LOUDNESS = float(_get("sarvam_tts_loudness", "1.5"))

# Update defaults for speaker and model
SARVAM_TTS_SPEAKER = _get("sarvam_tts_speaker", "pavithra")   # was: meera
SARVAM_TTS_MODEL   = _get("sarvam_tts_model", "bulbul:v2")    # was: bulbul:v1

# ── Gemini LLM ────────────────────────────────────────────────────────────────
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL         = _get("llm_model", os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-06-05"))
LLM_THINKING_BUDGET  = int(_get("llm_thinking_budget", "0"))

# ── Silence timeout ───────────────────────────────────────────────────────────
SILENCE_TIMEOUT_SECS = int(_get("silence_timeout_secs", "10"))
SILENCE_HANGUP_SECS  = int(_get("silence_hangup_secs", "8"))

# ── Adaptive hold (ms) ────────────────────────────────────────────────────────
ADAPTIVE_HOLD_SHORT_MS  = int(_get("adaptive_hold_short_ms", "400"))
ADAPTIVE_HOLD_NORMAL_MS = int(_get("adaptive_hold_normal_ms", "150"))
```

Also update the existing `LLM_PROVIDER` line to use `_get`:
```python
LLM_PROVIDER = _get("llm_provider", os.getenv("LLM_PROVIDER", "gemini"))
```

- [ ] **Step 4: Add dependencies to `requirements.txt`**

Add these two lines after the `httpx` line:
```
websockets>=12.0                # Sarvam STT WebSocket client
google-genai>=1.0.0             # Gemini 2.5 Pro LLM
```

- [ ] **Step 5: Install new dependencies**

```bash
pip install "websockets>=12.0" "google-genai>=1.0.0"
```

Expected: installs without errors.

- [ ] **Step 6: Commit**

```bash
git add config/user_settings.py config/base_config.py requirements.txt .gitignore
git commit -m "feat: add config foundation — user_settings JSON, new env vars for STT/TTS/LLM prosody"
```

---

## Task 2: STT Provider Router

**Files:**
- Rename: `services/stt.py` → `services/stt_deepgram.py`
- Create: `services/stt.py`
- Modify: `pipeline/voice_pipeline.py` (import only)

- [ ] **Step 1: Rename the existing Deepgram STT file**

```bash
git mv services/stt.py services/stt_deepgram.py
```

No logic changes to `stt_deepgram.py` — only the filename changes.

- [ ] **Step 2: Create `services/stt.py` router**

```python
"""STT provider router — returns SarvamSTT or DeepgramSTT based on STT_PROVIDER."""
from config.base_config import STT_PROVIDER


def get_stt(on_transcript, on_speech_started):
    """Return the configured STT client. Both implement connect/send_audio/close."""
    if STT_PROVIDER == "deepgram":
        from services.stt_deepgram import DeepgramSTT
        return DeepgramSTT(on_transcript, on_speech_started)
    from services.stt_sarvam import SarvamSTT
    return SarvamSTT(on_transcript, on_speech_started)
```

- [ ] **Step 3: Update the import in `pipeline/voice_pipeline.py`**

Find and replace the existing import:
```python
from services.stt import DeepgramSTT
```
with:
```python
from services.stt import get_stt
```

Find in `VoicePipeline.start()`:
```python
        self._stt = DeepgramSTT(
            on_transcript=self._on_transcript,
            on_speech_started=self._on_speech_started,
        )
```
Replace with:
```python
        self._stt = get_stt(
            on_transcript=self._on_transcript,
            on_speech_started=self._on_speech_started,
        )
```

- [ ] **Step 4: Verify import resolves**

```bash
python -c "from services.stt import get_stt; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add services/stt_deepgram.py services/stt.py pipeline/voice_pipeline.py
git commit -m "refactor: rename stt.py → stt_deepgram.py, add STT provider router"
```

---

## Task 3: Sarvam STT WebSocket Client

**Files:**
- Create: `services/stt_sarvam.py`
- Create: `tests/test_stt_sarvam.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stt_sarvam.py`:

```python
"""Tests for Sarvam STT WebSocket client."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def make_stt():
    """Factory: return a SarvamSTT with mock callbacks."""
    from services.stt_sarvam import SarvamSTT

    def _make(on_transcript=None, on_speech_started=None):
        transcripts = []
        speech_starts = []

        async def _on_transcript(text, is_final):
            transcripts.append((text, is_final))

        async def _on_speech_started():
            speech_starts.append(True)

        stt = SarvamSTT(
            on_transcript=on_transcript or _on_transcript,
            on_speech_started=on_speech_started or _on_speech_started,
        )
        stt._transcripts = transcripts
        stt._speech_starts = speech_starts
        return stt

    return _make


@pytest.mark.asyncio
async def test_speech_started_fires_callback(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "speech_started"})
    assert len(stt._speech_starts) == 1


@pytest.mark.asyncio
async def test_speech_started_clears_utterance_parts(make_stt):
    stt = make_stt()
    stt._utterance_parts = ["partial", "text"]
    await stt._handle_event({"type": "speech_started"})
    assert stt._utterance_parts == []


@pytest.mark.asyncio
async def test_final_transcript_fires_callback(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "transcript", "transcript": "hello world", "is_final": True})
    assert len(stt._transcripts) == 1
    assert stt._transcripts[0][0] == "hello world"
    assert stt._transcripts[0][1] is True


@pytest.mark.asyncio
async def test_non_final_transcript_accumulates(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "transcript", "transcript": "hello", "is_final": False})
    assert stt._utterance_parts == ["hello"]
    assert len(stt._transcripts) == 0


@pytest.mark.asyncio
async def test_final_transcript_appends_accumulated_parts(make_stt):
    stt = make_stt()
    stt._utterance_parts = ["I want to book"]
    await stt._handle_event({"type": "transcript", "transcript": "an appointment", "is_final": True})
    text = stt._transcripts[0][0]
    assert "I want to book" in text
    assert "an appointment" in text


@pytest.mark.asyncio
async def test_single_word_utterance_discarded(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "transcript", "transcript": "हाँ", "is_final": True})
    assert len(stt._transcripts) == 0


@pytest.mark.asyncio
async def test_empty_transcript_ignored(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "transcript", "transcript": "", "is_final": True})
    assert len(stt._transcripts) == 0


@pytest.mark.asyncio
async def test_unknown_event_type_ignored(make_stt):
    stt = make_stt()
    await stt._handle_event({"type": "heartbeat"})
    assert len(stt._speech_starts) == 0
    assert len(stt._transcripts) == 0
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

```bash
pytest tests/test_stt_sarvam.py -v
```

Expected: `ModuleNotFoundError: No module named 'services.stt_sarvam'`

- [ ] **Step 3: Create `services/stt_sarvam.py`**

```python
"""
Sarvam STT WebSocket client.

Same interface as DeepgramSTT — connect(), send_audio(), close().
Barge-in fires via on_speech_started() callback when Sarvam emits
a speech_started event.
"""
import asyncio
import json
import logging
import os
from typing import Callable, Awaitable, Optional

import websockets

from config.base_config import (
    SARVAM_API_KEY,
    SARVAM_STT_URL,
    SARVAM_STT_INTERRUPT_MIN_FRAMES,
    SARVAM_STT_MIN_SPEECH_FRAMES,
    SARVAM_STT_VOLUME_THRESHOLD,
    SARVAM_STT_HIGH_VAD,
    SARVAM_STT_NEGATIVE_FRAMES_COUNT,
    SARVAM_STT_NEGATIVE_FRAMES_WINDOW,
)

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, bool], Awaitable[None]]
SpeechStartedCallback = Callable[[], Awaitable[None]]


class SarvamSTT:
    """Streaming STT via Sarvam WebSocket with native Indian-acoustic VAD barge-in."""

    def __init__(
        self,
        on_transcript: TranscriptCallback,
        on_speech_started: Optional[SpeechStartedCallback] = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._ws = None
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._utterance_parts: list[str] = []

    async def connect(self) -> None:
        """Open WebSocket connection and send barge-in config."""
        api_key = (os.getenv("SARVAM_API_KEY") or SARVAM_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured")

        self._ws = await websockets.connect(
            SARVAM_STT_URL,
            additional_headers={"api-subscription-key": api_key},
        )

        config_msg = {
            "type": "config",
            "interrupt_min_speech_frames": SARVAM_STT_INTERRUPT_MIN_FRAMES,
            "min_speech_frames": SARVAM_STT_MIN_SPEECH_FRAMES,
            "start_speech_volume_threshold": SARVAM_STT_VOLUME_THRESHOLD,
            "high_vad_sensitivity": SARVAM_STT_HIGH_VAD,
            "negative_frames_count": SARVAM_STT_NEGATIVE_FRAMES_COUNT,
            "negative_frames_window": SARVAM_STT_NEGATIVE_FRAMES_WINDOW,
        }
        await self._ws.send(json.dumps(config_msg))

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info(
            "Sarvam STT connected (interrupt_min_frames=%d, vad_high=%s)",
            SARVAM_STT_INTERRUPT_MIN_FRAMES,
            SARVAM_STT_HIGH_VAD,
        )

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Forward a raw μ-law 8kHz audio chunk to Sarvam."""
        if self._ws:
            try:
                await self._ws.send(audio_chunk)
            except Exception as exc:
                logger.debug("Sarvam STT send_audio error: %s", exc)

    async def close(self) -> None:
        """Gracefully shut down the WebSocket and background tasks."""
        if self._receive_task:
            self._receive_task.cancel()
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Sarvam STT connection closed")

    async def _receive_loop(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue  # Sarvam sends JSON text frames only
                try:
                    data = json.loads(message)
                    await self._handle_event(data)
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Sarvam STT parse error: %s", exc)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Sarvam STT receive_loop ended: %s", exc)

    async def _handle_event(self, data: dict) -> None:
        event_type = data.get("type", "")

        if event_type == "speech_started":
            logger.debug("Sarvam STT: speech_started — barge-in trigger")
            self._utterance_parts = []
            if self._on_speech_started:
                await self._on_speech_started()

        elif event_type == "transcript":
            transcript = (data.get("transcript") or "").strip()
            is_final = bool(data.get("is_final", False))

            if not transcript:
                return

            if is_final:
                if self._utterance_parts:
                    full = " ".join(self._utterance_parts) + " " + transcript
                else:
                    full = transcript
                self._utterance_parts = []

                if len(full.split()) < 2:
                    logger.debug("Sarvam STT: discarding short utterance: %r", full)
                    return

                logger.info("Sarvam STT utterance complete: %s", full)
                await self._on_transcript(full, True)
            else:
                self._utterance_parts.append(transcript)
                logger.debug("Sarvam STT interim: %s", transcript)

    async def _keepalive_loop(self) -> None:
        try:
            while self._ws:
                await asyncio.sleep(8)
                if self._ws:
                    await self._ws.ping()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Sarvam STT keepalive stopped: %s", exc)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_stt_sarvam.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/stt_sarvam.py tests/test_stt_sarvam.py
git commit -m "feat: add Sarvam STT WebSocket client with Indian-acoustic barge-in params"
```

---

## Task 4: Sarvam TTS — HTTP Blocking Fix & Prosody Params

**Files:**
- Modify: `services/tts_sarvam.py`
- Create: `tests/test_tts_sarvam_barge_in.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tts_sarvam_barge_in.py`:

```python
"""Tests for Sarvam TTS HTTP-blocking barge-in fix and prosody params."""
import asyncio
import base64
import io
import wave
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_wav_b64(n_frames: int = 2000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * n_frames)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(autouse=True)
def clear_cache():
    from services import tts_sarvam
    tts_sarvam._audio_cache.clear()
    tts_sarvam._http_client = None
    yield
    tts_sarvam._audio_cache.clear()
    tts_sarvam._http_client = None


@pytest.mark.asyncio
async def test_pre_cancelled_skips_http_request():
    """If cancelled_flag is True before the call, no HTTP request is made."""
    mock_client = AsyncMock()

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [True]  # already cancelled
            chunks = []
            async for chunk in sarvam_synthesize("test", cancelled):
                chunks.append(chunk)

    assert chunks == []
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_during_http_fetch_stops_generation():
    """Cancel flag set while HTTP request is in flight stops synthesis."""
    wav_b64 = _make_wav_b64(n_frames=44100)

    async def slow_post(*args, **kwargs):
        await asyncio.sleep(0.2)  # simulate network latency
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"audios": [wav_b64]}
        return resp

    mock_client = AsyncMock()
    mock_client.post = slow_post
    cancelled = [False]

    async def cancel_after_50ms():
        await asyncio.sleep(0.05)
        cancelled[0] = True

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            chunks = []
            await asyncio.gather(
                cancel_after_50ms(),
                _collect(sarvam_synthesize("long text", cancelled), chunks),
            )

    assert chunks == [], "Should yield no chunks when cancelled mid-fetch"


async def _collect(gen, out: list):
    async for item in gen:
        out.append(item)


@pytest.mark.asyncio
async def test_prosody_params_sent_in_payload():
    """pace, pitch, loudness appear in the Sarvam API request payload."""
    wav_b64 = _make_wav_b64()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [wav_b64]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            import services.tts_sarvam as mod
            with patch.object(mod, "SARVAM_TTS_PACE", 0.9), \
                 patch.object(mod, "SARVAM_TTS_PITCH", 0.05), \
                 patch.object(mod, "SARVAM_TTS_LOUDNESS", 1.5):
                from services.tts_sarvam import sarvam_synthesize
                cancelled = [False]
                async for _ in sarvam_synthesize("test", cancelled, pitch_override=0.05):
                    pass

    payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args.args[1]
    assert payload["pace"] == 0.9
    assert payload["pitch"] == 0.05
    assert payload["loudness"] == 1.5
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_tts_sarvam_barge_in.py -v
```

Expected: FAIL — `pre_cancelled` passes (current code checks cache first), others fail.

- [ ] **Step 3: Modify `services/tts_sarvam.py`**

Add imports at the top:
```python
import asyncio
```

Update the function signature to accept `pitch_override`:
```python
async def sarvam_synthesize(
    text: str,
    cancelled_flag: list,
    language_code: str = "hi-IN",
    pitch_override: Optional[float] = None,
) -> AsyncIterator[bytes]:
```

Add these imports from base_config (add to existing import):
```python
from config.base_config import (
    SARVAM_API_KEY,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
    SARVAM_TTS_PACE,
    SARVAM_TTS_PITCH,
    SARVAM_TTS_LOUDNESS,
)
```

Replace the entire block from `if audio_bytes is None:` through the `client.post(...)` call with:

```python
    if audio_bytes is None:
        # Check cancel flag before starting the network request — handles the
        # common case where barge-in fires in the gap between two sentences.
        if cancelled_flag[0]:
            return

        _pitch = pitch_override if pitch_override is not None else SARVAM_TTS_PITCH
        payload = {
            "text": text,
            "target_language_code": language_code,
            "model": SARVAM_TTS_MODEL,
            "speaker": SARVAM_TTS_SPEAKER,
            "speech_sample_rate": 8000,
            "pace": SARVAM_TTS_PACE,
            "pitch": _pitch,
            "loudness": SARVAM_TTS_LOUDNESS,
        }
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }

        client = await _get_http_client()

        # Wrap the HTTP request in a task so we can cancel it if barge-in fires
        # during the network round-trip (Sarvam returns the full WAV at once —
        # there is no streaming, so this is the only cancellation window).
        fetch_task = asyncio.create_task(
            client.post(_SARVAM_TTS_URL, headers=headers, json=payload)
        )
        while not fetch_task.done():
            if cancelled_flag[0]:
                fetch_task.cancel()
                return
            await asyncio.sleep(0.05)
        response = await fetch_task

        # Some accounts use Bearer auth — retry once if subscription key rejected
        if response.status_code == 403:
            retry_headers = {**headers, "Authorization": f"Bearer {api_key}"}
            response = await client.post(_SARVAM_TTS_URL, headers=retry_headers, json=payload)
```

- [ ] **Step 4: Run all TTS tests**

```bash
pytest tests/test_tts_sarvam.py tests/test_tts_sarvam_barge_in.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/tts_sarvam.py tests/test_tts_sarvam_barge_in.py
git commit -m "fix: make Sarvam TTS HTTP fetch cancellable for barge-in + add pace/pitch/loudness"
```

---

## Task 5: Gemini 2.5 Pro LLM Client

**Files:**
- Create: `services/llm_gemini.py`
- Create: `tests/test_llm_gemini.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_gemini.py`:

```python
"""Tests for Gemini 2.5 Pro LLM client — message conversion and streaming."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Message conversion tests (no network needed) ──────────────────────────────

def test_system_message_extracted_as_instruction():
    from services.llm_gemini import _convert_messages
    msgs = [{"role": "system", "content": "You are Priya."}]
    system, contents = _convert_messages(msgs)
    assert system == "You are Priya."
    assert contents == []


def test_user_message_becomes_user_content():
    from services.llm_gemini import _convert_messages
    msgs = [{"role": "user", "content": "Hello"}]
    _, contents = _convert_messages(msgs)
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello"


def test_assistant_message_becomes_model_content():
    from services.llm_gemini import _convert_messages
    msgs = [{"role": "assistant", "content": "Hi there!"}]
    _, contents = _convert_messages(msgs)
    assert contents[0].role == "model"
    assert contents[0].parts[0].text == "Hi there!"


def test_tool_result_becomes_function_response():
    from services.llm_gemini import _convert_messages
    msgs = [{
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "check_doctor_slots",
        "content": json.dumps({"slots": ["9am", "11am"]}),
    }]
    _, contents = _convert_messages(msgs)
    assert contents[0].role == "user"
    part = contents[0].parts[0]
    assert part.function_response.name == "check_doctor_slots"
    assert part.function_response.response == {"slots": ["9am", "11am"]}


def test_assistant_with_tool_calls_becomes_function_call():
    from services.llm_gemini import _convert_messages
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "list_doctors", "arguments": '{"specialty": "cardiology"}'},
        }],
    }]
    _, contents = _convert_messages(msgs)
    assert contents[0].role == "model"
    assert contents[0].parts[0].function_call.name == "list_doctors"
    assert contents[0].parts[0].function_call.args == {"specialty": "cardiology"}


def test_tool_schema_conversion():
    from services.llm_gemini import _convert_tools
    openai_tools = [{
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment",
            "parameters": {
                "type": "object",
                "properties": {"doctor_name": {"type": "string"}},
                "required": ["doctor_name"],
            },
        }
    }]
    gemini_tools = _convert_tools(openai_tools)
    assert len(gemini_tools) == 1
    decl = gemini_tools[0].function_declarations[0]
    assert decl.name == "book_appointment"
    assert decl.description == "Book an appointment"


def test_empty_tool_list_returns_empty():
    from services.llm_gemini import _convert_tools
    assert _convert_tools([]) == []
    assert _convert_tools(None) == []
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_llm_gemini.py -v
```

Expected: `ModuleNotFoundError: No module named 'services.llm_gemini'`

- [ ] **Step 3: Create `services/llm_gemini.py`**

```python
"""
Gemini 2.5 Pro LLM streaming client.

Same stream_response() contract as services/llm.py:
  - yields str (sentences) as the LLM streams text
  - yields dict {"type": "tool_call", ...} for function calls

thinking_budget=0 disables Gemini's chain-of-thought mode — required
for real-time voice (prevents 3-5s silent pauses before each response).
"""
import json
import logging
import re
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types

from config.base_config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_THINKING_BUDGET,
)

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None

_SENTENCE_RE = re.compile(
    r"(?:(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Sr\.)(?<!Jr\.)(?<!Mrs\.)(?<!Prof\.)(?<!डॉ\.)(?<=[.!?])\s+|(?<=[।॥])\s*)"
)

# Duplicated from llm.py — importing from there would create a circular import
# (llm.py imports stream_response from llm_gemini.py via the routing dispatcher).
_CLAUSE_RE = re.compile(r',\s+(?=\S)| — |:\s+(?=\S)')


def _clause_split(sentence: str) -> list[str]:
    """Split a sentence at comma/em-dash/colon if the leading clause is ≥5 words."""
    raw_parts = _CLAUSE_RE.split(sentence)
    if len(raw_parts) == 1:
        return [sentence]
    result = []
    current = raw_parts[0]
    for part in raw_parts[1:]:
        if len(current.split()) >= 5:
            result.append(current.strip())
            current = part
        else:
            current = current + ", " + part
    if current.strip():
        result.append(current.strip())
    return [p for p in result if p]


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured. Add it to .env.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _convert_tools(openai_tools) -> list:
    """Convert OpenAI tool schemas → Gemini FunctionDeclaration list."""
    if not openai_tools:
        return []
    declarations = []
    for tool in openai_tools:
        fn = tool.get("function", {})
        declarations.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _convert_messages(messages: list[dict]) -> tuple[str, list]:
    """
    Split OpenAI messages into (system_instruction, gemini_contents).

    Role mapping:
      system    → system_instruction string (not a Content object)
      user      → user Content with text Part
      assistant → model Content (text Part or function_call Parts)
      tool      → user Content with function_response Part
    """
    system_instruction = ""
    contents = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "system":
            system_instruction = content
            continue

        if role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=content)])
            )

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts = [
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"].get("arguments", "{}")),
                        )
                    )
                    for tc in tool_calls
                ]
                contents.append(types.Content(role="model", parts=parts))
            elif content:
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=content)])
                )

        elif role == "tool":
            name = msg.get("name", "")
            try:
                result = json.loads(content) if content else {}
            except json.JSONDecodeError:
                result = {"content": content}
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=name,
                                response=result,
                            )
                        )
                    ],
                )
            )

    return system_instruction, contents


async def stream_response(
    messages: list[dict],
    tools=None,
    model: str = GEMINI_MODEL,
) -> AsyncIterator[str | dict]:
    """
    Yield sentences (str) and tool calls (dict) from Gemini 2.5 Pro.
    Identical contract to services/llm.py stream_response.
    """
    client = _get_client()
    system_instruction, contents = _convert_messages(messages)
    gemini_tools = _convert_tools(tools or [])

    config = types.GenerateContentConfig(
        system_instruction=system_instruction or None,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=LLM_THINKING_BUDGET),
        tools=gemini_tools or None,
    )

    buffer = ""
    function_calls: list[dict] = []

    try:
        async for chunk in client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    function_calls.append({
                        "type": "tool_call",
                        "id": f"call_{fc.name}",
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })
                elif part.text:
                    buffer += part.text
                    parts = _SENTENCE_RE.split(buffer)
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        if len(sentence.split()) < 3 and not sentence.endswith(("।", "॥", "!", "?")):
                            parts[-1] = sentence + " " + parts[-1]
                            continue
                        logger.debug("Gemini sentence: %s", sentence)
                        for sub in _clause_split(sentence):
                            yield sub
                    buffer = parts[-1]

        if buffer.strip():
            for sub in _clause_split(buffer.strip()):
                yield sub

        for fc in function_calls:
            logger.info("Gemini tool_call: %s(%s)", fc["name"], fc["arguments"])
            yield fc

    except Exception as exc:
        logger.error("Gemini LLM stream error: %s", exc)
        yield "I'm having a little trouble right now. Could you repeat that?"
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_llm_gemini.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/llm_gemini.py tests/test_llm_gemini.py
git commit -m "feat: add Gemini 2.5 Pro LLM client with thinking_budget=0 for voice latency"
```

---

## Task 6: LLM Routing & Sentence Splitter Update

**Files:**
- Modify: `services/llm.py`

- [ ] **Step 1: Add Gemini routing to `services/llm.py`**

Replace the module-level `stream_response` function definition with a dispatcher. Find the existing function header:

```python
async def stream_response(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = LLM_MODEL,
) -> AsyncIterator[str | dict]:
```

Add this block immediately BEFORE the existing function:

```python
if LLM_PROVIDER == "gemini":
    from services.llm_gemini import stream_response  # noqa: F401 — re-export
else:
    async def stream_response(  # type: ignore[misc]  # defined below
```

Actually that pattern is messy. Instead, rename the existing function to `_openai_stream_response` and add a top-level dispatcher:

At the TOP of `services/llm.py` (after imports), add:

```python
async def stream_response(
    messages: list[dict],
    tools=None,
    model: str = LLM_MODEL,
):
    """Route to Gemini or OpenAI-compatible provider based on LLM_PROVIDER."""
    if LLM_PROVIDER == "gemini":
        from services.llm_gemini import stream_response as _gemini_sr
        async for item in _gemini_sr(messages, tools=tools, model=model):
            yield item
    else:
        async for item in _openai_stream_response(messages, tools=tools, model=model):
            yield item
```

Rename the existing `stream_response` function to `_openai_stream_response` (just the `def` line — all body unchanged):

```python
async def _openai_stream_response(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = LLM_MODEL,
) -> AsyncIterator[str | dict]:
```

- [ ] **Step 2: Update the sentence splitter to add comma/clause splits**

In `services/llm.py`, replace:

```python
_SENTENCE_RE = re.compile(
    r"(?:(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Sr\.)(?<!Jr\.)(?<!Mrs\.)(?<!Prof\.)(?<!डॉ\.)(?<=[.!?])\s+|(?<=[।॥])\s*)"
)
```

with:

```python
_SENTENCE_RE = re.compile(
    r"(?:(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Sr\.)(?<!Jr\.)(?<!Mrs\.)(?<!Prof\.)(?<!डॉ\.)(?<=[.!?])\s+|(?<=[।॥])\s*)"
)

# Secondary split: break long comma-clauses and em-dashes into shorter TTS chunks.
# Only splits at a comma when the preceding segment is ≥5 words, to avoid splitting
# short phrases like "हाँ, sure" or "Hello, Priya" into separate TTS calls.
_CLAUSE_RE = re.compile(r',\s+(?=\S)| — |:\s+(?=\S)')


def _clause_split(sentence: str) -> list[str]:
    """Further split a sentence at comma/em-dash/colon if the leading clause is ≥5 words."""
    raw_parts = _CLAUSE_RE.split(sentence)
    if len(raw_parts) == 1:
        return [sentence]
    result = []
    current = raw_parts[0]
    for part in raw_parts[1:]:
        if len(current.split()) >= 5:
            result.append(current.strip())
            current = part
        else:
            current = current + ", " + part
    if current.strip():
        result.append(current.strip())
    return [p for p in result if p]
```

Then in `_openai_stream_response`, inside the `for sentence in parts[:-1]:` loop, after `yield sentence`, replace:

```python
                yield sentence
```

with:

```python
                for sub in _clause_split(sentence):
                    yield sub
```

Also add the same `_clause_split` call in the flush-remaining-buffer section:

```python
        # Flush remaining buffer
        if buffer.strip():
            for sub in _clause_split(buffer.strip()):
                yield sub
```

- [ ] **Step 3: Verify Gemini routing works end-to-end**

```bash
python -c "
import asyncio, os
os.environ['LLM_PROVIDER'] = 'groq'  # use existing provider to test routing
from services.llm import stream_response
print('routing import ok')
"
```

Expected: `routing import ok`

- [ ] **Step 4: Commit**

```bash
git add services/llm.py
git commit -m "feat: add Gemini LLM routing and comma-clause sentence splitter for lower TTS latency"
```

---

## Task 7: Voice Pipeline — Prosody, Anti-Markdown, Silence Timeout

**Files:**
- Modify: `pipeline/voice_pipeline.py`
- Create: `tests/test_prosody.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prosody.py`:

```python
"""Tests for prosody preprocessor and anti-markdown stripping."""
import pytest


def test_em_dash_replaced_with_comma_space():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("Doctor Bhaskar — he is available tomorrow")
    assert "—" not in result
    assert "," in result


def test_discourse_marker_haan_gets_comma():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("हाँ आपकी appointment book हो गई है")
    assert "हाँ," in result


def test_discourse_marker_actually_gets_comma():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("actually let me check that")
    assert "actually," in result


def test_discourse_marker_already_has_comma_unchanged():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("हाँ, sure बताइए")
    # Should not double-add a comma
    assert result.count("हाँ,") == 1


def test_ellipsis_replaced_with_comma():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("One moment... let me check")
    assert "..." not in result
    assert "," in result


def test_strip_markdown_bold():
    from pipeline.voice_pipeline import _strip_markdown
    assert _strip_markdown("**Doctor** Bhaskar") == "Doctor Bhaskar"


def test_strip_markdown_italic():
    from pipeline.voice_pipeline import _strip_markdown
    assert _strip_markdown("*please hold*") == "please hold"


def test_strip_markdown_bullet():
    from pipeline.voice_pipeline import _strip_markdown
    result = _strip_markdown("- Option one")
    assert result == "Option one"


def test_strip_markdown_trailing_colon():
    from pipeline.voice_pipeline import _strip_markdown
    result = _strip_markdown("Available slots :")
    assert result == "Available slots"


def test_strip_markdown_preserves_hindi():
    from pipeline.voice_pipeline import _strip_markdown
    result = _strip_markdown("**नमस्ते** Priya")
    assert result == "नमस्ते Priya"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_prosody.py -v
```

Expected: `ImportError` — `_add_prosody_markers` not defined yet.

- [ ] **Step 3: Add module-level functions to `pipeline/voice_pipeline.py`**

Add these two functions BEFORE the `VoicePipeline` class definition (after the module-level constants like `_EMERGENCY_KEYWORDS`):

```python
import re as _re

_DISCOURSE_RE = _re.compile(
    r'\b(हाँ|ठीक है|actually|so|well|okay|ok|sure|हाँ जी|अच्छा)(\s+)(?=[^\s,।])',
    _re.IGNORECASE,
)


def _add_prosody_markers(text: str) -> str:
    """Add natural pause cues before sending text to TTS."""
    text = text.replace("—", ", ")
    text = text.replace("...", ",")
    text = _DISCOURSE_RE.sub(lambda m: m.group(1) + "," + m.group(2), text)
    return text


def _strip_markdown(text: str) -> str:
    """Remove Gemini markdown artefacts that TTS reads literally."""
    text = _re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = _re.sub(r'\*(.+?)\*', r'\1', text)
    text = _re.sub(r'^\s*[-•]\s+', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'\s+:$', '', text)
    return text.strip()
```

- [ ] **Step 4: Wire `_strip_markdown` and `_add_prosody_markers` into `_normalize_for_tts`**

In `VoicePipeline._normalize_for_tts`, at the very END of the method body (before `return text`), add:

```python
        text = _strip_markdown(text)
        text = _add_prosody_markers(text)
        return text
```

Remove the existing bare `return text` at the end.

- [ ] **Step 5: Pass `pitch_override` to `sarvam_synthesize` for questions**

In `VoicePipeline._speak()`, replace:

```python
            async for chunk in self._tts.synthesize(text, language_code=lang_code):
```

with:

```python
            _pitch_bump = 0.05 if text.rstrip().endswith("?") else 0.0
            async for chunk in self._tts.synthesize(text, language_code=lang_code, pitch_override=_pitch_bump if _pitch_bump else None):
```

Also update `TTSService._sarvam()` in `services/tts.py` to accept and forward `pitch_override`:

```python
    async def synthesize(self, text: str, language_code: str = "hi-IN", pitch_override=None) -> AsyncIterator[bytes]:
```

And in `_sarvam`:
```python
    async def _sarvam(self, text: str, language_code: str = "hi-IN", pitch_override=None) -> AsyncIterator[bytes]:
        from services.tts_sarvam import sarvam_synthesize
        async for chunk in sarvam_synthesize(text, self._cancel_flag, language_code=language_code, pitch_override=pitch_override):
            yield chunk
```

- [ ] **Step 6: Add silence timeout to `VoicePipeline`**

Add these imports at the top of `pipeline/voice_pipeline.py` (to the existing imports block):

```python
from config.base_config import (
    SILENCE_TIMEOUT_SECS,
    SILENCE_HANGUP_SECS,
    ADAPTIVE_HOLD_SHORT_MS,
    ADAPTIVE_HOLD_NORMAL_MS,
)
```

In `VoicePipeline.__init__`, add:

```python
        self._last_activity_at: float = 0.0
```

In `VoicePipeline.start()`, after `asyncio.create_task(self._llm_loop())`, add:

```python
        self._last_activity_at = time.monotonic()
        asyncio.create_task(self._silence_watch_loop())
```

In `VoicePipeline._on_transcript()`, at the start of the method, add:

```python
        self._last_activity_at = time.monotonic()
```

In `VoicePipeline._speak()`, in the `finally` block (after `self._tts_playing = False`), add:

```python
            self._last_activity_at = time.monotonic()
```

Add this new method to `VoicePipeline` (after `_soft_close_after_booking`):

```python
    async def _silence_watch_loop(self) -> None:
        """Prompt 'are you still there?' after inactivity, then hang up."""
        while self._running:
            await asyncio.sleep(1.0)
            if self._tts_playing or self._agent_in_turn or not self._running:
                continue
            idle = time.monotonic() - self._last_activity_at
            if idle < SILENCE_TIMEOUT_SECS:
                continue

            logger.info("Silence timeout after %.0fs — prompting caller", idle)
            self._last_activity_at = time.monotonic()
            lang = self._caller_language
            if lang == "hindi":
                prompt = "क्या आप वहाँ हैं?"
            elif lang == "english":
                prompt = "Are you still there?"
            else:
                prompt = "क्या आप वहाँ हैं?"
            self._interruption.reset()
            await self._speak(prompt)

            # Wait for hangup window
            await asyncio.sleep(SILENCE_HANGUP_SECS)
            if not self._running:
                return
            idle2 = time.monotonic() - self._last_activity_at
            if idle2 < SILENCE_HANGUP_SECS:
                continue  # caller responded — back to normal watch

            logger.info("Caller still silent — hanging up")
            if lang == "hindi":
                bye = "ठीक है, आपसे बात करके अच्छा लगा। अलविदा।"
            elif lang == "english":
                bye = "Alright, have a good day. Goodbye."
            else:
                bye = "ठीक है, have a good day. Goodbye."
            self._interruption.reset()
            duration = await self._speak(bye)
            await asyncio.sleep(duration + 0.3)
            await self._telephony.hangup()
            self._running = False
```

Replace the hardcoded hold values in `_llm_loop`:

```python
            _hold_ms = 400 if len(user_text.split()) < 3 else 150
```

with:

```python
            _hold_ms = ADAPTIVE_HOLD_SHORT_MS if len(user_text.split()) < 3 else ADAPTIVE_HOLD_NORMAL_MS
```

- [ ] **Step 7: Run prosody tests**

```bash
pytest tests/test_prosody.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pipeline/voice_pipeline.py services/tts.py tests/test_prosody.py
git commit -m "feat: add prosody preprocessing, anti-markdown strip, silence timeout for natural speech"
```

---

## Task 8: System Prompt Update

**Files:**
- Modify: `prompts/system_prompt.py`

- [ ] **Step 1: Add Gemini prose-rhythm instruction**

In `prompts/system_prompt.py`, find the `## Critical voice rules (ALWAYS follow)` section inside the prompt string. After the last bullet in that section (the line ending `Go longer only if the caller asked for detail.`), add:

```python
- Write with natural spoken rhythm. Use commas generously at clause boundaries — they become spoken pauses. End questions with `?`. Never use colons, asterisks, or bullet points — these are read literally by the voice engine. Vary sentence length: short confirmations ("हाँ, sure."), medium explanations; never pack more than one idea into a single breath.
```

- [ ] **Step 2: Verify prompt builds without error**

```bash
python -c "
from prompts.system_prompt import build_system_prompt
p = build_system_prompt(caller_phone='+91999', caller_language='hinglish')
print(f'Prompt length: {len(p)} chars')
assert 'spoken rhythm' in p
print('ok')
"
```

Expected: prints prompt length and `ok`.

- [ ] **Step 3: Commit**

```bash
git add prompts/system_prompt.py
git commit -m "feat: add Gemini prose-rhythm instruction to system prompt for natural TTS output"
```

---

## Task 9: Config API Backend

**Files:**
- Modify: `main.py`
- Create: `tests/test_config_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_api.py`:

```python
"""Tests for config GET/POST endpoints."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    """TestClient with user_settings.json pointing to a temp dir."""
    import config.user_settings as us_mod
    tmp_settings = tmp_path / "user_settings.json"

    with patch.object(us_mod, "_PATH", tmp_settings):
        from main import app
        return TestClient(app)


def test_get_config_returns_empty_dict_when_no_file(client):
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json() == {}


def test_post_config_saves_settings(client, tmp_path):
    import config.user_settings as us_mod
    tmp_settings = tmp_path / "user_settings.json"

    with patch.object(us_mod, "_PATH", tmp_settings):
        payload = {"stt_provider": "sarvam", "sarvam_tts_pace": 0.9}
        response = client.post("/config", json=payload)
        assert response.status_code == 200
        assert response.json() == {"status": "saved"}
        saved = json.loads(tmp_settings.read_text())
        assert saved["stt_provider"] == "sarvam"
        assert saved["sarvam_tts_pace"] == 0.9


def test_get_config_returns_saved_settings(client, tmp_path):
    import config.user_settings as us_mod
    tmp_settings = tmp_path / "user_settings.json"
    tmp_settings.write_text(json.dumps({"llm_provider": "gemini"}))

    with patch.object(us_mod, "_PATH", tmp_settings):
        response = client.get("/config")
        assert response.json()["llm_provider"] == "gemini"


def test_config_ui_returns_html(client):
    response = client.get("/config/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_config_api.py -v
```

Expected: 404 errors — endpoints don't exist yet.

- [ ] **Step 3: Add config endpoints to `main.py`**

Add these imports at the top of `main.py` (after existing imports):

```python
from pathlib import Path
```

Add these three endpoints after the `/health` endpoint:

```python
# ── Config API ────────────────────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    """Return current user settings as JSON."""
    from config.user_settings import load_user_settings
    return JSONResponse(load_user_settings())


@app.post("/config")
async def post_config(request: Request):
    """Save user settings to config/user_settings.json."""
    from config.user_settings import save_user_settings
    data = await request.json()
    save_user_settings(data)
    return JSONResponse({"status": "saved"})


@app.get("/config/ui", response_class=HTMLResponse)
async def config_ui():
    """Serve the admin config page."""
    html_path = Path("static/config.html")
    if not html_path.exists():
        return HTMLResponse("<h1>Config UI not found</h1><p>static/config.html is missing.</p>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_config_api.py -v
```

Expected: all 4 tests PASS (the HTML test will pass after Task 10 creates the file — skip it for now with `-k "not html"`).

```bash
pytest tests/test_config_api.py -v -k "not html"
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_config_api.py
git commit -m "feat: add GET/POST /config and /config/ui endpoints for runtime settings"
```

---

## Task 10: Config UI — `static/config.html`

**Files:**
- Create: `static/config.html`

- [ ] **Step 1: Create `static/config.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Voice Agent Config</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f0f2f5; color: #1a1a1a; padding: 24px; }
    h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 20px; color: #111; }
    .card { background: #fff; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .card-header {
      padding: 14px 18px; cursor: pointer; display: flex; justify-content: space-between;
      align-items: center; font-weight: 600; font-size: 0.95rem; user-select: none;
    }
    .card-header:hover { background: #fafafa; border-radius: 10px; }
    .card-body { padding: 18px 18px 8px; display: none; border-top: 1px solid #f0f0f0; }
    .card-body.open { display: block; }
    .field { margin-bottom: 14px; }
    label { display: block; font-size: 0.82rem; font-weight: 500; margin-bottom: 5px; color: #555; }
    label small { font-weight: 400; color: #888; margin-left: 4px; }
    input[type=text], input[type=number], select, textarea {
      width: 100%; padding: 7px 10px; border: 1px solid #ddd; border-radius: 6px;
      font-size: 0.88rem; background: #fafafa;
    }
    input[type=text]:focus, input[type=number]:focus, select:focus, textarea:focus {
      outline: none; border-color: #2563eb; background: #fff;
    }
    textarea { height: 72px; resize: vertical; }
    .range-row { display: flex; align-items: center; gap: 10px; }
    input[type=range] { flex: 1; accent-color: #2563eb; }
    .range-val { font-size: 0.82rem; width: 44px; text-align: right; color: #444; font-family: monospace; }
    .toggle-row { display: flex; align-items: center; gap: 8px; }
    .toggle-row label { margin: 0; font-size: 0.88rem; }
    input[type=checkbox] { width: 16px; height: 16px; accent-color: #2563eb; }
    .save-area { text-align: center; margin-top: 20px; padding-bottom: 32px; }
    button {
      background: #2563eb; color: #fff; border: none; padding: 11px 36px;
      border-radius: 8px; font-size: 0.95rem; cursor: pointer; font-weight: 600;
    }
    button:hover { background: #1d4ed8; }
    button:active { background: #1e40af; }
    #msg { margin-top: 10px; font-size: 0.88rem; min-height: 1.3em; }
    #msg.ok { color: #16a34a; }
    #msg.err { color: #dc2626; }
    .chev { transition: transform 0.2s; display: inline-block; margin-left: 6px; }
    .chev.open { transform: rotate(180deg); }
  </style>
</head>
<body>
<h1>Voice Agent Settings</h1>

<!-- STT -->
<div class="card">
  <div class="card-header" onclick="toggle('stt')">
    STT — Speech Recognition <span class="chev" id="stt-chev">▼</span>
  </div>
  <div class="card-body" id="stt-body">
    <div class="field">
      <label>Provider</label>
      <select id="stt_provider">
        <option value="sarvam">Sarvam (recommended — Indian acoustic VAD)</option>
        <option value="deepgram">Deepgram (Nova-3)</option>
      </select>
    </div>
    <div class="field">
      <label>Interrupt min speech frames <small>(lower = faster barge-in; higher = fewer false triggers)</small></label>
      <input type="number" id="stt_interrupt_min_frames" min="1" max="20">
    </div>
    <div class="field">
      <label>Min speech frames <small>(frames of voice needed to register a valid turn)</small></label>
      <input type="number" id="stt_min_speech_frames" min="1" max="30">
    </div>
    <div class="field">
      <label>Volume threshold dB <small>(−40 = ignore background hum; 0 = hear everything)</small></label>
      <div class="range-row">
        <input type="range" id="stt_volume_threshold" min="-60" max="0" step="1"
               oninput="rv('stt_volume_threshold_v', this.value)">
        <span class="range-val" id="stt_volume_threshold_v">-40</span>
      </div>
    </div>
    <div class="field toggle-row">
      <input type="checkbox" id="stt_high_vad">
      <label for="stt_high_vad">High VAD sensitivity (stricter — fewer noise false-positives)</label>
    </div>
    <div class="field">
      <label>Negative frames count <small>(silence frames before turn ends)</small></label>
      <input type="number" id="stt_negative_frames_count" min="1" max="30">
    </div>
    <div class="field">
      <label>Negative frames window</label>
      <input type="number" id="stt_negative_frames_window" min="5" max="50">
    </div>
  </div>
</div>

<!-- TTS -->
<div class="card">
  <div class="card-header" onclick="toggle('tts')">
    TTS — Text to Speech <span class="chev" id="tts-chev">▼</span>
  </div>
  <div class="card-body" id="tts-body">
    <div class="field">
      <label>Provider</label>
      <select id="tts_provider">
        <option value="sarvam">Sarvam Bulbul</option>
        <option value="elevenlabs">ElevenLabs</option>
        <option value="azure">Azure</option>
        <option value="deepgram">Deepgram Aura</option>
      </select>
    </div>
    <div class="field">
      <label>Speaker</label>
      <select id="sarvam_tts_speaker">
        <option value="pavithra">pavithra (warm, conversational)</option>
        <option value="meera">meera</option>
        <option value="kalpana">kalpana</option>
        <option value="arvind">arvind (male)</option>
        <option value="amol">amol (male)</option>
        <option value="amartya">amartya (male)</option>
      </select>
    </div>
    <div class="field">
      <label>Model</label>
      <input type="text" id="sarvam_tts_model" placeholder="bulbul:v2">
    </div>
    <div class="field">
      <label>Pace <small>(0.5 = slow · 1.0 = normal · 2.0 = fast)</small></label>
      <div class="range-row">
        <input type="range" id="sarvam_tts_pace" min="0.5" max="2.0" step="0.05"
               oninput="rv('sarvam_tts_pace_v', parseFloat(this.value).toFixed(2))">
        <span class="range-val" id="sarvam_tts_pace_v">0.90</span>
      </div>
    </div>
    <div class="field">
      <label>Pitch <small>(−1.0 = lower · 0.0 = normal · 1.0 = higher)</small></label>
      <div class="range-row">
        <input type="range" id="sarvam_tts_pitch" min="-1.0" max="1.0" step="0.05"
               oninput="rv('sarvam_tts_pitch_v', parseFloat(this.value).toFixed(2))">
        <span class="range-val" id="sarvam_tts_pitch_v">0.00</span>
      </div>
    </div>
    <div class="field">
      <label>Loudness <small>(0.5 = quiet · 1.5 = normal+ · 3.0 = loud)</small></label>
      <div class="range-row">
        <input type="range" id="sarvam_tts_loudness" min="0.5" max="3.0" step="0.1"
               oninput="rv('sarvam_tts_loudness_v', parseFloat(this.value).toFixed(1))">
        <span class="range-val" id="sarvam_tts_loudness_v">1.5</span>
      </div>
    </div>
  </div>
</div>

<!-- LLM -->
<div class="card">
  <div class="card-header" onclick="toggle('llm')">
    LLM <span class="chev" id="llm-chev">▼</span>
  </div>
  <div class="card-body" id="llm-body">
    <div class="field">
      <label>Provider</label>
      <select id="llm_provider">
        <option value="gemini">Gemini (recommended)</option>
        <option value="groq">Groq</option>
        <option value="openrouter">OpenRouter</option>
        <option value="cerebras">Cerebras</option>
      </select>
    </div>
    <div class="field">
      <label>Model</label>
      <input type="text" id="llm_model" placeholder="gemini-2.5-pro-preview-06-05">
    </div>
    <div class="field">
      <label>Temperature <small>(0.0 = focused · 0.5 = balanced · 1.0 = creative)</small></label>
      <div class="range-row">
        <input type="range" id="llm_temperature" min="0.0" max="1.0" step="0.05"
               oninput="rv('llm_temperature_v', parseFloat(this.value).toFixed(2))">
        <span class="range-val" id="llm_temperature_v">0.50</span>
      </div>
    </div>
    <div class="field">
      <label>Max tokens</label>
      <input type="number" id="llm_max_tokens" min="50" max="1000" step="10">
    </div>
    <div class="field">
      <label>Thinking budget <small>(Gemini only — 0 disables chain-of-thought, required for voice speed)</small></label>
      <input type="number" id="llm_thinking_budget" min="0" max="10000" step="100">
    </div>
  </div>
</div>

<!-- Prosody -->
<div class="card">
  <div class="card-header" onclick="toggle('prosody')">
    Prosody &amp; Timing <span class="chev" id="prosody-chev">▼</span>
  </div>
  <div class="card-body" id="prosody-body">
    <div class="field">
      <label>Silence prompt timeout (seconds) <small>(how long before "are you still there?")</small></label>
      <input type="number" id="silence_timeout_secs" min="5" max="60">
    </div>
    <div class="field">
      <label>Silence hangup timeout (seconds) <small>(after prompt, how long to wait before hanging up)</small></label>
      <input type="number" id="silence_hangup_secs" min="5" max="60">
    </div>
    <div class="field">
      <label>Adaptive hold — short utterance (ms) <small>(hold for "haan", "okay" fillers)</small></label>
      <input type="number" id="adaptive_hold_short_ms" min="100" max="1000" step="50">
    </div>
    <div class="field">
      <label>Adaptive hold — normal utterance (ms)</label>
      <input type="number" id="adaptive_hold_normal_ms" min="50" max="500" step="25">
    </div>
  </div>
</div>

<!-- Agent -->
<div class="card">
  <div class="card-header" onclick="toggle('agent')">
    Agent <span class="chev" id="agent-chev">▼</span>
  </div>
  <div class="card-body" id="agent-body">
    <div class="field">
      <label>Agent name</label>
      <input type="text" id="agent_name" placeholder="Priya">
    </div>
    <div class="field">
      <label>Persona</label>
      <textarea id="agent_persona" placeholder="warm, professional, and caring"></textarea>
    </div>
    <div class="field">
      <label>Default language</label>
      <select id="agent_default_language">
        <option value="hinglish">Hinglish (Hindi-English mix)</option>
        <option value="hindi">Hindi only</option>
        <option value="english">English only</option>
      </select>
    </div>
  </div>
</div>

<div class="save-area">
  <button onclick="save()">Save Settings</button>
  <div id="msg"></div>
</div>

<script>
const FIELDS = [
  'stt_provider','stt_interrupt_min_frames','stt_min_speech_frames',
  'stt_volume_threshold','stt_high_vad','stt_negative_frames_count','stt_negative_frames_window',
  'tts_provider','sarvam_tts_speaker','sarvam_tts_model',
  'sarvam_tts_pace','sarvam_tts_pitch','sarvam_tts_loudness',
  'llm_provider','llm_model','llm_temperature','llm_max_tokens','llm_thinking_budget',
  'silence_timeout_secs','silence_hangup_secs',
  'adaptive_hold_short_ms','adaptive_hold_normal_ms',
  'agent_name','agent_persona','agent_default_language',
];
const RANGES = new Set(['stt_volume_threshold','sarvam_tts_pace','sarvam_tts_pitch','sarvam_tts_loudness','llm_temperature']);

function rv(valId, v) { document.getElementById(valId).textContent = v; }

function toggle(id) {
  const body = document.getElementById(id + '-body');
  const chev = document.getElementById(id + '-chev');
  const open = body.classList.toggle('open');
  chev.classList.toggle('open', open);
}

function getVal(id) {
  const el = document.getElementById(id);
  if (!el) return undefined;
  if (el.type === 'checkbox') return el.checked;
  if (el.type === 'number' || RANGES.has(id)) return parseFloat(el.value);
  return el.value;
}

function setVal(id, v) {
  const el = document.getElementById(id);
  if (!el || v === undefined || v === null) return;
  if (el.type === 'checkbox') { el.checked = Boolean(v); return; }
  el.value = v;
  if (RANGES.has(id)) {
    const suffix = '_v';
    const display = document.getElementById(id + suffix);
    if (display) display.textContent = parseFloat(v).toFixed(id === 'sarvam_tts_loudness' ? 1 : 2);
  }
}

async function load() {
  try {
    const res = await fetch('/config');
    const data = await res.json();
    FIELDS.forEach(id => setVal(id, data[id]));
  } catch(e) {
    showMsg('Could not load settings: ' + e.message, 'err');
  }
}

async function save() {
  const data = {};
  FIELDS.forEach(id => { data[id] = getVal(id); });
  try {
    const res = await fetch('/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (res.ok) {
      showMsg('Settings saved. Changes apply to new calls.', 'ok');
    } else {
      showMsg('Save failed: ' + await res.text(), 'err');
    }
  } catch(e) {
    showMsg('Save failed: ' + e.message, 'err');
  }
}

function showMsg(text, cls) {
  const el = document.getElementById('msg');
  el.textContent = text;
  el.className = cls;
}

load();
</script>
</body>
</html>
```

- [ ] **Step 2: Run all config tests including the HTML test**

```bash
pytest tests/test_config_api.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 3: Smoke-test the UI in browser**

Start the server:
```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/config/ui` — verify:
- All 5 sections expand/collapse on click
- Page loads current settings from `GET /config`
- Sliders show live values when dragged
- Clicking Save shows green "Settings saved" message
- `config/user_settings.json` is created/updated with saved values

- [ ] **Step 4: Commit**

```bash
git add static/config.html
git commit -m "feat: add voice agent admin config UI — STT, TTS, LLM, prosody, agent settings"
```

---

## Task 11: Full Test Suite & Final Verification

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS. Note any failures and fix before proceeding.

- [ ] **Step 2: Verify Sarvam STT URL is correct**

Check `https://docs.sarvam.ai/api-reference-docs/speech-to-text-translate/translate/ws` for the current WebSocket endpoint URL. Update `SARVAM_STT_URL` default in `config/base_config.py` if it differs from the placeholder.

- [ ] **Step 3: Verify Gemini model ID**

Check `https://aistudio.google.com` for the current `gemini-2.5-pro` preview model ID. Update `GEMINI_MODEL` default in `config/base_config.py` if needed.

- [ ] **Step 4: Verify `bulbul:v2` availability**

Make a test TTS call with `model=bulbul:v2`. If Sarvam returns a 400/422, set `SARVAM_TTS_MODEL=bulbul:v1` in `.env` or via the config UI.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final verification — all tests pass, Sarvam/Gemini endpoints confirmed"
```

---

## Environment Variables Checklist

Before deploying, ensure `.env` has:

```bash
# Required new vars
GEMINI_API_KEY=your_gemini_key
LLM_PROVIDER=gemini
STT_PROVIDER=sarvam
SARVAM_STT_URL=wss://api.sarvam.ai/speech-to-text-translate/subscribe  # verify URL

# Optional overrides (can also set via /config/ui)
SARVAM_TTS_SPEAKER=pavithra
SARVAM_TTS_MODEL=bulbul:v2
SARVAM_TTS_PACE=0.9
GEMINI_MODEL=gemini-2.5-pro-preview-06-05
```
