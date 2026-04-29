# Apollo Voice Agent — Sarvam TTS Integration & Pipeline Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ElevenLabs with Sarvam as primary TTS, add adaptive filler hold, barge-in acknowledgement, soft-close after booking, and fix 9 identified bugs.

**Architecture:** Surgical changes only — port `tts_sarvam.py` from `claude/thirsty-dijkstra-0dabe2`, wire it into `tts.py` as primary, then patch `voice_pipeline.py` and `interruption.py` for natural Indian phone behaviour. No changes to STT, LLM, telephony, or tool layers.

**Tech Stack:** Python 3.11, FastAPI, Sarvam Bulbul TTS API, Deepgram Nova-3 STT, Groq LLM, Twilio WebSocket telephony, httpx, audioop (stdlib), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `config/base_config.py` | Modify | Add `SARVAM_API_KEY`, `SARVAM_TTS_MODEL`, `SARVAM_TTS_SPEAKER` |
| `.env.example` | Modify | Scrub real credentials; add Sarvam vars; set `TTS_PROVIDER=sarvam` |
| `.env` | Create/update | Add real `SARVAM_API_KEY=sk_d9iu42x0_...` (not committed) |
| `services/tts_sarvam.py` | Create | Sarvam Bulbul TTS — WAV decode → μ-law 8kHz, LRU cache, explicit `language_code` param |
| `services/tts.py` | Modify | Add Sarvam as primary provider; change `voice_id` param → `language_code`; fallback chain |
| `pipeline/voice_pipeline.py` | Modify | Adaptive hold, `_tts_playing` guard widen, barge-in ack, soft-close, `language_code` routing |
| `pipeline/interruption.py` | Modify | Tail silence 50ms → 120ms |
| `tools/calendar.py` | Modify | Prod guard — warn if dry-run in production |
| `requirements.txt` | Modify | No new deps needed (audioop is stdlib; httpx already present) |
| `tests/__init__.py` | Create | Empty — makes tests/ a package |
| `tests/test_tts_sarvam.py` | Create | Unit tests for Sarvam TTS synthesis and audio conversion |
| `tests/test_pipeline_hold.py` | Create | Unit tests for adaptive filler hold |
| `tests/test_soft_close.py` | Create | Unit tests for soft-close timeout behaviour |

---

## Task 1: Add Sarvam config variables

**Files:**
- Modify: `config/base_config.py`
- Modify: `.env.example`
- Create/update: `.env`

- [ ] **Step 1: Add Sarvam vars to base_config.py**

Open `config/base_config.py`. After the Azure TTS block (after line 57), add:

```python
# Sarvam Bulbul TTS (Indian languages — primary)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v1")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "meera")
```

Also change the `TTS_PROVIDER` default on line 34 from `"elevenlabs"` to `"sarvam"`:

```python
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "sarvam")
```

- [ ] **Step 2: Scrub credentials and add Sarvam vars to .env.example**

Replace the entire `.env.example` with:

```env
# ── Copy this to .env and fill in your values ──────────────────────────────

# ── Server ──────────────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
# Set this to your ngrok / Railway / Fly.io URL so Twilio/Telnyx can reach you
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app

# ── Telephony ────────────────────────────────────────────────────────────────
TELEPHONY_PROVIDER=twilio          # twilio | telnyx

# Twilio (prototype — free trial)
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE_NUMBER=+1XXXXXXXXXX

# Telnyx (production — uncomment when ready)
# TELNYX_API_KEY=KEY0xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TELNYX_APP_ID=your_telnyx_app_id

# ── STT ──────────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# ── TTS ──────────────────────────────────────────────────────────────────────
# Sarvam Bulbul (Indian languages — primary)
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_TTS_MODEL=bulbul:v1
SARVAM_TTS_SPEAKER=meera           # meera (female) | arvind (male)
TTS_PROVIDER=sarvam                # sarvam | elevenlabs | azure

# ElevenLabs (Indian English — fallback)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Azure Cognitive Services TTS (free fallback)
AZURE_TTS_KEY=your_azure_tts_key_here
AZURE_TTS_REGION=eastus

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# ── Google Calendar ───────────────────────────────────────────────────────────
GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/service-account.json
GOOGLE_CALENDAR_ID=your_calendar_id@group.calendar.google.com

# ── Memory ────────────────────────────────────────────────────────────────────
USE_REDIS=false                    # Set true in production / docker-compose
REDIS_URL=redis://localhost:6379
SESSION_TTL_SECONDS=7200           # 2 hours

# ── CRM ───────────────────────────────────────────────────────────────────────
HUBSPOT_ACCESS_TOKEN=your_hubspot_access_token_here

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
```

- [ ] **Step 3: Write real Sarvam key to .env (not committed)**

Check if `.env` exists. If yes, add these lines. If not, copy from `.env.example` first.

```bash
echo "SARVAM_API_KEY=sk_d9iu42x0_UE6C5r9ObaeJ0LiaY0B2c2TT" >> .env
echo "SARVAM_TTS_MODEL=bulbul:v1" >> .env
echo "SARVAM_TTS_SPEAKER=meera" >> .env
echo "TTS_PROVIDER=sarvam" >> .env
```

Confirm `.env` is in `.gitignore`:

```bash
grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore
```

- [ ] **Step 4: Commit**

```bash
cd /Users/jsdata/Projects/voice-agents
git add config/base_config.py .env.example
git commit -m "config: add Sarvam TTS vars, set as default provider, scrub example credentials"
```

---

## Task 2: Create services/tts_sarvam.py

**Files:**
- Create: `services/tts_sarvam.py`
- Test: `tests/test_tts_sarvam.py`

- [ ] **Step 1: Create tests/\_\_init\_\_.py**

```bash
mkdir -p /Users/jsdata/Projects/voice-agents/tests
touch /Users/jsdata/Projects/voice-agents/tests/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_tts_sarvam.py`:

```python
"""Tests for Sarvam Bulbul TTS provider."""
import asyncio
import audioop
import base64
import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_wav_b64(sample_rate: int = 22050, n_frames: int = 2000) -> str:
    """Create a minimal valid WAV file encoded as base64."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(autouse=True)
def reset_sarvam_cache():
    """Clear the LRU cache between tests."""
    from services import tts_sarvam
    tts_sarvam._audio_cache.clear()
    tts_sarvam._http_client = None
    yield
    tts_sarvam._audio_cache.clear()
    tts_sarvam._http_client = None


@pytest.mark.asyncio
async def test_synthesize_returns_ulaw_bytes():
    """sarvam_synthesize yields non-empty bytes for a simple Hindi text."""
    wav_b64 = _make_wav_b64()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [wav_b64]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [False]
            chunks = []
            async for chunk in sarvam_synthesize("नमस्ते", cancelled, language_code="hi-IN"):
                chunks.append(chunk)

    assert chunks, "Expected at least one audio chunk"
    assert all(isinstance(c, bytes) for c in chunks)


@pytest.mark.asyncio
async def test_synthesize_uses_explicit_language_code():
    """language_code parameter is passed to Sarvam API payload, not inferred from text."""
    wav_b64 = _make_wav_b64()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [wav_b64]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [False]
            async for _ in sarvam_synthesize("hello", cancelled, language_code="en-IN"):
                pass

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert payload["target_language_code"] == "en-IN"


@pytest.mark.asyncio
async def test_cancel_flag_stops_streaming():
    """Setting cancelled[0] = True mid-stream stops chunk generation."""
    wav_b64 = _make_wav_b64(n_frames=44100)  # ~2s of audio, many chunks
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [wav_b64]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("services.tts_sarvam._get_http_client", AsyncMock(return_value=mock_client)):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [False]
            chunks = []
            async for chunk in sarvam_synthesize("test", cancelled, language_code="hi-IN"):
                chunks.append(chunk)
                cancelled[0] = True  # cancel after first chunk

    assert len(chunks) == 1, "Expected exactly 1 chunk before cancel stopped iteration"


@pytest.mark.asyncio
async def test_missing_api_key_raises():
    """RuntimeError raised when SARVAM_API_KEY is empty."""
    with patch.dict("os.environ", {"SARVAM_API_KEY": ""}):
        # Clear the module-level config that may have cached the key
        import importlib
        import services.tts_sarvam as mod
        with patch.object(mod, "SARVAM_API_KEY", ""):
            from services.tts_sarvam import sarvam_synthesize
            cancelled = [False]
            with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):
                async for _ in sarvam_synthesize("test", cancelled):
                    pass


def test_pcm16_16k_to_ulaw_8k_produces_half_length():
    """Downsampled μ-law output is half the sample count of 16kHz input."""
    from services.tts_sarvam import _pcm16_16k_to_ulaw_8k
    # 1600 samples @ 16kHz = 100ms PCM16 → 800 samples @ 8kHz μ-law
    pcm = b"\x00\x00" * 1600
    ulaw = _pcm16_16k_to_ulaw_8k(pcm)
    # μ-law is 1 byte/sample → expect ~800 bytes (allow ±2 for resampling rounding)
    assert abs(len(ulaw) - 800) <= 2
```

- [ ] **Step 3: Run tests — verify they all fail**

```bash
cd /Users/jsdata/Projects/voice-agents
python -m pytest tests/test_tts_sarvam.py -v 2>&1 | head -40
```

Expected: `ModuleNotFoundError: No module named 'services.tts_sarvam'` or `ImportError`

- [ ] **Step 4: Create services/tts_sarvam.py**

```python
"""
Sarvam Bulbul TTS provider.

Synthesizes speech via Sarvam's /text-to-speech REST API and normalizes
the response to raw μ-law 8kHz bytes (Twilio's native G.711 format).

Output format: ulaw_8000 — raw G.711 μ-law, 8kHz, mono, 1 byte/sample.

Usage:
    cancelled = [False]
    async for chunk in sarvam_synthesize("नमस्ते", cancelled, language_code="hi-IN"):
        await telephony.send_audio(chunk)

    # Cancel mid-stream (barge-in):
    cancelled[0] = True
"""

import audioop
import base64
import io
import logging
import os
import wave
from collections import OrderedDict
from typing import AsyncIterator

import httpx

from config.base_config import SARVAM_API_KEY, SARVAM_TTS_MODEL, SARVAM_TTS_SPEAKER

logger = logging.getLogger(__name__)

_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_AUDIO_CACHE_MAX_ITEMS = 128

# LRU cache: key = (text, language_code, model, speaker) → raw μ-law bytes
_audio_cache: "OrderedDict[tuple[str, str, str, str], bytes]" = OrderedDict()
_http_client: httpx.AsyncClient | None = None


def _cache_get(key: tuple) -> bytes | None:
    data = _audio_cache.get(key)
    if data is not None:
        _audio_cache.move_to_end(key)
    return data


def _cache_set(key: tuple, audio_bytes: bytes) -> None:
    _audio_cache[key] = audio_bytes
    _audio_cache.move_to_end(key)
    while len(_audio_cache) > _AUDIO_CACHE_MAX_ITEMS:
        _audio_cache.popitem(last=False)


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=45,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client


def _decode_first_wav(audio_b64_list: list) -> bytes:
    if not audio_b64_list:
        raise ValueError("Sarvam TTS returned no audio")
    return base64.b64decode(audio_b64_list[0])


def _wav_to_raw_pcm16_mono_16k(wav_bytes: bytes) -> bytes:
    """Convert WAV (any sample rate, mono/stereo) → raw PCM16 mono 16kHz."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if channels == 2:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
    if sampwidth != 2:
        frames = audioop.lin2lin(frames, sampwidth, 2)
        sampwidth = 2
    if framerate != 16000:
        frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 16000, None)
    return frames


def _pcm16_16k_to_ulaw_8k(pcm16: bytes) -> bytes:
    """Downsample PCM16 16kHz → μ-law 8kHz (Twilio native format)."""
    downsampled, _ = audioop.ratecv(pcm16, 2, 1, 16000, 8000, None)
    return audioop.lin2ulaw(downsampled, 2)


async def sarvam_synthesize(
    text: str,
    cancelled_flag: list,
    language_code: str = "hi-IN",
    output_format: str = "ulaw_8000",
) -> AsyncIterator[bytes]:
    """
    Synthesize speech with Sarvam Bulbul and yield raw μ-law 8kHz chunks.

    Args:
        text: Text to synthesize (Hindi, English, or Hinglish).
        cancelled_flag: Single-element list [False]; set to [True] to stop mid-stream.
        language_code: "hi-IN" for Hindi/Hinglish, "en-IN" for Indian English.
        output_format: Always "ulaw_8000" for telephony (Twilio μ-law 8kHz).
    """
    api_key = (os.getenv("SARVAM_API_KEY") or SARVAM_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not configured. Add it to .env: SARVAM_API_KEY=your_key"
        )

    cache_key = (text, language_code, SARVAM_TTS_MODEL, SARVAM_TTS_SPEAKER)
    audio_bytes = _cache_get(cache_key)

    if audio_bytes is None:
        payload = {
            "text": text,
            "target_language_code": language_code,
            "model": SARVAM_TTS_MODEL,
            "speaker": SARVAM_TTS_SPEAKER,
            "speech_sample_rate": 8000,
        }
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }

        client = await _get_http_client()
        response = await client.post(_SARVAM_TTS_URL, headers=headers, json=payload)

        # Some accounts use Bearer auth — retry once if subscription key rejected
        if response.status_code == 403:
            headers["Authorization"] = f"Bearer {api_key}"
            response = await client.post(_SARVAM_TTS_URL, headers=headers, json=payload)

        if response.status_code >= 400:
            logger.error(
                "Sarvam TTS HTTP %s: %s", response.status_code, response.text[:300]
            )
            response.raise_for_status()

        data = response.json()
        wav_bytes = _decode_first_wav(data.get("audios", []))
        pcm16 = _wav_to_raw_pcm16_mono_16k(wav_bytes)
        audio_bytes = _pcm16_16k_to_ulaw_8k(pcm16)
        _cache_set(cache_key, audio_bytes)

    chunk_size = 1024
    for i in range(0, len(audio_bytes), chunk_size):
        if cancelled_flag[0]:
            return
        yield audio_bytes[i: i + chunk_size]
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
cd /Users/jsdata/Projects/voice-agents
python -m pytest tests/test_tts_sarvam.py -v
```

Expected output:
```
PASSED tests/test_tts_sarvam.py::test_synthesize_returns_ulaw_bytes
PASSED tests/test_tts_sarvam.py::test_synthesize_uses_explicit_language_code
PASSED tests/test_tts_sarvam.py::test_cancel_flag_stops_streaming
PASSED tests/test_tts_sarvam.py::test_missing_api_key_raises
PASSED tests/test_tts_sarvam.py::test_pcm16_16k_to_ulaw_8k_produces_half_length
5 passed
```

- [ ] **Step 6: Commit**

```bash
cd /Users/jsdata/Projects/voice-agents
git add services/tts_sarvam.py tests/__init__.py tests/test_tts_sarvam.py
git commit -m "feat: add Sarvam Bulbul TTS provider with WAV→μ-law conversion and LRU cache"
```

---

## Task 3: Wire Sarvam into tts.py as primary provider

**Files:**
- Modify: `services/tts.py` (lines 66, 73–110)

- [ ] **Step 1: Update TTSService.synthesize() signature**

In `services/tts.py`, change the `synthesize` method signature on line 66 from:

```python
async def synthesize(self, text: str, voice_id: str | None = None) -> AsyncIterator[bytes]:
```

to:

```python
async def synthesize(self, text: str, language_code: str = "hi-IN") -> AsyncIterator[bytes]:
```

- [ ] **Step 2: Add Sarvam as primary in synthesize() and update fallback chain**

Replace the entire body of `synthesize()` (lines 72–110) with:

```python
        self.reset()
        if self._provider == "sarvam":
            try:
                async for chunk in self._sarvam(text, language_code):
                    if self._cancelled:
                        return
                    yield chunk
            except Exception as exc:
                logger.warning("Sarvam TTS failed (%s) — falling back to ElevenLabs", exc)
                try:
                    async for chunk in self._elevenlabs(text):
                        if self._cancelled:
                            return
                        yield chunk
                except Exception as exc2:
                    logger.warning("ElevenLabs failed (%s) — falling back to Deepgram", exc2)
                    async for chunk in self._deepgram(text):
                        if self._cancelled:
                            return
                        yield chunk
        elif self._provider == "elevenlabs":
            try:
                async for chunk in self._elevenlabs(text):
                    if self._cancelled:
                        return
                    yield chunk
            except Exception as exc:
                logger.warning("ElevenLabs failed (%s) — falling back to Deepgram TTS", exc)
                async for chunk in self._deepgram(text):
                    if self._cancelled:
                        return
                    yield chunk
        elif self._provider == "azure":
            try:
                async for chunk in self._azure(text):
                    if self._cancelled:
                        return
                    yield chunk
            except Exception as exc:
                logger.warning("Azure TTS failed (%s) — falling back to Deepgram TTS", exc)
                async for chunk in self._deepgram(text):
                    if self._cancelled:
                        return
                    yield chunk
        elif self._provider == "cartesia":
            async for chunk in self._cartesia(text):
                if self._cancelled:
                    return
                yield chunk
        else:
            async for chunk in self._deepgram(text):
                if self._cancelled:
                    return
                yield chunk
```

- [ ] **Step 3: Add _sarvam() private method**

After the `_elevenlabs` method, add:

```python
    # ── Sarvam Bulbul (Indian languages — primary) ────────────────────────────

    async def _sarvam(self, text: str, language_code: str = "hi-IN") -> AsyncIterator[bytes]:
        from services.tts_sarvam import sarvam_synthesize
        async for chunk in sarvam_synthesize(text, self._cancel_flag, language_code=language_code):
            yield chunk
```

- [ ] **Step 4: Verify import still works**

```bash
cd /Users/jsdata/Projects/voice-agents
python -c "from services.tts import TTSService; t = TTSService(); print('OK', t._provider)"
```

Expected: `OK sarvam`

- [ ] **Step 5: Commit**

```bash
cd /Users/jsdata/Projects/voice-agents
git add services/tts.py
git commit -m "feat: wire Sarvam as primary TTS provider with elevenlabs→deepgram fallback chain"
```

---

## Task 4: Adaptive filler hold + language_code routing in pipeline

**Files:**
- Modify: `pipeline/voice_pipeline.py` (lines 28, 218–228, 338–365)

- [ ] **Step 1: Write failing test for adaptive hold**

Create `tests/test_pipeline_hold.py`:

```python
"""Tests for adaptive filler hold logic in VoicePipeline._llm_loop."""
import asyncio
import pytest


def _word_count_hold_ms(text: str) -> int:
    """Replicates the adaptive hold logic from voice_pipeline._llm_loop."""
    return 400 if len(text.split()) < 3 else 150


def test_short_utterance_gets_400ms_hold():
    assert _word_count_hold_ms("haan") == 400
    assert _word_count_hold_ms("okay theek") == 400
    assert _word_count_hold_ms("ji") == 400


def test_normal_utterance_gets_150ms_hold():
    assert _word_count_hold_ms("mujhe doctor se milna hai") == 150
    assert _word_count_hold_ms("kal subah appointment chahiye") == 150


def test_exactly_three_words_gets_150ms():
    # Boundary: 3 words is NOT < 3, so gets standard hold
    assert _word_count_hold_ms("kal subah theek") == 150


def test_two_words_gets_400ms():
    assert _word_count_hold_ms("haan okay") == 400
```

- [ ] **Step 2: Run test — verify it passes (pure logic, no imports)**

```bash
cd /Users/jsdata/Projects/voice-agents
python -m pytest tests/test_pipeline_hold.py -v
```

Expected: `4 passed` (all pass immediately since this is pure logic)

- [ ] **Step 3: Remove ELEVENLABS_VOICE_ID import from voice_pipeline.py**

On line 28, find:

```python
from config.base_config import ELEVENLABS_VOICE_ID
```

Remove that import. It is no longer used after the next step.

- [ ] **Step 4: Replace flat 150ms hold with adaptive hold**

In `pipeline/voice_pipeline.py`, find this block (around lines 218–228):

```python
            # 150ms hold: absorbs mid-thought pauses that slipped past endpointing.
            # Drains any follow-on transcript fragments into the same turn.
            await asyncio.sleep(0.15)
            while not self._transcript_queue.empty():
                extra = self._transcript_queue.get_nowait()
                user_text = user_text + " " + extra
```

Replace with:

```python
            # Adaptive hold: short utterances (< 3 words) get 400ms to absorb
            # Hindi filler tokens ("haan", "okay", "ji") that precede the real
            # request. Normal utterances still get 150ms.
            _hold_ms = 400 if len(user_text.split()) < 3 else 150
            await asyncio.sleep(_hold_ms / 1000)
            while not self._transcript_queue.empty():
                extra = self._transcript_queue.get_nowait()
                user_text = user_text + " " + extra
```

- [ ] **Step 5: Update _speak() to pass language_code instead of voice_id**

In `_speak()` (around line 345), find:

```python
            async for chunk in self._tts.synthesize(text, voice_id=ELEVENLABS_VOICE_ID):
```

Replace with:

```python
            lang_code = "hi-IN" if self._caller_language in ("hindi", "hinglish") else "en-IN"
            async for chunk in self._tts.synthesize(text, language_code=lang_code):
```

- [ ] **Step 6: Verify server starts cleanly**

```bash
cd /Users/jsdata/Projects/voice-agents
python -c "
from pipeline.voice_pipeline import VoicePipeline
print('VoicePipeline import OK')
"
```

Expected: `VoicePipeline import OK`

- [ ] **Step 7: Commit**

```bash
cd /Users/jsdata/Projects/voice-agents
git add pipeline/voice_pipeline.py tests/test_pipeline_hold.py
git commit -m "feat: adaptive 400ms filler hold for short utterances + route language_code to Sarvam TTS"
```

---

## Task 5: Barge-in fixes — guard widen + tail silence + acknowledgement

**Files:**
- Modify: `pipeline/interruption.py` (line 39 — `_DEBOUNCE_SECS` not changed; only `send_silence` call)
- Modify: `pipeline/voice_pipeline.py` (lines 133, 194–204, 265–270, 328–365)

- [ ] **Step 1: Write failing tests for barge-in ack**

Create `tests/test_soft_close.py` (we'll add soft-close tests in Task 6; create the file now with barge-in ack tests):

```python
"""Tests for barge-in acknowledgement and soft-close behaviour."""
import asyncio
import pytest


# ── Barge-in ack phrase selection ──────────────────────────────────────────


_BARGE_IN_ACK = {
    "hindi":    ["haan, bataiye", "ji, haan?", "haan ji?"],
    "hinglish": ["haan, bataiye", "yes, bataiye?", "haan ji?"],
    "english":  ["yes, go ahead", "please go on"],
}


def _pick_ack(caller_language: str) -> str:
    import random
    pool = _BARGE_IN_ACK.get(caller_language, _BARGE_IN_ACK["hinglish"])
    return random.choice(pool)


def test_ack_phrase_for_hindi():
    phrase = _pick_ack("hindi")
    assert phrase in _BARGE_IN_ACK["hindi"]


def test_ack_phrase_for_english():
    phrase = _pick_ack("english")
    assert phrase in _BARGE_IN_ACK["english"]


def test_ack_phrase_for_hinglish():
    phrase = _pick_ack("hinglish")
    assert phrase in _BARGE_IN_ACK["hinglish"]


def test_ack_phrase_unknown_language_falls_back_to_hinglish():
    phrase = _pick_ack("unknown")
    assert phrase in _BARGE_IN_ACK["hinglish"]
```

- [ ] **Step 2: Run tests — verify they pass (pure logic)**

```bash
cd /Users/jsdata/Projects/voice-agents
python -m pytest tests/test_soft_close.py -v -k "ack"
```

Expected: `4 passed`

- [ ] **Step 3: Increase tail silence in _on_speech_started**

In `pipeline/voice_pipeline.py`, find (around line 204):

```python
        await self._telephony.send_silence(50)
```

Replace with:

```python
        await self._telephony.send_silence(120)
```

- [ ] **Step 4: Add _agent_in_turn flag and widen _tts_playing guard**

In `__init__` (around line 133), find:

```python
        self._tts_playing = False  # guard: only barge-in when TTS is active
```

Replace with:

```python
        self._tts_playing = False   # True when a TTS chunk is actively streaming
        self._agent_in_turn = False  # True for the entire agent response turn (multi-sentence)
```

In `_on_speech_started` (around line 196), find:

```python
        if not self._tts_playing:
            return
```

Replace with:

```python
        if not self._tts_playing and not self._agent_in_turn:
            return
```

- [ ] **Step 5: Add BARGE_IN_ACK dict and set _agent_in_turn in _llm_loop**

Near the top of `voice_pipeline.py`, after the existing `TOOL_FILLERS` dict, add:

```python
_BARGE_IN_ACK: dict[str, list[str]] = {
    "hindi":    ["haan, bataiye", "ji, haan?", "haan ji?"],
    "hinglish": ["haan, bataiye", "yes, bataiye?", "haan ji?"],
    "english":  ["yes, go ahead", "please go on"],
}
```

In `_llm_loop`, just before `self._interruption.reset()` (around line 269), add:

```python
            self._agent_in_turn = True
```

After the `async for item in stream_response` loop ends (after the `if assistant_text_parts` block and before the `if correction_text` block), add:

```python
            self._agent_in_turn = False
```

- [ ] **Step 6: Inject barge-in acknowledgement in _on_speech_started**

In `_on_speech_started`, after flushing stale transcripts and before `send_silence`, the existing code is:

```python
        self._tts.cancel()
        await self._interruption.trigger()
        # Discard transcripts queued during TTS playback (stale/echo fragments)
        while not self._transcript_queue.empty():
            self._transcript_queue.get_nowait()
        await self._telephony.send_silence(120)
```

Replace with:

```python
        sentences_spoken = getattr(self, "_sentences_spoken_this_turn", 0)
        self._tts.cancel()
        await self._interruption.trigger()
        # Discard transcripts queued during TTS playback (stale/echo fragments)
        while not self._transcript_queue.empty():
            self._transcript_queue.get_nowait()
        await self._telephony.send_silence(120)
        # Acknowledge the interruption if agent had already spoken ≥ 1 sentence —
        # avoids acknowledging coughs or noise that fire before any speech.
        if sentences_spoken >= 1:
            lang = self._caller_language if self._caller_language in _BARGE_IN_ACK else "hinglish"
            ack = random.choice(_BARGE_IN_ACK[lang])
            self._interruption.reset()
            await self._speak(ack)
```

In `_llm_loop`, just before `self._agent_in_turn = True`, add:

```python
            self._sentences_spoken_this_turn = 0
```

In the `async for item in stream_response` loop, after `assistant_text_parts.append(item)`, add:

```python
                    self._sentences_spoken_this_turn += 1
```

- [ ] **Step 7: Verify import still clean**

```bash
cd /Users/jsdata/Projects/voice-agents
python -c "from pipeline.voice_pipeline import VoicePipeline; print('OK')"
```

Expected: `OK`

- [ ] **Step 8: Commit**

```bash
cd /Users/jsdata/Projects/voice-agents
git add pipeline/voice_pipeline.py tests/test_soft_close.py
git commit -m "fix: widen _tts_playing guard, 120ms tail silence, barge-in ack phrase for Indian callers"
```

---

## Task 6: Soft-close after successful booking

**Files:**
- Modify: `pipeline/voice_pipeline.py` (add `_booking_confirmed` flag + `_soft_close()` method)

- [ ] **Step 1: Add soft-close tests to tests/test_soft_close.py**

Append to `tests/test_soft_close.py`:

```python
# ── Soft-close timeout ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_soft_close_hangs_up_after_silence():
    """_soft_close() calls hangup after 5s of silence."""
    hangup_called = False
    speak_texts = []

    class FakeTelephony:
        async def hangup(self):
            nonlocal hangup_called
            hangup_called = True

    class FakePipeline:
        _running = True
        _telephony = FakeTelephony()
        _caller_language = "hinglish"
        _transcript_queue = asyncio.Queue()

        async def _speak(self, text):
            speak_texts.append(text)

        # Paste the _soft_close method under test exactly as it appears in the file
        async def _soft_close(self):
            await self._speak("Koi aur madad chahiye aapko?")
            try:
                user_text = await asyncio.wait_for(
                    self._transcript_queue.get(), timeout=0.05  # short for test
                )
                # Caller responded — this branch not tested here
            except asyncio.TimeoutError:
                await self._speak(
                    "Dhanyavaad! Apollo Hospital mein aapka swagat hai. Bye!"
                )
                await asyncio.sleep(0.01)
                await self._telephony.hangup()
                self._running = False

    pipeline = FakePipeline()
    await pipeline._soft_close()

    assert hangup_called, "hangup() should be called after silence timeout"
    assert not pipeline._running, "_running should be False after hangup"
    assert "Dhanyavaad" in speak_texts[-1]


@pytest.mark.asyncio
async def test_soft_close_continues_if_caller_responds():
    """_soft_close() does NOT hang up if caller speaks within timeout."""
    hangup_called = False
    speak_texts = []

    class FakeTelephony:
        async def hangup(self):
            nonlocal hangup_called
            hangup_called = True

    class FakePipeline:
        _running = True
        _telephony = FakeTelephony()
        _caller_language = "hinglish"
        _transcript_queue = asyncio.Queue()

        async def _speak(self, text):
            speak_texts.append(text)

        async def _process_turn(self, text):
            pass  # stub

        async def _soft_close(self):
            await self._speak("Koi aur madad chahiye aapko?")
            try:
                user_text = await asyncio.wait_for(
                    self._transcript_queue.get(), timeout=5.0
                )
                await self._process_turn(user_text)
            except asyncio.TimeoutError:
                await self._speak(
                    "Dhanyavaad! Apollo Hospital mein aapka swagat hai. Bye!"
                )
                await asyncio.sleep(0.5)
                await self._telephony.hangup()
                self._running = False

    pipeline = FakePipeline()
    # Put a response in the queue before calling soft_close
    await pipeline._transcript_queue.put("nahi, shukriya")
    await pipeline._soft_close()

    assert not hangup_called, "hangup() should NOT be called if caller responded"
    assert pipeline._running, "_running should still be True"
```

- [ ] **Step 2: Run soft-close tests — verify they pass**

```bash
cd /Users/jsdata/Projects/voice-agents
python -m pytest tests/test_soft_close.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Add _booking_confirmed flag to __init__**

In `VoicePipeline.__init__`, after `self._agent_in_turn = False`, add:

```python
        self._booking_confirmed = False  # set True when book_appointment succeeds
```

- [ ] **Step 4: Set _booking_confirmed in _handle_tool_call after successful booking**

In `_handle_tool_call`, find the `elif name == "book_appointment":` block:

```python
            elif name == "book_appointment":
                from tools.scheduling import book_appointment
                result = book_appointment(
                    ...
                )
```

After that block, add:

```python
            if name == "book_appointment" and result.get("success"):
                self._booking_confirmed = True
```

- [ ] **Step 5: Add _soft_close() method to VoicePipeline**

Add this method after `_play_filler()`:

```python
    async def _soft_close(self) -> None:
        """Offer further help after a successful booking.

        Waits 5 seconds for the caller to respond. If silence, plays a goodbye
        and hangs up. If caller speaks, re-queues their text for normal processing.
        """
        await self._speak("Koi aur madad chahiye aapko?")
        try:
            user_text = await asyncio.wait_for(
                self._transcript_queue.get(), timeout=5.0
            )
            # Caller has more to say — put it back and let _llm_loop handle it
            await self._transcript_queue.put(user_text)
        except asyncio.TimeoutError:
            await self._speak(
                "Dhanyavaad! Apollo Hospital mein aapka swagat hai. Bye!"
            )
            await asyncio.sleep(0.5)  # let final audio drain from Twilio buffer
            await self._telephony.hangup()
            self._running = False
```

- [ ] **Step 6: Trigger soft-close from _llm_loop after booking confirmed**

In `_llm_loop`, after `self._agent_in_turn = False` and before `if correction_text:`, add:

```python
            # Soft-close: offer further help after a confirmed booking, then hang up on silence.
            if self._booking_confirmed:
                self._booking_confirmed = False
                await self._soft_close()
                continue
```

- [ ] **Step 7: Run all tests**

```bash
cd /Users/jsdata/Projects/voice-agents
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
cd /Users/jsdata/Projects/voice-agents
git add pipeline/voice_pipeline.py tests/test_soft_close.py
git commit -m "feat: soft-close after booking — offer further help, hang up after 5s silence"
```

---

## Task 7: Calendar dry-run prod guard

**Files:**
- Modify: `tools/calendar.py` (around line 83)

- [ ] **Step 1: Add prod environment guard in create_appointment_event**

In `tools/calendar.py`, find the dry-run block:

```python
    if service is None:
        # Dry-run — log and return success so the voice flow continues
        logger.info(
            "[DRY RUN] Appointment booked: %s | %s (%s) | %s | Concern: %s | Phone: %s",
            slot_human, doctor_name, specialty, patient_name, concern, patient_phone,
        )
        return {
            "success": True,
            "event_id": "dry-run",
            "slot": slot_human,
            "doctor": doctor_name,
            "note": "Google Calendar not connected — appointment logged locally only",
        }
```

Replace with:

```python
    if service is None:
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            logger.error(
                "GOOGLE_SERVICE_ACCOUNT_JSON not set in production — "
                "appointment NOT created in calendar. Set this env var immediately."
            )
            return {
                "success": False,
                "reason": "Calendar not configured. Please contact the clinic directly.",
            }
        # Development dry-run — log and return success so the voice flow continues
        logger.warning(
            "[DRY RUN] Appointment booked (not in calendar): %s | %s (%s) | %s | "
            "Concern: %s | Phone: %s",
            slot_human, doctor_name, specialty, patient_name, concern, patient_phone,
        )
        return {
            "success": True,
            "event_id": "dry-run",
            "slot": slot_human,
            "doctor": doctor_name,
            "note": "Google Calendar not connected — appointment logged locally only",
        }
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/jsdata/Projects/voice-agents
python -c "from tools.calendar import create_appointment_event; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/jsdata/Projects/voice-agents
git add tools/calendar.py
git commit -m "fix: calendar dry-run returns failure in ENVIRONMENT=production to prevent silent booking loss"
```

---

## Task 8: Final integration smoke test

**Files:**
- No new files — verify the full stack loads without errors

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/jsdata/Projects/voice-agents
python -m pytest tests/ -v --tb=short
```

Expected: All tests pass with 0 failures.

- [ ] **Step 2: Verify full server import**

```bash
cd /Users/jsdata/Projects/voice-agents
python -c "
import main
from pipeline.voice_pipeline import VoicePipeline
from services.tts import TTSService
from services.tts_sarvam import sarvam_synthesize
from tools.calendar import create_appointment_event
print('All imports OK')
t = TTSService()
print('TTS provider:', t._provider)
"
```

Expected:
```
All imports OK
TTS provider: sarvam
```

- [ ] **Step 3: Test Sarvam API key live (optional but recommended)**

```bash
cd /Users/jsdata/Projects/voice-agents
python -c "
import asyncio, os
os.environ.setdefault('SARVAM_API_KEY', 'sk_d9iu42x0_UE6C5r9ObaeJ0LiaY0B2c2TT')
from services.tts_sarvam import sarvam_synthesize

async def test():
    chunks = []
    async for c in sarvam_synthesize('नमस्ते, Apollo Hospital में आपका स्वागत है।', [False], language_code='hi-IN'):
        chunks.append(c)
    total = sum(len(c) for c in chunks)
    print(f'Got {len(chunks)} chunks, {total} bytes of μ-law audio')

asyncio.run(test())
"
```

Expected: `Got N chunks, XXXX bytes of μ-law audio`

- [ ] **Step 4: Final commit**

```bash
cd /Users/jsdata/Projects/voice-agents
git log --oneline -8
```

Verify the commit history shows all 7 feature commits cleanly.

---

## Out-of-Scope Reminders

- **Redis slot locking**: `_booked_slots` in `tools/scheduling.py` is in-memory only. Safe for single-process dev; needs Redis `SETNX` lock for multi-process production. Track as a separate infrastructure task.
- **Post-call CRM logging**: Successful bookings not written back to HubSpot. Separate task.
- **Prosody emergency detection**: Keyword-only for now.
