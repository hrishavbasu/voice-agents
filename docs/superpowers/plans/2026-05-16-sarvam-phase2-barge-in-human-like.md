# Sarvam Phase 2: Barge-in Latency & Human-like Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix live-call barge-in delay (~1–2s stop) and improve time-to-first-audio / voice naturalness on the Sarvam branch without changing providers.

**Architecture:** Split “stop playback now” from “debounce ack phrase”; reorder `_on_speech_started` so Twilio `clear` runs before any sleep; add one-slot TTS prefetch and startup warm-cache to hide Sarvam’s batch HTTP latency. Validate with a new mock-telephony E2E test (&lt;400ms stop).

**Tech Stack:** Python 3.11+, FastAPI, asyncio, pytest-asyncio, Sarvam STT/TTS HTTP+WS, Gemini (`google-genai`).

**Spec:** `docs/superpowers/specs/2026-05-16-sarvam-phase2-barge-in-human-like-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pipeline/interruption.py` | Modify | `trigger_immediate`, `debounce_ack_gate` |
| `pipeline/tts_prefetch.py` | Create | One-slot background Sarvam synthesis |
| `pipeline/voice_pipeline.py` | Modify | Barge-in order, cancel-aware `_speak`, ack mode, prefetch hook |
| `services/tts.py` | Modify | `is_cancelled` property |
| `services/tts_sarvam.py` | Modify | `warm_cache_phrases()`, optional `synthesize_to_bytes` |
| `services/llm_gemini.py` | Modify | First-clause 15-word cap |
| `config/base_config.py` | Modify | `BARGE_IN_ACK_MODE`, Phase 2 defaults |
| `prompts/system_prompt.py` | Modify | Short first-reply instruction |
| `main.py` | Modify | Call warm-cache in `lifespan` |
| `static/config.html` | Modify | `barge_in_ack_mode` dropdown |
| `.env.example` | Modify | Align with Sarvam/Gemini defaults |
| `tests/test_interruption.py` | Create | InterruptionController unit tests |
| `tests/test_barge_in_e2e.py` | Create | Mock telephony stop-latency test |
| `tests/test_tts_prefetch.py` | Create | Prefetch slot unit tests |
| `tests/test_llm_gemini.py` | Modify | First-clause split tests |

---

## Task 1: InterruptionController — immediate vs ack debounce

**Files:**
- Modify: `pipeline/interruption.py`
- Create: `tests/test_interruption.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_interruption.py`:

```python
import asyncio
import pytest
from pipeline.interruption import InterruptionController


@pytest.mark.asyncio
async def test_trigger_immediate_sets_flag_without_delay():
    ctrl = InterruptionController()
    t0 = asyncio.get_event_loop().time()
    await ctrl.trigger_immediate()
    elapsed = asyncio.get_event_loop().time() - t0
    assert ctrl.is_interrupted
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_debounce_ack_gate_waits_150ms():
    ctrl = InterruptionController()
    t0 = asyncio.get_event_loop().time()
    ok = await ctrl.debounce_ack_gate()
    elapsed = asyncio.get_event_loop().time() - t0
    assert ok is True
    assert elapsed >= 0.14


@pytest.mark.asyncio
async def test_debounce_ack_gate_returns_false_if_reset_during_wait():
    ctrl = InterruptionController()
    async def reset_soon():
        await asyncio.sleep(0.05)
        ctrl.reset()
    await asyncio.gather(ctrl.debounce_ack_gate(), reset_soon())
    assert ctrl.is_interrupted is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/jsdata/Projects/voice-agents && .venv/bin/python -m pytest tests/test_interruption.py -v
```

Expected: FAIL — `trigger_immediate` / `debounce_ack_gate` not defined.

- [ ] **Step 3: Implement `pipeline/interruption.py`**

Replace `trigger()` body with split methods; keep `trigger()` as alias:

```python
    async def trigger_immediate(self) -> None:
        """Stop TTS/LLM playback immediately — no debounce."""
        if self._interrupted:
            return
        self._interrupted = True
        self._debouncing = False
        self._event.set()
        logger.debug("Barge-in: immediate interrupt")

    async def debounce_ack_gate(self) -> bool:
        """Wait 150ms; return True if still interrupted (play ack). False if reset()."""
        if not self._interrupted:
            return False
        self._debouncing = True
        await asyncio.sleep(self._DEBOUNCE_SECS)
        self._debouncing = False
        return self._interrupted

    async def trigger(self) -> None:
        """Backward-compatible alias."""
        await self.trigger_immediate()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_interruption.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/interruption.py tests/test_interruption.py
git commit -m "feat: split immediate barge-in stop from ack debounce"
```

---

## Task 2: TTSService `is_cancelled` + cancel-aware `_speak`

**Files:**
- Modify: `services/tts.py`
- Modify: `pipeline/voice_pipeline.py` (`_speak` only in this task)
- Modify: `tests/test_tts_sarvam_barge_in.py` (add TTSService test) OR create snippet in Task 3

- [ ] **Step 1: Add property to `services/tts.py`**

Inside `class TTSService`, after `cancel()`:

```python
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
```

- [ ] **Step 2: Update `_speak` in `pipeline/voice_pipeline.py`**

Add helper at class level:

```python
    def _should_stop_speaking(self) -> bool:
        return self._interruption.is_interrupted or self._tts.is_cancelled
```

In `_speak`, replace every `if self._interruption.is_interrupted:` with `if self._should_stop_speaking():`.

Change the remainder flush block:

```python
            if buf and not self._should_stop_speaking():
                padded = bytes(buf) + bytes([0xFF] * (_FRAME - len(buf) % _FRAME))
                ...
```

- [ ] **Step 3: Run existing TTS tests**

```bash
.venv/bin/python -m pytest tests/test_tts_sarvam_barge_in.py tests/test_tts_sarvam.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add services/tts.py pipeline/voice_pipeline.py
git commit -m "feat: stop _speak loop on TTS cancel as well as interrupt flag"
```

---

## Task 3: Barge-in hot path reorder + E2E mock test

**Files:**
- Modify: `pipeline/voice_pipeline.py` (`_on_speech_started`)
- Create: `tests/test_barge_in_e2e.py`

- [ ] **Step 1: Write failing E2E test**

Create `tests/test_barge_in_e2e.py`:

```python
"""Mock telephony: barge-in must clear buffer and stop audio within 400ms."""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.interruption import InterruptionController
from services.tts import TTSService


class MockTelephony:
    def __init__(self):
        self.events: list[tuple[str, float]] = []
        self.stream_ready = asyncio.Event()
        self.stream_ready.set()

    async def send_audio(self, chunk: bytes) -> None:
        self.events.append(("audio", time.monotonic()))

    async def clear_playback_buffer(self) -> None:
        self.events.append(("clear", time.monotonic()))

    async def send_silence(self, ms: int) -> None:
        self.events.append(("silence", time.monotonic()))


@pytest.mark.asyncio
async def test_barge_in_stops_audio_within_400ms():
    """Simulate slow TTS stream; speech_started must clear and stop sends quickly."""
    telephony = MockTelephony()
    interruption = InterruptionController()
    tts = TTSService()

    async def slow_synth(text, language_code="hi-IN", pitch_override=None):
        for _ in range(50):  # 50 chunks ≈ long utterance
            if tts.is_cancelled:
                return
            await asyncio.sleep(0.02)  # 20ms per chunk
            yield b"\xff" * 160

    tts.synthesize = slow_synth  # type: ignore[method-assign]

    pipeline = MagicMock()
    pipeline._telephony = telephony
    pipeline._interruption = interruption
    pipeline._tts = tts
    pipeline._transcript_queue = asyncio.Queue()
    pipeline._tts_playing = True
    pipeline._agent_in_turn = True
    pipeline._playback_until = time.monotonic() + 10
    pipeline._sentences_spoken_this_turn = 1
    pipeline._caller_language = "hinglish"
    pipeline._transcript_queue = asyncio.Queue()

    from pipeline import voice_pipeline as vp

    async def run_speak():
        pipeline._interruption.reset()
        pipeline._tts.reset()
        pipeline._tts_playing = True
        total = 0
        async for chunk in pipeline._tts.synthesize("long"):
            if pipeline._interruption.is_interrupted or pipeline._tts.is_cancelled:
                break
            await telephony.send_audio(chunk)
            total += 1
        pipeline._tts_playing = False

    speak_task = asyncio.create_task(run_speak())
    await asyncio.sleep(0.15)  # let audio flow

    t_start = time.monotonic()
    # Inline hot path (matches spec order)
    pipeline._tts.cancel()
    await telephony.clear_playback_buffer()
    await interruption.trigger_immediate()
    while not pipeline._transcript_queue.empty():
        pipeline._transcript_queue.get_nowait()

    await asyncio.wait_for(speak_task, timeout=2.0)
    t_end = time.monotonic()

    clears = [t for ev, t in telephony.events if ev == "clear"]
    audios = [t for ev, t in telephony.events if ev == "audio"]
    assert clears, "clear_playback_buffer must be called"
    assert clears[0] - t_start < 0.05
    if audios:
        assert audios[-1] - t_start < 0.4, "no audio sends >400ms after barge-in start"
    assert t_end - t_start < 0.5
```

Adjust test to call real `VoicePipeline._on_speech_started` once Step 3 wires it — preferred final form:

```python
    from pipeline.voice_pipeline import VoicePipeline
    # ... build minimal VoicePipeline with mocks, patch _speak to slow_speak above
    await pipeline._on_speech_started()
```

- [ ] **Step 2: Run test — expect FAIL** (old order: clear after debounce)

```bash
.venv/bin/python -m pytest tests/test_barge_in_e2e.py -v
```

- [ ] **Step 3: Rewrite `_on_speech_started` in `pipeline/voice_pipeline.py`**

Replace method body with:

```python
    async def _on_speech_started(self) -> None:
        speaking_or_buffered = (
            self._tts_playing
            or self._agent_in_turn
            or (time.monotonic() < self._playback_until)
        )
        if not speaking_or_buffered:
            return

        sentences_spoken = getattr(self, "_sentences_spoken_this_turn", 0)
        self._tts.cancel()
        await self._telephony.clear_playback_buffer()
        await self._interruption.trigger_immediate()
        while not self._transcript_queue.empty():
            self._transcript_queue.get_nowait()
        await self._telephony.send_silence(80)
        self._cancel_prefetch()  # Task 5 — no-op until prefetch exists

        if not self._should_play_barge_in_ack(sentences_spoken):
            return

        if not await self._interruption.debounce_ack_gate():
            return
        lang = self._caller_language if self._caller_language in _BARGE_IN_ACK else "hinglish"
        ack = random.choice(_BARGE_IN_ACK[lang])
        self._interruption.reset()
        await self._speak(ack)
```

Add stub until Task 4:

```python
    def _should_play_barge_in_ack(self, sentences_spoken: int) -> bool:
        return sentences_spoken >= 1
```

- [ ] **Step 4: Run E2E test**

```bash
.venv/bin/python -m pytest tests/test_barge_in_e2e.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/voice_pipeline.py tests/test_barge_in_e2e.py
git commit -m "fix: barge-in clears Twilio buffer before debounce; add E2E latency test"
```

---

## Task 4: `BARGE_IN_ACK_MODE` config

**Files:**
- Modify: `config/base_config.py`
- Modify: `pipeline/voice_pipeline.py`
- Modify: `static/config.html`
- Modify: `main.py` (config API keys list if present)
- Create: tests in `tests/test_barge_in_ack.py` (small)

- [ ] **Step 1: Add config**

In `config/base_config.py` after silence timeout block:

```python
BARGE_IN_ACK_MODE = _get("barge_in_ack_mode", "sometimes").lower()  # silent | sometimes | always
```

- [ ] **Step 2: Implement `_should_play_barge_in_ack`**

```python
from config.base_config import BARGE_IN_ACK_MODE

    def _should_play_barge_in_ack(self, sentences_spoken: int) -> bool:
        if sentences_spoken < 1:
            return False
        mode = BARGE_IN_ACK_MODE
        if mode == "silent":
            return False
        if mode == "always":
            return True
        # sometimes: skip ack if caller already speaking (queue non-empty)
        return self._transcript_queue.empty()
```

- [ ] **Step 3: Config UI**

In `static/config.html`, add under STT card (or new “Agent behaviour” row):

```html
      <label>Barge-in acknowledgment</label>
      <select id="barge_in_ack_mode">
        <option value="silent">silent</option>
        <option value="sometimes">sometimes (default)</option>
        <option value="always">always</option>
      </select>
```

Add `'barge_in_ack_mode'` to the JS `FIELDS` array used by load/save.

- [ ] **Step 4: Test**

```python
# tests/test_barge_in_ack.py
def test_sometimes_skips_ack_when_transcript_queued():
    ...
```

Run: `.venv/bin/python -m pytest tests/test_barge_in_ack.py -v`

- [ ] **Step 5: Commit**

```bash
git add config/base_config.py pipeline/voice_pipeline.py static/config.html tests/test_barge_in_ack.py
git commit -m "feat: configurable barge-in ack mode (silent/sometimes/always)"
```

---

## Task 5: One-slot TTS prefetch

**Files:**
- Create: `pipeline/tts_prefetch.py`
- Modify: `services/tts_sarvam.py`
- Modify: `pipeline/voice_pipeline.py`
- Create: `tests/test_tts_prefetch.py`

- [ ] **Step 1: Add `synthesize_to_bytes` in `services/tts_sarvam.py`**

```python
async def sarvam_synthesize_to_bytes(
    text: str,
    language_code: str = "hi-IN",
    pitch_override: Optional[float] = None,
) -> bytes:
    cancelled = [False]
    parts = []
    async for chunk in sarvam_synthesize(text, cancelled, language_code, pitch_override):
        parts.append(chunk)
    return b"".join(parts)
```

- [ ] **Step 2: Write failing prefetch tests**

`tests/test_tts_prefetch.py`:

```python
@pytest.mark.asyncio
async def test_prefetch_returns_audio_bytes():
    with patch("pipeline.tts_prefetch.sarvam_synthesize_to_bytes", AsyncMock(return_value=b"\xff" * 320)):
        from pipeline.tts_prefetch import TtsPrefetchSlot
        slot = TtsPrefetchSlot()
        slot.start("hello", "hi-IN", None)
        await asyncio.sleep(0.05)
        assert slot.take() == b"\xff" * 320

@pytest.mark.asyncio
async def test_cancel_discards_prefetch():
    ...
```

- [ ] **Step 3: Create `pipeline/tts_prefetch.py`**

```python
import asyncio
import logging
from typing import Optional
from services.tts_sarvam import sarvam_synthesize_to_bytes

logger = logging.getLogger(__name__)


class TtsPrefetchSlot:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._text: Optional[str] = None
        self._result: Optional[bytes] = None

    def start(self, text: str, language_code: str, pitch_override: Optional[float]) -> None:
        self.cancel()
        self._text = text
        self._result = None

        async def _run():
            try:
                self._result = await sarvam_synthesize_to_bytes(text, language_code, pitch_override)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("Prefetch failed: %s", exc)

        self._task = asyncio.create_task(_run())

    def take(self) -> Optional[bytes]:
        if self._task and not self._task.done():
            return None
        data, self._result, self._text = self._result, None, None
        self._task = None
        return data

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self._result = None
        self._text = None
```

- [ ] **Step 4: Integrate in `VoicePipeline`**

In `__init__`:

```python
from pipeline.tts_prefetch import TtsPrefetchSlot
        self._prefetch = TtsPrefetchSlot()
        self._prefetch_text: Optional[str] = None
```

Add methods:

```python
    def _cancel_prefetch(self) -> None:
        self._prefetch.cancel()
        self._prefetch_text = None

    def _schedule_prefetch(self, text: str, lang_code: str, pitch) -> None:
        self._cancel_prefetch()
        self._prefetch_text = text
        self._prefetch.start(text, lang_code, pitch)
```

Update `_speak` signature:

```python
    async def _speak(self, text: str, pitch_override: float = None, prefetched_audio: bytes | None = None) -> float:
```

If `prefetched_audio` is not None, skip `self._tts.synthesize` and iterate chunks from `prefetched_audio` directly (same frame loop).

In `_llm_loop`, track `next_sentence: str | None = None`. On each `str` item:

```python
                prefetched = None
                if next_sentence and next_sentence == getattr(self, "_prefetch_text", None):
                    prefetched = self._prefetch.take()
                await self._speak(item, prefetched_audio=prefetched)
                # After scheduling speak, we don't know next yet — set next_sentence when we peek
```

Simpler integration pattern:

```python
            pending_next: str | None = None
            async for item in stream_response(...):
                ...
                if isinstance(item, str):
                    if pending_next:
                        self._schedule_prefetch(pending_next, lang_code, pitch_for(pending_next))
                    prefetched = self._prefetch.take() if self._prefetch_text == item else None
                    await self._speak(item, prefetched_audio=prefetched)
                    pending_next = None  # will set on next str
                # use a list buffer: collect item, on next str prefetch previous... 
```

**Recommended loop (explicit):**

```python
            sentence_buffer: list[str] = []
            async for item in stream_response(...):
                if isinstance(item, str):
                    sentence_buffer.append(item)
                    if len(sentence_buffer) >= 2:
                        prev, cur = sentence_buffer[-2], sentence_buffer[-1]
                        self._schedule_prefetch(cur, lang_code, None)
                        prefetched = self._prefetch.take() if self._prefetch_text == prev else None
                        await self._speak(prev, prefetched_audio=prefetched)
                        sentence_buffer = [cur]
            for s in sentence_buffer:
                prefetched = self._prefetch.take() if self._prefetch_text == s else None
                await self._speak(s, prefetched_audio=prefetched)
```

Refine during implementation so first sentence is not delayed; only prefetch **N+1** while speaking **N**.

On barge-in / turn end: `self._cancel_prefetch()`.

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/test_tts_prefetch.py tests/test_barge_in_e2e.py -v
```

- [ ] **Step 6: Commit**

```bash
git add pipeline/tts_prefetch.py pipeline/voice_pipeline.py services/tts_sarvam.py tests/test_tts_prefetch.py
git commit -m "feat: one-slot Sarvam TTS prefetch for faster multi-sentence turns"
```

---

## Task 6: Startup warm-cache

**Files:**
- Modify: `services/tts_sarvam.py`
- Modify: `main.py`
- Modify: `pipeline/voice_pipeline.py` (export phrase lists or duplicate in warm_cache module)

- [ ] **Step 1: Add `warm_cache_phrases` to `services/tts_sarvam.py`**

```python
async def warm_cache_phrases(phrases: list[tuple[str, str]]) -> None:
    """Fire-and-forget cache warm for (text, language_code) pairs."""
    for text, lang in phrases:
        if not text.strip():
            continue
        cancelled = [False]
        try:
            async for _ in sarvam_synthesize(text.strip(), cancelled, lang):
                pass
        except Exception as exc:
            logger.debug("Warm-cache skip %r: %s", text[:40], exc)
```

- [ ] **Step 2: Build phrase list helper**

In `pipeline/voice_pipeline.py` add module function:

```python
def collect_warm_cache_phrases() -> list[tuple[str, str]]:
    phrases = []
    for lang, fillers in TOOL_FILLERS.items():
        code = "hi-IN" if lang in ("hindi", "hinglish") else "en-IN"
        phrases.extend((f, code) for f in fillers)
    for lang, acks in _BARGE_IN_ACK.items():
        code = "hi-IN" if lang in ("hindi", "hinglish") else "en-IN"
        phrases.extend((a, code) for a in acks)
    return phrases
```

- [ ] **Step 3: Call from `main.py` lifespan**

```python
from config.base_config import TTS_PROVIDER, SARVAM_API_KEY
from services.tts_sarvam import warm_cache_phrases
from pipeline.voice_pipeline import collect_warm_cache_phrases

@asynccontextmanager
async def lifespan(app: FastAPI):
    ...
    if TTS_PROVIDER.lower() == "sarvam" and SARVAM_API_KEY:
        asyncio.create_task(warm_cache_phrases(collect_warm_cache_phrases()))
    yield
```

- [ ] **Step 4: Manual smoke**

```bash
.venv/bin/python -c "import asyncio; from services.tts_sarvam import warm_cache_phrases; asyncio.run(warm_cache_phrases([('namaste','hi-IN')]))"
```

(Skip if no API key; log only.)

- [ ] **Step 5: Commit**

```bash
git add services/tts_sarvam.py main.py pipeline/voice_pipeline.py
git commit -m "feat: warm Sarvam TTS cache for fillers and ack phrases at startup"
```

---

## Task 7: First-clause length guard + system prompt

**Files:**
- Modify: `services/llm_gemini.py`
- Modify: `prompts/system_prompt.py`
- Modify: `tests/test_llm_gemini.py`

- [ ] **Step 1: Add failing test**

In `tests/test_llm_gemini.py`:

```python
from services.llm_gemini import _limit_first_clause

def test_limit_first_clause_splits_long_first_sentence():
    long = "word " * 20 + "?"
    parts = _limit_first_clause(long, is_first=True)
    assert len(parts[0].split()) <= 15
    assert len(parts) >= 2
```

- [ ] **Step 2: Implement in `services/llm_gemini.py`**

```python
_MAX_FIRST_CLAUSE_WORDS = 15

def _limit_first_clause(sentence: str, is_first: bool) -> list[str]:
    if not is_first or len(sentence.split()) <= _MAX_FIRST_CLAUSE_WORDS:
        return [sentence]
    words = sentence.split()
    first = " ".join(words[:_MAX_FIRST_CLAUSE_WORDS])
    rest = " ".join(words[_MAX_FIRST_CLAUSE_WORDS:])
    return [first, rest] if rest.strip() else [first]
```

In `stream_response`, add `first_yielded = False` before the loop; when yielding subs:

```python
                        for sub in _clause_split(sentence):
                            for part in _limit_first_clause(sub, is_first=not first_yielded):
                                first_yielded = True
                                yield part
```

- [ ] **Step 3: System prompt one-liner**

In `prompts/system_prompt.py` persona section:

```python
"- Keep your very first spoken reply to at most 15 words before your first question.\n"
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_llm_gemini.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/llm_gemini.py prompts/system_prompt.py tests/test_llm_gemini.py
git commit -m "feat: cap first spoken clause at 15 words for faster time-to-audio"
```

---

## Task 8: Phase 2 config defaults + `.env.example`

**Files:**
- Modify: `config/base_config.py`
- Modify: `.env.example`

- [ ] **Step 1: Update defaults in `config/base_config.py`**

```python
SARVAM_TTS_MODEL   = _get("sarvam_tts_model", "bulbul:v3")
SARVAM_STT_INTERRUPT_MIN_FRAMES = int(_get("stt_interrupt_min_frames", "2"))
SARVAM_TTS_PACE     = float(_get("sarvam_tts_pace", "0.92"))
SARVAM_TTS_LOUDNESS = float(_get("sarvam_tts_loudness", "1.3"))
```

- [ ] **Step 2: Rewrite `.env.example` provider block** (placeholders only)

```bash
STT_PROVIDER=sarvam
TTS_PROVIDER=sarvam
TTS_ALLOW_FALLBACK=false
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_SPEAKER=pavithra
SARVAM_TTS_PACE=0.92
SARVAM_TTS_LOUDNESS=1.3
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here
SARVAM_API_KEY=your_sarvam_key_here
BARGE_IN_ACK_MODE=sometimes
```

Remove or comment Groq/OpenRouter as primary (keep keys optional if code still references).

- [ ] **Step 3: Commit**

```bash
git add config/base_config.py .env.example
git commit -m "chore: Phase 2 Sarvam/Gemini defaults and env example sync"
```

---

## Task 9: Full regression + manual live checklist

- [ ] **Step 1: Install missing test dep if needed**

```bash
.venv/bin/pip install google-genai
```

- [ ] **Step 2: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests pass (40+ including new files).

- [ ] **Step 3: Manual live checklist** (requires ngrok + Twilio)

| # | Action | Pass? |
|---|--------|-------|
| 1 | Interrupt mid-sentence during booking | Agent stops &lt; ~0.5s |
| 2 | Interrupt during tool filler | Filler stops |
| 3 | Interrupt during long slot list | No full repeat |

- [ ] **Step 4: Final commit if any test fixes**

```bash
git commit -m "test: Phase 2 regression green"
```

---

## Spec Coverage Self-Review

| Spec requirement | Task |
|------------------|------|
| `trigger_immediate` / ack debounce split | 1 |
| `_on_speech_started` reorder | 3 |
| Cancel-aware `_speak` | 2 |
| `BARGE_IN_ACK_MODE` | 4 |
| STT `interrupt_min_frames=2` | 8 |
| One-slot prefetch | 5 |
| Startup warm-cache | 6 |
| First-clause 15 words | 7 |
| `bulbul:v3`, pace/loudness defaults | 8 |
| E2E &lt;400ms test | 3 |
| Live checklist | 9 |
| Out of scope items | None in plan ✓ |

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-sarvam-phase2-barge-in-human-like.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement tasks in this session with checkpoints  

Which approach do you want?
