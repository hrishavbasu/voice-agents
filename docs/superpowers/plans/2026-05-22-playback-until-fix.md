# _playback_until Ghost Window Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the false barge-in detection triggered by the ghost window created by an outdated `_playback_until` formula in `pipeline/voice_pipeline.py`.

**Architecture:** One-line fix in `_speak()`. The formula `time.monotonic() + duration + 0.6` was correct when `_speak()` sent audio instantly, but audio is now paced frame-by-frame with `asyncio.sleep(0.018)` so `_speak()` takes ~`duration` seconds to return — leaving `_playback_until` pointing `duration` seconds into the future after TTS has already finished. Fix: set `_playback_until = time.monotonic() + 0.3` (a short post-TTS drain guard, not a full duration window).

**Tech Stack:** Python, asyncio, pytest

---

## Root Cause (for context)

`_speak()` in `pipeline/voice_pipeline.py` paces 160-byte frames with `asyncio.sleep(0.018)` (line 681). This makes the function take approximately `duration` seconds to return. Line 694 runs **after** the loop:

```python
duration = total_bytes / 8000
self._playback_until = time.monotonic() + duration + 0.6  # ← runs after ~duration seconds
```

So `_playback_until` is set to `now + duration + 0.6` where `now` is already `duration` seconds past the send start. The Twilio buffer has ~1 frame of audio left (~20 ms). But `_playback_until` stays live for `duration + 0.6` more seconds — any START_SPEECH in that window falsely fires the barge-in guard and cancels the LLM response.

**Fix:** Replace the formula with a fixed 300 ms post-completion drain guard:

```python
self._playback_until = time.monotonic() + 0.3
```

---

## File Map

| File | Action |
|------|--------|
| `pipeline/voice_pipeline.py` | Change line 694: `+ duration + 0.6` → `+ 0.3` |
| `tests/test_barge_in_pipeline.py` | Add regression test verifying `_playback_until` is within ~0.5s of `monotonic()` after `_speak()` returns |

---

### Task 1: Fix _playback_until formula

**Files:**
- Modify: `pipeline/voice_pipeline.py:694`
- Test: `tests/test_barge_in_pipeline.py`

- [ ] **Step 1: Write the failing test**

Read `tests/test_barge_in_pipeline.py` first to understand existing test patterns, then add:

```python
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_playback_until_is_short_after_speak(tmp_path):
    """_playback_until must be within 0.5s of now after _speak() returns.

    Regression: old formula set _playback_until = now + duration + 0.6 AFTER
    the real-time frame loop, creating a ghost window of ~duration seconds.
    """
    from pipeline.voice_pipeline import VoicePipeline

    telephony = MagicMock()
    telephony.send_audio = AsyncMock()

    pipeline = VoicePipeline.__new__(VoicePipeline)
    pipeline._telephony = telephony
    pipeline._tts_playing = False
    pipeline._running = True
    pipeline._greeting_finished = True
    pipeline._last_activity_at = 0.0
    pipeline._should_stop_speaking = MagicMock(return_value=False)
    pipeline._playback_until = 0.0
    pipeline.call_id = "test"

    # Mock TTS to return ~1 second of audio (8000 bytes of mulaw at 8kHz)
    fake_audio = b"\xff" * 8000  # 1 second
    with patch("pipeline.voice_pipeline.synthesize_speech", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = fake_audio
        with patch("pipeline.voice_pipeline.append_message", new_callable=AsyncMock):
            await pipeline._speak("test utterance", record_transcript=False)

    # _playback_until must be at most 0.5s in the future
    delta = pipeline._playback_until - time.monotonic()
    assert delta <= 0.5, (
        f"_playback_until is {delta:.2f}s in the future after _speak() returned — "
        f"ghost window too long. Expected ≤ 0.5s (the 0.3s drain guard)."
    )
    assert delta > 0, "_playback_until should still be in the future (drain guard active)"
```

Add to the top of the file if not already present:
```python
import pytest
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/hrishavbasuchoudhury/Developer/Customer_support_MVP
python -m pytest tests/test_barge_in_pipeline.py::test_playback_until_is_short_after_speak -v
```

Expected: FAIL — `AssertionError: _playback_until is ~1.6s in the future after _speak() returned`

- [ ] **Step 3: Apply the one-line fix**

In `pipeline/voice_pipeline.py`, line 694, change:

```python
        self._playback_until = time.monotonic() + duration + 0.6
```

To:

```python
        self._playback_until = time.monotonic() + 0.3
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_barge_in_pipeline.py::test_playback_until_is_short_after_speak -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite to catch regressions**

```bash
python -m pytest --tb=short -q
```

Expected: All existing tests pass. Any failure is a regression — fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add pipeline/voice_pipeline.py tests/test_barge_in_pipeline.py
git commit -m "fix: _playback_until ghost window — use 0.3s drain guard instead of duration+0.6"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|------------|------|
| Fix `_playback_until` formula at line 694 | Task 1 Step 3 |
| Failing test that catches the regression | Task 1 Step 1 |
| Test verifies post-fix: `_playback_until ≤ 0.5s` from now | Task 1 Step 4 |

All requirements covered.

**Placeholder scan:** No TBDs. All code complete.

**Type consistency:** `time.monotonic()` returns float; `0.3` is float — consistent.
