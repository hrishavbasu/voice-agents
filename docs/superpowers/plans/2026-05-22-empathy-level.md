# Empathy Level Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable `empathy_level` (1–5) integer that controls the agent's persona line and frustrated-caller handling section in the LLM system prompt, with a slider in the admin UI.

**Architecture:** One new config key reads via the existing `_get()` pattern and feeds two string lookups in `build_system_prompt()`. No new endpoints, no new tables — everything slots into the existing infrastructure.

**Tech Stack:** Python, FastAPI, HTML/JS (existing admin UI), pytest

---

## File Map

| File | Action |
|------|--------|
| `config/base_config.py` | Add `EMPATHY_LEVEL = int(_get("empathy_level", "3"))` with clamping |
| `prompts/system_prompt.py` | Import `EMPATHY_LEVEL`; replace persona string and frustrated-caller section with level-driven dicts |
| `static/config.html` | Add slider + live description label to Agent section; add `empathy_level` to FIELDS array and RANGES set |
| `tests/test_system_prompt.py` | New file — unit tests for `build_system_prompt()` at each empathy level |

---

### Task 1: Add EMPATHY_LEVEL to base_config

**Files:**
- Modify: `config/base_config.py` (after `BARGE_IN_ACK_MODE` line, around line 103)
- Test: `tests/test_system_prompt.py` (create new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_system_prompt.py`:

```python
"""Unit tests for build_system_prompt() empathy level behaviour."""
import importlib
import sys
import pytest
from unittest.mock import patch


def _reload_base_config(empathy_level: int):
    """Force-reload base_config with a given empathy_level in user_settings."""
    import config.user_settings as us_mod
    with patch.object(us_mod, "_PATH", "/nonexistent_path.json"):
        with patch.dict("os.environ", {"EMPATHY_LEVEL": str(empathy_level)}, clear=False):
            if "config.base_config" in sys.modules:
                del sys.modules["config.base_config"]
            import config.base_config as bc
            return bc


def test_empathy_level_default_is_3():
    """Default empathy level is 3 when no config or env var is set."""
    import config.user_settings as us_mod
    with patch.object(us_mod, "_PATH", "/nonexistent_path.json"):
        if "config.base_config" in sys.modules:
            del sys.modules["config.base_config"]
        import config.base_config as bc
        assert bc.EMPATHY_LEVEL == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/hrishavbasuchoudhury/Developer/Customer_support_MVP
python -m pytest tests/test_system_prompt.py::test_empathy_level_default_is_3 -v
```

Expected: FAIL — `AttributeError: module 'config.base_config' has no attribute 'EMPATHY_LEVEL'`

- [ ] **Step 3: Add EMPATHY_LEVEL to base_config.py**

In `config/base_config.py`, after line 103 (`BARGE_IN_ACK_MODE = ...`), add:

```python
# ── Empathy ───────────────────────────────────────────────────────────────────
EMPATHY_LEVEL = max(1, min(5, int(_get("empathy_level", "3"))))  # 1 (efficient) – 5 (deeply empathetic)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_system_prompt.py::test_empathy_level_default_is_3 -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/base_config.py tests/test_system_prompt.py
git commit -m "feat: add EMPATHY_LEVEL config key (default 3, clamped 1-5)"
```

---

### Task 2: Replace persona and frustrated-caller sections in system_prompt.py

**Files:**
- Modify: `prompts/system_prompt.py`
- Test: `tests/test_system_prompt.py`

The current code (line 66) uses:
```python
persona = cfg.get("persona", "warm, professional, and caring")
```

And the frustrated-caller section (lines 258–264) is hardcoded for level 3.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_system_prompt.py`:

```python
from prompts.system_prompt import build_system_prompt


def _build(empathy_level: int) -> str:
    """Build a system prompt with a patched EMPATHY_LEVEL."""
    import prompts.system_prompt as spm
    with patch.object(spm, "EMPATHY_LEVEL", empathy_level):
        return build_system_prompt()


def test_level_1_persona_is_efficient():
    prompt = _build(1)
    assert "efficient, precise, and solution-focused" in prompt


def test_level_3_persona_is_warm():
    prompt = _build(3)
    assert "warm, professional, and caring" in prompt


def test_level_5_persona_is_deeply_empathetic():
    prompt = _build(5)
    assert "deeply empathetic and emotionally present" in prompt


def test_level_1_frustrated_caller_acknowledge_once():
    prompt = _build(1)
    assert "Acknowledge once" in prompt or "pivot immediately" in prompt.lower()


def test_level_5_frustrated_caller_proactive_handoff():
    prompt = _build(5)
    assert "first signal of distress" in prompt or "proactively" in prompt.lower()


def test_level_3_frustrated_caller_after_2_turns():
    prompt = _build(3)
    assert "2 more frustrated" in prompt or "two more" in prompt.lower() or "after 2" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_system_prompt.py -k "persona or frustrated" -v
```

Expected: All FAIL — prompts return level-3 hardcoded text regardless.

- [ ] **Step 3: Implement empathy-driven prompt sections in system_prompt.py**

At the top of `prompts/system_prompt.py`, add the import after the existing imports:

```python
from config.base_config import KB_RELOAD_INTERVAL_SECONDS, EMPATHY_LEVEL
```

(Replace the existing `from config.base_config import KB_RELOAD_INTERVAL_SECONDS` line.)

Add two dicts just before `build_system_prompt()` (after `_kb_cache`):

```python
_PERSONA_BY_LEVEL = {
    1: "efficient, precise, and solution-focused. Get to the point quickly.",
    2: "professional and helpful. Acknowledge briefly, then act.",
    3: "warm, professional, and caring.",
    4: "warm, empathetic, and patient. Acknowledge feelings before acting.",
    5: "deeply empathetic and emotionally present. Lead with feelings, never rush the caller.",
}

_FRUSTRATED_CALLER_BY_LEVEL = {
    1: (
        "## Handling frustrated or irate callers\n"
        "If the caller sounds frustrated, upset, or uses signals like \"yaar\", \"kya hua\", \"itni der\", "
        "\"baat nahi sun rahe\", \"bahut time ho gaya\", raised voice, or repeated complaints:\n"
        "1. Acknowledge once (\"sorry for that\") and pivot immediately to the solution. No repeat acknowledgment.\n"
        "2. Do NOT repeat the question that triggered the frustration.\n"
        "3. Offer a concrete next step immediately: an alternate slot, a different doctor, or escalation.\n"
        "4. Offer `escalate_to_human` only if the caller explicitly requests it."
    ),
    2: (
        "## Handling frustrated or irate callers\n"
        "If the caller sounds frustrated, upset, or uses signals like \"yaar\", \"kya hua\", \"itni der\", "
        "\"baat nahi sun rahe\", \"bahut time ho gaya\", raised voice, or repeated complaints:\n"
        "1. Use a brief empathy phrase, then pivot to a concrete fix immediately.\n"
        "2. Do NOT repeat the question that triggered the frustration.\n"
        "3. Offer a concrete next step immediately: an alternate slot, a different doctor, or escalation.\n"
        "4. If the caller remains frustrated for 3 more turns after your empathy response, "
        "proactively offer `escalate_to_human`."
    ),
    3: (
        "## Handling frustrated or irate callers\n"
        "If the caller sounds frustrated, upset, or uses signals like \"yaar\", \"kya hua\", \"itni der\", "
        "\"baat nahi sun rahe\", \"bahut time ho gaya\", raised voice, or repeated complaints:\n"
        "1. Acknowledge first — always lead with empathy before anything else: "
        "\"Samajh mein aata hai, sorry for the inconvenience.\" / \"समझ में आता है, माफ़ी।\"\n"
        "2. Do NOT repeat the same question that triggered the frustration.\n"
        "3. Offer a concrete next step immediately: an alternate slot, a different doctor, or escalation.\n"
        "4. If the caller remains frustrated for 2 more turns after your empathy response, "
        "proactively offer `escalate_to_human` — do not wait for them to ask."
    ),
    4: (
        "## Handling frustrated or irate callers\n"
        "If the caller sounds frustrated, upset, or uses signals like \"yaar\", \"kya hua\", \"itni der\", "
        "\"baat nahi sun rahe\", \"bahut time ho gaya\", raised voice, or repeated complaints:\n"
        "1. Use strong acknowledgment — name the feeling explicitly: "
        "\"I can hear this has been frustrating\" / \"मैं समझ सकता/सकती हूँ — यह परेशान करने वाला है।\"\n"
        "2. Do NOT repeat the question that triggered the frustration.\n"
        "3. Offer a concrete next step immediately: an alternate slot, a different doctor, or escalation.\n"
        "4. If the caller remains frustrated for 1 more turn after your empathy response, "
        "proactively offer `escalate_to_human`."
    ),
    5: (
        "## Handling frustrated or irate callers\n"
        "If the caller sounds frustrated, upset, or uses signals like \"yaar\", \"kya hua\", \"itni der\", "
        "\"baat nahi sun rahe\", \"bahut time ho gaya\", raised voice, or repeated complaints:\n"
        "1. Always lead with emotion — acknowledge deeply before ANY action.\n"
        "2. Check in proactively: \"Are you okay to continue?\" / \"क्या आप ठीक हैं?\"\n"
        "3. Offer `escalate_to_human` on the first signal of distress — do NOT wait for the caller to ask.\n"
        "4. Never rush the caller through steps while they are distressed."
    ),
}
```

In `build_system_prompt()`, replace line 66:
```python
# OLD:
persona = cfg.get("persona", "warm, professional, and caring")
# NEW:
persona = _PERSONA_BY_LEVEL.get(EMPATHY_LEVEL, _PERSONA_BY_LEVEL[3])
```

Replace the entire `## Handling frustrated or irate callers` section (lines 258–264 in current file) in the prompt f-string. Find:

```python
## Handling frustrated or irate callers
If the caller sounds frustrated, upset, or uses signals like "yaar", "kya hua", "itni der", "baat nahi sun rahe", "bahut time ho gaya", raised voice, or repeated complaints:
1. Acknowledge first — always lead with empathy before anything else: "Samajh mein aata hai, sorry for the inconvenience." / "समझ में आता है, माफ़ी।"
2. Do NOT repeat the same question that triggered the frustration.
3. Offer a concrete next step immediately: an alternate slot, a different doctor, or escalation.
4. If the caller remains frustrated for 2 more turns after your empathy response, proactively offer `escalate_to_human` — do not wait for them to ask.
```

Replace with:

```python
{_FRUSTRATED_CALLER_BY_LEVEL.get(EMPATHY_LEVEL, _FRUSTRATED_CALLER_BY_LEVEL[3])}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_system_prompt.py -v
```

Expected: All PASS

- [ ] **Step 5: Run full test suite to catch regressions**

```bash
python -m pytest --tb=short -q
```

Expected: All existing tests pass. Any failure is a regression — fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add prompts/system_prompt.py tests/test_system_prompt.py
git commit -m "feat: drive persona and frustrated-caller prompt from EMPATHY_LEVEL"
```

---

### Task 3: Add empathy slider to admin UI

**Files:**
- Modify: `static/config.html`

- [ ] **Step 1: Add the slider HTML to the Agent section**

In `static/config.html`, find the Agent section's card body. It contains `agent_name`, `agent_persona`, `barge_in_ack_mode`, `agent_default_language`. Add the empathy slider AFTER the `agent_persona` textarea field block and BEFORE `barge_in_ack_mode`:

Find this block in the agent-body div:
```html
    <div class="field">
      <label>Persona</label>
      <textarea id="agent_persona" placeholder="warm, professional, and caring"></textarea>
    </div>
    <div class="field">
      <label>Barge-in acknowledgment</label>
```

Replace with:
```html
    <div class="field">
      <label>Persona</label>
      <textarea id="agent_persona" placeholder="warm, professional, and caring"></textarea>
    </div>
    <div class="field">
      <label>Empathy Level <small>(controls tone and frustrated-caller handling)</small></label>
      <div class="range-row">
        <input type="range" id="empathy_level" min="1" max="5" step="1"
               oninput="rv('empathy_level_v', this.value); updateEmpathyLabel(this.value)">
        <span class="range-val" id="empathy_level_v">3</span>
      </div>
      <div id="empathy_label" style="font-size:0.82rem; color:#555; margin-top:4px;">
        Warm — balanced empathy and efficiency (default)
      </div>
    </div>
    <div class="field">
      <label>Barge-in acknowledgment</label>
```

- [ ] **Step 2: Add empathy_level to FIELDS array**

Find the FIELDS array in the `<script>` section:
```javascript
  'agent_name','agent_persona','agent_default_language','barge_in_ack_mode',
```

Replace with:
```javascript
  'agent_name','agent_persona','agent_default_language','barge_in_ack_mode','empathy_level',
```

- [ ] **Step 3: Add empathy_level to RANGES set**

Find:
```javascript
const RANGES = new Set(['stt_volume_threshold','sarvam_tts_pace','sarvam_tts_pitch','sarvam_tts_loudness','llm_temperature']);
```

Replace with:
```javascript
const RANGES = new Set(['stt_volume_threshold','sarvam_tts_pace','sarvam_tts_pitch','sarvam_tts_loudness','llm_temperature','empathy_level']);
```

- [ ] **Step 4: Add the updateEmpathyLabel function**

Find the `function rv(id, val)` function in the script section and add the new function right after it:

```javascript
const EMPATHY_LABELS = {
  1: 'Efficient — task-focused, minimal emotional acknowledgment',
  2: 'Professional — brief acknowledgment, quick to solution',
  3: 'Warm — balanced empathy and efficiency (default)',
  4: 'Empathetic — feelings-first, patient',
  5: 'Deeply Empathetic — lead with emotion, proactive human handoff',
};
function updateEmpathyLabel(val) {
  const el = document.getElementById('empathy_label');
  if (el) el.textContent = EMPATHY_LABELS[parseInt(val)] || '';
}
```

- [ ] **Step 5: Initialise slider value on page load**

The existing `loadConfig()` function sets slider values from loaded settings. Check that the `empathy_level` key gets its integer value set correctly. The RANGES set ensures it uses `parseInt` instead of `parseFloat`. Find the load logic:

```javascript
if (RANGES.has(id)) {
```

Verify `empathy_level` is in RANGES (done in Step 3). Also add a call to `updateEmpathyLabel` after the slider value is set. In the `loadConfig()` function, find where RANGE fields are populated and ensure `updateEmpathyLabel` is called with the loaded value. Add after the existing RANGES block:

```javascript
// Sync empathy label on load
const empathyEl = document.getElementById('empathy_level');
if (empathyEl) updateEmpathyLabel(empathyEl.value);
```

- [ ] **Step 6: Verify UI manually**

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/config/ui` → expand Agent section → verify slider is visible with label "Warm — balanced empathy and efficiency (default)" → drag to 5, verify label changes to "Deeply Empathetic..." → click Save → verify `POST /config` receives `{"empathy_level": 5}`.

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest --tb=short -q
```

Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add static/config.html
git commit -m "feat: add empathy level slider to admin UI"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| `EMPATHY_LEVEL = int(_get("empathy_level", "3"))` with clamp | Task 1 |
| 5-level persona strings in prompt | Task 2 |
| 5-level frustrated-caller handling in prompt | Task 2 |
| Slider `min=1 max=5 step=1` with live description label | Task 3 |
| `empathy_level` saved via existing `POST /config` | Task 3 (FIELDS + RANGES) |
| Default 3 — no behaviour change on existing deployments | Task 1 (default "3") |
| Out-of-range values clamped to 1–5 | Task 1 (`max(1, min(5, ...))`) |
| Unit test: each level produces correct persona + frustrated-caller text | Task 2 |

All spec requirements covered. No gaps found.

**Placeholder scan:** No TBDs, no "fill in details" — all code blocks complete.

**Type consistency:** `EMPATHY_LEVEL` (int) used as dict key (int) throughout — consistent.
