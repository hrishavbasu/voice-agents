# Empathy Level Configuration — Design Spec

**Date:** 2026-05-22
**Status:** Approved

---

## Goal

Add a configurable `empathy_level` (1–5) that controls both the agent's general tone and its frustrated-caller handling, so clinic operators can tune the agent's emotional register without touching code.

## Architecture

One new integer config key (`empathy_level`, default 3) feeds two sections of the LLM system prompt. No new endpoints, no new storage mechanism — everything uses the existing `_get()` / `user_settings.json` / `POST /config` pattern.

## Data Model

**`config/base_config.py`**

```python
EMPATHY_LEVEL = int(_get("empathy_level", "3"))  # 1 (efficient) – 5 (deeply empathetic)
```

Stored in `config/user_settings.json` under key `"empathy_level"`. The existing `/config` POST endpoint saves it automatically.

Valid range: 1–5 integer. Values outside range are clamped at runtime (`max(1, min(5, value))`).

## System Prompt Changes

`build_system_prompt()` in `prompts/system_prompt.py` imports `EMPATHY_LEVEL` and uses it in two places:

### Block 1 — Persona line

Replaces `cfg.get("persona", ...)` with a level-driven string:

| Level | Persona string |
|-------|----------------|
| 1 | `"efficient, precise, and solution-focused. Get to the point quickly."` |
| 2 | `"professional and helpful. Acknowledge briefly, then act."` |
| 3 | `"warm, professional, and caring."` *(current default)* |
| 4 | `"warm, empathetic, and patient. Acknowledge feelings before acting."` |
| 5 | `"deeply empathetic and emotionally present. Lead with feelings, never rush the caller."` |

The company config `persona` field is no longer used for the prompt persona line (it was rarely customised). The `empathy_level` fully controls this.

### Block 2 — Frustrated-caller handling section

Replaces the current hardcoded section with a level-specific block:

| Level | Acknowledgment style | Escalation trigger |
|-------|----------------------|--------------------|
| 1 | Acknowledge once ("sorry for that"), pivot immediately to solution. No repeat acknowledgment. | Explicit request only |
| 2 | Brief empathy phrase, pivot to concrete fix. | After 3 frustrated turns |
| 3 | Lead with empathy before solution, offer concrete next step. *(current)* | After 2 more frustrated turns |
| 4 | Strong acknowledgment, name the feeling explicitly ("I can hear this has been frustrating"). | After 1 more frustrated turn |
| 5 | Always lead with emotion. Check in proactively ("Are you okay to continue?"). Offer human handoff on first signal of distress. | Proactively on first distress signal |

Frustration signals stay the same across all levels (yaar, kya hua, itni der, raised voice, etc.) — only the response and escalation threshold change.

## Admin UI

**`static/config.html`** — Agent Settings section:

- Label: **Empathy Level**
- Input: `<input type="range" min="1" max="5" step="1">` with numeric readout
- Live description label that updates as the slider moves:
  - 1 → "Efficient — task-focused, minimal emotional acknowledgment"
  - 2 → "Professional — brief acknowledgment, quick to solution"
  - 3 → "Warm — balanced empathy and efficiency (default)"
  - 4 → "Empathetic — feelings-first, patient"
  - 5 → "Deeply Empathetic — lead with emotion, proactive human handoff"
- Saved as `{ "empathy_level": <int> }` via existing `POST /config`

## Files Changed

| File | Change |
|------|--------|
| `config/base_config.py` | Add `EMPATHY_LEVEL = int(_get("empathy_level", "3"))` |
| `prompts/system_prompt.py` | Import `EMPATHY_LEVEL`; replace persona string and frustrated-caller section with level-driven dicts |
| `static/config.html` | Add slider + live label to Agent Settings section |

## Error Handling

- Non-integer or out-of-range values from `user_settings.json` are clamped to 1–5 at read time in `base_config.py`.
- Missing key falls back to default 3 — no behaviour change on existing deployments.

## Testing

- Unit test: `build_system_prompt()` called with each level produces the correct persona and frustrated-caller text.
- Manual: set level 1 via `/config` UI, make a call, verify the agent is terse; set level 5, verify stronger emotional language.
