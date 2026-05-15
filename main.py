"""
Customer Support AI — FastAPI entry point.

Endpoints:
  POST /twilio/voice          — Twilio webhook: returns TwiML to open a Media Stream
  POST /telnyx/voice          — Telnyx webhook: starts call-control flow
  WS   /ws/stream/{call_id}   — Carrier WebSocket: bi-directional audio stream
  GET  /health                — Health check

Set TELEPHONY_PROVIDER=twilio (default) or telnyx in your .env.
"""

import asyncio
from functools import lru_cache
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from config.base_config import HOST, PORT, PUBLIC_URL, LOG_LEVEL
from pipeline.voice_pipeline import VoicePipeline
from services.telephony import TelephonySession
from services.pronunciation_dict import ensure_pronunciation_dict

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TELEPHONY_PROVIDER = os.getenv("TELEPHONY_PROVIDER", "twilio").lower()

# Active pipelines: call_id → VoicePipeline
_active_pipelines: dict[str, VoicePipeline] = {}


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Customer Support AI starting — provider=%s public_url=%s",
        TELEPHONY_PROVIDER,
        PUBLIC_URL or "(not set — set PUBLIC_URL for webhooks)",
    )
    await ensure_pronunciation_dict()

    # Pre-warm Sarvam TTS cache in background so filler phrases and the
    # greeting have near-zero latency (served from LRU cache, no API round-trip).
    if os.getenv("TTS_PROVIDER", "").lower() == "sarvam":
        from pipeline.voice_pipeline import TOOL_FILLERS, BACKCHANNEL_SOUNDS
        from services.tts_sarvam import prewarm_sarvam_cache
        from config.company_config import COMPANY_CONFIG
        phrases: list[str] = []
        for lang_phrases in TOOL_FILLERS.values():
            phrases.extend(lang_phrases)
        for lang_phrases in BACKCHANNEL_SOUNDS.values():
            phrases.extend(lang_phrases)
        # Pre-warm all 3 language variants of the greeting so call connect is instant
        company = COMPANY_CONFIG.get("company_name", "our hospital")
        agent = COMPANY_CONFIG.get("agent_name", "Priya")
        phrases += [
            f"नमस्ते, {company} में आपका स्वागत है। मैं {agent} बोल रही हूँ। बताइए, आज आपको किस डॉक्टर से मिलना है?",
            f"नमस्ते, {company} में आपका स्वागत है। मैं {agent} बोल रही हूँ। बताइए, आज आपको किस डॉक्टर से अपॉइंटमेंट चाहिए?",
            f"Hello, thank you for calling {company}. I am {agent}. How can I help you today?",
        ]
        asyncio.create_task(prewarm_sarvam_cache(phrases))

    yield
    # Shutdown: close any lingering pipelines
    for pipeline in list(_active_pipelines.values()):
        await pipeline.shutdown()
    logger.info("Customer Support AI stopped")


app = FastAPI(title="Customer Support AI", lifespan=lifespan)

# In-memory browser sessions: session_id → message history
_browser_sessions: dict[str, list[dict]] = {}
_BROWSER_MAX_CONTEXT_MESSAGES = int(os.getenv("BROWSER_MAX_CONTEXT_MESSAGES", "6"))
_BROWSER_FAST_MODEL = os.getenv("BROWSER_FAST_LLM_MODEL", "").strip()


_WAITING_NOISE_RE = re.compile(
    r"\(\s*waiting\s+for\s+user(?:\s+response)?\s*\)|"
    r"\(\s*कृपया\s+इंतज़ार\s+करें\s*\)",
    re.IGNORECASE,
)


def _neutralize_caller_gender_hindi(text: str) -> str:
    """Avoid gendered caller-address forms while preserving female self-reference."""
    if not text:
        return text
    replacements = {
        "यह बताएँगी": "यह बताइए",
        "ये बताएँगी": "ये बताइए",
        "बताएँगी?": "बताइए?",
        "बताएँगी।": "बताइए।",
        "बताएँगी": "बताइए",
        "चाहती हैं?": "चाहेंगे?",
        "चाहती हैं।": "चाहेंगे।",
        "चाहती हैं": "चाहेंगे",
        "बता सकती हैं?": "बताइए?",
        "बता सकती हैं।": "बताइए।",
        "बता सकती हैं": "बताइए",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _clean_assistant_text(text: str) -> str:
    """Post-process assistant text for cleaner browser UX."""
    if not text:
        return text

    cleaned = _WAITING_NOISE_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _neutralize_caller_gender_hindi(cleaned)

    # Enforce one-question style in browser mode: keep at most one question.
    q_idx = cleaned.find("?")
    if q_idx >= 0:
        prefix = cleaned[: q_idx + 1]
        # Keep short leading context, drop trailing extra prompts/questions.
        cleaned = prefix

    return cleaned.strip()


def _build_browser_system_prompt(caller_language: str) -> str:
    """Compact browser prompt for lower latency while preserving behavior."""
    lang_line = {
        "english": "Reply only in English.",
        "hindi": "Reply only in Hindi (Devanagari script).",
        "hinglish": "Reply in natural Hinglish, with Hindi words in Devanagari script.",
    }.get(caller_language, "Reply in natural Hinglish, with Hindi words in Devanagari script.")
    return (
        "You are Priya, Apollo Hospitals Navi Mumbai receptionist. "
        "Be warm and concise. Ask exactly one question per turn. "
        "Use feminine self-reference, but keep caller-directed language gender-neutral. "
        "Never claim doctor availability without tools. "
        "For booking collect: name, concern, doctor/specialty, date, then check slots and confirm before booking. "
        "Escalate insurance/billing/reports/prescription/medical advice and emergencies to human; mention 108 for emergencies. "
        f"{lang_line}"
    )


def _pick_browser_model(user_text: str) -> str | None:
    """Route short/simple turns to a faster model when configured."""
    if not _BROWSER_FAST_MODEL:
        return None
    txt = (user_text or "").strip()
    words = len(txt.split())
    if words <= 8:
        return _BROWSER_FAST_MODEL
    if any(k in txt.lower() for k in ("hello", "hi", "हेलो", "नमस्ते", "appointment", "अपॉइंटमेंट")):
        return _BROWSER_FAST_MODEL
    return None


def _quick_browser_reply(user_text: str, caller_language: str) -> str | None:
    """Deterministic fast replies for common opening intents."""
    txt = (user_text or "").strip()
    low = txt.lower()
    greet_hits = ("hello", "hi", "हेलो", "नमस्ते")

    if len(txt.split()) <= 3 and any(k in low for k in greet_hits):
        if caller_language == "english":
            return "Hello, please tell me which doctor or specialty you want to consult."
        if caller_language == "hindi":
            return "जी, बताइए। आपको किस डॉक्टर या किस विभाग में अपॉइंटमेंट चाहिए?"
        return "जी, बताइए। आपको किस डॉक्टर या किस विभाग में दिखाना है?"

    if any(k in low for k in ("general", "physician", "internal", "जनरल", "फिजिशियन", "general physician")):
        names = _cached_browser_doctor_names("internal")
        if caller_language == "english":
            return f"For general physician consultation, we have {names}. Would you like me to check slots?"
        return f"जनरल फिजिशियन के लिए हमारे पास {names} हैं। क्या मैं उपलब्ध स्लॉट चेक करूँ?"

    if any(k in low for k in ("sunday", "रविवार")) and any(k in low for k in ("open", "closed", "बंद", "खुला")):
        if caller_language == "english":
            return "Our OPD is closed on Sunday. I can help you check Monday to Saturday availability."
        return "हमारी ओपीडी रविवार को बंद रहती है। मैं सोमवार से शनिवार के उपलब्ध स्लॉट चेक कर सकती हूँ।"

    day_aliases = {
        "monday": "monday", "mon": "monday", "सोमवार": "monday",
        "tuesday": "tuesday", "tue": "tuesday", "मंगलवार": "tuesday",
        "wednesday": "wednesday", "wed": "wednesday", "बुधवार": "wednesday",
        "thursday": "thursday", "thu": "thursday", "गुरुवार": "thursday",
        "friday": "friday", "fri": "friday", "शुक्रवार": "friday",
        "saturday": "saturday", "sat": "saturday", "शनिवार": "saturday",
        "sunday": "sunday", "sun": "sunday", "रविवार": "sunday",
    }
    requested_day = next((v for k, v in day_aliases.items() if k in low), None)
    if requested_day:
        try:
            from config.company_config import COMPANY_CONFIG
            from tools.scheduling import check_doctor_slots
            import re as _re
            doctors = COMPANY_CONFIG.get("doctors", [])
            plain_text = _re.sub(r"[^a-z0-9\s]", " ", low)
            matched_doctor = next(
                (d.get("name", "") for d in doctors if d.get("name", "").lower() in low),
                "",
            )
            if not matched_doctor:
                matched_doctor = next(
                    (
                        d.get("name", "")
                        for d in doctors
                        if all(
                            token in plain_text
                            for token in _re.sub(r"[^a-z0-9\s]", " ", d.get("name", "").lower()).split()
                            if token not in {"dr", "doctor"}
                        )
                    ),
                    "",
                )
            if matched_doctor:
                slot_info = check_doctor_slots(doctor_name=matched_doctor, preferred_date=requested_day)
                if slot_info.get("success") and slot_info.get("available_on_requested_date") is False:
                    next_day = (slot_info.get("next_available_date", "") or "").split(",")[0] or "next available day"
                    if caller_language == "english":
                        return (
                            f"{matched_doctor} is not available on {requested_day.capitalize()}. "
                            f"They are available on {slot_info.get('available_days', '')}. "
                            f"Next available is {next_day}. Would you like me to book that?"
                        )
                    return (
                        f"{matched_doctor} {requested_day.capitalize()} को उपलब्ध नहीं हैं। "
                        f"ये {slot_info.get('available_days', '')} को उपलब्ध हैं। "
                        f"अगली उपलब्ध तारीख {next_day} है। क्या मैं वही बुक कर दूँ?"
                    )
        except Exception:
            pass
    return None


@lru_cache(maxsize=16)
def _cached_browser_doctor_names(key: str) -> str:
    from config.company_config import COMPANY_CONFIG
    doctors = COMPANY_CONFIG.get("doctors", [])
    matched = [d.get("name", "") for d in doctors if key in d.get("specialty", "").lower()]
    return ", ".join(n for n in matched[:3] if n) or "Dr. S V Kulkarni"


def _is_weak_english_signal(text: str) -> bool:
    low = (text or "").strip().lower()
    return low in {"hello", "hi", "hey", "hello priya", "hi priya"}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "active_calls": len(_active_pipelines)}


# ── Browser chat UI ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def browser_ui():
    return HTMLResponse(_BROWSER_UI_HTML)


@app.post("/chat")
async def chat(request: Request):
    """
    Browser chat endpoint.
    Accepts JSON: {"text": "...", "session_id": "..."}
    Returns: {"text": "...", "audio_b64": "<base64 mulaw>"}
    Handles tool calls (check_doctor_slots, list_doctors, book_appointment) inline.
    """
    from services.tts import TTSService
    from tools.definitions import get_tool_schemas
    import base64, json as _json

    body = await request.json()
    user_text = body.get("text", "").strip()
    include_audio = bool(body.get("include_audio", True))
    session_id = body.get("session_id") or str(uuid.uuid4())

    if not user_text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    if session_id not in _browser_sessions:
        _browser_sessions[session_id] = []

    messages = _browser_sessions[session_id]

    # Detect language from accumulated conversation; update session meta
    from pipeline.voice_pipeline import _detect_language
    session_meta = _browser_sessions.get(f"{session_id}:meta", {})
    detected_lang = _detect_language(user_text)
    # Accumulate: once we know caller is english-only, keep it
    prev_lang = session_meta.get("caller_language", "hinglish")
    if detected_lang == "english" and _is_weak_english_signal(user_text):
        detected_lang = prev_lang
    if detected_lang == "english" and prev_lang == "hinglish":
        session_meta["caller_language"] = "english"
    elif detected_lang == "hindi" and prev_lang == "hinglish":
        session_meta["caller_language"] = "hindi"
    caller_language = session_meta.get("caller_language", "hinglish")
    _browser_sessions[f"{session_id}:meta"] = session_meta

    system = _build_browser_system_prompt(caller_language)
    selected_model = _pick_browser_model(user_text)
    messages.append({"role": "user", "content": user_text})

    from pipeline.guardrails import (
        run_guardrails, GuardrailContext, extract_caller_name_from_text,
    )

    tool_schemas = get_tool_schemas()

    caller_name = session_meta.get("caller_name")
    if not caller_name:
        detected = extract_caller_name_from_text(user_text)
        if detected:
            caller_name = detected
            session_meta["caller_name"] = caller_name
            _browser_sessions[f"{session_id}:meta"] = session_meta

    quick = _quick_browser_reply(user_text, caller_language)
    if quick:
        response_text, tool_was_called = quick, False
    else:
        response_text, tool_was_called = await _run_llm_with_tools(
            system,
            messages,
            tool_schemas,
            caller_phone="browser-test",
            model=selected_model,
        )
    response_text = _clean_assistant_text(response_text)

    gr_ctx = GuardrailContext(
        caller_name=caller_name,
        tool_was_called=tool_was_called,
        session_messages=messages,
    )
    gr = run_guardrails(response_text, gr_ctx)
    if gr.blocked:
        logger.warning("Guardrail blocked response (reason=%s): %r", gr.reason, response_text[:80])
        response_text = gr.fallback

    messages.append({"role": "assistant", "content": response_text})

    audio_b64 = ""
    audio_format = ""
    if include_audio:
        # Browser path uses PCM16 for much clearer playback than telephony μ-law.
        audio_format = "pcm_16000"
        tts = TTSService(output_format=audio_format)
        audio_chunks = []
        async for chunk in tts.synthesize(response_text):
            audio_chunks.append(chunk)
        audio_b64 = base64.b64encode(b"".join(audio_chunks)).decode()

    return JSONResponse({
        "text": response_text,
        "audio_b64": audio_b64,
        "audio_format": audio_format,
        "session_id": session_id,
    })


@app.post("/tts")
async def browser_tts(request: Request):
    """Generate browser audio for a text reply (decoupled from /chat latency)."""
    from services.tts import TTSService
    import base64

    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    audio_format = "pcm_16000"
    tts = TTSService(output_format=audio_format)
    audio_chunks: list[bytes] = []
    async for chunk in tts.synthesize(text):
        audio_chunks.append(chunk)

    return JSONResponse({
        "audio_b64": base64.b64encode(b"".join(audio_chunks)).decode(),
        "audio_format": audio_format,
    })


async def _run_llm_with_tools(
    system: str,
    messages: list[dict],
    tool_schemas: list[dict],
    caller_phone: str,
    model: str | None = None,
    depth: int = 0,
    _tool_called: bool = False,
) -> tuple[str, bool]:
    """Call LLM, execute any tool calls, feed results back, return (final_text, tool_was_called)."""
    import json as _json
    from services.llm import stream_response

    if depth > 3:
        return "Could you please repeat that once?", _tool_called

    full_messages = [{"role": "system", "content": system}] + messages[-_BROWSER_MAX_CONTEXT_MESSAGES:]

    text_parts: list[str] = []
    tool_calls: list[dict] = []

    stream_kwargs = {"tools": tool_schemas}
    if model:
        stream_kwargs["model"] = model
    async for item in stream_response(full_messages, **stream_kwargs):
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "tool_call":
            tool_calls.append(item)

    logger.debug("_run_llm depth=%d text_parts=%r tool_calls=%r", depth, text_parts, [tc.get("name") for tc in tool_calls])

    if not tool_calls:
        return (" ".join(text_parts) if text_parts else "एक पल, मैं check करती हूँ।"), _tool_called

    messages.append({
        "role": "assistant",
        "content": " ".join(text_parts) if text_parts else None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": _json.dumps(tc["arguments"])},
            }
            for tc in tool_calls
        ],
    })

    for tc in tool_calls:
        result = _execute_tool(tc["name"], tc["arguments"], caller_phone)
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": tc["name"],
            "content": _json.dumps(result),
        })
        logger.info("Browser tool %s → %s", tc["name"], result)

    return await _run_llm_with_tools(
        system,
        messages,
        tool_schemas,
        caller_phone,
        model=model,
        depth=depth + 1,
        _tool_called=True,
    )


def _execute_tool(name: str, args: dict, caller_phone: str) -> dict:
    """Execute a tool synchronously and return its result dict."""
    try:
        if name == "check_doctor_slots":
            from tools.scheduling import check_doctor_slots
            return check_doctor_slots(
                doctor_name=args.get("doctor_name", ""),
                preferred_date=args.get("preferred_date"),
            )
        elif name == "list_doctors":
            from tools.scheduling import list_doctors
            return list_doctors(specialty=args.get("specialty"))
        elif name == "book_appointment":
            from tools.scheduling import book_appointment
            return book_appointment(
                caller_phone=caller_phone,
                patient_name=args.get("patient_name", ""),
                concern=args.get("concern", ""),
                doctor_name=args.get("doctor_name"),
                specialty=args.get("specialty"),
                preferred_date=args.get("preferred_date"),
                preferred_time=args.get("preferred_time"),
            )
        elif name == "escalate_to_human":
            return {"success": True, "message": "Escalation noted — in browser test mode, no actual transfer."}
        else:
            return {"success": False, "reason": f"Unknown tool: {name}"}
    except Exception as exc:
        logger.error("Tool %s error: %s", name, exc)
        return {"success": False, "reason": str(exc)}


@app.delete("/chat/{session_id}")
async def reset_session(session_id: str):
    _browser_sessions.pop(session_id, None)
    return {"status": "reset"}


_BROWSER_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Apollo Hospitals — Voice Bot Test</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; height: 100vh; display: flex;
         flex-direction: column; }
  header { background: #1a1d27; padding: 16px 24px; border-bottom: 1px solid #2a2d3a;
           display: flex; align-items: center; gap: 12px; }
  .logo { width: 36px; height: 36px; background: #3b82f6; border-radius: 50%;
          display: flex; align-items: center; justify-content: center; font-size: 18px; }
  h1 { font-size: 17px; font-weight: 600; }
  .subtitle { font-size: 12px; color: #6b7280; margin-top: 2px; }
  .status-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%;
                margin-left: auto; box-shadow: 0 0 6px #22c55e; }

  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex;
          flex-direction: column; gap: 12px; }
  .bubble { max-width: 75%; padding: 12px 16px; border-radius: 16px;
            line-height: 1.5; font-size: 14px; }
  .user { background: #3b82f6; color: white; border-bottom-right-radius: 4px;
          align-self: flex-end; }
  .bot  { background: #1e2130; border: 1px solid #2a2d3a;
          border-bottom-left-radius: 4px; align-self: flex-start; }
  .bot.thinking { color: #6b7280; font-style: italic; }
  .bot .speaker { font-size: 11px; color: #3b82f6; margin-bottom: 4px;
                  font-weight: 600; letter-spacing: 0.5px; }

  footer { background: #1a1d27; border-top: 1px solid #2a2d3a;
           padding: 16px 20px; display: flex; gap: 10px; align-items: center; }
  #input { flex: 1; background: #0f1117; border: 1px solid #2a2d3a; border-radius: 24px;
           padding: 12px 18px; color: #e0e0e0; font-size: 14px; outline: none; }
  #input:focus { border-color: #3b82f6; }
  #input::placeholder { color: #4b5563; }
  button { border: none; cursor: pointer; border-radius: 50%;
           width: 44px; height: 44px; display: flex; align-items: center;
           justify-content: center; font-size: 18px; transition: all .2s; }
  #sendBtn { background: #3b82f6; color: white; }
  #sendBtn:hover { background: #2563eb; }
  #sendBtn:disabled { background: #374151; cursor: not-allowed; }
  #micBtn  { background: #1e2130; border: 1px solid #2a2d3a; color: #e0e0e0; }
  #micBtn.recording { background: #ef4444; border-color: #ef4444;
                      animation: pulse 1s infinite; }
  #micBtn:hover:not(.recording) { background: #2a2d3a; }
  @keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)}
                     50%{box-shadow:0 0 0 8px rgba(239,68,68,0)} }
  #resetBtn { background: transparent; border: 1px solid #374151; color: #6b7280;
              border-radius: 8px; width: auto; padding: 0 12px; font-size: 12px; height: 36px; }
  #resetBtn:hover { border-color: #6b7280; color: #e0e0e0; }
  .lang-badge { font-size: 10px; padding: 2px 6px; border-radius: 4px;
                background: #1e2130; color: #6b7280; margin-left: 8px; }
  #hint { font-size: 11px; color: #4b5563; text-align: center; padding: 6px;
          background: #0f1117; }
</style>
</head>
<body>

<header>
  <div class="logo">🏥</div>
  <div>
    <h1>Apollo Hospitals — Priya (AI Receptionist)</h1>
    <div class="subtitle">Hindi / English / Hinglish • Voice + Text test</div>
  </div>
  <div class="status-dot" title="Server online"></div>
</header>

<div id="hint">
  💡 Use Chrome/Edge for voice input — click 🎤 and speak in Hindi, English or Hinglish
</div>

<div id="chat">
  <div class="bubble bot">
    <div class="speaker">PRIYA</div>
    नमस्ते! Apollo Hospitals Navi Mumbai में आपका स्वागत है। मैं Priya बोल रही हूँ। आज मैं आपकी कैसे मदद कर सकती हूँ?
  </div>
</div>

<footer>
  <button id="micBtn" title="Hold to speak">🎤</button>
  <input id="input" placeholder="Type in Hindi, English or Hinglish…" autocomplete="off"/>
  <button id="sendBtn" title="Send">➤</button>
  <button id="resetBtn">↺ New call</button>
</footer>

<script>
const SESSION_KEY = 'vb_session_' + Math.random().toString(36).slice(2);
let sessionId = null;
let isProcessing = false;

const chat    = document.getElementById('chat');
const input   = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const micBtn  = document.getElementById('micBtn');
const resetBtn= document.getElementById('resetBtn');

// ── helpers ───────────────────────────────────────────────────────────────────
function addBubble(role, text) {
  const d = document.createElement('div');
  d.className = 'bubble ' + role;
  if (role === 'bot') {
    d.innerHTML = '<div class="speaker">PRIYA</div>' + escHtml(text);
  } else {
    d.textContent = text;
  }
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
          .replace(/\\n/g,'<br>');
}

function setLoading(on) {
  isProcessing = on;
  sendBtn.disabled = on;
  input.disabled   = on;
  micBtn.disabled  = on;
}

// ── play μ-law audio from base64 ─────────────────────────────────────────────
async function playMulaw(b64) {
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const pcm = new Int16Array(raw.length);
  for (let i = 0; i < raw.length; i++) pcm[i] = mulawDecode(raw[i]);

  const ctx = new AudioContext({ sampleRate: 8000 });
  const buf = ctx.createBuffer(1, pcm.length, 8000);
  const ch  = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768.0;

  return new Promise(resolve => {
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.onended = () => { ctx.close(); resolve(); };
    src.start();
  });
}

// ── play PCM16 (little-endian) at 16kHz ─────────────────────────────────────
async function playPcm16(b64, sampleRate = 16000) {
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const pcm = new Int16Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 2));
  const ctx = new AudioContext({ sampleRate });
  const buf = ctx.createBuffer(1, pcm.length, sampleRate);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768.0;

  return new Promise(resolve => {
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.onended = () => { ctx.close(); resolve(); };
    src.start();
  });
}

// μ-law decoder (ITU-T G.711)
function mulawDecode(u) {
  u = ~u & 0xFF;
  const sign = u & 0x80 ? -1 : 1;
  const exp  = (u >> 4) & 0x07;
  const mant = u & 0x0F;
  return sign * ((mant << 1 | 1) << (exp + 2)) - sign * 33;
}

// ── send message ──────────────────────────────────────────────────────────────
async function send(text) {
  if (!text.trim() || isProcessing) return;
  setLoading(true);
  addBubble('user', text);
  input.value = '';

  const thinking = addBubble('bot thinking', 'Priya is thinking…');

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        session_id: sessionId,
        include_audio: false
      }),
    });
    const data = await res.json();
    sessionId = data.session_id;

    thinking.remove();
    addBubble('bot', data.text);
    // Unblock typing immediately; synth/play audio in background.
    setLoading(false);
    input.focus();
    (async () => {
      try {
        const ttsRes = await fetch('/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: data.text }),
        });
        const ttsData = await ttsRes.json();
        if (ttsData.audio_b64) {
          if (ttsData.audio_format === 'pcm_16000') {
            await playPcm16(ttsData.audio_b64, 16000);
          } else {
            await playMulaw(ttsData.audio_b64);
          }
        }
      } catch (_) {
        // Keep text UX smooth if TTS fails.
      }
    })();

  } catch (e) {
    thinking.textContent = '⚠ Error: ' + e.message;
    thinking.classList.remove('thinking');
  } finally {
    if (isProcessing) {
      setLoading(false);
      input.focus();
    }
  }
}

sendBtn.addEventListener('click', () => send(input.value));
input.addEventListener('keydown', e => { if (e.key === 'Enter') send(input.value); });

resetBtn.addEventListener('click', async () => {
  if (sessionId) await fetch('/chat/' + sessionId, { method: 'DELETE' });
  sessionId = null;
  chat.innerHTML = '';
  addBubble('bot', 'नमस्ते! Apollo Hospitals Navi Mumbai में आपका स्वागत है। मैं Priya बोल रही हूँ। आज मैं आपकी कैसे मदद कर सकती हूँ?');
});

// ── Web Speech API (mic button) ───────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRecognition) {
  micBtn.title = 'Speech not supported in this browser — use Chrome/Edge';
  micBtn.style.opacity = '0.4';
  micBtn.disabled = true;
} else {
  const rec = new SpeechRecognition();
  rec.continuous = false;
  rec.interimResults = true;
  rec.lang = 'hi-IN';  // Hindi — also recognises English words naturally

  rec.onresult = e => {
    const t = Array.from(e.results).map(r => r[0].transcript).join('');
    input.value = t;
    if (e.results[e.results.length - 1].isFinal) send(t);
  };

  rec.onerror = e => {
    micBtn.classList.remove('recording');
    if (e.error !== 'no-speech') console.warn('STT error', e.error);
  };

  rec.onend = () => micBtn.classList.remove('recording');

  micBtn.addEventListener('click', () => {
    if (isProcessing) return;
    if (micBtn.classList.contains('recording')) {
      rec.stop();
    } else {
      micBtn.classList.add('recording');
      input.value = '';
      rec.start();
    }
  });
}
</script>
</body>
</html>
"""


# ── Transfer TwiML endpoint ───────────────────────────────────────────────────

@app.post("/transfer-twiml")
@app.get("/transfer-twiml")
async def transfer_twiml(request: Request):
    """
    Called by Twilio when redirecting a call for human escalation.
    Returns TwiML that dials the escalation phone number.
    """
    params = dict(request.query_params)
    destination = params.get("to", "")
    if not destination:
        from config.company_config import COMPANY_CONFIG
        destination = COMPANY_CONFIG.get("escalation_phone", "")

    logger.info("Transfer TwiML → %s", destination)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="woman">Please hold while we connect you to our staff.</Say>
  <Dial timeout="30" action="/transfer-fallback">
    <Number>{destination}</Number>
  </Dial>
</Response>"""
    return HTMLResponse(content=twiml, media_type="text/xml")


@app.post("/transfer-fallback")
async def transfer_fallback(request: Request):
    """Called by Twilio if the transfer dial times out or fails."""
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="woman">We're sorry, our staff is unavailable right now. Please call back during business hours. Goodbye.</Say>
  <Hangup/>
</Response>"""
    return HTMLResponse(content=twiml, media_type="text/xml")


# ── Twilio webhook ────────────────────────────────────────────────────────────

@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(request: Request):
    """
    Twilio calls this when a call arrives on the configured phone number.
    We respond with TwiML that connects Twilio to our WebSocket stream.
    """
    call_sid = str(uuid.uuid4())
    caller = "unknown"
    try:
        if request.method == "POST":
            form = await request.form()
            call_sid = form.get("CallSid", call_sid)
            caller = form.get("From", caller)
        else:
            q = request.query_params
            call_sid = q.get("CallSid", call_sid)
            caller = q.get("From", caller)
    except Exception as exc:
        logger.warning("Twilio request parse fallback: %s", exc)
    logger.info("Incoming Twilio call: call_sid=%s from=%s", call_sid, caller)

    ws_url = _ws_url(call_sid, caller)
    status_url = f"{PUBLIC_URL.rstrip('/')}/twilio/stream-status" if PUBLIC_URL else "/twilio/stream-status"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" statusCallback="{status_url}" statusCallbackMethod="POST">
      <Parameter name="caller_phone" value="{caller}"/>
    </Stream>
  </Connect>
</Response>"""
    return HTMLResponse(content=twiml, media_type="text/xml")


@app.post("/twilio/stream-status")
async def twilio_stream_status(request: Request):
    """Receive Twilio Media Stream status callbacks for troubleshooting."""
    try:
        form = await request.form()
        logger.info(
            "Twilio stream status: call_sid=%s stream_sid=%s status=%s message=%s",
            form.get("CallSid", ""),
            form.get("StreamSid", ""),
            form.get("StreamEvent", ""),
            form.get("StreamError", ""),
        )
    except Exception as exc:
        logger.warning("Twilio stream-status parse error: %s", exc)
    return JSONResponse({"ok": True})


# ── Telnyx webhook ────────────────────────────────────────────────────────────

@app.post("/telnyx/voice")
async def telnyx_voice(request: Request):
    """
    Telnyx calls this for call-control events.
    We respond to 'call.initiated' with a stream action.
    """
    body = await request.json()
    event_type = body.get("data", {}).get("event_type", "")
    payload = body.get("data", {}).get("payload", {})

    if event_type == "call.initiated":
        call_id = payload.get("call_control_id", str(uuid.uuid4()))
        caller = payload.get("from", "unknown")
        logger.info("Incoming Telnyx call: call_id=%s from=%s", call_id, caller)
        ws_url = _ws_url(call_id)
        return JSONResponse({
            "command": "streaming.start",
            "params": {
                "stream_url": ws_url,
                "stream_track": "both_tracks",
            },
        })

    return JSONResponse({"status": "ignored"})


# ── WebSocket stream handler ──────────────────────────────────────────────────

@app.websocket("/ws/stream/{call_id}")
async def websocket_stream(websocket: WebSocket, call_id: str, caller: str = "unknown"):
    """
    Carrier WebSocket — receives μ-law audio from Twilio/Telnyx and sends
    TTS audio back in the same format.
    """
    await websocket.accept()
    logger.info("WebSocket connected: call_id=%s", call_id)

    caller_phone = caller
    pipeline: Optional[VoicePipeline] = None
    telephony: Optional[TelephonySession] = None

    async def on_audio(chunk: bytes) -> None:
        if pipeline:
            await pipeline.on_audio(chunk)

    telephony = TelephonySession(
        websocket=websocket,
        on_audio=on_audio,
        provider=TELEPHONY_PROVIDER,
    )

    try:
        # Create pipeline immediately — it will wait for stream_ready before greeting
        pipeline = VoicePipeline(
            call_id=call_id,
            caller_phone=caller_phone,
            telephony_session=telephony,
        )
        _active_pipelines[call_id] = pipeline

        # Run pipeline.start() and telephony.receive_loop() concurrently so the
        # "start" frame from Twilio (which sets stream_ready) is read while the
        # pipeline is waiting for it.
        await asyncio.gather(
            pipeline.start(),
            telephony.receive_loop(),
        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: call_id=%s", call_id)
    except Exception as exc:
        logger.error("WebSocket error on call_id=%s: %s", call_id, exc, exc_info=True)
    finally:
        if pipeline:
            await pipeline.shutdown()
            _active_pipelines.pop(call_id, None)
        logger.info("Call ended: call_id=%s", call_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ws_url(call_id: str, caller_phone: str = "") -> str:
    """Build the WebSocket URL for the carrier to connect to."""
    base = PUBLIC_URL.rstrip("/") if PUBLIC_URL else f"http://localhost:{PORT}"
    # Twilio requires wss:// for production; ngrok provides https → wss automatically
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{ws_base}/ws/stream/{call_id}"
    if caller_phone:
        from urllib.parse import quote
        url += f"?caller={quote(caller_phone)}"
    return url


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )
