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
import logging
import os
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
    yield
    # Shutdown: close any lingering pipelines
    for pipeline in list(_active_pipelines.values()):
        await pipeline.shutdown()
    logger.info("Customer Support AI stopped")


app = FastAPI(title="Customer Support AI", lifespan=lifespan)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "active_calls": len(_active_pipelines)}


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

@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """
    Twilio calls this when a call arrives on the configured phone number.
    We respond with TwiML that connects Twilio to our WebSocket stream.
    """
    form = await request.form()
    call_sid = form.get("CallSid", str(uuid.uuid4()))
    caller = form.get("From", "unknown")
    logger.info("Incoming Twilio call: call_sid=%s from=%s", call_sid, caller)

    ws_url = _ws_url(call_sid, caller)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="caller_phone" value="{caller}"/>
    </Stream>
  </Connect>
</Response>"""
    return HTMLResponse(content=twiml, media_type="text/xml")


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
