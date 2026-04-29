"""
Base configuration — reads environment variables and sets global defaults.
All secrets come from environment variables; never hardcode credentials here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Telephony ──────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

TELNYX_API_KEY = os.getenv("TELNYX_API_KEY", "")
TELNYX_APP_ID = os.getenv("TELNYX_APP_ID", "")  # TeXML app / call-control app

# ── Voice Pipeline ─────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# STT options (Deepgram Nova-3)
STT_MODEL = os.getenv("STT_MODEL", "nova-3")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "hi")      # Devanagari Hindi (Nova-3)
STT_SMART_FORMAT = True
STT_INTERIM_RESULTS = True
STT_ENDPOINTING_MS = int(os.getenv("STT_ENDPOINTING_MS", "700"))
STT_CONFIDENCE_THRESHOLD = float(os.getenv("STT_CONFIDENCE_THRESHOLD", "0.88"))
KB_RELOAD_INTERVAL_SECONDS = int(os.getenv("KB_RELOAD_INTERVAL_SECONDS", "300"))  # 5 min

# TTS options
# "sarvam" (Indian languages, primary) | "elevenlabs" (fallback) | "azure" (free fallback) | "deepgram" (prototype)
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "sarvam")
TTS_VOICE_ID = os.getenv(
    "TTS_VOICE_ID",
    "aura-asteria-en",  # Deepgram Aura fallback (not used when TTS_PROVIDER=elevenlabs)
)
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "8000"))  # Twilio μ-law 8kHz
TTS_ENCODING = os.getenv("TTS_ENCODING", "mulaw")

# ElevenLabs (Indian English voice — primary TTS for clinic agent)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
# Alekhya — professional, warm, soothing Indian English female
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "m28sDRnudtExG3WLAufB")
# Mahi — voice-bot optimised Hindi female (optional, for Hindi callers)
ELEVENLABS_VOICE_ID_HINDI = os.getenv("ELEVENLABS_VOICE_ID_HINDI", "OwA6IqdLakQOd19pSLOn")
# Pronunciation Dictionary — created once at startup via services/pronunciation_dict.py
# Leave blank to auto-create from clinic_config tts_pronunciation_entries on first run.
ELEVENLABS_PRONUNCIATION_DICT_ID = os.getenv("ELEVENLABS_PRONUNCIATION_DICT_ID", "")
ELEVENLABS_PRONUNCIATION_DICT_VERSION_ID = os.getenv("ELEVENLABS_PRONUNCIATION_DICT_VERSION_ID", "")

# Azure Cognitive Services TTS (Indian English — free 0.5M chars/month)
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY", "")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION", "eastus")
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "en-IN-NeerjaNeural")

# Sarvam Bulbul TTS (Indian languages — primary)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v1")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "meera")

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Groq — direct API (lowest latency)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Cerebras — same speed as Groq, higher free limits
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# LLM_PROVIDER: "groq" (default, lowest latency) | "openrouter" (fallback)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "200"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
LLM_STREAM = True

# ── Memory / Session ──────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(2 * 60 * 60)))  # 2 h
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"

# ── CRM ───────────────────────────────────────────────────────────────────────
HUBSPOT_ACCESS_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN", "")

# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
PUBLIC_URL = os.getenv("PUBLIC_URL", "")  # e.g. https://xxxx.ngrok.io

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
