"""
Shared fixtures and environment setup for the test suite.
All external service calls (Deepgram, OpenAI, HubSpot, Redis) are mocked so
tests run without API keys or network access.
"""

import os
import sys

# Ensure repo root is on sys.path so all package imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub out every env var the config layer reads so modules import cleanly
_ENV_DEFAULTS = {
    "DEEPGRAM_API_KEY": "test-deepgram-key",
    "OPENROUTER_API_KEY": "test-or-key",
    "GROQ_API_KEY": "test-groq-key",
    "CEREBRAS_API_KEY": "test-cerebras-key",
    "ELEVENLABS_API_KEY": "test-el-key",
    "TWILIO_ACCOUNT_SID": "ACtest",
    "TWILIO_AUTH_TOKEN": "test-token",
    "TWILIO_PHONE_NUMBER": "+10000000000",
    "HUBSPOT_ACCESS_TOKEN": "test-hubspot",
    "LLM_PROVIDER": "openrouter",
    "USE_REDIS": "false",
}
for k, v in _ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)
