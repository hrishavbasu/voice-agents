"""
LLM service — Google Gemini (OpenAI-compatible) with streaming.

Primary: Google Gemini 2.0 Flash (gemini-2.0-flash)
Fallback chain: Cerebras → OpenRouter free models.

To switch model, set LLM_MODEL in .env, e.g.:
    LLM_MODEL=gemini-1.5-pro

Usage:
    async for sentence in stream_response(messages, tools):
        await tts.speak(sentence)
"""

import asyncio
import json
import logging
import re
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI, RateLimitError, NotFoundError

from config.base_config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GOOGLE_API_KEY,
    GOOGLE_BASE_URL,
    CEREBRAS_API_KEY,
    CEREBRAS_BASE_URL,
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

_KNOWN_SAMPLE_GOOGLE_KEYS = {
    "AIzaSyAYizSqADqjEFrAzuU4JFR4pkT3o5noY1o",
}


def _is_usable_google_key(key: str) -> bool:
    """Reject empty or known sample keys."""
    return bool(key) and key not in _KNOWN_SAMPLE_GOOGLE_KEYS


# Primary: Google Gemini. Fallback chain: Cerebras → OpenRouter free.
if LLM_PROVIDER == "google" and _is_usable_google_key(GOOGLE_API_KEY):
    # Disable SDK auto-retries so we can fail over quickly on 429.
    _client = AsyncOpenAI(
        api_key=GOOGLE_API_KEY,
        base_url=GOOGLE_BASE_URL,
        max_retries=0,
    )
    logger.info("LLM provider: Google Gemini model=%s", LLM_MODEL)
else:
    if LLM_PROVIDER == "google":
        logger.error(
            "Google LLM provider requested but GOOGLE_API_KEY is missing or sample key. "
            "Set your paid key in .env to avoid 429 bursts."
        )
    _client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        max_retries=0,
    )
    logger.info("LLM provider: OpenRouter model=%s", LLM_MODEL)

_cerebras_client = AsyncOpenAI(
    api_key=CEREBRAS_API_KEY,
    base_url=CEREBRAS_BASE_URL,
    max_retries=0,
)
_openrouter_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    max_retries=0,
)

# Sentence boundary — two cases:
#   1. Latin .!? with lookbehinds to skip abbreviations (Dr., Mr., etc.)
#   2. Devanagari danda । and double-danda ॥ — always sentence boundaries,
#      no abbreviation ambiguity; allow zero or more trailing spaces.
_SENTENCE_RE = re.compile(
    r"(?:(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Sr\.)(?<!Jr\.)(?<!Mrs\.)(?<!Prof\.)(?<!डॉ\.)(?<=[.!?])\s+|(?<=[।॥])\s*)"
)


async def stream_response(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = LLM_MODEL,
) -> AsyncIterator[str | dict]:
    """
    Yield complete sentences (str) as the LLM streams them, so TTS can start
    speaking the first sentence while the LLM is still generating the rest.

    If the LLM emits a tool_call, yields a dict:
        {"type": "tool_call", "name": str, "arguments": dict}
    instead of a sentence.
    """
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "stream": True,
    }
    if LLM_PROVIDER not in ("google",):
        kwargs["extra_headers"] = {
            "X-Title": "Customer Support AI",
            "HTTP-Referer": "https://customer-support-mvp.local",
        }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    buffer = ""
    tool_calls_acc: dict[int, dict] = {}  # index → accumulated tool call

    try:
        try:
            stream = await _client.chat.completions.create(**kwargs)
        except RateLimitError:
            # Google quota exhausted — try Cerebras first (same speed), then OpenRouter
            stream = None

            # 1. Cerebras — ~600ms, free, higher limits
            if CEREBRAS_API_KEY:
                try:
                    cerebras_kwargs = {**kwargs, "model": "qwen-3-235b-a22b-instruct-2507"}
                    logger.warning("Google 429 — trying Cerebras")
                    stream = await _cerebras_client.chat.completions.create(**cerebras_kwargs)
                except (RateLimitError, NotFoundError):
                    logger.warning("Cerebras unavailable, trying OpenRouter")

            # 2. OpenRouter free models — slower but unlimited
            if stream is None:
                _OR_FALLBACKS = [
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "openai/gpt-oss-120b:free",
                    "nousresearch/hermes-3-llama-3.1-405b:free",
                ]
                or_kwargs = {**kwargs, "extra_headers": {
                    "X-Title": "Customer Support AI",
                    "HTTP-Referer": "https://customer-support-mvp.local",
                }}
                for fb_model in _OR_FALLBACKS:
                    try:
                        or_kwargs["model"] = fb_model
                        logger.warning("Trying OpenRouter fallback: %s", fb_model)
                        stream = await _openrouter_client.chat.completions.create(**or_kwargs)
                        break
                    except RateLimitError:
                        logger.warning("%s rate-limited, trying next", fb_model)

            if stream is None:
                raise RateLimitError("All LLM providers exhausted", response=None, body=None)

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # ── Accumulate tool calls ──────────────────────────────────────
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments_raw": "",
                        }
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments_raw"] += (
                                tc.function.arguments
                            )
                continue

            # ── Accumulate text ───────────────────────────────────────────
            content = delta.content or ""
            buffer += content

            # Yield complete sentences eagerly
            parts = _SENTENCE_RE.split(buffer)
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if not sentence:
                    continue
                # Fewer than 3 words ending in a Latin period = likely an
                # abbreviation fragment (e.g. "डॉ.", "एस.वी.") — merge back.
                # Danda-terminated sentences (।) are always genuine regardless of length.
                if len(sentence.split()) < 3 and not sentence.endswith(("।", "॥", "!", "?")):
                    parts[-1] = sentence + " " + parts[-1]
                    continue
                logger.debug("LLM sentence: %s", sentence)
                yield sentence
            buffer = parts[-1]

        # Flush remaining buffer
        if buffer.strip():
            yield buffer.strip()

        # Yield tool calls
        for tc in tool_calls_acc.values():
            try:
                arguments = json.loads(tc["arguments_raw"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            logger.info("LLM tool_call: %s(%s)", tc["name"], arguments)
            yield {
                "type": "tool_call",
                "id": tc["id"] or f"call_{tc['name']}",
                "name": tc["name"],
                "arguments": arguments,
            }

    except Exception as exc:
        logger.error("LLM stream error: %s", exc)
        yield "I'm having a little trouble right now. Could you repeat that?"
