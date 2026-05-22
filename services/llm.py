"""
LLM service — OpenRouter (OpenAI-compatible) with streaming.

Prototype: meta-llama/llama-3.3-70b-instruct:free  (zero cost)
Production: change LLM_MODEL in .env to "groq/llama-3.3-70b" or
            "anthropic/claude-haiku-4-5" — no code change needed.

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
    GROQ_API_KEY,
    GROQ_BASE_URL,
    CEREBRAS_API_KEY,
    CEREBRAS_BASE_URL,
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    GEMINI_MODEL,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)

# Primary: Groq (lowest latency). Fallback chain: Cerebras → OpenRouter free.
if LLM_PROVIDER == "groq" and GROQ_API_KEY:
    _client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    logger.info("LLM provider: Groq model=%s", LLM_MODEL)
else:
    _client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    logger.info("LLM provider: OpenRouter model=%s", LLM_MODEL)

_cerebras_client = AsyncOpenAI(api_key=CEREBRAS_API_KEY, base_url=CEREBRAS_BASE_URL)
_openrouter_client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# Gemini via OpenAI-compatible endpoint — avoids google-genai AFC bug where
# function_call parts are swallowed in streaming mode (google-genai 1.x).
# This uses the same _openai_stream_response path as other providers, which
# correctly accumulates streamed tool call deltas into structured dicts.
_gemini_openai_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
) if GEMINI_API_KEY else None

# Sentence boundary — two cases:
#   1. Latin .!? with lookbehinds to skip abbreviations (Dr., Mr., etc.)
#   2. Devanagari danda । and double-danda ॥ — always sentence boundaries,
#      no abbreviation ambiguity; allow zero or more trailing spaces.
_SENTENCE_RE = re.compile(
    r"(?:(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Sr\.)(?<!Jr\.)(?<!Mrs\.)(?<!Prof\.)(?<!डॉ\.)(?<=[.!?])\s+|(?<=[।॥])\s*)"
)

# Secondary split: break long comma-clauses and em-dashes into shorter TTS chunks.
# Only splits at a comma when the preceding segment is ≥5 words, to avoid splitting
# short phrases like "हाँ, sure" or "Hello, Priya" into separate TTS calls.
_CLAUSE_RE = re.compile(r',\s+(?=\S)| — |:\s+(?=\S)')


def _clause_split(sentence: str) -> list[str]:
    """Further split a sentence at comma/em-dash/colon if the leading clause is ≥5 words."""
    raw_parts = _CLAUSE_RE.split(sentence)
    if len(raw_parts) == 1:
        return [sentence]
    result = []
    current = raw_parts[0]
    for part in raw_parts[1:]:
        if len(current.split()) >= 5:
            result.append(current.strip())
            current = part
        else:
            current = current + ", " + part
    if current.strip():
        result.append(current.strip())
    return [p for p in result if p]


async def stream_response(
    messages: list[dict],
    tools=None,
    model=None,
):
    """Route to Gemini or OpenAI-compatible provider based on LLM_PROVIDER."""
    if LLM_PROVIDER == "gemini":
        # Use Gemini's OpenAI-compatible REST endpoint. The native google-genai SDK
        # (llm_gemini.py) silently drops function_call parts in streaming mode due
        # to Automatic Function Calling (AFC) interception — tool calls would be
        # emitted as plain text and read aloud by TTS. The OpenAI-compat path
        # accumulates streamed tool call deltas correctly.
        async for item in _openai_stream_response(
            messages,
            tools=tools,
            model=model or GEMINI_MODEL,
            _direct_client=_gemini_openai_client,
        ):
            yield item
    else:
        async for item in _openai_stream_response(messages, tools=tools, model=model or LLM_MODEL):
            yield item


async def _openai_stream_response(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = LLM_MODEL,
    _direct_client=None,
):
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
    # Extra headers for OpenRouter attribution — not sent to Groq or Gemini
    if LLM_PROVIDER not in ("groq",) and _direct_client is None:
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
        if _direct_client is not None:
            # Direct client path (e.g. Gemini OpenAI-compat) — no fallback chain
            stream = await _direct_client.chat.completions.create(**kwargs)
        else:
            try:
                stream = await _client.chat.completions.create(**kwargs)
            except RateLimitError:
                # Groq quota exhausted — try Cerebras first (same speed), then OpenRouter
                stream = None

                # 1. Cerebras — ~600ms, free, higher limits than Groq
                if CEREBRAS_API_KEY:
                    try:
                        cerebras_kwargs = {**kwargs, "model": "qwen-3-235b-a22b-instruct-2507"}
                        logger.warning("Groq 429 — trying Cerebras")
                        stream = await _cerebras_client.chat.completions.create(**cerebras_kwargs)
                    except (RateLimitError, NotFoundError):
                        logger.warning("Cerebras unavailable, trying OpenRouter")

                # 2. OpenRouter free models — slower but unlimited
                if stream is None:
                    _OR_FALLBACKS = [
                        "openai/gpt-oss-120b:free",
                        "meta-llama/llama-3.3-70b-instruct:free",
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
                for sub in _clause_split(sentence):
                    yield sub
            buffer = parts[-1]

        # Flush remaining buffer
        if buffer.strip():
            for sub in _clause_split(buffer.strip()):
                yield sub

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
