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

from openai import AsyncOpenAI

from config.base_config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# Sentence boundary — split on .  !  ? followed by whitespace
# Lookbehinds prevent splitting after known abbreviations (ASCII + Devanagari).
# Also guarded by word-count check in stream_response: fragments < 3 words are
# merged back (catches initials like "एस.वी." not coverable by fixed lookbehind).
_SENTENCE_RE = re.compile(
    r"(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Sr\.)(?<!Jr\.)(?<!Mrs\.)(?<!Prof\.)(?<!डॉ\.)(?<=[.!?])\s+"
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
        "extra_headers": {
            "X-Title": "Customer Support AI",
            "HTTP-Referer": "https://customer-support-mvp.local",
        },
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    buffer = ""
    tool_calls_acc: dict[int, dict] = {}  # index → accumulated tool call

    try:
        stream = await _client.chat.completions.create(**kwargs)

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
                # Fewer than 3 words = almost certainly an abbreviation fragment
                # (e.g. "डॉ.", "एस.वी.") — merge back rather than yield a micro-TTS call.
                if len(sentence.split()) < 3:
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
