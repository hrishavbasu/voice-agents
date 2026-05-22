"""
Gemini 2.5 Pro LLM streaming client.

Same stream_response() contract as services/llm.py:
  - yields str (sentences) as the LLM streams text
  - yields dict {"type": "tool_call", ...} for function calls

thinking_budget=0 disables Gemini's chain-of-thought mode — required
for real-time voice (prevents 3-5s silent pauses before each response).
"""
import json
import logging
import re
from typing import AsyncIterator, List, Optional, Union

from google import genai
from google.genai import types

from config.base_config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_THINKING_BUDGET,
)

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None

_SENTENCE_RE = re.compile(
    r"(?:(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Sr\.)(?<!Jr\.)(?<!Mrs\.)(?<!Prof\.)(?<!डॉ\.)(?<=[.!?])\s+|(?<=[।॥])\s*)"
)

# Duplicated from llm.py — importing from there would create a circular import
# (llm.py imports stream_response from llm_gemini.py via the routing dispatcher).
_CLAUSE_RE = re.compile(r',\s+(?=\S)| — |:\s+(?=\S)')


from services.llm_splitting import limit_first_clause

def _clause_split(sentence: str) -> List[str]:
    """Split a sentence at comma/em-dash/colon if the leading clause is ≥5 words."""
    # Hindi/Devanagari reads more naturally as full sentences; splitting at
    # commas sounds robotic and introduces extra TTS turn latency.
    if re.search(r"[\u0900-\u097f]", sentence):
        return [sentence]
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


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured. Add it to .env.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _convert_tools(openai_tools) -> list:
    """Convert OpenAI tool schemas → Gemini FunctionDeclaration list."""
    if not openai_tools:
        return []
    declarations = []
    for tool in openai_tools:
        fn = tool.get("function", {})
        declarations.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _convert_messages(messages: List[dict]) -> tuple:
    """
    Split OpenAI messages into (system_instruction, gemini_contents).

    Role mapping:
      system    → system_instruction string (not a Content object)
      user      → user Content with text Part
      assistant → model Content (text Part or function_call Parts)
      tool      → user Content with function_response Part
    """
    system_instruction = ""
    contents = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "system":
            system_instruction = content
            continue

        if role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=content)])
            )

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts = []
                if content:
                    parts.append(types.Part(text=content))
                parts += [
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"].get("arguments", "{}")),
                        )
                    )
                    for tc in tool_calls
                ]
                contents.append(types.Content(role="model", parts=parts))
            elif content:
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=content)])
                )

        elif role == "tool":
            name = msg.get("name", "")
            try:
                result = json.loads(content) if content else {}
            except json.JSONDecodeError:
                result = {"content": content}
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=name,
                                response=result,
                            )
                        )
                    ],
                )
            )

    return system_instruction, contents


def sanitize_messages_for_gemini(messages: List[dict]) -> List[dict]:
    """
    Normalize OpenAI-style history for Gemini.

    - Keep system messages as-is (handled separately).
    - Merge consecutive assistant text turns (invalid back-to-back model turns).
    - Drop empty assistant rows and orphan tool / tool_calls pairs.
    - When trimming, never leave assistant tool_calls without the following tool row.
    """
    if not messages:
        return []

    systems = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    cleaned: List[dict] = []
    i = 0
    while i < len(rest):
        msg = rest[i]
        role = msg.get("role", "")
        if role == "assistant":
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls")
            if not content and not tool_calls:
                i += 1
                continue
            if tool_calls:
                if i + 1 < len(rest) and rest[i + 1].get("role") == "tool":
                    cleaned.append(dict(msg))
                    cleaned.append(dict(rest[i + 1]))
                    i += 2
                else:
                    i += 1
                continue
            if (
                cleaned
                and cleaned[-1].get("role") == "assistant"
                and not cleaned[-1].get("tool_calls")
            ):
                prev = (cleaned[-1].get("content") or "").strip()
                merged = f"{prev} {content}".strip() if prev else content
                cleaned[-1] = {**cleaned[-1], "content": merged}
                i += 1
                continue
            cleaned.append(dict(msg))
            i += 1
            continue
        if role == "tool":
            i += 1
            continue
        cleaned.append(dict(msg))
        i += 1

    # Drop trailing assistant+tool_calls without tool result (truncated history).
    while cleaned and cleaned[-1].get("role") == "assistant" and cleaned[-1].get("tool_calls"):
        if len(cleaned) >= 2 and cleaned[-2].get("role") == "tool":
            break
        cleaned.pop()

    return systems + cleaned


def trim_messages_for_llm(messages: List[dict], max_non_system: int = 20) -> List[dict]:
    """Return the last N non-system messages, keeping tool call pairs intact."""
    systems = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) <= max_non_system:
        return systems + rest

    trimmed = rest[-max_non_system:]
    # Drop orphaned assistant+tool_calls at start (tool result was trimmed away).
    while trimmed and trimmed[0].get("role") == "assistant" and trimmed[0].get("tool_calls"):
        if len(trimmed) >= 2 and trimmed[1].get("role") == "tool":
            break
        trimmed = trimmed[1:]
    # Drop orphaned tool messages at start (their assistant+tool_calls was trimmed away).
    # Without the preceding assistant message, the Gemini OpenAI-compat endpoint cannot
    # determine the function name → function_response.name becomes empty → 400 error.
    while trimmed and trimmed[0].get("role") == "tool":
        trimmed = trimmed[1:]
    return systems + trimmed


async def stream_response(
    messages: List[dict],
    tools=None,
    model: str = GEMINI_MODEL,
) -> AsyncIterator[Union[str, dict]]:
    """
    Yield sentences (str) and tool calls (dict) from Gemini 2.5 Pro.
    Identical contract to services/llm.py stream_response.
    """
    client = _get_client()
    prepared = sanitize_messages_for_gemini(messages)
    system_instruction, contents = _convert_messages(prepared)
    gemini_tools = _convert_tools(tools or [])

    config = types.GenerateContentConfig(
        system_instruction=system_instruction or None,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=LLM_THINKING_BUDGET),
        tools=gemini_tools or None,
    )

    buffer = ""
    function_calls: List[dict] = []
    _call_idx = 0
    first_yielded = False

    try:
        stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    _call_idx += 1
                    function_calls.append({
                        "type": "tool_call",
                        "id": "call_{}_{}".format(fc.name, _call_idx),
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })
                elif part.text:
                    buffer += part.text
                    parts = _SENTENCE_RE.split(buffer)
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        if len(sentence.split()) < 3 and not sentence.endswith(("।", "॥", "!", "?")):
                            parts[-1] = sentence + " " + parts[-1]
                            continue
                        logger.debug("Gemini sentence: %s", sentence)
                        for sub in _clause_split(sentence):
                            for part in limit_first_clause(sub, is_first=not first_yielded):
                                first_yielded = True
                                yield part
                    buffer = parts[-1]

        if buffer.strip():
            for sub in _clause_split(buffer.strip()):
                for part in limit_first_clause(sub, is_first=not first_yielded):
                    first_yielded = True
                    yield part

        for fc in function_calls:
            logger.info("Gemini tool_call: %s(%s)", fc["name"], fc["arguments"])
            yield fc

    except Exception as exc:
        logger.error("Gemini LLM stream error: %s", exc)
        yield (
            "माफ़ कीजिए, एक पल रुकिए। क्या आप दोबारा बता सकते हैं?"
        )
