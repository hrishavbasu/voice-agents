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


def _clause_split(sentence: str) -> List[str]:
    """Split a sentence at comma/em-dash/colon if the leading clause is ≥5 words."""
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
                parts = [
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
    system_instruction, contents = _convert_messages(messages)
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

    try:
        async for chunk in client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    function_calls.append({
                        "type": "tool_call",
                        "id": "call_{}".format(fc.name),
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
                            yield sub
                    buffer = parts[-1]

        if buffer.strip():
            for sub in _clause_split(buffer.strip()):
                yield sub

        for fc in function_calls:
            logger.info("Gemini tool_call: %s(%s)", fc["name"], fc["arguments"])
            yield fc

    except Exception as exc:
        logger.error("Gemini LLM stream error: %s", exc)
        yield "I'm having a little trouble right now. Could you repeat that?"
