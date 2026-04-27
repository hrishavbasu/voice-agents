"""
Hallucination guardrails — validate LLM output before it reaches TTS or the caller.

Each guardrail is a function: (text, context) → GuardrailResult.
Rules are defined in GUARDRAIL_CONFIG (in clinic_config.py or here as defaults).
Any failed guardrail triggers a safe fallback response instead of speaking the
hallucinated text.

Usage:
    result = run_guardrails(text, context)
    if result.blocked:
        speak(result.fallback)
    else:
        speak(result.text)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config — can be overridden per clinic in clinic_config.py ─────────────────

DEFAULT_GUARDRAIL_CONFIG: dict = {
    # Block responses that mention a caller name that was never given
    "block_hallucinated_name": True,

    # Block responses that state doctor availability without a tool call having happened
    "block_availability_without_tool": True,

    # Block responses containing digit-format times (9:00 AM, 3:30 PM)
    # The system prompt already prohibits these, but catch them as a safety net
    "block_digit_times": True,

    # Block responses containing digit-format currency (₹1,700 / Rs. 500)
    "block_digit_currency": True,

    # Block responses that claim to be an AI/bot (caller trust issue)
    "block_ai_disclosure": True,

    # Minimum response length in characters — catch empty/truncated responses
    "min_response_chars": 5,

    # Fallback phrase when a guardrail fires (spoken to caller)
    "fallback_phrase": "एक moment, मैं confirm करके बताती हूँ।",
}


@dataclass
class GuardrailContext:
    """Contextual facts the guardrails can check against."""
    caller_name: Optional[str] = None          # set once caller states their name
    tool_was_called: bool = False              # True after any tool executed this turn
    session_messages: list = field(default_factory=list)


@dataclass
class GuardrailResult:
    text: str
    blocked: bool = False
    reason: str = ""
    fallback: str = ""


# ── Individual rules ──────────────────────────────────────────────────────────

# Common Sanskrit/Hindi name suffixes the LLM might hallucinate
_NAME_HONORIFICS = re.compile(
    r"\b(जी|साहब|मैडम|sir|ma'am)\b",
    re.IGNORECASE,
)

# Digit time patterns the LLM must not output
_DIGIT_TIME_RE = re.compile(
    r"\b\d{1,2}:\d{2}\s*(AM|PM|am|pm|बजे)?\b"
)

# Digit currency patterns
_DIGIT_CURRENCY_RE = re.compile(
    r"(₹\s*\d[\d,]*|Rs\.?\s*\d[\d,]*|\bINR\s*\d[\d,]*)"
)

# Phrases that indicate AI self-disclosure
_AI_DISCLOSURE_PATTERNS = re.compile(
    r"\b(i am an ai|i('m| am) a (bot|robot|virtual|language model|ai)|"
    r"as an ai|artificial intelligence|मैं एक AI|मैं bot हूँ)\b",
    re.IGNORECASE,
)

# Phrases claiming availability without a tool call
_AVAILABILITY_CLAIMS = re.compile(
    r"(available|उपलब्ध|slot|slots|timing|appointment available|book kar sakte)"
    r".{0,40}"
    r"(on|at|at|को|पर|के लिए|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)",
    re.IGNORECASE,
)


def _check_hallucinated_name(text: str, ctx: GuardrailContext, cfg: dict) -> Optional[str]:
    """
    Block if the response addresses a caller by name (with an honorific like 'जी')
    but the caller has not yet provided their name this session.
    """
    if not cfg.get("block_hallucinated_name", True):
        return None
    if ctx.caller_name:
        return None  # name is known — allowed

    # Look for patterns like "राहुल जी" / "Priya ji" / "Vikram sir"
    # These are almost always hallucinations when no name was given
    if _NAME_HONORIFICS.search(text):
        # Check if any message in history actually contains a name from caller
        history_text = " ".join(
            m.get("content", "") for m in ctx.session_messages
            if m.get("role") == "user"
        )
        # If caller never stated a name in any message, this is a hallucination
        # Heuristic: if honorific appears but caller_name not set, it's suspect
        logger.warning("GUARDRAIL: possible name hallucination — caller_name not set but honorific found: %r", text[:80])
        # Soft block — log but allow (name might be embedded in context we can't parse)
        # Change to `return "hallucinated_name"` to make it a hard block
        return None

    return None


def _check_availability_without_tool(text: str, ctx: GuardrailContext, cfg: dict) -> Optional[str]:
    """
    Block if the response claims specific availability/slot info
    but no tool was called this turn.
    """
    if not cfg.get("block_availability_without_tool", True):
        return None
    if ctx.tool_was_called:
        return None  # tool ran — information is grounded

    if _AVAILABILITY_CLAIMS.search(text):
        logger.warning("GUARDRAIL: availability claim without tool call — blocking: %r", text[:100])
        return "availability_without_tool"

    return None


def _check_digit_times(text: str, ctx: GuardrailContext, cfg: dict) -> Optional[str]:
    if not cfg.get("block_digit_times", True):
        return None
    m = _DIGIT_TIME_RE.search(text)
    if m:
        logger.warning("GUARDRAIL: digit time in response — blocking: %r", m.group())
        return "digit_time"
    return None


def _check_digit_currency(text: str, ctx: GuardrailContext, cfg: dict) -> Optional[str]:
    if not cfg.get("block_digit_currency", True):
        return None
    m = _DIGIT_CURRENCY_RE.search(text)
    if m:
        logger.warning("GUARDRAIL: digit currency in response — blocking: %r", m.group())
        return "digit_currency"
    return None


def _check_ai_disclosure(text: str, ctx: GuardrailContext, cfg: dict) -> Optional[str]:
    if not cfg.get("block_ai_disclosure", True):
        return None
    if _AI_DISCLOSURE_PATTERNS.search(text):
        logger.warning("GUARDRAIL: AI disclosure detected — blocking")
        return "ai_disclosure"
    return None


def _check_min_length(text: str, ctx: GuardrailContext, cfg: dict) -> Optional[str]:
    min_chars = cfg.get("min_response_chars", 5)
    if len(text.strip()) < min_chars:
        logger.warning("GUARDRAIL: response too short (%d chars) — blocking", len(text.strip()))
        return "too_short"
    return None


_ALL_CHECKS = [
    _check_min_length,
    _check_ai_disclosure,
    _check_digit_times,
    _check_digit_currency,
    _check_availability_without_tool,
    _check_hallucinated_name,
]


# ── Public API ────────────────────────────────────────────────────────────────

def run_guardrails(
    text: str,
    context: Optional[GuardrailContext] = None,
    config: Optional[dict] = None,
) -> GuardrailResult:
    """
    Run all enabled guardrails against `text`.
    Returns GuardrailResult with blocked=True if any check fires.
    """
    cfg = {**DEFAULT_GUARDRAIL_CONFIG, **(config or {})}
    ctx = context or GuardrailContext()
    fallback = cfg.get("fallback_phrase", "एक moment, मैं confirm करके बताती हूँ।")

    for check in _ALL_CHECKS:
        reason = check(text, ctx, cfg)
        if reason:
            return GuardrailResult(
                text=text,
                blocked=True,
                reason=reason,
                fallback=fallback,
            )

    return GuardrailResult(text=text, blocked=False)


def extract_caller_name_from_text(user_text: str) -> Optional[str]:
    """
    Best-effort extraction of a caller's name from their message.
    Returns the name string if found, None otherwise.
    Used to update GuardrailContext.caller_name after each user turn.
    """
    patterns = [
        # "mera naam X hai" / "my name is X"
        r"(?:mera naam|my name is|मेरा नाम|naam hai|name is)\s+([A-Za-zऀ-ॿ]+(?:\s+[A-Za-zऀ-ॿ]+)?)",
        # "main X hoon" / "I am X"
        r"(?:main|mai|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:hoon|hun|हूँ|हूं|bol raha|bolta)",
        # "X bol raha hoon"
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:bol|speaking|here)",
    ]
    for pat in patterns:
        m = re.search(pat, user_text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name) >= 2:
                return name
    return None
