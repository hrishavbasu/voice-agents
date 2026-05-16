#!/usr/bin/env python3
"""
Simulate Hindi / English / Hinglish conversation turns (no Twilio).

Validates: language detection, prosody, Gemini LLM, Sarvam TTS, response quality.

Usage:
    .venv/bin/python scripts/simulate_hindi_english.py
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from config.base_config import GEMINI_API_KEY, SARVAM_API_KEY, TTS_PROVIDER, LLM_PROVIDER
from config.user_settings import load_user_settings
from pipeline.voice_pipeline import VoicePipeline, _detect_language
from prompts.system_prompt import build_system_prompt
from services.llm import stream_response
from tools.definitions import get_tool_schemas

# (name, user_utterance, expected_lang for _detect_language)
# Roman-only Hinglish has <5% Devanagari → detected as "english" until caller uses Hindi script.
SCENARIOS = [
    ("roman_hinglish_as_english", "Namaste, mujhe kal cardiologist se appointment chahiye.", "english"),
    ("english_booking", "Hello, I need to book an appointment with a cardiologist tomorrow.", "english"),
    ("hindi_booking", "नमस्ते, मुझे कल डॉक्टर से मिलना है।", "hindi"),
    ("hinglish_mixed_script", "Mujhe कल appointment चाहिए, cardiologist के साथ please.", "hinglish"),
    ("english_slot", "What time slots are available in the morning?", "english"),
    ("hindi_confirm", "हाँ, तीन बजे ठीक है।", "hindi"),
    ("hindi_short", "हाँ जी।", "hindi"),
    ("english_emergency_style", "I have mild chest discomfort, not severe.", "english"),
]

TTS_ONLY = [
    ("tts_hindi", "नमस्ते, Apollo Hospitals में आपका स्वागत है।", "hi-IN"),
    ("tts_english", "Welcome to Apollo Hospitals. How may I help you today?", "en-IN"),
    ("tts_hinglish", "Main aapki appointment book kar deti hoon, ek minute.", "hi-IN"),
]


def _has_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097f]", text))


def _has_markdown_artifacts(text: str) -> bool:
    return bool(re.search(r"\*\*|^[\s]*[-•]", text, re.MULTILINE))


async def _run_llm_turn(user_text: str, caller_language: str) -> tuple[str, float, list[str]]:
    """Return (full_response, latency_secs, issues)."""
    issues: list[str] = []
    system = build_system_prompt(caller_phone="+919999999999", caller_language=caller_language)
    messages = [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "Namaste! Main Priya bol rahi hoon Apollo Hospitals se. Main aapki kya madad kar sakti hoon?"},
        {"role": "user", "content": user_text},
    ]
    t0 = time.perf_counter()
    parts: list[str] = []
    async for item in stream_response(messages, tools=get_tool_schemas()):
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "tool_call":
            parts.append(f"[tool:{item.get('name')}]")
    latency = time.perf_counter() - t0
    full = " ".join(parts)
    if not full.strip():
        issues.append("empty LLM response")
    if _has_markdown_artifacts(full):
        issues.append("markdown in response")
    if caller_language in ("hindi", "hinglish") and not _has_devanagari(full):
        issues.append("expected Devanagari for hindi/hinglish")
    if len(full.split()) > 80:
        issues.append("response too long for voice")
    return full, latency, issues


async def _run_tts_snippet(text: str, lang_code: str) -> tuple[int, float, list[str]]:
    """Synthesize first sentence chunk; return (bytes, latency, issues)."""
    from services.tts_sarvam import sarvam_synthesize_to_bytes

    issues: list[str] = []
    if not text.strip():
        return 0, 0.0, ["no text for TTS"]
    sample = text[:120] if len(text) > 120 else text
    t0 = time.perf_counter()
    try:
        audio = await sarvam_synthesize_to_bytes(sample, lang_code)
    except Exception as exc:
        return 0, time.perf_counter() - t0, [f"TTS error: {exc}"]
    latency = time.perf_counter() - t0
    if len(audio) < 800:
        issues.append("TTS audio too short")
    return len(audio), latency, issues


async def main() -> int:
    settings = load_user_settings()
    print("=" * 60)
    print("Voice Agent — Hindi / English / Hinglish Simulation")
    print("=" * 60)
    print(f"  LLM_PROVIDER={LLM_PROVIDER}  TTS_PROVIDER={TTS_PROVIDER}")
    print(f"  GEMINI_API_KEY={'set' if GEMINI_API_KEY else 'MISSING'}")
    print(f"  SARVAM_API_KEY={'set' if SARVAM_API_KEY else 'MISSING'}")
    print(f"  user_settings keys: {len(settings)}")
    print()

    if not SARVAM_API_KEY:
        print("FAIL: SARVAM_API_KEY required for TTS simulation.")
        return 1

    passed = 0
    failed = 0
    skipped = 0

    # ── Language detection (offline) ─────────────────────────────────────────
    print("── Language detection ──")
    for name, utterance, expected in SCENARIOS:
        detected = _detect_language(utterance)
        ok = detected == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detected!r} (expected {expected!r})")
        if ok:
            passed += 1
        else:
            failed += 1
    print()

    print("── Sarvam TTS (no LLM) ──")
    for name, text, lang_code in TTS_ONLY:
        nbytes, lat, issues = await _run_tts_snippet(text, lang_code)
        ok = not issues
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {nbytes} bytes in {lat:.2f}s" + (f" — {issues}" if issues else ""))
        if ok:
            passed += 1
        else:
            failed += 1
    print()

    if not GEMINI_API_KEY:
        print("── LLM + TTS turns (SKIPPED — set GEMINI_API_KEY in .env) ──\n")
        skipped = len(SCENARIOS)
    else:
        print("── LLM + TTS end-to-end turns ──")
        for name, utterance, expected_lang in SCENARIOS:
            lang_code = "hi-IN" if expected_lang in ("hindi", "hinglish") else "en-IN"
            issues: list[str] = []
            try:
                response, llm_lat, llm_issues = await _run_llm_turn(utterance, expected_lang)
                issues.extend(llm_issues)
                tts_bytes, tts_lat, tts_issues = await _run_tts_snippet(response, lang_code)
                issues.extend(tts_issues)
                ok = not issues
                status = "PASS" if ok else "FAIL"
                print(f"\n  [{status}] {name}")
                print(f"       User: {utterance[:70]}{'…' if len(utterance) > 70 else ''}")
                print(f"       Agent: {response[:100]}{'…' if len(response) > 100 else ''}")
                print(f"       LLM {llm_lat:.2f}s | TTS {tts_lat:.2f}s | audio {tts_bytes} bytes")
                if issues:
                    print(f"       Issues: {', '.join(issues)}")
                if ok:
                    passed += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f"\n  [FAIL] {name}: {exc}")
                failed += 1
        print()

    # ── Prosody smoke ────────────────────────────────────────────────────────
    print("── Prosody normalization ──")
    from pipeline.voice_pipeline import VoicePipeline as VP

    sample = VP._normalize_for_tts("**Dr. Sharma** — fee is ₹1,700")
    prosody_ok = "₹" not in sample and "**" not in sample
    print(f"  [{'PASS' if prosody_ok else 'FAIL'}] markdown/currency stripped: {sample!r}")
    if prosody_ok:
        passed += 1
    else:
        failed += 1
    print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped (no Gemini key)")
    print("=" * 60)
    if failed:
        return 1
    if skipped and not GEMINI_API_KEY:
        print("\nAdd GEMINI_API_KEY to .env and re-run for full LLM+TTS validation.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
