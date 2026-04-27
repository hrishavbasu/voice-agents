"""
Local call simulator — test the full voice pipeline without Twilio.

Modes:
  1. TEXT mode (default) — type messages, see LLM + TTS responses
  2. AUDIO mode (--audio) — speak into mic, hear TTS back through speakers

Usage:
    # Text mode (no extra deps):
    .venv/bin/python test_local_call.py

    # Audio mode (needs pyaudio):
    .venv/bin/pip install pyaudio
    .venv/bin/python test_local_call.py --audio

Scenarios tested automatically in text mode:
  - Hindi greeting & booking flow
  - English appointment enquiry
  - Doctor listing
  - Emergency keyword detection
  - Out-of-hours handling
"""

import argparse
import asyncio
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_local_call")

sys.path.insert(0, ".")

from config.base_config import ELEVENLABS_API_KEY, DEEPGRAM_API_KEY, GOOGLE_API_KEY
from services.llm import stream_response
from services.tts import TTSService
from prompts.system_prompt import build_system_prompt


# ── Colours ───────────────────────────────────────────────────────────────────
RESET = "\033[0m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"


def _banner():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Apollo Hospitals — Local Call Simulator{RESET}")
    print(f"{'='*60}")
    print(f"  GOOGLE_API_KEY : {'✓ set' if GOOGLE_API_KEY else '✗ missing'}")
    print(f"  DEEPGRAM_KEY   : {'✓ set' if DEEPGRAM_API_KEY else '✗ missing'}")
    print(f"  ELEVENLABS_KEY : {'✓ set' if ELEVENLABS_API_KEY else '✗ missing'}")
    print(f"{'='*60}\n")


# ── Text-mode simulator ───────────────────────────────────────────────────────

SCENARIOS = [
    ("Hindi booking", "हाँ, मुझे Dr. Anuj Sathe से appointment chahiye kal ke liye."),
    ("English enquiry", "I need to see a cardiologist. What are the timings?"),
    ("List doctors",    "Which orthopedic doctors do you have?"),
    ("Out of hours",    "Are you open on Sunday?"),
    ("Emergency",       "I have chest pain and I can't breathe."),
    ("Hinglish",        "Mujhe Dr. Kulkarni ka slot check karna hai Monday ko."),
]

CALLER_PHONE = "+919876543210"


async def run_text_mode(scenario_only: bool = False):
    _banner()
    tts = TTSService()
    system = build_system_prompt(
        caller_phone=CALLER_PHONE,
        crm_contact=None,
        caller_language="hinglish",
    )
    messages: list[dict] = []

    greeting = "नमस्ते, Apollo Hospitals Navi Mumbai में आपका स्वागत है। मैं Priya हूँ। How may I assist you today?"
    print(f"{GREEN}[PRIYA]{RESET} {greeting}\n")
    messages.append({"role": "assistant", "content": greeting})

    if scenario_only:
        print(f"{YELLOW}Running {len(SCENARIOS)} preset scenarios...{RESET}\n")
        for label, user_text in SCENARIOS:
            await _single_turn(tts, system, messages, user_text, label)
        print(f"\n{BOLD}{GREEN}All scenarios complete.{RESET}\n")
        return

    # Interactive loop
    print(f"{YELLOW}Type your message (or 'quit' to exit, 'scenario' to run presets):{RESET}\n")
    while True:
        try:
            user_input = input(f"{CYAN}[YOU]{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Ending call. Goodbye!")
            break
        if user_input.lower() == "scenario":
            print(f"\n{YELLOW}Running preset scenarios...{RESET}\n")
            for label, text in SCENARIOS:
                await _single_turn(tts, system, messages, text, label)
            print(f"\n{YELLOW}Back to interactive mode:{RESET}\n")
            continue

        await _single_turn(tts, system, messages, user_input)


async def _single_turn(
    tts: TTSService,
    system: str,
    messages: list[dict],
    user_text: str,
    label: str = "",
):
    if label:
        print(f"\n{YELLOW}[SCENARIO: {label}]{RESET}")
    print(f"{CYAN}[YOU]{RESET} {user_text}")

    messages.append({"role": "user", "content": user_text})
    full_messages = [{"role": "system", "content": system}] + messages[-10:]

    t0 = time.time()
    response_parts = []
    first_token = None

    async for item in stream_response(full_messages):
        if isinstance(item, str):
            if first_token is None:
                first_token = time.time() - t0
            response_parts.append(item)

    ttft = f"{first_token:.2f}s" if first_token else "N/A"
    full_response = " ".join(response_parts)
    print(f"{GREEN}[PRIYA]{RESET} {full_response}")
    print(f"  {YELLOW}↳ TTFT: {ttft} | total: {time.time()-t0:.2f}s{RESET}")

    # TTS test — synthesize and measure
    if full_response and "trouble" not in full_response:
        tts_t0 = time.time()
        chunks = 0
        total_bytes = 0
        async for chunk in tts.synthesize(full_response):
            chunks += 1
            total_bytes += len(chunk)
        dur = total_bytes / 8000
        print(f"  {YELLOW}↳ TTS: {time.time()-tts_t0:.2f}s synthesis | ~{dur:.1f}s audio{RESET}")
    else:
        print(f"  {RED}↳ LLM error — check API key / quota{RESET}")

    messages.append({"role": "assistant", "content": full_response})
    print()


# ── Audio-mode simulator ──────────────────────────────────────────────────────

async def run_audio_mode():
    """Full audio loop: mic → Deepgram STT → LLM → ElevenLabs TTS → speakers."""
    try:
        import pyaudio  # type: ignore
    except ImportError:
        print(f"{RED}pyaudio not installed. Run: .venv/bin/pip install pyaudio{RESET}")
        sys.exit(1)

    _banner()
    print(f"{YELLOW}Audio mode active — speak into your mic. Press Ctrl+C to stop.{RESET}\n")

    from services.stt import DeepgramSTT

    tts = TTSService()
    pa = pyaudio.PyAudio()
    system = build_system_prompt(
        caller_phone=CALLER_PHONE,
        crm_contact=None,
        caller_language="hinglish",
    )
    messages: list[dict] = []

    # Greeting
    greeting = "नमस्ते, Apollo Hospitals में आपका स्वागत है। मैं Priya हूँ। How may I help you?"
    print(f"{GREEN}[PRIYA]{RESET} {greeting}")
    messages.append({"role": "assistant", "content": greeting})

    # Play greeting through speakers (μ-law → PCM conversion)
    await _play_tts(tts, pa, greeting)

    transcript_queue: asyncio.Queue[str] = asyncio.Queue()

    async def on_transcript(text: str, is_final: bool):
        if is_final:
            await transcript_queue.put(text)

    stt = DeepgramSTT(on_transcript=on_transcript)
    await stt.connect()

    # Mic capture task
    mic_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=8000,
        input=True,
        frames_per_buffer=1600,
    )

    async def mic_loop():
        loop = asyncio.get_event_loop()
        while True:
            chunk = await loop.run_in_executor(None, mic_stream.read, 1600, False)
            # Convert PCM16 to μ-law for Deepgram
            import audioop
            mulaw = audioop.lin2ulaw(chunk, 2)
            await stt.send_audio(mulaw)

    mic_task = asyncio.create_task(mic_loop())

    try:
        while True:
            user_text = await transcript_queue.get()
            print(f"{CYAN}[YOU]{RESET} {user_text}")
            messages.append({"role": "user", "content": user_text})
            full_messages = [{"role": "system", "content": system}] + messages[-10:]

            response_parts = []
            async for item in stream_response(full_messages):
                if isinstance(item, str):
                    response_parts.append(item)

            full_response = " ".join(response_parts)
            print(f"{GREEN}[PRIYA]{RESET} {full_response}")
            messages.append({"role": "assistant", "content": full_response})

            if full_response:
                await _play_tts(tts, pa, full_response)
    except KeyboardInterrupt:
        print("\nEnding call.")
    finally:
        mic_task.cancel()
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()
        await stt.close()


async def _play_tts(tts: TTSService, pa, text: str):
    """Synthesize TTS (μ-law 8kHz) and play through speakers."""
    import audioop
    stream = pa.open(
        format=pa.get_format_from_width(2),
        channels=1,
        rate=8000,
        output=True,
    )
    try:
        async for chunk in tts.synthesize(text):
            pcm = audioop.ulaw2lin(chunk, 2)
            stream.write(pcm)
    finally:
        stream.stop_stream()
        stream.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local voice bot simulator")
    parser.add_argument("--audio", action="store_true", help="Use mic + speakers (needs pyaudio)")
    parser.add_argument("--scenarios", action="store_true", help="Auto-run preset test scenarios and exit")
    args = parser.parse_args()

    if args.audio:
        asyncio.run(run_audio_mode())
    else:
        asyncio.run(run_text_mode(scenario_only=args.scenarios))
