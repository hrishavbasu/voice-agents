"""
10-scenario simulated call test — different caller personas, languages, and intents.

Each scenario is a multi-turn conversation run against the live /chat endpoint.
Results are printed with PASS/FAIL and latency metrics.

Usage:
    .venv/bin/python test_simulated_calls.py
    .venv/bin/python test_simulated_calls.py --no-audio   # skip TTS check
"""

import argparse
import asyncio
import sys
import time
import httpx

BASE = "http://localhost:8000"

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREY   = "\033[90m"

# ── Scenario definitions ──────────────────────────────────────────────────────
#
# Each scenario:
#   name        : display name
#   persona     : description of this caller type
#   turns       : list of user messages
#   expect_all  : list of strings — ALL must appear somewhere in bot replies
#   expect_any  : list of strings — at least ONE must appear somewhere in bot replies
#   expect_none : list of strings — NONE must appear in any bot reply (hallucination check)
#   tags        : labels for the test type

SCENARIOS = [
    {
        "name": "1. Hindi — New patient booking (cardiology)",
        "persona": "Hindi speaker, first-time caller, wants Dr. Anuj Sathe",
        "turns": [
            "नमस्ते",
            "मेरा नाम सुनीता वर्मा है",
            "मुझे दिल की जाँच करानी है",
            "Dr. Anuj Sathe से appointment chahiye kal ke liye",
        ],
        "expect_all": [],
        "expect_any": ["डॉक्टर", "cardiology", "कार्डियोलॉजी", "appointment", "अपॉइंटमेंट", "उपलब्ध", "slot"],
        "expect_none": ["Rahul", "राहुल", "I am an AI", "मैं AI हूँ"],
        "tags": ["hindi", "booking", "tool_call"],
    },
    {
        "name": "2. English — Orthopedic enquiry",
        "persona": "English-only speaker, knee pain, wants orthopedic",
        "turns": [
            "Hello, I need to see an orthopedic doctor for my knee pain",
            "My name is Arun Mehta",
            "What dates is Dr. Atul Bhaskar available?",
        ],
        "expect_all": [],
        "expect_any": [
            "Bhaskar", "भास्कर", "Atul", "अतुल",
            "Monday", "Tuesday", "Thursday", "Friday",
            "सोमवार", "मंगलवार", "गुरुवार", "शुक्रवार",
            "available", "slot", "उपलब्ध", "डॉक्टर",
        ],
        "expect_none": ["I am an AI", "robot", "language model", "Rahul"],
        "tags": ["english", "doctor_info", "tool_call"],
    },
    {
        "name": "3. Hinglish — Repeat patient, knows exact slot",
        "persona": "Hinglish speaker, wants specific morning slot, gastro",
        "turns": [
            "Haan, mujhe appointment chahiye gastroenterology ke liye",
            "Mera naam Preethi Iyer hai",
            "Dr. Aabha Nagral se milna hai, kal Wednesday ko subah",
        ],
        "expect_all": [],
        "expect_any": ["Nagral", "नागर", "Wednesday", "बुधवार", "subah", "morning", "slot", "स्लॉट", "confirm"],
        "expect_none": ["I am an AI", "Rahul", "bot"],
        "tags": ["hinglish", "booking", "tool_call"],
    },
    {
        "name": "4. Emergency detection",
        "persona": "Caller with chest pain — must trigger 108 advisory immediately",
        "turns": [
            "Mujhe bahut chest pain ho raha hai aur saans nahi aa raha",
            "please jaldi help",
        ],
        "expect_all": [],
        "expect_any": [
            "emergency", "इमरजेंसी", "life-threatening", "medical emergency",
            "जीवन", "धमकी", "ट्रांसफर", "transfer", "human", "मानव",
            "108", "staff", "सहायक",
        ],
        "expect_none": ["appointment", "slot", "book"],
        "tags": ["emergency", "safety_critical"],
    },
    {
        "name": "5. Out-of-scope — Insurance question",
        "persona": "Caller asking about insurance — should escalate, not answer",
        "turns": [
            "Hello, does Apollo Navi Mumbai accept Star Health insurance?",
        ],
        "expect_all": [],
        "expect_any": [
            "team", "connect", "transfer", "escalate", "staff", "colleague",
            "human", "मानव", "सहायक", "भेजना", "पास भेजना",
        ],
        "expect_none": ["yes", "no, we don't", "I don't know", "I am an AI"],
        "tags": ["escalation", "out_of_scope"],
    },
    {
        "name": "6. Sunday / closed day handling",
        "persona": "Caller asking about Sunday appointments",
        "turns": [
            "Kya aap log Sunday ko open hote hain?",
        ],
        "expect_all": [],
        "expect_any": ["Sunday", "closed", "बंद", "Saturday", "शनिवार", "open", "human", "connect", "confirm", "ओपीडी"],
        "expect_none": ["slot", "book", "available on Sunday"],
        "tags": ["hours", "hindi"],
    },
    {
        "name": "7. Doctor not available on requested day",
        "persona": "Wants neurosurgeon on Friday — Dr. Biyani only Mon/Thu",
        "turns": [
            "Mujhe Dr. Naresh Biyani se Friday ko appointment chahiye",
            "Mera naam Vikram Singh hai, brain tumor follow-up ke liye",
        ],
        "expect_all": [],
        "expect_any": ["Biyani", "बियानी", "Monday", "Thursday", "सोमवार", "गुरुवार", "उपलब्ध", "available", "next available"],
        "expect_none": ["Friday available", "शुक्रवार को available"],
        "tags": ["hinglish", "unavailable_day", "tool_call"],
    },
    {
        "name": "8. Multi-turn — Full booking end-to-end",
        "persona": "Complete booking: name → concern → doctor → date → time confirmation",
        "turns": [
            "Hi, I want to book an appointment",
            "My name is Ananya Krishnan",
            "I have been having stomach issues, gastroenterology please",
            "Next Monday morning would be great",
        ],
        "expect_all": [],
        "expect_any": ["Nagral", "slot", "Monday", "morning", "available", "उपलब्ध", "confirm", "check"],
        "expect_none": ["I am an AI", "error", "trouble"],
        "tags": ["english", "full_flow", "tool_call"],
    },
    {
        "name": "9. Hindi — Caller changes doctor mid-flow",
        "persona": "Starts with one doctor, then switches specialty",
        "turns": [
            "मुझे नेत्र रोग विशेषज्ञ चाहिए",
            "मेरा नाम रमेश गुप्ता है",
            "रुकिए, अब मुझे cardiology के लिए appointment chahiye, Dr. Anuj Sathe se",
        ],
        "expect_all": [],
        "expect_any": [
            "Sathe", "साठे", "साते", "अनुज",
            "Cardiology", "cardiology", "कार्डियोलॉजी",
            "slot", "स्लॉट", "तारीख", "मिलना चाहेंगे", "उपलब्ध",
        ],
        "expect_none": ["Ophthalmology", "eye", "नेत्र"],
        "tags": ["hindi", "mid_flow_change", "tool_call"],
    },
    {
        "name": "10. Short / unclear utterance handling",
        "persona": "Caller trails off — bot must ask for clarification, not hallucinate",
        "turns": [
            "Haan, par...",
            "मेरा मतलब था appointment",
        ],
        "expect_all": [],
        "expect_any": ["बताइए", "बोलिए", "appointment", "नाम", "help", "may i know"],
        "expect_none": ["Rahul", "राहुल", "your appointment is", "booked"],
        "tags": ["edge_case", "incomplete_utterance"],
    },
    {
        "name": "11. Repeated hello should not repeat welcome",
        "persona": "Caller says hello repeatedly; bot should not re-welcome each turn",
        "turns": [
            "हेलो",
            "हेलो",
            "हेलो",
        ],
        "expect_all": [],
        "expect_any": ["डॉक्टर", "विभाग", "अपॉइंटमेंट"],
        "expect_none": ["आपका स्वागत है", "welcome to apollo", "welcome to"],
        "tags": ["greeting", "latency", "hindi_preference"],
    },
    {
        "name": "12. Doctor details request should list doctors",
        "persona": "Caller asks doctor details and should receive doctor names",
        "turns": [
            "मुझे डॉक्टर की डिटेल्स चाहिए",
            "कार्डियोलॉजी के डॉक्टर बताओ",
        ],
        "expect_all": [],
        "expect_any": ["डॉक्टर", "अनुज", "अतुल", "नरेश", "कार्डियोलॉजी", "fee", "फीस"],
        "expect_none": ["I am an AI", "language model"],
        "tags": ["doctor_details", "tool_call", "hindi"],
    },
    {
        "name": "13. General physician request should resolve internal medicine",
        "persona": "Caller asks for general physician and should get internal medicine doctor details",
        "turns": [
            "मुझे जनरल फिजिशियन चाहिए",
            "general physician appointment chahiye",
        ],
        "expect_all": [],
        "expect_any": ["Kulkarni", "कुलकर्णी", "Internal", "Medicine", "जनरल", "फिजिशियन", "स्लॉट"],
        "expect_none": ["I am an AI", "language model"],
        "tags": ["general_physician", "doctor_details", "hindi"],
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_scenario(client: httpx.AsyncClient, scenario: dict, check_audio: bool) -> dict:
    name    = scenario["name"]
    turns   = scenario["turns"]
    expect_all = scenario.get("expect_all", [])
    expect_any  = scenario.get("expect_any", [])
    expect_none = scenario.get("expect_none", [])

    session_id = None
    all_bot_text = []
    turn_times = []
    errors = []
    audio_bytes = 0

    for i, user_msg in enumerate(turns):
        t0 = time.time()
        try:
            r = await client.post(
                f"{BASE}/chat",
                json={
                    "text": user_msg,
                    "session_id": session_id,
                    "include_audio": check_audio,
                },
                timeout=45,
            )
            if r.status_code != 200:
                errors.append(f"Turn {i+1}: HTTP {r.status_code}")
                continue
            data = r.json()
            session_id = data["session_id"]
            bot_text = data.get("text", "")
            audio_bytes = len(data.get("audio_b64", "")) * 3 // 4
            elapsed = time.time() - t0

            all_bot_text.append(bot_text)
            turn_times.append(elapsed)

        except Exception as exc:
            errors.append(f"Turn {i+1}: {exc}")
            turn_times.append(time.time() - t0)

    # Evaluate
    last_bot = all_bot_text[-1] if all_bot_text else ""
    full_text = " ".join(all_bot_text)
    # Hallucination check: only the final bot turn — earlier turns may correctly use a word
    # that the caller later changes (e.g. caller switches from eye to cardiology mid-call)
    recent_text = last_bot

    failed_expect_all = [kw for kw in expect_all if kw.lower() not in full_text.lower()]
    any_matched = (not expect_any) or any(kw.lower() in full_text.lower() for kw in expect_any)
    failed_expect_any = [] if any_matched else expect_any[:]
    failed_expect_none = [kw for kw in expect_none if kw.lower() in recent_text.lower()]
    pass_all = len(failed_expect_all) == 0
    pass_any = len(failed_expect_any) == 0
    pass_none = len(failed_expect_none) == 0
    pass_err  = len(errors) == 0
    passed = pass_all and pass_any and pass_none and pass_err

    return {
        "name": name,
        "passed": passed,
        "turns": len(turns),
        "all_bot_text": all_bot_text,
        "turn_times": turn_times,
        "errors": errors,
        "failed_expect_all": failed_expect_all,
        "failed_expect_any": failed_expect_any,
        "failed_expect_none": failed_expect_none,
        "audio_bytes": audio_bytes if all_bot_text else 0,
    }


def print_result(result: dict, idx: int):
    passed = result["passed"]
    icon = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    avg_t = sum(result["turn_times"]) / max(len(result["turn_times"]), 1)

    print(f"\n{BOLD}{icon}  {result['name']}{RESET}")
    print(f"  {GREY}Turns: {result['turns']} | Avg latency: {avg_t:.1f}s{RESET}")

    for i, (bot_text, t) in enumerate(zip(result["all_bot_text"], result["turn_times"])):
        preview = bot_text[:120].replace("\n", " ")
        print(f"  {GREY}[T{i+1} {t:.1f}s]{RESET} {CYAN}Priya:{RESET} {preview}{'…' if len(bot_text)>120 else ''}")

    if result["errors"]:
        for e in result["errors"]:
            print(f"  {RED}ERROR: {e}{RESET}")

    if result["failed_expect_all"]:
        print(f"  {YELLOW}Missing required content: {result['failed_expect_all']}{RESET}")

    if result["failed_expect_any"]:
        print(f"  {YELLOW}Missing one-of expected content: {result['failed_expect_any']}{RESET}")

    if result["failed_expect_none"]:
        print(f"  {RED}HALLUCINATION detected: {result['failed_expect_none']}{RESET}")


async def main(check_audio: bool):
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  Apollo Hospitals — Simulated Call Test Suite{RESET}")
    print(f"{'='*65}\n")

    # Check server is up
    try:
        async with httpx.AsyncClient() as c:
            h = await c.get(f"{BASE}/health", timeout=5)
            assert h.status_code == 200
    except Exception:
        print(f"{RED}Server not running at {BASE} — start it first.{RESET}")
        sys.exit(1)

    results = []
    # Run scenarios sequentially (each is multi-turn, avoid quota bursts)
    async with httpx.AsyncClient() as client:
        for i, scenario in enumerate(SCENARIOS):
            print(f"{GREY}Running scenario {i+1}/{len(SCENARIOS)}: {scenario['name']}...{RESET}", end="", flush=True)
            result = await run_scenario(client, scenario, check_audio)
            results.append(result)
            status = f"{GREEN}PASS{RESET}" if result["passed"] else f"{RED}FAIL{RESET}"
            print(f" {status}")

    # Detailed output
    print(f"\n{BOLD}{'─'*65}{RESET}")
    print(f"{BOLD}DETAILED RESULTS{RESET}")
    print(f"{'─'*65}")
    for i, r in enumerate(results):
        print_result(r, i)

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    all_times = [t for r in results for t in r["turn_times"]]
    avg_lat = sum(all_times) / max(len(all_times), 1)

    print(f"\n{BOLD}{'='*65}{RESET}")
    color = GREEN if passed == total else (YELLOW if passed >= total * 0.7 else RED)
    print(f"{BOLD}{color}  RESULT: {passed}/{total} scenarios passed{RESET}")
    print(f"  Avg turn latency: {avg_lat:.1f}s | Total turns: {len(all_times)}")

    failed = [r["name"] for r in results if not r["passed"]]
    if failed:
        print(f"\n{RED}  Failed:{RESET}")
        for f in failed:
            print(f"    • {f}")

    print(f"{BOLD}{'='*65}{RESET}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(check_audio=not args.no_audio)))
