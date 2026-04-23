"""
LLM tool schemas for Apollo Clinic voice agent.

Only tools listed in COMPANY_CONFIG["tools_enabled"] are sent to the LLM.

Usage:
    from tools.definitions import get_tool_schemas
    tools = get_tool_schemas()
"""

from config.company_config import COMPANY_CONFIG

TOOL_REGISTRY: dict[str, dict] = {
    "check_doctor_slots": {
        "type": "function",
        "function": {
            "name": "check_doctor_slots",
            "description": (
                "Check available appointment slots for a specific doctor on a given date. "
                "ALWAYS call this before book_appointment to show the caller what times are open. "
                "If the doctor is unavailable on that date, returns their next available date "
                "and slots, plus any alternative doctors of the same specialty available on the requested date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {
                        "type": "string",
                        "description": "Name of the doctor to check (e.g. 'Dr. Atul Bhaskar').",
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": (
                            "Date to check availability for "
                            "(e.g. 'tomorrow', 'Monday', '20 April'). "
                            "Leave blank to check next available day."
                        ),
                    },
                },
                "required": ["doctor_name"],
            },
        },
    },
    "list_doctors": {
        "type": "function",
        "function": {
            "name": "list_doctors",
            "description": (
                "List available doctors at Apollo Hospitals. "
                "Use when the caller asks which doctors are available, "
                "wants to know about a specialty, or needs help choosing a doctor. "
                "Optionally filter by specialty keyword."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": (
                            "Optional specialty or keyword to filter by "
                            "(e.g. 'heart', 'cardiology', 'eye', 'bone', 'cancer'). "
                            "Leave empty to list all doctors."
                        ),
                    }
                },
                "required": [],
            },
        },
    },

    "book_appointment": {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book a doctor's appointment at Apollo Hospitals Navi Mumbai. "
                "Collect patient name, concern/symptoms, doctor or specialty preference, "
                "and preferred date/time before calling this tool. "
                "The appointment will be created in Google Calendar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full name of the patient.",
                    },
                    "concern": {
                        "type": "string",
                        "description": (
                            "Brief description of the patient's symptoms or reason for visit "
                            "(e.g. 'chest pain', 'knee pain after fall', 'routine eye checkup')."
                        ),
                    },
                    "doctor_name": {
                        "type": "string",
                        "description": (
                            "Name of the preferred doctor (e.g. 'Dr. Anuj Sathe'). "
                            "If the caller only knows the specialty, leave blank and "
                            "use the specialty field instead."
                        ),
                    },
                    "specialty": {
                        "type": "string",
                        "description": (
                            "Specialty if no specific doctor was chosen "
                            "(e.g. 'Cardiology', 'Orthopedic Surgery')."
                        ),
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": (
                            "Preferred date as mentioned by caller "
                            "(e.g. 'Monday', 'tomorrow', '25th April', 'next week')."
                        ),
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": (
                            "Preferred time as mentioned by caller "
                            "(e.g. 'morning', '11am', 'afternoon', '3pm')."
                        ),
                    },
                },
                "required": ["patient_name", "concern"],
            },
        },
    },

    "escalate_to_human": {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Transfer this call to a human staff member at Apollo Hospitals. "
                "Use when: (1) the caller explicitly asks to speak with a person, "
                "(2) there is a medical emergency (also advise to call 108), "
                "(3) the issue cannot be resolved after 2 attempts, or "
                "(4) the caller sounds very distressed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Brief reason for escalation "
                            "(e.g. 'Caller requested human', 'Medical emergency', "
                            "'Billing query')."
                        ),
                    }
                },
                "required": ["reason"],
            },
        },
    },
}


def get_tool_schemas() -> list[dict]:
    """Return only the tool schemas enabled in the current config."""
    enabled = COMPANY_CONFIG.get("tools_enabled", list(TOOL_REGISTRY.keys()))
    return [TOOL_REGISTRY[name] for name in enabled if name in TOOL_REGISTRY]


def get_tool_names() -> list[str]:
    """Return enabled tool names."""
    enabled = COMPANY_CONFIG.get("tools_enabled", list(TOOL_REGISTRY.keys()))
    return [name for name in enabled if name in TOOL_REGISTRY]
