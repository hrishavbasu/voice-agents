"""
Escalation tool — transfers the call to a human agent and exports the
transcript to HubSpot so the agent has full context on pick-up.
"""

import logging
from typing import Optional

from config.company_config import COMPANY_CONFIG

logger = logging.getLogger(__name__)


async def escalate_to_human(
    call_id: str,
    caller_phone: str,
    transcript: str,
    reason: str = "Customer requested human agent",
    telephony_session=None,
    crm_contact_id: Optional[str] = None,
    priority: str = "NORMAL",
) -> dict:
    """
    1. Log transcript + escalation reason to HubSpot.
    2. Trigger SIP transfer to the company escalation queue.

    Returns:
        {"success": True, "destination": "+1-800-xxx"}
    or
        {"success": False, "reason": "..."}
    """
    destination = COMPANY_CONFIG.get("escalation_phone", "")
    if not destination:
        logger.error("No escalation_phone configured")
        return {"success": False, "reason": "No escalation number configured."}

    # ── Log to CRM ────────────────────────────────────────────────────────────
    try:
        from tools.crm import log_call_activity, create_support_ticket

        if crm_contact_id:
            subject = (
                f"🚨 EMERGENCY Escalated Call — {reason}"
                if priority == "EMERGENCY"
                else f"Escalated Call — {reason}"
            )
            create_support_ticket(
                contact_id=crm_contact_id,
                subject=subject,
                description=(
                    f"PRIORITY: {priority}\n"
                    f"Call ID: {call_id}\n\n"
                    f"Reason: {reason}\n\n"
                    f"Transcript:\n{transcript}"
                ),
            )
            disposition = "EMERGENCY_ESCALATED" if priority == "EMERGENCY" else "ESCALATED"
            log_call_activity(
                contact_id=crm_contact_id,
                call_sid=call_id,
                duration_seconds=0,
                transcript=transcript,
                disposition=disposition,
            )
            if priority == "EMERGENCY":
                logger.warning(
                    "EMERGENCY escalation logged to CRM for call %s — caller: %s",
                    call_id,
                    caller_phone,
                )
    except Exception as exc:
        logger.error("CRM escalation logging failed: %s", exc)

    # ── Transfer call ─────────────────────────────────────────────────────────
    if telephony_session:
        try:
            await telephony_session.transfer(destination)
            logger.info("Call %s transferred to %s", call_id, destination)
        except Exception as exc:
            logger.error("Call transfer failed: %s", exc)
            return {"success": False, "reason": f"Transfer failed: {exc}"}
    else:
        logger.warning("No telephony session — cannot transfer call %s", call_id)

    return {"success": True, "destination": destination, "reason": reason}
