"""
Google Calendar integration for Apollo clinic appointment booking.

Auth: Google Service Account (no OAuth browser flow — works server-side).

Setup (one-time, ~5 minutes):
  1. Go to console.cloud.google.com → Create project (free)
  2. APIs & Services → Enable "Google Calendar API"
  3. IAM & Admin → Service Accounts → Create → Download JSON key
  4. Open Google Calendar → Settings → your clinic calendar
     → Share with: <service-account-email> (give "Make changes to events" role)
  5. Copy the Calendar ID from Settings (looks like xxx@group.calendar.google.com)
  6. Add to .env:
       GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/service-account.json
       GOOGLE_CALENDAR_ID=xxx@group.calendar.google.com

If GOOGLE_SERVICE_ACCOUNT_JSON is not set, runs in dry-run mode:
  appointment is logged to console and booking returns success
  so the call flow completes normally.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _get_service():
    """Build an authenticated Google Calendar service. Returns None on failure."""
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not sa_path:
        return None

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        logger.error(
            "google-api-python-client not installed. "
            "Run: pip install google-api-python-client google-auth"
        )
        return None

    try:
        if os.path.exists(sa_path):
            creds = service_account.Credentials.from_service_account_file(
                sa_path, scopes=_SCOPES
            )
        else:
            # Allow raw JSON string in env var instead of a file path
            info = json.loads(sa_path)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=_SCOPES
            )
        return build("calendar", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        logger.error("Google Calendar auth failed: %s", exc)
        return None


def create_appointment_event(
    patient_name: str,
    patient_phone: str,
    doctor_name: str,
    specialty: str,
    concern: str,
    start_dt: datetime,
    duration_minutes: int = 30,
    fee_inr: Optional[int] = None,
) -> dict:
    """
    Create a Google Calendar appointment event.

    Returns:
        {
            "success": True,
            "event_id": "...",
            "event_link": "...",      # Google Calendar event URL
            "slot": "Monday, 21 April at 10:00 AM",
            "doctor": "Dr. Anuj Sathe",
        }
    or
        {"success": False, "reason": "..."}
    """
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    service = _get_service()
    env_name = (os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "development").lower()
    allow_dry_run = os.getenv("CALENDAR_ALLOW_DRY_RUN", "true").lower() == "true"

    slot_human = start_dt.strftime("%A, %d %B at %-I:%M %p")

    if service is None:
        if env_name in {"prod", "production"} and not allow_dry_run:
            reason = "Google Calendar is not configured in production; dry-run is disabled"
            logger.error(reason)
            return {"success": False, "reason": reason}
        # Dry-run — log and return success so the voice flow continues
        logger.info(
            "[DRY RUN] Appointment booked: %s | %s (%s) | %s | Concern: %s | Phone: %s",
            slot_human, doctor_name, specialty, patient_name, concern, patient_phone,
        )
        return {
            "success": True,
            "event_id": "dry-run",
            "slot": slot_human,
            "doctor": doctor_name,
            "note": "Google Calendar not connected — appointment logged locally only",
        }

    end_dt = start_dt + timedelta(minutes=duration_minutes)
    fee_line = f"\nConsultation Fee: ₹{fee_inr:,}" if fee_inr else ""

    description = (
        f"Patient: {patient_name}\n"
        f"Phone: {patient_phone}\n"
        f"Reason for visit: {concern}"
        f"{fee_line}\n\n"
        "Booked via Apollo Clinic Voice Agent (Priya)"
    )

    event_body = {
        "summary": f"Apollo Clinic — {doctor_name} | {patient_name}",
        "description": description,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 24 * 60},
                {"method": "popup", "minutes": 60},
            ],
        },
    }

    try:
        created = (
            service.events()
            .insert(calendarId=calendar_id, body=event_body)
            .execute()
        )
        logger.info(
            "Calendar event created: id=%s patient=%s doctor=%s slot=%s",
            created.get("id"), patient_name, doctor_name, slot_human,
        )
        return {
            "success": True,
            "event_id": created.get("id"),
            "event_link": created.get("htmlLink"),
            "slot": slot_human,
            "doctor": doctor_name,
        }
    except Exception as exc:
        logger.error("Google Calendar insert failed: %s", exc)
        return {"success": False, "reason": str(exc)}
