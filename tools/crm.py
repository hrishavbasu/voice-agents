"""
HubSpot CRM integration (hubspot-api-client v12).

Operations:
  - lookup_contact_by_phone(phone) → dict | None
  - create_or_update_contact(phone, name, email) → contact_id
  - log_call_activity(contact_id, call_sid, duration_s, transcript) → None
  - create_support_ticket(contact_id, subject, description) → ticket_id

Falls back gracefully when HUBSPOT_ACCESS_TOKEN is not set (dev mode).
"""

import logging
from typing import Optional

from config.base_config import HUBSPOT_ACCESS_TOKEN

logger = logging.getLogger(__name__)

_hubspot = None


def _get_client():
    global _hubspot
    if _hubspot is None:
        # Skip if token is missing or still the example placeholder
        if not HUBSPOT_ACCESS_TOKEN or HUBSPOT_ACCESS_TOKEN.startswith("pat-na1-xxxx"):
            return None
        from hubspot import HubSpot
        _hubspot = HubSpot(access_token=HUBSPOT_ACCESS_TOKEN)
    return _hubspot


# ── Contact operations ────────────────────────────────────────────────────────

def lookup_contact_by_phone(phone: str) -> Optional[dict]:
    """Return HubSpot contact dict or None if not found."""
    client = _get_client()
    if client is None:
        return None
    try:
        from hubspot.crm.contacts import PublicObjectSearchRequest, Filter, FilterGroup
        f = Filter(
            property_name="phone",
            operator="EQ",
            value=phone,
        )
        fg = FilterGroup(filters=[f])
        req = PublicObjectSearchRequest(
            filter_groups=[fg],
            properties=["firstname", "lastname", "email", "phone"],
        )
        result = client.crm.contacts.search_api.do_search(
            public_object_search_request=req
        )
        if result.results:
            contact = result.results[0]
            return {
                "id": contact.id,
                "properties": contact.properties,
            }
        return None
    except Exception as exc:
        logger.error("HubSpot contact lookup error: %s", exc)
        return None


def create_or_update_contact(
    phone: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[str]:
    """Create a new HubSpot contact or update existing one.  Returns contact ID."""
    client = _get_client()
    if client is None:
        logger.info("CRM dry-run: would create/update contact phone=%s", phone)
        return "dry-run-contact-id"
    try:
        from hubspot.crm.contacts import SimplePublicObjectInputForCreate

        existing = lookup_contact_by_phone(phone)
        properties = {"phone": phone}
        if name:
            parts = name.split(" ", 1)
            properties["firstname"] = parts[0]
            if len(parts) > 1:
                properties["lastname"] = parts[1]
        if email:
            properties["email"] = email

        if existing:
            contact_id = existing["id"]
            from hubspot.crm.contacts import SimplePublicObjectInput
            client.crm.contacts.basic_api.update(
                contact_id=contact_id,
                simple_public_object_input=SimplePublicObjectInput(
                    properties=properties
                ),
            )
            logger.info("Updated HubSpot contact %s", contact_id)
            return contact_id
        else:
            new_contact = client.crm.contacts.basic_api.create(
                simple_public_object_input_for_create=SimplePublicObjectInputForCreate(
                    properties=properties
                )
            )
            contact_id = new_contact.id
            logger.info("Created HubSpot contact %s", contact_id)
            return contact_id
    except Exception as exc:
        logger.error("HubSpot create/update contact error: %s", exc)
        return None


# ── Call logging ──────────────────────────────────────────────────────────────

def log_call_activity(
    contact_id: str,
    call_sid: str,
    duration_seconds: int,
    transcript: str,
    disposition: str = "COMPLETED",
) -> None:
    """Log a call engagement on the HubSpot contact timeline."""
    client = _get_client()
    if client is None:
        logger.info(
            "CRM dry-run: would log call sid=%s duration=%ds", call_sid, duration_seconds
        )
        return
    try:
        from hubspot.crm.objects.calls import SimplePublicObjectInputForCreate

        props = {
            "hs_call_title": f"AI Support Call {call_sid}",
            "hs_call_body": transcript[:5000],  # HubSpot 5000-char limit
            "hs_call_duration": str(duration_seconds * 1000),  # milliseconds
            "hs_call_status": disposition,
            "hs_call_direction": "INBOUND",
        }
        call_obj = client.crm.objects.calls.basic_api.create(
            simple_public_object_input_for_create=SimplePublicObjectInputForCreate(
                properties=props
            )
        )
        # Associate call with contact
        from hubspot.crm.objects.calls import AssociationSpec
        client.crm.objects.calls.associations_api.create(
            call_id=call_obj.id,
            to_object_type="contacts",
            to_object_id=contact_id,
            association_spec=[
                AssociationSpec(
                    association_category="HUBSPOT_DEFINED",
                    association_type_id=194,
                )
            ],
        )
        logger.info("Logged call activity %s for contact %s", call_obj.id, contact_id)
    except Exception as exc:
        logger.error("HubSpot log_call_activity error: %s", exc)


# ── Ticket creation ───────────────────────────────────────────────────────────

def create_support_ticket(
    contact_id: str,
    subject: str,
    description: str,
    pipeline_stage: str = "1",
) -> Optional[str]:
    """Create a support ticket and associate it with the contact."""
    client = _get_client()
    if client is None:
        logger.info("CRM dry-run: would create ticket '%s'", subject)
        return "dry-run-ticket-id"
    try:
        from hubspot.crm.tickets import SimplePublicObjectInputForCreate

        ticket = client.crm.tickets.basic_api.create(
            simple_public_object_input_for_create=SimplePublicObjectInputForCreate(
                properties={
                    "subject": subject,
                    "content": description,
                    "hs_pipeline_stage": pipeline_stage,
                }
            )
        )
        ticket_id = ticket.id
        # Associate with contact
        from hubspot.crm.associations import BatchInputPublicObjectId
        client.crm.associations.v4.basic_api.create(
            object_type="tickets",
            object_id=ticket_id,
            to_object_type="contacts",
            to_object_id=contact_id,
            association_spec=[],
        )
        logger.info("Created HubSpot ticket %s for contact %s", ticket_id, contact_id)
        return ticket_id
    except Exception as exc:
        logger.error("HubSpot create_support_ticket error: %s", exc)
        return None
