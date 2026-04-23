"""
ElevenLabs Pronunciation Dictionary — one-time setup utility.

Creates (or re-uses) a named pronunciation dictionary with PLS entries
for proper nouns that the TTS voice mispronounces in Latin script.

Usage (called once at app startup in main.py):
    from services.pronunciation_dict import ensure_pronunciation_dict
    dict_id, version_id = await ensure_pronunciation_dict()

The returned IDs are injected into every TTS request so ElevenLabs
uses the correct IPA phonemes for the listed words.

Why PLS?
    ElevenLabs accepts PLS (W3C Pronunciation Lexicon Specification) files.
    IPA entries in PLS override the voice's default grapheme-to-phoneme model,
    which mispronounces "Navi" as /naːvi/ (long-ā) instead of /nʌvi/.
"""

import logging
import os
from typing import Optional

import httpx

from config.base_config import ELEVENLABS_API_KEY
from config.company_config import COMPANY_CONFIG

logger = logging.getLogger(__name__)

_BASE = "https://api.elevenlabs.io/v1"
_DICT_NAME = "clinic_proper_nouns"

# Cached at module level — populated once by ensure_pronunciation_dict()
_dict_id: Optional[str] = None
_version_id: Optional[str] = None


def _build_pls(entries: dict[str, str]) -> bytes:
    """
    Build a PLS XML document from {grapheme: ipa_phoneme} entries.

    ElevenLabs expects UTF-8 encoded PLS bytes with the standard W3C namespace.
    - alphabet goes on <lexicon> only, NOT on <phoneme> (non-standard and rejected)
    - xml:lang must be a standard BCP-47 tag; "en-US" is safest
    """
    items = ""
    for grapheme, ipa in entries.items():
        items += (
            f"  <lexeme>\n"
            f"    <grapheme>{grapheme}</grapheme>\n"
            f"    <phoneme>{ipa}</phoneme>\n"
            f"  </lexeme>\n"
        )
    pls = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<lexicon version="1.0"\n'
        '  xmlns="http://www.w3.org/2005/01/pronunciation-lexicon"\n'
        '  alphabet="ipa"\n'
        '  xml:lang="en-US">\n'
        f"{items}"
        "</lexicon>\n"
    )
    return pls.encode("utf-8")


async def _list_dicts(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get(
        f"{_BASE}/pronunciation-dictionaries",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        params={"page_size": 100},
    )
    r.raise_for_status()
    return r.json().get("pronunciation_dictionaries", [])


async def _create_dict(
    client: httpx.AsyncClient, entries: dict[str, str]
) -> tuple[str, str]:
    """Upload PLS file and create a new pronunciation dictionary."""
    pls_bytes = _build_pls(entries)
    logger.debug("PLS content:\n%s", pls_bytes.decode("utf-8"))
    r = await client.post(
        f"{_BASE}/pronunciation-dictionaries/add-from-file",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        files={"file": ("clinic_proper_nouns.pls", pls_bytes, "text/xml")},
        data={"name": _DICT_NAME},
    )
    r.raise_for_status()
    data = r.json()
    return data["id"], data["version_id"]


async def _add_rules(
    client: httpx.AsyncClient, dict_id: str, entries: dict[str, str]
) -> str:
    """Add/update phoneme rules in an existing dictionary, returns new version_id."""
    rules = [
        {"type": "phoneme", "string_to_replace": g, "phoneme": ipa, "alphabet": "ipa"}
        for g, ipa in entries.items()
    ]
    r = await client.post(
        f"{_BASE}/pronunciation-dictionaries/{dict_id}/add-rules",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        json={"rules": rules},
    )
    r.raise_for_status()
    return r.json()["version_id"]


async def ensure_pronunciation_dict() -> tuple[str, str]:
    """
    Ensure the clinic pronunciation dictionary exists in ElevenLabs.

    Resolution order:
    1. Env vars already set (ELEVENLABS_PRONUNCIATION_DICT_ID / VERSION_ID) → use them.
    2. Dictionary with name "clinic_proper_nouns" already exists → update rules + return.
    3. Neither → create from PLS file, return new IDs.

    Returns:
        (dict_id, version_id) — empty strings if no entries configured or API key missing.
    """
    global _dict_id, _version_id

    if _dict_id and _version_id:
        return _dict_id, _version_id

    entries: dict[str, str] = COMPANY_CONFIG.get("tts_pronunciation_entries", {})
    if not entries:
        logger.info("No tts_pronunciation_entries configured — skipping pronunciation dict setup")
        return "", ""

    if not ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY not set — skipping pronunciation dict setup")
        return "", ""

    # Check env vars first (set after first run to avoid repeated API calls)
    env_dict_id = os.getenv("ELEVENLABS_PRONUNCIATION_DICT_ID", "")
    env_version_id = os.getenv("ELEVENLABS_PRONUNCIATION_DICT_VERSION_ID", "")
    if env_dict_id and env_version_id:
        logger.info(
            "Pronunciation dict loaded from env: id=%s version=%s", env_dict_id, env_version_id
        )
        _dict_id, _version_id = env_dict_id, env_version_id
        return _dict_id, _version_id

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            existing = await _list_dicts(client)
            match = next((d for d in existing if d.get("name") == _DICT_NAME), None)

            if match:
                dict_id = match["id"]
                version_id = await _add_rules(client, dict_id, entries)
                logger.info(
                    "Pronunciation dict updated: id=%s new_version=%s", dict_id, version_id
                )
            else:
                dict_id, version_id = await _create_dict(client, entries)
                logger.info(
                    "Pronunciation dict created: id=%s version=%s", dict_id, version_id
                )

            logger.info(
                "Add to .env to skip re-creation:\n"
                "  ELEVENLABS_PRONUNCIATION_DICT_ID=%s\n"
                "  ELEVENLABS_PRONUNCIATION_DICT_VERSION_ID=%s",
                dict_id,
                version_id,
            )
            _dict_id, _version_id = dict_id, version_id
            return _dict_id, _version_id

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Pronunciation dict setup failed (HTTP %s): %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return "", ""
        except Exception as exc:
            logger.error("Pronunciation dict setup error: %s", exc)
            return "", ""


def get_pronunciation_dict_locators() -> list[dict] | None:
    """
    Return the pronunciation_dictionary_locators payload for ElevenLabs TTS requests,
    or None if no dictionary has been configured.
    """
    if _dict_id and _version_id:
        return [{"pronunciation_dictionary_id": _dict_id, "version_id": _version_id}]
    return None
