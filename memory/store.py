"""
Session memory store — swappable in-memory dict (prototype) or Redis (production).

Set USE_REDIS=true in .env to use Redis; otherwise falls back to an in-process
dict (single-process only, lost on restart).

API:
    store = get_store()
    await store.set(call_id, session_dict)
    session = await store.get(call_id)
    await store.delete(call_id)
"""

import asyncio
import json
import logging
from typing import Optional

from config.base_config import REDIS_URL, SESSION_TTL_SECONDS, USE_REDIS

logger = logging.getLogger(__name__)


# ── In-memory store (prototype) ───────────────────────────────────────────────

class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    async def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    async def set(self, key: str, value: dict, ttl: int = SESSION_TTL_SECONDS) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def close(self) -> None:
        pass


# ── Redis store (production) ──────────────────────────────────────────────────

class RedisStore:
    def __init__(self) -> None:
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                REDIS_URL, encoding="utf-8", decode_responses=True
            )
            logger.info("Connected to Redis at %s", REDIS_URL)
        return self._client

    async def get(self, key: str) -> Optional[dict]:
        client = await self._ensure_client()
        raw = await client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set(self, key: str, value: dict, ttl: int = SESSION_TTL_SECONDS) -> None:
        client = await self._ensure_client()
        await client.setex(key, ttl, json.dumps(value))

    async def delete(self, key: str) -> None:
        client = await self._ensure_client()
        await client.delete(key)

    async def exists(self, key: str) -> bool:
        client = await self._ensure_client()
        return bool(await client.exists(key))

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


# ── Factory ───────────────────────────────────────────────────────────────────

_store_instance: Optional[InMemoryStore | RedisStore] = None


def get_store() -> InMemoryStore | RedisStore:
    global _store_instance
    if _store_instance is None:
        if USE_REDIS:
            logger.info("Using Redis session store")
            _store_instance = RedisStore()
        else:
            logger.info("Using in-memory session store")
            _store_instance = InMemoryStore()
    return _store_instance
