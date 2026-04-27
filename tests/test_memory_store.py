"""Tests for InMemoryStore (memory/store.py)."""

import pytest

from memory.store import InMemoryStore


@pytest.mark.asyncio
async def test_set_and_get():
    store = InMemoryStore()
    await store.set("call1", {"caller": "+91999"})
    result = await store.get("call1")
    assert result == {"caller": "+91999"}


@pytest.mark.asyncio
async def test_get_missing_key_returns_none():
    store = InMemoryStore()
    assert await store.get("nonexistent") is None


@pytest.mark.asyncio
async def test_set_overwrites_existing():
    store = InMemoryStore()
    await store.set("call1", {"v": 1})
    await store.set("call1", {"v": 2})
    result = await store.get("call1")
    assert result == {"v": 2}


@pytest.mark.asyncio
async def test_delete_removes_key():
    store = InMemoryStore()
    await store.set("call1", {"x": 1})
    await store.delete("call1")
    assert await store.get("call1") is None


@pytest.mark.asyncio
async def test_delete_missing_key_is_no_op():
    store = InMemoryStore()
    await store.delete("no-such-key")  # should not raise


@pytest.mark.asyncio
async def test_exists_true_and_false():
    store = InMemoryStore()
    assert await store.exists("k") is False
    await store.set("k", {})
    assert await store.exists("k") is True


@pytest.mark.asyncio
async def test_multiple_keys_are_independent():
    store = InMemoryStore()
    await store.set("a", {"n": 1})
    await store.set("b", {"n": 2})
    assert (await store.get("a"))["n"] == 1
    assert (await store.get("b"))["n"] == 2


@pytest.mark.asyncio
async def test_close_is_safe():
    store = InMemoryStore()
    await store.set("x", {})
    await store.close()
    # After close the data should still be accessible (in-memory store)
    assert await store.get("x") == {}
