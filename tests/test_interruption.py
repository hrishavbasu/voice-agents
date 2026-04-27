"""Tests for InterruptionController (pipeline/interruption.py)."""

import asyncio
import pytest

from pipeline.interruption import InterruptionController


@pytest.mark.asyncio
async def test_initial_state_not_interrupted():
    ctrl = InterruptionController()
    assert ctrl.is_interrupted is False


@pytest.mark.asyncio
async def test_trigger_sets_interrupted_after_debounce():
    ctrl = InterruptionController()
    await ctrl.trigger()
    assert ctrl.is_interrupted is True


@pytest.mark.asyncio
async def test_reset_clears_interrupted():
    ctrl = InterruptionController()
    await ctrl.trigger()
    ctrl.reset()
    assert ctrl.is_interrupted is False


@pytest.mark.asyncio
async def test_second_trigger_while_interrupted_is_no_op():
    ctrl = InterruptionController()
    await ctrl.trigger()
    # Calling trigger again should not raise and state stays interrupted
    await ctrl.trigger()
    assert ctrl.is_interrupted is True


@pytest.mark.asyncio
async def test_reset_during_debounce_prevents_interrupt():
    """If reset() is called while debouncing, the interrupt should not fire."""
    ctrl = InterruptionController()

    async def trigger_then_reset():
        task = asyncio.create_task(ctrl.trigger())
        # Yield control so trigger() enters its sleep
        await asyncio.sleep(0)
        ctrl.reset()
        await task

    await trigger_then_reset()
    assert ctrl.is_interrupted is False


@pytest.mark.asyncio
async def test_wait_for_interruption_resolves():
    ctrl = InterruptionController()

    async def do_trigger():
        await ctrl.trigger()

    task = asyncio.create_task(do_trigger())
    await ctrl.wait_for_interruption()
    await task
    assert ctrl.is_interrupted is True


@pytest.mark.asyncio
async def test_multiple_reset_trigger_cycles():
    ctrl = InterruptionController()
    for _ in range(3):
        await ctrl.trigger()
        assert ctrl.is_interrupted is True
        ctrl.reset()
        assert ctrl.is_interrupted is False
