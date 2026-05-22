import asyncio
import pytest
from pipeline.interruption import InterruptionController


@pytest.mark.asyncio
async def test_trigger_immediate_sets_flag_without_delay():
    ctrl = InterruptionController()
    t0 = asyncio.get_event_loop().time()
    await ctrl.trigger_immediate()
    elapsed = asyncio.get_event_loop().time() - t0
    assert ctrl.is_interrupted
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_debounce_ack_gate_waits_150ms():
    ctrl = InterruptionController()
    await ctrl.trigger_immediate()
    t0 = asyncio.get_event_loop().time()
    ok = await ctrl.debounce_ack_gate()
    elapsed = asyncio.get_event_loop().time() - t0
    assert ok is True
    assert elapsed >= 0.14


@pytest.mark.asyncio
async def test_debounce_ack_gate_returns_false_if_reset_during_wait():
    ctrl = InterruptionController()
    await ctrl.trigger_immediate()

    async def reset_soon():
        await asyncio.sleep(0.05)
        ctrl.reset()

    ok = await asyncio.gather(ctrl.debounce_ack_gate(), reset_soon())
    assert ok[0] is False
    assert ctrl.is_interrupted is False
