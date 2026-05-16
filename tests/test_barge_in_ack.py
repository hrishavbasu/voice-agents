import asyncio
import pytest
from unittest.mock import MagicMock, patch


def _make_pipeline(queue_empty: bool):
    from pipeline.voice_pipeline import VoicePipeline

    p = MagicMock(spec=VoicePipeline)
    p._transcript_queue = asyncio.Queue()
    if not queue_empty:
        p._transcript_queue.put_nowait("caller speaking")
    p._should_play_barge_in_ack = VoicePipeline._should_play_barge_in_ack.__get__(p, VoicePipeline)
    return p


def test_sometimes_skips_ack_when_transcript_queued():
    with patch("pipeline.voice_pipeline.BARGE_IN_ACK_MODE", "sometimes"):
        p = _make_pipeline(queue_empty=False)
        assert p._should_play_barge_in_ack(1) is False


def test_sometimes_plays_ack_when_queue_empty():
    with patch("pipeline.voice_pipeline.BARGE_IN_ACK_MODE", "sometimes"):
        p = _make_pipeline(queue_empty=True)
        assert p._should_play_barge_in_ack(1) is True


def test_silent_never_plays_ack():
    with patch("pipeline.voice_pipeline.BARGE_IN_ACK_MODE", "silent"):
        p = _make_pipeline(queue_empty=True)
        assert p._should_play_barge_in_ack(2) is False
