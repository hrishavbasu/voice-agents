from pipeline.voice_pipeline import VoicePipeline


def test_short_fragment_needs_extra_hold():
    assert VoicePipeline._needs_extra_short_hold("मैम मेरे") is True


def test_meaningful_request_does_not_need_extra_hold():
    assert VoicePipeline._needs_extra_short_hold("नौ बजे कर दो मैम") is False
