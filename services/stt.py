"""STT provider router — returns SarvamSTT or DeepgramSTT based on STT_PROVIDER."""
from config.base_config import STT_PROVIDER


def get_stt(on_transcript, on_speech_started):
    """Return the configured STT client. Both implement connect/send_audio/close."""
    if STT_PROVIDER == "deepgram":
        from services.stt_deepgram import DeepgramSTT
        return DeepgramSTT(on_transcript, on_speech_started)
    from services.stt_sarvam import SarvamSTT
    return SarvamSTT(on_transcript, on_speech_started)
