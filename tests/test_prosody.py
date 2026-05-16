"""Tests for prosody preprocessor and anti-markdown stripping."""
import pytest


def test_em_dash_replaced_with_comma_space():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("Doctor Bhaskar — he is available tomorrow")
    assert "—" not in result
    assert "," in result


def test_discourse_marker_haan_gets_comma():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("हाँ आपकी appointment book हो गई है")
    assert "हाँ," in result


def test_discourse_marker_actually_gets_comma():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("actually let me check that")
    assert "actually," in result


def test_discourse_marker_already_has_comma_unchanged():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("हाँ, sure बताइए")
    # Should not double-add a comma
    assert result.count("हाँ,") == 1


def test_ellipsis_replaced_with_comma():
    from pipeline.voice_pipeline import _add_prosody_markers
    result = _add_prosody_markers("One moment... let me check")
    assert "..." not in result
    assert "," in result


def test_strip_markdown_bold():
    from pipeline.voice_pipeline import _strip_markdown
    assert _strip_markdown("**Doctor** Bhaskar") == "Doctor Bhaskar"


def test_strip_markdown_italic():
    from pipeline.voice_pipeline import _strip_markdown
    assert _strip_markdown("*please hold*") == "please hold"


def test_strip_markdown_bullet():
    from pipeline.voice_pipeline import _strip_markdown
    result = _strip_markdown("- Option one")
    assert result == "Option one"


def test_strip_markdown_trailing_colon():
    from pipeline.voice_pipeline import _strip_markdown
    result = _strip_markdown("Available slots :")
    assert result == "Available slots"


def test_strip_markdown_preserves_hindi():
    from pipeline.voice_pipeline import _strip_markdown
    result = _strip_markdown("**नमस्ते** Priya")
    assert result == "नमस्ते Priya"
