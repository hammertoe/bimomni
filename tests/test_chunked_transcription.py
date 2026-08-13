"""Deterministic helpers for the V4 multimodal TikTok transcription spike."""
from pathlib import Path
from unittest.mock import patch

from bimomni.transcription.chunked import (
    V4_AUDIO_PROMPT,
    V4_STITCH_PROMPT,
    AudioWindow,
    _extract_audio_window,
    _valid_model_continuation,
    audio_windows,
    concatenate_transcripts,
    frame_timestamps,
)


def test_audio_prompt_requires_verbatim_tiktok_transcription() -> None:
    assert "TikTok audio verbatim" in V4_AUDIO_PROMPT
    assert "Do not summarize" in V4_AUDIO_PROMPT
    assert "FM radio" not in V4_AUDIO_PROMPT


def test_stitch_prompt_only_removes_overlap() -> None:
    assert "Remove only its duplicated opening overlap" in V4_STITCH_PROMPT
    assert "summarize" in V4_STITCH_PROMPT
    assert "Do not" in V4_STITCH_PROMPT


def test_audio_window_is_normalized_for_v4() -> None:
    window = AudioWindow(index=0, start_s=0, end_s=29)

    with patch("bimomni.transcription.chunked.subprocess.run") as run:
        _extract_audio_window(Path("video.mp4"), window, Path("chunk.mp3"))

    command = run.call_args.args[0]
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-ac") + 1] == "1"


def test_audio_windows_cover_source_with_five_second_overlap() -> None:
    windows = audio_windows(129.85)

    assert [(round(window.start_s), round(window.end_s)) for window in windows] == [
        (0, 30), (25, 55), (50, 80), (75, 105), (100, 130),
    ]


def test_frame_timestamps_are_within_the_audio_window() -> None:
    window = AudioWindow(index=1, start_s=25, end_s=55)

    assert frame_timestamps(window) == (32.5, 40.0, 47.5)


def test_concatenation_keeps_audit_record_and_removes_exact_boundary_repeat() -> None:
    literal, stitched = concatenate_transcripts([
        {"start_s": 0.0, "end_s": 30.0, "transcript": "One two three four"},
        {"start_s": 25.0, "end_s": 55.0, "transcript": "three four five six"},
    ])

    assert literal == "[0.0s-30.0s] One two three four\n\n[25.0s-55.0s] three four five six"
    assert stitched == "One two three four five six"


def test_concatenation_removes_fuzzy_asr_boundary_repeat() -> None:
    _, stitched = concatenate_transcripts([
        {
            "start_s": 0.0,
            "end_s": 29.0,
            "transcript": "Visit Worthing Square, a vibrant open-air food garden with local eats. Grab lunch from Chefette, then",
        },
        {
            "start_s": 24.0,
            "end_s": 53.0,
            "transcript": "A vibrant open air food garden with local eats. Grab lunch from Chefette, then walk the boardwalk.",
        },
    ])

    assert stitched.count("food garden") == 1
    assert stitched.endswith("walk the boardwalk.")


def test_model_continuation_accepts_complete_chunk_tail() -> None:
    current = "one two three four five six seven eight nine ten"
    candidate = "four five six seven eight nine ten"

    assert _valid_model_continuation(candidate, current, candidate.split())


def test_model_continuation_rejects_truncated_chunk_tail() -> None:
    current = "one two three four five six seven eight nine ten"
    candidate = "one two three four five"

    assert not _valid_model_continuation(candidate, current, current.split())


def test_model_continuation_rejects_unremoved_overlap() -> None:
    current = "one two three four five six seven eight nine ten"
    deterministic = "four five six seven eight nine ten".split()

    assert not _valid_model_continuation(current, current, deterministic)
