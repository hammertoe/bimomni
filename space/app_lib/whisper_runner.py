"""Whisper-large-v3 baseline transcription.

Loaded lazily on first `compare()` call inside the @spaces.GPU task so the
Space starts instantly. The pipeline is cached at module level because a
single Cold start is plenty for the daily quota.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_pipe: Any = None
_loaded_for: str | None = None


def load_whisper(model_id: str) -> Any:
    """Idempotent load of the Whisper ASR pipeline."""
    global _pipe, _loaded_for
    if _pipe is not None and _loaded_for == model_id:
        return _pipe

    from transformers import pipeline  # heavy import; do it once

    log.info("loading Whisper pipeline: %s", model_id)
    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
        torch_dtype="bfloat16",
        device_map="cuda",
    )
    _loaded_for = model_id
    return _pipe


def transcribe(pipe: Any, audio_path: str) -> str:
    """Run Whisper on a file path; return plain transcript text."""
    result = pipe(audio_path, return_timestamps=False, generate_kwargs={"language": "en"})
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return str(result[0].get("text", "")).strip()
    return str(result).strip()
