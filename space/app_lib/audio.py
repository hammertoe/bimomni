"""Audio preprocessing for the BimOmni demo."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from .config import MAX_AUDIO_SECONDS, TARGET_SAMPLE_RATE


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=1).astype(np.float32)


def _resample(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Linear-interpolation resample — good enough at ≤30 s for ASR."""
    if sr == target_sr:
        return data
    duration = len(data) / sr
    target_length = max(round(duration * target_sr), 1)
    xp = np.linspace(0, len(data) - 1, target_length)
    return np.interp(xp, np.arange(len(data), dtype=np.float64), data.astype(np.float64)).astype(np.float32)


def validate_duration(audio_path: str, max_seconds: int = MAX_AUDIO_SECONDS) -> float:
    """Return the audio duration in seconds; raise if it exceeds max_seconds.

    Runs outside the @spaces.GPU decorator so a too-long upload does not burn
    the daily quota.
    """
    info = sf.info(audio_path)
    if info.duration > max_seconds + 0.5:
        raise ValueError(
            f"Audio is {info.duration:.1f}s; the demo accepts at most {max_seconds}s."
        )
    return float(info.duration)


def load_audio_array(
    audio_path: str, max_seconds: int = MAX_AUDIO_SECONDS
) -> tuple[np.ndarray, int]:
    """Load any audio file, return mono float32 numpy array at 16 kHz (≤ max_seconds)."""
    data, sr = sf.read(audio_path, always_2d=False)
    data = _to_mono(np.asarray(data, dtype=np.float32))
    if sr != TARGET_SAMPLE_RATE:
        data = _resample(data, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE
    max_samples = TARGET_SAMPLE_RATE * max_seconds
    if len(data) > max_samples:
        data = data[:max_samples]
    return data, TARGET_SAMPLE_RATE


def preprocess_to_pcm16_wav(
    audio_path: str, max_seconds: int = MAX_AUDIO_SECONDS
) -> bytes:
    """Same normalisation as load_audio_array, but return PCM_16 WAV bytes."""
    data, sr = load_audio_array(audio_path, max_seconds=max_seconds)
    buf = io.BytesIO()
    sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return buf.read()
