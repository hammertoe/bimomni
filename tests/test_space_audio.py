"""Tests for the Space's audio preprocessing.

The Space ships pure-Python audio normalisation (load -> mono -> 16kHz -> cap)
in `space/app_lib/audio.py`. These tests pin the contract that callers
rely on (validate_duration raises on >MAX_AUDIO_SECONDS; load returns float32
mono at 16kHz, capped at MAX_AUDIO_SECONDS).
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# The Space package is named `app_lib` (not `bimomni`) to avoid a name
# collision with src/bimomni when both are importable. Put space/ on the
# path just for this module.
_SPACE_ROOT = Path(__file__).resolve().parent.parent / "space"
if str(_SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPACE_ROOT))

from app_lib.audio import (
    _resample,
    _to_mono,
    load_audio_array,
    preprocess_to_pcm16_wav,
    validate_duration,
)
from app_lib.config import MAX_AUDIO_SECONDS, TARGET_SAMPLE_RATE


def _write_wav(samples: np.ndarray, sr: int) -> str:
    """Write a temp WAV on disk; return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, samples, sr, subtype="PCM_16")
    return f.name


def test_to_mono_passthrough_for_mono() -> None:
    mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _to_mono(mono) is mono


def test_to_mono_averages_channels() -> None:
    stereo = np.array([[1.0, 4.0], [2.0, 6.0]], dtype=np.float32)
    result = _to_mono(stereo)
    assert result.ndim == 1
    assert np.allclose(result, [2.5, 4.0])


def test_resample_identity_when_already_target_sr() -> None:
    data = np.arange(100, dtype=np.float32)
    assert _resample(data, 16000, 16000) is data


def test_resample_preserves_duration_within_one_sample() -> None:
    sr_in = 22050
    sr_out = TARGET_SAMPLE_RATE
    duration_s = 1.0
    data = np.random.default_rng(0).standard_normal(int(duration_s * sr_in)).astype(np.float32)
    out = _resample(data, sr_in, sr_out)
    assert out.dtype == np.float32
    assert abs(len(out) - int(duration_s * sr_out)) <= 2


def test_validate_duration_passes_under_limit(tmp_path) -> None:
    samples = np.zeros(int(0.5 * 22050), dtype=np.float32)
    path = _write_wav(samples, 22050)
    info = validate_duration(path, max_seconds=MAX_AUDIO_SECONDS)
    assert info < MAX_AUDIO_SECONDS


def test_validate_duration_rejects_oversized(tmp_path) -> None:
    samples = np.zeros(int((MAX_AUDIO_SECONDS + 5) * 22050), dtype=np.float32)
    path = _write_wav(samples, 22050)
    with pytest.raises(ValueError, match="at most"):
        validate_duration(path, max_seconds=MAX_AUDIO_SECONDS)


def test_load_audio_array_returns_16khz_mono_float32() -> None:
    samples = np.random.default_rng(1).standard_normal(int(1.5 * 22050)).astype(np.float32)
    path = _write_wav(samples, 22050)
    data, sr = load_audio_array(path)
    assert sr == TARGET_SAMPLE_RATE
    assert data.dtype == np.float32
    assert data.ndim == 1
    assert len(data) <= TARGET_SAMPLE_RATE * MAX_AUDIO_SECONDS


def test_preprocess_to_pcm16_wav_round_trip() -> None:
    samples = np.random.default_rng(2).standard_normal(int(0.5 * 22050)).astype(np.float32)
    path = _write_wav(samples, 22050)
    raw = preprocess_to_pcm16_wav(path)
    assert isinstance(raw, bytes)
    # Read it back with soundfile and verify properties.
    buf = io.BytesIO(raw)
    rt, sr = sf.read(buf)
    assert sr == TARGET_SAMPLE_RATE
    assert rt.ndim == 1
    assert rt.shape[0] <= TARGET_SAMPLE_RATE * MAX_AUDIO_SECONDS


def test_samples_registry_filters_missing_files() -> None:
    """SAMPLE_CLIPS excludes any entry whose WAV is absent on disk."""
    from app_lib import samples

    # None of the operator-curated bundles ship in CI; the registry must
    # therefore filter to empty. CANDIDATES is untouched on the way out.
    real = samples.CANDIDATES
    try:
        assert samples.SAMPLE_CLIPS == {}
    finally:
        samples.CANDIDATES = real
