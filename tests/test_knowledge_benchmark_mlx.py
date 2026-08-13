"""Regression tests for the MLX Qwen3-Omni knowledge benchmark."""

from __future__ import annotations

import platform

import pytest

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - macOS-only
    pytest.skip("mlx is only available on Apple Silicon", allow_module_level=True)

from bimomni.evaluation.knowledge import Probe
from bimomni.inference.mlx import _score_probe_mlx

if platform.system() != "Darwin":  # pragma: no cover - macOS-only
    pytest.skip("mlx is only available on Apple Silicon", allow_module_level=True)


def test_score_probe_uses_qwen_omni_thinker_logits() -> None:
    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            return {"input_ids": self.encode(text)}

        def encode(self, text: str) -> list[int]:
            if text == "Question ":
                return [0, 1]
            return [0, 1, 2]

    class Thinker:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, input_ids: mx.array):
            self.calls += 1
            return type("Output", (), {"logits": mx.zeros((1, input_ids.shape[1], 3))})()

    thinker = Thinker()
    model = type("Model", (), {"thinker": thinker})()
    probe = Probe(
        id="probe",
        track="control",
        category="test",
        prompt="Question",
        choices=("a", "b", "c", "d"),
        answer_index=0,
        fact_id="probe",
        variant="canonical",
        source_group="test",
    )

    scores = _score_probe_mlx(model, Tokenizer(), probe)

    assert len(scores) == 4
    assert thinker.calls == 4
