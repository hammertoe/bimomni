"""Tests for bimomni.publish.upload: model card rendering and PEFT metadata."""

from __future__ import annotations


from bimomni.publish.upload import (
    BaseModelInfo,
    build_model_card,
    build_fused_card,
    build_mlx_card,
    FusedModelInfo,
    MODEL_REPO,
)


def test_build_model_card_contains_required_sections() -> None:
    info = BaseModelInfo(
        base_model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        base_revision="abc123",
        record_count=100_000,
        token_estimate=40_000_000,
        hyperparameters={"lora_rank": 64, "learning_rate": 1e-4},
        budget_hours=12.0,
        adapter_repo=MODEL_REPO,
    )
    card = build_model_card(info)

    assert "Qwen/Qwen3-Omni-30B-A3B-Instruct" in card
    assert "abc123" in card
    assert "100,000" in card
    assert "40,000,000" in card
    assert "--q-bits 4 --q-group-size 64" in card
    assert "--skip-talker" in card
    assert "talker" in card.lower()
    assert "text-only" in card.lower()
    assert "Prime Intellect" in card
    assert "hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v3" in card
    assert "12" in card


def test_build_fused_card_mentions_tower_handling() -> None:
    info = FusedModelInfo(
        base_model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        base_revision="rev123",
        adapter_repo=MODEL_REPO,
    )
    card = build_fused_card(info)

    assert "rev123" in card
    assert "enable_audio_output" in card
    assert "audio_tower" in card
    assert "visual" in card
    assert "talker" in card
    assert "dropped" in card.lower()


def test_build_mlx_card_mentions_4bit_and_usage() -> None:
    info = FusedModelInfo(
        base_model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        base_revision="rev123",
        adapter_repo=MODEL_REPO,
    )
    card = build_mlx_card(info)

    assert "--q-bits 4 --q-group-size 64" in card
    assert "mlx_lm.generate" in card
    assert "4bit" in card.lower()
    assert "rev123" in card
