"""Tests for bimomni.publish.strip_talker: filtering non-thinker weights for MLX."""

from __future__ import annotations

from pathlib import Path

import pytest

from bimomni.publish.strip_talker import (
    is_talker_weight,
    filter_thinker_weights,
    rewrite_mlx_config,
    strip_mlx_safetensors,
    KNOWN_NON_TALKER_PREFIXES,
)


def test_rewrite_mlx_config_backfills_rope_theta_and_num_experts(tmp_path: Path) -> None:
    """transformers-5.x rope_parameters / num_local_experts -> mlx_vlm fields."""
    import json

    config = {
        "thinker_config": {
            "text_config": {
                "num_experts": 128,
                "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
            }
        },
        "talker_config": {
            "text_config": {
                "num_local_experts": 128,
                "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
            }
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    rewrite_mlx_config(path)

    rewritten = json.loads(path.read_text())
    thinker = rewritten["thinker_config"]["text_config"]
    talker = rewritten["talker_config"]["text_config"]
    assert thinker["rope_theta"] == 1_000_000
    assert thinker["rope_scaling"] == {"type": "default"}
    assert "rope_parameters" not in thinker
    assert talker["num_experts"] == 128
    assert "num_local_experts" not in talker
    assert talker["rope_theta"] == 1_000_000


def test_rewrite_mlx_config_parses_in_mlx_vlm(tmp_path: Path) -> None:
    """The rewritten config must satisfy mlx_vlm's ModelConfig.from_dict."""
    import json
    import shutil

    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.qwen3_omni_moe import config as mlx_config

    config = {
        "architectures": ["Qwen3Omni"],
        "model_type": "qwen3_omni_moe",
        "enable_audio_output": False,
        "thinker_config": {
            "model_type": "qwen3_omni_moe_thinker",
            "text_config": {
                "num_hidden_layers": 48,
                "hidden_size": 2048,
                "intermediate_size": 3584,
                "num_attention_heads": 16,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [0, 1],
                "moe_intermediate_size": 512,
                "rms_norm_eps": 1e-6,
                "vocab_size": 151936,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 131072,
                "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
            },
            "vision_config": {"hidden_size": 1024, "image_size": 448, "patch_size": 14},
            "audio_config": {"hidden_size": 768},
        },
        "talker_config": {
            "model_type": "qwen3_omni_moe_talker",
            "text_config": {
                "num_hidden_layers": 18,
                "hidden_size": 1024,
                "intermediate_size": 1792,
                "num_attention_heads": 16,
                "num_local_experts": 128,
                "num_experts_per_tok": 6,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "moe_intermediate_size": 384,
                "rms_norm_eps": 1e-6,
                "vocab_size": 3072,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "max_position_embeddings": 16384,
                "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
            },
            "code_predictor_config": {},
        },
        "code2wav_config": {},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    rewrite_mlx_config(path)

    parsed = mlx_config.ModelConfig.from_dict(json.loads(path.read_text()))
    assert parsed.thinker_config.text_config.rope_theta == 1_000_000
    assert parsed.talker_config.text_config.num_experts == 128
    assert parsed.enable_audio_output is False


def test_is_talker_weight_identifies_multimodal_heads() -> None:
    assert is_talker_weight("talker.model.embed_tokens.weight")
    assert is_talker_weight("code2wav.embedding.weight")
    assert is_talker_weight("audio_tower.audio_encoder.blocks.0.weight")
    assert is_talker_weight("visual.patch_embed.weight")


def test_is_talker_weight_keeps_input_towers_when_keep_inputs_true() -> None:
    """keep_inputs preserves audio_tower and visual so ingestion still works."""
    assert not is_talker_weight("audio_tower.layers.0.weight", keep_inputs=True)
    assert not is_talker_weight("audio_tower.audio_encoder.blocks.0.weight", keep_inputs=True)
    assert not is_talker_weight("visual.patch_embed.weight", keep_inputs=True)
    assert is_talker_weight("talker.model.embed_tokens.weight", keep_inputs=True)
    assert is_talker_weight("code2wav.embedding.weight", keep_inputs=True)
    assert is_talker_weight("talking_head.bias", keep_inputs=True)


def test_is_talker_weight_keeps_thinker() -> None:
    assert not is_talker_weight("model.layers.0.self_attn.q_proj.weight")
    assert not is_talker_weight("model.embed_tokens.weight")
    assert not is_talker_weight("lm_head.weight")


def test_filter_thinker_weights_drops_talker_names() -> None:
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "talker.model.embed_tokens.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "visual.patch_embed.weight",
        "lm_head.weight",
    ]
    assert filter_thinker_weights(names) == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "lm_head.weight",
    ]


def test_filter_thinker_weights_keeps_known_prefixes() -> None:
    names = [p + ".weight" for p in KNOWN_NON_TALKER_PREFIXES]
    filtered = filter_thinker_weights(names)
    assert len(filtered) == len(names)


def test_strip_mlx_safetensors_drops_talker_weights(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        "talker.model.embed_tokens.weight": torch.ones(2, 2),
        "lm_head.weight": torch.full((2, 2), 2.0),
    }
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))

    kept, dropped = strip_mlx_safetensors(tmp_path)

    assert kept == 2
    assert dropped == 1
    from safetensors.torch import load_file

    remaining = load_file(str(weights))
    assert set(remaining) == {
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    }


def test_strip_mlx_safetensors_handles_sharded_index(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
            "talker.model.embed_tokens.weight": torch.ones(2, 2),
        },
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"lm_head.weight": torch.full((2, 2), 2.0)},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {'
        '"model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors", '
        '"talker.model.embed_tokens.weight": "model-00001-of-00002.safetensors", '
        '"lm_head.weight": "model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )

    kept, dropped = strip_mlx_safetensors(tmp_path)

    assert kept == 2
    assert dropped == 1
    from safetensors.torch import load_file

    assert set(load_file(str(tmp_path / "model-00001-of-00002.safetensors"))) == {
        "model.layers.0.self_attn.q_proj.weight"
    }


def test_strip_mlx_safetensors_keeps_input_towers_when_keep_inputs_true(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        "audio_tower.layers.0.weight": torch.ones(2, 2),
        "visual.patch_embed.weight": torch.ones(2, 2),
        "talker.model.embed_tokens.weight": torch.ones(2, 2),
        "code2wav.embedding.weight": torch.ones(2, 2),
        "talking_head.bias": torch.ones(2, 2),
    }
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))

    kept, dropped = strip_mlx_safetensors(tmp_path, keep_inputs=True)

    assert kept == 3
    assert dropped == 3
    remaining = load_file(str(weights))
    assert set(remaining) == {
        "model.layers.0.self_attn.q_proj.weight",
        "audio_tower.layers.0.weight",
        "visual.patch_embed.weight",
    }
