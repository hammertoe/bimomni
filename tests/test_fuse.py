"""Tests for bimomni.publish.fuse: merging a PEFT LoRA into a base checkpoint.

The production path fuses the trained thinker-attention adapter into the full
Qwen3-Omni checkpoint on the pod. These tests exercise the same code with a
tiny torch model so the merge math is verified exactly, locally.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from bimomni.publish.fuse import (
    disable_talker_output,
    fuse_adapter,
    fuse_model,
    write_provenance,
)


class TinyConfig(PretrainedConfig):
    model_type = "tiny-fuse-test"


class TinyModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config: TinyConfig) -> None:
        super().__init__(config)
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x):  # pragma: no cover - unused by the tests
        return self.linear(x)


class TinyTalkerModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config: TinyConfig) -> None:
        super().__init__(config)
        self.linear = nn.Linear(4, 4, bias=False)
        self.talker = nn.Linear(4, 4, bias=False)
        config.enable_audio_output = True

    def disable_talker(self) -> None:
        del self.talker
        self.has_talker = False

    def forward(self, x):  # pragma: no cover - unused by the tests
        return self.linear(x)


def _trained_adapter_dir(tmp_path):
    from peft import LoraConfig, get_peft_model

    config = TinyConfig()
    model = TinyModel(config)
    with torch.no_grad():
        model.linear.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    peft = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=4, target_modules=["linear"])
    )
    lora_linear = peft.base_model.model.linear
    with torch.no_grad():
        lora_linear.lora_A["default"].weight.copy_(
            torch.tensor([[1.0, 0.0, -1.0, 0.5], [0.0, 2.0, 1.0, -1.0]])
        )
        lora_linear.lora_B["default"].weight.copy_(
            torch.tensor([[1.0, -1.0], [0.5, 0.25], [-2.0, 1.0], [0.0, 3.0]])
        )
    adapter_dir = tmp_path / "checkpoint-1"
    peft.save_pretrained(str(adapter_dir))
    base_dir = tmp_path / "base"
    model.save_pretrained(str(base_dir))
    return base_dir, adapter_dir


def test_fuse_model_merges_lora_math_exactly(tmp_path) -> None:
    _base_dir, adapter_dir = _trained_adapter_dir(tmp_path)
    model = TinyModel(TinyConfig())
    with torch.no_grad():
        model.linear.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))

    fused = fuse_model(model, adapter_dir)

    lora_a = torch.tensor([[1.0, 0.0, -1.0, 0.5], [0.0, 2.0, 1.0, -1.0]])
    lora_b = torch.tensor([[1.0, -1.0], [0.5, 0.25], [-2.0, 1.0], [0.0, 3.0]])
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 2.0 * lora_b @ lora_a
    assert torch.allclose(fused.linear.weight, expected, atol=1e-6)


def test_fuse_model_drops_lora_wrappers(tmp_path) -> None:
    _base_dir, adapter_dir = _trained_adapter_dir(tmp_path)
    fused = fuse_model(TinyModel(TinyConfig()), adapter_dir)

    assert type(fused.linear) is nn.Linear
    assert not any("lora_" in name for name, _ in fused.state_dict().items())


def test_fuse_adapter_writes_complete_checkpoint(tmp_path) -> None:
    base_dir, adapter_dir = _trained_adapter_dir(tmp_path)
    output_dir = tmp_path / "fused"

    def loader(path):
        config = TinyConfig.from_pretrained(str(path))
        model = TinyModel(config)
        with torch.no_grad():
            model.linear.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        return model

    fuse_adapter(base_dir, adapter_dir, output_dir, loader=loader)

    assert (output_dir / "config.json").exists()
    assert any(output_dir.glob("*.safetensors"))
    provenance = json.loads((output_dir / "fusion_provenance.json").read_text())
    assert provenance["base_model"] == str(base_dir)
    assert provenance["adapter"] == str(adapter_dir)


def test_write_provenance_records_sources(tmp_path) -> None:
    write_provenance(tmp_path, base_model="/data/hf/base", adapter="/data/checkpoints/checkpoint-500")
    provenance = json.loads((tmp_path / "fusion_provenance.json").read_text())
    assert provenance == {
        "base_model": "/data/hf/base",
        "adapter": "/data/checkpoints/checkpoint-500",
    }


def test_fuse_adapter_requires_existing_output_free(tmp_path) -> None:
    base_dir, adapter_dir = _trained_adapter_dir(tmp_path)
    output_dir = tmp_path / "fused"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        fuse_adapter(base_dir, adapter_dir, output_dir, loader=lambda p: TinyModel(TinyConfig()))


def test_disable_talker_output_removes_talker_and_flips_config() -> None:
    model = TinyTalkerModel(TinyConfig())
    assert hasattr(model, "talker")
    assert model.config.enable_audio_output is True

    disable_talker_output(model)

    assert not hasattr(model, "talker")
    assert model.config.enable_audio_output is False


def test_disable_talker_output_tolerates_model_without_talker(tmp_path) -> None:
    model = TinyModel(TinyConfig())
    disable_talker_output(model)  # must not raise
    assert model.config.enable_audio_output is False


def _load_state(output_dir) -> dict:
    import torch

    weights = output_dir / "model.safetensors"
    if weights.exists():
        from safetensors.torch import load_file

        return load_file(str(weights))
    return torch.load(output_dir / "pytorch_model.bin")


def test_fuse_adapter_drop_talker_saves_text_only_config(tmp_path) -> None:
    base_dir, adapter_dir = _trained_adapter_dir(tmp_path)
    output_dir = tmp_path / "fused-text-only"

    def loader(path):
        config = TinyConfig.from_pretrained(str(path))
        model = TinyTalkerModel(config)
        with torch.no_grad():
            model.linear.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        return model

    fuse_adapter(base_dir, adapter_dir, output_dir, loader=loader, drop_talker=True)

    config = json.loads((output_dir / "config.json").read_text())
    assert config["enable_audio_output"] is False
    state = _load_state(output_dir)
    assert "talker.weight" not in state
    assert "linear.weight" in state


def test_fuse_adapter_keeps_talker_by_default(tmp_path) -> None:
    base_dir, adapter_dir = _trained_adapter_dir(tmp_path)
    output_dir = tmp_path / "fused-with-talker"

    def loader(path):
        config = TinyConfig.from_pretrained(str(path))
        return TinyTalkerModel(config)

    fuse_adapter(base_dir, adapter_dir, output_dir, loader=loader)

    config = json.loads((output_dir / "config.json").read_text())
    assert config["enable_audio_output"] is True
    state = _load_state(output_dir)
    assert "talker.weight" in state
