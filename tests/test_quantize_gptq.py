"""Tests for the one-off Qwen3-Omni GPTQ publishing workflow."""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from bimomni.publish.quantize_gptq import (
    CalibrationConfig,
    LlmCompressorRuntime,
    QuantizationConfig,
    preprocess_audio_sample,
    quantize_checkpoint,
)


class FakeSavedObject:
    def __init__(self) -> None:
        self.saved_to = None

    def save_pretrained(self, output_dir, **kwargs) -> None:
        self.saved_to = (output_dir, kwargs)


class FakeModel(FakeSavedObject):
    def __init__(self) -> None:
        super().__init__()
        self.thinker = SimpleNamespace(visual=SimpleNamespace())
        self.config = SimpleNamespace(enable_audio_output=True)
        self.talker_disabled = False

    def disable_talker(self) -> None:
        self.talker_disabled = True


class FakeDataset:
    column_names = ["audio", "text"]

    def __init__(self) -> None:
        self.map_call = None

    def map(self, function, *, remove_columns):
        self.map_call = (function, remove_columns)
        return "prepared-dataset"


class FakeRuntime:
    model_class = object

    def __init__(self) -> None:
        self.model = FakeModel()
        self.processor = FakeSavedObject()
        self.dataset = FakeDataset()
        self.oneshot_call = None
        self.recipe_call = None
        self.modified = None
        self.dispatched = None
        self.uploaded = None
        self.model_loaded = False
        self.audio_decoding_disabled = False
        self.validated = None

    def load_context(self, model_class):
        assert model_class is self.model_class
        return nullcontext()

    def load_model(self, source, revision):
        self.model_loaded = True
        self.loaded_model = (source, revision)
        return self.model

    def load_processor(self, source, revision):
        self.loaded_processor = (source, revision)
        return self.processor

    def patch_visual(self, visual) -> None:
        self.patched_visual = visual

    def load_dataset(self, dataset_id, subset, split):
        self.loaded_dataset = (dataset_id, subset, split)
        return self.dataset

    def disable_audio_decoding(self, dataset):
        assert dataset is self.dataset
        self.audio_decoding_disabled = True
        return dataset

    def make_recipe(self, *, ignore, offload_hessians):
        self.recipe_call = (ignore, offload_hessians)
        return "gptq-recipe"

    def oneshot(self, **kwargs) -> None:
        self.oneshot_call = kwargs

    def validate_weight_scales(self, model) -> None:
        self.validated = model

    def make_image_collator(self):
        return "image-collator"

    def prepare_image_dataset(self, dataset_id, split, processor, max_sequence_length):
        self.prepared_image_dataset = (dataset_id, split, processor, max_sequence_length)
        return "prepared-image-dataset"

    def dispatch_model(self, model) -> None:
        self.dispatched = model

    def modify_save_pretrained(self, model) -> None:
        self.modified = model

    def upload_checkpoint(self, output_dir, repo_id, private) -> None:
        self.uploaded = (output_dir, repo_id, private)


class FakeTensor:
    def __init__(self, value) -> None:
        self.value = value

    def __getitem__(self, index):
        assert index == 0
        return self.value


class FakeProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is False
        return "rendered-chat"

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return {"input_ids": FakeTensor([1, 2]), "input_features": FakeTensor([3, 4])}


def test_preprocess_audio_sample_includes_audio_prompt_and_transcript() -> None:
    processor = FakeProcessor()
    sample = {
        "audio": {"array": [0.1, -0.1], "sampling_rate": 16_000},
        "text": "Bridgetown",
    }

    result = preprocess_audio_sample(sample, processor)

    assert processor.messages[-1]["role"] == "assistant"
    assert processor.messages[-1]["content"][0]["text"] == "Bridgetown"
    assert processor.inputs == {
        "text": "rendered-chat",
        "audio": [[0.1, -0.1]],
        "sampling_rate": 16_000,
        "return_tensors": "pt",
    }
    assert result == {"input_ids": [1, 2], "input_features": [3, 4]}


def test_preprocess_audio_sample_decodes_raw_audio_bytes(monkeypatch) -> None:
    processor = FakeProcessor()
    decoded = ([0.25, -0.25], 8_000)

    def fake_read(source, *, dtype):
        assert source.read() == b"flac-data"
        assert dtype == "float32"
        return decoded

    monkeypatch.setattr("soundfile.read", fake_read)

    preprocess_audio_sample(
        {"audio": {"bytes": b"flac-data", "path": None}, "text": "Oistins"},
        processor,
    )

    assert processor.inputs["audio"] == [[0.25, -0.25]]
    assert processor.inputs["sampling_rate"] == 8_000


@pytest.mark.parametrize("invalid_scale", [float("nan"), float("inf"), 0.0])
def test_validate_weight_scales_reports_invalid_modules(invalid_scale) -> None:
    model = torch.nn.Module()
    model.valid = torch.nn.Linear(2, 2)
    model.invalid = torch.nn.Linear(2, 2)
    model.valid.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones(1), requires_grad=False),
    )
    model.invalid.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.tensor([invalid_scale]), requires_grad=False),
    )
    runtime = LlmCompressorRuntime.__new__(LlmCompressorRuntime)

    with pytest.raises(ValueError, match=r"1 modules.*invalid"):
        runtime.validate_weight_scales(model)


def test_quantize_checkpoint_targets_thinker_and_saves_complete_model(tmp_path) -> None:
    runtime = FakeRuntime()
    output_dir = tmp_path / "quantized"
    config = QuantizationConfig(
        source="hammertoe/BimOmni-30B-A3B",
        revision="abc123",
        output_dir=output_dir,
        repo_id="hammertoe/BimOmni-30B-A3B-GPTQ-4bit",
        calibration=CalibrationConfig(samples=8, max_sequence_length=512),
        offload_hessians=True,
    )

    quantize_checkpoint(config, runtime=runtime)

    assert runtime.loaded_model == (config.source, config.revision)
    assert runtime.loaded_processor == (config.source, config.revision)
    assert runtime.model.talker_disabled is True
    assert runtime.model.config.enable_audio_output is False
    assert runtime.recipe_call == (
        ("lm_head", r"re:.*visual.*", r"re:.*audio_tower.*", r"re:.*code2wav.*"),
        True,
    )
    assert runtime.oneshot_call["model"] is runtime.model.thinker
    assert runtime.oneshot_call["processor"] is runtime.processor
    assert runtime.oneshot_call["dataset"] == "prepared-dataset"
    assert runtime.oneshot_call["recipe"] == "gptq-recipe"
    assert runtime.oneshot_call["num_calibration_samples"] == 8
    assert runtime.oneshot_call["max_seq_length"] == 512
    assert runtime.oneshot_call["sequential_targets"] == ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    assert runtime.validated is runtime.model.thinker
    assert runtime.modified is runtime.model
    assert runtime.model.saved_to == (str(output_dir), {"save_compressed": True})
    assert runtime.processor.saved_to == (str(output_dir), {})
    provenance = json.loads((output_dir / "quantization_provenance.json").read_text())
    assert provenance["source"] == config.source
    assert provenance["revision"] == config.revision
    assert provenance["algorithm"] == "GPTQ W4A16"
    assert provenance["calibration_samples"] == 8
    assert runtime.uploaded == (
        output_dir,
        "hammertoe/BimOmni-30B-A3B-GPTQ-4bit",
        True,
    )


def test_quantize_checkpoint_can_use_upstream_image_calibration(tmp_path) -> None:
    runtime = FakeRuntime()
    config = QuantizationConfig(
        source="model",
        output_dir=tmp_path / "quantized",
        calibration=CalibrationConfig(
            mode="image",
            dataset_id="flickr30k",
            subset=None,
            split="test",
            samples=8,
        ),
    )

    quantize_checkpoint(config, runtime=runtime)

    assert not hasattr(runtime, "loaded_dataset")
    assert runtime.prepared_image_dataset == ("flickr30k", "test[:8]", runtime.processor, 2048)
    assert runtime.oneshot_call["dataset"] == "prepared-image-dataset"
    assert runtime.oneshot_call["data_collator"] == "image-collator"


def test_quantize_checkpoint_refuses_existing_output(tmp_path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    config = QuantizationConfig(source="model", output_dir=output_dir)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        quantize_checkpoint(config, runtime=FakeRuntime())


def test_quantize_checkpoint_preflight_stops_before_model_load(tmp_path) -> None:
    runtime = FakeRuntime()
    config = QuantizationConfig(
        source="model",
        output_dir=tmp_path / "unused",
        calibration=CalibrationConfig(samples=1),
        preflight=True,
    )

    quantize_checkpoint(config, runtime=runtime)

    assert runtime.dataset.map_call is not None
    assert runtime.audio_decoding_disabled is True
    assert runtime.model_loaded is False
    assert not config.output_dir.exists()
