"""Quantize a talker-free Qwen3-Omni checkpoint with GPTQ W4A16.

This follows llm-compressor's Qwen3-Omni path and quantizes the Thinker while
leaving the audio and visual input towers in their original precision. Run the
command once for the pinned base checkpoint and once for the fused BimOmni
checkpoint so the demo compares equivalent compressed model layouts.
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "MLCommons/peoples_speech"
DEFAULT_DATASET_SUBSET = "test"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_IGNORE = (
    "lm_head",
    r"re:.*visual.*",
    r"re:.*audio_tower.*",
    r"re:.*code2wav.*",
)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Audio calibration dataset and sampling limits."""

    mode: str = "audio"
    dataset_id: str = DEFAULT_DATASET
    subset: str | None = DEFAULT_DATASET_SUBSET
    split: str = DEFAULT_DATASET_SPLIT
    samples: int = 512
    max_sequence_length: int = 2048


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Inputs for one deterministic GPTQ checkpoint conversion."""

    source: str
    output_dir: Path
    revision: str | None = None
    repo_id: str | None = None
    private: bool = True
    preflight: bool = False
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    offload_hessians: bool = False


class LlmCompressorRuntime:
    """Lazy imports and adapters for the CUDA-only quantization stack."""

    def __init__(self) -> None:
        from transformers import Qwen3OmniMoeForConditionalGeneration

        self.model_class = Qwen3OmniMoeForConditionalGeneration

    def load_context(self, model_class):
        from llmcompressor.utils import load_context

        return load_context(model_class)

    def load_model(self, source: str, revision: str | None):
        kwargs = {"revision": revision} if revision else {}
        return self.model_class.from_pretrained(source, **kwargs)

    def load_processor(self, source: str, revision: str | None):
        from transformers import AutoProcessor

        kwargs = {"revision": revision} if revision else {}
        return AutoProcessor.from_pretrained(source, **kwargs)

    def patch_visual(self, visual) -> None:
        from llmcompressor.modeling.patch.qwen3_omni_patch import (
            fast_pos_embed_interpolate,
        )

        visual.fast_pos_embed_interpolate = fast_pos_embed_interpolate.__get__(visual)

    def load_dataset(self, dataset_id: str, subset: str | None, split: str):
        from datasets import load_dataset

        return load_dataset(dataset_id, subset, split=split)

    def disable_audio_decoding(self, dataset):
        from datasets import Audio

        return dataset.cast_column("audio", Audio(decode=False))

    def make_recipe(self, *, ignore: tuple[str, ...], offload_hessians: bool):
        from llmcompressor.modifiers.gptq import GPTQModifier

        return GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=list(ignore),
            offload_hessians=offload_hessians,
        )

    def oneshot(self, **kwargs) -> None:
        from llmcompressor import oneshot

        oneshot(**kwargs)

    def validate_weight_scales(self, model) -> None:
        import torch

        invalid = []
        for name, module in model.named_modules():
            scale = getattr(module, "weight_scale", None)
            if scale is None:
                continue
            if scale.device.type == "meta" or not torch.all(torch.isfinite(scale) & (scale > 0)):
                invalid.append(name or "<root>")

        if invalid:
            examples = ", ".join(invalid[:10])
            raise ValueError(
                f"GPTQ produced invalid weight_scale values in {len(invalid)} "
                f"modules; first affected modules: {examples}"
            )

    def make_image_collator(self):
        from transformers import default_data_collator

        def data_collator(features):
            batch = default_data_collator(features)
            batch["image_grid_thw"] = batch["image_grid_thw"].squeeze(0)
            return batch

        return data_collator

    def prepare_image_dataset(
        self,
        dataset_id: str,
        split: str,
        processor: Any,
        max_sequence_length: int,
    ):
        from llmcompressor.args import DatasetArguments
        from llmcompressor.transformers.data import TextGenerationDataset

        dataset_args = DatasetArguments(
            dataset=dataset_id,
            max_seq_length=max_sequence_length,
        )
        manager = TextGenerationDataset.load_from_registry(
            dataset_id,
            dataset_args=dataset_args,
            split=split,
            processor=processor,
        )
        return manager(add_labels=False)

    def dispatch_model(self, model) -> None:
        from compressed_tensors.offload import dispatch_model

        dispatch_model(model)

    def modify_save_pretrained(self, model) -> None:
        from llmcompressor.transformers.compression.compressed_tensors_utils import (
            modify_save_pretrained,
        )

        modify_save_pretrained(model)

    def upload_checkpoint(self, output_dir: Path, repo_id: str, private: bool) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=output_dir,
            commit_message="Publish Qwen3-Omni GPTQ W4A16 checkpoint",
        )


def preprocess_audio_sample(sample: dict[str, Any], processor: Any) -> dict[str, Any]:
    """Convert one audio/transcript sample into Qwen3-Omni model inputs."""
    audio = sample["audio"]
    if "array" in audio:
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
    else:
        import soundfile as sf

        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
        audio_array, sampling_rate = sf.read(source, dtype="float32")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": "Transcribe this audio verbatim."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text"]}],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(
        text=prompt,
        audio=[audio_array],
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    return {name: value[0] for name, value in inputs.items()}


def quantize_checkpoint(
    config: QuantizationConfig,
    *,
    runtime: Any | None = None,
) -> None:
    """Quantize one checkpoint and save a self-contained compressed model."""
    if config.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing {config.output_dir}")
    if config.calibration.samples < 1:
        raise ValueError("calibration samples must be at least 1")

    runtime = runtime or LlmCompressorRuntime()
    processor = runtime.load_processor(config.source, config.revision)

    calibration = config.calibration
    split = f"{calibration.split}[:{calibration.samples}]"
    if calibration.mode == "audio":
        dataset = runtime.load_dataset(calibration.dataset_id, calibration.subset, split)
        dataset = runtime.disable_audio_decoding(dataset)
        dataset = dataset.map(
            lambda sample: preprocess_audio_sample(sample, processor),
            remove_columns=dataset.column_names,
        )
        calibration_kwargs = {"dataset": dataset}
    elif calibration.mode == "image":
        dataset = runtime.prepare_image_dataset(
            calibration.dataset_id,
            split,
            processor,
            calibration.max_sequence_length,
        )
        calibration_kwargs = {
            "dataset": dataset,
            "data_collator": runtime.make_image_collator(),
        }
    else:
        raise ValueError(f"unsupported calibration mode: {calibration.mode}")
    if config.preflight:
        print("[quantize] calibration preflight passed", flush=True)
        return

    # Prepare calibration before loading tens of gigabytes of model weights so
    # dataset or codec failures fail cheaply.
    with runtime.load_context(runtime.model_class):
        model = runtime.load_model(config.source, config.revision)
    disable_talker = getattr(model, "disable_talker", None)
    if callable(disable_talker):
        disable_talker()
    model.config.enable_audio_output = False
    runtime.patch_visual(model.thinker.visual)

    recipe = runtime.make_recipe(
        ignore=DEFAULT_IGNORE,
        offload_hessians=config.offload_hessians,
    )
    runtime.oneshot(
        model=model.thinker,
        processor=processor,
        **calibration_kwargs,
        recipe=recipe,
        batch_size=1,
        max_seq_length=calibration.max_sequence_length,
        num_calibration_samples=calibration.samples,
        moe_calibrate_all_experts=True,
    )
    runtime.validate_weight_scales(model.thinker)

    runtime.dispatch_model(model)
    runtime.modify_save_pretrained(model)
    config.output_dir.mkdir(parents=True)
    model.save_pretrained(str(config.output_dir), save_compressed=True)
    processor.save_pretrained(str(config.output_dir))
    provenance = {
        "source": config.source,
        "revision": config.revision,
        "algorithm": "GPTQ W4A16",
        "ignored_modules": list(DEFAULT_IGNORE),
        "calibration_samples": calibration.samples,
        "calibration": asdict(calibration),
        "offload_hessians": config.offload_hessians,
    }
    (config.output_dir / "quantization_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    if config.repo_id:
        runtime.upload_checkpoint(config.output_dir, config.repo_id, config.private)


def main(argv: list[str] | None = None) -> int:
    """Run the one-off GPTQ quantization command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="checkpoint path or Hub model id")
    parser.add_argument("--revision", help="optional pinned Hub revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", help="optional Hub model repo to create and upload")
    parser.add_argument(
        "--public",
        action="store_true",
        help="make a newly created --repo-id public (default: private)",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-subset", default=DEFAULT_DATASET_SUBSET)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--calibration-mode", choices=("audio", "image"), default="audio")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--offload-hessians", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="prepare calibration data, then exit before loading model weights",
    )
    args = parser.parse_args(argv)

    config = QuantizationConfig(
        source=args.source,
        revision=args.revision,
        output_dir=args.output,
        repo_id=args.repo_id,
        private=not args.public,
        preflight=args.preflight,
        calibration=CalibrationConfig(
            mode=args.calibration_mode,
            dataset_id=args.dataset,
            subset=args.dataset_subset or None,
            split=args.dataset_split,
            samples=args.samples,
            max_sequence_length=args.max_sequence_length,
        ),
        offload_hessians=args.offload_hessians,
    )
    quantize_checkpoint(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
