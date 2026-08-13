"""Fuse a trained PEFT LoRA adapter into the full base checkpoint.

Production use: merge the thinker-attention adapter into the complete
Qwen3-Omni model on the pod, producing a standard Hugging Face checkpoint.
The fused checkpoint converts cleanly with `mlx_vlm.convert` because its key
layout matches the untouched base model (mlx_vlm's `sanitize()` remaps
`thinker.model.*` itself).

Runs on CPU: the merged model is bf16 (~61 GB), well inside the pod's RAM.

Pass `drop_talker=True` to also drop the speech-output towers: the talker and
code2wav modules are deleted and `enable_audio_output` is flipped off in the
saved config, so the checkpoint loads without missing-weight warnings. Input
towers (audio_tower, visual) are always preserved by the merge itself; they
live inside the thinker and are untouched by the LoRA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimomni.training.train import MODEL_ID, find_latest_checkpoint

DEFAULT_CHECKPOINT_DIR = "/data/checkpoints"


def fuse_model(model, adapter_dir) -> object:
    """Attach the adapter to the base model and merge LoRA weights in place."""
    from peft import PeftModel

    adapted = PeftModel.from_pretrained(model, str(adapter_dir))
    return adapted.merge_and_unload()


def write_provenance(output_dir, base_model: str, adapter: str) -> None:
    """Record which base checkpoint and adapter produced this fusion."""
    provenance = {"base_model": base_model, "adapter": adapter}
    Path(output_dir, "fusion_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def disable_talker_output(model) -> None:
    """Drop the speech-output towers and mark the model text-only.

    Deletes the talker/code2wav modules (via `model.disable_talker()` when
    available) and flips `enable_audio_output` off in the config so the saved
    checkpoint reloads without any missing-weight warnings. Input towers
    (audio_tower, visual) live inside the thinker and are unaffected.
    """
    disable = getattr(model, "disable_talker", None)
    if callable(disable):
        disable()
    config = getattr(model, "config", None)
    if config is not None:
        try:
            config.enable_audio_output = False
        except (AttributeError, TypeError):
            pass


def _load_omni(base_dir):
    """Load the full Qwen3-Omni model in bf16 on CPU for merging."""
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration

    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(base_dir), torch_dtype=torch.bfloat16, device_map="cpu"
    )


def fuse_adapter(
    base_dir,
    adapter_dir,
    output_dir,
    loader=None,
    save_processor: bool = True,
    drop_talker: bool = False,
) -> None:
    """Merge adapter into base and write a complete HF checkpoint.

    `loader` loads the base model from `base_dir`; defaults to the Qwen3-Omni
    bf16 CPU loader. The output directory must not already exist so a fused
    checkpoint is never silently overwritten. With `drop_talker=True` the
    speech-output towers are removed from the merged model before saving.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing {output_dir}")
    if loader is None:
        loader = _load_omni
    model = loader(base_dir)
    fused = fuse_model(model, adapter_dir)
    if drop_talker:
        disable_talker_output(fused)
    fused.save_pretrained(str(output_dir))
    if save_processor:
        try:
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained(str(base_dir)).save_pretrained(str(output_dir))
        except Exception as exc:  # processor optional for text-only conversion
            print(f"[fuse] processor save skipped: {exc}", flush=True)
    write_provenance(output_dir, str(base_dir), str(adapter_dir))
    print(f"[fuse] wrote fused checkpoint to {output_dir}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=MODEL_ID, help="base model path or Hub id")
    parser.add_argument(
        "--adapter",
        default=None,
        help="adapter checkpoint dir (default: latest under --checkpoint-dir)",
    )
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", required=True, help="output dir for the fused model")
    parser.add_argument(
        "--drop-talker",
        action="store_true",
        help="drop speech-output towers (talker/code2wav) before saving",
    )
    args = parser.parse_args(argv)

    adapter = args.adapter or find_latest_checkpoint(args.checkpoint_dir)
    if adapter is None:
        parser.error(f"no checkpoint found under {args.checkpoint_dir}")
    print(
        f"[fuse] base={args.base} adapter={adapter} output={args.output} "
        f"drop_talker={args.drop_talker}",
        flush=True,
    )
    fuse_adapter(args.base, adapter, args.output, drop_talker=args.drop_talker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
