"""Strip Qwen3-Omni output-side weights from an MLX snapshot.

Qwen3-Omni has separate input and output towers beyond the thinker:

- `thinker.*` — text reasoning, kept.
- `audio_tower.*` — encodes incoming audio into the thinker; keep if you
  want to ingest audio.
- `visual.*` — encodes incoming images; keep if you want to ingest images.
- `lm_head` — text output; keep.
- `talker.*`, `code2wav.*`, `talking_head.*` — generate outgoing speech;
  drop for text-only output.

`strip_mlx_safetensors` defaults to text-only output (drops everything
non-thinker). Pass `keep_inputs=True` to preserve `audio_tower` and `visual`
weights so audio/image ingestion still works.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_TOWER_PREFIXES = ("talker.", "code2wav.", "talking_head.")
INPUT_TOWER_PREFIXES = ("audio_tower.", "visual.")
ALL_DROPPABLE_PREFIXES = OUTPUT_TOWER_PREFIXES + INPUT_TOWER_PREFIXES
KNOWN_NON_TALKER_PREFIXES = (
    "model.",
    "lm_head.",
    "qwen3_omni_headscale.",
    "qwen3_omni_padscale.",
    "qwen3_omni_last_hidden_state_scale.",
    "qwen3_omni_last_hidden_state_padscale.",
)


def _fix_text_config(text: dict) -> None:
    """Repair a Qwen3-Moe text config for mlx_vlm 0.6.5 in place.

    transformers 5.x serializes rope settings under `rope_parameters`
    (with `rope_theta` nested inside) instead of a flat `rope_theta`, and
    uses `num_local_experts` where mlx_vlm expects `num_experts`. mlx_vlm's
    `TextConfig` requires a positional `rope_theta` and `num_experts`, so the
    fused transformers config must be rewritten before conversion.
    """
    rope = text.pop("rope_parameters", None)
    if isinstance(rope, dict):
        if "rope_theta" in rope and "rope_theta" not in text:
            text["rope_theta"] = rope["rope_theta"]
        text.setdefault("rope_scaling", {"type": "default"})
    if "num_experts" not in text and "num_local_experts" in text:
        text["num_experts"] = text.pop("num_local_experts")


def rewrite_mlx_config(config_path: Path) -> None:
    """Rewrite a transformers 5.x Qwen3-Omni config.json for mlx_vlm 0.6.5.

    Applies `_fix_text_config` to the thinker and talker sub-configs in place
    and rewrites the file. Other required mlx_vlm fields (vision/audio/codec)
    match the transformers layout already.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for section in ("thinker_config", "talker_config"):
        sub = config.get(section)
        if isinstance(sub, dict) and isinstance(sub.get("text_config"), dict):
            _fix_text_config(sub["text_config"])
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def is_talker_weight(name: str, keep_inputs: bool = False) -> bool:
    """True for weights that belong to non-thinker towers.

    With `keep_inputs=False` (default), input towers (audio_tower, visual)
    are also dropped (text-only MLX). Set `keep_inputs=True` to keep them
    so the model can still ingest audio/images.
    """
    if name.startswith(KNOWN_NON_TALKER_PREFIXES):
        return False
    if name.startswith(OUTPUT_TOWER_PREFIXES):
        return True
    if not keep_inputs and name.startswith(INPUT_TOWER_PREFIXES):
        return True
    return False


def filter_thinker_weights(names: list[str], keep_inputs: bool = False) -> list[str]:
    """Return only the thinker weights; optionally keep input towers."""
    return [name for name in names if not is_talker_weight(name, keep_inputs=keep_inputs)]


def strip_mlx_safetensors(mlx_path: Path, keep_inputs: bool = False) -> tuple[int, int]:
    """Rewrite the safetensors in an MLX snapshot, dropping output towers.

    Handles both single-file (model.safetensors) and sharded
    (model-00001-of-0000N.safetensors + model.safetensors.index.json) layouts.
    Pass `keep_inputs=True` to preserve audio_tower and visual weights for
    audio/image ingestion. Returns (kept, dropped) weight counts.
    """
    index_path = mlx_path / "model.safetensors.index.json"
    if index_path.exists():
        return _strip_sharded(mlx_path, index_path, keep_inputs=keep_inputs)
    single = mlx_path / "model.safetensors"
    if single.exists():
        return _strip_single(single, keep_inputs=keep_inputs)
    raise FileNotFoundError(f"no safetensors found under {mlx_path}")


def _strip_single(single: Path, keep_inputs: bool = False) -> tuple[int, int]:

    from safetensors.torch import load_file, save_file

    tensors = load_file(str(single))
    kept, dropped = {}, 0
    for name, tensor in tensors.items():
        if is_talker_weight(name, keep_inputs=keep_inputs):
            dropped += 1
        else:
            kept[name] = tensor
    if dropped:
        save_file(kept, str(single))
    return len(kept), dropped


def _strip_sharded(mlx_path: Path, index_path: Path, keep_inputs: bool = False) -> tuple[int, int]:
    import torch  # noqa: F401  (tensor dtype types used by safetensors)
    from safetensors.torch import load_file, save_file

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    dropped = 0
    for filename in sorted({str(v) for v in weight_map.values()}):
        shard = mlx_path / filename
        tensors = load_file(str(shard))
        kept = {
            name: tensor
            for name, tensor in tensors.items()
            if not is_talker_weight(name, keep_inputs=keep_inputs)
        }
        dropped += len(tensors) - len(kept)
        if len(kept) != len(tensors):
            if kept:
                save_file(kept, str(shard))
            else:
                shard.unlink()
        for name in list(tensors):
            if name not in kept:
                del weight_map[name]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return len(weight_map), dropped
