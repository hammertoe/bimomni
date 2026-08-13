"""Upload the trained adapter to the Hugging Face Hub.

Asserts the adapter loads cleanly via PEFT, then pushes only the adapter
artefacts (adapter_config.json, adapter_model.safetensors, tokenizer,
chat template, and model card). Full optimizer state, scheduler, RNG, and
runs/ stay in the private HF Bucket; they must never be uploaded as part
of the public adapter deliverable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REPO = "hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4"


@dataclass(frozen=True)
class BaseModelInfo:
    base_model: str = MODEL_ID
    base_revision: str = ""
    record_count: int = 0
    token_estimate: int = 0
    hyperparameters: dict = field(default_factory=dict)
    budget_hours: float = 12.0
    adapter_repo: str = MODEL_REPO


def build_model_card(info: BaseModelInfo) -> str:
    """Render the markdown model card for the adapter repo."""
    hparams = "\n".join(
        f"| {key} | {value} |" for key, value in sorted(info.hyperparameters.items())
    )
    return f"""---
base_model: Qwen/Qwen3-Omni-30B-A3B-Instruct
base_model_revision: {info.base_revision}
library_name: peft
pipeline_tag: text-generation
tags:
  - qwen
  - omni
  - lora
  - dapt
  - barbados
---

# Qwen3-Omni Barbados LoRA

Domain-adaptive pretraining (DAPT) adapter for `{info.base_model}` on the
Barbados newspaper corpus. Text-only: the audio talker head is disabled and
never trained.

## Adapter

| Property | Value |
| --- | --- |
| Base revision | `{info.base_revision}` |
| Corpus source | Prime Intellect persistent disk (private; not redistributed) |
| Records after dedup | {info.record_count:,} |
| Token estimate | {info.token_estimate:,} |
| Budget cap | {info.budget_hours} GPU-hours |

## Hyperparameters

| Parameter | Value |
| --- | --- |
{hparams}

## Known limitations

- Text-only. The talker (audio output) head is disabled via
  `model.disable_talker()`; this adapter never supervises speech targets.
- The corpus is not published; only this adapter (and the tokenizer files
  needed to run it) is on the Hub.

## Local quantization (Apple Silicon)

```bash
# 1. Convert base + adapter to MLX, stripping talker weights
mlx_lm.convert \\
    --hf-model {info.base_model} --revision {info.base_revision} \\
    --mlx-path ./mlx/qwen3-omni-barbados-bf16 \\
    --skip-talker

# 2. Fuse adapter into the converted model
python bimomni/strip_talker.py \\
    --mlx-path ./mlx/qwen3-omni-barbados-bf16 \\
    --adapter-repo {info.adapter_repo} \\
    --out ./mlx/qwen3-omni-barbados-fused

# 3. Quantize to 4 bits
mlx_lm.quantize \\
    --mlx-path ./mlx/qwen3-omni-barbados-fused \\
    --quantize --q-bits 4 --q-group-size 64 \\
    --out ./mlx/qwen3-omni-barbados-4bit

# 4. Smoke test
mlx_lm.generate \\
    --model ./mlx/qwen3-omni-barbados-4bit \\
    --prompt "The Crop Over festival in Barbados" \\
    --max-tokens 100
```

Flag values: `--quantize --q-bits 4 --q-group-size 64`.
"""


def assert_adapter_loads(adapter_dir: Path) -> None:
    """Fail fast if the adapter is not a loadable PEFT LoRA checkpoint."""
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"missing {adapter_config}")
    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise ValueError(f"expected a LORA adapter, got {config.get('peft_type')!r}")
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists():
        raise FileNotFoundError(f"missing {adapter_weights}")


ADAPTER_IGNORE_PATTERNS: tuple[str, ...] = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*",
    "*.pth",
    "runs/",
    "logging.jsonl",
    "args.json",
    "trainer_state.json",
    "trainer_state*",
    "_COMPLETE.json",
)


def upload(
    adapter_dir: Path,
    info: BaseModelInfo,
    repo_id: str = MODEL_REPO,
    token: str | None = None,
) -> str:
    """Push adapter artefacts + model card to the Hub. Returns the repo URL."""
    assert_adapter_loads(adapter_dir)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    # Write the model card FIRST so upload_folder sees the corrected metadata,
    # not whatever stale README.md the adapter dir shipped with.
    card_path = adapter_dir / "README.md"
    card_path.write_text(build_model_card(info), encoding="utf-8")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(adapter_dir),
        repo_type="model",
        ignore_patterns=list(ADAPTER_IGNORE_PATTERNS),
    )
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_type="model",
    )
    return f"https://huggingface.co/{repo_id}"


@dataclass(frozen=True)
class FusedModelInfo:
    base_model: str = MODEL_ID
    base_revision: str = ""
    adapter_repo: str = ""
    drop_talker: bool = True
    keep_inputs: bool = True


def build_fused_card(info: FusedModelInfo) -> str:
    """Render the markdown model card for the fused bf16 repo."""
    return f"""---
base_model: {info.base_model}
base_model_revision: {info.base_revision}
adapter_repo: {info.adapter_repo}
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen
  - omni
  - dapt
  - barbados
  - fused
---

# Qwen3-Omni Barbados Fused bf16

Base model with the {info.adapter_repo} DAPT LoRA merged and unloaded.

## What is kept and dropped

| Tower | Weights | Status |
| --- | --- | --- |
| thinker (text) | `thinker.*` | fused with the LoRA |
| audio input | `thinker.audio_tower.*` | preserved |
| visual input | `thinker.visual.*` | preserved |
| text output | `thinker.lm_head.*` | preserved |
| speech output | `talker.*`, `code2wav.*` | dropped (`enable_audio_output=false`) |

The model is text-only for output: `model.disable_talker()` was called before
saving, so `enable_audio_output` is `false` and there are no missing-weight
warnings on load. Audio and image ingestion still works.

## Provenance

- Base: `{info.base_model}` @ `{info.base_revision}`
- Adapter: `{info.adapter_repo}`
- Output: bf16 safetensors shards + `model.safetensors.index.json`
"""


def build_mlx_card(info: FusedModelInfo) -> str:
    """Render the markdown model card for the 4-bit MLX repo."""
    return f"""---
base_model: {info.base_model}
base_model_revision: {info.base_revision}
fused_repo: hammertoe/BimOmni-30B-A3B
adapter_repo: {info.adapter_repo}
library_name: mlx
pipeline_tag: text-generation
tags:
  - qwen
  - omni
  - dapt
  - barbados
  - mlx
  - 4bit
---

# Qwen3-Omni Barbados MLX 4-bit

MLX snapshot of the fused bf16 checkpoint, quantized to 4 bits
(`--quantize --q-bits 4 --q-group-size 64`). Same tower treatment as the
fused source: text-only output, audio/visual input towers kept.

## Usage

```bash
mlx_lm.generate \
    --model hammertoe/BimOmni-30B-A3B-MLX-4bit \
    --prompt "The Crop Over festival in Barbados" \
    --max-tokens 100
```

For audio/image input, use the mlx-vlm API and pass the fused bf16 model's
processor alongside this snapshot's weights.
"""


def upload_folder_to_repo(
    repo_id: str,
    folder_path: Path,
    token: str | None,
    *,
    commit_message: str,
    ignore_patterns: list[str] | str | None = None,
) -> str:
    """Push a generated artifact directory to a model repo. Returns the URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder_path),
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )
    return f"https://huggingface.co/{repo_id}"


def upload_fused(
    folder_path: Path,
    info: FusedModelInfo,
    repo_id: str,
    token: str | None = None,
) -> str:
    """Upload a fused checkpoint plus its model card. Returns the repo URL."""
    card_path = folder_path / "README.md"
    card_path.write_text(build_fused_card(info), encoding="utf-8")
    return upload_folder_to_repo(
        repo_id,
        folder_path,
        token,
        commit_message="Fused bf16: base + Barbados LoRA, talker dropped",
    )


def upload_mlx(
    folder_path: Path,
    info: FusedModelInfo,
    repo_id: str,
    token: str | None = None,
) -> str:
    """Upload an MLX snapshot plus its model card. Returns the repo URL."""
    card_path = folder_path / "README.md"
    card_path.write_text(build_mlx_card(info), encoding="utf-8")
    return upload_folder_to_repo(
        repo_id,
        folder_path,
        token,
        commit_message="MLX 4-bit: fused bf16 quantized (q-bits 4, group-size 64)",
    )
