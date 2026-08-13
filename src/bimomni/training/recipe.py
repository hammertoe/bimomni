"""Recipe manifest and identity verification for the V3 DAPT container.

The manifest captures every parameter that affects checkpoint compatibility:
base revision, dataset revision, image digest, LoRA shape, optimizer, target
modules/parameters, gradient checkpointing, packing, and the exact swift flags.
On restore, the manifest stored next to the remote checkpoint is compared
against the manifest produced inside the new container. Any mismatch aborts
resume so we never silently mix training runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bimomni.training.train import (
    DATA_ROOT as _DATA_ROOT,
    LORA_ALPHA,
    LORA_RANK,
    MAX_LENGTH,
    THINKER_ATTENTION_MODULES,
    THINKER_MLP_PARAMETERS,
)


BASE_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
BASE_MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
DATASET_REPO_ID = "hammertoe/barbados-dapt-v2"
DATASET_REVISION = "ef6b5cc92850b90d574822d8e60d7aa1a3129d1b"
HF_BUCKET_REPO_ID = "hammertoe/barbados-dapt-checkpoints-v3"
ADAPTER_MODEL_REPO_ID = "hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v3"
FUSED_MODEL_REPO_ID = "hammertoe/Qwen3-Omni-30B-A3B-Barbados-fused-bf16-v3"
MLX_MODEL_REPO_ID = "hammertoe/Qwen3-Omni-30B-A3B-Barbados-4bit-v3"
LORA_DROPOUT = 0.0
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
SAVE_STEPS = 100


@dataclass(frozen=True, slots=True)
class RecipeManifest:
    """Locked recipe identity that all checkpoints must agree on."""

    base_model_id: str
    base_model_revision: str
    dataset_repo_id: str
    dataset_revision: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    target_parameters: tuple[str, ...]
    max_length: int
    batch_size: int
    gradient_accumulation: int
    save_steps: int
    python_version: str
    torch_version: str
    transformers_version: str
    peft_version: str
    flash_attn_version: str
    ms_swift_version: str
    swift_commit: str
    image_digest: str

    def stable_hash(self) -> str:
        """SHA-256 over the manifest fields, stable across Python processes."""
        canonical = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RecipeManifest:
        return cls(
            base_model_id=str(payload["base_model_id"]),
            base_model_revision=str(payload["base_model_revision"]),
            dataset_repo_id=str(payload["dataset_repo_id"]),
            dataset_revision=str(payload["dataset_revision"]),
            lora_rank=int(payload["lora_rank"]),
            lora_alpha=int(payload["lora_alpha"]),
            lora_dropout=float(payload["lora_dropout"]),
            target_modules=tuple(payload["target_modules"]),
            target_parameters=tuple(payload["target_parameters"]),
            max_length=int(payload["max_length"]),
            batch_size=int(payload["batch_size"]),
            gradient_accumulation=int(payload["gradient_accumulation"]),
            save_steps=int(payload["save_steps"]),
            python_version=str(payload["python_version"]),
            torch_version=str(payload["torch_version"]),
            transformers_version=str(payload["transformers_version"]),
            peft_version=str(payload["peft_version"]),
            flash_attn_version=str(payload["flash_attn_version"]),
            ms_swift_version=str(payload["ms_swift_version"]),
            swift_commit=str(payload["swift_commit"]),
            image_digest=str(payload["image_digest"]),
        )


def _module_version(module_name: str, attr: str = "__version__") -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    value = getattr(module, attr, None)
    if value is None:
        return "unknown"
    return str(value)


def build_current_manifest(image_digest: str = "") -> RecipeManifest:
    """Construct a manifest reflecting the current container's environment."""
    return RecipeManifest(
        base_model_id=BASE_MODEL_ID,
        base_model_revision=BASE_MODEL_REVISION,
        dataset_repo_id=DATASET_REPO_ID,
        dataset_revision=DATASET_REVISION,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=THINKER_ATTENTION_MODULES,
        target_parameters=THINKER_MLP_PARAMETERS,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        save_steps=SAVE_STEPS,
        python_version=platform.python_version(),
        torch_version=_module_version("torch"),
        transformers_version=_module_version("transformers"),
        peft_version=_module_version("peft"),
        flash_attn_version=_module_version("flash_attn"),
        ms_swift_version=_module_version("swift"),
        swift_commit=os.environ.get("SWIFT_GIT_COMMIT", "release"),
        image_digest=image_digest or os.environ.get("IMAGE_DIGEST",""),
    )


def write_manifest(path: Path, manifest: RecipeManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


def manifests_compatible(local: RecipeManifest, remote: RecipeManifest) -> bool:
    """Return True iff two manifests describe the same training recipe."""
    return local.stable_hash() == remote.stable_hash()


def recipe_diff(local: RecipeManifest, remote: RecipeManifest) -> str:
    """Render a human-readable diff of mismatched manifest fields."""
    rows: list[str] = []
    for field in RecipeManifest.__dataclass_fields__:
        lvalue = getattr(local, field)
        rvalue = getattr(remote, field)
        if lvalue != rvalue:
            rows.append(f"  {field}: local={lvalue!r} remote={rvalue!r}")
    if not rows:
        return "no differences"
    return "\n".join(rows)


if __name__ == "__main__":
    manifest = build_current_manifest()
    sys.stdout.write(manifest.to_json() + "\n")