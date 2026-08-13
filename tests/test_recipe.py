"""Tests for the recipe manifest identity and integrity checks."""

from __future__ import annotations

import json
from pathlib import Path

from bimomni.training.recipe import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    DATASET_REPO_ID,
    DATASET_REVISION,
    RecipeManifest,
    build_current_manifest,
    manifests_compatible,
    recipe_diff,
    write_manifest,
)


def _sample_manifest(**overrides) -> RecipeManifest:
    base = dict(
        base_model_id=BASE_MODEL_ID,
        base_model_revision=BASE_MODEL_REVISION,
        dataset_repo_id=DATASET_REPO_ID,
        dataset_revision=DATASET_REVISION,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        target_parameters=("gate_up_proj", "down_proj"),
        max_length=4096,
        batch_size=1,
        gradient_accumulation=16,
        save_steps=100,
        python_version="3.10",
        torch_version="2.7.1",
        transformers_version="5.8.1",
        peft_version="0.18.1",
        flash_attn_version="2.7.4.post1",
        ms_swift_version="4.4.2",
        swift_commit="release",
        image_digest="sha256:deadbeef",
    )
    base.update(overrides)
    return RecipeManifest(**base)


def test_stable_hash_is_deterministic() -> None:
    manifest = _sample_manifest()
    assert manifest.stable_hash() == manifest.stable_hash()
    assert len(manifest.stable_hash()) == 64


def test_stable_hash_changes_on_recipe_drift() -> None:
    original = _sample_manifest()
    drifted = _sample_manifest(lora_rank=32)
    assert original.stable_hash() != drifted.stable_hash()


def test_manifests_compatible_identical() -> None:
    a = _sample_manifest()
    b = _sample_manifest()
    assert manifests_compatible(a, b)
    assert not recipe_diff(a, b).startswith(" ")


def test_manifests_compatible_differs_on_base_revision() -> None:
    a = _sample_manifest()
    b = _sample_manifest(base_model_revision="other-sha")
    assert not manifests_compatible(a, b)
    diff = recipe_diff(a, b)
    assert "base_model_revision" in diff


def test_manifests_compatible_differs_on_target_modules() -> None:
    a = _sample_manifest()
    b = _sample_manifest(target_modules=("q_proj",))
    assert not manifests_compatible(a, b)


def test_manifest_json_roundtrip() -> None:
    manifest = _sample_manifest()
    payload = json.loads(manifest.to_json())
    restored = RecipeManifest.from_json(payload)
    assert manifests_compatible(manifest, restored)


def test_write_manifest_persists(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    target = tmp_path / "recipe.json"
    write_manifest(target, manifest)
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["lora_rank"] == 64
    assert payload["base_model_revision"] == BASE_MODEL_REVISION


def test_build_current_manifest_uses_locked_values() -> None:
    manifest = build_current_manifest(image_digest="sha256:test")
    assert manifest.base_model_id == BASE_MODEL_ID
    assert manifest.base_model_revision == BASE_MODEL_REVISION
    assert manifest.dataset_repo_id == DATASET_REPO_ID
    assert manifest.dataset_revision == DATASET_REVISION
    assert manifest.lora_rank == 64
    assert manifest.lora_alpha == 128
    assert manifest.lora_dropout == 0.0
    assert manifest.image_digest == "sha256:test"
