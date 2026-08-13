"""Tests for HF Bucket checkpoint sync (completion, restore, prune, upload)."""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import bimomni.training.checkpoint_sync as cs
from bimomni.training.checkpoint_sync import (
    BATCH_BYTES,
    PARALLEL_UPLOADS,
    UPLOAD_RETRIES,
    dir_size,
    expected_checkpoint_files,
    hard_link_checkpoint,
    is_checkpoint_complete,
    marker_is_complete,
    parse_step,
    partition_uploads,
    sha256_file,
    upload_checkpoint,
    write_complete_marker,
)
from bimomni.training.recipe import RecipeManifest


def _sample_manifest(**overrides) -> RecipeManifest:
    base = dict(
        base_model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        base_model_revision="r",
        dataset_repo_id="hammertoe/barbados-dapt-v2",
        dataset_revision="d",
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=("q_proj",),
        target_parameters=("gate_up_proj",),
        max_length=4096,
        batch_size=1,
        gradient_accumulation=16,
        save_steps=100,
        python_version="3.10",
        torch_version="2.0",
        transformers_version="5.0",
        peft_version="0.18",
        flash_attn_version="2.7",
        ms_swift_version="4.4",
        swift_commit="release",
        image_digest="sha256:abc",
    )
    base.update(overrides)
    return RecipeManifest(**base)


def _adapter_files(directory: Path, *, step: int, extra: Sequence[str] = ()) -> Path:
    target = directory / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (target / name).write_text(name, encoding="utf-8")
    for name in extra:
        (target / name).write_text(name, encoding="utf-8")
    return target


def test_parse_step_extracts_int() -> None:
    assert parse_step("checkpoint-100") == 100
    assert parse_step("checkpoint-14087") == 14087
    assert parse_step("100") == 100
    assert parse_step("runs") is None
    assert parse_step("checkpoint-abc") is None


def test_dir_size_sums_files(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"x" * 100)
    (tmp_path / "b").write_bytes(b"y" * 250)
    assert dir_size(tmp_path) == 350


def test_sha256_file_hashes_content(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello world")
    assert sha256_file(path) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_expected_checkpoint_files_returns_adapter_set(tmp_path: Path) -> None:
    checkpoint = _adapter_files(tmp_path, step=100)
    files = expected_checkpoint_files(checkpoint)
    assert "adapter_config.json" in files
    assert "adapter_model.safetensors" in files


def test_expected_checkpoint_files_when_no_adapter(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "anything.bin").write_text("x")
    assert expected_checkpoint_files(checkpoint) == ("anything.bin",)


def test_is_checkpoint_complete_with_stable_check(tmp_path: Path) -> None:
    checkpoint = _adapter_files(tmp_path, step=100)
    files = expected_checkpoint_files(checkpoint)
    assert is_checkpoint_complete(checkpoint, expected=files, stable_for_seconds=0.0)


def test_is_checkpoint_complete_rejects_missing_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("x")
    assert not is_checkpoint_complete(
        checkpoint,
        expected=("adapter_config.json", "adapter_model.safetensors"),
        stable_for_seconds=0.0,
    )


def test_is_checkpoint_complete_adapter_only_without_tokenizer(tmp_path: Path) -> None:
    """ms-swift 4.4 skips tokenizer for LoRA checkpoints; adapter files suffice."""
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (checkpoint / "optimizer.pt").write_text("x", encoding="utf-8")
    assert is_checkpoint_complete(
        checkpoint,
        expected=expected_checkpoint_files(checkpoint),
        stable_for_seconds=0.0,
    )


def test_is_checkpoint_complete_rejects_growing_file(tmp_path: Path) -> None:
    checkpoint = _adapter_files(tmp_path, step=100)
    files = expected_checkpoint_files(checkpoint)
    target = checkpoint / "adapter_model.safetensors"

    first_pass_done = {"flag": False}

    def flaky_stat(self, *args, **kwargs):
        result = Path.stat_orig(self, *args, **kwargs)
        # After the first pass through the expected files, grow one of them so
        # the second-pass comparison detects the change.
        if (
            self == target
            and first_pass_done["flag"]
        ):
            with open(target, "ab") as handle:
                handle.write(b"x")
        if self == target:
            first_pass_done["flag"] = True
        return result

    Path.stat_orig = Path.stat  # type: ignore[attr-defined]
    Path.stat = flaky_stat  # type: ignore[assignment]
    try:
        assert not is_checkpoint_complete(
            checkpoint, expected=files, stable_for_seconds=0.0
        )
    finally:
        Path.stat = Path.stat_orig  # type: ignore[assignment]


def test_write_complete_marker_roundtrip(tmp_path: Path) -> None:
    checkpoint = _adapter_files(tmp_path, step=200)
    manifest = _sample_manifest()
    marker = write_complete_marker(
        checkpoint_dir=checkpoint,
        manifest=manifest,
        run_id="run-1",
        step=200,
    )
    assert marker.exists()
    assert marker_is_complete(marker, manifest)
    assert not marker_is_complete(marker, _sample_manifest(lora_rank=32))


def test_hard_link_checkpoint_protects_source(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    checkpoint = _adapter_files(source_root, step=50)
    spool = tmp_path / "spool"
    spool.mkdir()
    spooled = hard_link_checkpoint(checkpoint, spool_root=spool, step=50)
    assert spooled.exists()
    shutil.rmtree(checkpoint)
    assert spooled.exists()
    assert (spooled / "adapter_config.json").read_text() == "adapter_config.json"


def test_uploader_run_once_no_new_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    output.mkdir()
    spool = tmp_path / "spool"
    manifest = _sample_manifest()
    monkeypatch.setattr(cs, "create_bucket", lambda *a, **kw: None)
    monkeypatch.setattr(cs, "list_remote_checkpoints", lambda *a, **kw: [])
    uploader = cs.Uploader(
        output_dir=output,
        spool_root=spool,
        bucket_id="hammertoe/test",
        run_id="run-1",
        manifest=manifest,
        hf_token="dummy",
        poll_seconds=1.0,
        stable_for_seconds=0.0,
    )
    assert uploader.run_once() == []


def test_uploader_run_once_uploads_new_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _adapter_files(output, step=100)
    spool = tmp_path / "spool"
    manifest = _sample_manifest()
    captured = []

    def fake_upload_checkpoint(*, spool_root, step, bucket_id, run_id, hf_token):
        captured.append((bucket_id, run_id, step))
        return 1024

    monkeypatch.setattr(cs, "create_bucket", lambda *a, **kw: None)
    monkeypatch.setattr(cs, "list_remote_checkpoints", lambda *a, **kw: [])
    monkeypatch.setattr(cs, "upload_checkpoint", fake_upload_checkpoint)
    monkeypatch.setattr(cs, "prune_old_checkpoints", lambda **kw: [])
    uploader = cs.Uploader(
        output_dir=output,
        spool_root=spool,
        bucket_id="hammertoe/test",
        run_id="run-1",
        manifest=manifest,
        hf_token="dummy",
        poll_seconds=1.0,
        stable_for_seconds=0.0,
    )
    assert uploader.run_once() == [100]
    assert ("hammertoe/test", "run-1", 100) in captured
    assert uploader.run_once() == []


def test_prune_old_checkpoints_keeps_last_two(monkeypatch: pytest.MonkeyPatch) -> None:
    deleted: list[tuple[str, ...]] = []

    top_level = [
        MagicMock(type="directory", path="runs/run-1/checkpoints/checkpoint-100"),
        MagicMock(type="directory", path="runs/run-1/checkpoints/checkpoint-200"),
        MagicMock(type="directory", path="runs/run-1/checkpoints/checkpoint-300"),
    ]
    marker_files = [
        MagicMock(type="file", path="runs/run-1/checkpoints/checkpoint-100/_COMPLETE.json"),
        MagicMock(type="file", path="runs/run-1/checkpoints/checkpoint-200/_COMPLETE.json"),
        MagicMock(type="file", path="runs/run-1/checkpoints/checkpoint-300/_COMPLETE.json"),
    ]

    def fake_list_bucket_tree(bucket_id, *, prefix, recursive=True):
        if not recursive:
            return top_level
        return marker_files

    def fake_batch(bucket_id, *, add=(), delete=()):
        if delete:
            deleted.append(tuple(delete))
        return None

    monkeypatch.setattr(cs, "list_bucket_tree", fake_list_bucket_tree)
    monkeypatch.setattr(cs, "batch_bucket_files", fake_batch)
    stale = cs.prune_old_checkpoints(
        bucket_id="hammertoe/test",
        run_id="run-1",
        keep=2,
    )
    assert stale == [100]
    assert deleted, "expected at least one delete batch"


def test_uploader_seed_seen_steps_with_remote_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _sample_manifest()

    def fake_list_remote(bucket_id, *, run_id):
        return [
            cs.RemoteCheckpoint(
                step=100,
                prefix="",
                size_bytes=1,
                marker_present=True,
                manifest_hash=manifest.stable_hash(),
            ),
            cs.RemoteCheckpoint(
                step=50,
                prefix="",
                size_bytes=1,
                marker_present=False,
                manifest_hash="",
            ),
        ]

    monkeypatch.setattr(cs, "list_remote_checkpoints", fake_list_remote)
    monkeypatch.setattr(cs, "create_bucket", lambda *a, **kw: None)
    uploader = cs.Uploader(
        output_dir=Path("/nonexistent"),
        spool_root=Path("/tmp/spool"),
        bucket_id="hammertoe/test",
        run_id="run-1",
        manifest=manifest,
        hf_token="dummy",
        poll_seconds=1.0,
        stable_for_seconds=0.0,
    )
    assert uploader._seen_steps == {100}


def test_latest_complete_remote_step_skips_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _sample_manifest()
    mismatch = _sample_manifest(lora_rank=32)

    def fake_list_remote(bucket_id, *, run_id):
        return [
            cs.RemoteCheckpoint(
                step=100,
                prefix="",
                size_bytes=1,
                marker_present=True,
                manifest_hash=mismatch.stable_hash(),
            ),
            cs.RemoteCheckpoint(
                step=50,
                prefix="",
                size_bytes=1,
                marker_present=True,
                manifest_hash=manifest.stable_hash(),
            ),
        ]

    monkeypatch.setattr(cs, "list_remote_checkpoints", fake_list_remote)
    step = cs.latest_complete_remote_step(
        "hammertoe/test", run_id="run-1", manifest=manifest
    )
    assert step == 50


def test_read_remote_marker_hash_reads_downloaded_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _sample_manifest()

    def fake_download(bucket_id, files, *, token=None):
        assert bucket_id == "hammertoe/test"
        assert token == "secret"
        remote, local = files[0]
        assert remote.endswith("_COMPLETE.json")
        Path(local).write_text(
            json.dumps({"manifest_hash": manifest.stable_hash()}),
            encoding="utf-8",
        )
        return None

    monkeypatch.setattr(cs, "download_bucket_files", fake_download)
    assert cs._read_remote_marker_hash(
        "hammertoe/test",
        "runs/run-1/checkpoints/100/_COMPLETE.json",
        hf_token="secret",
    ) == manifest.stable_hash()


def test_partition_uploads_respects_batch_bytes() -> None:
    items = [
        ("a.bin", 100),
        ("b.bin", 1_500_000_000),
        ("c.bin", 100),
        ("d.bin", 1_500_000_000),
    ]
    chunks = partition_uploads(items, batch_bytes=2_000_000_000)
    flat = [(src, size) for chunk in chunks for src, size in chunk]
    assert flat == items
    assert all(sum(size for _, size in chunk) <= 2_000_000_000 for chunk in chunks)
    assert len(chunks) >= 2


def test_partition_uploads_keeps_oversized_item_alone() -> None:
    items = [("huge.bin", 9_000_000_000), ("tiny.bin", 100)]
    chunks = partition_uploads(items, batch_bytes=2_000_000_000)
    assert chunks == [[("huge.bin", 9_000_000_000)], [("tiny.bin", 100)]]


def test_partition_uploads_returns_empty_for_empty_input() -> None:
    assert partition_uploads([], batch_bytes=1_000_000_000) == []


def test_upload_checkpoint_uploads_marker_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The marker must land after every data file, so a partial upload never advertises completeness."""
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_bytes(b"config")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "optimizer.pt").write_bytes(b"opt")
    manifest = _sample_manifest()
    write_complete_marker(
        checkpoint_dir=checkpoint,
        manifest=manifest,
        run_id="run-1",
        step=100,
    )
    spool = tmp_path / "spool"
    hard_link_checkpoint(checkpoint, spool_root=spool, step=100)

    calls: list[list[tuple[str, str]]] = []

    def fake_batch(bucket_id, *, add=(), delete=()):
        calls.append(list(add))

    monkeypatch.setattr(cs, "batch_bucket_files", fake_batch)

    upload_checkpoint(
        spool_root=spool,
        step=100,
        bucket_id="hammertoe/test",
        run_id="run-1",
        hf_token="secret",
    )

    assert len(calls) >= 2, "expected at least one data batch + final marker batch"
    final = calls[-1]
    assert len(final) == 1
    assert final[0][1].endswith("_COMPLETE.json")
    for prior in calls[:-1]:
        for _, dest in prior:
            assert not dest.endswith("_COMPLETE.json")


def test_upload_checkpoint_chunks_large_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single 31 GB optimizer must be split across multiple batch_bucket_files calls.

    The fake batch blocks until the test signals so we can observe concurrency
    deterministically (chunks enter before any of them finish).
    """
    checkpoint = tmp_path / "checkpoint-200"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"w" * 200)
    (checkpoint / "adapter_config.json").write_bytes(b"c" * 200)
    (checkpoint / "optimizer.pt").write_bytes(b"o" * 500)
    manifest = _sample_manifest()
    write_complete_marker(
        checkpoint_dir=checkpoint,
        manifest=manifest,
        run_id="run-1",
        step=200,
    )
    spool = tmp_path / "spool"
    hard_link_checkpoint(checkpoint, spool_root=spool, step=200)

    calls: list[list[tuple[str, str]]] = []
    in_flight = 0
    max_in_flight = 0
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    def fake_batch(bucket_id, *, add=(), delete=()):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if in_flight >= 2:
                entered.set()
        calls.append(list(add))
        release.wait(timeout=5)
        with lock:
            in_flight -= 1

    monkeypatch.setattr(cs, "batch_bucket_files", fake_batch)

    def driver():
        assert entered.wait(timeout=5), "expected ≥2 chunks in flight"
        release.set()

    threading.Thread(target=driver, daemon=True).start()

    upload_checkpoint(
        spool_root=spool,
        step=200,
        bucket_id="hammertoe/test",
        run_id="run-1",
        hf_token="secret",
        batch_bytes=300,
        parallelism=PARALLEL_UPLOADS,
    )

    data_calls = calls[:-1]
    assert len(data_calls) >= 3, "optimizer + adapter should split across multiple chunks"
    assert max_in_flight >= 2, f"expected parallel uploads, observed max_in_flight={max_in_flight}"


def test_upload_checkpoint_retries_failed_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint-300"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_bytes(b"c")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"w")
    manifest = _sample_manifest()
    write_complete_marker(
        checkpoint_dir=checkpoint,
        manifest=manifest,
        run_id="run-1",
        step=300,
    )
    spool = tmp_path / "spool"
    hard_link_checkpoint(checkpoint, spool_root=spool, step=300)

    attempts = {"count": 0}

    def flaky_batch(bucket_id, *, add=(), delete=()):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("simulated network blip")

    monkeypatch.setattr(cs, "batch_bucket_files", flaky_batch)

    upload_checkpoint(
        spool_root=spool,
        step=300,
        bucket_id="hammertoe/test",
        run_id="run-1",
        hf_token="secret",
    )

    assert attempts["count"] >= 2, "retry should have re-attempted at least once"


def test_upload_checkpoint_raises_after_max_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint-400"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_bytes(b"c")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"w")
    manifest = _sample_manifest()
    write_complete_marker(
        checkpoint_dir=checkpoint,
        manifest=manifest,
        run_id="run-1",
        step=400,
    )
    spool = tmp_path / "spool"
    hard_link_checkpoint(checkpoint, spool_root=spool, step=400)

    def always_fail(bucket_id, *, add=(), delete=()):
        raise RuntimeError("dead")

    sleeps: list[float] = []
    monkeypatch.setattr(cs, "batch_bucket_files", always_fail)
    monkeypatch.setattr(cs.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="dead"):
        upload_checkpoint(
            spool_root=spool,
            step=400,
            bucket_id="hammertoe/test",
            run_id="run-1",
            hf_token="secret",
            retries=2,
        )

    assert len(sleeps) == 1, "should sleep between retries only"


def test_upload_checkpoint_constants_have_safe_defaults() -> None:
    assert BATCH_BYTES >= 1_000_000_000, "batch too small to be useful"
    assert BATCH_BYTES <= 4_000_000_000, "batch too large for HF Hub request body"
    assert PARALLEL_UPLOADS >= 2
    assert PARALLEL_UPLOADS <= 8
    assert UPLOAD_RETRIES >= 2
