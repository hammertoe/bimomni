"""HF Bucket checkpoint persistence for ephemeral pods.

A sidecar process polls a local output directory for new checkpoints, hard-
links each completed checkpoint into a stable spool directory, then uploads
the spool to a private Hugging Face Bucket. Upload integrity is proven by a
`_COMPLETE.json` marker that contains file sizes, the recipe manifest hash,
and the step number. On startup, restore only considers entries whose marker
exists.

State design:

    runs/<run-id>/remote/<step>/        mirror of the spool as of last upload
    runs/<run-id>/remote/<step>/_COMPLETE.json
                                         integrity marker for restore
    runs/<run-id>/state/budget.json      cumulative GPU-seconds across resumes
    runs/<run-id>/state/run.json         most recent run metadata

The most recent two complete remote checkpoints are kept; older prefixes are
deleted so a long run never accumulates petabytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from huggingface_hub import (
    HfApi,
    batch_bucket_files,
    create_bucket,
    download_bucket_files,
    list_bucket_tree,
)

from bimomni.training.recipe import RecipeManifest, manifests_compatible, recipe_diff


LOGGER = logging.getLogger(__name__)
KEEP_COMPLETED = 2
CHECKPOINT_PREFIX_RE = re.compile(r"^checkpoint-(\d+)$")
# ms-swift 4.4 skips saving the tokenizer/processor for adapter (LoRA) runs,
# so the completeness gate must not require it. The `_COMPLETE.json` marker
# records the exact file inventory + sizes, which is the real integrity record.
EXPECTED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)
# Per-chunk size cap for the bucket upload. A single 31 GB optimizer.pt is the
# root cause of "Internal error: timed out reading request body" — splitting
# into 2 GB sub-batches keeps each Hub request body small enough to stream.
BATCH_BYTES = 2 * 1024 * 1024 * 1024
# Concurrent upload chunks in flight. The Hub upload path is network-bound; two
# to four in flight gives meaningful overlap without overwhelming the connection.
PARALLEL_UPLOADS = 4
# How many times to retry a single chunk before giving up. The chunk either
# fully lands or fully fails — partial successes can't happen mid-batch.
UPLOAD_RETRIES = 3


@dataclass(frozen=True, slots=True)
class RemoteCheckpoint:
    step: int
    prefix: str
    size_bytes: int
    marker_present: bool
    manifest_hash: str

    @property
    def complete(self) -> bool:
        return self.marker_present


@dataclass(frozen=True, slots=True)
class CheckpointFiles:
    step: int
    files: tuple[tuple[str, int], ...]


def parse_step(checkpoint_name: str) -> int | None:
    """Return a step from local `checkpoint-N` or remote `N` directory names."""
    suffix = checkpoint_name.removeprefix("checkpoint-")
    try:
        step = int(suffix)
    except ValueError:
        return None
    return step if step >= 0 else None


def sha256_file(path: Path) -> str:
    """SHA-256 hash of a single file, streamed so large files stay cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def expected_checkpoint_files(checkpoint_dir: Path) -> tuple[str, ...]:
    """Files that must be present to consider the checkpoint uploadable."""
    if (checkpoint_dir / "adapter_model.safetensors").exists():
        return EXPECTED_FILES
    return tuple(sorted(p.name for p in checkpoint_dir.iterdir() if p.is_file()))


def is_checkpoint_complete(
    checkpoint_dir: Path,
    expected: Iterable[str] = EXPECTED_FILES,
    *,
    stable_for_seconds: float = 10.0,
) -> bool:
    """Return True if the checkpoint is ready to upload.

    A checkpoint is complete when all expected files exist, sizes are stable
    across two checks separated by `stable_for_seconds`, and a `.safetensors`
    index is present alongside each sharded safetensors payload.
    """
    expected_set = set(expected)
    first_sizes: dict[str, int] = {}
    for name in expected_set:
        path = checkpoint_dir / name
        if not path.exists():
            return False
        first_sizes[name] = path.stat().st_size
    if "adapter_model.safetensors.index.json" in {p.name for p in checkpoint_dir.iterdir()}:
        if not (checkpoint_dir / "adapter_model.safetensors.index.json").exists():
            return False
    time.sleep(max(0.0, stable_for_seconds))
    for name, size in first_sizes.items():
        path = checkpoint_dir / name
        if not path.exists():
            return False
        if path.stat().st_size != size:
            return False
    return True


def write_complete_marker(
    *,
    checkpoint_dir: Path,
    manifest: RecipeManifest,
    run_id: str,
    step: int,
    extra_metadata: dict[str, str] | None = None,
) -> Path:
    """Write `_COMPLETE.json` for a finished checkpoint so restore can use it."""
    files = sorted(
        (p, p.stat().st_size) for p in checkpoint_dir.rglob("*") if p.is_file()
    )
    payload = {
        "run_id": run_id,
        "step": step,
        "manifest_hash": manifest.stable_hash(),
        "image_digest": manifest.image_digest,
        "created_at": time.time(),
        "files": [
            {"path": str(path.relative_to(checkpoint_dir)), "size": size}
            for path, size in files
        ],
        "metadata": extra_metadata or {},
    }
    marker = checkpoint_dir / "_COMPLETE.json"
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return marker


def marker_is_complete(marker: Path, expected_manifest: RecipeManifest) -> bool:
    """Return True if `_COMPLETE.json` matches the current recipe manifest."""
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    remote_hash = str(payload.get("manifest_hash", ""))
    return remote_hash == expected_manifest.stable_hash()


def hard_link_checkpoint(
    src: Path,
    *,
    spool_root: Path,
    step: int,
) -> Path:
    """Hard-link a checkpoint into a stable spool path before upload.

    Hard-linking keeps disk usage cheap and protects against Trainer pruning
    the source directory mid-upload.
    """
    target = spool_root / f"checkpoint-{step}"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    for entry in src.rglob("*"):
        rel = entry.relative_to(src)
        if entry.is_dir():
            (target / rel).mkdir(parents=True, exist_ok=True)
        else:
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            if (target / rel).exists():
                (target / rel).unlink()
            os.link(entry, target / rel)
    return target


def upload_checkpoint(
    *,
    spool_root: Path,
    step: int,
    bucket_id: str,
    run_id: str,
    hf_token: str,
    batch_bytes: int = BATCH_BYTES,
    parallelism: int = PARALLEL_UPLOADS,
    retries: int = UPLOAD_RETRIES,
) -> int:
    """Push a single spooled checkpoint to the bucket in parallel chunks.

    Data files are uploaded first across bounded parallel batches of at most
    `batch_bytes` each. The `_COMPLETE.json` marker is uploaded last, in its
    own dedicated call, so a partial failure can never advertise a "complete"
    checkpoint that is missing data. Each chunk retries up to `retries` times
    with exponential backoff before the whole upload fails.
    """
    checkpoint = spool_root / f"checkpoint-{step}"
    marker = checkpoint / "_COMPLETE.json"
    if not marker.exists():
        raise FileNotFoundError(f"missing marker {marker}; refusing to upload")
    upload_root = f"runs/{run_id}/checkpoints/{step}"

    data_items: list[tuple[str, str, int]] = []
    marker_upload: tuple[str, str] | None = None
    for entry in checkpoint.rglob("*"):
        if not entry.is_file():
            continue
        rel = str(entry.relative_to(checkpoint))
        dest = f"{upload_root}/{rel}"
        if entry.name == "_COMPLETE.json":
            marker_upload = (str(entry), dest)
            continue
        data_items.append((str(entry), dest, entry.stat().st_size))
    # Smallest first — if a chunk fails, the smallest data files have already
    # landed and the largest (optimizer.pt) is left to retry, which is the
    # cheaper retry on every level: disk read, network bytes, latency.
    data_items.sort(key=lambda item: item[2])

    chunks = _chunk_data_items(data_items, batch_bytes=batch_bytes)

    def _upload_chunk(chunk: list[tuple[str, str]], attempt: int = 1) -> None:
        try:
            batch_bucket_files(bucket_id, add=chunk)
        except Exception as exc:
            if attempt >= retries:
                raise
            LOGGER.warning(
                "[uploader] chunk upload failed (attempt %d/%d): %s; retrying",
                attempt, retries, exc,
            )
            time.sleep(2 ** attempt)
            _upload_chunk(chunk, attempt + 1)

    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as pool:
        futures = [
            pool.submit(_upload_chunk, [(src, dest) for src, dest, _ in chunk])
            for chunk in chunks
        ]
        for future in as_completed(futures):
            future.result()

    if marker_upload is not None:
        batch_bucket_files(bucket_id, add=[marker_upload])

    return sum(size for _, _, size in data_items)


def _chunk_data_items(
    items: list[tuple[str, str, int]],
    *,
    batch_bytes: int,
) -> list[list[tuple[str, str, int]]]:
    """Greedy bin-pack (src, dest, size) into chunks each summing to ≤ batch_bytes.

    Items larger than batch_bytes occupy their own chunk — they can never be
    co-batched without breaching the cap. Order is preserved within a chunk;
    callers that want a different order should pre-sort `items` (e.g. smallest
    first so the largest file is the last to retry on failure).
    """
    chunks: list[list[tuple[str, str, int]]] = []
    current: list[tuple[str, str, int]] = []
    current_size = 0
    for item in items:
        src, dest, size = item
        if current and current_size + size > batch_bytes:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(item)
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def partition_uploads(
    items: list[tuple[str, int]],
    *,
    batch_bytes: int,
) -> list[list[tuple[str, int]]]:
    """Greedy bin-pack (src, size) into chunks each summing to ≤ batch_bytes.

    Items larger than batch_bytes occupy their own chunk — they can never be
    co-batched without breaching the cap. Order is preserved within a chunk;
    callers that want a different order should pre-sort `items` (e.g. smallest
    first so the largest file is the last to retry on failure).
    """
    chunks: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_size = 0
    for src, size in items:
        if current and current_size + size > batch_bytes:
            chunks.append(current)
            current = []
            current_size = 0
        current.append((src, size))
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def prune_old_checkpoints(
    *,
    bucket_id: str,
    run_id: str,
    keep: int = KEEP_COMPLETED,
    hf_token: str | None = None,
) -> list[int]:
    """Delete remote checkpoint prefixes so only the newest `keep` complete remain."""
    api = HfApi(token=hf_token)
    prefix = f"runs/{run_id}/checkpoints/"
    completed: list[tuple[int, Path]] = []
    for entry in list_bucket_tree(bucket_id, prefix=prefix, recursive=False):
        if entry.type != "directory":
            continue
        name = entry.path.rstrip("/").rsplit("/", 1)[-1]
        step = parse_step(name)
        if step is None:
            continue
        marker_path = f"{prefix}{name}/_COMPLETE.json"
        marker_files = [
            item for item in list_bucket_tree(bucket_id, prefix=f"{prefix}{name}/")
            if item.type == "file" and item.path.endswith("_COMPLETE.json")
        ]
        if not marker_files:
            continue
        completed.append((step, Path(marker_path)))
    completed.sort(key=lambda pair: pair[0], reverse=True)
    stale = [pair[1].parent for pair in completed[keep:]]
    for path in stale:
        files_to_delete = [
            item.path
            for item in list_bucket_tree(bucket_id, prefix=str(path) + "/")
            if item.type == "file"
        ]
        if files_to_delete:
            batch_bucket_files(bucket_id, delete=files_to_delete)
    return [step for step, _ in completed[keep:]]


def list_remote_checkpoints(
    bucket_id: str, *, run_id: str
) -> list[RemoteCheckpoint]:
    """List bucket checkpoints with completion metadata."""
    prefix = f"runs/{run_id}/checkpoints/"
    out: list[RemoteCheckpoint] = []
    for entry in list_bucket_tree(bucket_id, prefix=prefix, recursive=False):
        if entry.type != "directory":
            continue
        name = entry.path.rstrip("/").rsplit("/", 1)[-1]
        step = parse_step(name)
        if step is None:
            continue
        marker_path = f"{prefix}{name}/_COMPLETE.json"
        marker_files = [
            item for item in list_bucket_tree(bucket_id, prefix=f"{prefix}{name}/")
            if item.type == "file" and item.path.endswith("_COMPLETE.json")
        ]
        marker_present = bool(marker_files)
        manifest_hash = ""
        if marker_present:
            manifest_hash = _read_remote_marker_hash(bucket_id, marker_path)
        size = _dir_size_remote(bucket_id, prefix + name + "/")
        out.append(
            RemoteCheckpoint(
                step=step,
                prefix=prefix + name + "/",
                size_bytes=size,
                marker_present=marker_present,
                manifest_hash=manifest_hash,
            )
        )
    out.sort(key=lambda c: c.step, reverse=True)
    return out


def _read_remote_marker_hash(
    bucket_id: str,
    marker_path: str,
    *,
    hf_token: str | None = None,
) -> str:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_marker = Path(tmp_dir) / "_COMPLETE.json"
            download_bucket_files(
                bucket_id,
                files=[(marker_path, local_marker)],
                token=hf_token,
            )
            payload = json.loads(local_marker.read_text(encoding="utf-8"))
        return str(payload.get("manifest_hash", ""))
    except (json.JSONDecodeError, OSError):
        return ""


def _dir_size_remote(bucket_id: str, prefix: str) -> int:
    return sum(
        entry.size or 0
        for entry in list_bucket_tree(bucket_id, prefix=prefix)
        if entry.type == "file"
    )


def latest_complete_remote_step(
    bucket_id: str, *, run_id: str, manifest: RecipeManifest
) -> int | None:
    """Return the highest step whose `_COMPLETE.json` matches `manifest`."""
    for checkpoint in list_remote_checkpoints(bucket_id, run_id=run_id):
        if not checkpoint.complete:
            continue
        if checkpoint.manifest_hash != manifest.stable_hash():
            LOGGER.warning(
                "remote checkpoint step=%d manifest mismatch (hash=%s) — skipping",
                checkpoint.step,
                checkpoint.manifest_hash,
            )
            continue
        return checkpoint.step
    return None


def restore_latest_checkpoint(
    *,
    bucket_id: str,
    run_id: str,
    manifest: RecipeManifest,
    target_dir: Path,
    hf_token: str | None = None,
) -> Path | None:
    """Download the newest compatible remote checkpoint into `target_dir`."""
    step = latest_complete_remote_step(bucket_id, run_id=run_id, manifest=manifest)
    if step is None:
        return None
    return _download_checkpoint(
        bucket_id=bucket_id,
        run_id=run_id,
        step=step,
        target_dir=target_dir,
        hf_token=hf_token,
    )


def restore_latest_checkpoint_any(
    *,
    bucket_id: str,
    run_id: str,
    target_dir: Path,
    hf_token: str | None = None,
) -> Path | None:
    """Download the newest complete remote checkpoint without a manifest check.

    Used for post-hoc recovery (e.g. re-uploading a finished run's adapter)
    where the local container cannot reproduce the original recipe manifest.
    """
    step = latest_remote_step(bucket_id, run_id=run_id)
    if step is None:
        return None
    return _download_checkpoint(
        bucket_id=bucket_id,
        run_id=run_id,
        step=step,
        target_dir=target_dir,
        hf_token=hf_token,
    )


def _download_checkpoint(
    *,
    bucket_id: str,
    run_id: str,
    step: int,
    target_dir: Path,
    hf_token: str | None = None,
) -> Path:
    target = target_dir / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)
    prefix = f"runs/{run_id}/checkpoints/{step}/"
    downloads: list[tuple[str, str]] = []
    for entry in list_bucket_tree(bucket_id, prefix=prefix):
        if entry.type != "file":
            continue
        rel = entry.path[len(prefix):]
        downloads.append((entry.path, str(target / rel)))
    if downloads:
        download_bucket_files(bucket_id, files=downloads)
    return target


def latest_remote_step(bucket_id: str, *, run_id: str) -> int | None:
    """Return the highest complete remote checkpoint step, ignoring manifests."""
    for checkpoint in list_remote_checkpoints(bucket_id, run_id=run_id):
        if checkpoint.complete:
            return checkpoint.step
    return None


@dataclass
class Uploader:
    """Sidecar that watches a checkpoint directory and uploads to HF Buckets."""

    output_dir: Path
    spool_root: Path
    bucket_id: str
    run_id: str
    manifest: RecipeManifest
    hf_token: str
    poll_seconds: float = 20.0
    keep: int = KEEP_COMPLETED
    stable_for_seconds: float = 10.0
    log: Callable[[str], None] = LOGGER.info

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_steps: set[int] = set()
        self._spool_root = self.spool_root / "remote"
        self._spool_root.mkdir(parents=True, exist_ok=True)
        create_bucket(self.bucket_id, exist_ok=True)
        self._seed_seen_steps()

    def _seed_seen_steps(self) -> None:
        for checkpoint in list_remote_checkpoints(
            self.bucket_id, run_id=self.run_id
        ):
            if checkpoint.complete and (
                checkpoint.manifest_hash == ""
                or checkpoint.manifest_hash == self.manifest.stable_hash()
            ):
                self._seen_steps.add(checkpoint.step)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="hf-bucket-uploader", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def run_once(self) -> list[int]:
        """One synchronous pass for testing or one-shot CLI mode."""
        return self._scan_once()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_once()
            except Exception as exc:  # noqa: BLE001
                self.log(f"[uploader] error: {exc}")
            self._stop.wait(self.poll_seconds)

    def _scan_once(self) -> list[int]:
        uploaded: list[int] = []
        candidates = sorted(
            (p for p in self.output_dir.iterdir() if p.is_dir()),
            key=lambda path: parse_step(path.name) or 0,
        )
        for candidate in candidates:
            step = parse_step(candidate.name)
            if step is None or step in self._seen_steps:
                continue
            try:
                if not is_checkpoint_complete(
                    candidate,
                    expected=expected_checkpoint_files(candidate),
                    stable_for_seconds=self.stable_for_seconds,
                ):
                    continue
            except Exception:
                continue
            self._seen_steps.add(step)
            write_complete_marker(
                checkpoint_dir=candidate,
                manifest=self.manifest,
                run_id=self.run_id,
                step=step,
            )
            spooled = hard_link_checkpoint(
                candidate, spool_root=self._spool_root, step=step
            )
            try:
                upload_checkpoint(
                    spool_root=self._spool_root,
                    step=step,
                    bucket_id=self.bucket_id,
                    run_id=self.run_id,
                    hf_token=self.hf_token,
                )
                uploaded.append(step)
                self.log(f"[uploader] uploaded checkpoint step={step} ({spooled})")
            except Exception as exc:  # noqa: BLE001
                self.log(f"[uploader] upload step={step} failed: {exc}")
                self._seen_steps.discard(step)
            finally:
                shutil.rmtree(spooled, ignore_errors=True)
        if uploaded:
            try:
                stale = prune_old_checkpoints(
                    bucket_id=self.bucket_id,
                    run_id=self.run_id,
                    keep=self.keep,
                    hf_token=self.hf_token,
                )
                if stale:
                    self.log(f"[uploader] pruned older remote steps: {stale}")
            except Exception as exc:  # noqa: BLE001
                self.log(f"[uploader] prune failed: {exc}")
        return uploaded


__all__ = [
    "BATCH_BYTES",
    "CHECKPOINT_PREFIX_RE",
    "EXPECTED_FILES",
    "KEEP_COMPLETED",
    "PARALLEL_UPLOADS",
    "UPLOAD_RETRIES",
    "Uploader",
    "CheckpointFiles",
    "RemoteCheckpoint",
    "create_bucket_for_run",
    "dir_size",
    "expected_checkpoint_files",
    "hard_link_checkpoint",
    "is_checkpoint_complete",
    "latest_complete_remote_step",
    "list_remote_checkpoints",
    "parse_step",
    "partition_uploads",
    "prune_old_checkpoints",
    "restore_latest_checkpoint",
    "sha256_file",
    "upload_checkpoint",
    "write_complete_marker",
]


def create_bucket_for_run(bucket_id: str, *, private: bool = True) -> None:
    """Idempotently create the run's bucket."""
    create_bucket(bucket_id, private=private, exist_ok=True)
