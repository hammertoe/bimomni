"""Container supervisor for the Qwen3-Omni Barbados DAPT image.

Run modes:
- doctor       Verify environment, CUDA, and HF credentials.
- smoke        Two-step training pass into `DATA_ROOT/smoke-checkpoints`.
- sync-once    One synchronous checkpoint upload pass (for tests).
- train (def)  Full training under a persistent BudgetGuard.
- fuse         Merge the published adapter into the base, drop the talker,
               and upload the fused bf16 checkpoint (CPU job).
- mlx          Convert a fused bf16 checkpoint to 4-bit MLX and upload (CPU).
- finalise     Run `fuse` then `mlx` in a single CPU job.
- push-adapter Restore the newest checkpoint from the bucket and upload the
               adapter (recovery when the train job's upload failed).

Startup order for `train`:
1. Verify environment and credentials.
2. Download the base model revision into the local HF cache.
3. Download the packed train/eval JSONL from the pinned dataset revision.
4. Restore the newest compatible remote checkpoint (if any).
5. Spawn the HF Bucket uploader sidecar.
6. Run `swift pt` under a persistent BudgetGuard.
7. On normal completion or graceful termination, flush final artefacts and
   the adapter to the model repo.

All state lives under `$DATA_ROOT/state/` and survives ephemeral disks because
checkpoints, run metadata, and budget counters are continuously pushed to a
private Hugging Face Bucket.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bimomni.publish.fuse import fuse_adapter, write_provenance
from bimomni.publish.strip_talker import rewrite_mlx_config, strip_mlx_safetensors
from bimomni.publish.upload import (
    BaseModelInfo,
    FusedModelInfo,
    assert_adapter_loads,
    upload_fused,
    upload_mlx,
)
from bimomni.publish.upload import (
    upload as upload_adapter,
)
from bimomni.training.budget_guard import (
    build_guard,
)
from bimomni.training.checkpoint_sync import (
    Uploader,
    create_bucket_for_run,
    restore_latest_checkpoint,
    restore_latest_checkpoint_any,
)
from bimomni.training.recipe import (
    ADAPTER_MODEL_REPO_ID,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    DATASET_REPO_ID,
    DATASET_REVISION,
    FUSED_MODEL_REPO_ID,
    HF_BUCKET_REPO_ID,
    MLX_MODEL_REPO_ID,
    build_current_manifest,
    write_manifest,
)
from bimomni.training.train import (
    DAPTConfig,
    smoke_config,
)

LOGGER = logging.getLogger("bimomni.training.supervisor")
logging.basicConfig(
    level=os.environ.get("HF_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DATA_ROOT = Path(os.environ.get("DAPT_DATA_ROOT", "/data/v3"))
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
SMOKE_DIR = DATA_ROOT / "smoke-checkpoints"
HF_CACHE = DATA_ROOT / "hf"
STATE_DIR = DATA_ROOT / "state"
MANIFEST_PATH = STATE_DIR / "recipe.json"
BUDGET_STATE = STATE_DIR / "budget.json"
RUN_STATE = STATE_DIR / "run.json"
ADAPTER_DIR = DATA_ROOT / "adapter"
FUSED_DIR = DATA_ROOT / "fused-bf16"
MLX_DIR = DATA_ROOT / "mlx-4bit"

REQUIRED_ENV = ("HF_TOKEN",)


@dataclass(frozen=True, slots=True)
class Env:
    hf_token: str
    run_id: str
    image_digest: str
    bucket_id: str

    @classmethod
    def load(cls) -> Env:
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"missing required env vars: {missing}")
        return cls(
            hf_token=os.environ["HF_TOKEN"],
            run_id=os.environ.get("RUN_ID", _default_run_id()),
            image_digest=os.environ.get("IMAGE_DIGEST", ""),
            bucket_id=os.environ.get("HF_BUCKET_REPO_ID", HF_BUCKET_REPO_ID),
        )


def _default_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def ensure_directories() -> None:
    for directory in (DATA_ROOT, CHECKPOINT_DIR, HF_CACHE, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def verify_environment() -> None:
    """Sanity-check CUDA, the verified package set, and HF credentials."""
    import huggingface_hub
    import peft
    import torch
    import transformers

    LOGGER.info("torch=%s transformers=%s peft=%s huggingface_hub=%s",
                torch.__version__, transformers.__version__, peft.__version__,
                huggingface_hub.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the supervisor; no GPU detected")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        LOGGER.info("gpu[%d]: %s (%.1f GiB)", idx, props.name, props.total_memory / 2**30)
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()
    LOGGER.info("HF authenticated as %s", user.get("name", "unknown"))


def download_base_model() -> Path:
    """Cache the exact base revision used for V3."""
    from huggingface_hub import snapshot_download

    target = HF_CACHE / "models" / BASE_MODEL_ID.replace("/", "_") / BASE_MODEL_REVISION
    snapshot_download(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        cache_dir=str(HF_CACHE),
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json",
            "*.txt",
            "merges.txt",
            "vocab.json",
            "*.model",
            "*.safetensors",
            "*.py",
        ],
    )
    return target


def download_dataset_files(*, target_dir: Path) -> tuple[Path, Path]:
    """Fetch the packed train/eval JSONL files from the pinned dataset revision."""
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)
    train = Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename="barbados_dapt_packed.jsonl",
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=str(HF_CACHE),
            local_dir=str(target_dir),
        )
    )
    eval_path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename="barbados_dapt_eval.jsonl",
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=str(HF_CACHE),
            local_dir=str(target_dir),
        )
    )
    return train, eval_path


def cmd_doctor(_args: argparse.Namespace) -> int:
    ensure_directories()
    Env.load()
    verify_environment()
    download_base_model()
    LOGGER.info("doctor OK")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    ensure_directories()
    env = Env.load()
    verify_environment()
    manifest = build_current_manifest(image_digest=env.image_digest)
    write_manifest(MANIFEST_PATH, manifest)
    create_bucket_for_run(env.bucket_id, private=True)
    download_base_model()
    train_path, eval_path = download_dataset_files(target_dir=DATA_ROOT)
    LOGGER.info("dataset pinned at %s and %s", train_path, eval_path)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    restore_latest_checkpoint(
        bucket_id=env.bucket_id,
        run_id=env.run_id,
        manifest=manifest,
        target_dir=SMOKE_DIR,
        hf_token=env.hf_token,
    )
    config = smoke_config()
    config = DAPTConfig(**{**config.__dict__,
                            "train_dataset": str(train_path),
                            "eval_dataset": str(eval_path),
                            "output_dir": str(SMOKE_DIR)})
    from bimomni.training.train import run_train

    code = run_train(config)
    uploader = Uploader(
        output_dir=SMOKE_DIR,
        spool_root=DATA_ROOT / "spool",
        bucket_id=env.bucket_id,
        run_id=env.run_id,
        manifest=manifest,
        hf_token=env.hf_token,
        stable_for_seconds=2.0,
    )
    uploaded = uploader.run_once()
    LOGGER.info("smoke checkpoint sync uploaded steps=%s", uploaded)
    return code if code >= 0 else 0


def cmd_sync_once(args: argparse.Namespace) -> int:
    ensure_directories()
    env = Env.load()
    manifest = build_current_manifest(image_digest=env.image_digest)
    write_manifest(MANIFEST_PATH, manifest)
    create_bucket_for_run(env.bucket_id, private=True)
    output_dir = Path(args.output_dir) if args.output_dir else CHECKPOINT_DIR
    uploader = Uploader(
        output_dir=output_dir,
        spool_root=DATA_ROOT / "spool",
        bucket_id=env.bucket_id,
        run_id=env.run_id,
        manifest=manifest,
        hf_token=env.hf_token,
        poll_seconds=2.0,
        stable_for_seconds=2.0,
    )
    uploaded = uploader.run_once()
    LOGGER.info("sync-once uploaded %s", uploaded)
    return 0


def download_adapter_repo(repo_id: str = ADAPTER_MODEL_REPO_ID) -> Path:
    """Fetch the published LoRA adapter into the local HF cache."""
    from huggingface_hub import snapshot_download

    target = HF_CACHE / "adapters" / repo_id.replace("/", "_")
    snapshot_download(
        repo_id,
        cache_dir=str(HF_CACHE),
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return target


def download_fused_repo(repo_id: str = FUSED_MODEL_REPO_ID) -> Path:
    """Fetch a published fused bf16 checkpoint into the local HF cache."""
    from huggingface_hub import snapshot_download

    target = HF_CACHE / "models" / repo_id.replace("/", "_")
    snapshot_download(
        repo_id,
        cache_dir=str(HF_CACHE),
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return target


def _run_fuse(args: argparse.Namespace, env: Env) -> Path:
    """Fuse adapter into the base, drop the talker, and publish the result."""
    base_dir = download_base_model()
    adapter_repo = args.adapter_repo or ADAPTER_MODEL_REPO_ID
    adapter_dir = download_adapter_repo(adapter_repo)
    output_dir = Path(args.output_dir) if args.output_dir else FUSED_DIR
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "fusing base=%s adapter=%s -> %s (drop_talker=True)",
        base_dir,
        adapter_dir,
        output_dir,
    )
    fuse_adapter(base_dir, adapter_dir, output_dir, drop_talker=True)
    write_provenance(output_dir, base_model=BASE_MODEL_ID, adapter=adapter_repo)
    info = FusedModelInfo(
        base_model=BASE_MODEL_ID,
        base_revision=BASE_MODEL_REVISION,
        adapter_repo=adapter_repo,
    )
    fused_repo = args.fused_repo or FUSED_MODEL_REPO_ID
    url = upload_fused(output_dir, info, fused_repo, token=env.hf_token)
    LOGGER.info("fused checkpoint uploaded to %s", url)
    return output_dir


def _convert_mlx(fused_dir: Path, mlx_dir: Path) -> None:
    """Convert a fused bf16 checkpoint to a 4-bit MLX snapshot."""
    from mlx_vlm import convert

    rewrite_mlx_config(fused_dir / "config.json")
    LOGGER.info("converting %s -> %s (4-bit, group size 64)", fused_dir, mlx_dir)
    convert(
        hf_path=str(fused_dir),
        mlx_path=str(mlx_dir),
        quantize=True,
        q_bits=4,
        q_group_size=64,
    )


def _run_mlx(args: argparse.Namespace, env: Env, source_dir: Path | None = None) -> Path:
    """Convert a fused bf16 checkpoint to 4-bit MLX and publish it."""
    if source_dir is None:
        fused_repo = args.fused_repo or FUSED_MODEL_REPO_ID
        source_dir = download_fused_repo(fused_repo)
    mlx_dir = Path(args.mlx_dir) if args.mlx_dir else MLX_DIR
    if mlx_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing {mlx_dir}")
    mlx_dir.parent.mkdir(parents=True, exist_ok=True)
    _convert_mlx(source_dir, mlx_dir)
    kept, dropped = strip_mlx_safetensors(mlx_dir, keep_inputs=True)
    LOGGER.info("mlx safety strip kept=%d dropped=%d", kept, dropped)
    info = FusedModelInfo(
        base_model=BASE_MODEL_ID,
        base_revision=BASE_MODEL_REVISION,
        adapter_repo=args.adapter_repo or ADAPTER_MODEL_REPO_ID,
    )
    mlx_repo = args.mlx_repo or MLX_MODEL_REPO_ID
    url = upload_mlx(mlx_dir, info, mlx_repo, token=env.hf_token)
    LOGGER.info("mlx 4-bit uploaded to %s", url)
    return mlx_dir


def cmd_fuse(args: argparse.Namespace) -> int:
    ensure_directories()
    env = Env.load()
    _run_fuse(args, env)
    LOGGER.info("fuse complete")
    return 0


def cmd_push_adapter(args: argparse.Namespace) -> int:
    """Re-upload a finished run's adapter to the Hub from the checkpoint bucket.

    Used when the train job's in-process `_finalise` upload failed (e.g. an
    uploader regression) but the checkpoint sidecar already persisted the
    adapter to the bucket. Restores the newest complete checkpoint for the
    run regardless of recipe-manifest match and uploads it to the LoRA repo.
    """
    ensure_directories()
    env = Env.load()
    run_id = args.run_id or env.run_id
    target_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINT_DIR
    restored = restore_latest_checkpoint_any(
        bucket_id=env.bucket_id,
        run_id=run_id,
        target_dir=target_dir,
        hf_token=env.hf_token,
    )
    if restored is None:
        LOGGER.error("no complete checkpoint found for run %s", run_id)
        return 1
    LOGGER.info("restored adapter candidate from %s", restored)
    try:
        assert_adapter_loads(restored)
    except Exception as exc:
        LOGGER.error("adapter validation failed: %s", exc)
        return 1
    info = BaseModelInfo(
        base_revision=BASE_MODEL_REVISION,
        adapter_repo=ADAPTER_MODEL_REPO_ID,
        hyperparameters={"lora_rank": 64, "lora_alpha": 128},
    )
    try:
        url = upload_adapter(restored, info, token=env.hf_token)
    except Exception as exc:
        LOGGER.error("adapter upload failed: %s", exc)
        return 1
    LOGGER.info("adapter re-uploaded to %s", url)
    return 0


def cmd_mlx(args: argparse.Namespace) -> int:
    ensure_directories()
    env = Env.load()
    _run_mlx(args, env)
    LOGGER.info("mlx complete")
    return 0


def cmd_finalise(args: argparse.Namespace) -> int:
    ensure_directories()
    env = Env.load()
    fused_dir = _run_fuse(args, env)
    _run_mlx(args, env, source_dir=fused_dir)
    LOGGER.info("finalise complete")
    return 0


def _save_run_state(manifest_hash: str, env: Env, extra: dict[str, Any]) -> None:
    payload = {
        "run_id": env.run_id,
        "image_digest": env.image_digest,
        "manifest_hash": manifest_hash,
        "started_at": time.time(),
        **extra,
    }
    RUN_STATE.parent.mkdir(parents=True, exist_ok=True)
    RUN_STATE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> int:
    ensure_directories()
    env = Env.load()
    verify_environment()
    manifest = build_current_manifest(image_digest=env.image_digest)
    write_manifest(MANIFEST_PATH, manifest)
    create_bucket_for_run(env.bucket_id, private=True)

    download_base_model()
    train_path, eval_path = download_dataset_files(target_dir=DATA_ROOT)

    restored = restore_latest_checkpoint(
        bucket_id=env.bucket_id,
        run_id=env.run_id,
        manifest=manifest,
        target_dir=CHECKPOINT_DIR,
        hf_token=env.hf_token,
    )
    if restored is not None:
        LOGGER.info("restored checkpoint from %s", restored)
    _save_run_state(
        manifest.stable_hash(),
        env,
        {"restored_checkpoint": str(restored) if restored else None},
    )

    uploader = Uploader(
        output_dir=CHECKPOINT_DIR,
        spool_root=DATA_ROOT / "spool",
        bucket_id=env.bucket_id,
        run_id=env.run_id,
        manifest=manifest,
        hf_token=env.hf_token,
    )
    uploader.start()

    guard = build_guard(BUDGET_STATE, budget_hours=float(args.budget))
    LOGGER.info(
        "budget guard loaded; persisted_hours=%.3f cap=%.3f",
        guard.state.consumed_seconds / 3600.0,
        guard.budget_hours,
    )
    config = DAPTConfig(
        budget_hours=float(args.budget),
        train_dataset=str(train_path),
        eval_dataset=str(eval_path),
        output_dir=str(CHECKPOINT_DIR),
    )
    if restored is not None and config.resume_from_checkpoint is None:
        config = DAPTConfig(
            **{
                **config.__dict__,
                "resume_from_checkpoint": str(restored),
            }
        )

    from bimomni.training.train import run_train

    try:
        rc = run_train(
            config,
            budget_state_path=str(BUDGET_STATE),
            persist_budget=True,
        )
    finally:
        uploader.stop()

    if rc < 0:
        LOGGER.warning("training ended via signal %s; finalising", rc)
    _finalise(env=env, manifest=manifest)
    return rc if rc >= 0 else 0


def _finalise(*, env: Env, manifest: Any) -> None:
    latest = max(
        CHECKPOINT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
        default=None,
    )
    if latest is None:
        LOGGER.warning("no checkpoint to upload")
        return
    try:
        assert_adapter_loads(latest)
    except Exception as exc:
        LOGGER.error("adapter validation failed: %s", exc)
        return
    info = BaseModelInfo(
        base_revision=BASE_MODEL_REVISION,
        adapter_repo=ADAPTER_MODEL_REPO_ID,
        hyperparameters={"lora_rank": 64, "lora_alpha": 128},
    )
    try:
        url = upload_adapter(latest, info, token=env.hf_token)
        LOGGER.info("uploaded adapter to %s", url)
    except Exception as exc:
        LOGGER.error("adapter upload failed: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="supervisor", description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("doctor", help="verify CUDA and HF credentials")
    sub.add_parser("smoke", help="two-step training smoke run")
    sync_once = sub.add_parser("sync-once", help="run one checkpoint sync pass")
    sync_once.add_argument("--output-dir", default=str(CHECKPOINT_DIR))

    train = sub.add_parser(
        "train", help="run training under the supervisor (default)"
    )
    train.add_argument("--budget", type=float, default=12.0)

    fuse = sub.add_parser(
        "fuse", help="fuse adapter into base, drop talker, upload fused bf16"
    )
    fuse.add_argument("--adapter-repo", default=None, help="override adapter repo")
    fuse.add_argument("--fused-repo", default=None, help="override fused repo")
    fuse.add_argument("--output-dir", default=None, help="override fused output dir")

    mlx = sub.add_parser(
        "mlx", help="convert fused bf16 to 4-bit MLX and upload"
    )
    mlx.add_argument("--fused-repo", default=None, help="override fused source repo")
    mlx.add_argument("--mlx-repo", default=None, help="override MLX repo")
    mlx.add_argument("--mlx-dir", default=None, help="override MLX output dir")
    mlx.add_argument("--adapter-repo", default=None, help="adapter id for the model card")

    finalise = sub.add_parser(
        "finalise", help="fuse adapter, then convert the result to 4-bit MLX"
    )
    finalise.add_argument("--adapter-repo", default=None, help="override adapter repo")
    finalise.add_argument("--fused-repo", default=None, help="override fused repo")
    finalise.add_argument("--mlx-repo", default=None, help="override MLX repo")
    finalise.add_argument("--output-dir", default=None, help="override fused output dir")
    finalise.add_argument("--mlx-dir", default=None, help="override MLX output dir")

    push_adapter = sub.add_parser(
        "push-adapter",
        help="restore the newest checkpoint from the bucket and upload the adapter",
    )
    push_adapter.add_argument("--run-id", default=None, help="training run to recover")
    push_adapter.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))

    args = parser.parse_args(argv)
    handlers = {
        "doctor": cmd_doctor,
        "smoke": cmd_smoke,
        "sync-once": cmd_sync_once,
        "train": cmd_train,
        "fuse": cmd_fuse,
        "mlx": cmd_mlx,
        "finalise": cmd_finalise,
        "push-adapter": cmd_push_adapter,
    }
    return handlers[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
