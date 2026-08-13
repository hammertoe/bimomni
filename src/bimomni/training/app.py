"""BimOmni training CLI entrypoint: download / prepare / train / evaluate / upload.

Each stage is idempotent and can be re-run; upload is gated on a clean
evaluate exit. Intended to run inside the BimOmni training container on the rental.

The recommended container path is `python -m bimomni.training.supervisor <stage>`
which manages restore, upload, budget persistence, and signal handling.
This CLI is retained for unit-test harnesses and CI smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bimomni.training.recipe import (
    BASE_MODEL_ID as MODEL_ID,
)
from bimomni.training.recipe import (
    DATASET_REPO_ID,
    DATASET_REVISION,
)

DATA_ROOT = Path(os.environ.get("DAPT_DATA_ROOT", "/data/v3"))
PACKED_PATH = DATA_ROOT / "barbados_dapt_packed.jsonl"
EVAL_PATH = DATA_ROOT / "barbados_dapt_eval.jsonl"
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
HF_CACHE = DATA_ROOT / "hf"
ADAPTER_REPO = "hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4"


def cmd_download(_args: argparse.Namespace) -> int:
    """Pull the base model and pinned dataset files into the local HF cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(MODEL_ID, cache_dir=str(HF_CACHE))
    PACKED_PATH.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="barbados_dapt_packed.jsonl",
        repo_type="dataset",
        revision=DATASET_REVISION,
        cache_dir=str(HF_CACHE),
        local_dir=str(PACKED_PATH.parent),
    )
    hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="barbados_dapt_eval.jsonl",
        repo_type="dataset",
        revision=DATASET_REVISION,
        cache_dir=str(HF_CACHE),
        local_dir=str(EVAL_PATH.parent),
    )
    if not PACKED_PATH.exists() or not EVAL_PATH.exists():
        print("[download] dataset files missing", file=sys.stderr)
        return 1
    print(f"[download] base model cached and dataset files staged at {DATA_ROOT}")
    return 0


def cmd_prepare(_args: argparse.Namespace) -> int:
    """Verify the staged train/eval JSONL files exist and are well-formed."""
    if not PACKED_PATH.exists() or not EVAL_PATH.exists():
        print("[prepare] missing dataset files; run download first", file=sys.stderr)
        return 1
    with PACKED_PATH.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    if not first.strip():
        print(f"[prepare] {PACKED_PATH} is empty", file=sys.stderr)
        return 1
    train_count = sum(1 for _ in PACKED_PATH.open("r", encoding="utf-8"))
    eval_count = sum(1 for _ in EVAL_PATH.open("r", encoding="utf-8"))
    print(json.dumps({
        "packed_path": str(PACKED_PATH),
        "eval_path": str(EVAL_PATH),
        "train_records": train_count,
        "eval_records": eval_count,
    }, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run ms-swift DAPT under a BudgetGuard cap.

    If the budget guard terminates training (negative return code = killed by
    signal), persist whatever checkpoint exists: evaluate, then upload.
    """
    from bimomni.training.train import DAPTConfig, run_train

    if args.smoke:
        from bimomni.training.train import smoke_config

        config = smoke_config()
        print("[train] SMOKE MODE: 2 steps into /data/smoke_checkpoints", flush=True)
    else:
        config = DAPTConfig(
            budget_hours=float(os.environ.get("PRIME_BUDGET_HOURS", args.budget)),
        )
    code = run_train(config)
    if code >= 0:
        return code

    print(f"[train] training stopped (signal {code}); running evaluate + upload", flush=True)
    if _latest_checkpoint() is None:
        print("[train] no checkpoint to persist; skipping evaluate/upload", file=sys.stderr)
        return 1
    eval_code = cmd_evaluate(args)
    if eval_code != 0:
        return eval_code
    return cmd_upload(args)


def cmd_evaluate(_args: argparse.Namespace) -> int:
    """Run the four evaluation gates; exit non-zero on any failure."""
    from bimomni.evaluation.evaluate import run_all_gates

    adapter = _latest_checkpoint()
    if adapter is None:
        print("[evaluate] no adapter found under /data/checkpoints", file=sys.stderr)
        return 1
    reports = run_all_gates(adapter=adapter)
    failed = [r for r in reports if not r.passed]
    for report in reports:
        print(f"[evaluate] {report.name}: {'PASS' if report.passed else 'FAIL'}")
        if report.detail:
            print(report.detail)
    return 1 if failed else 0


def cmd_upload(args: argparse.Namespace) -> int:
    """Assert the adapter loads, then push adapter + card to the Hub."""
    from bimomni.publish.upload import BaseModelInfo, assert_adapter_loads, upload

    adapter = _latest_checkpoint()
    if adapter is None:
        print("[upload] no adapter found under /data/checkpoints", file=sys.stderr)
        return 1
    assert_adapter_loads(adapter)
    token = args.token or os.environ.get("HF_TOKEN")
    info = BaseModelInfo(
        base_revision=_base_revision(adapter),
        record_count=args.records,
        token_estimate=args.tokens,
        budget_hours=float(os.environ.get("PRIME_BUDGET_HOURS", 12.0)),
        hyperparameters={"lora_rank": 64, "lora_alpha": 128},
    )
    url = upload(adapter, info, token=token)
    print(f"[upload] pushed adapter to {url}")
    return 0


def _latest_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        (p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def _base_revision(_adapter: Path) -> str:
    return os.environ.get("BASE_REVISION", "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.py", description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("download", help="cache base model + verify corpus")
    sub.add_parser("prepare", help="pack corpus into train/eval DAPT JSONL")

    train = sub.add_parser("train", help="run ms-swift DAPT under budget guard")
    train.add_argument("--budget", type=float, default=12.0, help="GPU-hour cap")
    train.add_argument(
        "--smoke", action="store_true", help="2-step smoke run into a scratch dir"
    )
    sub.add_parser("evaluate", help="run the four eval gates")
    upload = sub.add_parser("upload", help="push adapter + model card to Hub")
    upload.add_argument("--records", type=int, default=0, help="records after dedup")
    upload.add_argument("--tokens", type=int, default=0, help="token estimate")
    upload.add_argument("--token", default=None, help="HF_TOKEN (or env)")

    args = parser.parse_args(argv)
    handlers = {
        "download": cmd_download,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "upload": cmd_upload,
    }
    return handlers[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
