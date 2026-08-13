"""ms-swift DAPT training command for Qwen3-Omni, text-only thinker LoRA.

Builds and runs the `swift pt` continued-pretraining command with the locked
recipe (LoRA r=64 alpha=128, thinker self-attention only, bf16, 4096-token
sequences, pure Barbados corpus) and wraps it with a BudgetGuard.

The run is checkpoint/resume safe against spot preemption: checkpoints land
in a stable output dir (add_version disabled) and a re-run auto-resumes from
the latest checkpoint with optimizer + RNG state restored.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from bimomni.training.budget_guard import (
    BudgetGuard,
    BudgetState,
    GPU_LIMIT_HOURS,
    build_guard,
    load_state,
    save_state,
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DATA_ROOT = os.environ.get("DAPT_DATA_ROOT", "/data/v3")
LORA_RANK = 64
LORA_ALPHA = 128
MAX_LENGTH = 4096
LEARNING_RATE = 1e-4
THINKER_ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
THINKER_MLP_PARAMETERS = ("gate_up_proj", "down_proj")


def default_dataset_num_proc() -> int:
    """Default worker count for ms-swift dataset preprocessing.

    Forced to 1: multiprocess tokenization of the 225k-row DAPT corpus on the
    h200 flavor (os.cpu_count()==192) spawned 192 workers that exhausted the
    256 GiB pod RAM (OOMKilled). Single-proc runs the same pass at ~160
    examples/s with zero loss in throughput and a fraction of the memory.
    """
    return 1


@dataclass(frozen=True)
class DAPTConfig:
    model_id: str = MODEL_ID
    train_dataset: str = f"{DATA_ROOT}/barbados_dapt_packed.jsonl"
    eval_dataset: str = f"{DATA_ROOT}/barbados_dapt_eval.jsonl"
    output_dir: str = f"{DATA_ROOT}/checkpoints"
    lora_rank: int = LORA_RANK
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = 0.0
    max_length: int = MAX_LENGTH
    learning_rate: float = LEARNING_RATE
    num_train_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation: int = 16
    max_steps: int | None = None
    save_steps: int = 100
    train_samples: int | None = None
    budget_hours: float = GPU_LIMIT_HOURS
    resume_from_checkpoint: str | None = None
    dataset_num_proc: int = field(default_factory=default_dataset_num_proc)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the highest-step checkpoint dir under output_dir, or None."""
    root = Path(output_dir)
    if not root.exists():
        return None
    checkpoints = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def build_swift_command(config: DAPTConfig) -> list[str]:
    """Build the ms-swift `swift pt` command from a locked config.

    Argument names follow ms-swift 4.4 official examples: `swift pt` with
    `--dataset` (not --train_dataset), `--tuner_type lora` (not --train_type),
    and `--use_hf true` so the model + dataset come from Hugging Face instead
    of the ModelScope default.
    """
    command = [
        "swift",
        "pt",
        "--model",
        config.model_id,
        "--use_hf",
        "true",
        "--tuner_type",
        "lora",
        "--lora_rank",
        str(config.lora_rank),
        "--lora_alpha",
        str(config.lora_alpha),
        "--lora_dropout",
        str(config.lora_dropout),
        "--target_modules",
        *THINKER_ATTENTION_MODULES,
        "--target_parameters",
        *THINKER_MLP_PARAMETERS,
        "--torch_dtype",
        "bfloat16",
        "--dataset",
        (
            f"{config.train_dataset}#{config.train_samples}"
            if config.train_samples is not None
            else config.train_dataset
        ),
        "--val_dataset",
        config.eval_dataset,
        "--eval_strategy",
        "no",
        "--output_dir",
        config.output_dir,
        "--add_version",
        "false",
        "--num_train_epochs",
        str(config.num_train_epochs),
        "--max_length",
        str(config.max_length),
        "--per_device_train_batch_size",
        str(config.batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation),
        "--learning_rate",
        str(config.learning_rate),
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.03",
        "--packing",
        "true",
        "--use_chat_template",
        "false",
        "--dataset_num_proc",
        str(config.dataset_num_proc),
        "--attn_impl",
        "flash_attn",
        "--gradient_checkpointing",
        "true",
        "--logging_steps",
        "10",
        "--save_steps",
        str(config.save_steps),
        "--save_total_limit",
        "2",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--freeze_llm",
        "false",
    ]
    if config.max_steps is not None:
        command += ["--max_steps", str(config.max_steps)]
    if config.resume_from_checkpoint is not None:
        command += ["--resume_from_checkpoint", config.resume_from_checkpoint]
    return command


def run_train(
    config: DAPTConfig = DAPTConfig(),
    *,
    budget_state_path: str | None = None,
    persist_budget: bool = True,
) -> int:
    """Run swift pt under a BudgetGuard; returns the subprocess return code.

    Auto-resumes from the latest checkpoint in output_dir unless an explicit
    checkpoint is supplied, so a spot-preempted run picks up where it stopped.

    The anchored target regex confines adapters to thinker self-attention.
    `ENABLE_AUDIO_OUTPUT=0`, `--freeze_vit true`, and `--freeze_aligner true`
    additionally keep multimodal output and towers disabled for text-only DAPT.

    When `budget_state_path` is provided, the prior accumulated GPU-seconds
    counter is restored before training and updated in place during the run.
    The state file lets a 12-hour cap span multiple pod lifetimes.
    """
    if config.resume_from_checkpoint is None:
        config = _resume_config(config)
    command = build_swift_command(config)
    state_path = (
        Path(budget_state_path) if budget_state_path
        else Path(os.environ.get("DAPT_BUDGET_STATE", f"{DATA_ROOT}/state/budget.json"))
    )
    if state_path.exists() and persist_budget:
        guard = build_guard(state_path, budget_hours=config.budget_hours)
        print(
            f"[train] resumed budget: {guard.state.consumed_seconds:.0f}s "
            f"of {config.budget_hours * 3600.0:.0f}s",
            flush=True,
        )
    else:
        guard = BudgetGuard(
            budget_hours=config.budget_hours,
            gpu_count=1,
            state=BudgetState(),
        )
    print(f"[train] budget guard: {config.budget_hours} GPU-hours cap", flush=True)
    if config.resume_from_checkpoint:
        print(f"[train] resuming from {config.resume_from_checkpoint}", flush=True)
    print(f"[train] command: {' '.join(command)}", flush=True)

    env = {**os.environ, "ENABLE_AUDIO_OUTPUT": "0"}
    process = subprocess.Popen(command, env=env)
    try:
        while True:
            try:
                code = process.wait(timeout=guard.poll_seconds)
                guard.sample()
                if persist_budget:
                    save_state(state_path, guard.state)
                return code
            except subprocess.TimeoutExpired:
                guard.sample()
                if persist_budget:
                    save_state(state_path, guard.state)
                guard.check_and_abort(signal_fn=lambda sig: process.terminate())
                continue
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        if persist_budget:
            save_state(state_path, guard.state)


def _resume_config(config: DAPTConfig) -> DAPTConfig:
    """Copy config with resume_from_checkpoint set to the latest checkpoint."""
    checkpoint = find_latest_checkpoint(config.output_dir)
    if checkpoint is None:
        return config
    return DAPTConfig(**{**config.__dict__, "resume_from_checkpoint": checkpoint})


def smoke_config() -> DAPTConfig:
    """A 2-step run into a scratch dir so resume never touches the real run."""
    return DAPTConfig(
        output_dir=f"{DATA_ROOT}/smoke-checkpoints",
        max_steps=2,
        save_steps=2,
        num_train_epochs=1,
        train_samples=64,
    )
