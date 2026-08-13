"""Tests for bimomni.training.train: ms-swift DAPT command construction."""

from __future__ import annotations


from bimomni.training.train import build_swift_command, DAPTConfig, find_latest_checkpoint


def test_build_swift_command_uses_pt_subcommand() -> None:
    config = DAPTConfig()
    command = build_swift_command(config)
    assert command[0] == "swift"
    assert command[1] == "pt"


def test_build_swift_command_locked_recipe() -> None:
    config = DAPTConfig()
    command = build_swift_command(config)
    text = " ".join(command)

    assert "--model Qwen/Qwen3-Omni-30B-A3B-Instruct" in text
    assert "--use_hf true" in text
    assert "--tuner_type lora" in text
    assert "--lora_rank 64" in text
    assert "--lora_alpha 128" in text
    assert "--lora_dropout 0.0" in text
    assert "--torch_dtype bfloat16" in text
    assert command[command.index("--target_modules") + 1 : command.index("--target_parameters")] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert command[command.index("--target_parameters") + 1 : command.index("--torch_dtype")] == [
        "gate_up_proj",
        "down_proj",
    ]
    assert "--freeze_vit true" in text
    assert "--freeze_aligner true" in text
    assert "--freeze_llm false" in text
    assert "--packing true" in text
    assert "--max_length 4096" in text
    assert "--num_train_epochs 1" in text
    assert "--learning_rate 0.0001" in text  # == 1e-4
    assert "--lr_scheduler_type cosine" in text
    assert "--warmup_ratio 0.03" in text
    assert "--attn_impl flash_attn" in text
    assert "--gradient_checkpointing true" in text
    assert "--eval_strategy no" in text
    assert "--save_steps 100" in text


def test_build_swift_command_uses_dataset_paths() -> None:
    config = DAPTConfig(
        train_dataset="/data/v3/barbados_dapt_packed.jsonl",
        eval_dataset="/data/v3/barbados_dapt_eval.jsonl",
        output_dir="/data/v3/checkpoints",
    )
    command = build_swift_command(config)
    text = " ".join(command)
    assert "--dataset /data/v3/barbados_dapt_packed.jsonl" in text
    assert "--val_dataset /data/v3/barbados_dapt_eval.jsonl" in text
    assert "--output_dir /data/v3/checkpoints" in text


def test_build_swift_command_disables_chat_template_for_pt() -> None:
    command = build_swift_command(DAPTConfig())
    text = " ".join(command)
    assert "--use_chat_template false" in text


def test_build_swift_command_defaults_dataset_num_proc_to_one() -> None:
    command = build_swift_command(DAPTConfig())
    text = " ".join(command)
    assert "--dataset_num_proc" in text
    assert "--dataset_num_proc 1" in text


def test_build_swift_command_honours_dataset_num_proc_override() -> None:
    config = DAPTConfig(dataset_num_proc=32)
    text = " ".join(build_swift_command(config))
    assert "--dataset_num_proc 32" in text


def test_build_swift_command_disables_add_version_for_stable_resume() -> None:
    text = " ".join(build_swift_command(DAPTConfig()))
    assert "--add_version false" in text


def test_build_swift_command_adds_resume_flag_when_configured() -> None:
    config = DAPTConfig(resume_from_checkpoint="/data/checkpoints/checkpoint-1000")
    text = " ".join(build_swift_command(config))
    assert "--resume_from_checkpoint /data/checkpoints/checkpoint-1000" in text


def test_build_swift_command_omits_resume_flag_by_default() -> None:
    assert "--resume_from_checkpoint" not in " ".join(build_swift_command(DAPTConfig()))


def test_build_swift_command_samples_train_dataset_when_configured() -> None:
    config = DAPTConfig(
        train_dataset="/data/v3/barbados_dapt_packed.jsonl",
        train_samples=64,
    )
    command = build_swift_command(config)
    assert command.count("--dataset") == 1
    assert "--dataset /data/v3/barbados_dapt_packed.jsonl#64" in " ".join(command)


def test_build_swift_command_keeps_full_dataset_by_default() -> None:
    config = DAPTConfig(train_dataset="/data/v3/barbados_dapt_packed.jsonl")
    command = build_swift_command(config)
    assert "--dataset /data/v3/barbados_dapt_packed.jsonl" in " ".join(command)
    assert "#" not in command[command.index("--dataset") + 1]


def test_find_latest_checkpoint_returns_highest_step(tmp_path) -> None:
    output = tmp_path / "checkpoints"
    for step in ("checkpoint-100", "checkpoint-5000", "checkpoint-50"):
        (output / step).mkdir(parents=True)

    assert find_latest_checkpoint(str(output)) == str(output / "checkpoint-5000")


def test_find_latest_checkpoint_none_when_missing(tmp_path) -> None:
    assert find_latest_checkpoint(str(tmp_path / "nope")) is None


def test_find_latest_checkpoint_ignores_non_checkpoint_dirs(tmp_path) -> None:
    output = tmp_path / "checkpoints"
    (output / "checkpoint-100").mkdir(parents=True)
    (output / "runs").mkdir()

    assert find_latest_checkpoint(str(output)) == str(output / "checkpoint-100")


def test_targeting_restores_v1_attention_and_mlp_parameters() -> None:
    command = build_swift_command(DAPTConfig())
    targets = command[command.index("--target_modules") + 1 : command.index("--target_parameters")]
    parameters = command[command.index("--target_parameters") + 1 : command.index("--torch_dtype")]
    assert targets == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert parameters == ["gate_up_proj", "down_proj"]


def test_smoke_config_runs_into_scratch_dir() -> None:
    from bimomni.training.train import smoke_config

    config = smoke_config()
    assert config.output_dir == "/data/v3/smoke-checkpoints"
    assert config.max_steps == 2
    assert config.save_steps == 2
