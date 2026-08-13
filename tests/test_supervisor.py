"""Tests for the supervisor CLI and Env loader."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

import bimomni.training.supervisor as supervisor


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        stage="doctor",
        budget=12.0,
        output_dir=str(Path("/tmp/checkpoints")),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_env_load_requires_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        supervisor.Env.load()


def test_env_load_uses_provided_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("RUN_ID", "run-x")
    monkeypatch.setenv("IMAGE_DIGEST", "sha256:abc")
    monkeypatch.setenv("HF_BUCKET_REPO_ID", "hammertoe/test")
    env = supervisor.Env.load()
    assert env.hf_token == "secret"
    assert env.run_id == "run-x"
    assert env.image_digest == "sha256:abc"
    assert env.bucket_id == "hammertoe/test"


def test_main_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit):
        supervisor.main(["frobnicate"])


def test_main_routes_train_to_handler() -> None:
    captured = {}

    def fake_train(args):
        captured["train"] = args
        return 0

    with patch.object(supervisor, "cmd_train", side_effect=fake_train):
        assert supervisor.main(["train", "--budget", "5"]) == 0
    assert captured["train"].budget == 5.0


def test_main_routes_doctor() -> None:
    captured = {}

    def fake_doctor(args):
        captured["doctor"] = args
        return 0

    with patch.object(supervisor, "cmd_doctor", side_effect=fake_doctor):
        assert supervisor.main(["doctor"]) == 0
    assert "doctor" in captured


def test_main_routes_sync_once() -> None:
    captured = {}

    def fake_sync(args):
        captured["sync"] = args
        return 0

    with patch.object(supervisor, "cmd_sync_once", side_effect=fake_sync):
        assert supervisor.main(["sync-once", "--output-dir", "/tmp/out"]) == 0
    assert captured["sync"].output_dir == "/tmp/out"


def test_main_routes_fuse() -> None:
    captured = {}

    def fake_fuse(args):
        captured["fuse"] = args
        return 0

    with patch.object(supervisor, "cmd_fuse", side_effect=fake_fuse):
        assert supervisor.main(["fuse"]) == 0
    assert "fuse" in captured


def test_main_routes_mlx() -> None:
    captured = {}

    def fake_mlx(args):
        captured["mlx"] = args
        return 0

    with patch.object(supervisor, "cmd_mlx", side_effect=fake_mlx):
        assert supervisor.main(["mlx"]) == 0
    assert "mlx" in captured


def test_main_routes_finalise() -> None:
    captured = {}

    def fake_finalise(args):
        captured["finalise"] = args
        return 0

    with patch.object(supervisor, "cmd_finalise", side_effect=fake_finalise):
        assert supervisor.main(["finalise"]) == 0
    assert "finalise" in captured


def test_main_routes_push_adapter() -> None:
    captured = {}

    def fake_push(args):
        captured["push"] = args
        return 0

    with patch.object(supervisor, "cmd_push_adapter", side_effect=fake_push):
        assert supervisor.main(["push-adapter", "--run-id", "run-x"]) == 0
    assert captured["push"].run_id == "run-x"


def test_cmd_push_adapter_uploads_restored_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    @dataclass
    class FakeEnv:
        hf_token: str = "secret"
        run_id: str = "recovery-run"
        image_digest: str = ""
        bucket_id: str = "hammertoe/test"

    monkeypatch.setattr(supervisor, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(supervisor, "MANIFEST_PATH", tmp_path / "state/recipe.json")
    monkeypatch.setattr(supervisor, "ensure_directories", lambda: None)
    monkeypatch.setattr(supervisor.Env, "load", classmethod(lambda cls: FakeEnv()))

    def fake_restore_any(**kwargs):
        calls.update(restore=kwargs)
        return tmp_path / "checkpoint-1000"

    monkeypatch.setattr(supervisor, "restore_latest_checkpoint_any", fake_restore_any)

    def fake_assert(path):
        assert str(path).endswith("checkpoint-1000")

    monkeypatch.setattr(supervisor, "assert_adapter_loads", fake_assert)

    def fake_upload(path, info, token):
        calls.update(upload=(path, info.adapter_repo, token))
        return "https://hf.co/adapter"

    monkeypatch.setattr(supervisor, "upload_adapter", fake_upload)

    args = argparse.Namespace(run_id="train-run-x", checkpoint_dir=None)
    assert supervisor.cmd_push_adapter(args) == 0
    assert calls["restore"]["run_id"] == "train-run-x"
    assert calls["upload"][1] == supervisor.ADAPTER_MODEL_REPO_ID


def test_run_fuse_fuses_with_drop_talker_and_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}
    monkeypatch.setattr(supervisor, "download_base_model", lambda: tmp_path / "base")
    monkeypatch.setattr(supervisor, "download_adapter_repo", lambda repo: tmp_path / "adapter")
    monkeypatch.setattr(supervisor, "fuse_adapter", lambda base, adapter, out, drop_talker: calls.update(fused=(base, adapter, out, drop_talker)) or None)
    monkeypatch.setattr(supervisor, "write_provenance", lambda *a, **k: None)
    monkeypatch.setattr(supervisor, "upload_fused", lambda out, info, repo, token: f"https://hf.co/{repo}")
    monkeypatch.setenv("HF_TOKEN", "secret")

    out_dir = tmp_path / "fused-bf16"
    args = argparse.Namespace(
        adapter_repo=None,
        fused_repo=None,
        output_dir=str(out_dir),
        mlx_dir=None,
        mlx_repo=None,
    )
    env = supervisor.Env.load()

    result = supervisor._run_fuse(args, env)

    assert result == out_dir
    assert calls["fused"][3] is True
    assert calls["fused"][2] == out_dir


def test_run_mlx_converts_and_strips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}
    monkeypatch.setattr(supervisor, "download_fused_repo", lambda repo: tmp_path / "fused")
    monkeypatch.setattr(
        supervisor,
        "_convert_mlx",
        lambda fused, out: calls.update(convert=(fused, out)) or None,
    )
    monkeypatch.setattr(supervisor, "strip_mlx_safetensors", lambda path, keep_inputs=True: (0, 0))
    monkeypatch.setattr(supervisor, "upload_mlx", lambda out, info, repo, token: f"https://hf.co/{repo}")
    monkeypatch.setenv("HF_TOKEN", "secret")

    mlx_dir = tmp_path / "mlx-4bit"
    args = argparse.Namespace(
        adapter_repo=None,
        fused_repo=None,
        mlx_repo=None,
        mlx_dir=str(mlx_dir),
        output_dir=None,
    )
    env = supervisor.Env.load()

    result = supervisor._run_mlx(args, env)

    assert result == mlx_dir
    assert calls["convert"][1] == mlx_dir


def test_ensure_directories_creates_state_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.setenv("DAPT_DATA_ROOT", str(tmp_path))
    reloaded = importlib.reload(supervisor)
    reloaded.ensure_directories()
    assert (tmp_path / "checkpoints").exists()
    assert (tmp_path / "state").exists()
    assert (tmp_path / "hf").exists()
    importlib.reload(supervisor)


def test_cmd_smoke_restores_and_uploads_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class FakeEnv:
        hf_token: str = "secret"
        run_id: str = "smoke-run"
        image_digest: str = "sha256:abc"
        bucket_id: str = "hammertoe/test"

    class FakeUploader:
        def __init__(self, **kwargs):
            assert kwargs["output_dir"] == tmp_path / "smoke"

        def run_once(self):
            return [2]

    manifest = object()
    restored: dict[str, object] = {}
    monkeypatch.setattr(supervisor, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(supervisor, "MANIFEST_PATH", tmp_path / "state/recipe.json")
    monkeypatch.setattr(supervisor.Env, "load", classmethod(lambda cls: FakeEnv()))
    monkeypatch.setattr(supervisor, "ensure_directories", lambda: None)
    monkeypatch.setattr(supervisor, "verify_environment", lambda: None)
    monkeypatch.setattr(supervisor, "build_current_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(supervisor, "write_manifest", lambda *args: None)
    monkeypatch.setattr(supervisor, "create_bucket_for_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "download_base_model", lambda: tmp_path / "base")
    monkeypatch.setattr(
        supervisor,
        "download_dataset_files",
        lambda **kwargs: (tmp_path / "train.jsonl", tmp_path / "eval.jsonl"),
    )

    def fake_restore(**kwargs):
        restored.update(kwargs)
        return None

    monkeypatch.setattr(supervisor, "restore_latest_checkpoint", fake_restore)
    monkeypatch.setattr(supervisor, "Uploader", FakeUploader)
    monkeypatch.setattr("bimomni.training.train.run_train", lambda config: 0)

    assert supervisor.cmd_smoke(_args(stage="smoke")) == 0
    assert restored["run_id"] == "smoke-run"
    assert restored["target_dir"] == tmp_path / "smoke"
