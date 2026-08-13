"""Tests for bimomni.cli.app CLI dispatch and checkpoint selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from bimomni.training import app


def test_latest_checkpoint_returns_highest_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-5000").mkdir()
    (tmp_path / "checkpoint-50").mkdir()

    latest = app._latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-5000"


def test_latest_checkpoint_none_when_missing(tmp_path: Path) -> None:
    assert app._latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_ignores_non_checkpoint_dirs(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "runs").mkdir()
    assert app._latest_checkpoint(tmp_path).name == "checkpoint-100"


def test_main_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit):
        app.main(["frobnicate"])
