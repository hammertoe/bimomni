"""Tests for bimomni.training.budget_guard: GPU-hour accounting and abort logic."""

from __future__ import annotations

import pytest

from bimomni.training.budget_guard import (
    GPU_LIMIT_HOURS,
    elapsed_gpu_hours,
    parse_nvidia_smi_power,
    exceeds_budget,
    BudgetGuard,
)


def test_elapsed_gpu_hours_scales_with_gpu_count() -> None:
    assert elapsed_gpu_hours(started=0.0, now=7200.0, gpu_count=1) == pytest.approx(2.0)
    assert elapsed_gpu_hours(started=0.0, now=7200.0, gpu_count=4) == pytest.approx(8.0)


def test_elapsed_gpu_hours_zero_before_start() -> None:
    assert elapsed_gpu_hours(started=10.0, now=5.0, gpu_count=1) == 0.0


def test_exceeds_budget_uses_default_limit() -> None:
    assert exceeds_budget(hours=GPU_LIMIT_HOURS - 0.01) is False
    assert exceeds_budget(hours=GPU_LIMIT_HOURS + 0.01) is True


def test_exceeds_budget_scales_gpu_count_before_comparison() -> None:
    assert exceeds_budget(hours=elapsed_gpu_hours(0, 3 * 3600, 4)) is True


def test_parse_nvidia_smi_power_handles_common_formats() -> None:
    assert parse_nvidia_smi_power("457.23W") == 457.23
    assert parse_nvidia_smi_power("457.23") == 457.23
    assert parse_nvidia_smi_power("0 W") == 0.0
    assert parse_nvidia_smi_power("N/A") == 0.0


def test_parse_nvidia_smi_power_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_nvidia_smi_power("bogus")


def test_budget_guard_aborts_at_budget(tmp_path) -> None:
    """Active GPU across a 1h window exhausts the 1h budget."""
    import signal

    calls: list[int] = []
    current_time = {"value": 0.0}

    def fake_smi_power() -> float:
        return 400.0

    def clock() -> float:
        return current_time["value"]

    guard = BudgetGuard(
        budget_hours=1.0,
        gpu_count=1,
        smi_power_fn=fake_smi_power,
        wall_clock=clock,
        poll_seconds=0.001,
    )

    # First sample just opens the window.
    current_time["value"] = 0.0
    guard.sample()
    # Second sample after 3600 active seconds.
    current_time["value"] = 3600.0
    guard.sample()
    assert guard.exceeds() is True
    guard.check_and_abort(signal_fn=calls.append)
    assert calls == [signal.SIGTERM]


def test_budget_guard_no_abort_under_budget(tmp_path) -> None:
    calls: list[str] = []

    def fake_smi_power() -> float:
        return 400.0

    guard = BudgetGuard(
        budget_hours=10.0,
        gpu_count=1,
        smi_power_fn=fake_smi_power,
        wall_clock=lambda: 3600.0,
    )

    assert guard.exceeds() is False
    guard.check_and_abort(signal_fn=calls.append)
    assert calls == []


def test_budget_guard_ignores_wall_time_when_gpu_idle(tmp_path) -> None:
    """Idle GPU (low/zero power draw) should not burn budget."""
    calls: list[str] = []

    def fake_smi_power() -> float:
        return 0.0  # idle

    guard = BudgetGuard(
        budget_hours=1.0,
        gpu_count=1,
        smi_power_fn=fake_smi_power,
        wall_clock=lambda: 7200.0,  # 2h wall but idle
    )

    assert guard.exceeds() is False
    guard.check_and_abort(signal_fn=calls.append)
    assert calls == []
