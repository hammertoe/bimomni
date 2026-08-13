"""Tests for the persistent GPU-second budget guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from bimomni.training.budget_guard import (
    BudgetGuard,
    BudgetState,
    build_guard,
    load_state,
    parse_nvidia_smi_power,
    save_state,
)


def test_parse_nvidia_smi_power_handles_units_and_idle() -> None:
    assert parse_nvidia_smi_power("457.23W") == 457.23
    assert parse_nvidia_smi_power("457.23") == 457.23
    assert parse_nvidia_smi_power("0 W") == 0.0
    assert parse_nvidia_smi_power("") == 0.0
    assert parse_nvidia_smi_power("N/A") == 0.0


def test_parse_nvidia_smi_power_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_nvidia_smi_power("not-a-number")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    state = BudgetState(consumed_seconds=7200.0, last_updated_monotonic=42.0)
    target = tmp_path / "budget.json"
    save_state(target, state)
    loaded = load_state(target)
    assert loaded.consumed_seconds == 7200.0
    assert loaded.consumed_hours == 2.0


def test_load_missing_returns_zero(tmp_path: Path) -> None:
    state = load_state(tmp_path / "missing.json")
    assert state.consumed_seconds == 0.0


def test_load_corrupt_returns_zero(tmp_path: Path) -> None:
    target = tmp_path / "budget.json"
    target.write_text("not json", encoding="utf-8")
    assert load_state(target).consumed_seconds == 0.0


def test_budget_guard_only_counts_active_intervals() -> None:
    power = {"watts": 100.0}
    wall = {"now": 100.0}

    def fake_smi() -> float:
        return power["watts"]

    def fake_clock() -> float:
        return wall["now"]

    guard = BudgetGuard(
        budget_hours=1.0,
        smi_power_fn=fake_smi,
        wall_clock=fake_clock,
        poll_seconds=0.001,
    )
    wall["now"] = 100.0
    guard.sample()
    wall["now"] = 200.0  # active interval
    guard.sample()
    wall["now"] = 400.0  # idle interval (next sample sees power=0)
    power["watts"] = 0.0
    guard.sample()
    wall["now"] = 700.0  # active interval again
    power["watts"] = 500.0
    guard.sample()
    wall["now"] = 1000.0  # active 700→1000
    guard.sample()
    # 100→200 (100s) + 200→400 (200s) + 700→1000 (300s) = 600s; idle not counted.
    assert guard.state.consumed_seconds == pytest.approx(600.0)


def test_budget_guard_exceeds_only_when_over_cap() -> None:
    power = {"watts": 600.0}
    wall = {"now": 0.0}

    def fake_smi() -> float:
        return power["watts"]

    def fake_clock() -> float:
        return wall["now"]

    guard = BudgetGuard(
        budget_hours=0.001,
        smi_power_fn=fake_smi,
        wall_clock=fake_clock,
        poll_seconds=0.001,
    )
    wall["now"] = 0.0
    guard.sample()
    wall["now"] = 60.0
    guard.sample()
    assert guard.exceeds()


def test_build_guard_restores_state(tmp_path: Path) -> None:
    state_path = tmp_path / "budget.json"
    save_state(state_path, BudgetState(consumed_seconds=3600.0))
    guard = build_guard(state_path, budget_hours=12.0)
    assert guard.state.consumed_seconds == 3600.0
    assert guard.budget_hours == 12.0


def test_persistence_across_instances(tmp_path: Path) -> None:
    state_path = tmp_path / "budget.json"
    power = {"watts": 400.0}
    wall = {"now": 0.0}

    def fake_smi() -> float:
        return power["watts"]

    def fake_clock() -> float:
        return wall["now"]

    guard_a = BudgetGuard(
        budget_hours=12.0,
        smi_power_fn=fake_smi,
        wall_clock=fake_clock,
        poll_seconds=0.001,
    )
    wall["now"] = 0.0
    guard_a.sample()
    wall["now"] = 1500.0  # active for 1500s
    guard_a.sample()
    save_state(state_path, guard_a.state)
    assert guard_a.state.consumed_seconds == pytest.approx(1500.0)

    guard_b = build_guard(
        state_path,
        budget_hours=12.0,
        smi_power_fn=fake_smi,
        wall_clock=fake_clock,
    )
    # The post-restore first sample sets the new window; the next active
    # sample accumulates the time since restore.
    wall["now"] = 2000.0
    guard_b.sample()
    wall["now"] = 2500.0  # 500s after restore
    guard_b.sample()
    assert guard_b.state.consumed_seconds == pytest.approx(2000.0)
    save_state(state_path, guard_b.state)
    final = load_state(state_path)
    assert final.consumed_seconds == pytest.approx(2000.0)
