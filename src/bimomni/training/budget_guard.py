"""GPU-hour budget enforcement for the training run.

Counts only active GPU seconds (power >= IDLE_POWER_WATTS) and persists the
running total so a preempted pod cannot reset the 12-hour cap on the next
instance. The BudgetGuard class is the in-memory view; a thin persistence
helper writes the counter under `<DATA_ROOT>/state/budget.json` so resume
inherits the prior spend.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

GPU_LIMIT_HOURS = 12.0
IDLE_POWER_WATTS = 25.0
POLL_SECONDS = 60.0

SmiPowerFn = Callable[[], float]
WallClockFn = Callable[[], float]
SignalFn = Callable[[int], None]


def elapsed_gpu_hours(started: float, now: float, gpu_count: int = 1) -> float:
    """Wall seconds converted to GPU-hours, never negative."""
    if now <= started:
        return 0.0
    return (now - started) / 3600.0 * gpu_count


def exceeds_budget(hours: float, limit: float = GPU_LIMIT_HOURS) -> bool:
    return hours >= limit


def parse_nvidia_smi_power(raw: str) -> float:
    """Parse `nvidia-smi --query-gpu=power.draw` output into watts.

    Accepts values like "457.23W", "457.23", "0 W". Sentinels for a missing
    or idle GPU ("N/A", empty) return 0.0 so an idle sample never burns budget.
    """
    text = raw.strip()
    if not text or text.upper() == "N/A":
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"unparseable power.draw value: {raw!r}")
    return float(match.group(1))


def read_nvidia_smi_power() -> float:
    """Sample power.draw from the first GPU via nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return 0.0
    first = out.splitlines()[0] if out.splitlines() else ""
    return parse_nvidia_smi_power(first)


@dataclass
class BudgetState:
    """Persistent GPU-second counter shared across pod lifetimes."""

    consumed_seconds: float = 0.0
    last_updated_monotonic: float | None = None

    @property
    def consumed_hours(self) -> float:
        return self.consumed_seconds / 3600.0


def load_state(path: Path) -> BudgetState:
    """Read the persisted budget counter; missing file yields zero seconds."""
    if not path.exists():
        return BudgetState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return BudgetState()
    consumed = float(payload.get("consumed_seconds", 0.0))
    return BudgetState(consumed_seconds=max(0.0, consumed))


def save_state(path: Path, state: BudgetState) -> None:
    """Atomically persist the budget counter so resume continues the same cap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "consumed_seconds": state.consumed_seconds,
        "last_updated_monotonic": state.last_updated_monotonic,
    }
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class BudgetGuard:
    """Tracks active GPU-hours and signals the caller at the budget cap."""

    budget_hours: float = GPU_LIMIT_HOURS
    gpu_count: int = 1
    idle_power_watts: float = IDLE_POWER_WATTS
    smi_power_fn: SmiPowerFn = read_nvidia_smi_power
    wall_clock: WallClockFn = time.monotonic
    poll_seconds: float = POLL_SECONDS
    state: BudgetState | None = None

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = BudgetState()
        self._window_started: float = self.wall_clock()
        self._last_sample_was_active: bool = False

    def reset_window(self) -> None:
        """Start a fresh active-interval window from the current wall clock."""
        self._window_started = self.wall_clock()
        self._last_sample_was_active = False

    def sample(self) -> float:
        """Poll power once; accumulate the prior window if it was active."""
        now = self.wall_clock()
        is_active = self.smi_power_fn() >= self.idle_power_watts
        if self._last_sample_was_active and now >= self._window_started:
            elapsed = now - self._window_started
            if elapsed > 0.0:
                self.state.consumed_seconds += elapsed * self.gpu_count
        self._window_started = now
        self._last_sample_was_active = is_active
        return self.consumed_hours()

    def consumed_hours(self) -> float:
        """Return the total active GPU-hours so far, including this window."""
        live = self.state.consumed_seconds
        if self._last_sample_was_active:
            live += max(0.0, self.wall_clock() - self._window_started) * self.gpu_count
        return live / 3600.0

    def exceeds(self) -> bool:
        return self.consumed_hours() >= self.budget_hours

    def check_and_abort(self, signal_fn: SignalFn | None = None) -> None:
        """Signal the training process (default SIGTERM) when over budget."""
        if not self.exceeds():
            return
        if signal_fn is None:
            signal_fn = lambda sig: _send_signal(sig)  # noqa: E731
        signal_fn(signal.SIGTERM)


def _send_signal(sig: int) -> None:
    signal.raise_signal(sig)


def build_guard(
    state_path: Path,
    budget_hours: float,
    *,
    smi_power_fn: SmiPowerFn | None = None,
    wall_clock: WallClockFn | None = None,
) -> BudgetGuard:
    """Construct a BudgetGuard pre-loaded with persisted spend."""
    kwargs: dict[str, object] = {"budget_hours": budget_hours, "state": load_state(state_path)}
    if smi_power_fn is not None:
        kwargs["smi_power_fn"] = smi_power_fn
    if wall_clock is not None:
        kwargs["wall_clock"] = wall_clock
    return BudgetGuard(**kwargs)