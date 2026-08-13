"""Generate the V2 training-progress chart from the ms-swift logging.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/v2_logging.jsonl")
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artefacts/barbados_dapt_v2_loss.png")

xs_loss: list[int] = []
ys_loss: list[float] = []
xs_acc: list[int] = []
ys_acc: list[float] = []

with LOG_PATH.open() as fh:
    for line in fh:
        m = json.loads(line)
        if "loss" in m and "global_step/max_steps" in m:
            step = int(m["global_step/max_steps"].split("/")[0])
            xs_loss.append(step)
            ys_loss.append(float(m["loss"]))
            if "token_acc" in m:
                xs_acc.append(step)
                ys_acc.append(float(m["token_acc"]))

fig, ax_l = plt.subplots(figsize=(12, 6.5))
ax_l.plot(xs_loss, ys_loss, color="#c62828", linewidth=1.4, label="train loss")
ax_l.set_xlabel("optimizer step")
ax_l.set_ylabel("train loss", color="#c62828")
ax_l.tick_params(axis="y", labelcolor="#c62828")
ax_l.grid(True, linestyle=":", color="#bbbbbb", alpha=0.7)

ax_r = ax_l.twinx()
ax_r.plot(xs_acc, ys_acc, color="#1565c0", linewidth=1.0, alpha=0.85, label="token accuracy")
ax_r.set_ylabel("token accuracy", color="#1565c0")
ax_r.tick_params(axis="y", labelcolor="#1565c0")

fig.suptitle(
    "Barbados DAPT V2 — Qwen3-Omni thinker LoRA",
    fontsize=14,
    y=0.97,
)
ax_l.set_title(
    f"{len(xs_loss):,} logged steps, last loss {ys_loss[-1]:.4f}, "
    f"min {min(ys_loss):.4f} at step {xs_loss[ys_loss.index(min(ys_loss))]:,}",
    fontsize=10,
    color="#444444",
)

fig.tight_layout(rect=(0, 0, 1, 0.95))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
print(f"wrote {OUT_PATH} ({len(xs_loss)} points, steps {xs_loss[0]}-{xs_loss[-1]})")
