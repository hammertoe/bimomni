"""Score a single fused MLX Barbados model on the knowledge probes.

The `hammertoe/BimOmni-30B-A3B-MLX-4bit` artifact is a *fused* model (the LoRA
is merged into the base, talker dropped), so the paired base-vs-adapter
benchmark in `bimomni.inference.mlx` does not apply directly. This runner
scores one MLX snapshot on the shared probe set and reports absolute
accuracy per track, mirroring the PyTorch scoring path (mean token
log-probability over the completion tokens).

Scoring path mirrors the PyTorch benchmark (mean token log-probability over
the completion tokens); `model.thinker(ids)` returns a `LanguageModelOutput`
carrying `.logits` in mlx-vlm 0.6.x.

MLX is only available on Apple Silicon; the heavy imports are deferred to
runtime so this module loads on Linux CI.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from bimomni.evaluation.knowledge import (
    PROBES_PATH,
    Probe,
    ProbeResult,
    _completion_start,
    load_probes,
)

if TYPE_CHECKING:
    pass  # pragma: no cover

TRACKS = ("local", "rare_local", "control")


def _score_probe(model, tokenizer, probe: Probe) -> tuple[float, ...]:
    """Score the four completions by mean token log-probability."""
    import mlx.core as mx
    import mlx.nn as nn

    prefix = probe.prompt.rstrip() + " "
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    scores: list[float] = []
    for choice in probe.choices:
        full_ids = tokenizer(prefix + choice, add_special_tokens=True)["input_ids"]
        ids = mx.array(full_ids)[None, :]  # (1, T)
        out = model.thinker(ids)  # LanguageModelOutput(logits=(1, T, V))
        logits = out.logits
        log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)[0]  # (T, V)
        start = _completion_start(prefix_ids, full_ids)
        if start < 1 or len(full_ids) <= start:
            raise RuntimeError(f"probe {probe.id!r}: empty completion")
        positions = mx.arange(start - 1, len(full_ids) - 1)
        targets = mx.array(full_ids[start:])
        token_logprobs = log_probs[positions, targets]
        scores.append(float(mx.mean(token_logprobs)))
    return tuple(scores)


def score_model(model, tokenizer, probes: list[Probe]) -> list[ProbeResult]:
    return [
        ProbeResult.from_scores(probe, _score_probe(model, tokenizer, probe))
        for probe in probes
    ]


def _track_accuracy(results: list[ProbeResult], track: str) -> dict:
    subset = [r for r in results if r.track == track]
    if not subset:
        return {"track": track, "count": 0, "correct": 0, "accuracy": None}
    correct = sum(r.correct for r in subset)
    return {
        "track": track,
        "count": len(subset),
        "correct": correct,
        "accuracy": correct / len(subset),
    }


def _write_reports(
    output_dir: Path,
    model_id: str,
    probes: list[Probe],
    results: list[ProbeResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = [_track_accuracy(results, t) for t in TRACKS]
    overall = {
        "track": "overall",
        "count": len(results),
        "correct": sum(r.correct for r in results),
        "accuracy": sum(r.correct for r in results) / len(results),
    }
    by_category: Counter[str] = Counter()
    for r in results:
        by_category[f"{r.track}::{r.category}"] += 1
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "model_repo": "hammertoe/BimOmni-30B-A3B-MLX-4bit",
        "summary": {"tracks": [*tracks, overall]},
        "results": [
            {
                "probe": asdict(probe),
                "predicted_index": result.predicted_index,
                "answer_index": result.answer_index,
                "correct": result.correct,
                "correct_score": result.correct_score,
                "margin": result.margin,
                "scores": list(result.scores),
            }
            for probe, result in zip(probes, results, strict=True)
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Barbados Knowledge — Fused 4-bit MLX",
        "",
        f"Model: `{model_id}`",
        "",
        "Single-snapshot accuracy on the shared 60-probe set (no base/adapter",
        "delta; the adapter is already merged). Mean token log-probability scoring.",
        "",
        "| Track | N | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in [*tracks, overall]:
        acc = "—" if row["accuracy"] is None else f"{row['accuracy']:.1%}"
        lines.append(f"| {row['track']} | {row['count']} | {row['correct']} | {acc} |")
    lines.extend(["", "## Wrong probes", ""])
    for r in results:
        if not r.correct:
            lines.append(f"- `{r.probe_id}` ({r.track})")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fused_benchmark(
    model_path: str,
    *,
    probes_path: Path = PROBES_PATH,
    output_dir: Path = Path("knowledge-benchmark-mlx-fused"),
) -> dict:
    probes = load_probes(probes_path)
    print(f"[mlx-fused] loading {model_path}", flush=True)
    from mlx_vlm import load as load_mlx_vlm
    model, processor = load_mlx_vlm(model_path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    print(f"[mlx-fused] scoring {len(probes)} probes", flush=True)
    results = score_model(model, tokenizer, probes)
    _write_reports(output_dir, model_path, probes, results)
    summary = {
        t["track"]: t["accuracy"] for t in [_track_accuracy(results, t) for t in TRACKS]
    }
    summary["overall"] = sum(r.correct for r in results) / len(results)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fused MLX Barbados knowledge benchmark")
    parser.add_argument("--model", default="model/qwen3-omni-4bit-v4")
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("knowledge-benchmark-mlx-fused"))
    args = parser.parse_args()
    summary = run_fused_benchmark(args.model, probes_path=args.probes, output_dir=args.output_dir)
    print(f"[mlx-fused] overall accuracy {summary.get('overall', 0):.1%}", flush=True)
