"""MLX-backed Barbados knowledge benchmark.

Reuses the pure-Python dataclasses from `bimomni.evaluation.knowledge` so the
probe set, track taxonomy, and reporting stay identical to the GPU run.
Replaces only the model scoring path with mlx_lm + mlx.core.

MLX is only available on Apple Silicon, so the heavy imports are deferred to
runtime: importing this module on Linux CI does not require mlx_vlm.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from bimomni.evaluation.knowledge import (
    EVAL_DATA_PATH,
    PROBES_PATH,
    TRAIN_DATA_PATH,
    BenchmarkComparison,
    Probe,
    ProbeResult,
    _completion_start,
    build_markdown_report,
    compare_results,
    load_probes,
    resolve_probe_membership,
)

if TYPE_CHECKING:
    pass  # pragma: no cover


def _score_probe_mlx(
    model, tokenizer, probe: Probe
) -> tuple[float, ...]:
    """Score probe completions with MLX logprobs.

    For each of the 4 choices, tokenize prefix+choice, run a forward pass, and
    return the mean token log-probability over the completion tokens.
    """
    import mlx.core as mx
    import mlx.nn as nn

    prefix = probe.prompt.rstrip() + " "
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    scores: list[float] = []
    for choice in probe.choices:
        full_ids = tokenizer(prefix + choice, add_special_tokens=True)["input_ids"]
        # Convert to mlx arrays.
        ids = mx.array(full_ids)[None, :]  # (1, T)
        logits = model.thinker(ids).logits  # (1, T, V)
        log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)[0]  # (T, V)
        completion_start = _completion_start(prefix_ids, full_ids)
        if completion_start < 1 or len(full_ids) <= completion_start:
            raise RuntimeError(f"probe {probe.id!r}: empty completion")
        positions = mx.arange(completion_start - 1, len(full_ids) - 1)
        targets = mx.array(full_ids[completion_start:])
        token_logprobs = log_probs[positions, targets]
        scores.append(float(mx.mean(token_logprobs)))
    return tuple(scores)


def score_model_mlx(model, tokenizer, probes: list[Probe]) -> list[ProbeResult]:
    """Score every probe with the given MLX model."""
    return [ProbeResult.from_scores(p, _score_probe_mlx(model, tokenizer, p)) for p in probes]


def load_model_and_tokenizer_mlx(model_path: str):
    """Load the Qwen3-Omni Thinker with MLX-VLM and return its tokenizer."""
    from mlx_vlm import load as load_mlx_vlm

    model, processor = load_mlx_vlm(model_path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return model, tokenizer


def _write_reports_mlx(
    output_dir: Path,
    model_id: str,
    adapter_path: str,
    probes: list[Probe],
    comparison: BenchmarkComparison,
    base: list[ProbeResult],
    adapter: list[ProbeResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "adapter_path": adapter_path,
        "comparison": asdict(comparison),
        "results": [
            {
                "probe": asdict(probe),
                "base": asdict(base_result),
                "adapter": asdict(adapter_result),
            }
            for probe, base_result, adapter_result in zip(probes, base, adapter, strict=True)
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        build_markdown_report(comparison, base, adapter, model_id=model_id),
        encoding="utf-8",
    )


def run_benchmark_mlx(
    base_model_path: str,
    adapter_model_path: str,
    *,
    probes_path: Path = PROBES_PATH,
    output_dir: Path = Path("/data/knowledge-benchmark"),
    train_data_path: Path | None = TRAIN_DATA_PATH,
    eval_data_path: Path | None = EVAL_DATA_PATH,
) -> BenchmarkComparison:
    """Score base vs adapter on the local probes and write reports."""
    probes = load_probes(probes_path)
    if (
        train_data_path is not None
        and eval_data_path is not None
        and train_data_path.exists()
        and eval_data_path.exists()
    ):
        probes = resolve_probe_membership(probes, train_data_path, eval_data_path)

    print(f"[mlx-bench] scoring base from {base_model_path}", flush=True)
    base_model, base_tok = load_model_and_tokenizer_mlx(base_model_path)
    base = score_model_mlx(base_model, base_tok, probes)
    del base_model

    print(f"[mlx-bench] scoring adapter from {adapter_model_path}", flush=True)
    adapter_model, adapter_tok = load_model_and_tokenizer_mlx(adapter_model_path)
    adapter = score_model_mlx(adapter_model, adapter_tok, probes)
    del adapter_model

    comparison = compare_results(base, adapter)
    _write_reports_mlx(
        output_dir, base_model_path, adapter_model_path, probes, comparison, base, adapter
    )
    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLX Barbados knowledge benchmark")
    parser.add_argument("--base-model", required=True, help="MLX-format base directory")
    parser.add_argument("--adapter-model", required=True, help="MLX-format adapter directory")
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("./knowledge-benchmark"))
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA_PATH)
    args = parser.parse_args()

    comparison = run_benchmark_mlx(
        base_model_path=args.base_model,
        adapter_model_path=args.adapter_model,
        probes_path=args.probes,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
    )
    print(
        f"[mlx-bench] local accuracy delta {comparison.local.accuracy_delta:+.1%}; "
        f"control-adjusted margin delta {comparison.difference_in_differences:+.3f}",
        flush=True,
    )
