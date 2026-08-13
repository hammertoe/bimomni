"""Standalone text-only benchmark for Barbados local knowledge.

The benchmark compares a base model with a PEFT adapter using the mean token
log-probability of four candidate completions. Corpus-derived probes measure
knowledge acquisition; unrelated controls expose broad regressions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PROBES_PATH = Path(__file__).with_name("barbados_knowledge_benchmark.jsonl")
OUTPUT_DIR = Path("/data/knowledge-benchmark")
TRAIN_DATA_PATH = Path("/data/barbados_dapt_packed.jsonl")
EVAL_DATA_PATH = Path("/data/barbados_dapt_eval.jsonl")
TRACKS = ("local", "rare_local", "control")
VARIANTS = ("canonical", "paraphrase")


@dataclass(frozen=True, slots=True)
class Probe:
    """One four-choice completion probe with source provenance."""

    id: str
    track: str
    category: str
    prompt: str
    choices: tuple[str, ...]
    answer_index: int
    sources: tuple[str, ...] = ()
    stability: str = "stable"
    exposure: str = "control"
    fact_id: str = ""
    variant: str = "canonical"
    source_group: str = ""
    source_digests: tuple[str, ...] = ()
    membership: str = "unverified"

    @property
    def answer(self) -> str:
        """Return the correct completion text."""
        return self.choices[self.answer_index]


@dataclass(frozen=True, slots=True)
class ChoiceOutcome:
    """The deterministic outcome from one vector of completion scores."""

    predicted_index: int
    correct: bool
    correct_score: float
    margin: float


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Scores and outcome for one model on one probe."""

    probe_id: str
    track: str
    category: str
    exposure: str
    scores: tuple[float, ...]
    predicted_index: int
    answer_index: int
    correct: bool
    correct_score: float
    margin: float
    fact_id: str = ""
    variant: str = "canonical"
    source_group: str = ""
    membership: str = "unverified"

    @classmethod
    def from_scores(cls, probe: Probe, scores: Sequence[float]) -> ProbeResult:
        """Create a result by scoring a probe's candidate completions."""
        values = tuple(float(score) for score in scores)
        outcome = score_choices(values, probe.answer_index)
        return cls(
            probe_id=probe.id,
            track=probe.track,
            category=probe.category,
            exposure=probe.exposure,
            scores=values,
            predicted_index=outcome.predicted_index,
            answer_index=probe.answer_index,
            correct=outcome.correct,
            correct_score=outcome.correct_score,
            margin=outcome.margin,
            fact_id=probe.fact_id or probe.id,
            variant=probe.variant,
            source_group=probe.source_group or probe.id,
            membership=probe.membership,
        )


@dataclass(frozen=True, slots=True)
class TrackComparison:
    """Aggregate base-versus-adapter measurements for one track."""

    name: str
    count: int
    base_accuracy: float
    adapter_accuracy: float
    accuracy_delta: float
    base_mean_margin: float
    adapter_mean_margin: float
    margin_delta: float
    margin_delta_ci_low: float
    margin_delta_ci_high: float
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True, slots=True)
class FamilyComparison:
    """Strict result for independently worded probes of the same fact."""

    fact_id: str
    track: str
    variant_count: int
    base_all_correct: bool
    adapter_all_correct: bool
    base_min_margin: float
    adapter_min_margin: float
    min_margin_delta: float


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Complete paired comparison, including the control-adjusted delta."""

    overall: TrackComparison
    local: TrackComparison
    rare_local: TrackComparison
    control: TrackComparison
    categories: tuple[TrackComparison, ...]
    exposures: tuple[TrackComparison, ...]
    memberships: tuple[TrackComparison, ...]
    families: tuple[FamilyComparison, ...]
    difference_in_differences: float
    source_macro_margin_delta: float
    robust_acquisitions: int
    robust_regressions: int


def load_probes(path: Path = PROBES_PATH) -> list[Probe]:
    """Load and strictly validate benchmark probes from JSONL."""
    required = {
        "id",
        "track",
        "category",
        "prompt",
        "choices",
        "answer_index",
        "stability",
        "exposure",
        "fact_id",
        "variant",
        "source_digests",
    }
    probes: list[Probe] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: probe must be an object")
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields: {sorted(missing)}")
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError(f"{path}:{line_number}: probe needs exactly four choices")
        if not all(isinstance(choice, str) and choice.strip() for choice in choices):
            raise ValueError(f"{path}:{line_number}: choices must be non-empty strings")
        answer_index = payload["answer_index"]
        if not isinstance(answer_index, int) or not 0 <= answer_index < 4:
            raise ValueError(f"{path}:{line_number}: answer_index must be in [0, 3]")
        probe_id = payload["id"]
        if not isinstance(probe_id, str) or not probe_id:
            raise ValueError(f"{path}:{line_number}: id must be non-empty text")
        if probe_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {probe_id!r}")
        track = payload["track"]
        if track not in TRACKS:
            raise ValueError(f"{path}:{line_number}: unknown track {track!r}")
        sources = payload.get("sources", [])
        if not isinstance(sources, list) or not all(
            isinstance(item, str) for item in sources
        ):
            raise ValueError(f"{path}:{line_number}: sources must be a list of strings")
        fact_id = payload["fact_id"]
        if not isinstance(fact_id, str) or not fact_id:
            raise ValueError(f"{path}:{line_number}: fact_id must be non-empty text")
        variant = payload["variant"]
        if variant not in VARIANTS:
            raise ValueError(f"{path}:{line_number}: unknown variant {variant!r}")
        source_group = payload.get("source_group", "") or probe_id
        if not isinstance(source_group, str):
            raise ValueError(
                f"{path}:{line_number}: source_group must be non-empty text"
            )
        source_digests = payload["source_digests"]
        if not isinstance(source_digests, list) or not all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in source_digests
        ):
            raise ValueError(
                f"{path}:{line_number}: source_digests must contain SHA-256 hex"
            )
        if track != "control" and not source_digests:
            raise ValueError(
                f"{path}:{line_number}: local probes require source digests"
            )
        seen.add(probe_id)
        probes.append(
            Probe(
                id=probe_id,
                track=track,
                category=str(payload["category"]),
                prompt=str(payload["prompt"]),
                choices=tuple(choices),
                answer_index=answer_index,
                sources=tuple(sources),
                stability=str(payload["stability"]),
                exposure=str(payload["exposure"]),
                fact_id=fact_id,
                variant=variant,
                source_group=source_group,
                source_digests=tuple(source_digests),
                membership="control" if track == "control" else "unverified",
            )
        )
    if not probes:
        raise ValueError(f"{path}: no probes found")
    variants_by_fact: dict[str, set[str]] = {}
    for probe in probes:
        variants_by_fact.setdefault(probe.fact_id, set()).add(probe.variant)
    invalid_facts = {
        fact_id: variants
        for fact_id, variants in variants_by_fact.items()
        if variants != set(VARIANTS)
        or sum(probe.fact_id == fact_id for probe in probes) != len(VARIANTS)
    }
    if invalid_facts:
        raise ValueError(
            f"{path}: every fact needs canonical and paraphrase probes: {invalid_facts}"
        )
    return probes


def resolve_probe_membership(
    probes: Sequence[Probe], train_path: Path, eval_path: Path
) -> list[Probe]:
    """Classify corpus probes against the exact packed train/eval records."""
    train_digests = _packed_record_digests(train_path)
    eval_digests = _packed_record_digests(eval_path)
    resolved: list[Probe] = []
    for probe in probes:
        if probe.track == "control":
            membership = "control"
        else:
            memberships = {
                "train"
                if digest in train_digests
                else "eval"
                if digest in eval_digests
                else "absent"
                for digest in probe.source_digests
            }
            membership = memberships.pop() if len(memberships) == 1 else "mixed"
        resolved.append(replace(probe, membership=membership))
    return resolved


def _packed_record_digests(path: Path) -> set[str]:
    digests: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{line_number}: packed record needs messages")
        content = " ".join(str(message["content"]) for message in messages).strip()
        digests.add(hashlib.sha256(content.encode("utf-8")).hexdigest())
    return digests


def score_choices(scores: Sequence[float], answer_index: int) -> ChoiceOutcome:
    """Select the highest-scoring choice and calculate its correctness margin."""
    if len(scores) != 4:
        raise ValueError("expected exactly four choice scores")
    if not 0 <= answer_index < 4:
        raise ValueError("answer_index must be in [0, 3]")
    predicted = max(range(4), key=lambda index: scores[index])
    incorrect_best = max(
        score for index, score in enumerate(scores) if index != answer_index
    )
    correct_score = float(scores[answer_index])
    return ChoiceOutcome(
        predicted_index=predicted,
        correct=predicted == answer_index,
        correct_score=correct_score,
        margin=correct_score - float(incorrect_best),
    )


def compare_results(
    base: Sequence[ProbeResult], adapter: Sequence[ProbeResult]
) -> BenchmarkComparison:
    """Build a paired aggregate comparison from base and adapter results."""
    base_by_id = {result.probe_id: result for result in base}
    adapter_by_id = {result.probe_id: result for result in adapter}
    if len(base_by_id) != len(base) or len(adapter_by_id) != len(adapter):
        raise ValueError("duplicate probe results")
    if base_by_id.keys() != adapter_by_id.keys():
        raise ValueError("base and adapter results must contain identical probe ids")

    pairs = [(base_by_id[key], adapter_by_id[key]) for key in sorted(base_by_id)]
    summaries = {
        track: _summarize_track(
            track, [pair for pair in pairs if pair[0].track == track]
        )
        for track in TRACKS
    }
    overall = _summarize_track("overall", pairs)
    categories = tuple(
        _summarize_track(name, [pair for pair in pairs if pair[0].category == name])
        for name in sorted({base.category for base, _ in pairs})
    )
    exposures = tuple(
        _summarize_track(name, [pair for pair in pairs if pair[0].exposure == name])
        for name in sorted({base.exposure for base, _ in pairs})
    )
    memberships = tuple(
        _summarize_track(name, [pair for pair in pairs if pair[0].membership == name])
        for name in sorted({base.membership for base, _ in pairs})
    )
    families = _compare_families(pairs)
    acquisition_families = [family for family in families if family.track != "control"]
    local_delta = _combined_local_margin_delta(
        summaries["local"], summaries["rare_local"]
    )
    return BenchmarkComparison(
        overall=overall,
        local=summaries["local"],
        rare_local=summaries["rare_local"],
        control=summaries["control"],
        categories=categories,
        exposures=exposures,
        memberships=memberships,
        families=families,
        difference_in_differences=local_delta - summaries["control"].margin_delta,
        source_macro_margin_delta=_source_macro_margin_delta(pairs),
        robust_acquisitions=sum(
            not family.base_all_correct and family.adapter_all_correct
            for family in acquisition_families
        ),
        robust_regressions=sum(
            family.base_all_correct and not family.adapter_all_correct
            for family in acquisition_families
        ),
    )


def _compare_families(
    pairs: Sequence[tuple[ProbeResult, ProbeResult]],
) -> tuple[FamilyComparison, ...]:
    grouped: dict[str, list[tuple[ProbeResult, ProbeResult]]] = {}
    for pair in pairs:
        grouped.setdefault(pair[0].fact_id or pair[0].probe_id, []).append(pair)
    families = []
    for fact_id, family_pairs in sorted(grouped.items()):
        tracks = {base.track for base, _ in family_pairs}
        if len(tracks) != 1:
            raise ValueError(f"fact family {fact_id!r} spans multiple tracks")
        base_min = min(base.margin for base, _ in family_pairs)
        adapter_min = min(adapter.margin for _, adapter in family_pairs)
        families.append(
            FamilyComparison(
                fact_id=fact_id,
                track=tracks.pop(),
                variant_count=len(family_pairs),
                base_all_correct=all(base.correct for base, _ in family_pairs),
                adapter_all_correct=all(adapter.correct for _, adapter in family_pairs),
                base_min_margin=base_min,
                adapter_min_margin=adapter_min,
                min_margin_delta=adapter_min - base_min,
            )
        )
    return tuple(families)


def _source_macro_margin_delta(
    pairs: Sequence[tuple[ProbeResult, ProbeResult]],
) -> float:
    grouped: dict[str, list[float]] = {}
    for base, adapter in pairs:
        if base.track == "control":
            continue
        grouped.setdefault(base.source_group or base.probe_id, []).append(
            adapter.margin - base.margin
        )
    if not grouped:
        return 0.0
    return sum(sum(deltas) / len(deltas) for deltas in grouped.values()) / len(grouped)


def _summarize_track(
    name: str, pairs: Sequence[tuple[ProbeResult, ProbeResult]]
) -> TrackComparison:
    if not pairs:
        return TrackComparison(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)
    count = len(pairs)
    base_accuracy = sum(base.correct for base, _ in pairs) / count
    adapter_accuracy = sum(adapted.correct for _, adapted in pairs) / count
    base_margin = sum(base.margin for base, _ in pairs) / count
    adapter_margin = sum(adapted.margin for _, adapted in pairs) / count
    deltas = [adapted.margin - base.margin for base, adapted in pairs]
    low, high = _bootstrap_mean_interval(deltas)
    return TrackComparison(
        name=name,
        count=count,
        base_accuracy=base_accuracy,
        adapter_accuracy=adapter_accuracy,
        accuracy_delta=adapter_accuracy - base_accuracy,
        base_mean_margin=base_margin,
        adapter_mean_margin=adapter_margin,
        margin_delta=sum(deltas) / count,
        margin_delta_ci_low=low,
        margin_delta_ci_high=high,
        wins=sum(delta > 1e-9 for delta in deltas),
        ties=sum(abs(delta) <= 1e-9 for delta in deltas),
        losses=sum(delta < -1e-9 for delta in deltas),
    )


def _bootstrap_mean_interval(
    values: Sequence[float], *, samples: int = 2_000, seed: int = 42
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a paired mean."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)
    )
    return means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]


def _combined_local_margin_delta(
    local: TrackComparison, rare_local: TrackComparison
) -> float:
    count = local.count + rare_local.count
    if count == 0:
        return 0.0
    return (
        local.margin_delta * local.count + rare_local.margin_delta * rare_local.count
    ) / count


def score_model(model, tokenizer, probes: Sequence[Probe]) -> list[ProbeResult]:
    """Score all probe completions using mean causal token log-probability."""
    return [
        ProbeResult.from_scores(probe, _score_probe(model, tokenizer, probe))
        for probe in probes
    ]


def _completion_start(prefix_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    """Find the first completion-bearing token, including a boundary merge."""
    common = 0
    for prefix_token, full_token in zip(prefix_ids, full_ids):
        if prefix_token != full_token:
            break
        common += 1
    return common


def _score_probe(model, tokenizer, probe: Probe) -> tuple[float, ...]:
    import torch

    prefix = probe.prompt.rstrip() + " "
    texts = [prefix + choice for choice in probe.choices]
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    scores: list[float] = []
    for row in range(len(texts)):
        sequence_length = int(encoded["attention_mask"][row].sum())
        full_ids = encoded["input_ids"][row, :sequence_length].tolist()
        completion_start = _completion_start(prefix_ids, full_ids)
        if completion_start < 1 or sequence_length <= completion_start:
            raise RuntimeError(f"probe {probe.id!r} produced no completion tokens")
        positions = torch.arange(
            completion_start - 1, sequence_length - 1, device=device
        )
        targets = encoded["input_ids"][row, completion_start:sequence_length]
        token_scores = log_probs[row, positions, targets]
        scores.append(float(token_scores.mean().cpu()))
    return tuple(scores)


def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load the text path of Qwen Omni without enabling audio output."""
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    try:
        model.disable_talker()
    except AttributeError:
        pass
    return model, tokenizer


def _load_adapter(model, adapter_path: str):
    """Attach the adapter to the complete Omni model before using its thinker."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path).eval()


def run_benchmark(
    adapter_path: str,
    *,
    model_id: str = MODEL_ID,
    probes_path: Path = PROBES_PATH,
    output_dir: Path = OUTPUT_DIR,
    train_data_path: Path | None = TRAIN_DATA_PATH,
    eval_data_path: Path | None = EVAL_DATA_PATH,
) -> BenchmarkComparison:
    """Run the paired benchmark and write JSON and Markdown reports."""
    probes = load_probes(probes_path)
    if (
        train_data_path is not None
        and eval_data_path is not None
        and train_data_path.exists()
        and eval_data_path.exists()
    ):
        probes = resolve_probe_membership(probes, train_data_path, eval_data_path)
    model, tokenizer = load_model_and_tokenizer(model_id)
    thinker = model.thinker
    print(
        f"[knowledge-benchmark] scoring base model on {len(probes)} probes", flush=True
    )
    base = score_model(thinker, tokenizer, probes)

    adapted_model = _load_adapter(model, adapter_path)
    adapted_thinker = adapted_model.model.thinker
    print(f"[knowledge-benchmark] scoring adapter {adapter_path}", flush=True)
    adapter = score_model(adapted_thinker, tokenizer, probes)
    comparison = compare_results(base, adapter)
    _write_reports(
        output_dir, model_id, adapter_path, probes, comparison, base, adapter
    )
    return comparison


def _write_reports(
    output_dir: Path,
    model_id: str,
    adapter_path: str,
    probes: Sequence[Probe],
    comparison: BenchmarkComparison,
    base: Sequence[ProbeResult],
    adapter: Sequence[ProbeResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "comparison": asdict(comparison),
        "results": [
            {
                "probe": asdict(probe),
                "base": asdict(base_result),
                "adapter": asdict(adapter_result),
            }
            for probe, base_result, adapter_result in zip(
                probes, base, adapter, strict=True
            )
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        build_markdown_report(comparison, base, adapter, model_id=model_id),
        encoding="utf-8",
    )
    print(f"[knowledge-benchmark] wrote {output_dir / 'results.json'}", flush=True)
    print(f"[knowledge-benchmark] wrote {output_dir / 'report.md'}", flush=True)


def build_markdown_report(
    comparison: BenchmarkComparison,
    base: Sequence[ProbeResult],
    adapter: Sequence[ProbeResult],
    *,
    model_id: str,
) -> str:
    """Render a concise human-readable paired benchmark report."""
    lines = [
        "# Barbados Knowledge Delta",
        "",
        f"Model: `{model_id}`",
        "",
        "Corpus-derived probes measure knowledge acquisition, not unseen generalisation.",
        "",
        "## Summary",
        "",
        "| Track | N | Base accuracy | Adapter accuracy | Accuracy delta | Margin delta (95% CI) | W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in (
        comparison.local,
        comparison.rare_local,
        comparison.control,
        comparison.overall,
    ):
        lines.append(
            f"| {summary.name} | {summary.count} | {summary.base_accuracy:.1%} | "
            f"{summary.adapter_accuracy:.1%} | {summary.accuracy_delta:+.1%} | "
            f"{summary.margin_delta:+.3f} "
            f"[{summary.margin_delta_ci_low:+.3f}, {summary.margin_delta_ci_high:+.3f}] | "
            f"{summary.wins}/{summary.ties}/{summary.losses} |"
        )
    lines.extend(
        [
            "",
            f"Control-adjusted local margin delta: **{comparison.difference_in_differences:+.3f}**",
        ]
    )
    _append_detail_table(lines, "Category detail", comparison.categories)
    _append_detail_table(lines, "Exposure detail", comparison.exposures)
    _append_detail_table(lines, "Training membership", comparison.memberships)
    acquisition_families = [
        family for family in comparison.families if family.track != "control"
    ]
    control_families = [
        family for family in comparison.families if family.track == "control"
    ]
    lines.extend(
        [
            "",
            "## Strict fact-family results",
            "",
            "A fact passes only when both independently worded variants are correct.",
            "",
            "| Group | Facts | Base pass | Adapter pass | Acquired | Regressed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, families in (
        ("acquisition", acquisition_families),
        ("control", control_families),
    ):
        count = len(families)
        base_passes = sum(family.base_all_correct for family in families)
        adapter_passes = sum(family.adapter_all_correct for family in families)
        acquired = sum(
            not family.base_all_correct and family.adapter_all_correct
            for family in families
        )
        regressed = sum(
            family.base_all_correct and not family.adapter_all_correct
            for family in families
        )
        lines.append(
            f"| {name} | {count} | {base_passes}/{count} | {adapter_passes}/{count} | "
            f"{acquired} | {regressed} |"
        )
    lines.extend(
        [
            "",
            f"Source-macro local margin delta: **{comparison.source_macro_margin_delta:+.3f}**",
        ]
    )
    lines.extend(
        [
            "",
            "## Regressed probes",
            "",
            "| Probe | Track | Base margin | Adapter margin | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    regressed = []
    for base_result, adapter_result in zip(base, adapter, strict=True):
        delta = adapter_result.margin - base_result.margin
        if delta < 0:
            regressed.append((delta, base_result, adapter_result))
    for delta, base_result, adapter_result in sorted(regressed):
        lines.append(
            f"| {base_result.probe_id} | {base_result.track} | {base_result.margin:+.3f} | "
            f"{adapter_result.margin:+.3f} | {delta:+.3f} |"
        )
    if not regressed:
        lines.append("| None | - | - | - | - |")
    return "\n".join(lines) + "\n"


def _append_detail_table(
    lines: list[str], title: str, summaries: Sequence[TrackComparison]
) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| Group | N | Base accuracy | Adapter accuracy | Accuracy delta | Margin delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary.name} | {summary.count} | {summary.base_accuracy:.1%} | "
            f"{summary.adapter_accuracy:.1%} | {summary.accuracy_delta:+.1%} | "
            f"{summary.margin_delta:+.3f} |"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse standalone benchmark command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", required=True, help="PEFT adapter checkpoint or Hub ID"
    )
    parser.add_argument("--model", default=MODEL_ID, help="base model path or Hub ID")
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone benchmark CLI."""
    args = parse_args(argv)
    comparison = run_benchmark(
        args.adapter,
        model_id=args.model,
        probes_path=args.probes,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
    )
    print(
        f"[knowledge-benchmark] local accuracy delta "
        f"{comparison.local.accuracy_delta:+.1%}; "
        f"control-adjusted margin delta {comparison.difference_in_differences:+.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
