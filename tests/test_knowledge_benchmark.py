"""Tests for the standalone Barbados local-knowledge benchmark."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from bimomni.evaluation.evaluate import _attach_adapter
from bimomni.evaluation.knowledge import (
    Probe,
    ProbeResult,
    _completion_start,
    _load_adapter,
    build_markdown_report,
    compare_results,
    load_probes,
    resolve_probe_membership,
    score_choices,
)

PROBES_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "knowledge"
    / "probes"
    / "barbados-knowledge-v1.jsonl"
)


def test_curated_benchmark_has_paired_facts_and_independent_sources() -> None:
    probes = load_probes(PROBES_PATH)
    repo_root = Path(__file__).resolve().parents[1]

    assert len(probes) == 60
    assert sum(probe.track == "local" for probe in probes) == 20
    assert sum(probe.track == "rare_local" for probe in probes) == 20
    assert sum(probe.track == "control" for probe in probes) == 20
    assert [
        sum(probe.answer_index == index for probe in probes) for index in range(4)
    ] == [
        15,
        15,
        15,
        15,
    ]
    fact_ids = {probe.fact_id for probe in probes}
    assert len(fact_ids) == 30
    assert all(
        sum(probe.fact_id == fact_id for probe in probes) == 2 for fact_id in fact_ids
    )
    assert all(
        {probe.variant for probe in probes if probe.fact_id == fact_id}
        == {"canonical", "paraphrase"}
        for fact_id in fact_ids
    )
    acquisition = [probe for probe in probes if probe.track != "control"]
    # Public probes may have source_digests but not source paths, so the
    # validation step only runs when the corpus is available locally.
    for probe in acquisition:
        assert probe.source_digests
        for digest in probe.source_digests:
            assert len(digest) == 64
    local_corpus = repo_root / "model" / "pdfs"
    if local_corpus.exists():
        assert len({probe.source_group for probe in acquisition}) == 20
        for probe in probes:
            for source in probe.sources:
                source_path, _, line = source.rpartition(":")
                assert source_path.endswith(".jsonl")
                assert line.isdigit()
                assert (repo_root / source_path).exists()
                payload = json.loads(
                    (repo_root / source_path)
                    .read_text(encoding="utf-8")
                    .splitlines()[int(line) - 1]
                )
                content = " ".join(
                    message["content"] for message in payload["messages"]
                ).strip()
                assert (
                    hashlib.sha256(content.encode("utf-8")).hexdigest()
                    in probe.source_digests
                )


def test_load_probes_parses_strict_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "probes.jsonl"
    payload = {
        "id": "history-001",
        "track": "local",
        "category": "history",
        "prompt": "Barbados gained independence in",
        "choices": ["1965", "1966", "1967", "1968"],
        "answer_index": 1,
        "sources": ["model/pdfs/example.jsonl:1"],
        "stability": "stable",
        "exposure": "frequent",
        "fact_id": "independence-date",
        "variant": "canonical",
        "source_group": "example:1",
        "source_digests": ["a" * 64],
    }
    paraphrase = {
        **payload,
        "id": "history-001-paraphrase",
        "prompt": "In what year did Barbados become independent?",
        "variant": "paraphrase",
    }
    path.write_text(
        json.dumps(payload) + "\n" + json.dumps(paraphrase) + "\n",
        encoding="utf-8",
    )

    probes = load_probes(path)

    assert len(probes) == 2
    assert probes[0] == Probe(
        id="history-001",
        track="local",
        category="history",
        prompt="Barbados gained independence in",
        choices=("1965", "1966", "1967", "1968"),
        answer_index=1,
        sources=("model/pdfs/example.jsonl:1",),
        stability="stable",
        exposure="frequent",
        fact_id="independence-date",
        variant="canonical",
        source_group="example:1",
        source_digests=("a" * 64,),
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "missing fields"),
        (
            {
                "id": "bad",
                "track": "local",
                "category": "history",
                "prompt": "Question",
                "choices": ["one", "two"],
                "answer_index": 0,
                "sources": ["source"],
                "stability": "stable",
                "exposure": "rare",
                "fact_id": "bad",
                "variant": "canonical",
                "source_group": "bad",
                "source_digests": ["a" * 64],
            },
            "exactly four choices",
        ),
    ],
)
def test_load_probes_rejects_invalid_records(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_probes(path)


def test_score_choices_calculates_top_choice_and_margin() -> None:
    outcome = score_choices((-2.5, -0.4, -1.2, -3.0), answer_index=1)

    assert outcome.predicted_index == 1
    assert outcome.correct is True
    assert outcome.correct_score == pytest.approx(-0.4)
    assert outcome.margin == pytest.approx(0.8)


def test_completion_start_includes_a_token_merged_at_prompt_boundary() -> None:
    prefix_ids = [1, 10, 20, 30]
    full_ids = [1, 10, 20, 99, 40]

    assert _completion_start(prefix_ids, full_ids) == 3


def test_load_adapter_attaches_to_the_complete_omni_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, str]] = []

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model: object, adapter_path: str) -> object:
            calls.append((model, adapter_path))
            return model

    monkeypatch.setitem(sys.modules, "peft", type("Peft", (), {"PeftModel": FakePeftModel}))

    class Model:
        def eval(self) -> Model:
            return self

    model = Model()

    assert _load_adapter(model, "/adapter") is model
    assert calls == [(model, "/adapter")]


def test_evaluation_attaches_adapter_before_selecting_thinker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, str]] = []
    thinker = object()

    class AdaptedModel:
        model = type("Omni", (), {"thinker": thinker})()

        def eval(self) -> AdaptedModel:
            return self

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model: object, adapter_path: str) -> AdaptedModel:
            calls.append((model, adapter_path))
            return AdaptedModel()

    monkeypatch.setitem(sys.modules, "peft", type("Peft", (), {"PeftModel": FakePeftModel}))
    model = object()

    assert _attach_adapter(model, "/adapter") is thinker
    assert calls == [(model, "/adapter")]


def test_compare_results_reports_local_delta_and_control_guardrail() -> None:
    probes = [
        Probe(
            "local-1a",
            "local",
            "history",
            "Q",
            ("a", "b", "c", "d"),
            0,
            ("s",),
            exposure="rare",
            fact_id="local-1",
            variant="canonical",
            source_group="source-1",
        ),
        Probe(
            "local-1b",
            "local",
            "history",
            "Q2",
            ("b", "a", "c", "d"),
            1,
            ("s",),
            exposure="rare",
            fact_id="local-1",
            variant="paraphrase",
            source_group="source-1",
        ),
        Probe(
            "control-1a",
            "control",
            "general",
            "Q",
            ("a", "b", "c", "d"),
            0,
            fact_id="control-1",
            variant="canonical",
            source_group="control-1",
        ),
        Probe(
            "control-1b",
            "control",
            "general",
            "Q2",
            ("b", "a", "c", "d"),
            1,
            fact_id="control-1",
            variant="paraphrase",
            source_group="control-1",
        ),
    ]
    base = [
        ProbeResult.from_scores(probes[0], (-2.0, -1.0, -1.5, -3.0)),
        ProbeResult.from_scores(probes[1], (-1.0, -2.0, -1.5, -3.0)),
        ProbeResult.from_scores(probes[2], (-0.5, -2.0, -2.5, -3.0)),
        ProbeResult.from_scores(probes[3], (-2.0, -0.5, -2.5, -3.0)),
    ]
    adapter = [
        ProbeResult.from_scores(probes[0], (-0.3, -1.1, -1.5, -3.0)),
        ProbeResult.from_scores(probes[1], (-1.1, -0.3, -1.5, -3.0)),
        ProbeResult.from_scores(probes[2], (-0.6, -2.0, -2.5, -3.0)),
        ProbeResult.from_scores(probes[3], (-2.0, -0.6, -2.5, -3.0)),
    ]

    comparison = compare_results(base, adapter)

    assert comparison.local.base_accuracy == 0.0
    assert comparison.local.adapter_accuracy == 1.0
    assert comparison.local.accuracy_delta == 1.0
    assert comparison.control.accuracy_delta == 0.0
    assert comparison.difference_in_differences > 0
    assert [summary.name for summary in comparison.categories] == ["general", "history"]
    assert [summary.name for summary in comparison.exposures] == ["control", "rare"]
    assert comparison.robust_acquisitions == 1
    assert comparison.robust_regressions == 0
    assert comparison.source_macro_margin_delta > 0
    assert comparison.families[0].fact_id == "control-1"
    assert comparison.families[1].fact_id == "local-1"
    assert comparison.families[1].adapter_all_correct is True


def test_resolve_probe_membership_uses_packed_record_digests(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    train.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "trained fact"}]})
        + "\n"
    )
    evaluation.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "held out fact"}]})
        + "\n"
    )
    train_digest = hashlib.sha256(b"trained fact").hexdigest()
    eval_digest = hashlib.sha256(b"held out fact").hexdigest()
    probes = [
        Probe(
            "train",
            "local",
            "history",
            "Q",
            ("a", "b", "c", "d"),
            0,
            ("s",),
            fact_id="train",
            source_group="train",
            source_digests=(train_digest,),
        ),
        Probe(
            "eval",
            "rare_local",
            "history",
            "Q",
            ("a", "b", "c", "d"),
            0,
            ("s",),
            fact_id="eval",
            source_group="eval",
            source_digests=(eval_digest,),
        ),
        Probe(
            "absent",
            "local",
            "history",
            "Q",
            ("a", "b", "c", "d"),
            0,
            ("s",),
            fact_id="absent",
            source_group="absent",
            source_digests=("f" * 64,),
        ),
        Probe(
            "control",
            "control",
            "general",
            "Q",
            ("a", "b", "c", "d"),
            0,
            fact_id="control",
            source_group="control",
        ),
    ]

    resolved = resolve_probe_membership(probes, train, evaluation)

    assert [probe.membership for probe in resolved] == [
        "train",
        "eval",
        "absent",
        "control",
    ]


def test_build_markdown_report_contains_summary_and_regressions() -> None:
    probe = Probe("history-1", "local", "history", "Q", ("a", "b", "c", "d"), 0, ("s",))
    base = [ProbeResult.from_scores(probe, (-0.5, -1.0, -2.0, -3.0))]
    adapter = [ProbeResult.from_scores(probe, (-1.5, -0.2, -2.0, -3.0))]
    comparison = compare_results(base, adapter)

    report = build_markdown_report(comparison, base, adapter, model_id="test-model")

    assert "# Barbados Knowledge Delta" in report
    assert "test-model" in report
    assert "Category detail" in report
    assert "Exposure detail" in report
    assert "Strict fact-family results" in report
    assert "Source-macro local margin delta" in report
    assert "Regressed probes" in report
    assert "history-1" in report
