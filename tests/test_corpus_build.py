"""Tests for the article-aware newspaper Corpus V2 builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bimomni.corpus.build import build_corpus, deduplicate_records


def _article(headline: str, body: str) -> str:
    return f"{headline}\n\n{body}\n"


def test_deduplicate_records_removes_exact_and_near_duplicates() -> None:
    common = (
        "Barbados tourism officials announced a community programme involving "
        "hotels, schools, farmers and cultural organisations across the island. "
    )
    records = [
        ("a", 0, common * 4),
        ("b", 0, (common * 4).replace("island.", "island!")),
        ("c", 0, common * 4),
        ("d", 0, "A completely different Barbadian cricket report. " * 12),
    ]

    kept, exact, near = deduplicate_records(records)

    assert exact == 1
    assert near == 1
    assert [(record.source, record.ordinal) for record in kept] == [("a", 0), ("d", 0)]


def test_build_corpus_reads_limited_text_files_into_separate_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "v2"
    input_dir.mkdir()
    body = (
        "The Barbados programme will support communities across the island while "
        "creating opportunities for residents and local businesses. "
    ) * 8
    for stem in ("2023-01-01", "2023-01-02", "2023-01-03"):
        input_dir.joinpath(f"{stem}.txt").write_text(
            _article(f"Local programme announced {stem}", body), encoding="utf-8"
        )

    summary = build_corpus(input_dir, output_dir, limit=2)

    assert summary.scanned_files == 2
    assert summary.output_records == 2
    outputs = sorted(output_dir.glob("*.jsonl"))
    assert [path.stem for path in outputs] == ["2023-01-01", "2023-01-02"]
    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert payload["messages"][0]["role"] == "assistant"
    assert set(payload) == {"messages"}


def test_build_corpus_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "v2"
    input_dir.mkdir()
    output_dir.mkdir()
    output_dir.joinpath("old.jsonl").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        build_corpus(input_dir, output_dir)


def test_build_corpus_requires_an_existing_input_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="input directory"):
        build_corpus(tmp_path / "missing", tmp_path / "v2")
