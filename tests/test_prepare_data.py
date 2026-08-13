"""Tests for bimomni.training.prepare_data: corpus validation, dedup, packing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bimomni.training.prepare_data import (
    EVAL_FRACTION,
    MIN_TOKENS,
    MAX_TOKENS,
    RecordError,
    prepare_corpus,
    validate_record,
)


def _encode(text: str) -> list[int]:
    return list(range(len(text)))


def _long(text: str, times: int = 30) -> str:
    return (text + " ") * times


def _write_shard(dirpath: Path, stem: str, records: list[dict]) -> Path:
    path = dirpath / f"{stem}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _record(content: str) -> dict:
    return {"messages": [{"role": "assistant", "content": content}]}


def test_validate_record_accepts_assistant_only(tmp_path: Path) -> None:
    text = "Crop Over is Barbados' biggest cultural festival."
    record = validate_record(_record(text), _encode)
    assert record.content == text
    assert record.tokens == len(text)
    assert len(record.digest) == 64


def test_minimum_length_retains_short_coherent_articles() -> None:
    assert MIN_TOKENS == 128


def test_validate_record_rejects_missing_messages() -> None:
    with pytest.raises(RecordError, match="messages"):
        validate_record({}, _encode)


def test_validate_record_rejects_empty_messages() -> None:
    with pytest.raises(RecordError, match="messages"):
        validate_record({"messages": []}, _encode)


def test_validate_record_rejects_non_assistant_role() -> None:
    with pytest.raises(RecordError, match="assistant"):
        validate_record(
            {"messages": [{"role": "user", "content": "hi"}]}, _encode
        )


def test_validate_record_rejects_non_string_content() -> None:
    with pytest.raises(RecordError, match="content"):
        validate_record({"messages": [{"role": "assistant", "content": 42}]}, _encode)


def test_validate_record_rejects_blank_content() -> None:
    with pytest.raises(RecordError, match="content"):
        validate_record({"messages": [{"role": "assistant", "content": "   "}]}, _encode)


def test_prepare_dedups_by_content_across_shards(tmp_path: Path) -> None:
    dup = _long("The hurricane season runs from June to November.")
    _write_shard(tmp_path, "a", [_record(dup), _record(_long("First unique article."))])
    _write_shard(tmp_path, "b", [_record(dup), _record(_long("Second unique article."))])

    packed = tmp_path / "packed.jsonl"
    eval_out = tmp_path / "eval.jsonl"
    summary = prepare_corpus(tmp_path, packed, eval_out, _encode)

    assert summary.total_records == 4
    assert summary.valid_records == 4
    assert summary.deduplicated == 1
    assert summary.below_min_tokens == 0
    assert summary.train_records + summary.eval_records == 3
    assert len(packed.read_text().splitlines()) == summary.train_records
    if summary.eval_records:
        assert len(eval_out.read_text().splitlines()) == summary.eval_records


def test_prepare_drops_records_outside_token_window(tmp_path: Path) -> None:
    short = _record("tiny")
    long_text = "x" * (MAX_TOKENS + 50)
    _write_shard(tmp_path, "a", [short, _record(long_text), _record(_long("just right"))])

    packed = tmp_path / "packed.jsonl"
    eval_out = tmp_path / "eval.jsonl"
    summary = prepare_corpus(tmp_path, packed, eval_out, _encode)

    assert summary.below_min_tokens == 1
    assert summary.above_max_tokens == 1
    assert summary.train_records + summary.eval_records == 1
    for line in packed.read_text().splitlines():
        assert "just right" in line


def test_prepare_eval_split_fraction_is_2_percent(tmp_path: Path) -> None:
    text = "A distinct enough sentence to stay unique. " * 30
    records = [_record(f"{text} {i}") for i in range(200)]
    _write_shard(tmp_path, "a", records)

    packed = tmp_path / "packed.jsonl"
    eval_out = tmp_path / "eval.jsonl"
    summary = prepare_corpus(tmp_path, packed, eval_out, _encode)

    expected = round(200 * EVAL_FRACTION)
    assert summary.eval_records == expected
    assert summary.train_records == 200 - expected


def test_prepare_eval_split_is_deterministic(tmp_path: Path) -> None:
    text = "A distinct enough sentence to stay unique. " * 30
    records = [_record(f"{text} {i}") for i in range(100)]
    _write_shard(tmp_path, "a", records)

    summary_a = prepare_corpus(tmp_path, tmp_path / "p1.jsonl", tmp_path / "e1.jsonl", _encode)
    summary_b = prepare_corpus(tmp_path, tmp_path / "p2.jsonl", tmp_path / "e2.jsonl", _encode)

    assert tmp_path.joinpath("e1.jsonl").read_text() == tmp_path.joinpath("e2.jsonl").read_text()
    assert summary_a.eval_records == summary_b.eval_records


def test_prepare_sorts_train_records_by_token_length(tmp_path: Path) -> None:
    _write_shard(
        tmp_path,
        "a",
        [_record("long" * 500), _record("short"), _record("medium" * 100)],
    )

    packed = tmp_path / "packed.jsonl"
    eval_out = tmp_path / "eval.jsonl"
    summary = prepare_corpus(tmp_path, packed, eval_out, _encode)

    lengths = [len(json.loads(line)["messages"][0]["content"]) for line in packed.read_text().splitlines()]
    assert lengths == sorted(lengths)
    assert summary.train_tokens == sum(lengths)


def test_prepare_skips_blank_jsonl_and_empty_corpus(tmp_path: Path) -> None:
    _write_shard(tmp_path, "empty", [])

    packed = tmp_path / "packed.jsonl"
    eval_out = tmp_path / "eval.jsonl"
    summary = prepare_corpus(tmp_path, packed, eval_out, _encode)

    assert summary.total_records == 0
    assert summary.train_records == 0
    assert not packed.exists()
    assert not eval_out.exists()
