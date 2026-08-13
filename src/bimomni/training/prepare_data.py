"""Corpus preparation for Qwen3-Omni DAPT.

Reads the assistant-only JSONL shards under /data/corpus, validates each
record, dedups by SHA-256 of content, drops records outside the token
window, length-sorts, and packs into ms-swift DAPT JSONL with a held-out
eval split for perplexity checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MIN_TOKENS = 128
MAX_TOKENS = 4096
EVAL_FRACTION = 0.02

EncodeFn = Callable[[str], list[int]]


class RecordError(ValueError):
    """Raised when a JSONL line is not a valid assistant-only record."""


@dataclass(frozen=True)
class ValidatedRecord:
    content: str
    tokens: int
    digest: str


@dataclass(frozen=True)
class PrepareSummary:
    total_records: int
    valid_records: int
    deduplicated: int
    below_min_tokens: int
    above_max_tokens: int
    train_records: int
    eval_records: int
    train_tokens: int
    eval_tokens: int


def validate_record(obj: object, encode: EncodeFn) -> ValidatedRecord:
    """Validate one parsed JSONL object as an assistant-only record.

    Raises RecordError with a message describing the first problem found.
    """
    if not isinstance(obj, dict):
        raise RecordError("record is not a JSON object")
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RecordError("record needs a non-empty 'messages' list")
    for message in messages:
        if not isinstance(message, dict):
            raise RecordError("message is not an object")
        if message.get("role") != "assistant":
            raise RecordError(f"expected role 'assistant', got {message.get('role')!r}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RecordError("assistant content must be non-empty text")
    content = " ".join(
        str(message["content"]) for message in messages if isinstance(message.get("content"), str)
    ).strip()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    tokens = len(encode(content))
    return ValidatedRecord(content=content, tokens=tokens, digest=digest)


def _eval_choice(digest: str) -> int:
    """Deterministic pseudo-random rank in [0, 2^32) from the record digest."""
    return int(digest[:8], 16)


def prepare_corpus(
    corpus_dir: Path,
    train_out: Path,
    eval_out: Path,
    encode: EncodeFn,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    eval_fraction: float = EVAL_FRACTION,
) -> PrepareSummary:
    """Pack validated shards into train/eval ms-swift DAPT JSONL files.

    Records are deduplicated by SHA-256 of content (first occurrence wins),
    dropped when outside [min_tokens, max_tokens], length-sorted ascending,
    then deterministically split into eval_out (eval_fraction) and train_out.
    """
    seen: set[str] = set()
    keep: list[ValidatedRecord] = []
    summary = PrepareSummary(
        total_records=0,
        valid_records=0,
        deduplicated=0,
        below_min_tokens=0,
        above_max_tokens=0,
        train_records=0,
        eval_records=0,
        train_tokens=0,
        eval_tokens=0,
    )

    for shard in sorted(corpus_dir.glob("*.jsonl")):
        for raw in shard.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            summary = _replace_count(summary, total_records=summary.total_records + 1)
            obj = json.loads(line)
            record = validate_record(obj, encode)
            summary = _replace_count(summary, valid_records=summary.valid_records + 1)
            if record.digest in seen:
                summary = _replace_count(summary, deduplicated=summary.deduplicated + 1)
                continue
            seen.add(record.digest)
            if record.tokens < min_tokens:
                summary = _replace_count(summary, below_min_tokens=summary.below_min_tokens + 1)
                continue
            if record.tokens > max_tokens:
                summary = _replace_count(summary, above_max_tokens=summary.above_max_tokens + 1)
                continue
            keep.append(record)

    keep.sort(key=lambda r: r.tokens)

    eval_count = round(len(keep) * eval_fraction)
    eval_digests = {
        record.digest for record in sorted(keep, key=lambda r: _eval_choice(r.digest))[:eval_count]
    }
    eval_records = [r for r in keep if r.digest in eval_digests]
    train_records = [r for r in keep if r.digest not in eval_digests]

    if train_records:
        _write_records(train_out, train_records)
    if eval_records:
        _write_records(eval_out, eval_records)

    summary = _replace_count(summary, train_records=len(train_records))
    summary = _replace_count(summary, eval_records=len(eval_records))
    summary = _replace_count(summary, train_tokens=sum(r.tokens for r in train_records))
    summary = _replace_count(summary, eval_tokens=sum(r.tokens for r in eval_records))
    return summary


def _write_records(path: Path, records: list[ValidatedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {"messages": [{"role": "assistant", "content": record.content}]}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _replace_count(summary: PrepareSummary, **kwargs: int) -> PrepareSummary:
    return PrepareSummary(
        total_records=kwargs.get("total_records", summary.total_records),
        valid_records=kwargs.get("valid_records", summary.valid_records),
        deduplicated=kwargs.get("deduplicated", summary.deduplicated),
        below_min_tokens=kwargs.get("below_min_tokens", summary.below_min_tokens),
        above_max_tokens=kwargs.get("above_max_tokens", summary.above_max_tokens),
        train_records=kwargs.get("train_records", summary.train_records),
        eval_records=kwargs.get("eval_records", summary.eval_records),
        train_tokens=kwargs.get("train_tokens", summary.train_tokens),
        eval_tokens=kwargs.get("eval_tokens", summary.eval_tokens),
    )
