#!/usr/bin/env python3
"""Build an article-aware Corpus V2 from existing newspaper text files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from bimomni.corpus.extract import WORD_RE, chunk_blocks, extract_blocks

LOCAL_TERMS = re.compile(
    r"\b(?:Barbados|Barbadian|Barbadians|Bajan|Bridgetown|CARICOM|Caribbean|"
    r"Christ Church|Saint Michael|St\. Michael|Saint Philip|St\. Philip|"
    r"Saint Peter|St\. Peter|Saint James|St\. James|Saint Lucy|St\. Lucy|"
    r"Saint George|St\. George|Saint Thomas|St\. Thomas|Saint Andrew|St\. Andrew|"
    r"Saint Joseph|St\. Joseph|Saint John|St\. John|UWI|Cave Hill)\b",
    re.IGNORECASE,
)
CLEAR_FOREIGN_TERMS = re.compile(
    r"\b(?:Israel|Israeli|Tunisia|Tunisian|Ukraine|Ukrainian|United States|"
    r"Washington|New York|London|European Union|Virgin Galactic)\b|"
    r"\((?:AP|Reuters)\)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """A transient source association used only while building the corpus."""

    source: str
    ordinal: int
    content: str


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """Aggregate counts from one bounded V2 corpus build."""

    scanned_files: int
    extracted_records: int
    below_minimum_records: int
    off_domain_records: int
    exact_duplicates: int
    near_duplicates: int
    output_records: int


def _canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _simhash(text: str) -> int:
    tokens = [token.casefold() for token in WORD_RE.findall(text)]
    shingles = {
        " ".join(tokens[index : index + 5])
        for index in range(max(1, len(tokens) - 4))
    }
    if not shingles:
        return 0
    totals = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            totals[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, total in enumerate(totals) if total >= 0)


def _coerce_record(record: CorpusRecord | tuple[str, int, str]) -> CorpusRecord:
    return record if isinstance(record, CorpusRecord) else CorpusRecord(*record)


def deduplicate_records(
    records: Iterable[CorpusRecord | tuple[str, int, str]],
    *,
    max_hamming_distance: int = 3,
) -> tuple[list[CorpusRecord], int, int]:
    """Remove exact and conservative near duplicates, preferring longer text."""
    unique: dict[str, CorpusRecord] = {}
    exact_duplicates = 0
    for raw_record in records:
        record = _coerce_record(raw_record)
        key = _canonical(record.content)
        previous = unique.get(key)
        if previous is None:
            unique[key] = record
        else:
            exact_duplicates += 1
            if len(record.content) > len(previous.content):
                unique[key] = record

    ordered = sorted(
        unique.values(),
        key=lambda record: (-len(record.content), record.source, record.ordinal),
    )
    buckets: dict[tuple[int, int], list[tuple[int, CorpusRecord]]] = defaultdict(list)
    kept: list[CorpusRecord] = []
    near_duplicates = 0
    for record in ordered:
        fingerprint = _simhash(record.content)
        candidates: dict[tuple[str, int], tuple[int, CorpusRecord]] = {}
        for band in range(4):
            value = (fingerprint >> (band * 16)) & 0xFFFF
            for candidate_fingerprint, candidate in buckets[(band, value)]:
                candidates[(candidate.source, candidate.ordinal)] = (
                    candidate_fingerprint,
                    candidate,
                )
        if any(
            (fingerprint ^ candidate_fingerprint).bit_count() <= max_hamming_distance
            for candidate_fingerprint, _ in candidates.values()
        ):
            near_duplicates += 1
            continue
        kept.append(record)
        for band in range(4):
            value = (fingerprint >> (band * 16)) & 0xFFFF
            buckets[(band, value)].append((fingerprint, record))

    kept.sort(key=lambda record: (record.source, record.ordinal))
    return kept, exact_duplicates, near_duplicates


def _is_clearly_off_domain(content: str) -> bool:
    return not LOCAL_TERMS.search(content) and bool(CLEAR_FOREIGN_TERMS.search(content))


def _write_shards(records: list[CorpusRecord], output_dir: Path) -> None:
    grouped: dict[str, list[CorpusRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source].append(record)
    for source, source_records in sorted(grouped.items()):
        output_path = output_dir / f"{source}.jsonl"
        temporary_path = output_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            for record in source_records:
                payload = {
                    "messages": [{"role": "assistant", "content": record.content}]
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)


def build_corpus(
    input_dir: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    max_words: int = 1_000,
    min_record_words: int = 80,
    local_only: bool = True,
) -> BuildSummary:
    """Build deduplicated V2 shards from existing extracted newspaper text."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if min_record_words < 1 or max_words < min_record_words:
        raise ValueError("require 1 <= min_record_words <= max_words")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    text_paths = sorted(input_dir.glob("*.txt"))
    if limit is not None:
        text_paths = text_paths[:limit]

    records: list[CorpusRecord] = []
    extracted_records = 0
    below_minimum_records = 0
    off_domain_records = 0
    for text_path in text_paths:
        source = text_path.read_text(encoding="utf-8")
        chunks = chunk_blocks(extract_blocks(source), max_words=max_words)
        extracted_records += len(chunks)
        for ordinal, content in enumerate(chunks):
            if len(WORD_RE.findall(content)) < min_record_words:
                below_minimum_records += 1
                continue
            if local_only and _is_clearly_off_domain(content):
                off_domain_records += 1
                continue
            records.append(CorpusRecord(text_path.stem, ordinal, content))

    kept, exact_duplicates, near_duplicates = deduplicate_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_shards(kept, output_dir)
    return BuildSummary(
        scanned_files=len(text_paths),
        extracted_records=extracted_records,
        below_minimum_records=below_minimum_records,
        off_domain_records=off_domain_records,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
        output_records=len(kept),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-words", type=int, default=1_000)
    parser.add_argument("--min-record-words", type=int, default=80)
    parser.add_argument(
        "--include-clear-foreign",
        action="store_true",
        help="Retain clear foreign wire stories with no Barbados/Caribbean signal",
    )
    return parser.parse_args()


def main() -> None:
    """Build the requested bounded corpus and print aggregate counts."""
    args = parse_args()
    summary = build_corpus(
        args.input_dir,
        args.output_dir,
        limit=args.limit,
        max_words=args.max_words,
        min_record_words=args.min_record_words,
        local_only=not args.include_clear_foreign,
    )
    print(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":
    main()
