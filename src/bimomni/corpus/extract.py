#!/usr/bin/env python3
"""Convert extracted newspaper text into ms-swift pretraining JSONL."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
HEADING_RE = re.compile(r"^##\s+(?P<title>.+)$")
BYLINE_RE = re.compile(r"^(?:by)\s+[A-Z][A-Za-z.' -]+$", re.IGNORECASE)
FIXTURE_RE = re.compile(
    r"^(?:[A-Z0-9&.' -]+)\s+V(?:S\.?)?\s+(?:[A-Z0-9&.' -]+)$"
)
NOISE_PATTERNS = (
    re.compile(r"^<!--.*-->$"),
    re.compile(r"^.*\bAdvocate\b.*\bEstablished October 1895\b.*$", re.IGNORECASE),
    re.compile(r"^.*\bEstablished October 1895\b.*$", re.IGNORECASE),
    re.compile(r"^\$\d+(?:\.\d{2})?\s+VAT Inclusive$", re.IGNORECASE),
    re.compile(r"^Page\s+\d+$", re.IGNORECASE),
    re.compile(r"^.*\b\d{1,2}:\d{2}$"),
    re.compile(r"^RUNOVER\b.*$", re.IGNORECASE),
    re.compile(r"^\(?Cont['’]?d\b.*(?:page\s+\d+|next page).*\)?$", re.IGNORECASE),
    re.compile(r"^.{1,50}\b(?:from|on)\s+(?:Front\s+)?Page\s+\d+$", re.IGNORECASE),
    re.compile(r"^\(?See .*\bPage\s+\d+.*\)?$", re.IGNORECASE),
    re.compile(r"^(?:ABOVE|BELOW|RIGHT|AT RIGHT|BELOW RIGHT)\b.*", re.IGNORECASE),
    re.compile(r"^\((?:DB|JB)\)$"),
    re.compile(r"^(?:SCAN ME|SUBSCRIBE NOW\b.*|SEE MORE\b.*|VISIT US AT\b.*)$", re.IGNORECASE),
    re.compile(r"^(?:Telephone|Contact|Email|Website):\s*.*$", re.IGNORECASE),
    re.compile(r"^(?:https?://|www\.)\S+", re.IGNORECASE),
)
CAPTION_PATTERNS = (
    re.compile(r"^\(?(?:From left|From right|Pictured|Photo)\b", re.IGNORECASE),
    re.compile(r"\((?:[^)]*/)?(?:BGIS|AP|Reuters|Photo[^)]*|Picture[^)]*)\)\s*$", re.IGNORECASE),
    re.compile(r"\b(?:photo|picture)\s+(?:by|courtesy of)\b", re.IGNORECASE),
    re.compile(r"\bPictured (?:here|from left|from right)\b", re.IGNORECASE),
)
LEGAL_ARTICLE_PATTERNS = (
    re.compile(r"\bSUPREME COURT OF BARBADOS\b", re.IGNORECASE),
    re.compile(r"\bNOTICE OF APPLICATION FOR DECLARATION OF OWNERSHIP\b", re.IGNORECASE),
    re.compile(r"\bIN THE MATTER OF THE LAND \(TITLE PROCEEDINGS\) ACT\b", re.IGNORECASE),
    re.compile(r"\bCLAIM NO\.\s*[A-Z0-9/-]+", re.IGNORECASE),
)
OCR_PREFIXES = {
    "A s ": "As ",
    "B arbados ": "Barbados ",
    "C ricket ": "Cricket ",
    "T he ": "The ",
    "T wo ": "Two ",
    "W e've ": "We've ",
    "W ho ": "Who ",
}


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A cleaned heading or prose paragraph from the source document."""

    text: str
    is_heading: bool


def word_count(text: str) -> int:
    """Return the approximate number of language-model words in text."""
    return len(WORD_RE.findall(text))


def normalise_text(text: str) -> str:
    """Normalise HTML, punctuation, whitespace, and common OCR artefacts."""
    text = html.unescape(text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": " - ",
        "\u2014": " - ",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        r"\s*\(Cont'd\s+(?:from|on)\s+(?:next\s+)?Page\s+\d+\.?\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+SUBSCRIBE NOW\b.*$", "", text, flags=re.IGNORECASE)
    for source, replacement in OCR_PREFIXES.items():
        if text.startswith(source):
            text = replacement + text[len(source) :]
            break
    return text


def is_noise(text: str) -> bool:
    """Return whether a source block is extraction or publication boilerplate."""
    return any(pattern.match(text) for pattern in NOISE_PATTERNS)


def is_caption(text: str) -> bool:
    """Return whether a block is a newspaper photograph caption."""
    if any(pattern.search(text) for pattern in CAPTION_PATTERNS):
        return True
    caption_action = re.search(
        r"\b(?:poses?|present(?:s|ed|ing)|accepts?|greeting|looking on|helping)\b",
        text,
        re.IGNORECASE,
    )
    return word_count(text) <= 45 and caption_action is not None and '"' not in text


def is_corrupt_prose(text: str) -> bool:
    """Detect blocks dominated by isolated OCR characters rather than prose."""
    words = WORD_RE.findall(text)
    if len(words) < 12:
        return False
    isolated = sum(len(word) == 1 for word in words)
    return isolated / len(words) > 0.15


def is_prose(text: str, *, min_words: int = 8) -> bool:
    """Return whether text contains enough sentence-like material to train on."""
    if word_count(text) < min_words:
        return False
    if not re.search(r"[a-z]{3}", text):
        return False
    return not is_corrupt_prose(text)


def is_useful_heading(text: str) -> bool:
    """Reject bylines and table-like headings while retaining article headings."""
    if BYLINE_RE.fullmatch(text) or FIXTURE_RE.fullmatch(text):
        return False
    words = WORD_RE.findall(text)
    if not words or len(words) > 24:
        return False
    return True


def is_plain_heading(text: str, next_text: str | None) -> bool:
    """Detect a standalone headline in Docling's plain-text newspaper output."""
    words = WORD_RE.findall(text)
    if not next_text or not 1 <= len(words) <= 14 or len(text) > 160:
        return False
    if text[-1:] in ".?!;:" or text.startswith(('"', "'", "(")):
        return False
    if is_noise(text) or is_caption(text) or not is_useful_heading(text):
        return False
    if not is_prose(next_text, min_words=12):
        return False

    capitalised = sum(word[0].isupper() for word in words if word)
    has_uppercase_word = any(len(word) > 1 and word.isupper() for word in words)
    return text.isupper() or has_uppercase_word or capitalised / len(words) >= 0.25


def extract_blocks(source: str, *, min_paragraph_words: int = 8) -> list[TextBlock]:
    """Extract cleaned headings and prose paragraphs from PDF-derived Markdown."""
    candidates: list[tuple[str, bool]] = []
    for raw_block in re.split(r"\n\s*\n", source):
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        heading_match = HEADING_RE.fullmatch(raw_block)
        text = normalise_text(heading_match.group("title") if heading_match else raw_block)
        if text:
            candidates.append((text, heading_match is not None))

    next_usable: str | None = None
    next_texts: list[str | None] = [None] * len(candidates)
    for index in range(len(candidates) - 1, -1, -1):
        next_texts[index] = next_usable
        candidate = candidates[index][0]
        if not is_noise(candidate) and not is_caption(candidate):
            next_usable = candidate

    blocks: list[TextBlock] = []
    for index, (text, explicit_heading) in enumerate(candidates):
        if is_noise(text) or is_caption(text):
            continue

        next_text = next_texts[index]
        if explicit_heading or is_plain_heading(text, next_text):
            if is_useful_heading(text):
                blocks.append(TextBlock(text, is_heading=True))
            continue

        if (
            BYLINE_RE.fullmatch(text)
            or not is_prose(text, min_words=min_paragraph_words)
        ):
            continue
        blocks.append(TextBlock(text, is_heading=False))
    return blocks


def _split_long_text(text: str, max_words: int) -> list[str]:
    """Split oversized prose at sentence boundaries, falling back to word spans."""
    if word_count(text) <= max_words:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1:
        matches = list(WORD_RE.finditer(text))
        return [
            text[matches[start].start() : matches[min(start + max_words, len(matches)) - 1].end()]
            for start in range(0, len(matches), max_words)
        ]

    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = word_count(sentence)
        if sentence_words > max_words:
            if current:
                parts.append(" ".join(current))
                current = []
                current_words = 0
            parts.extend(_split_long_text(sentence, max_words))
        elif current and current_words + sentence_words > max_words:
            parts.append(" ".join(current))
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_blocks(
    blocks: Sequence[TextBlock],
    *,
    min_words: int = 250,
    max_words: int = 1_000,
) -> list[str]:
    """Build bounded chunks without crossing detected article boundaries."""
    if min_words < 1 or max_words < min_words:
        raise ValueError("Require 1 <= min_words <= max_words")

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    has_prose = False

    def flush() -> None:
        nonlocal current, current_words, has_prose
        if current and has_prose:
            article = "\n\n".join(current)
            prefix = article[:500]
            if not any(pattern.search(prefix) for pattern in LEGAL_ARTICLE_PATTERNS):
                chunks.append(article)
            current = []
            current_words = 0
            has_prose = False

    for block in blocks:
        if block.is_heading:
            if has_prose:
                flush()
            current.append(block.text)
            current_words += word_count(block.text)
            continue

        for part in _split_long_text(block.text, max_words):
            part_words = word_count(part)
            if has_prose and current_words + part_words > max_words:
                flush()
            current.append(part)
            current_words += part_words
            has_prose = True

    flush()
    return chunks


def write_jsonl(chunks: Iterable[str], output_path: Path) -> int:
    """Write chunks as assistant-only ms-swift pretraining records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for chunk in chunks:
                record = {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": chunk,
                        }
                    ]
                }
                json.dump(record, output, ensure_ascii=False)
                output.write("\n")
                count += 1
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


def generate_chunks(
    input_path: Path,
    output_path: Path,
    *,
    min_words: int = 250,
    max_words: int = 1_000,
    min_paragraph_words: int = 8,
) -> tuple[int, int]:
    """Read extracted text and write cleaned pretraining chunks."""
    source = input_path.read_text(encoding="utf-8")
    blocks = extract_blocks(source, min_paragraph_words=min_paragraph_words)
    chunks = chunk_blocks(blocks, min_words=min_words, max_words=max_words)
    return len(blocks), write_jsonl(chunks, output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PDF-derived Markdown text file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output JSONL path (defaults to the input path with .jsonl)",
    )
    parser.add_argument("--min-words", type=int, default=250)
    parser.add_argument("--max-words", type=int, default=1_000)
    parser.add_argument("--min-paragraph-words", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Run the newspaper-to-JSONL conversion."""
    args = parse_args()
    output_path = args.output or args.input.with_suffix(".jsonl")
    blocks, chunks = generate_chunks(
        args.input,
        output_path,
        min_words=args.min_words,
        max_words=args.max_words,
        min_paragraph_words=args.min_paragraph_words,
    )
    print(
        f"Wrote {chunks} training chunks by combining {blocks} cleaned "
        f"headings/paragraphs into {output_path}"
    )


if __name__ == "__main__":
    main()
