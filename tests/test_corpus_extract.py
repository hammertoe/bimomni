"""Tests for preparing newspaper text for domain-adaptive pretraining."""

from __future__ import annotations

import json
from pathlib import Path

from bimomni.corpus.extract import (
    TextBlock,
    chunk_blocks,
    extract_blocks,
    write_jsonl,
)


def test_extract_blocks_removes_pdf_noise_and_normalises_ocr() -> None:
    source = """
<!-- image -->

## By Janelle Brathwaite

RUNOVER from Page 3

## Barbados in the spotlight!

B arbados is home to the iconic Kensington Öval, where local cricket fans
gather throughout the season to support the West Indies team.

Telephone: 626-4300 Emergencies: 626-9000
"""

    assert extract_blocks(source) == [
        TextBlock("Barbados in the spotlight!", is_heading=True),
        TextBlock(
            "Barbados is home to the iconic Kensington Oval, where local cricket "
            "fans gather throughout the season to support the West Indies team.",
            is_heading=False,
        ),
    ]


def test_extract_blocks_rejects_schedule_rows_and_corrupt_ocr() -> None:
    source = """
## WEST INDIES V UGANDA

JUN

08

GUYANA NATIONAL STADIUM I 20:30

The Barbados Workers' Union has represented workers across the island for
many years and remains an important institution in Barbadian public life.

Members from left are George Headley and Joe Small assistar t t rrt t t -i
tt Lt t tr L t rge Francis seated at front.
"""

    assert extract_blocks(source) == [
        TextBlock(
            "The Barbados Workers' Union has represented workers across the island "
            "for many years and remains an important institution in Barbadian public life.",
            is_heading=False,
        )
    ]


def test_extract_blocks_removes_inline_continuation_and_subscription_text() -> None:
    source = """
## (Cont'd from Page 19)

Today, we shine the spotlight on memorable moments from Kensington's fans.
(Čont'd on Page 32)

With sincere appreciation, The Barbados Advocate SUBSCRIBE NOW FOR OUR
ENHANCED ONLINE COPY FOR ONLY $2.
"""

    assert extract_blocks(source) == [
        TextBlock(
            "Today, we shine the spotlight on memorable moments from Kensington's fans.",
            is_heading=False,
        ),
    ]


def test_chunk_blocks_keeps_coherent_paragraphs_and_soft_heading_boundaries() -> None:
    blocks = [
        TextBlock("Barbados", is_heading=True),
        TextBlock("One two three four five six seven eight nine ten.", is_heading=False),
        TextBlock("Cricket", is_heading=True),
        TextBlock("Eleven twelve thirteen fourteen fifteen sixteen.", is_heading=False),
        TextBlock("Tourism", is_heading=True),
        TextBlock("Seventeen eighteen nineteen twenty twenty-one.", is_heading=False),
    ]

    chunks = chunk_blocks(blocks, min_words=10, max_words=20)

    assert chunks == [
        "Barbados\n\nOne two three four five six seven eight nine ten.",
        "Cricket\n\nEleven twelve thirteen fourteen fifteen sixteen.",
        "Tourism\n\nSeventeen eighteen nineteen twenty twenty-one.",
    ]


def test_extract_blocks_detects_plain_text_newspaper_headlines() -> None:
    source = """
Advocate Established October 1895 Wednesday September 23, 2015

$1 VAT Inclusive

NCC workers' case to be heard next week

After a year of waiting, retrenched workers will finally present their case
to the Employment Rights Tribunal at the conference centre next Wednesday.

Alexandra School re-opens tomorrow

The Ministry of Education advised that Alexandra School in St Peter will
re-open tomorrow after it was closed because of flooding earlier this week.
"""

    assert extract_blocks(source) == [
        TextBlock("NCC workers' case to be heard next week", is_heading=True),
        TextBlock(
            "After a year of waiting, retrenched workers will finally present "
            "their case to the Employment Rights Tribunal at the conference "
            "centre next Wednesday.",
            is_heading=False,
        ),
        TextBlock("Alexandra School re-opens tomorrow", is_heading=True),
        TextBlock(
            "The Ministry of Education advised that Alexandra School in St Peter "
            "will re-open tomorrow after it was closed because of flooding earlier "
            "this week.",
            is_heading=False,
        ),
    ]


def test_chunk_blocks_never_combines_articles_to_reach_minimum() -> None:
    blocks = [
        TextBlock("First local headline", is_heading=True),
        TextBlock(
            "This first Barbados article contains enough meaningful prose to be useful.",
            is_heading=False,
        ),
        TextBlock("Second local headline", is_heading=True),
        TextBlock(
            "This second Barbados article must remain a separate training record.",
            is_heading=False,
        ),
    ]

    assert chunk_blocks(blocks, min_words=250, max_words=1_000) == [
        "First local headline\n\n"
        "This first Barbados article contains enough meaningful prose to be useful.",
        "Second local headline\n\n"
        "This second Barbados article must remain a separate training record.",
    ]


def test_extract_blocks_removes_mastheads_captions_and_page_markers() -> None:
    source = """
Advocate Established October 1895 Monday June 19, 2023

Page 4

(From left) Officials inspect the plans at the historic property.

From left: Tourism officials presenting an award after the ceremony.

Green thumbs up! Prime Minister Mia Mottley poses for a photograph. (C. Pitt/BGIS)

Sam Lord's Castle restoration begins

Workers have started preparing the historic St Philip property for restoration,
and the project is expected to provide employment for Barbadians.

CLEAN UP from Page 1
"""

    assert extract_blocks(source) == [
        TextBlock("Sam Lord's Castle restoration begins", is_heading=True),
        TextBlock(
            "Workers have started preparing the historic St Philip property for "
            "restoration, and the project is expected to provide employment for "
            "Barbadians.",
            is_heading=False,
        ),
    ]


def test_write_jsonl_uses_ms_swift_pretraining_format(tmp_path: Path) -> None:
    output = tmp_path / "training" / "barbados.jsonl"

    count = write_jsonl(["BTMI announced a new campaign."], output)

    assert count == 1
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records == [
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "BTMI announced a new campaign.",
                }
            ]
        }
    ]
