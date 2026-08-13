"""Tests for batch conversion of newspaper PDFs into training JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bimomni.corpus.batch_extract import (
    ProcessStatus,
    process_directory,
    process_pdf,
)


PROSE = (
    "The Barbados Tourism Marketing Inc. announced a new campaign for visitors "
    "and tourism partners across the island."
)


def test_process_pdf_skips_existing_jsonl(tmp_path: Path) -> None:
    pdf = tmp_path / "2024-06-08.pdf"
    pdf.touch()
    pdf.with_suffix(".jsonl").write_text("already complete\n")

    def unexpected_runner(*args: object, **kwargs: object) -> None:
        raise AssertionError("Docling should not run for a completed edition")

    status = process_pdf(pdf, run_command=unexpected_runner)

    assert status is ProcessStatus.SKIPPED
    assert pdf.with_suffix(".jsonl").read_text() == "already complete\n"


def test_process_pdf_reuses_existing_text(tmp_path: Path) -> None:
    pdf = tmp_path / "2024-06-08.pdf"
    pdf.touch()
    pdf.with_suffix(".txt").write_text(PROSE)

    def unexpected_runner(*args: object, **kwargs: object) -> None:
        raise AssertionError("Docling should not run when extracted text exists")

    status = process_pdf(
        pdf,
        run_command=unexpected_runner,
        min_words=1,
        max_words=100,
    )

    assert status is ProcessStatus.REUSED_TEXT
    record = json.loads(pdf.with_suffix(".jsonl").read_text())
    assert record["messages"][0] == {"role": "assistant", "content": PROSE}


def test_process_pdf_runs_docling_into_temporary_directory(tmp_path: Path) -> None:
    pdf = tmp_path / "2024-06-08.pdf"
    pdf.touch()
    commands: list[list[str]] = []

    def fake_runner(command: list[str], *, check: bool) -> None:
        assert check is True
        commands.append(command)
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.joinpath("2024-06-08.txt").write_text(PROSE)

    status = process_pdf(
        pdf,
        run_command=fake_runner,
        min_words=1,
        max_words=100,
    )

    assert status is ProcessStatus.EXTRACTED
    assert commands[0][:6] == [
        "docling",
        "convert",
        "--to",
        "text",
        "--ocr",
        "--ocr-mode",
    ]
    assert commands[0][6] == "full_page"
    assert commands[0][-1] == str(pdf)
    assert "--no-tables" in commands[0]
    assert "--pdf-backend" in commands[0]
    assert "--num-threads" in commands[0]
    assert "--page-batch-size" in commands[0]
    assert pdf.with_suffix(".txt").read_text() == PROSE
    assert pdf.with_suffix(".jsonl").exists()


def test_process_pdf_uses_daemon_converter_when_provided(tmp_path: Path) -> None:
    pdf = tmp_path / "2024-06-08.pdf"
    pdf.touch()

    class FakeConverter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def convert(self, path: str) -> object:
            self.calls.append(path)
            return _FakeResult(PROSE)

    converter = FakeConverter()
    status = process_pdf(
        pdf,
        run_command=lambda command, check: None,
        converter=converter,
        min_words=1,
        max_words=100,
    )

    assert status is ProcessStatus.EXTRACTED
    assert converter.calls == [str(pdf)]
    assert pdf.with_suffix(".txt").read_text() == PROSE
    assert pdf.with_suffix(".jsonl").exists()


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.document = _FakeDocument(text)


class _FakeDocument:
    def __init__(self, text: str) -> None:
        self._text = text

    def export_to_text(self) -> str:
        return self._text


def test_process_pdf_fails_if_docling_does_not_create_text(tmp_path: Path) -> None:
    pdf = tmp_path / "2024-06-08.pdf"
    pdf.touch()

    with pytest.raises(RuntimeError, match="did not create"):
        process_pdf(pdf, run_command=lambda command, check: None)

    assert not pdf.with_suffix(".txt").exists()
    assert not pdf.with_suffix(".jsonl").exists()


def test_process_directory_uses_daemon_converter_for_pool(tmp_path: Path) -> None:
    for stem in ("2024-06-08", "2024-06-09", "2024-06-10"):
        (tmp_path / f"{stem}.pdf").touch()

    class FakeConverter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def convert(self, path: str) -> object:
            self.calls.append(path)
            return _FakeResult(PROSE)

    converter = FakeConverter()
    summary = process_directory(
        tmp_path,
        run_command=lambda command, check: None,
        converter=converter,
        min_words=1,
        max_words=100,
        workers=2,
    )

    assert summary.scanned == 3
    assert summary.skipped == 0
    assert summary.extracted == 3
    assert summary.failures == ()
    for stem in ("2024-06-08", "2024-06-09", "2024-06-10"):
        assert (tmp_path / f"{stem}.jsonl").exists()
    assert sorted(converter.calls) == sorted(
        [
            str(tmp_path / "2024-06-08.pdf"),
            str(tmp_path / "2024-06-09.pdf"),
            str(tmp_path / "2024-06-10.pdf"),
        ]
    )


def test_process_directory_rejects_converter_without_workers(tmp_path: Path) -> None:
    (tmp_path / "2024-06-08.pdf").touch()

    class FakeConverter:
        def convert(self, path: str) -> object:  # pragma: no cover
            return _FakeResult(PROSE)

    with pytest.raises(ValueError, match="converter implies workers"):
        process_directory(
            tmp_path,
            run_command=lambda command, check: None,
            converter=FakeConverter(),
            workers=1,
        )


def test_process_directory_records_elapsed_time(tmp_path: Path) -> None:
    (tmp_path / "2024-06-08.pdf").touch()

    def fake_runner(command: list[str], *, check: bool) -> None:
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.joinpath("2024-06-08.txt").write_text(PROSE)

    summary = process_directory(
        tmp_path,
        run_command=fake_runner,
        min_words=1,
        max_words=100,
    )

    assert summary.elapsed_seconds >= 0.0
