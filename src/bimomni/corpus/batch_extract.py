#!/usr/bin/env python3
"""Convert every unfinished newspaper PDF into pretraining JSONL."""

from __future__ import annotations

import argparse
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from bimomni.corpus.extract import generate_chunks


RunCommand = Callable[..., object]


class ProcessStatus(StrEnum):
    """Outcome for one newspaper edition."""

    SKIPPED = "skipped"
    REUSED_TEXT = "reused text"
    EXTRACTED = "extracted"


@dataclass(frozen=True, slots=True)
class BatchSummary:
    """Aggregate outcomes from a directory conversion."""

    scanned: int
    skipped: int
    reused_text: int
    extracted: int
    failures: tuple[tuple[Path, str], ...]
    elapsed_seconds: float


def build_docling_command(
    pdf_path: Path,
    output_directory: Path,
    *,
    docling_executable: str = "docling",
    ocr_mode: str = "full_page",
    num_threads: int = 8,
    page_batch_size: int = 8,
    pdf_backend: str = "pypdfium2",
    enable_tables: bool = False,
) -> list[str]:
    """Build the Docling command tuned for fast text-only extraction."""
    command = [
        docling_executable,
        "convert",
        "--to",
        "text",
        "--ocr",
        "--ocr-mode",
        ocr_mode,
        "--pdf-backend",
        pdf_backend,
        "--num-threads",
        str(num_threads),
        "--page-batch-size",
        str(page_batch_size),
        "--quiet",
    ]
    if not enable_tables:
        command.append("--no-tables")
    command.extend(["--output", str(output_directory), str(pdf_path)])
    return command


def extract_text(
    pdf_path: Path,
    text_path: Path,
    *,
    docling_executable: str = "docling",
    run_command: RunCommand = subprocess.run,
    ocr_mode: str = "full_page",
    num_threads: int = 8,
    page_batch_size: int = 8,
    pdf_backend: str = "pypdfium2",
    enable_tables: bool = False,
) -> None:
    """Run Docling and atomically publish its text output beside the PDF."""
    with TemporaryDirectory(prefix=f".{pdf_path.stem}-docling-", dir=pdf_path.parent) as temporary:
        output_directory = Path(temporary)
        command = build_docling_command(
            pdf_path,
            output_directory,
            docling_executable=docling_executable,
            ocr_mode=ocr_mode,
            num_threads=num_threads,
            page_batch_size=page_batch_size,
            pdf_backend=pdf_backend,
            enable_tables=enable_tables,
        )
        run_command(command, check=True)

        generated_text = output_directory / text_path.name
        if not generated_text.is_file() or generated_text.stat().st_size == 0:
            raise RuntimeError(
                f"Docling did not create non-empty text output for {pdf_path.name}"
            )
        generated_text.replace(text_path)


def process_pdf(
    pdf_path: Path,
    *,
    docling_executable: str = "docling",
    run_command: RunCommand = subprocess.run,
    converter: object | None = None,
    min_words: int = 250,
    max_words: int = 1_000,
    min_paragraph_words: int = 8,
    ocr_mode: str = "full_page",
    num_threads: int = 8,
    page_batch_size: int = 8,
    pdf_backend: str = "pypdfium2",
    enable_tables: bool = False,
) -> ProcessStatus:
    """Create one edition's JSONL unless a completed output already exists."""
    jsonl_path = pdf_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        return ProcessStatus.SKIPPED

    text_path = pdf_path.with_suffix(".txt")
    if text_path.is_file() and text_path.stat().st_size > 0:
        status = ProcessStatus.REUSED_TEXT
    else:
        text_path.unlink(missing_ok=True)
        if converter is not None:
            convert_with_daemon(
                pdf_path=pdf_path,
                text_path=text_path,
                converter=converter,
            )
        else:
            extract_text(
                pdf_path,
                text_path,
                docling_executable=docling_executable,
                run_command=run_command,
                ocr_mode=ocr_mode,
                num_threads=num_threads,
                page_batch_size=page_batch_size,
                pdf_backend=pdf_backend,
                enable_tables=enable_tables,
            )
        status = ProcessStatus.EXTRACTED

    _, chunk_count = generate_chunks(
        text_path,
        jsonl_path,
        min_words=min_words,
        max_words=max_words,
        min_paragraph_words=min_paragraph_words,
    )
    if chunk_count == 0:
        jsonl_path.unlink(missing_ok=True)
        raise RuntimeError(f"No usable training chunks were found in {text_path.name}")
    return status


def convert_with_daemon(
    *,
    pdf_path: Path,
    text_path: Path,
    converter: object,
) -> None:
    """Convert one PDF using a long-lived Docling `DocumentConverter`."""
    result = converter.convert(str(pdf_path))
    document = getattr(result, "document", None)
    if document is None:
        raise RuntimeError(f"Docling produced no document for {pdf_path.name}")
    markdown = getattr(document, "export_to_text", None)
    if markdown is None:
        raise RuntimeError("Docling document is missing export_to_text()")
    text_path.write_text(markdown(), encoding="utf-8")
    if text_path.stat().st_size == 0:
        raise RuntimeError(
            f"Docling produced empty text output for {pdf_path.name}"
        )


def _process_one(
    pdf_path: Path,
    *,
    process_kwargs: dict[str, object],
) -> tuple[Path, ProcessStatus | str]:
    """Run process_pdf for a single PDF, returning (path, outcome)."""
    try:
        return pdf_path, process_pdf(pdf_path, **process_kwargs)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return pdf_path, str(error)


def build_daemon_converter(
    *,
    ocr_mode: str = "full_page",
    num_threads: int = 8,
    page_batch_size: int = 8,
    pdf_backend: str = "pypdfium2",
    enable_tables: bool = False,
    artifacts_path: str | None = None,
) -> object:
    """Construct a long-lived Docling `DocumentConverter`."""
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415

    options = PdfPipelineOptions()
    options.do_ocr = True
    if not enable_tables:
        options.do_table_structure = False
    from docling.datamodel.pipeline_options import OcrMode  # noqa: PLC0415

    options.ocr_options.mode = (
        OcrMode.FULL_PAGE if ocr_mode == "full_page" else OcrMode.DEFAULT
    )
    options.ocr_batch_size = page_batch_size
    options.layout_batch_size = page_batch_size
    options.table_batch_size = page_batch_size
    options.accelerator_options.num_threads = num_threads
    if artifacts_path:
        options.artifacts_path = artifacts_path

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    )


def process_directory(
    directory: Path,
    *,
    docling_executable: str = "docling",
    run_command: RunCommand = subprocess.run,
    converter: object | None = None,
    min_words: int = 250,
    max_words: int = 1_000,
    min_paragraph_words: int = 8,
    limit: int | None = None,
    fail_fast: bool = False,
    workers: int = 1,
    ocr_mode: str = "full_page",
    num_threads: int = 8,
    page_batch_size: int = 8,
    pdf_backend: str = "pypdfium2",
    enable_tables: bool = False,
    artifacts_path: str | None = None,
) -> BatchSummary:
    """Process PDFs in a directory, retaining failures for a final summary."""
    pdf_paths = sorted(directory.glob("*.pdf"))
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    counts = {status: 0 for status in ProcessStatus}
    failures: list[tuple[Path, str]] = []
    total = len(pdf_paths)
    started = time.monotonic()

    if workers <= 1 and converter is not None:
        raise ValueError(
            "converter implies workers > 1; pass workers=2 or higher"
        )

    def handle(
        index: int,
        pdf_path: Path,
        outcome: ProcessStatus | str,
        counts: dict[ProcessStatus, int],
        failures: list[tuple[Path, str]],
    ) -> None:
        if isinstance(outcome, ProcessStatus):
            counts[outcome] += 1
            print(f"[{index}/{total}] {outcome.value}: {pdf_path.name}", flush=True)
            return
        failures.append((pdf_path, outcome))
        print(f"[{index}/{total}] failed: {pdf_path.name}: {outcome}", flush=True)
        if fail_fast:
            raise RuntimeError(f"{pdf_path.name}: {outcome}")

    if workers <= 1:
        process_kwargs: dict[str, object] = {
            "docling_executable": docling_executable,
            "run_command": run_command,
            "min_words": min_words,
            "max_words": max_words,
            "min_paragraph_words": min_paragraph_words,
            "ocr_mode": ocr_mode,
            "num_threads": num_threads,
            "page_batch_size": page_batch_size,
            "pdf_backend": pdf_backend,
            "enable_tables": enable_tables,
        }
        for index, pdf_path in enumerate(pdf_paths, start=1):
            pdf_path, outcome = _process_one(pdf_path, process_kwargs=process_kwargs)
            handle(index, pdf_path, outcome, counts, failures)
    else:
        if converter is None:
            converter = build_daemon_converter(
                ocr_mode=ocr_mode,
                num_threads=num_threads,
                page_batch_size=page_batch_size,
                pdf_backend=pdf_backend,
                enable_tables=enable_tables,
                artifacts_path=artifacts_path,
            )
        process_kwargs = {
            "docling_executable": docling_executable,
            "run_command": run_command,
            "converter": converter,
            "min_words": min_words,
            "max_words": max_words,
            "min_paragraph_words": min_paragraph_words,
            "ocr_mode": ocr_mode,
            "num_threads": num_threads,
            "page_batch_size": page_batch_size,
            "pdf_backend": pdf_backend,
            "enable_tables": enable_tables,
        }
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_one, pdf_path, process_kwargs=process_kwargs): (idx, pdf_path)
                for idx, pdf_path in enumerate(pdf_paths, start=1)
            }
            for future in as_completed(futures):
                index, pdf_path = futures[future]
                pdf_path, outcome = future.result()
                handle(index, pdf_path, outcome, counts, failures)

    elapsed = time.monotonic() - started
    return BatchSummary(
        scanned=total,
        skipped=counts[ProcessStatus.SKIPPED],
        reused_text=counts[ProcessStatus.REUSED_TEXT],
        extracted=counts[ProcessStatus.EXTRACTED],
        failures=tuple(failures),
        elapsed_seconds=elapsed,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("model/pdfs"))
    parser.add_argument("--docling-executable", default="docling")
    parser.add_argument("--min-words", type=int, default=250)
    parser.add_argument("--max-words", type=int, default=1_000)
    parser.add_argument("--min-paragraph-words", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Docling workers")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Use a long-lived in-process Docling converter (faster, uses --workers)",
    )
    parser.add_argument("--ocr-mode", choices=("default", "full_page"), default="full_page")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--page-batch-size", type=int, default=8)
    parser.add_argument(
        "--pdf-backend",
        choices=("pypdfium2", "docling_parse"),
        default="pypdfium2",
    )
    parser.add_argument("--enable-tables", action="store_true")
    parser.add_argument("--artifacts-path", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the resumable PDF conversion batch."""
    args = parse_args()
    if not args.directory.is_dir():
        raise SystemExit(f"Not a directory: {args.directory}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.daemon and args.workers <= 1:
        print(
            "--daemon implies a worker pool; defaulting --workers to 2",
            flush=True,
        )
        args.workers = 2

    summary = process_directory(
        args.directory,
        docling_executable=args.docling_executable,
        min_words=args.min_words,
        max_words=args.max_words,
        min_paragraph_words=args.min_paragraph_words,
        limit=args.limit,
        fail_fast=args.fail_fast,
        workers=args.workers,
        ocr_mode=args.ocr_mode,
        num_threads=args.num_threads,
        page_batch_size=args.page_batch_size,
        pdf_backend=args.pdf_backend,
        enable_tables=args.enable_tables,
        artifacts_path=args.artifacts_path,
    )
    elapsed_minutes = summary.elapsed_seconds / 60
    print(
        "Summary: "
        f"scanned={summary.scanned} skipped={summary.skipped} "
        f"reused_text={summary.reused_text} extracted={summary.extracted} "
        f"failed={len(summary.failures)} "
        f"elapsed={elapsed_minutes:.1f}min"
    )
    if summary.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()