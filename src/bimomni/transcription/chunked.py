"""Transcribe an audio/video file with the local BimOmni model using overlapping
audio windows.

Each BimOmni request receives at most 30 seconds of audio (frames are an
optional disambiguation aid that is off by default). The output is a verbatim
audio transcript. Results retain both the literal chunk concatenation and a
boundary de-duplicated version for inspection.

The module is independent of any specific application: pass it a video or
audio file, the local model path, and an output directory. It does not assume
Pulse fixtures.
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

# MLX is Apple-Silicon only. The patch is a no-op on Linux CI and on
# mlx-vlm >= 0.6.10, so this is safe to import unconditionally.
from bimomni.inference.audio_compat import ensure_audio_patch as _ensure_audio_patch

# The MLX audio patch is applied lazily inside ``_transcribe_window`` and
# ``main`` so the rest of this module (helpers, dataclasses, prompts) is
# importable on Linux CI where mlx-vlm is unavailable.

if TYPE_CHECKING:
    import mlx.core as mx  # pragma: no cover
    import mlx_vlm  # pragma: no cover

import numpy as np

DEFAULT_MODEL_PATH = Path("model/qwen3-omni-4bit")
DEFAULT_OUTPUT_DIR = Path("transcripts")
WINDOW_SECONDS = 30.0
OVERLAP_SECONDS = 5.0
FRAMES_PER_WINDOW = 3
FRAME_HEIGHT = 720
MAX_TOKENS = 2_048
STITCH_MAX_TOKENS = 2_048

V4_AUDIO_PROMPT = (
    "Transcribe every intelligible spoken word in this TikTok audio verbatim and in order. "
    "Preserve exact wording, repetitions, and Bajan dialect. Correct only obvious proper-noun "
    "spelling. Do not summarize, classify, explain, or describe the audio. Do not transcribe "
    "background song lyrics. Return only the transcript."
)

V4_STITCH_PROMPT = (
    "Two consecutive transcript chunks overlap. Return only the portion of NEXT CHUNK that should "
    "be appended after PREVIOUS TAIL. Remove only its duplicated opening overlap. Preserve every "
    "remaining word from NEXT CHUNK in order, including its exact ending. Do not rewrite the "
    "previous transcript, summarize, paraphrase, add facts, or omit non-overlapping details. "
    "Return only the continuation text without labels or commentary."
)

log = logging.getLogger("spike.tiktok_v4_multimodal_transcription")


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One bounded chronological audio interval."""

    index: int
    start_s: float
    end_s: float


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def audio_windows(
    duration_s: float,
    *,
    window_s: float = WINDOW_SECONDS,
    overlap_s: float = OVERLAP_SECONDS,
) -> tuple[AudioWindow, ...]:
    """Return full-coverage windows, advancing by ``window_s - overlap_s``."""
    if duration_s <= 0:
        return ()
    if not 0 <= overlap_s < window_s:
        raise ValueError("overlap_s must be non-negative and shorter than window_s")
    windows: list[AudioWindow] = []
    start_s = 0.0
    while start_s < duration_s:
        end_s = min(start_s + window_s, duration_s)
        windows.append(AudioWindow(len(windows), start_s, end_s))
        if end_s >= duration_s:
            break
        start_s += window_s - overlap_s
    return tuple(windows)


def frame_timestamps(window: AudioWindow, *, count: int = FRAMES_PER_WINDOW) -> tuple[float, ...]:
    """Return evenly spaced interior source-video positions for a window."""
    duration_s = window.end_s - window.start_s
    return tuple(window.start_s + duration_s * (index + 1) / (count + 1) for index in range(count))


def _extract_audio_window(video_path: Path, window: AudioWindow, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{window.start_s:.3f}",
            "-i", str(video_path), "-t", f"{window.end_s - window.start_s:.3f}",
            "-vn", "-ar", "16000", "-ac", "1", "-acodec", "libmp3lame", "-b:a", "64k",
            str(output_path),
        ],
        check=True,
    )


def _extract_frames(video_path: Path, window: AudioWindow, output_dir: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, timestamp_s in enumerate(frame_timestamps(window)):
        output_path = output_dir / f"frame-{index:02d}-{timestamp_s:07.3f}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{timestamp_s:.3f}",
                "-i", str(video_path), "-frames:v", "1", "-vf", f"scale=-2:{FRAME_HEIGHT}", str(output_path),
            ],
            check=True,
        )
        paths.append(output_path)
    return tuple(paths)


def _messages(audio_path: Path, frame_paths: tuple[Path, ...], window: AudioWindow) -> list[dict[str, Any]]:
    frame_guidance = (
        " The supplied frames are from this same time window and may only disambiguate a spoken "
        "name; do not describe frames or add visual-only text to the transcript."
        if frame_paths else ""
    )
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": V4_AUDIO_PROMPT if not frame_paths else (
                    "Transcribe the spoken audio faithfully and in chronological order. "
                    "Preserve wording, repetitions, and Bajan dialect. Correct only obvious "
                    f"spelling of proper nouns.{frame_guidance} Write [unintelligible] for "
                    "speech you cannot hear. Return only the transcript, no labels or commentary."
                ),
            }],
        },
        {
            "role": "user",
                "content": [
                *[{"type": "image", "image": str(path)} for path in frame_paths],
                {"type": "audio", "audio": str(audio_path)},
                {
                    "type": "text",
                    "text": (
                        (f"Transcribe audio from {window.start_s:.1f}s to {window.end_s:.1f}s. "
                         "Use the images only to spell a word that is audibly spoken.")
                        if frame_paths else "Transcribe this TikTok audio verbatim."
                    ),
                },
            ],
        },
    ]


def _transcribe_window(model: Any, processor: Any, audio_path: Path, frame_paths: tuple[Path, ...], window: AudioWindow) -> dict[str, Any]:
    """Run one BimOmni inference for the given audio window.

    Heavy MLX imports are deferred so this module remains importable on
    Linux CI where mlx-vlm is unavailable.
    """
    import mlx.core as mx
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.generate import generate_step
    from mlx_vlm.models.qwen3_omni_moe.omni_utils import process_multimodal_info
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_audio

    messages = _messages(audio_path, frame_paths, window)
    if not frame_paths:
        chat_prompt = apply_chat_template(
            processor, model.config, messages, num_audios=1,
            add_generation_prompt=True, enable_thinking=True,
        )
        started = time.monotonic()
        result = mlx_generate(
            model, processor, chat_prompt, audio=[str(audio_path)], max_tokens=MAX_TOKENS,
            temperature=0.0, repetition_penalty=1.05, repetition_context_size=64, verbose=False,
        )
        raw_text = (result.text or "").strip()
        return {
            "raw_text": raw_text,
            "transcript": raw_text,
            "generation_tokens": getattr(result, "generation_tokens", None),
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    audios, images, videos = process_multimodal_info(messages, use_audio_in_video=False)
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    inputs = processor(
        text=[text],
        audio=[load_audio(path, sr=processor.feature_extractor.sampling_rate) for path in audios],
        images=images or None,
        videos=videos or None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    model_inputs = {
        key: mx.array(value.numpy()) if hasattr(value, "numpy") else value
        for key, value in inputs.items()
    }
    if "feature_attention_mask" in model_inputs and "audio_feature_lengths" not in model_inputs:
        model_inputs["audio_feature_lengths"] = model_inputs["feature_attention_mask"].sum(axis=1).astype(mx.int32)
    extra = {
        key: model_inputs[key]
        for key in (
            "pixel_values_videos", "image_grid_thw", "video_grid_thw",
            "input_features", "feature_attention_mask", "audio_feature_lengths",
        )
        if key in model_inputs
    }
    started = time.monotonic()
    generator = generate_step(
        model_inputs["input_ids"], model.thinker, model_inputs.get("pixel_values"), None,
        max_tokens=MAX_TOKENS, temperature=0.0, repetition_penalty=1.05,
        repetition_context_size=64, verbose=False, **extra,
    )
    stop_tokens = {
        value for value in (
            processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            processor.tokenizer.eos_token_id,
        ) if value is not None
    }
    tokens: list[int] = []
    for token, _ in generator:
        token_id = int(token)
        if token_id in stop_tokens or len(tokens) >= MAX_TOKENS:
            break
        tokens.append(token_id)
    mx.eval(mx.zeros(1))
    raw_text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return {
        "raw_text": raw_text,
        "transcript": raw_text,
        "generation_tokens": len(tokens),
        "latency_ms": int((time.monotonic() - started) * 1000),
    }


def concatenate_transcripts(rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Return literal time-marked and overlap-de-duplicated concatenations.

    De-duplication is deliberately conservative. The literal version remains
    the audit record if adjacent ASR chunks phrase the overlap differently.
    """
    literal = "\n\n".join(
        f"[{row['start_s']:.1f}s-{row['end_s']:.1f}s] {row['transcript']}" for row in rows
    )
    stitched: list[str] = []
    for row in rows:
        words = row["transcript"].split()
        if stitched:
            words = _deterministic_continuation(stitched, words)
        stitched.extend(words)
    return literal, " ".join(stitched)


def _normalise_overlap(words: list[str]) -> str:
    return " ".join(re.sub(r"[^\w']", "", word.casefold()) for word in words)


def _deterministic_continuation(previous: list[str], current: list[str]) -> list[str]:
    """Remove a conservatively matched overlap prefix from one transcript chunk."""
    max_overlap = min(60, len(previous), len(current))
    best_prefix_size = 0
    best_span_size = 0
    best_ratio = 0.0
    for suffix_size in range(2, max_overlap + 1):
        left = _normalise_overlap(previous[-suffix_size:])
        for prefix_size in range(max(2, suffix_size - 4), min(max_overlap, suffix_size + 4) + 1):
            right = _normalise_overlap(current[:prefix_size])
            ratio = SequenceMatcher(None, left, right).ratio()
            span_size = min(suffix_size, prefix_size)
            if ratio >= 0.72 and (
                ratio > best_ratio + 0.02
                or (abs(ratio - best_ratio) <= 0.02 and span_size > best_span_size)
            ):
                best_prefix_size = prefix_size
                best_span_size = span_size
                best_ratio = ratio
    return current[best_prefix_size:]


def _clean_audio_output(text: str) -> str:
    """Match the established V4 ASR cleanup before concatenation."""
    cleaned = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return "" if cleaned.upper().startswith("MUSIC") else cleaned


def _model_continuation(
    model: Any,
    processor: Any,
    previous_words: list[str],
    current_transcript: str,
) -> dict[str, Any]:
    """Ask V4 to remove one duplicated boundary without rewriting prior text."""
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    previous_tail = " ".join(previous_words[-80:])
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": V4_STITCH_PROMPT}],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    f"PREVIOUS TAIL:\n{previous_tail}\n\n"
                    f"NEXT CHUNK:\n{current_transcript}"
                ),
            }],
        },
    ]
    prompt = apply_chat_template(
        processor,
        model.config,
        messages,
        num_audios=0,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    started = time.monotonic()
    result = mlx_generate(
        model,
        processor,
        prompt,
        max_tokens=STITCH_MAX_TOKENS,
        temperature=0.0,
        repetition_penalty=1.05,
        repetition_context_size=64,
        verbose=False,
    )
    raw_text = (result.text or "").strip()
    return {
        "raw_text": raw_text,
        "transcript": _clean_audio_output(raw_text),
        "generation_tokens": getattr(result, "generation_tokens", None),
        "latency_ms": int((time.monotonic() - started) * 1000),
    }


def _valid_model_continuation(
    candidate: str,
    current: str,
    deterministic_continuation: list[str],
) -> bool:
    """Reject continuations that truncate, expand, or lose the source chunk ending."""
    candidate_words = candidate.split()
    current_words = current.split()
    if not candidate_words or len(candidate_words) < len(current_words) * 0.45:
        return False
    if len(candidate_words) > len(current_words) * 1.1:
        return False
    overlap_words = len(current_words) - len(deterministic_continuation)
    if overlap_words >= 2 and len(candidate_words) > len(deterministic_continuation) + 2:
        return False
    tail_size = min(12, len(candidate_words), len(current_words))
    candidate_tail = _normalise_overlap(candidate_words[-tail_size:])
    current_tail = _normalise_overlap(current_words[-tail_size:])
    return SequenceMatcher(None, candidate_tail, current_tail).ratio() >= 0.85


def _stitch_pairwise_with_model(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a complete transcript while letting V4 resolve one boundary at a time."""
    if not rows:
        return {"transcript": "", "steps": [], "latency_ms": 0, "accepted_steps": 0}
    stitched = rows[0]["transcript"].split()
    steps: list[dict[str, Any]] = []
    for row in rows[1:]:
        current = row["transcript"]
        result = _model_continuation(model, processor, stitched, current)
        deterministic = _deterministic_continuation(stitched, current.split())
        accepted = _valid_model_continuation(
            result["transcript"], current, deterministic,
        )
        continuation = (
            result["transcript"].split()
            if accepted
            else deterministic
        )
        stitched.extend(continuation)
        steps.append({
            **result,
            "chunk_index": row["index"],
            "accepted": accepted,
            "continuation": " ".join(continuation),
        })
    return {
        "transcript": " ".join(stitched),
        "steps": steps,
        "latency_ms": sum(step["latency_ms"] for step in steps),
        "generation_tokens": sum((step["generation_tokens"] or 0) for step in steps),
        "accepted_steps": sum(step["accepted"] for step in steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to the input audio or video file.")
    parser.add_argument("--label", default="transcription", help="Label used in the output directory name.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", action="store_true", help="Include aligned video frames (default: audio only).")
    parser.add_argument("--max-windows", type=int, default=0, help="Limit windows for a focused smoke test.")
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=OVERLAP_SECONDS)
    parser.add_argument("--no-model-stitch", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.video.exists():
        parser.error(f"video not found: {args.video}")
    if not args.model.exists():
        parser.error(f"model not found: {args.model}")
    duration_s = _probe_duration(args.video)
    windows = audio_windows(
        duration_s, window_s=args.window_seconds, overlap_s=args.overlap_seconds,
    )
    if args.max_windows:
        windows = windows[:args.max_windows]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output / f"bimomni-transcription-{args.label}-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("loading BimOmni from %s", args.model)
    _ensure_audio_patch()
    from mlx_vlm.utils import load as mlx_load
    model, processor = mlx_load(args.model)
    rows: list[dict[str, Any]] = []
    model_stitch: dict[str, Any] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="bimomni-v4-") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            for window in windows:
                chunk_dir = temp_dir / f"chunk-{window.index:02d}"
                chunk_dir.mkdir()
                audio_path = chunk_dir / "audio.mp3"
                _extract_audio_window(video_path, window, audio_path)
                frames = _extract_frames(video_path, window, chunk_dir) if args.frames else ()
                log.info("transcribing %d: %.1fs-%.1fs with %d frames", window.index, window.start_s, window.end_s, len(frames))
                row = _transcribe_window(model, processor, audio_path, frames, window)
                if not args.frames:
                    row["transcript"] = _clean_audio_output(row["raw_text"])
                row.update({
                    "index": window.index,
                    "start_s": window.start_s,
                    "end_s": window.end_s,
                    "frame_timestamps_s": list(frame_timestamps(window)),
                })
                rows.append(row)
                log.info("chunk %d: %d tokens, %dms, %d chars", window.index, row["generation_tokens"], row["latency_ms"], len(row["transcript"]))
        literal, deduplicated = concatenate_transcripts(rows)
        if not args.no_model_stitch:
            log.info("stitching %d transcript chunks pairwise with V4", len(rows))
            model_stitch = _stitch_pairwise_with_model(model, processor, rows)
            log.info(
                "model stitch: %d/%d boundaries accepted, %d tokens, %dms, %d chars",
                model_stitch["accepted_steps"],
                len(model_stitch["steps"]),
                model_stitch["generation_tokens"],
                model_stitch["latency_ms"],
                len(model_stitch["transcript"]),
            )
    finally:
        del model, processor
        mx.clear_cache()
    final_transcript = model_stitch["transcript"] if model_stitch else deduplicated
    stitch_source = "pairwise_model" if model_stitch else "deterministic"
    payload = {
        "video": str(video_path),
        "duration_s": duration_s,
        "window_seconds": args.window_seconds,
        "overlap_seconds": args.overlap_seconds,
        "frames_per_window": FRAMES_PER_WINDOW,
        "frames_enabled": args.frames,
        "frame_height": FRAME_HEIGHT,
        "model": str(args.model),
        "chunks": rows,
        "literal_concatenation": literal,
        "deduplicated_concatenation": deduplicated,
        "model_stitch": model_stitch,
        "final_transcript": final_transcript,
        "stitch_source": stitch_source,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "literal-transcript.txt").write_text(literal + "\n")
    (out_dir / "deduplicated-transcript.txt").write_text(deduplicated + "\n")
    if model_stitch is not None:
        (out_dir / "model-stitched-transcript.txt").write_text(
            model_stitch["transcript"] + "\n"
        )
    (out_dir / "final-transcript.txt").write_text(final_transcript + "\n")
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
