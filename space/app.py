"""BimOmni ZeroGPU Space — three-way audio transcription comparison.

Workflow (single @spaces.GPU task per Compare click):
  1. Whisper-large-v3  (ASR baseline, cached after first load)
  2. Qwen3-Omni base   (multimodal Thinker, no Barbados data)
  3. BimOmni           (same Qwen with the V4 LoRA active)

Single model in memory; the LoRA is toggled on/off via PEFT's
`disable_adapter()` context manager so the daily 5-min GPU budget counts
honestly.

Activation probe runs on the first `compare()` invocation: if the adapter
doesn't shift the Barbados-completion logprob by more than epsilon, the
demo refuses to serve the comparison (V2 silent-inactive-adapter bug,
rendered as a hard gate).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Make `import app_lib` resolve to this Space's package, not src/bimomni.
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ENABLE_AUDIO_OUTPUT", "0")

import gradio as gr
import spaces
from app_lib._qwen_omni_compat import ensure_compat as ensure_qwen_omni_compat
from app_lib.audio import load_audio_array, validate_duration
from app_lib.config import (
    ADAPTER_ID,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    WHISPER_MODEL_ID,
)
from app_lib.probe import ensure_probed
from app_lib.qwen_runner import load_qwen
from app_lib.qwen_runner import transcribe as qwen_transcribe
from app_lib.samples import SAMPLE_CLIPS
from app_lib.whisper_runner import load_whisper
from app_lib.whisper_runner import transcribe as whisper_transcribe
from rich.console import Console
from rich.logging import RichHandler

# Run the defensive qwen_omni_utils patch BEFORE transformers touches it.
ensure_qwen_omni_compat()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=False)],
)
log = logging.getLogger("bimomni")
console = Console()


# Module-level sentinels — loaded lazily on the first compare() call inside
# the @spaces.GPU decorator. ZeroGPU doesn't allow real CUDA at import time.
WHISPER_PIPE = None
QWEN_MODEL = None
QWEN_PROCESSOR = None


def _load_sample(label: str | None) -> str:
    return SAMPLE_CLIPS.get(label or "", "")


def _ensure_whisper() -> None:
    global WHISPER_PIPE
    if WHISPER_PIPE is None:
        log.info("loading Whisper-large-v3…")
        WHISPER_PIPE = load_whisper(WHISPER_MODEL_ID)


def _ensure_qwen() -> tuple:
    global QWEN_MODEL, QWEN_PROCESSOR
    if QWEN_MODEL is None:
        log.info("loading Qwen3-Omni base + Barbados LoRA (cold start: minutes)")
        QWEN_MODEL, QWEN_PROCESSOR = load_qwen(
            BASE_MODEL_ID, BASE_MODEL_REVISION, ADAPTER_ID
        )
        ensure_probed(QWEN_MODEL, QWEN_PROCESSOR)
        log.info("[probe] adapter is active — demo ready")
    return QWEN_MODEL, QWEN_PROCESSOR


@spaces.GPU(duration=180)
def compare(
    audio_path: str,
) -> tuple[str, str, str, str, str, str, str]:
    """Run all three transcriptions on `audio_path`. Yields partial UI state."""
    if not audio_path:
        yield "", "", "", "Upload a clip or pick a sample.", "—", "—", "—"
        return

    try:
        duration = validate_duration(audio_path)
    except ValueError as exc:
        yield "", "", "", f"⚠ {exc}", "—", "—", "—"
        return

    log.info("compare(audio=%s, duration=%.1fs)", audio_path, duration)

    try:
        _ensure_whisper()
    except Exception as exc:
        log.exception("Whisper load failed")
        yield "", "", "", f"Whisper load failed: {type(exc).__name__}: {exc}", "error", "—", "—"
        return
    yield "", "", "", "Transcribing with Whisper-large-v3…", "running…", "—", "—"

    try:
        t0 = time.monotonic()
        whisper_text = whisper_transcribe(WHISPER_PIPE, audio_path)
        t_whisper = time.monotonic() - t0
    except Exception as exc:
        log.exception("Whisper transcription failed")
        yield "", "", "", f"Whisper failed: {type(exc).__name__}: {exc}", "error", "—", "—"
        return

    yield whisper_text, "", "", "Transcribing with Qwen3-Omni (base)…", f"{t_whisper:.1f}s", "running…", "—"

    try:
        model, processor = _ensure_qwen()
    except Exception as exc:
        log.exception("Qwen load or activation probe failed")
        yield (
            whisper_text,
            "",
            "",
            f"Qwen load/probe failed: {type(exc).__name__}: {exc}",
            f"{t_whisper:.1f}s",
            "error",
            "—",
        )
        return
    audio_array, sample_rate = load_audio_array(audio_path)

    try:
        t0 = time.monotonic()
        base_text = qwen_transcribe(
            model, processor, audio_array, sample_rate, apply_adapter=False
        )
        t_base = time.monotonic() - t0
    except Exception as exc:
        log.exception("Qwen base transcription failed")
        yield (
            whisper_text,
            "",
            "",
            f"Qwen base failed: {type(exc).__name__}: {exc}",
            f"{t_whisper:.1f}s",
            "error",
            "—",
        )
        return

    yield (
        whisper_text,
        base_text,
        "",
        "Transcribing with BimOmni (adapter on)…",
        f"{t_whisper:.1f}s",
        f"{t_base:.1f}s",
        "running…",
    )

    try:
        t0 = time.monotonic()
        bimomni_text = qwen_transcribe(
            model, processor, audio_array, sample_rate, apply_adapter=True
        )
        t_bimomni = time.monotonic() - t0
    except Exception as exc:
        log.exception("BimOmni transcription failed")
        yield (
            whisper_text,
            base_text,
            "",
            f"BimOmni failed: {type(exc).__name__}: {exc}",
            f"{t_whisper:.1f}s",
            f"{t_base:.1f}s",
            "error",
        )
        return

    yield (
        whisper_text,
        base_text,
        bimomni_text,
        "✓ Done — look for proper nouns on the right column.",
        f"{t_whisper:.1f}s",
        f"{t_base:.1f}s",
        f"{t_bimomni:.1f}s",
    )


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    title="BimOmni Audio Demo",
) as demo:
    gr.Markdown(
        """
        # BimOmni — Caribbean-aware audio transcription

        Upload a clip of up to **29 seconds** (or pick a sample). Three models
        transcribe it in a single GPU run:

        | Column | Model | Notes |
        |---|---|---|
        | Whisper-large-v3 | `openai/whisper-large-v3` | ASR baseline |
        | Qwen3-Omni (base) | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (pinned rev) | multimodal Thinker, no Barbados data |
        | BimOmni | base + `hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4` | same Qwen with the V4 LoRA active |

        Differences usually surface on **proper nouns** — place names,
        festivals, institutions — where BimOmni should match the audio and
        the base may scramble the spelling.

        Free tier: ~5 minutes of GPU time per day. First compare() pays a
        one-time cold start (~minutes) to download the 63 GB base model.
        """
    )

    sample_labels = list(SAMPLE_CLIPS.keys())

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio (≤29 s)",
            )
            sample_dropdown = gr.Dropdown(
                choices=sample_labels,
                label="Sample clips (when bundled)",
                value=sample_labels[0] if sample_labels else None,
                interactive=bool(sample_labels),
            )
            go = gr.Button("Compare", variant="primary")

        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Whisper-large-v3")
                    out_whisper = gr.Textbox(label="Transcript", lines=8, interactive=False)
                    lat_whisper = gr.Markdown("—")

                with gr.Column():
                    gr.Markdown("### Qwen3-Omni (base)")
                    out_base = gr.Textbox(label="Transcript", lines=8, interactive=False)
                    lat_base = gr.Markdown("—")

                with gr.Column():
                    gr.Markdown("### BimOmni")
                    out_bimomni = gr.Textbox(label="Transcript", lines=8, interactive=False)
                    lat_bimomni = gr.Markdown("—")

            status = gr.Markdown("Ready.")

    sample_dropdown.change(fn=_load_sample, inputs=sample_dropdown, outputs=audio)
    go.click(
        fn=compare,
        inputs=audio,
        outputs=[
            out_whisper,
            out_base,
            out_bimomni,
            status,
            lat_whisper,
            lat_base,
            lat_bimomni,
        ],
    )

    gr.Markdown(
        """
        ---
        BimOmni is a Barbados-adapted DAPT pass over the
        [*Barbados Advocate*](https://en.wikipedia.org/wiki/Daily_Advocate_(Barbados))
        2013-2023. 75.0% on the 60-probe Barbados knowledge set.

        Built on the [`bimomni`](https://github.com/hammertoe/bimomni) repo.
        """
    )


if __name__ == "__main__":
    demo.queue(max_size=4).launch()
