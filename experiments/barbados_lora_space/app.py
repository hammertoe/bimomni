"""BimOmni Barbados fine-tune demo on ZeroGPU.

Three-way side-by-side comparison in a single @spaces.GPU task:
  1. Whisper-large-v3          (ASR baseline)
  2. Qwen3-Omni base           (Barbados LoRA adapter disabled)
  3. BimOmni fine-tune         (same weights, adapter enabled)

One bf16 Qwen3-Omni checkpoint; PEFT injects the Barbados LoRA adapter
in-place and each transcription toggles it via `disable_adapter()`.
No quantization anywhere.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext

import gradio as gr
import numpy as np
import soundfile as sf
import spaces
import torch
from peft import PeftModel
from qwen_compat import disable_cuda_allocator_warmup, patch_qwen3_omni_embedding_accessors
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration, pipeline

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
ADAPTER_ID = "hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4"
WHISPER_MODEL_ID = "openai/whisper-large-v3"

TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 29
MAX_NEW_TOKENS = 256

PROBES = [
    ("The Crop Over festival is held in", " Barbados", " Iceland"),
    ("Start your trip on the south coast at", " Worthing Square", " Weddings Square"),
    ("Grab lunch or a snack from", " Chefette", " Chez Fete"),
]

TRANSCRIPT_PROMPT = (
    "Transcribe every intelligible spoken word verbatim and in order. "
    "Preserve exact wording, repetitions, and Bajan dialect. Correct only "
    "obvious proper-noun spelling. Do not summarize, classify, explain, or "
    "describe the audio. Do not transcribe background song lyrics. Return only "
    "the transcript."
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bimomni-barbados")

patch_qwen3_omni_embedding_accessors()
disable_cuda_allocator_warmup()

PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

BASE_MODEL = None
PEFT_MODEL = None
WHISPER_PIPE = None


def _ensure_models() -> None:
    """Load base + adapter + Whisper inside the GPU allocation (real GPU)."""
    global BASE_MODEL, PEFT_MODEL, WHISPER_PIPE
    if BASE_MODEL is None:
        log.info("loading %s @ %s", MODEL_ID, MODEL_REVISION[:8])
        BASE_MODEL = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            enable_audio_output=False,
        )
    if PEFT_MODEL is None:
        log.info("injecting LoRA adapter: %s", ADAPTER_ID)
        PEFT_MODEL = PeftModel.from_pretrained(BASE_MODEL, ADAPTER_ID)
        PEFT_MODEL.eval()
    if WHISPER_PIPE is None:
        log.info("loading Whisper pipeline: %s", WHISPER_MODEL_ID)
        WHISPER_PIPE = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )


def load_audio(audio_path: str) -> np.ndarray:
    """Load one bounded clip as mono float32 audio at 16 kHz."""
    info = sf.info(audio_path)
    if info.duration > MAX_AUDIO_SECONDS + 0.5:
        raise ValueError(
            f"Audio is {info.duration:.1f}s; the demo accepts at most {MAX_AUDIO_SECONDS}s."
        )
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        old_positions = np.arange(len(audio), dtype=np.float64) / sample_rate
        new_length = round(len(audio) * TARGET_SAMPLE_RATE / sample_rate)
        new_positions = np.arange(new_length, dtype=np.float64) / TARGET_SAMPLE_RATE
        audio = np.interp(new_positions, old_positions, audio).astype(np.float32)
    return audio


def _adapter_context(adapter_enabled: bool):
    """Context manager selecting fine-tune (on) or base (off) behaviour."""
    if adapter_enabled:
        return nullcontext()
    return PEFT_MODEL.disable_adapter()


_MERGED = False


def _set_merged(merged: bool) -> None:
    """Merge or unmerge the LoRA delta into the base weights.

    The unmerged runtime path materialises the all-experts delta on every
    sparse expert access (~12x slower), so generation always runs either on
    merged weights (fine-tune) or with the adapter disabled (base).
    """
    global _MERGED
    if merged == _MERGED:
        return
    started = time.monotonic()
    if merged:
        PEFT_MODEL.merge_adapter()
    else:
        PEFT_MODEL.unmerge_adapter()
    _MERGED = merged
    log.info("%s adapter in %.1fs", "merged" if merged else "unmerged", time.monotonic() - started)


def _build_inputs(audio: np.ndarray, device: torch.device, dtype: torch.dtype) -> dict:
    """Build processor inputs for one audio clip on `device`."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": TRANSCRIPT_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Transcribe this audio verbatim."},
            ],
        },
    ]
    inputs = PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        name: value.to(device=device, dtype=dtype)
        if torch.is_floating_point(value)
        else value.to(device)
        for name, value in inputs.items()
        if hasattr(value, "to")
    }


def _decode_transcript(output: torch.Tensor, input_ids: torch.Tensor) -> str:
    token_ids = output[0] if isinstance(output, tuple) else output
    trimmed = [ids[len(input_ids) :] for ids, input_ids in zip(token_ids, input_ids, strict=False)]
    return PROCESSOR.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _transcribe(audio: np.ndarray, adapter_enabled: bool) -> tuple[str, float]:
    """Transcribe once with the LoRA adapter merged (FT) or disabled (base)."""
    _set_merged(adapter_enabled)
    inputs = _build_inputs(audio, BASE_MODEL.device, BASE_MODEL.dtype)
    started = time.monotonic()
    with torch.inference_mode(), _adapter_context(adapter_enabled):
        output = BASE_MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
        )
    elapsed = time.monotonic() - started
    return _decode_transcript(output, inputs["input_ids"]), elapsed


def _transcribe_whisper(audio: np.ndarray) -> tuple[str, float]:
    """Transcribe with the Whisper-large-v3 ASR baseline."""
    started = time.monotonic()
    result = WHISPER_PIPE(
        audio,
        return_timestamps=False,
        generate_kwargs={"language": "en"},
    )
    elapsed = time.monotonic() - started
    if isinstance(result, dict):
        return str(result.get("text", "")).strip(), elapsed
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return str(result[0].get("text", "")).strip(), elapsed
    return str(result).strip(), elapsed


def _mean_logprob(prompt: str, completion: str, adapter_enabled: bool) -> float:
    """Mean token log-probability of `completion` given `prompt`."""
    _set_merged(adapter_enabled)
    tokenizer = PROCESSOR.tokenizer
    full_ids = tokenizer(prompt + completion, return_tensors="pt").input_ids
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    n_prompt = prompt_ids.shape[1]
    if full_ids.shape[1] < n_prompt + 1:
        return float("-inf")
    full_ids = full_ids.to(BASE_MODEL.device)
    with torch.inference_mode(), _adapter_context(adapter_enabled):
        out = BASE_MODEL.thinker(input_ids=full_ids, return_dict=True)
    logits = out.logits[0, n_prompt - 1 : -1].float()
    target = full_ids[0, n_prompt:]
    logprobs = torch.log_softmax(logits, dim=-1)
    chosen = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(chosen.mean().item())


@spaces.GPU(duration=900, size="xlarge")
def run_probe() -> str:
    _ensure_models()
    base_scores = {
        prompt: (
            _mean_logprob(prompt, correct, adapter_enabled=False),
            _mean_logprob(prompt, wrong, adapter_enabled=False),
        )
        for prompt, correct, wrong in PROBES
    }
    ft_scores = {
        prompt: (
            _mean_logprob(prompt, correct, adapter_enabled=True),
            _mean_logprob(prompt, wrong, adapter_enabled=True),
        )
        for prompt, correct, wrong in PROBES
    }
    _set_merged(False)
    lines = [
        "| Prompt | Correct | base pref | FT pref | FT−base |",
        "|---|---:|---:|---:|---:|",
    ]
    for prompt, correct, wrong in PROBES:
        base_c, base_w = base_scores[prompt]
        ft_c, ft_w = ft_scores[prompt]
        base_pref = base_c - base_w
        ft_pref = ft_c - ft_w
        delta = ft_c - base_c
        short = prompt.strip()[-35:]
        if len(prompt) > 35:
            short = "…" + short
        lines.append(
            f"| {short} | `{correct.strip()}` | {base_pref:+.3f} | {ft_pref:+.3f} | {delta:+.3f} |"
        )
    lines.append("")
    lines.append("_pref = logprob(correct) − logprob(wrong). Positive = prefers correct._")
    lines.append("_Same weights throughout; the adapter is toggled off (base) and on (FT)._")
    lines.append("_FT−base = how much the fine-tune shifts preference for the correct token._")
    return "\n".join(lines)


@spaces.GPU(duration=900, size="xlarge")
def compare(audio_path: str) -> tuple[str, str, str, str, str, str, str]:
    """Run all three transcriptions on `audio_path`, yielding partial UI state."""
    if not audio_path:
        yield "", "", "", "Upload an audio clip first.", "—", "—", "—"
        return

    try:
        audio = load_audio(audio_path)
    except (OSError, RuntimeError, ValueError) as error:
        yield "", "", "", f"Audio error: {type(error).__name__}: {error}", "—", "—", "—"
        return

    try:
        _ensure_models()
    except Exception as error:
        log.exception("Model loading failed")
        yield "", "", "", f"Model loading failed: {type(error).__name__}: {error}", "—", "—", "—"
        return

    try:
        whisper_text, t_whisper = _transcribe_whisper(audio)
    except Exception as error:
        log.exception("Whisper transcription failed")
        yield "", "", "", f"Whisper failed: {type(error).__name__}: {error}", "—", "—", "—"
        return
    yield whisper_text, "", "", "Transcribing with Qwen3-Omni (base)…", f"{t_whisper:.1f}s", "running…", "—"

    try:
        base_text, t_base = _transcribe(audio, adapter_enabled=False)
    except Exception as error:
        log.exception("Base Qwen transcription failed")
        yield whisper_text, "", "", f"Qwen base failed: {type(error).__name__}: {error}", f"{t_whisper:.1f}s", "—", "—"
        return
    yield whisper_text, base_text, "", "Transcribing with BimOmni fine-tune…", f"{t_whisper:.1f}s", f"{t_base:.1f}s", "running…"

    try:
        ft_text, t_ft = _transcribe(audio, adapter_enabled=True)
    except Exception as error:
        log.exception("Fine-tune transcription failed")
        yield (
            whisper_text,
            base_text,
            "",
            f"BimOmni fine-tune failed: {type(error).__name__}: {error}",
            f"{t_whisper:.1f}s",
            f"{t_base:.1f}s",
            "—",
        )
        return

    yield (
        whisper_text,
        base_text,
        ft_text,
        "✓ Done — look for proper nouns on the right column.",
        f"{t_whisper:.1f}s",
        f"{t_base:.1f}s",
        f"{t_ft:.1f}s",
    )


with gr.Blocks(title="BimOmni Barbados Demo") as demo:
    gr.Markdown(
        """
        # BimOmni Barbados fine-tune — audio transcription

        Upload up to **29 seconds** of audio. Three models transcribe the same
        clip in a single GPU run:

        | Column | Model | Notes |
        |---|---|---|
        | Whisper-large-v3 | `openai/whisper-large-v3` | ASR baseline |
        | Qwen3-Omni (base) | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (pinned rev) | Barbados LoRA **disabled** |
        | BimOmni fine-tune | same weights | Barbados LoRA v4 **enabled** |

        The base and fine-tune columns are literally the same bf16 weights —
        the LoRA adapter is toggled off and on — so any difference is purely
        the fine-tune. Differences usually surface on **proper nouns**: place
        names, festivals, institutions.
        """
    )
    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
    go = gr.Button("Compare", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Whisper-large-v3")
            out_whisper = gr.Textbox(label="Transcript", lines=10, interactive=False)
            lat_whisper = gr.Markdown("—")
        with gr.Column():
            gr.Markdown("### Qwen3-Omni (base)")
            out_base = gr.Textbox(label="Transcript", lines=10, interactive=False)
            lat_base = gr.Markdown("—")
        with gr.Column():
            gr.Markdown("### BimOmni fine-tune")
            out_ft = gr.Textbox(label="Transcript", lines=10, interactive=False)
            lat_ft = gr.Markdown("—")

    status = gr.Markdown("Ready.")

    go.click(
        fn=compare,
        inputs=audio,
        outputs=[
            out_whisper,
            out_base,
            out_ft,
            status,
            lat_whisper,
            lat_base,
            lat_ft,
        ],
    )

    probe_btn = gr.Button("Run adaptation probe", variant="secondary")
    probe_out = gr.Markdown(
        "Click **Run** to measure how the LoRA fine-tune shifts token "
        "preferences for Barbados proper nouns (adapter off vs on, same weights)."
    )
    probe_btn.click(fn=run_probe, outputs=[probe_out], api_name="probe")


if __name__ == "__main__":
    demo.queue(max_size=4, default_concurrency_limit=1).launch()
