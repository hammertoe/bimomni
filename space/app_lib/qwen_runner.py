"""Qwen3-Omni inference with PEFT LoRA toggle.

The Space holds a single Qwen3-Omni base model in memory and toggles the
Barbados LoRA on/off between the base and adapted runs via PEFT's
`disable_adapter()` context manager. That keeps the daily GPU budget
honest: one model load, two transcriptions, one set of weights.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

_model: Any = None
_processor: Any = None
_loaded_signature: tuple[str, str, str] | None = None


def _disable_allocator_warmup() -> None:
    """Disable a transformers loading optimization unsupported by ZeroGPU.

    transformers 5.x pre-allocates the model's full quantized footprint with
    one large ``torch.empty`` call. ZeroGPU's virtual CUDA/NVML allocator
    crashes on that call before any weights load. The warm-up only improves
    load speed; normal shard-by-shard allocation remains correct without it.
    """
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils.caching_allocator_warmup, "_bimomni_disabled", False):
        return

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    _noop._bimomni_disabled = True  # type: ignore[attr-defined]
    modeling_utils.caching_allocator_warmup = _noop
    log.info("disabled transformers CUDA allocator warm-up for ZeroGPU")


def _bnb_config() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_qwen(
    base_id: str, base_revision: str, adapter_id: str
) -> tuple[Any, Any]:
    """Idempotent load of Qwen3-Omni base + PEFT LoRA adapter.

    Must be called inside an active @spaces.GPU task — `device_map="cuda"`
    requires a live CUDA context.
    """
    global _model, _processor, _loaded_signature
    sig = (base_id, base_revision, adapter_id)
    if _model is not None and _loaded_signature == sig:
        return _model, _processor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen3-Omni requires CUDA; call load_qwen from inside @spaces.GPU."
        )

    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        Qwen3OmniMoeForConditionalGeneration,
    )

    _disable_allocator_warmup()
    log.info("loading Qwen3-Omni base: %s @ %s…", base_id, base_revision[:8])
    base = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        base_id,
        revision=base_revision,
        quantization_config=_bnb_config(),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        enable_audio_output=False,
    )
    log.info("attaching PEFT adapter: %s", adapter_id)
    model = PeftModel.from_pretrained(base, adapter_id, is_trainable=False)
    log.info("loading processor")
    processor = AutoProcessor.from_pretrained(base_id, revision=base_revision)

    _model, _processor, _loaded_signature = model, processor, sig
    return _model, _processor


def transcribe(
    model: Any,
    processor: Any,
    audio_array: np.ndarray,
    sample_rate: int,
    *,
    apply_adapter: bool,
    max_new_tokens: int = 256,
) -> str:
    """Single Qwen3-Omni transcription pass.

    `apply_adapter=False` wraps the call in `model.disable_adapter()` so the
    output reflects the base model only; `apply_adapter=True` (default for
    a PeftModel) leaves the LoRA active.
    """
    del sample_rate  # AutoProcessor takes the raw array; resampling is upstream.

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": "Transcribe this audio verbatim."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

    cm = model.disable_adapter() if not apply_adapter else contextlib.nullcontext()
    with cm:
        text_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [ids[len(in_ids):] for ids, in_ids in zip(text_ids, inputs["input_ids"], strict=False)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return text.strip()
