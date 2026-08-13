"""Evaluation gates for the trained adapter.

Runs four checks before upload is allowed:

1. bf16 smoke: reload base + adapter, disable talker, forward a tiny batch.
2. Perplexity delta: base vs adapter on the held-out eval split; fail if the
   adapter made perplexity worse (positive delta).
3. Generation check: a handful of Barbados prompts, greedy, for eyeball review.
4. CPU 4-bit proxy: bitsandbytes 4-bit load + adapter + small forward.

Each check is a standalone function returning a report dict; the run_* wrappers
stay thin and lazy-import heavy libraries so this module imports cheaply.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DATA_ROOT = os.environ.get("DAPT_DATA_ROOT", "/data/v3")
EVAL_DATASET = f"{DATA_ROOT}/barbados_dapt_eval.jsonl"
ADAPTER_REPO = "hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v3"

BARBADOS_PROMPTS = (
    "The Crop Over festival in Barbados",
    "Kensington Oval was packed for",
    "Prices in BDS$ at the local supermarket",
    "Hurricane season forecast for the Caribbean",
    "The Bridgetown parliamentary constituency",
)


@dataclass(frozen=True)
class EvalReport:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def perplexity_from_loss(loss: float) -> float:
    """Convert a mean negative log-likelihood to perplexity."""
    return math.exp(loss)


def assert_perplexity_improved(base: float, adapter: float, tolerance: float = 0.0) -> EvalReport:
    """Adapter perplexity must not exceed base by more than tolerance."""
    delta = adapter - base
    passed = delta <= tolerance
    return EvalReport(
        name="perplexity_delta",
        passed=passed,
        detail=f"base={base:.4f} adapter={adapter:.4f} delta={delta:+.4f}",
    )


def _load_base(model_id: str, dtype: str = "bf16"):
    """Load the Qwen3-Omni multimodal model with talker disabled.

    Returns (model, processor). Dtype is one of 'bf16' or '4bit'.
    """
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    kwargs: dict = {
        "trust_remote_code": True,
    }
    if dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    elif dtype == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = "auto"
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, **kwargs)
    try:
        model.disable_talker()
    except AttributeError:
        pass
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def _attach_adapter(model, adapter: str):
    """Attach PEFT at the Omni root, then return its adapted thinker."""
    from peft import PeftModel

    adapted_model = PeftModel.from_pretrained(model, adapter).eval()
    return adapted_model.model.thinker


def run_bf16_smoke(model_id: str = MODEL_ID, adapter: str = ADAPTER_REPO) -> EvalReport:
    """Reload base + adapter in bf16, disable talker, forward a tiny batch."""
    import torch

    model, processor = _load_base(model_id, dtype="bf16")
    adapted = _attach_adapter(model, adapter)
    text = "The Barbados Tourism Marketing Inc. announced a new campaign."
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(adapted.device)
    with torch.no_grad():
        outputs = adapted(**inputs, labels=inputs["input_ids"])
    loss = float(outputs.loss)
    ok = math.isfinite(loss) and loss < 12.0
    detail = f"loss={loss:.4f}"
    del adapted, model, processor
    _cuda_empty_cache()
    return EvalReport("bf16_smoke", ok, detail=detail)


def run_perplexity_on_eval(
    model_id: str = MODEL_ID,
    adapter: str = ADAPTER_REPO,
    eval_dataset: str = EVAL_DATASET,
) -> EvalReport:
    """Compare base vs adapter perplexity on the held-out eval split.

    Loads the base model once and swaps PEFT adapters in-place to avoid
    paying the 66 GB base load twice.
    """
    import torch

    model, processor = _load_base(model_id, dtype="bf16")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    records = [
        json.loads(line.strip())
        for line in Path(eval_dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [record["messages"][0]["content"] for record in records]

    def nll(m) -> float:
        total, count = 0.0, 0
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                enc = {k: v.to(m.device) for k, v in enc.items()}
                outputs = m(**enc, labels=enc["input_ids"])
                total += float(outputs.loss)
                count += 1
        return total / max(count, 1)

    base_thinker = model.thinker.eval()
    base_ppl = perplexity_from_loss(nll(base_thinker))

    adapted = _attach_adapter(model, adapter)
    adapter_ppl = perplexity_from_loss(nll(adapted))
    report = assert_perplexity_improved(base_ppl, adapter_ppl)
    del adapted, base_thinker, model, processor
    _cuda_empty_cache()
    return report


def run_generation_check(
    model_id: str = MODEL_ID,
    adapter: str = ADAPTER_REPO,
    prompts: tuple[str, ...] = BARBADOS_PROMPTS,
) -> EvalReport:
    """Greedy generation on Barbados prompts; report outputs for review."""
    import torch

    model, processor = _load_base(model_id, dtype="bf16")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    adapted = _attach_adapter(model, adapter)

    outputs: list[str] = []
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt").to(adapted.device)
            gen = adapted.generate(**enc, max_new_tokens=100, do_sample=False, temperature=1.0)
            outputs.append(tokenizer.decode(gen[0], skip_special_tokens=True))
    del adapted, model, processor
    _cuda_empty_cache()
    return EvalReport("generation", passed=True, detail="\n".join(outputs))


def run_cpu_4bit_proxy(
    model_id: str = MODEL_ID, adapter: str = ADAPTER_REPO
) -> EvalReport:
    """4-bit bitsandbytes load + adapter + small forward (surfaces bad LoRA targets)."""
    import torch
    model, processor = _load_base(model_id, dtype="4bit")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    adapted = _attach_adapter(model, adapter)
    enc = tokenizer("The Crop Over festival in Barbados", return_tensors="pt").to(adapted.device)
    with torch.no_grad():
        outputs = adapted(**enc, labels=enc["input_ids"])
    loss = float(outputs.loss)
    ok = math.isfinite(loss)
    detail = f"loss={loss:.4f}"
    del adapted, model, processor
    _cuda_empty_cache()
    return EvalReport("cpu_4bit_proxy", ok, detail=detail)


def run_all_gates(model_id: str = MODEL_ID, adapter: str = ADAPTER_REPO) -> list[EvalReport]:
    """Run all four gates in order; any failure means upload stays gated."""
    reports = [
        run_bf16_smoke(model_id, adapter),
        run_perplexity_on_eval(model_id, adapter),
        run_generation_check(model_id, adapter),
        run_cpu_4bit_proxy(model_id, adapter),
    ]
    return reports


def _cuda_empty_cache() -> None:
    """Free CUDA memory between gates so the next gate starts fresh."""
    try:
        import torch

        torch.cuda.empty_cache()
        import gc

        gc.collect()
    except Exception:
        pass
