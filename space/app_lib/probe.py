"""Activation probe.

Refuse to run comparisons if the LoRA adapter is silently inactive. The V2
run shipped a demo where the adapter loaded but produced no measurable
effect — this gate turns that failure mode into a startup error rather
than a quiet comparison that lies.

The probe runs lazily on the first `compare()` call inside the @spaces.GPU
task (the ZeroGPU runtime doesn't allow real CUDA work at module import
time), with a module-level `_probe_done` flag to make it a one-shot.
"""
from __future__ import annotations

import logging

import torch

from .config import PROBE_BARBADOS, PROBE_OTHER, PROBE_PROMPT

log = logging.getLogger(__name__)


def _mean_logprob(model, processor, prompt: str, completion: str) -> float:
    """Average per-token log-prob of `completion` conditioned on `prompt`."""
    full_text = prompt + completion
    full_ids = processor.tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    n_prompt = prompt_ids.shape[1]
    if full_ids.shape[1] < n_prompt + 1:
        return float("-inf")
    with torch.no_grad():
        out = model(input_ids=full_ids, return_dict=True)
    # Predict each completion token from the prefix's last position + the
    # already-known previous token.
    logits = out.logits[0, n_prompt - 1 : -1]
    target = full_ids[0, n_prompt:]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    chosen = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(chosen.mean().item())


def probe(model, processor, *, atol: float = 1e-3) -> None:
    """Raise if the adapter does not shift the Barbados-completion logprob.

    Also checks that Iceland is NOT shifted as much — a moving Iceland logprob
    would suggest the adapter is misconfigured (e.g. attached to the wrong
    layer) rather than legitimately Barbados-aware.
    """
    with torch.no_grad():
        with model.disable_adapter():
            off_barbados = _mean_logprob(model, processor, PROBE_PROMPT, PROBE_BARBADOS)
            off_iceland = _mean_logprob(model, processor, PROBE_PROMPT, PROBE_OTHER)
        on_barbados = _mean_logprob(model, processor, PROBE_PROMPT, PROBE_BARBADOS)
        on_iceland = _mean_logprob(model, processor, PROBE_PROMPT, PROBE_OTHER)

    delta_barbados = on_barbados - off_barbados
    delta_iceland = on_iceland - off_iceland
    log.info(
        "[probe] Barbados logprob Δ = %+.4f  Iceland logprob Δ = %+.4f",
        delta_barbados,
        delta_iceland,
    )

    if abs(delta_barbados) < atol:
        raise RuntimeError(
            f"LoRA adapter appears inactive: "
            f"Δlogprob(Barbados) = {delta_barbados:+.6f} is below atol={atol}. "
            f"Verify peft==0.18.1 and that the adapter downloaded fully. "
            f"Refusing to serve comparisons from a silent base."
        )

    if abs(delta_iceland) > 2 * abs(delta_barbados) + 1e-3:
        log.warning(
            "[probe] Iceland logprob shifted more than expected "
            "(|ΔIceland|=%.4f vs |ΔBarbados|=%.4f). Adapter may be on the wrong layer.",
            abs(delta_iceland),
            abs(delta_barbados),
        )


_probe_done = False


def ensure_probed(model, processor) -> None:
    """Run the probe once, lazily on first use."""
    global _probe_done
    if _probe_done:
        return
    probe(model, processor)
    _probe_done = True
