"""Narrow Transformers compatibility for Qwen3 Omni GPTQModel loading."""

from typing import Any


def _get_input_embeddings(self: Any) -> Any:
    return self.thinker.get_input_embeddings()


def _set_input_embeddings(self: Any, value: Any) -> None:
    self.thinker.set_input_embeddings(value)


def _get_output_embeddings(self: Any) -> Any:
    return self.thinker.get_output_embeddings()


def _set_output_embeddings(self: Any, value: Any) -> None:
    self.thinker.set_output_embeddings(value)


def install_embedding_accessors(model_class: type[Any]) -> None:
    """Install missing top-level embedding accessors by delegating to Thinker."""
    accessors = {
        "get_input_embeddings": _get_input_embeddings,
        "set_input_embeddings": _set_input_embeddings,
        "get_output_embeddings": _get_output_embeddings,
        "set_output_embeddings": _set_output_embeddings,
    }
    for name, accessor in accessors.items():
        if name not in model_class.__dict__:
            setattr(model_class, name, accessor)


def patch_qwen3_omni_embedding_accessors() -> None:
    """Patch the exact Qwen3 Omni wrapper class used by Transformers 5.14.1."""
    from transformers import Qwen3OmniMoeForConditionalGeneration

    install_embedding_accessors(Qwen3OmniMoeForConditionalGeneration)


def disable_cuda_allocator_warmup() -> None:
    """Skip the CUDA caching-allocator warmup during from_pretrained.

    The warmup calls `torch.cuda.mem_get_info`, a driver-level query that the
    ZeroGPU CUDA emulation does not intercept at module level, so it raises
    "Found no NVIDIA driver". Weight placement itself works under emulation.
    """
    import transformers.modeling_utils as modeling_utils

    modeling_utils.caching_allocator_warmup = lambda *_args, **_kwargs: None
