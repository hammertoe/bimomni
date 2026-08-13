"""Text-only shim for qwen_omni_utils.

The upstream package imports torchvision, audioread, av and librosa at module
load. torchvision 0.24.x hits an import-ordering bug against torch 2.7.1
("operator torchvision::nms does not exist"), and librosa pulls in numba which
pins numpy to an ABI that breaks torch 2.7.1 / flash-attn. None of this is
needed for Barbados DAPT: the ms-swift Qwen3-Omni loader only imports
`vision_process` to build the processor, and text-only data never touches the
media code paths.

The real qwen_omni_utils==0.0.9 distribution stays installed (its dist-info is
kept) so ms-swift's require_version('qwen_omni_utils>=0.0.9') check passes.
"""

from . import vision_process
from .vision_process import fetch_image, fetch_video, process_mm_info, process_vision_info

__all__ = [
    "vision_process",
    "fetch_image",
    "fetch_video",
    "process_mm_info",
    "process_vision_info",
]
