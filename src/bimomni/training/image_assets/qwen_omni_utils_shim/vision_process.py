"""Import-safe stand-in for qwen_omni_utils.vision_process.

Exposes the constants ms-swift's patch_qwen_vl_utils() reads and sets, and
fail-fast media functions. Barbados DAPT is text-only, so the media functions
are never called during training or evaluation.
"""

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
MODEL_SEQ_LEN = 128000


def _not_supported(name):
    raise NotImplementedError(
        f"qwen_omni_utils.{name}: media input is not supported in the text-only Barbados DAPT image"
    )


def fetch_image(*args, **kwargs):
    _not_supported("fetch_image")


def fetch_video(*args, **kwargs):
    _not_supported("fetch_video")


def process_vision_info(*args, **kwargs):
    _not_supported("process_vision_info")


def process_mm_info(*args, **kwargs):
    _not_supported("process_mm_info")
