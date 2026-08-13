"""In-process defensive compat for qwen_omni_utils.

The training container solves this with a wholesale text-only shim (because
torch 2.7.1 + torchvision 0.24 + librosa are mutually incompatible). The demo
runs on a ZeroGPU runtime with whatever torchvision/numpy wheels ship
there — we don't control the image — so instead of replacing the package
we patch the small set of functions that, if broken, would surface as noisy
imports or wrong-shape inputs in the audio path.

Called once at app startup, BEFORE transformers resolves the Qwen3-Omni
processor.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

log = logging.getLogger(__name__)


def _identity(*args: Any, **kwargs: Any) -> Any:
    """No-op stand-in for media helpers we never call."""
    if args:
        return args[0]
    if "url" in kwargs:
        return kwargs["url"]
    return None


def _safe_vision_process_stub() -> Any:
    """Return a stand-in module exposing the symbols the shim checks for."""
    stub = sys.modules.get("qwen_omni_utils_vision_process_stub")
    if stub is not None:
        return stub
    stub = type(sys)("qwen_omni_utils_vision_process_stub")
    stub.FPS = 2.0  # mirror the training shim's assertion
    stub.fetch_image = _identity
    stub.fetch_video = _identity
    return stub


def _apply_local_patches() -> None:
    """Patch the real qwen_omni_utils package in place.

    Idempotent. Targets the two import-time failures observed in pulse:
      - `vision_process.FPS` (training image asserts == 2.0)
      - `vision_process.fetch_image` (audio path never invokes it; we want
        any accidental call to degrade to a no-op, not crash)
    """
    try:
        pkg = sys.modules["qwen_omni_utils"]
    except KeyError:
        try:
            import qwen_omni_utils  # noqa: F401
            pkg = sys.modules["qwen_omni_utils"]
        except Exception as exc:
            log.warning("qwen_omni_utils unavailable (%s); audio path relies on AutoProcessor only", exc)
            return

    vp = getattr(pkg, "vision_process", None)
    if vp is None:
        pkg.vision_process = _safe_vision_process_stub()
        log.info("installed qwen_omni_utils.vision_process stub")
    else:
        if not callable(getattr(vp, "fetch_image", None)):
            try:
                vp.fetch_image = _identity
            except Exception:  # pragma: no cover
                pkg.vision_process = _safe_vision_process_stub()


_apply_done = False


def ensure_compat() -> None:
    """Idempotent entry point — call once at app startup, before transformers."""
    global _apply_done
    if _apply_done:
        return
    _apply_local_patches()
    _apply_done = True
