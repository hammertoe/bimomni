"""Configuration for the BimOmni ZeroGPU Space.

All model identifiers are pinned to the exact revisions used to publish V4
so the Space reproduces the published artefact without drift.
"""
from __future__ import annotations

from pathlib import Path

# Model identifiers.
BASE_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
BASE_MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
ADAPTER_ID = "hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4"

WHISPER_MODEL_ID = "openai/whisper-large-v3"

# Audio constraints. Qwen3-Omni's audio tower accepts arbitrary lengths but
# 30 s windows are what the published eval used and keep VRAM predictable.
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 29

# Bundled clips directory (operator-curated). The launcher filters to files
# that actually exist so a partial bundle does not break the dropdown.
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

# Generation parameters for Qwen3-Omni.
QWEN_MAX_NEW_TOKENS = 256
QWEN_DO_SAMPLE = False

# Probe constants. Differentiating signal: the adapter shifts the Barbados
# token's logprob far more than the Iceland token's — both are checkable.
PROBE_PROMPT = "The Crop Over festival is held in"
PROBE_BARBADOS = " Barbados"
PROBE_OTHER = " Iceland"
