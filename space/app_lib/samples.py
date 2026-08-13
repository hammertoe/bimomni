"""Bundled audio clips registry.

The Space ships with a `space/samples/` directory checked into the repo
(operator-curated). `SAMPLE_CLIPS` exposes only files that actually exist
so a partial bundle never breaks the launcher or the Gradio dropdown.
"""
from __future__ import annotations

from .config import SAMPLES_DIR

# Operator-curated. Add new entries here; the launcher filters by file
# existence so deleting one file is a valid way to drop a clip.
CANDIDATES: dict[str, str] = {
    "Itinerary (TikTok)": "itinerary_29s.wav",
    "Harbour Lights (TikTok)": "harbour_lights_29s.wav",
    "HOTT 95.3 broadcast": "radio_hott_29s.wav",
    "VOB 92.9 broadcast": "radio_vob_29s.wav",
    "CBC news extract": "radio_cbc_news_29s.wav",
}


def sample_clips() -> dict[str, str]:
    """Label → absolute path, restricted to clips that are actually on disk."""
    return {
        label: str(SAMPLES_DIR / name)
        for label, name in CANDIDATES.items()
        if (SAMPLES_DIR / name).is_file()
    }


SAMPLE_CLIPS: dict[str, str] = sample_clips()
