"""BimOmni: a Barbados-adapted Qwen3-Omni-30B-A3B model.

BimOmni is the public training, publishing, evaluation, and inference
package for the Barbados-adapted Qwen3-Omni DAPT work produced during the
Future Caribbean buildathon.

Modules:

- :mod:`bimomni.corpus` — newspaper PDF → training JSONL preparation.
- :mod:`bimomni.training` — recipe, supervisor, budget guard, and
  checkpoint sidecar for Hugging Face Jobs.
- :mod:`bimomni.publish` — LoRA fusion, talker removal, MLX 4-bit
  quantisation, and Hub publication.
- :mod:`bimomni.inference` — MLX loaders, MLX audio compatibility shim,
  and the knowledge-benchmark scorers.
- :mod:`bimomni.evaluation` — the 60-probe Barbados knowledge benchmark
  and the four training-time evaluation gates.
- :mod:`bimomni.transcription` — chunked audio/video transcription with
  model-assisted stitching.
"""

__version__ = "0.1.0"
