# BimOmni

> BimOmni hears Barbados properly.

**BimOmni** is a Barbados-adapted version of
[Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).
It was developed as part of
[Future Caribbean](https://futurecaribbean.com/) to test a simple idea: when
audio is ambiguous, can stronger knowledge of Barbados help a multimodal
model choose the correct local name, place, institution, event, or phrase?

The project contains:

- a text-only domain-adaptive pretraining (DAPT) recipe for the
  Qwen3-Omni Thinker;
- a Hugging Face Jobs supervisor with checkpoint sidecar, budget guard,
  recipe identity, and resume-from-bucket;
- LoRA fusion, talker removal, and 4-bit MLX quantisation on Apple
  Silicon;
- a 60-probe Barbados knowledge evaluation set with a paired MLX scorer;
- a chunked audio-transcription pipeline for local BimOmni inference.

The published weights live on the Hugging Face Hub under the **BimOmni**
family:

| Model | What it is |
| --- | --- |
| [hammertoe/BimOmni-30B-A3B](https://huggingface.co/hammertoe/BimOmni-30B-A3B) | fused bf16 (LoRA merged, talker dropped) |
| [hammertoe/BimOmni-30B-A3B-MLX-4bit](https://huggingface.co/hammertoe/BimOmni-30B-A3B-MLX-4bit) | 4-bit MLX snapshot for Apple Silicon |

A standard text-only LoRA is published as the provenance entry-point:

- [hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4](https://huggingface.co/hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4)

## Project background

BimOmni was developed as part of
[Future Caribbean](https://futurecaribbean.com/), a regional initiative
connecting technology, talent, and opportunity across the Caribbean
through a global agentic AI buildathon. The model is one component of
**Pulse**, a public-signal intelligence system for Barbados.

The model is not Pulse-specific. It can be used independently for
Caribbean-domain transcription, information extraction, search,
summarisation, and other multimodal applications. You do not need to be
running Pulse to benefit from the training.

## How to use this repository

```text
.
├── src/
│   └── bimomni/
│       ├── corpus/         # newspaper PDF → training JSONL
│       ├── training/       # recipe, ms-swift driver, supervisor, sidecar
│       ├── publish/        # adapter fuse, talker strip, MLX 4-bit, upload
│       ├── inference/      # MLX loaders + audio compat shim
│       ├── evaluation/     # knowledge benchmark + run-all-gates
│       └── transcription/  # chunked audio/video transcription
├── containers/             # pinned training + finalise Dockerfiles
├── benchmarks/             # probe set + canonical V4 result
├── scripts/                # operator-facing run scripts
├── docs/                   # design, executed runs, blog articles
└── tests/                  # unit tests for every public module
```

The `docs/` directory is the recommended starting point for new readers:

- `docs/training-recipe.md` — the locked training decision matrix.
- `docs/runs/v2-prime-intellect.md` — the executed V2 run on a rented H200.
- `docs/runs/v3-v4-hugging-face-jobs.md` — V3 and V4 on Hugging Face Jobs,
  including the resume-from-bucket flow.
- `docs/inference.md` — running BimOmni locally, library versions, and
  the activation check.
- `docs/articles/` — the published blog posts that walk through the
  motivation, recipe evolution, and benchmark design.

## Quick start

Install the project locally:

```bash
uv sync --extra mlx        # add the Apple Silicon MLX stack
uv sync --extra train      # add the GPU training stack
uv sync --extra dev        # add tests and lint
uv sync --extra dev --extra test   # run the unit suite locally
```

### Local 4-bit inference (Apple Silicon)

Download a snapshot:

```bash
hf download hammertoe/BimOmni-30B-A3B-MLX-4bit --local-dir model/bimomni-4bit
```

Run a one-shot generation through the MLX-LM CLI:

```bash
mlx_lm.generate \
    --model model/bimomni-4bit \
    --prompt "The Crop Over festival in Barbados" \
    --max-tokens 100
```

For audio, images, or the knowledge benchmark, use the `bimomni.inference`
package. See `docs/inference.md` for a full walkthrough.

### Chunked local audio transcription

Transcribe a video or audio file with overlapping 30-second windows and
model-assisted stitching:

```bash
uv run python -m bimomni.transcription.chunked \
    --video path/to/video.mp4 \
    --label my-clip \
    --model model/bimomni-4bit \
    --output transcripts/
```

Each window is decoded as audio only by default. Pass `--frames` to feed
up to three video frames per window; the prompt keeps the output as an
audio transcript.

### Run the knowledge benchmark

Score a local snapshot on the curated 60-probe set:

```bash
uv run --extra mlx python -m bimomni.inference.mlx_fused \
    --model-path model/bimomni-4bit \
    --probes-path benchmarks/knowledge/probes/barbados-knowledge-v1.jsonl \
    --output-dir benchmarks/knowledge/results
```

The current canonical result for the published `BimOmni-30B-A3B-MLX-4bit`
artefact is committed at
[`benchmarks/knowledge/results-v4.json`](benchmarks/knowledge/results-v4.json)
and [`benchmarks/knowledge/results-v4.md`](benchmarks/knowledge/results-v4.md).

## Container pipeline

The training and publishing pipeline runs end-to-end on Hugging Face
Jobs. The supervisor is `bimomni.training.supervisor`, and the immutable
images are defined in `containers/Dockerfile.train` and
`containers/Dockerfile.finalise`.

Stages:

| Stage | Flavor | Purpose |
| --- | --- | --- |
| `doctor` | h200 | verify CUDA + HF credentials |
| `smoke` | h200 | two-step training into a scratch dir |
| `train` | h200 | full DAPT under a wall-clock budget guard |
| `sync-once` | h200 | one checkpoint-sync pass (sidecar) |
| `fuse` | cpu-performance | base + adapter → fused bf16 (talker dropped) |
| `mlx` | cpu-performance | fused bf16 → 4-bit MLX snapshot |
| `finalise` | cpu-performance | fuse then mlx in one job |
| `push-adapter` | cpu-performance | restore newest checkpoint from bucket + re-upload adapter |

To run a smoke test locally:

```bash
uv sync --extra train
uv run python -m bimomni.training.app download
uv run python -m bimomni.training.app prepare
uv run python -m bimomni.training.app train --smoke
```

## Testing

```bash
uv sync --extra dev
uv run pytest
```

The unit tests cover the recipe identity, checkpoint sync, budget guard,
training command construction, knowledge benchmark loaders, fuse
verification, strip-talker correctness, supervisor dispatch, upload
gating, and the chunked transcription helpers. MLX-only paths are
imported lazily so the suite runs on Linux CI without Apple Silicon.

## Demo Space

A ZeroGPU Gradio Space under [`space/`](space/) transcribes a single ≤29 s clip
three ways — Whisper-large-v3, Qwen3-Omni (base), and BimOmni (base + the V4
LoRA) — in one `@spaces.GPU` task, toggling the adapter on/off via PEFT's
`disable_adapter()`. An activation probe refuses to serve comparisons if the
adapter is silently inactive.

The Space source is self-contained in `space/` (app + `app_lib/` package +
requirements + Space card). See [`space/README.md`](space/README.md) for the
HF Space front-matter and deployment steps, and
[`space/samples/README.md`](space/samples/README.md) for the operator-curated
clip list (not committed — add WAVs at deploy time).

## Project status

BimOmni is a working, published model. The training pipeline, evaluation
set, and publishing flow are stable. The corpus, the checkpoint bucket,
and any notebook environments are private and never leave the cluster.

BimOmni is not a finished product — it is a research artefact built
during a 21-day buildathon. Treat its outputs as a starting point for
human review rather than as ground truth.

## Author

Built by [Matt Hamilton](https://dharach.com/) — [matt@dharach.com](mailto:matt@dharach.com).

## Licence

This repository is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for the full text.

The BimOmni model weights are released under the same terms as the
upstream Qwen3-Omni-30B-A3B-Instruct base model. Review the original
[Qwen licence terms](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
before redistribution or deployment.
