# 14 - Fine-tuning Qwen3-Omni on Barbados Newspapers

## Objective

Domain-adaptive pretraining (DAPT) of `Qwen/Qwen3-Omni-30B-A3B-Instruct` on the
Barbados newspaper JSONL corpus produced by `scripts/generate_all_domain_chunks.py`,
then publish the resulting adapter to the Hugging Face Hub for downstream 4-bit
MLX quantization on Apple Silicon.

Scope is text-only: the audio "talker" head is disabled so we never need to
supervise speech targets we do not have data for. The "thinker" MoE consumes
the corpus as assistant-only pretraining records.

## Decisions locked in

| Decision | Value |
| --- | --- |
| Training scope | Text-only DAPT, thinker only |
| Method | LoRA, r=64, alpha=128, target attention + MLP linears |
| Precision | bf16 base, fp32 LoRA weights |
| Hardware | Prime Intellect on-demand 1x H200 80 GB |
| Budget cap | 12 GPU-hours (aborts and uploads whatever exists) |
| Data mix | Pure Barbados newspaper JSONL, no general-text wash |
| Eval gate | bf16 smoke + perplexity delta + Barbados-prompt generation + CPU 4-bit proxy |
| Upload | Adapter-only repo, PEFT format, text-only tokenizer exported |
| Quantization (local) | 4-bit MLX on Apple Silicon via `mlx_lm.convert` |

## High-level flow

```
laptop                              prime intellect (H200)
------                              --------------------
1. rsync JSONL --ssh-->             persistent volume
   shards into                      /data/corpus/*.jsonl
   Prime Intellect disk

                                    2. python app.py download   (base model only)
                                    3. python app.py prepare    (pack corpus)
                                    4. python app.py train
                                       (budget_guard.py watches wall time)
                                    5. python app.py evaluate
                                       (bf16 + perplexity + gen + 4-bit proxy)
                                    6. python app.py upload
                                       (adapter + tokenizer + model card)

7. mlx_lm.convert   (Apple Silicon)
   mlx_lm.generate  smoke test
   mlx_lm.generate  real prompts
```

> **Note on corpus prep speed**: when generating the JSONL shards locally
> with `scripts/generate_all_domain_chunks.py`, pass `--daemon --workers 2`
> so Docling loads its models once and reuses them across all PDFs. This
> avoids paying the model-load cost per PDF and roughly halves total runtime
> compared to spawning one CLI subprocess per file.

## Compute: Prime Intellect

- 1x H200 80 GB on-demand. Price fluctuates by region ($1.23-$1.99/hr);
  budget cap of 12 hours therefore covers $15-$25 plus $1-$2 setup overhead.
- Provision via the Prime Intellect CLI or web UI. Request an Ubuntu 22.04 +
  CUDA 12.4 image with at least 400 GB of persistent storage attached.
- Network: pull the 60 GB base model from `huggingface.co`; push the small
  adapter back to the Hub. Confirm egress is included or budget ~$3 for it.
- SSH into the instance for all stages; no public endpoint required.

## Stage 1: Upload JSONL to Prime Intellect persistent storage (laptop)

Goal: make the corpus available to the rental without putting it on the
public Hugging Face Hub. Newspaper material may not be cleared for public
distribution, and Prime Intellect persistent disks are private.

1. Confirm `scripts/generate_all_domain_chunks.py` has produced all desired
   `*.jsonl` files under `model/pdfs/`.
2. Create a persistent disk via the Prime Intellect UI (`Storage` tab →
   `Create Disk`) or CLI:

   ```bash
   prime disks create --name barbados-corpus --size-gb 50 --region <region>
   ```

3. Provision an H200 instance with the disk attached:

   ```bash
   prime pods create \
       --gpu-type H200_80GB \
       --gpu-count 1 \
       --disk-size 400 \
       --disks <disk-id>
   ```

4. Note the instance SSH target. Push the JSONL shards directly with `rsync`:

   ```bash
   prime pods ssh <pod-id> mkdir -p /data/corpus
   rsync -avz --progress model/pdfs/*.jsonl \
       primeintellect://<pod-id>:/data/corpus/
   ```

   The exact transport depends on Prime Intellect's supported methods; if
   `rsync` over SSH is not exposed, use `scp` or the web-based file manager.
   For very large corpora, tar the shards first to reduce overhead:

   ```bash
   tar -czf corpus.tar.gz -C model/pdfs *.jsonl
   scp corpus.tar.gz primeintellect://<pod-id>:/data/corpus/
   # inside the instance
   tar -xzf /data/corpus/corpus.tar.gz -C /data/corpus/
   ```

5. Verify the corpus landed correctly by counting lines in a few shards
   inside the instance before moving to Stage 2.

The base model still downloads from Hugging Face in Stage 3; only the corpus
goes through Prime Intellect storage.

### Stage 1b (optional): Wikipedia corpus

`scripts/fetch_wikipedia.py` builds a separate Barbados Wikipedia corpus so
the "Pure Barbados newspaper JSONL" mix locked above can stay the default
while an experimental wash is tried later without touching the newspaper
shards.

```bash
WIKIMEDIA_USER_AGENT="pulse-wikipedia-corpus/1.0 (contact@example.com)" \
  uv run python scripts/fetch_wikipedia.py --output-dir model/wikipedia
```

Discovery is layered: a geosearch grid tiles the island (10 km max radius
per query, capped points split into overlapping sub-searches so nothing is
silently dropped), then curated Barbados categories are traversed
(`--category-depth`, default 2). Stub categories (e.g. "Category:Barbados
stubs") are not traversed, and articles whose cleaned prose falls below
`--min-article-words` (default 200, roughly the 256-token floor used by
`prepare_data.py`) are skipped — the Barbados category tree is stub-heavy and
short stubs would otherwise be fetched and then dropped downstream. Outputs:

- `model/wikipedia/barbados_wikipedia.jsonl` — assistant-only records, one
  per 250-1,000-word chunk with the article title prefixed.
- `model/wikipedia/barbados_wikipedia_sources.jsonl` — one provenance record
  per corpus line (aligned by index): digest, title, pageid, revid, URLs,
  categories, coordinates, discovery source.
- `model/wikipedia/.cache/` — per-page API cache for resumable runs
  (git-ignored); `--refresh` ignores it.
- `model/wikipedia/summary.json` and `fetch-failures.tsv`.

Wikipedia text is CC BY-SA 4.0; keep the provenance sidecar so attribution
survives into the model card. To run an experiment with a mixed wash, copy
the desired shards into the same `/data/corpus` target and record the mix
ratio in `summary.json` and the model card; do not mix shards in the
newspaper-only default.

## Stage 2: Provision the rental

1. Pick an H200 80 GB on-demand instance in a region close to your HF dataset
   bucket to minimize download latency.
2. Mount persistent storage at `/data`. Confirm at least 400 GB free
   (80 GB base weights, 50 GB HF cache, 50 GB checkpoints, room for logs).
3. SSH in and clone the project repo:

   ```bash
   git clone <pulse repo url> pulse
   cd pulse
   ```

4. Build the container (uses the project Dockerfile in `hf_space/`):

   ```bash
   docker build -t pulse/qwen3-omni-dapt hf_space/
   ```

5. Export required secrets:

   ```bash
   export HF_TOKEN=...             # write access to base and adapter repos
   export PRIME_BUDGET_HOURS=12    # for budget_guard.py
   ```

## Stage 3: app.py stages

All stages run inside the container via `python app.py <stage>`. They are
idempotent: re-running a stage skips work it has already completed.

### download

- `huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct` into
  `/data/hf/transformers`. The base model comes from the public Hub.
- Verifies the corpus shards at `/data/corpus/*.jsonl` are present and
  non-empty. The corpus is uploaded from the laptop in Stage 1 via the
  Prime Intellect persistent disk; this stage does not re-download it.
- Resumable: skips base model files already present and matching the expected
  size; reports missing corpus files so you know to re-run Stage 1.

### prepare

- Reads `/data/corpus/*.jsonl` (uploaded in Stage 1).
- Validates each record is `{messages: [{role: assistant, content: str}]}`.
- Length-sorts, dedups by SHA-256 of `content`, drops records outside
  256-4096 tokens (computed with the Qwen tokenizer).
- Packs into ms-swift DAPT JSONL at `/data/barbados_dapt_packed.jsonl`.
- Writes a held-out 2% split at `/data/barbados_dapt_eval.jsonl` for perplexity.
- Prints record counts and total token estimate.

### train

- Loads base with `Qwen3OmniMoeForConditionalGeneration.from_pretrained`,
  pins `dtype=bf16`, `attn_implementation=flash_attention_2`.
- Calls `model.disable_talker()` immediately after load.
- Wraps with `peft.get_peft_model` and the locked LoRA target list.
- Invokes ms-swift DAPT CLI with the locked recipe.
- `budget_guard.py` runs as a sidecar that:
  - Samples GPU-hours via `nvidia-smi --query-gpu=power.draw --format=csv`
    plus elapsed wall time.
  - At 12 GPU-hours sends SIGTERM to the training process and runs the upload
    stage with whatever checkpoint exists.

### evaluate

Runs four checks before allowing upload:

1. **bf16 smoke**: reload base + adapter, call `disable_talker()`, do a forward
   pass on a tiny batch, assert no exceptions and reasonable loss.
2. **Perplexity delta**: compute base + adapter perplexity on
   `/data/barbados_dapt_eval.jsonl`. Print delta. Fail if delta is positive
   (adapter made it worse).
3. **Generation check**: a handful of Barbados prompts (Crop Over, Kensington
   Oval, BDS$ pricing, hurricane season, parliamentary constituencies) with
   greedy decoding, 100 tokens. Print outputs for eyeball review.
4. **CPU 4-bit proxy**: load via `transformers` + `bitsandbytes` 4-bit, apply
   adapter, run a small forward. Surfaces wrong LoRA targets, broken chat
   template, or missing base layers before any MLX work happens.

If any check fails, exit non-zero. Upload is gated on a clean exit.

### upload

- Asserts the adapter loads cleanly with `PeftModel.from_pretrained`.
- Pushes the adapter, adapter config, base revision pin, text-only tokenizer
  files, and chat template to `pulse/Qwen3-Omni-30B-A3B-Barbados-LoRA`.
- Writes a model card with:
  - Base model revision SHA
  - Corpus source: Prime Intellect persistent disk; record count after
    dedup; total token estimate (no public dataset revision since the
    corpus is private)
  - Hyperparameters and budget cap
  - Exact `mlx_lm.convert` + `mlx_lm.generate` commands
  - Flag values: `--quantize --q-bits 4 --q-group-size 64`
  - Known limitations (text-only, talker disabled, corpus not redistributed)

## Stage 4: Local quantization (Apple Silicon)

After the adapter is on the Hub:

```bash
# 1. Convert base + adapter to MLX, stripping talker weights
mlx_lm.convert \
    --hf-model Qwen/Qwen3-Omni-30B-A3B-Instruct --revision <base sha> \
    --mlx-path ./mlx/qwen3-omni-barbados-bf16 \
    --skip-talker

# 2. Fuse adapter into the converted model
python hf_space/strip_talker.py \
    --mlx-path ./mlx/qwen3-omni-barbados-bf16 \
    --adapter-repo pulse/Qwen3-Omni-30B-A3B-Barbados-LoRA \
    --out ./mlx/qwen3-omni-barbados-fused

# 3. Quantize to 4 bits
mlx_lm.quantize \
    --mlx-path ./mlx/qwen3-omni-barbados-fused \
    --quantize --q-bits 4 --q-group-size 64 \
    --out ./mlx/qwen3-omni-barbados-4bit

# 4. Smoke test
mlx_lm.generate \
    --model ./mlx/qwen3-omni-barbados-4bit \
    --prompt "The Crop Over festival in Barbados" \
    --max-tokens 100

mlx_lm.generate \
    --model ./mlx/qwen3-omni-barbados-4bit \
    --prompt "Kensington Oval was packed for" \
    --max-tokens 100
```

Memory budget on Apple Silicon: a 30B model at 4-bit group-size 64 is
~15-17 GB of weights. A 32 GB M2/M3 Pro is realistic. A 16 GB machine will
OOM at long contexts; keep `--max-model-len` at or below 8192.

## Stage 5: Teardown

- Stop the Prime Intellect instance from the web UI or CLI.
- Confirm no further charges.
- Archive `/data/checkpoints` to local disk if you want a safety copy
  (uploaded adapter is canonical).

## Rollback

If the rental run fails or the perplexity check is negative:

1. The adapter is not uploaded (upload is gated).
2. Stop the instance; nothing to clean up.
3. Inspect logs under `/data/logs/` for the failing stage.
4. Re-run with adjusted hyperparameters (LR, rank, or sequence length).
   The persistent disk keeps the corpus, so re-runs do not need to
   re-upload the JSONL shards.

## Risk register

| Risk | Mitigation |
| --- | --- |
| H200 unavailable in region | Fallback: on-demand 1x A100 80 GB at $3.14/hr |
| transformers 5.x drift | Pin `transformers==5.8.1` in the Dockerfile |
| Talker disable insufficient for text-only forward | bf16 smoke test in `evaluate.py` catches it |
| Pure Barbados causes forgetting | Acceptable per locked decision; revisit if perplexity on a small general-text slice regresses badly |
| Budget overrun | `budget_guard.py` aborts at 12 GPU-hours and uploads whatever exists |
| Adapter format incompatible with MLX | PEFT assertion in `upload.py` + CPU 4-bit proxy in `evaluate.py` |

## Open questions to revisit after first run

1. Held-out perplexity delta vs base: target negative. If positive, lower LR
   to 5e-5 and retry.
2. Generation quality: if completions feel too generic, increase rank to r=128.
3. Speed: if wall-clock exceeds 12 hours, drop sequence length to 2046 or
   micro-batch to 1 with accumulation 16.