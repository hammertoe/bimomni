---
title: BimOmni Audio Demo
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Caribbean-aware Whisper vs Qwen3-Omni vs BimOmni
---

# BimOmni — Caribbean-aware audio transcription

Three models transcribe the same ≤29 s clip side-by-side in a single GPU run:

- **Whisper-large-v3** — ASR baseline (`openai/whisper-large-v3`).
- **Qwen3-Omni (base)** — `Qwen/Qwen3-Omni-30B-A3B-Instruct` at the pinned
  revision used to publish the V4 adapter.
- **BimOmni** — same Qwen base with
  [`hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4`](https://huggingface.co/hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4)
  attached via PEFT.

Single model in memory, adapter toggled on/off between the two Qwen runs via
PEFT's `disable_adapter()` context manager. Differences usually show on
**proper nouns** (Caribbean place names, festivals, institutions) — BimOmni
should match the audio and the base often scrambles the spelling.

## Use

Upload a clip (≤29 s) or pick a bundled sample. Free tier quota is ~5 min of
GPU time per day. The first compare() pays a one-time cold start (~minutes)
while the 63 GB base model is fetched and attached to PEFT.

## How it works

A `@spaces.GPU(duration=180)` task holds the model and runs Whisper, the
base Qwen3-Omni, and the adapted BimOmni in sequence. A startup probe
checks that the Barbados-completion logprob actually moves between base
and adapted runs; if it doesn't, the demo refuses to serve comparisons.
(This was the V2 silent-inactive-adapter bug, rendered as a hard gate.)

## Sources

- Base model: [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) @ `26291f793822fb6be9555850f06dfe95f2d7e695`
- Adapter: [hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4](https://huggingface.co/hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4)
- Fused / 4-bit MLX variants: [hammertoe/BimOmni-30B-A3B](https://huggingface.co/hammertoe/BimOmni-30B-A3B), [hammertoe/BimOmni-30B-A3B-MLX-4bit](https://huggingface.co/hammertoe/BimOmni-30B-A3B-MLX-4bit)

## Credits

DAPT pass over the *Barbados Advocate* 2013–2023 (~51.6 M tokens). 75.0%
on the 60-probe Barbados knowledge set (20 local / 20 rare-local /
20 control). Source: [github.com/hammertoe/bimomni](https://github.com/hammertoe/bimomni).
