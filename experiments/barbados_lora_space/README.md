---
title: BimOmni Barbados Demo
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
startup_duration_timeout: 1h
license: apache-2.0
---

# BimOmni Barbados fine-tune — side-by-side audio transcription

ZeroGPU demo of the Barbados LoRA fine-tune for Qwen3-Omni. Three models
    transcribe the same ≤29 s clip in a single GPU run:

- **Whisper-large-v3** — generic ASR baseline (`openai/whisper-large-v3`).
- **Qwen3-Omni (base)** — `Qwen/Qwen3-Omni-30B-A3B-Instruct` at the pinned
  revision the adapter was trained on, with the LoRA adapter **disabled**.
- **BimOmni fine-tune** — the same bf16 weights with
  `hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4` **enabled**.

The base and fine-tune columns share one set of weights: PEFT injects the
adapter in place and each transcription toggles it via `disable_adapter()`.
Any difference between those two columns is purely the fine-tune. No
quantization anywhere.

## Use

    Upload a clip (≤29 s). The first compare() pays a one-time cold start
(~minutes) while the base Qwen3-Omni checkpoint (~63 GB bf16) and the
adapter (~10 GB) are fetched inside the GPU task. Differences surface on
**proper nouns** — Caribbean place names, festivals, institutions.

The **adaptation probe** button measures log-probability preferences for
Barbados proper nouns with the adapter toggled off and on.
