---
language:
  - en
license: other
license_name: apache-2.0
library_name: transformers
pipeline_tag: any-to-any
base_model:
  - Qwen/Qwen3-Omni-30B-A3B-Instruct
tags:
  - multimodal
  - audio
  - barbados
  - caribbean
  - domain-adaptive-pretraining
  - lora
---

# BimOmni 30B-A3B

**BimOmni** is a Barbados-adapted version of
[Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).

It was developed to test a simple idea: when audio is ambiguous, can stronger
knowledge of Barbados help a multimodal model choose the correct local name,
place, institution, event, or phrase?

BimOmni keeps Qwen3-Omni's audio and visual input towers, but adapts the
Thinker's language model through text-only domain-adaptive pretraining on a
Barbados newspaper archive.

This repository contains the fused bf16 checkpoint. The trained LoRA has been
merged into the base weights and unloaded.

> **BimOmni hears Barbados properly.**

## Project background

BimOmni was developed as part of
[Future Caribbean](https://futurecaribbean.com/), a regional initiative
connecting technology, talent, and opportunity across the Caribbean through a
global agentic AI buildathon.

The model is one component of **Pulse**, a public-signal intelligence system
for Barbados. Pulse ingests fragmented social, broadcast, and news signals,
then resolves them into a graph of sources, places, events, people, and
communities.

The model itself is not tied to Pulse. It can be used independently for
Caribbean-domain transcription, information extraction, search,
summarisation, research, and other multimodal applications.

## Why BimOmni?

Automatic speech recognition works well until somebody mentions a local
school, village, politician, festival, restaurant, performer, or cricket
ground.

At that point, transcription is not purely acoustic. The model must choose
between several plausible sequences.

For Barbados, useful context includes names such as:

- Kensington Oval
- Cave Hill
- Samuel Jackman Prescod Polytechnic
- Crop Over
- Oistins Fish Fry
- Richard Haynes Boardwalk
- Puddin' and Souse
- Animal Flower Cave

BimOmni was trained to strengthen that local language prior:

```text
ambiguous audio + stronger Barbados context -> better local word choice
```

It is important to be precise about what this means. BimOmni was adapted using
text, not paired Barbadian audio and transcripts. It does not have a newly
trained acoustic encoder and should not be described as having learnt a
Barbadian accent. The intended improvement is in how the Thinker interprets
ambiguous multimodal evidence and selects its output tokens.

## Training

BimOmni uses domain-adaptive pretraining, or DAPT. Rather than converting the
corpus into question-and-answer pairs, the model continued its next-token
training on cleaned Barbados newspaper text.

The working corpus was built from 1,286 extracted editions of the *Barbados
Advocate*, covering 2013 to 2023. The initial prepared snapshot contained
approximately 51.6 million tokens.

The corpus is not redistributed.

### Architecture and recipe

The training adapted Qwen3-Omni's **Thinker**:

- Base model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Base revision: `26291f793822fb6be9555850f06dfe95f2d7e695`
- Method: LoRA domain-adaptive pretraining
- LoRA targets: attention and MLP/MoE projections
- Trainable parameters: approximately 2.59 billion
- Trainable proportion: approximately 7.54%
- Context length: 4,096 tokens
- Precision: bfloat16
- Final schedule: 1,624 of 1,624 planned optimiser steps
- Audio and visual encoders: frozen
- Talker: disabled

The V4 schedule was completed on a rented NVIDIA H200 through Hugging Face
Jobs. Checkpoints were persisted separately so that the run could resume after
its original 12-hour training window ended at step 1,000.

## Evaluation

### Barbados knowledge evaluation

The text evaluation contains 60 four-choice completion probes:

- 20 local Barbados probes
- 20 rare-local probes
- 20 general-knowledge controls

Each answer is scored using its mean token log-probability rather than asking
the model to generate a multiple-choice letter.

| Model | Training recipe | Schedule | Overall accuracy |
| --- | --- | ---: | ---: |
| Qwen3-Omni base, 4-bit | No adaptation | - | 61.7% |
| V2 | Attention-only LoRA | 500 / 801 | 55.0% |
| V3 | Attention + MLP/MoE LoRA | 1,000 / 1,624 | 70.0% |
| **BimOmni (V4)** | **Attention + MLP/MoE LoRA** | **1,624 / 1,624** | **75.0%** |

Re-running the base, V3, and V4 artefacts on the same MLX evaluation stack
reproduced their original results to zero percentage points.

BimOmni improved by **13.3 percentage points overall** compared with the
unadapted 4-bit base model.

| Track | Qwen3 base | V3 | BimOmni |
| --- | ---: | ---: | ---: |
| Local | 55.0% | 65.0% | **70.0%** |
| Rare local | 40.0% | 55.0% | **60.0%** |
| General controls | 90.0% | 90.0% | **95.0%** |
| **Overall** | **61.7%** | **70.0%** | **75.0%** |

This is a small, project-specific knowledge probe rather than a broad
measurement of cultural understanding.

### TikTok transcription comparison

We also compared three local transcription paths on two Barbados TikTok
videos:

- **Whisper**: the Whisper transcription path used by Pulse
- **Qwen3**: unadapted `Qwen3-Omni-30B-A3B-Instruct`, 4-bit MLX
- **BimOmni**: `BimOmni-30B-A3B-MLX-4bit`

The test used 29-second audio windows with five seconds of overlap. Audio was
normalised to 16 kHz mono before inference. Overlapping chunks were stitched
after transcription.

| Video | Whisper | Qwen3 | BimOmni | Reported character similarity |
| --- | ---: | ---: | ---: | ---: |
| Itinerary | 386 words / **7.8 s** | 387 words / 30.7 s | 384 words / 25.8 s | 0.372 |
| Harbour Lights | 206 words / **2.8 s** | 208 words / 12.4 s | 207 words / 11.2 s | 0.564 |

Whisper was the fastest path on both samples. BimOmni was faster than the
unadapted Qwen3 model, but its main advantage was the handling of
Barbados-specific proper nouns.

### Itinerary proper nouns

The following manually reviewed examples come from the itinerary video used in
the comparison.

| Reference | Whisper | Qwen3 | BimOmni |
| --- | --- | --- | --- |
| Worthing Square | Wooding Square | Willing Square | **Worthing Square** |
| Chefette | Shafet | **Chefette** | **Chefette** |
| Richard Haynes | Richard Hines | Richard Hinds | **Richard Haynes** |
| Q.P. Bistro | Kupi Bistro | QPB Bistro | **Q.P. Bistro** |
| Oistins Fish Fry | Oistin's Fish Fry | Oyston's Fish Fry | **Oistins Fish Fry** |
| Puddin' and Souse | putting on south | Puddin' and Sauce | **Puddin' and Souse** |
| Farley Hill | **Farley Hill** | Carlisle Hill | **Farley Hill** |
| Animal Flower Cave | **Animal Flower Cave** | Animaux Flower Cave | **Animal Flower Cave** |
| Carlisle Bay | Carlyle Bay | **Carlisle Bay** | **Carlisle Bay** |
| Champers | Champas | **Champers** | **Champers** |

| Model | Correct examples |
| --- | ---: |
| Whisper | 2 / 10 |
| Qwen3 | 3 / 10 |
| **BimOmni** | **10 / 10** |

These ten examples were selected to examine local proper-noun handling. They
are not a complete word-error-rate benchmark, and the two-video comparison
should not be interpreted as a general ranking of transcription systems.

The practical result is narrower:

- Whisper remained substantially faster.
- The unadapted Qwen3 model corrected some local names but was the slowest.
- BimOmni produced the strongest Barbados-specific proper nouns in these
  samples.

## Model structure

| Component | Status |
| --- | --- |
| Thinker language model | Adapted and fused |
| Audio input tower | Preserved |
| Visual input tower | Preserved |
| Text output | Preserved |
| Talker and speech output | Removed |
| `enable_audio_output` | `false` |

The model accepts text, audio, image, and video inputs, but produces text only.

## Usage

Download the fused bf16 checkpoint:

```bash
hf download hammertoe/BimOmni-30B-A3B \
    --local-dir BimOmni-30B-A3B
```

Use the same processor and prompt structure documented for the original
Qwen3-Omni model.

For local Apple Silicon inference, use the 4-bit MLX snapshot:

```bash
hf download hammertoe/BimOmni-30B-A3B-MLX-4bit \
    --local-dir BimOmni-30B-A3B-MLX-4bit
```

For audio with `mlx-vlm`, keep individual audio windows below 30 seconds. The
project's validated transcription path uses 29-second windows with a
five-second overlap and 16 kHz mono audio.

## Limitations

- BimOmni was adapted using newspaper text, not paired Barbadian speech data.
- It should not be treated as proof of improved accent or dialect recognition.
- The newspaper corpus contains OCR errors and reflects the editorial choices
  and historical biases of its source.
- The model may over-prioritise familiar Barbados entities when acoustic
  evidence is weak.
- The knowledge evaluation contains only 60 probes.
- The transcription comparison contains only two videos.
- The proper-noun table contains a deliberately selected set of ten local
  entities and is not a general ASR benchmark.
- Latency depends on hardware, software versions, quantisation, chunking, and
  generation settings.
- The model's speech-generation components were removed. It cannot produce
  audio output.
- As with the base model, generated information should be verified before use
  in consequential settings.

## Training story

The project and its training process are documented in two articles:

1. [Teaching an Audio Model More About Barbados](https://dev.to/hammertoe/teaching-an-audio-model-more-about-barbados-32o2)  
   The motivation, corpus preparation, first training recipe, initial knowledge
   evaluation, and the inference bug that initially made the adapter appear
   inactive.

2. [From a half-finished training run to a reproducible eval](https://dev.to/hammertoe/from-a-half-finished-training-run-to-a-reproducible-eval-1k78)  
   The broader V3/V4 recipe, checkpoint recovery, completed 1,624-step
   schedule, reproducibility work, and text, radio, and TikTok evaluations.

## Author

Created by [Matt Hamilton](https://dharach.com/).

- Website: [dharach.com](https://dharach.com/)
- Email: [matt@dharach.com](mailto:matt@dharach.com)

## Provenance

- Project: [Future Caribbean](https://futurecaribbean.com/)
- System: Pulse
- Author: [Matt Hamilton](https://dharach.com/)
- Base model:
  [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- Base revision: `26291f793822fb6be9555850f06dfe95f2d7e695`
- Adapter:
  [`hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4`](https://huggingface.co/hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4)
- Fused bf16:
  [`hammertoe/BimOmni-30B-A3B`](https://huggingface.co/hammertoe/BimOmni-30B-A3B)
- MLX 4-bit:
  [`hammertoe/BimOmni-30B-A3B-MLX-4bit`](https://huggingface.co/hammertoe/BimOmni-30B-A3B-MLX-4bit)

## Licence

The base model identifies its licence as Apache-2.0. Users should review the
licence and terms of the original Qwen3-Omni model before redistribution or
deployment.
