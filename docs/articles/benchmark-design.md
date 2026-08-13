# When your benchmark is wrong and your model is right

We fine-tuned a 30-billion-parameter model on Barbados newspapers. Then we had to write three different benchmarks to figure out whether it actually got better. It did — but not in the way any single benchmark could show on its own, and one of them made it look worse before it looked better.

This is about the benchmarks, not the model. The model is [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) with a LoRA adapter trained on a cleaned Barbados newspaper archive. We're building Pulse, a live public-signal intelligence system for Barbados — radio, newspapers, TikTok, and government feeds turned into a queryable knowledge graph — as part of [Future Caribbean](https://futurecaribbean.com), a 21-day agentic AI buildathon. Pulse is still in active development. The fine-tuned model is one of its domain-knowledge components, and it's also published openly for anyone working on Caribbean-domain text.

![Full training loss: V3 stopped at step 1000, V4 resumed to step 1624](barbados-dapt-v4-full-loss.png)

## Three benchmarks, three answers

The model went through two versions. V3 stopped halfway through its training schedule (step 1000 of 1624, when the 12-hour GPU budget ran out). V4 resumed and completed the full schedule to step 1624. We scored both against the unmodified base model on three different eval tracks.

### 1. Text knowledge probe — 60 multiple-choice questions

Twenty well-known Barbados facts, twenty niche ones, and twenty general-knowledge controls. Scored by mean token log-probability over four candidate completions. Clean, reproducible, deterministic.

| Model | local | rare_local | control | overall |
|---|---:|---:|---:|---:|
| base | 55% | 40% | 90% | 61.7% |
| V3 | 65% | 55% | 90% | 70.0% |
| V4 | 70% | 60% | 95% | 75.0% |

![Text knowledge probe — three models across four tracks](bench-text-probe.png)

V4 wins across the board. The DAPT contribution is real (+13.3pp overall vs base). This benchmark is the one I'd cite if someone asked "did the fine-tuning work?".

### 2. Radio audio extraction — 17 five-minute FM windows

Real Barbados radio captures (HOTT 95.3, VOB 92.9, CBC 94.7, and seven others). The model hears the audio and emits structured JSON: corrected transcript, events, entities, facts. Scored on quality, hallucination rate, proper-noun precision/recall, and evidence-quote presence.

| Model | avg quality | avg latency |
|---|---:|---:|
| base | 0.135 | 9.2s |
| V3 | 0.140 | 15.5s |
| V4 | **0.230** | 17.8s |

![Radio audio extraction — average quality score per model](bench-radio.png)

V3 was essentially flat against base on this benchmark. V4 nearly doubled the quality score. The difference is proper nouns: the model that has read a thousand editions of the *Barbados Advocate* hears "Rise Together" (a Crop Over event) instead of "Rice Together". That is the entire reason we fine-tuned.

### 3. TikTok extraction — 10 short videos

Barbados TikToks: golf promotions, catamaran cruises, cave tours, farmers' markets. Each has hand-written ground truth for the structured observations the pipeline should extract. Scored on field-level precision, recall, and F1.

| Model | precision | recall | F1 |
|---|---:|---:|---:|
| base | 0.26 | 0.36 | 0.30 |
| V3 | **0.60** | 0.55 | **0.57** |
| V4 | 0.34 | 0.55 | 0.42 |

![TikTok extraction — precision, recall, and F1 per model](bench-tiktok.png)

V4 is worse than V3 here. That is the interesting bit.

## What "worse" means

V4 and V3 find the same number of correct observations (12 true positives each). V4 emits 23 false positives; V3 emits 8. The difference is 15 extra observations that don't match anything in the ground truth.

I dug into where those 15 extras come from. Two patterns.

**Over-emission.** On a 65-second catamaran cruise video, V3 emitted 1 observation. V4 emitted 9 — the same catamaran cruise, three times (as duplicates), plus separate "promotion" objects for "drinks were already flowing", "so many options and variety", and "staff were super friendly". Every positive sentence in the review got its own JSON object. V4 found both expected observations (V3 found neither), but buried them in noise.

**Type confusion.** On an itinerary TikTok listing eight things to do, V3 emitted five `event_occurrence` objects: racehorses at Pebble Beach, sunrise on the east side, yacht charter, St Nicholas Abbey, Hammers Market. V4 saw the same content but emitted three of them as `entity_mention` instead — just the place name, without the activity context. The scorer matches on observation type first, so those became false positives (wrong type) while the expected events became false negatives (unmatched).

Both patterns share a root cause. Completing the training schedule broadened the model's sensitivity to Barbados content. Its output layer now fires more readily on domain tokens — place names, venue names, activity descriptions. In the text probe this shows up as higher accuracy. In audio extraction it shows up as better proper-noun transcription. In structured extraction it shows up as over-triggering: the model sees Barbados signal everywhere because it has been trained to.

## The benchmark is the bottleneck

The TikTok eval has a scoring problem, and it took V4 to expose it.

The ground truth for each fixture is hand-written: one to six expected observations per video, each with specific fields (name, location, category, price). It is a small, fixed target. When V4 emits a correct observation that isn't in the ground truth — say, "snorkelling at the first stop" on the catamaran video — the scorer counts it as a false positive. The model isn't wrong; the ground truth is incomplete.

This is a known problem in information-extraction evaluation. Strict precision against a small gold set punishes models that extract more than the annotator thought to write down. The right fix is either a larger gold set, a human-in-the-loop review of every "false positive" to separate real hallucinations from under-annotated ground truth, or a softer scoring rubric that rewards partially-correct observations instead of treating them as binary hits or misses.

We're now moving toward the second option: a review pass over every V4 "false positive" to classify it as a real hallucination (the model invented something that isn't in the audio) or an annotation gap (the model found something correct that the ground truth missed). That will give us a corrected precision number that reflects the model's actual quality, not the ground truth's coverage.

## Benchmarks evolve with the model

When we started, the 60-probe text eval was the only benchmark. It was designed to answer a simple question: does the adapter know more about Barbados than the base model? It does, reproducibly, and that question is answered.

But the text probe doesn't test what Pulse actually does. Pulse ingests audio — radio and TikTok — and extracts structured events from it. The radio eval was built next, and it revealed that V3 (the half-finished model) was barely better than base on audio extraction despite scoring +8pp on the text probe. That finding motivated resuming the training to produce V4, which is where the audio quality jumped.

Then the TikTok eval revealed that V4, the better model by every other measure, over-triggers on structured extraction. That finding is now driving the ground-truth review, which will in turn produce a better benchmark for the next model version.

Each benchmark answered one question and exposed the next one. The text probe asked "does it know more?". The radio eval asked "can it hear better?". The TikTok eval asked "can it extract cleanly?". The answers are yes, yes, and "it depends on how you score it".

That is not a failure of evaluation. It is what evaluation looks like when you are building something new. You write the benchmark you can write. You run it. It tells you something true but incomplete. The incompleteness becomes visible only when the model outgrows the benchmark's assumptions. Then you write the next one.

The alternative — waiting until you have a perfect benchmark before training — is a good way to never train at all.

## Where we are now

We're still in development, not production. The V4 numbers are encouraging — its audio extraction quality (0.230 vs base's 0.135) is the difference between a pipeline that transcribes "Rice Together" and one that transcribes "Rise Together". For a system whose job is to tell you what's happening in Barbados tonight, that is the whole game. But encouraging numbers on a small eval set are not the same as a model ready to ship.

What we're doing now is refining the benchmarks themselves. The TikTok precision regression made it clear that our extraction eval was scoring the wrong thing — punishing correct observations that the ground truth didn't anticipate. So we're splitting the evals along the lines of what actually matters to Pulse's use cases:

- **Raw transcription accuracy** — does the model hear the right words?
This is the proper-noun problem: "Rise Together" vs "Rice Together", "Leadpipe and Sadis" vs plumbing and a typo. It's a clean, scoreable metric that directly reflects whether the DAPT investment paid off.
- **Knowledge extraction quality** — given the right words, does the
model produce the right structured output? This is where the over-emission and type confusion show up, and where the scoring rubric needs to evolve beyond binary hit/miss against a small gold set.

These are different skills and they need different benchmarks. The current eval conflates them — a model can transcribe perfectly but still score zero on extraction because it emitted the wrong observation type, or because the ground truth didn't list the entity it found. Splitting them apart will tell us whether V4's regression is a hearing problem (it isn't — transcription is better than ever) or a formatting problem (it is — the model over-triggers on domain content).

The model artefacts are published openly at [hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4](https://huggingface.co/hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4) (adapter), [hammertoe/BimOmni-30B-A3B](https://huggingface.co/hammertoe/BimOmni-30B-A3B) (fused weights), and [hammertoe/BimOmni-30B-A3B-MLX-4bit](https://huggingface.co/hammertoe/BimOmni-30B-A3B-MLX-4bit) (4-bit MLX for Apple Silicon). You don't need to be running Pulse to use them — any Caribbean-domain NLP task can benefit.

If you have ideas for better extraction scoring against under-annotated ground truth, I'd love to hear them.
