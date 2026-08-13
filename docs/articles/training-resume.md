# From a half-finished training run to a reproducible eval

So our last training pass on `Qwen3-Omni-30B-A3B-Instruct` — fine-tuning it on Barbados newspapers — ran out of time halfway through. Twelve hours of GPU budget, planned for 1624 training steps, finished at step 1000. Loss was still falling. The full checkpoint, optimiser state, scheduler — everything you need to keep going — was sitting safely in a private cloud bucket. So we restarted it. Then the interesting bit happened.

The headline finding is the +5 percentage points on the 60-probe text knowledge eval (reproducible to 0pp on the same stack). The bigger finding is the radio audio number, which roughly doubled on completion. Both point to the same conclusion: completing the training schedule to step 1624 mattered.

![Full training loss curve, V3 run (step 0–1000) and V4 resume (step 1000–1624)](barbados-dapt-v4-full-loss.png)

## Context: Future Caribbean and Pulse

This work is part of [Future Caribbean](https://futurecaribbean.com), a regional initiative that runs a global agentic AI buildathon over a 21-day sprint each summer. The framing is "build deployable AI systems that strengthen how economies coordinate" — ten tracks, $70K+ in prizes, and a final showcase at the NYSE. We're one of the selected teams.

The programme advertised NVIDIA H200-class compute for selected teams, and we planned this training work on that basis. In practice, compute was not made available to our team during the build window. Rather than drop the model work, I rented the H200 capacity personally through Hugging Face Jobs. That is a decision worth flagging: access to compute materially shapes what small teams can attempt in a three-week buildathon, and going it alone means we could only train the model once, on the budget I could afford.

Our build is **Pulse** — a live public-signal intelligence system for Barbados that continuously ingests fragmented social, broadcast, and news signals, resolves them into a trusted graph of sources, places, events, people, and communities, and exposes that intelligence through agentic workflows. The Barbados LoRA is one of the domain-knowledge components feeding into Pulse's natural language layer.

The model itself isn't Pulse-specific. It's a standard text-only LoRA on a standard base model, trained on a Barbados newspaper archive. The source corpus is private, but the adapter is public. Anyone working on Caribbean-domain text — search, summarisation, knowledge probes, a personal project on Bajan history — can download and use the V4 artefacts directly. You don't need to be running Pulse to benefit from the training, and Pulse isn't the only reason the model exists.

## The setup, briefly

Domain-adaptive pretraining (DAPT) is the simplest "teach a model something new" technique you can run: take a model that's already been trained on the open web, and continue its next-token prediction on a corpus from your target domain. No labels, no chat format, no reward model. Just keep showing it text and let the loss fall. Gururangan and colleagues at ACL 2020 called this [Don't Stop Pretraining](https://aclanthology.org/2020.acl-main.740/) — the follow-up paper that established the technique.

In our case the corpus is the cleaned Barbados newspaper archive, the base model is a 30-billion-parameter multimodal MoE, and the fine-tune is a small low-rank adapter (LoRA) on the text-only "thinker" half of the model. Training runs on a rented H200 GPU inside a Hugging Face Job. This post is about what happened when we tried to finish the schedule and what re-scoring the older artefacts taught us.

## The mistake that shaped the recipe

The first real training run only touched the Thinker's attention projections — the `q`, `k`, `v`, and `o` projection layers around each attention head. That made 53.5 million parameters trainable: 0.17% of the 31.8-billion-parameter base model. The reasoning at the time was that attention is where the model decides which knowledge to retrieve, and retargeting that was the cheapest plausible intervention. We called it V2.

The mistake was thinking that was enough. On the GPU paired path (LoRA loaded at runtime into a bf16 base), V2 helped. On the 4-bit fused path that Pulse actually runs — the LoRA merged into the weights, then quantised — V2 *regressed*. 55% overall, against 61.7% for the unfine-tuned 4-bit base. Fusing an attention-only LoRA into a 4-bit snapshot loses more fidelity than the same adapter applied at runtime. The base model still knew things the fused V2 didn't.

Imagine the model as a library. Attention is the librarian deciding which shelves and passages matter for the question. The MLP/MoE pathway is more like the subject specialists who read that material, transform it, and connect it to other facts. Retraining the librarian — V2 — changed where the model looked. It did not change the specialists' working knowledge. On a paired eval the librarian can be enough: the specialists still know what they knew. Fused into 4-bit weights, the librarian's retraining doesn't survive the round-trip cleanly, and the specialists never moved at all.

The later recipe kept the attention targets and added LoRA parameters to the Thinker's MLP/MoE pathway (`gate_up_proj` and `down_proj`). That expanded the trainable surface to about 2.59 billion parameters, or 7.54% of the base model. In library terms, we could now retrain both the librarian and the subject specialists. V3 stopped partway through that broader schedule. V4 is the completed version. The +15pp jump from V2 (55%) to V3 (70%) is the recipe change doing real work — not the schedule completing.

## Picking up where we left off

Resuming a Hugging Face Job is conceptually simple: launch a new job that points at the same run identifier. The bucket-and-sidecar architecture (the worker continuously uploads checkpoint snapshots to a private bucket as it trains) handles the bookkeeping — when a fresh training job starts, it asks the bucket "what's the latest complete snapshot for this run?" and picks up from there.

In practice there were two things to be careful about. First, the resume check enforces recipe identity strictly — it compares a hash of the training config (base model, dataset, LoRA shape, library versions, the *image digest of the container*) against the snapshot it's about to restore. So we couldn't use the rebuilt container image that fixes an upload bug elsewhere in the pipeline; the manifest wouldn't match, the resume would silently skip the bucket, and we'd start a brand new run instead. We used the original container and dealt with the upload bug the same way we did for V3: don't trust the in-process adapter upload, publish the final adapter in a separate recovery job.

Second, that recovery job hardcoded the destination repository name in the supervisor code, pointing at V3. There was no flag to override it. Three-line code change to fix, but the cleanest workaround without rebuilding the container was an inline Python script that calls the same upload primitives with a different target — published `…-LoRA-v4` in 61 seconds. Then a single `finalise` job produced the fused bf16 and the 4-bit MLX snapshots (the latter is what you actually run on a Mac). Both published, both verified — the provenance file correctly lists `adapter: …-LoRA-v4` rather than the V3 adapter a publishing bug had caught in an earlier round.

The full resume, including one sidecar-marker hiccup that took a two-minute manual fix, is recorded in `docs/19` §10.2.

## The result

Three runs on the same 60-probe Barbados knowledge set — twenty well-known local facts (parliament year, Crop Over revival, parish count), twenty niche ones (Barrow constituency, Leadpipe Glitch, Grantley Adams Secondary), and twenty general-knowledge controls (longest European river, deepest trench, Marie Curie). The model picks one of four candidate completions for each probe by ranking them on mean log-probability over the completion tokens. Right or wrong.

| Run | Recipe | Schedule | Overall |
|---|---|---|---:|
| V2 | attention only (`q/k/v/o_proj`) | 500 / 801 steps · 53.5M params | 55.0% |
| V3 | attention + MLP/MoE | 1000 / 1624 steps · 2.59B params | 70.0% |
| V4 | attention + MLP/MoE (completed) | 1624 / 1624 steps · 2.59B params | 75.0% |
| base | unfine-tuned 4-bit | — | 61.7% |

![Overall accuracy on the 60-probe Barbados text knowledge eval — base 4-bit, then V2, V3, V4 fused 4-bit](v2-v3-v4-text-probe.png)

The V2 number is the most important one in the table. The attention-only recipe *helped* on a GPU paired eval (LoRA loaded at runtime into a bf16 base) — a result from earlier in the project that I wrote up at the time. But on the 4-bit fused path that Pulse actually runs, V2 scored worse than base: 55% against 61.7%. Fusing an attention-only LoRA into a 4-bit snapshot loses more fidelity than fusing one that also has MLP/MoE parameters. V3's recipe change — adding LoRA to `gate_up_proj` and `down_proj` — is what brought the fused 4-bit path back to parity. The +15pp jump from V2 (55%) to V3 (70%) is the recipe change doing real work, not the schedule completing.

V4 then completes the schedule V3 started, lifting another 5pp on top.

The reproducibility question was whether +5pp on the V3 → V4 step was real or evaluator drift. The eval script's hardcoded model-repo field pointed at V3, the MLX backend is sensitive to small numerical changes between versions, and 5pp on 60 probes is roughly three probes — within sample noise. The honest test was to re-score the earlier artefacts on the same stack — V3's fused 4-bit snapshot and the unfine-tuned base, both downloaded from the Hub.

| Track | base | V3 | V4 |
|---|---:|---:|---:|
| local | 55.0% | 65.0% | 70.0% |
| rare_local | 40.0% | 55.0% | 60.0% |
| control | 90.0% | 90.0% | 95.0% |
| **overall** | **61.7%** | **70.0%** | **75.0%** |

All three reproduce to **0pp** of their originally-recorded numbers. The eval path is stable. The +5pp V3 → V4 is real weight movement. The cumulative DAPT contribution (base → V4) is **+13.3pp overall**: +3 local, +4 rare_local, +1 control.

## The interesting bit

So the +5pp is real, but the mechanism is interesting. On the 34 probes that both the base model and V4 got right, V4's mean margin (the gap between the top-ranked correct completion and the runner-up) is **0.56 log-prob *smaller*** than the base model's. V4 is *less* confident on the easy probes it shares with the base model, yet wins more probes overall.

This is the signature of a model that's been retargeted toward a domain. The output layer has shifted to put more weight on Barbados-relevant tokens, which compresses the log-probability gap on general-completion probes where those tokens weren't relevant, while flipping close calls on the domain probes where they are. Same total budget, redistributed. The accuracy gain comes from breaking ties on the borderline probes, not from being more confident on what's already easy.

The +1 control-probe gain (a paraphrasing probe the base and V3 both missed) is also real and slightly unexpected. V4 picks up some paraphrasing robustness from incidental general prose in the corpus (sports, world news, op-eds). One probe on N=20 is suggestive rather than load-bearing, but it does mean V4 is genuinely a better model, not just a better Barbados model.

## Does it survive contact with actual audio?

The 60-probe eval is text-only — it asks whether the model knows facts about Barbados. Pulse's actual workload is audio: five-minute FM radio windows and short-form TikToks that need transcription and structured extraction (events, venues, prices, promotions). Two more evals cover that path.

**Radio — 17 five-minute windows** from real Barbados FM (HOTT 95.3, VOB 92.9, CBC 94.7, Life 97.5, Q 100.7, The Beat 104.1, The One 98.1, Y 103.3, Radio Bimshire 106.1, SLAM 101.1). Scored on quality, hallucination rate, proper-noun precision/recall, event false-positive and false-negative rates, and whether evidence quotes survive.

| Model | Windows | Avg quality | Avg latency |
|---|---:|---:|---:|
| base | 17 | 0.135 | 9.2s |
| V3 | 17 | 0.140 | 15.5s |
| **V4** | **17** | **0.230** | 17.8s |

![Radio audio extraction — V4 nearly doubles V3's quality score](v4-table-radio.png)

V3 was essentially flat against base on radio. Completing the LR schedule to step 1624 nearly doubled the audio extraction quality (0.140 → 0.230). The place where DAPT shows up most clearly is the proper-noun layer: "Rise Together" instead of "Rice Together", "Leadpipe and Sadis" instead of plumbing+typo. These errors are unrecoverable once cheap Whisper has destroyed them, so the audio-tower-bearing model is the only place to fix them.

**TikTok — 10 Barbados TikToks** (golf promotions, catamaran cruises, cave tours, farmers' markets, flower forests, rug-making workshops). Each fixture has hand-written ground truth for the observations the pipeline should extract.

| Model | Precision | Recall | F1 | Field recall |
|---|---:|---:|---:|---:|
| base | 0.26 | 0.36 | 0.30 | 0.42 |
| V3 | 0.60 | 0.55 | **0.57** | 0.38 |
| V4 | 0.34 | 0.55 | 0.42 | **0.52** |

![TikTok extraction — V3 wins on precision, V4 wins on field recall](v4-table-tiktok.png)

V4 and V3 are tied on recall (0.55) — both find the right observations. They diverge on what they do around the edges. V3 is more conservative: higher precision (fewer false positives), lower field recall (less detail per observation). V4 emits more candidate entities per fixture (35 vs V3's 20 across all 10) and fills more fields correctly per observation, but at the cost of more false positives. That's the same calibration shift showing up in a different modality: V4's unembedding has broadened its candidate distribution, which helps coverage and hurts precision.

There is a fair question about whether V4's precision drop is a real regression or an artefact of the benchmark itself. The ground truth for each fixture is hand-written — one to six expected observations per video. When V4 emits a correct observation that isn't in that small gold set, the scorer counts it as a false positive. The model isn't wrong; the ground truth is incomplete. This is a known problem in information-extraction evaluation, and we're working on better benchmarks that separate raw transcription accuracy from knowledge extraction quality, and that don't punish a model for finding more than the annotator thought to write down.

If that benchmark design question is more interesting to you than the model, I wrote a whole separate post on it: [When your benchmark is wrong and your model is right](https://dev.to/hammertoe/when-your-benchmark-is-wrong-and-your-model-is-right-p74).

The V4 artefacts are at:

- `hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4`
- `hammertoe/BimOmni-30B-A3B`
- `hammertoe/BimOmni-30B-A3B-MLX-4bit`

If you want to take V4 for a spin, the model card on each Hub repo walks through download, the supported prompt format, and the activation-check probe that proves the adapter is wired up. The 60-probe set is published as part of the model card on the V4 adapter repo. I'd be curious whether you see the same −0.56 log-prob margin shift on the easy probes.

Stay tuned for the next iteration. I want to try a longer schedule to see if +13.3pp is in fact a ceiling for this recipe, or whether the calibration shift keeps compounding as the model spends more time in the corpus.
