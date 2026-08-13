# 16 - Running the Qwen3-Omni Barbados Adapter

This is the reproducible inference and benchmarking guide for the trained
`hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA` adapter. It records the working
library versions, the Qwen3-Omni-specific PEFT loading rule, proof that the
adapter is active, and the valid benchmark result from 2026-08-06.

The training history is in `docs/15_qwen3_omni_dapt_executed.md`. The benchmark
implementation is `hf_space/knowledge_benchmark.py`.

## 1. Known-good artefacts

V2 (docs/15) is the GPU paired-benchmark baseline; V3 (docs/19 §9) and
V4 (docs/19 §10) are the fused 4-bit MLX snapshots. **V4 is the current
recommended artefact for local inference** — finished the planned 1624-step
schedule, +5pp on every track of the 60-probe Barbados set versus V3.

| Item | V2 (paired baseline) | V3 (fused 4-bit) | V4 (fused 4-bit, current) |
| --- | --- | --- | --- |
| Base model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | (fused — no base needed) | (fused — no base needed) |
| Base revision | `26291f793822fb6be9555850f06dfe95f2d7e695` | — | — |
| Adapter | `…-LoRA` | `…-LoRA-v3` | `…-LoRA-v4` |
| Final fused | n/a | `…-fused-bf16-v3` | `…-fused-bf16-v4` |
| 4-bit MLX | n/a | `…-4bit-v3` | `…-4bit-v4` |
| Final step | step 500 of 801 | step 1000 of 1624 | step 1624 of 1624 |
| Train loss at stop | 2.07486992 | 2.156 | 2.159 |
| Benchmark overall | 70.0% (paired, GPU) | 70.0% (fused 4-bit) | 75.0% (fused 4-bit) |
| GPU used for validation | NVIDIA H200 141 GB | — (CPU MLX on Apple Silicon) | — (CPU MLX on Apple Silicon) |

Library versions (unchanged across V2/V3/V4):
- PyTorch `2.7.1+cu126`
- Transformers `5.8.1`
- PEFT for paired GPU inference `0.20.0`
- Apple Silicon local inference: `mlx==0.32.0`, `mlx-vlm==0.6.10`

## 2. Keep training and inference environments separate

Do not upgrade PEFT inside the environment used to resume ms-swift training:

- ms-swift 4.4.2 declares `peft>=0.11,<0.20`.
- PEFT 0.19.1 cannot load this adapter with Transformers 5.8.1. It fails in
  `convert_peft_adapter_state_dict_for_transformers` because Transformers'
  `WeightConverter` does not accept PEFT's `distributed_operation` argument.
- PEFT 0.20.0 loads and activates the adapter when it is attached to the full
  Qwen3-Omni model, but it is outside ms-swift 4.4.2's supported range.

Use the original training venv only for `swift pt`. Create a dedicated
inference venv for validation and benchmarking.

```bash
python3 -m venv /workspace/inference-venv
/workspace/inference-venv/bin/pip install --upgrade pip wheel
/workspace/inference-venv/bin/pip install \
  torch==2.7.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
/workspace/inference-venv/bin/pip install \
  transformers==5.8.1 peft==0.20.0 accelerate safetensors \
  huggingface_hub qwen_omni_utils
```

The benchmark pod reused `/workspace/venv` and upgraded PEFT after training had
finished. A future run should use the separate environment above.

## 3. Download on the GPU server

Downloading directly on the server avoids transferring roughly 86 GB through
the MacBook.

```bash
export HF_TOKEN="$(grep '^HF_TOKEN=' /workspace/.env | cut -d= -f2-)"
export HF_HOME=/data/hf

/workspace/inference-venv/bin/hf download \
  Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --local-dir /data/hf/base_local

/workspace/inference-venv/bin/hf download \
  hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA \
  --local-dir /data/adapter_pull
```

Before loading, verify the expected files:

```bash
test -f /data/hf/base_local/config.json
test -f /data/adapter_pull/adapter_config.json
test -f /data/adapter_pull/adapter_model.safetensors
du -sh /data/hf/base_local /data/adapter_pull
```

## 4. The critical loading rule

Attach PEFT to the complete `Qwen3OmniMoeForConditionalGeneration`, then select
the adapted thinker for text inference.

```python
import os

import torch
from peft import PeftModel
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

os.environ["ENABLE_AUDIO_OUTPUT"] = "0"

base_path = "/data/hf/base_local"
adapter_path = "/data/adapter_pull"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    base_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()
model.disable_talker()

processor = Qwen3OmniMoeProcessor.from_pretrained(
    base_path,
    trust_remote_code=True,
)
tokenizer = processor.tokenizer

adapted_root = PeftModel.from_pretrained(model, adapter_path).eval()
adapted_thinker = adapted_root.model.thinker
```

Do not do this:

```python
# Wrong for this checkpoint: adapter keys are rooted under model.thinker.
adapted_thinker = PeftModel.from_pretrained(model.thinker, adapter_path)
```

The adapter state dict contains names such as:

```text
base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_B.weight
```

Wrapping only `model.thinker` changes the expected module namespace. PEFT can
return without making the intended thinker projections active, which is more
dangerous than a hard failure. The first invalid benchmark did exactly this and
therefore produced byte-for-byte-equivalent base and adapter scores.

Qwen3-Omni's top-level generation path is not the text path used here. Loading
must happen at the root for namespace alignment; scoring and generation happen
through `adapted_root.model.thinker`.

## 5. Prove activation before trusting output

Successful deserialisation is not sufficient. Check the layer type, loaded
weights, and output delta on identical input.

```python
text = "The Crop Over festival in Barbados"
inputs = tokenizer(text, return_tensors="pt")
device = model.thinker.get_input_embeddings().weight.device
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

base_thinker = model.thinker.eval()
with torch.inference_mode():
    base_loss = base_thinker(
        input_ids=inputs["input_ids"],
        labels=inputs["input_ids"],
    ).loss

adapted_root = PeftModel.from_pretrained(model, adapter_path).eval()
adapted_thinker = adapted_root.model.thinker
q_proj = adapted_thinker.model.layers[0].self_attn.q_proj

assert type(q_proj).__module__.startswith("peft.tuners.lora")
assert "default" in q_proj.lora_A
assert float(q_proj.lora_B["default"].weight.norm()) > 0

with torch.inference_mode():
    adapted_loss = adapted_thinker(
        input_ids=inputs["input_ids"],
        labels=inputs["input_ids"],
    ).loss

delta = float(adapted_loss - base_loss)
assert delta != 0.0
print(float(base_loss), float(adapted_loss), delta)
```

On the validation pod, this probe produced:

```text
q_proj: peft.tuners.lora.layer.Linear
lora_A norm: 4.654426097869873
base loss: 3.560260534286499
adapted loss: 3.1088740825653076
delta: -0.4513864517211914
```

The safetensors file independently showed that 288 of 384 `lora_B` tensors
were non-zero. The zero tensors were mainly frozen audio-tower targets, not an
empty adapter.

## 6. Run the paired benchmark

```bash
export HF_HOME=/data/hf
export ENABLE_AUDIO_OUTPUT=0

setsid nohup /workspace/inference-venv/bin/python \
  /workspace/repo/hf_space/knowledge_benchmark.py \
  --model /data/hf/base_local \
  --adapter /data/adapter_pull \
  --output-dir /data/knowledge-benchmark-valid \
  > /workspace/knowledge-benchmark-valid.log 2>&1 < /dev/null &
```

The script scores all base probes before mutating the loaded root model with
PEFT, then scores the same probes through `adapted_root.model.thinker`. It
writes `results.json` and `report.md`.

Copy only the small results back:

```bash
mkdir -p artefacts/qwen3-omni-barbados-benchmark
scp -i ~/.ssh/id_ecdsa \
  ubuntu@<pod-ip>:/data/knowledge-benchmark-valid/results.json \
  artefacts/qwen3-omni-barbados-benchmark/results.json
scp -i ~/.ssh/id_ecdsa \
  ubuntu@<pod-ip>:/data/knowledge-benchmark-valid/report.md \
  artefacts/qwen3-omni-barbados-benchmark/report.md
```

## 7. Valid result

| Track | N | Base | Adapter | Accuracy delta | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local | 20 | 55.0% | 65.0% | +10.0% | +0.247 |
| Rare local | 20 | 45.0% | 60.0% | +15.0% | +0.198 |
| Control | 20 | 90.0% | 85.0% | -5.0% | -0.344 |
| Overall | 60 | 63.3% | 70.0% | +6.7% | +0.034 |

The control-adjusted local margin delta was `+0.567`. Strict paired fact
families improved from 7/20 to 11/20, with four acquired and zero regressed
local fact families. Controls dropped from 8/10 to 7/10 strict families.

This supports domain-knowledge acquisition, but the confidence intervals are
wide and the control regression is real. It does not establish unseen
generalisation. Full output is preserved in:

- `artefacts/qwen3-omni-barbados-benchmark/results.json`
- `artefacts/qwen3-omni-barbados-benchmark/report.md`

## 8. Failure modes and diagnosis

| Symptom | Root cause | Fix |
| --- | --- | --- |
| `WeightConverter` rejects `distributed_operation` | PEFT 0.19.1 and Transformers 5.8.1 incompatibility | Use PEFT 0.20.0 in a separate inference venv |
| `q_proj` remains `torch.nn.Linear` | Adapter attached below its saved root namespace | Attach to the complete Omni model |
| Base and adapter scores are exactly equal | LoRA is inactive even if loading returned successfully | Require layer, norm, and loss/logit delta checks |
| `ParamWrapper` rejects dropout during training | Qwen Omni parameter-target LoRA does not support non-zero dropout | Train with `--lora_dropout 0.0` |
| `qwen_omni_utils` import error | Transformers does not install this runtime dependency | Install `qwen_omni_utils>=0.0.9` |
| CUDA out of memory near model load/training | The 4096-token recipe used about 138/141 GB | Keep batch size 1 and gradient accumulation 16 |
| `mlx.core` cannot find `libmlx.so` on Linux | Installed PyPI MLX wheel was unusable on this CUDA pod | Use PyTorch/PEFT on Linux; perform MLX conversion on Apple Silicon |

## 9. MLX status

The PyTorch/PEFT adapter path above is validated. The 4-bit MLX deliverable is
now **validated too**, produced by the V3/V4 HF Jobs toolchain rather than Apple
Silicon: the finalise image runs `mlx_vlm.convert` (group size 64, affine 4-bit)
and publishes `hammertoe/BimOmni-30B-A3B-MLX-4bit` (4 shards,
~18.2 GB, audio + vision towers kept, talker dropped). The fused bf16 source is
`hammertoe/BimOmni-30B-A3B`.

V3 (`…-4bit-v3`) is the "62% / 12 h budget" baseline (step 1000 of 1624).
V4 (now `hammertoe/BimOmni-30B-A3B-MLX-4bit`) is the current recommended
artefact — finished the planned schedule (step 1624) and is +5pp on every track
of the 60-probe Barbados set versus V3.

Local download:

```bash
hf download hammertoe/BimOmni-30B-A3B-MLX-4bit \
    --local-dir model/qwen3-omni-4bit-v4
```

Local benchmark (Apple Silicon, fused snapshot, 60-probe set, mean token
log-probability):

```bash
.venv/bin/python -m hf_space.knowledge_benchmark_mlx_fused \
    --model model/qwen3-omni-4bit-v4 \
    --output-dir artefacts/qwen3-omni-barbados-benchmark-mlx-fused-4bit-v4
```

See `docs/19_qwen3_omni_dapt_v3_hf_jobs.md` for the executed runs and the
`barbados-dapt-v3-hf-jobs` skill for the pipeline and gotchas.

## 10. Teardown checklist

After the reports are copied and checksummed:

```bash
sha256sum /data/knowledge-benchmark-valid/results.json \
  /data/knowledge-benchmark-valid/report.md
prime pods terminate <pod-id> --yes --plain
prime disks list --output json
```

Delete any separately created persistent disk as well. Confirm both pod and
disk inventories are empty rather than assuming deletion succeeded.
