# QVQ YAQA best ~W3.2 reproducibility record

This document pins the best protocol-valid ~W3.2 Llama 3.2 1B result known as
of 2026-08-31. It records the quantizer revision, complete mixed-format
configuration, source-model and dataset bindings, corpus construction, data
splits, quantization command, and score-producing evaluation commands.

The experiment is `f6_yaqa182_nm10000`. It must not be confused with the
historical `551/1209` result whose evaluator/code provenance was subsequently
superseded by the Wave-10 audit.

## Result

| Field | Value |
| --- | ---: |
| Model | Llama 3.2 1B Instruct |
| Effective BPW | **3.1783404137931034** |
| GSM8K Platinum | **547/1209 = 45.2440033085%** |
| D300 aligned-token Top-1 | **3222/9600 = 33.5625%** |
| D300 exact trajectories at 32 | **23/300 = 7.6667%** |
| D300 mean first divergence | **9.8066666667** |
| YAQA seed | **0** |
| Quantizer commit | `3ebcf9a307231187178e29ae7ad91e156a84d6ef` |

The otherwise identical NM10k seed-1 arm scored `533/1209 = 44.0860%` on
GSM8K Platinum. The seed-0 result is therefore the best observed result, not a
seed-robust mean estimate.

Canonical artifacts:

```text
Checkpoint:
/root/qvq-results/calibration-fisher-scaling-v2/llama32-1b-f6_yaqa182_nm10000-anchor-up4-l6-l8

Quantization run:
/root/qvq-results/calibration-fisher-scaling-v2/llama32-1b-f6_yaqa182_nm10000-anchor-up4-l6-l8/qvq_quantize_run.json

GSM8K result:
/root/qvq-results/calibration-fisher-scaling-v2/llama32-1b-f6_yaqa182_nm10000-anchor-up4-l6-l8/post_quant_eval_gsm8k_platinum.json

D300 result:
/root/qvq-results/calibration-fisher-scaling-v2/llama32-1b-f6_yaqa182_nm10000-anchor-up4-l6-l8/post_quant_eval_divergence300.json

Checkpoint tensor manifest:
/root/qvq-results/calibration-fisher-scaling-v2/llama32-1b-f6_yaqa182_nm10000-anchor-up4-l6-l8/checkpoint_hashes.json
```

Checkpoint identity:

```text
quantize_config SHA-256:
92c6d2545fa1b098070a58cc9b56645a0cf342af30b65f208a08196b9d4c29e1

qvq_quantize_run.json SHA-256:
780461c4e9059debb0bf40c4ea05ee84292eb863cb4511d1534c355546252ee4

model.safetensors.index.json SHA-256:
6ad9b35d6dcc10bccfa9bf47ba5ed4958795beeb145050415be03e1e6e81b697

Serialized tensor count: 558
Quantized projection modules: 112
```

## Effective allocation

There are 16 transformer layers. The following rates apply to all layers
unless a layer exception is stated.

| Projection | Bits | Format |
| --- | ---: | --- |
| `self_attn.q_proj` | 2.0 | `qvq_v2b2_p32` |
| `self_attn.k_proj` | 2.5 | `qvq_v2b2_p32` |
| `self_attn.v_proj` | 3.5 | `qvq_v2b2_p32` |
| `self_attn.o_proj` | 4.0 | ordinary `qvq` L16/V2 |
| `mlp.gate_proj` | 3.0 | `qvq_v2b2_p32` |
| `mlp.up_proj` | 3.5 | `qvq_v2b2_p32` |
| `mlp.down_proj` | 3.0 | `qvq_v2b2_p32` |
| layers 6 and 8 `mlp.up_proj` | 4.0 | ordinary `qvq` L16/V2 |
| embeddings and LM head | dense | not quantized |

The two layer-specific W4 Up rules and the W4 O rule use ordinary QVQ because
V2B2-P32 supports rates only through W3.5. Dynamic rules are resolved using
first-match-wins semantics, so specific layer rules must precede broad role
rules.

## Complete quantization input config

Canonical repository path:

```text
scripts/configs/llama32_1b_fisher_scaling_yaqa182_nm10000.json
SHA-256 fe66d8969f8efea66cb37febf050144d474109caa6d85ec3df008b48da046a22
```

Exact contents:

```json
{
  "bits": 2,
  "format": "qvq_v2b2_p32",
  "bank_count": 2,
  "rounding": "yaqa",
  "yaqa": {
    "seed": 0,
    "regularization": 0.15,
    "regularization_by_rate": [
      [2.0, 0.15],
      [2.5, 0.02],
      [3.0, 0.02],
      [3.5, 0.02],
      [4.0, 0.02]
    ],
    "minimum_sequences": 10178,
    "batch_size": 1,
    "sequence_sort": "desc",
    "activation_checkpointing": true,
    "v2b2_family_mode": "reselect",
    "sample_strategy": "full"
  },
  "device": "cuda:0",
  "offload_to_disk": false,
  "dynamic": {
    "+:^model\\.layers\\.(6|8)\\.(mlp.up_proj)$": {
      "bits": 4,
      "format": "qvq"
    },
    "+:^model\\.layers\\.[0-9]+\\.(self_attn.q_proj)$": {
      "bits": 2
    },
    "+:^model\\.layers\\.[0-9]+\\.(self_attn.k_proj)$": {
      "bits": 2.5
    },
    "+:^model\\.layers\\.[0-9]+\\.(self_attn.v_proj)$": {
      "bits": 3.5
    },
    "+:^model\\.layers\\.[0-9]+\\.(self_attn.o_proj)$": {
      "bits": 4,
      "format": "qvq"
    },
    "+:^model\\.layers\\.[0-9]+\\.(mlp.gate_proj|mlp.down_proj)$": {
      "bits": 3
    },
    "+:^model\\.layers\\.[0-9]+\\.(mlp.up_proj)$": {
      "bits": 3.5
    }
  }
}
```

Resolved defaults saved in the checkpoint include:

```text
group_size:                         -1
lm_head:                            false
sym:                                true
pack_dtype:                         int32
codebook:                           pgc16-v1
trellis_window:                     16
vector_size:                        2
base bank_count:                    2
tile_rows / tile_cols:              16 / 16
incoherence:                        rht
viterbi_objective:                  euclidean
tail_biting_candidates:             1
viterbi_minimum_proxy_improvement:  0.0
viterbi pruning:                    auto / norm_band / exact / baseline fallback
module_scale_search:                false
output_channel_scale_optimization:  false
spectral_refinement:                false
YAQA chat template:                 disabled
YAQA full-model Fisher:             enabled
YAQA sequence loss reduction:       per_sequence_token_sum
```

The fully expanded saved configuration is the checkpoint's
`quantize_config.json`; that file, rather than defaults from a later checkout,
is authoritative for inspecting the serialized model.

## Dense source model

```text
Path: /monster/data/model/Llama-3.2-1B-Instruct
Architecture: LlamaForCausalLM
Layers: 16
Hidden size: 2048
Intermediate size: 8192
Dense dtype: bfloat16
```

File bindings:

| File | SHA-256 |
| --- | --- |
| `model.safetensors` | `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f` |
| `config.json` | `2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb` |
| `generation_config.json` | `88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36` |
| `tokenizer.json` | `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` |
| `tokenizer_config.json` | `9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e` |
| `special_tokens_map.json` | `6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec` |

## Quantization datasets and splits

### Lifecycle-forward stream

```text
Dataset name: NM calibration
Path: /monster/data/model/dataset/nm-calibration/llm.parquet
Split: train
Selection: row_start=0, rows=128
Selected rows: [0, 128)
File SHA-256: 26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef
```

For `rounding="yaqa"`, this is a lifecycle-forward stream. The actual QVQ
module input/output Fisher factors come from the YAQA stream below. No module
replay, replay-search, replay-confirmation, or validation dataset was used.

### YAQA full-model Fisher stream

```text
Dataset name: yaqa182_nm10000
Path: /root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet
Split: train
Selection: row_start=0, rows=10178
Independent sequences: 10,178
Valid Fisher/output token samples: 3,961,260
File SHA-256: 5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39
```

Immutable corpus manifest:

```text
/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.manifest.json
SHA-256 90f17c84200edc4cf25bd745729507c6e32b76f34622b3f44b34aabe8e10131e

Ordered calibration-manifest SHA-256:
330c1039a50d8a71b7a64255f4084187594471b25183feadeecfb0a6568e61d7
```

The corpus is the deduplicated union of:

| Source | Path and split | Selected rows | Raw valid tokens | Kept rows/tokens |
| --- | --- | ---: | ---: | ---: |
| YAQA optimized | `/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet`, `train` | 182, rows `[0,182)` | 302,193 | 182 / 302,193 |
| NM full | `/monster/data/model/dataset/nm-calibration/llm.parquet`, `train` | 10,000, rows `[0,10000)` | 3,661,493 | 9,996 / 3,659,067 |

Source-file bindings:

```text
YAQA optimized source SHA-256:
2140541facb66112428212b3a36d51a7735393b28c79db59c2429f6e51ed57ef

NM full source SHA-256:
26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef

Selected NM row-list SHA-256:
a74675bfed3834ac1c3140b2d7fb7d93d90a76608cc08515a5019a1cb2075ff4
```

Deduplication uses the normalized user-turn SHA-256 after NFKC, case folding,
and whitespace/punctuation normalization. Source precedence is `YAQA > NM`.
Four duplicate groups occur within NM, yielding `182 + 9,996 = 10,178`
independent sequences. The artifact's stored order is YAQA first and NM
second; YAQA capture subsequently uses `sequence_sort="desc"`.

### Disjointness binding

```text
Path: /root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.disjointness.json
SHA-256: f283eca649cbf1d2dcc160c202bc131c2462d4b3485b34c0d4bb5237d85a9d4e
Status: pass
Strict required: true
```

The manifest binds both calibration sources and the generated Fisher artifact
to GSM8K Platinum and both D300 manifests.

## Software and runtime provenance

Quantization:

```text
Git commit: 3ebcf9a307231187178e29ae7ad91e156a84d6ef
GPTQModel: 7.3.3+ultra-local-git-3ebcf9a3
Python: 3.14.7, GIL enabled
Torch: 2.13.0+cu130
Device: NVIDIA PG506-230, 96 GiB
TF32 during YAQA Fisher capture: false
YAQA Monte Carlo samples per output: 1
YAQA factor dtype: float32
Packed symmetric accumulators: false
```

Published GSM8K evaluation:

```text
Evalution: 0.0.14
GPTQModel: 7.3.3+ultra
Transformers: 5.15.1
Torch: 2.13.0
LogBar: 0.4.13
Inference dtype: float16
Attention: paged|sdpa
Continuous batching: required
Batch size: 8
```

Pin the historical commit when reproducing the checkpoint. Do not quantize
from a moving checkout: the Wave-10 audit demonstrated that quantizer code
drift changes discrete bank/trellis selections across the entire model.

## Hash preflight

Run before quantization:

```bash
sha256sum -c <<'EOF'
1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f  /monster/data/model/Llama-3.2-1B-Instruct/model.safetensors
2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb  /monster/data/model/Llama-3.2-1B-Instruct/config.json
79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4  /monster/data/model/Llama-3.2-1B-Instruct/tokenizer.json
9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e  /monster/data/model/Llama-3.2-1B-Instruct/tokenizer_config.json
26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef  /monster/data/model/dataset/nm-calibration/llm.parquet
2140541facb66112428212b3a36d51a7735393b28c79db59c2429f6e51ed57ef  /root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet
5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39  /root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet
90f17c84200edc4cf25bd745729507c6e32b76f34622b3f44b34aabe8e10131e  /root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.manifest.json
f283eca649cbf1d2dcc160c202bc131c2462d4b3485b34c0d4bb5237d85a9d4e  /root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.disjointness.json
EOF
```

## Exact quantization CLI

`run_in_worktree.py` is required because the development environment may have
an editable GPTQModel installation. Directly running the historical script can
otherwise import the current checkout and silently invalidate provenance.

```bash
set -euo pipefail

ROOT=/root/QvQ-score-updates
PINNED=3ebcf9a307231187178e29ae7ad91e156a84d6ef
WORKBASE="$(mktemp -d /tmp/qvq-f6-reproduction.XXXXXX)"
WORKTREE="$WORKBASE/source"
OUTPUT=/root/qvq-results/f6-yaqa182-nm10000-reproduction

git -C "$ROOT" worktree add --detach "$WORKTREE" "$PINNED"

env \
  PYTHONHASHSEED=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0 \
python "$ROOT/scripts/run_in_worktree.py" \
  --worktree "$WORKTREE" \
  --script scripts/qvq_quantize.py -- \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --output "$OUTPUT" \
  --quant-config "$WORKTREE/scripts/configs/llama32_1b_fisher_scaling_yaqa182_nm10000.json" \
  --calibration-dataset /monster/data/model/dataset/nm-calibration/llm.parquet \
  --calibration-dataset-split train \
  --calibration-row-start 0 \
  --calibration-rows 128 \
  --yaqa-dataset /root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet \
  --yaqa-dataset-split train \
  --yaqa-row-start 0 \
  --yaqa-rows 10178 \
  --batch-size 1 \
  --concat-size 0 \
  --calibration-sort desc \
  --device cuda:0 \
  --disjointness-manifest /root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.disjointness.json \
  --require-disjointness \
  --qvq-telemetry
```

For the cleanest scientific replication, use one quantizer on an otherwise
idle physical GPU even though the 1B model can fit multiple sessions.

## Exact GSM8K Platinum score CLI

Dataset and task:

```text
Hugging Face dataset: madrylab/gsm8k-platinum
Dataset config: main
Split: test
Rows: 1,209
Evalution task: gsm8k_platinum_cot
```

Task protocol from Evalution 0.0.14:

```text
Prompt: "Q: {question}\nA:"
Fixed few-shot examples: 8
Apply chat template: false
Few-shot seed: 0
Maximum new tokens: 256
Sampling: false
Temperature: 0.0
Stop strings: "Q:", "</s>", "<|im_end|>", "</assistant>"
Metric: format-insensitive numeric exact match (`acc,num`)
```

The published `547/1209` artifact used the following runtime exactly:

```bash
env \
  PYTHONHASHSEED=0 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0 \
python "$ROOT/scripts/run_in_worktree.py" \
  --worktree "$WORKTREE" \
  --script scripts/qvq_evaluate.py -- tasks \
  --checkpoint "$OUTPUT" \
  --output "$OUTPUT/post_quant_eval_gsm8k_platinum.json" \
  --task gsm8k_platinum_cot \
  --batch-size 8 \
  --device cuda:0 \
  --attn-implementation 'paged|sdpa'
```

The newer paged-FA2/CUDA-graph evaluator is intended to preserve the exact
metric at higher throughput, but it is not the provenance of the published
547 result. Use the command above when validating literal reproduction, then
run the fast evaluator as a parity-checked secondary report.

## Exact D300 diagnostic CLI

Dataset:

```text
Path: /root/qvq-data/divergence300-v1/divergence300-development.jsonl
SHA-256: 701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2
Prompts: 300
Prompt tokens: 889,765
Prompts at 16,384-token cap: 49
```

Protocol:

```text
Dense reference: the exact source model bound above
Exactly 32 independent greedy argmax steps
EOS is an ordinary token
Chat template enabled
Sampling disabled
Inference dtype float16
Attention implementation sdpa
```

```bash
env \
  PYTHONHASHSEED=0 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0 \
python "$ROOT/scripts/run_in_worktree.py" \
  --worktree "$WORKTREE" \
  --script scripts/qvq_evaluate.py -- divergence300 \
  --dense-model /monster/data/model/Llama-3.2-1B-Instruct \
  --checkpoint "$OUTPUT" \
  --dataset /root/qvq-data/divergence300-v1/divergence300-development.jsonl \
  --device cuda:0 \
  --output "$OUTPUT/post_quant_eval_divergence300.json" \
  --max-prompt-tokens 16384 \
  --dtype float16 \
  --attn-implementation sdpa
```

The locked D300 dataset was included in the contamination binding but was not
used to produce the score above:

```text
/root/qvq-data/divergence300-v1/divergence300-locked.jsonl
SHA-256 17151e98b2e34587c9af58a6736c875c82854564f45c763f027090f35b8e8f58
```

## Rebuild the Fisher corpus from its sources

The canonical artifact should be reused when reproducing the score. To audit
or rebuild it from its two immutable source Parquets, run the builder at the
pinned revision:

```bash
REBUILD=/root/qvq-data/calibration-fisher-scaling-v2-rebuild
mkdir -p "$REBUILD/configs"

python "$ROOT/scripts/run_in_worktree.py" \
  --worktree "$WORKTREE" \
  --script scripts/build_calibration_fisher_scaling.py -- \
  --nm /monster/data/model/dataset/nm-calibration/llm.parquet \
  --yaqa /root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet \
  --yaqa-rows 182 \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --base-config "$WORKTREE/scripts/configs/llama32_1b_frontier_w5_anchor_up4_l6_l8.json" \
  --config-dir "$REBUILD/configs" \
  --output-dir "$REBUILD" \
  --registry "$REBUILD/registry.json" \
  --d300 /root/qvq-data/divergence300-v1/divergence300-development.jsonl \
  --d300-locked /root/qvq-data/divergence300-v1/divergence300-locked.jsonl

sha256sum "$REBUILD/yaqa182_nm10000.parquet"
```

The rebuilt `yaqa182_nm10000.parquet` must produce:

```text
5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39
```

## Repository records

The authoritative tracked experiment records are:

```text
docs/experiments/calibration_fisher_scaling_v2_queue_20260830.json
docs/experiments/calibration_fisher_scaling_v2_registry_20260830.json
scripts/run_llama32_calibration_fisher_scaling_v2.sh
scripts/build_calibration_fisher_scaling.py
```

