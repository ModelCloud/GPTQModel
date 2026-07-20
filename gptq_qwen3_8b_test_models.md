# Qwen3-8B GPTQ ScaleSearch test models

This file records the reusable post-quantized Qwen3-8B checkpoints generated for the ScaleSearch quality sweep.
Use these snapshots for inference, kernel, and evaluation comparisons without repeating quantization.

## Source and quantization setup

- Native model: `/monster/data/model/Qwen3-8B`
- Snapshot root: `/monster/data/model/Qwen3-8B/gptq-scale-search-snapshots`
- Quantization: GPTQ W4, group size 128, symmetric, BF16 compute
- Calibration: 512 samples, 182,272 total tokens, sequence length 2048, 89 batches
- Quantizer build: GPT-QModel Ultra `7.2.0+ultra+0d8e54ed`
- Software: Transformers 5.14.1, Torch 2.12.0+cu130
- Hardware: NVIDIA PG506-230/232 Ampere `sm_80`, 96 GB
- Created: 2026-07-20

Each snapshot is approximately 5.7 GiB, contains two safetensor shards, and includes its tokenizer,
`quantize_config.json`, and `quant_log.csv`. All five checkpoints passed post-save reload and shard/index validation.
The complete snapshot set occupies approximately 29 GiB.

## Saved checkpoints

| Snapshot path | ScaleSearch policy |
|---|---|
| `/monster/data/model/Qwen3-8B/gptq-scale-search-snapshots/disabled` | Disabled globally |
| `/monster/data/model/Qwen3-8B/gptq-scale-search-snapshots/activation` | Activation globally |
| `/monster/data/model/Qwen3-8B/gptq-scale-search-snapshots/hessian` | Hessian globally |
| `/monster/data/model/Qwen3-8B/gptq-scale-search-snapshots/hybrid` | Hybrid globally |
| `/monster/data/model/Qwen3-8B/gptq-scale-search-snapshots/qkvo-activation-mlp-hessian` | Q/K/V/O activation; MLP gate/up/down Hessian |

The mixed checkpoint uses a global Hessian objective and this attention override:

```python
scale_search = ScaleSearchConfig.HESSIAN
dynamic = {
    r"+:^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$": {
        "scale_search": ScaleSearchConfig.ACTIVATION,
    },
}
```

Qwen3-8B quantizes exactly these seven projection types per layer, so the override assigns activation to every
attention projection and leaves the three MLP projections on Hessian.

## Full-coverage quality results

The persisted files were regenerated in quantization-only mode from the same deterministic setup. The scores below
come from the preceding full post-quant evaluation sweep; evaluation was not repeated after moving to permanent
paths.

| Policy | ARC accuracy | ARC normalized | GSM8K Platinum | Mean | Mean vs disabled |
|---|---:|---:|---:|---:|---:|
| Disabled | 0.537543 | 0.541809 | 0.913151 | 0.664168 | - |
| All activation | 0.548635 | 0.550341 | 0.911497 | 0.670158 | +0.005990 / +0.902% |
| All Hessian | 0.543515 | 0.546075 | 0.913978 | 0.667856 | +0.003689 / +0.555% |
| All hybrid | 0.534983 | 0.530717 | 0.913978 | 0.659893 | -0.004275 / -0.644% |
| QKVO activation / MLP Hessian | 0.542662 | 0.544369 | 0.917287 | 0.668106 | +0.003938 / +0.593% |

## Findings and usage notes

- Activation is the general GPTQ ScaleSearch default and produced the highest three-metric mean in this sweep.
- The mixed policy is the strongest balanced Qwen-specific choice: all three metrics improved and GSM8K was best.
- All-Hessian improved every metric but trailed activation on mean and the mixed policy on GSM8K.
- Global hybrid regressed both ARC metrics and should not be used for Qwen3-8B based on this evaluation.
- Explicit `scale_search=None` remains the way to reproduce the pre-ScaleSearch disabled baseline.
- The complete experimental record, including projection-scoped results and timing, is in `gptq_scale_search.md`.
- Read the embedded `quantize_config.json` before reusing a snapshot; the mixed policy records its dynamic override.
