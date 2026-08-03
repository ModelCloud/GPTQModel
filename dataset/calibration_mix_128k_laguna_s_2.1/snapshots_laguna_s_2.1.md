# Laguna-S-2.1 quantized snapshot paths

Both snapshots were produced with `optimize/quantize_and_eval.py` using the same
calibration data: `dataset/calibration_mix_128k_laguna_s_2.1/calibration.parquet`
(41 rows / 59,075 tokens, MoE routing bypass coverage mix).

## Snapshots

| snapshot | path |
|---|---|
| W4G64 base quant | `/monster/data/model/Laguna-S-2.1-GPTQ-W4G64-CovMix` |
| W4G64 + embed/lm_head W8G128 requant | `/monster/data/model/Laguna-S-2.1-GPTQ-W4G64-CovMix_embed_lmhead_w8g128` |

## Recipes

W4G64 base quant (GPUs 4-7, GIL=0, `TORCHDYNAMO_DISABLE=1`, `CUDA_LAUNCH_BLOCKING=1`):

- GPTQ, bits=4, group_size=64, sym=true
- desc_act=false, act_group_aware=true, scale_search=activation
- MoE `ExpertsRoutingBypass`, dense/MoE `VramStrategy.BALANCED`,
  `calibration_data_device="balanced"`

Post-quant embed/lm_head requant (GPU 7 only):

- `model.embed_tokens` + `lm_head`: bits=8, group_size=128, sym=true
- desc_act=false, act_group_aware=true, scale_search=activation
- All other tensors are streamed unchanged from the W4G64 checkpoint
  (embedding-replacement save; only the embed/lm_head shard is rewritten)

## Size comparison (du -sBM, allocated)

```
+---------------------------------------------------------------------+---------+
| snapshot                                                            | size    |
+---------------------------------------------------------------------+---------+
| Laguna-S-2.1-GPTQ-W4G64-CovMix                                      | 61630 M |
| Laguna-S-2.1-GPTQ-W4G64-CovMix_embed_lmhead_w8g128                  | 61052 M |
+---------------------------------------------------------------------+---------+
| delta                                                               |  -578 M |
+---------------------------------------------------------------------+---------+
```

The requant snapshot is ~578 MB (~0.56 GiB) smaller: the dense F16
`model.embed_tokens.weight` + `lm_head.weight` (~617 MB each) are replaced by
W8G128 packed tensors (qweight/qzeros/scales/g_idx, ~602 MB rewritten shard
total for both modules).

## Validation

- 49 shards, 146,071 tensors; per-expert quantized keys intact (spot-checked
  nonzero `model.layers.1.mlp.experts.0.down_proj.qweight`).
- `model.embed_tokens.qweight` I32 [25088, 3072], `lm_head` W8G128 present.
- `quantize_config.json` `dynamic` carries the embed/lm_head W8G128 overrides.
