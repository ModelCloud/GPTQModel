# Komodo NPU Kernel Notes

Date: 2026-04-29

Komodo is the experimental Ascend NPU int4 kernel path for GPTQ and AWQ. These
notes record the corrected benchmark state after disabling dense dequantized
weight caching by default and restricting Komodo inference to FP16.

## Current Policy

- Komodo inference is FP16-only for now.
- BF16 inference is intentionally rejected at selector validation and at runtime.
- Dense dequantized weight caching is off by default and should not be used for
  performance claims.
- Current performance numbers below compare Torch quantized kernels against
  Komodo native int4 prepack on the same synthetic packed weights with Qwen3.6
  projection sizes.

The BF16 gate is deliberate. Running AWQ through the native NPU int4 BF16 path
was either extremely slow when routed through the exact no-cache fallback, or
showed large drift when forced through native int4. Upcasting an FP16 native
result back to BF16 would hide an FP16 accuracy profile behind a BF16 interface,
so the safer behavior is to fail clearly until BF16 has a real fix.

## Benchmark Setup

Baseline: existing Torch quantized implementation. This host has no GPU capable
of running Marlin, so Marlin/Machete are references for implementation direction
only, not runtime baselines.

Mode: `native_int4_prepack`, no dense dequantized cache.

Command shape:

```bash
python scripts/benchmark_komodo_npu_ab.py \
  --cases qwen3_6_27b_all \
  --dtype fp16 \
  --device 0 \
  --shard-index 0 \
  --num-shards 8 \
  --warmup 2 \
  --iters 5 \
  --komodo-native-int4
```

The full benchmark was sharded over `npu:0` through `npu:7`.

## Qwen3.6 Shapes

| Model set | Projection | In features | Out features | dtype | tokens | group size |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B | q_proj | 5120 | 6144 | fp16 | 1 | 32 |
| Qwen3.6-27B | k_proj | 5120 | 1024 | fp16 | 1 | 32 |
| Qwen3.6-27B | v_proj | 5120 | 1024 | fp16 | 1 | 32 |
| Qwen3.6-27B | gate_proj | 5120 | 17408 | fp16 | 1 | 32 |
| Qwen3.6-27B | up_proj | 5120 | 17408 | fp16 | 1 | 32 |
| Qwen3.6-27B | down_proj | 17408 | 5120 | fp16 | 1 | 32 |
| Qwen3.6-35B-A3B | q_proj | 2048 | 4096 | fp16 | 1 | 128 |
| Qwen3.6-35B-A3B | k_proj | 2048 | 512 | fp16 | 1 | 128 |
| Qwen3.6-35B-A3B | v_proj | 2048 | 512 | fp16 | 1 | 128 |
| Qwen3.6-35B-A3B | gate_proj | 2048 | 512 | fp16 | 1 | 128 |
| Qwen3.6-35B-A3B | up_proj | 2048 | 512 | fp16 | 1 | 128 |
| Qwen3.6-35B-A3B | down_proj | 512 | 2048 | fp16 | 1 | 128 |

## FP16 Summary

| Model | Method | Torch total s | Komodo total s | Speedup | Max abs drift | Max mean abs | Max rel | Min cosine | Path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B | GPTQ | 0.111584 | 0.000835 | 133.64x | 0.015625 | 0.003148 | 2.32143 | 0.999999046 | native int4 prepack |
| Qwen3.6-35B-A3B | AWQ | 0.120065 | 0.000893 | 134.46x | 1.0 | 0.137502 | 95.4507 | 0.999999166 | native int4 prepack |
| Qwen3.6-27B | GPTQ | 1.975834 | 0.001366 | 1446.18x | 0.0078125 | 0.00000238 | 0.002457 | 0.999997377 | native int4 prepack |
| Qwen3.6-27B | AWQ | 2.162875 | 0.001317 | 1642.18x | 2.0 | 0.000305 | 0.005181 | 0.999997795 | native int4 prepack |

Notes:

- Qwen3.6-35B-A3B AWQ has a high max relative error because some reference
  outputs are near zero. Mean absolute error and cosine are more useful for that
  row.
- Qwen3.6-27B AWQ looks much better than the smaller MoE expert projections
  despite a larger absolute max drift, because the output scale is much larger.

## BF16 Gate

Komodo BF16 inference is disabled for both GPTQ and AWQ.

Observed BF16 behavior before the gate:

| Model | Method | Path | Torch total s | Komodo total s | Speedup | Max abs drift | Max mean abs | Max rel | Min cosine |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | GPTQ | native int4 prepack | 0.112362 | 0.000851 | 132.05x | 0.125 | 0.014303 | 237.0 | 0.999995351 |
| Qwen3.6-35B-A3B | AWQ | no-cache exact fallback | 0.125424 | 0.126056 | 0.995x | 0 | 0 | 0 | 1.000000238 |
| Qwen3.6-27B | GPTQ | native int4 prepack | 1.984648 | 0.001378 | 1439.97x | 0.0625 | 0.00000359 | 0.030303 | 1.000000238 |
| Qwen3.6-27B | AWQ | no-cache exact fallback | 2.222606 | 2.225066 | 0.999x | 0 | 0 | 0 | 1.000000358 |

The exact BF16 fallback is correct but defeats the purpose of Komodo for LLM
inference because it is effectively Torch-speed. The native AWQ BF16 path had
large drift with AWQ's much larger scale range, so it is not enabled.

Runtime behavior now:

```text
RuntimeError: KomodoLinear currently supports only torch.float16 inference on NPU; got torch.bfloat16.
```

## Future Work

- Investigate why AWQ scale ranges are much larger than GPTQ scales in these
  synthetic packed benchmarks and why that amplifies native BF16 regressions.
- Determine whether the NPU `npu_weight_quant_batchmatmul` BF16 path has lower
  precision accumulation, scale/offset rounding behavior, or an unsupported
  antiquant layout for AWQ.
- Add a real-model layer extraction benchmark once local Qwen3.6 checkpoints are
  available, so scale distributions come from actual quantized layers instead of
  synthetic random packed weights.
- Re-enable Komodo BF16 only when native BF16 is both fast and passes drift
  thresholds without silently doing FP16 inference and upcasting the result.

## Reference Outputs

Benchmark result directories:

- `/tmp/komodo_qwen3_6_35b_a3b_all_fp16_only_gate`
- `/tmp/komodo_qwen3_6_27b_all_fp16_only_gate`
- `/tmp/komodo_qwen3_6_35b_a3b_all_bf16_awqfix`
- `/tmp/komodo_qwen3_6_27b_all_bf16_awqfix`

Relevant commits:

- `17d712a0 Restrict Komodo inference to fp16`
- `03d7fd8f Gate AWQ native int4 bf16 path`
- `f08a9b0a Disable Komodo dense cache by default`
- `39c47ca8 Add Qwen3.6 Komodo projection benchmarks`
