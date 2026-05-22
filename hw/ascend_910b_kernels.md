# Ascend 910B Kernel Benchmark Ledger

Last updated: 2026-05-04

This file is the hardware-specific benchmark ledger for the local Ascend 910B1
host. It lists every GPTQModel quantized linear kernel that can run on this
hardware, records the latest comparable performance data we have, and keeps an
append-only history so regressions are visible over time.

Use this as the first stop before making or evaluating an Ascend NPU kernel
change. The deeper implementation notes stay in `hw/komodo.md`, `hw/cannoe.md`,
and `hw/ascend_910b.md`.

## Test Policy

All local hardware tests for this ledger must use PCI bus ordering and only
physical NPUs 6 and 7:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=6 ...
CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=7 ...
```

Inside each process the visible device is addressed as logical `npu:0`. For a
two-process comparison, run one process with `ASCEND_RT_VISIBLE_DEVICES=6` and
one with `ASCEND_RT_VISIBLE_DEVICES=7`; do not use the other physical NPUs for
new benchmark claims in this repo.

Dense dequantized weight caching is not a valid Ascend NPU speedup mechanism.
Packed native plans may cache int4-packed weights, scales, offsets, group lists,
and small runtime metadata. They must not cache a full f16/f32 dequantized
weight matrix. The explicit dense-cache env is
`GPTQ_CACHE_DEQUANTIZED_WEIGHTS`; NPU paths must ignore it for persistent dense
weight storage.

## Applicable Kernels

| Kernel | Backend / selector | Quant method / format | NPU role | Persistent weight representation | Benchmark status |
|---|---|---|---|---|---|
| `TorchLinear` | `BACKEND.GPTQ_TORCH` | GPTQ / GPTQ_V2 | Quantization/runtime fallback; exact reference for GPTQ | Quantized source buffers; transient dequantized matmul | Correctness covered; no speed claims |
| `AwqTorchLinear` | `BACKEND.AWQ_TORCH` | AWQ GEMM | Quantization/runtime fallback; exact reference for AWQ | Quantized source buffers; transient dequantized matmul | Correctness covered; no speed claims |
| `KomodoLinear` | `BACKEND.GPTQ_KOMODO`, `BACKEND.KOMODO`, AUTO inference | GPTQ / GPTQ_V2 | Default GPTQ FP16 native int4 inference path | Packed int4 CANN plan plus scales/offsets; optional source drop | Current best GPTQ FP16 microbench |
| `AwqKomodoLinear` | `BACKEND.AWQ_KOMODO`, `BACKEND.KOMODO`, AUTO inference | AWQ GEMM | Default AWQ FP16 native int4 inference path | Packed int4 CANN plan plus scales/offsets; optional source drop | Current best AWQ FP16 microbench |
| `CannoeLinear` | `BACKEND.GPTQ_CANNOE`, `BACKEND.CANNOE` | GPTQ / GPTQ_V2 | CANN/Ascend C experiment; GPTQ FP16 and selected BF16 path | Packed int4 plan plus CANN tiling metadata | Best GPTQ BF16 path; FP16 still trails Komodo in quick sweeps |
| `AwqCannoeLinear` | `BACKEND.AWQ_CANNOE`, `BACKEND.CANNOE` | AWQ GEMM | CANN/Ascend C experiment for AWQ | Packed int4 plan plus CANN tiling metadata | Group-128 fused-bias win; group-32 remains gated |
| `ParoLinear` | `BACKEND.PAROQUANT_CUDA`, AUTO for Paro on NPU | ParoQuant | Rotates activations, then reuses AWQ/Komodo native int4 when eligible | Packed AWQ/Komodo native plan plus rotation buffers | Correctness/native-dispatch covered; standardized timing missing |
| `GGUFTorchLinear` | `BACKEND.GGUF_TORCH`, AUTO for GGUF on NPU | GGUF, including `q4_0` | Generic GGUF fallback; native q4_0 CANN int4 path for eligible FP16 shapes | q4_0 native path caches packed int4 plan; other qtypes use transient dequant/fused fallback | Correctness/native-dispatch covered; standardized timing missing |
| `QQQTorchLinear` | `BACKEND.QQQ_TORCH`, AUTO for QQQ on NPU | QQQ | Portable torch fallback on NPU | Quantized QQQ buffers; no persistent full runtime weight cache | Correctness covered; no speed claims |
| `ExllamaV3TorchLinear` | EXL3 torch runtime | EXL3 | Portable torch fallback on NPU | EXL3 buffers; transient dense reconstruction only | Correctness covered; no speed claims |

## Latest Snapshot

The latest comparable FP16 projection microbench data still comes from the
Komodo Qwen3.6 synthetic packed-weight suite. These are layer-level numbers,
not end-to-end generation throughput.

| Kernel | Case set | Dtype | Reference | Kernel total | Speedup / change | Accuracy note |
|---|---|---|---:|---:|---:|---|
| `KomodoLinear` | Qwen3.6-35B-A3B GPTQ projections | FP16 | Torch `0.111584s` | `0.000835s` | `133.64x` | max abs `0.015625`, min cosine `0.999999046` |
| `AwqKomodoLinear` | Qwen3.6-35B-A3B AWQ projections | FP16 | Torch `0.120065s` | `0.000893s` | `134.46x` | max abs `1.0`, min cosine `0.999999166` |
| `KomodoLinear` | Qwen3.6-27B GPTQ projections | FP16 | Torch `1.975834s` | `0.001366s` | `1446.18x` | max abs `0.0078125`, min cosine `0.999997377` |
| `AwqKomodoLinear` | Qwen3.6-27B AWQ projections | FP16 | Torch `2.162875s` | `0.001317s` | `1642.18x` | max abs `2.0`, min cosine `0.999997795` |
| `CannoeLinear` | Qwen3.6-27B GPTQ projections | BF16 | Torch reference | `1.360772ms` | `7.8%` faster than prior Cannoe BF16 bias path | max abs `0.5`; speedup vs Torch not recomputed after bias replacement |
| `CannoeLinear` | Qwen3.6-35B-A3B GPTQ projections | BF16 | Torch reference | `1.240929ms` | `16.2%` faster than prior Cannoe BF16 bias path | max abs `0.25`; speedup vs Torch not recomputed after bias replacement |
| `AwqCannoeLinear` | Qwen3.6-35B-A3B AWQ projections | FP16 | prior unfused Cannoe bias | `0.842399ms` mean | `0.8338x` paired new/old, faster | max abs `1.0` |
| `GGUFTorchLinear` q4_0 native | Synthetic q4_0 NPU test shape | FP16/BF16 correctness only | CPU torch | n/a | n/a | native q4_0 path asserted; timing not standardized |
| `ParoLinear` native | Synthetic ParoQuant NPU test shape | FP16 correctness only | CPU torch | n/a | n/a | Komodo native path asserted after rotation; timing not standardized |
| `QQQTorchLinear` | Synthetic QQQ NPU test shape | FP16/BF16 correctness only | CPU torch | n/a | n/a | no persistent full runtime weight cache |
| `ExllamaV3TorchLinear` | Synthetic EXL3 NPU test shape | FP16 correctness only | CPU torch | n/a | n/a | no persistent full runtime weight cache |

## Historical Performance Log

Append a row here for every benchmark-affecting kernel change. Keep the command
or result path so a later developer can reproduce the comparison.

| Date | Commit / change | Kernel(s) | Case set | Hardware / devices | Result | Regression read |
|---|---|---|---|---|---|---|
| 2026-04-29 | `39c47ca8` Qwen3.6 projection benchmark baseline | Komodo GPTQ/AWQ | Qwen3.6-27B and Qwen3.6-35B-A3B FP16 projections | local 910B1 | GPTQ `133.64x` to `1446.18x`; AWQ `134.46x` to `1642.18x` vs Torch | Strong native int4 win; becomes FP16 baseline |
| 2026-04-29 | `f08a9b0a` disable dense cache by default | Komodo fallback | Quick GPTQ+AWQ | local 910B1 | exact fallback `0.997x` vs Torch; native quick `63.852x` | Dense cache not valid for perf claims; native int4 is the speed path |
| 2026-04-29 | prefetch comparison | Komodo GPTQ/AWQ | Quick GPTQ+AWQ | NPU0-NPU3 in older sweep | native `63.852x`; native+prefetch `65.040x` | Small steady-state gain; first-forward pack latency improves |
| 2026-04-29 | source-drop validation | Komodo GPTQ/AWQ | Quick GPTQ+AWQ | local 910B1 | source drop `73.983x`; prefetch+drop `69.344x` | Source drop reduces resident source buffers; no math-path change |
| 2026-04-30 | profiler-guided host-path patch | Cannoe GPTQ | GPTQ group sizes and Qwen3.6-27B GPTQ | 8-NPU older sweep | Cannoe mostly slower than Komodo; best Qwen source-drop `1.2861ms` vs Komodo `1.2979ms` | Host prefetch is not enough; real fused Ascend C op still needed |
| 2026-04-30 | V3 bridge workspace cache | Cannoe GPTQ | q-like and balanced probes | local 910B1 | workspace cache improved q/k/down/balanced probes, but 8-NPU bridge still slower than native CANN | V3 API correctness confirmed; generic ACLNN boundary remains cost |
| 2026-05-03 | AWQ fused-bias gate | Cannoe AWQ | Qwen3.6-35B-A3B AWQ and Qwen3.6-27B AWQ | physical NPUs 0,1 in older sweep | group-128 fused bias `0.842399ms`, paired `0.8338`; group-32 left unfused | Fused bias is shape-sensitive; group-32 drift prevents enabling |
| 2026-05-04 | GPTQ BF16 Cannoe enablement | Cannoe GPTQ | Qwen3.6-27B and Qwen3.6-35B-A3B GPTQ BF16 | physical NPUs 0,1 in older sweep | `1.475314ms` at `91.65x`; `1.480676ms` at `4.77x`; FP16 group sweep `1.527064ms` mean at `3.50x` | BF16 GPTQ acceptable by casting native call to FP16 and returning BF16 |
| 2026-05-04 | BF16 bias replacement | Cannoe GPTQ | Qwen3.6 GPTQ BF16 | physical NPUs 0,1 in older sweep | Qwen27B `1.475314 -> 1.360772ms`; Qwen35B `1.480676 -> 1.240929ms` | Faster without a second resident bias cache |
| 2026-05-04 | `5aa487f5` native NPU int4 paths | ParoQuant, GGUF q4_0 | Synthetic NPU dispatch tests | local 910B1 | native path asserted for ParoQuant after rotation and GGUF q4_0 | Needs standardized timing rows before perf claims |
| 2026-05-04 | `eea92f03` dense runtime caches removed | QQQ, EXL3, Komodo fallback, Torch fallback | Targeted NPU tests | physical NPUs 2,3 in older sweep | 10-test cache/correctness subset passed on each device | Correctness/policy fix; no speedup claimed |
| 2026-05-04 | `f748485d` explicit dense-cache env rename | Torch fallback and benchmark scripts | Dense-cache guard test | physical NPUs 2,3 in older sweep | renamed to `GPTQ_CACHE_DEQUANTIZED_WEIGHTS`; one NPU regression passed on each device | Naming fix; no perf change |

## Benchmark Matrix To Keep Current

Run these after any NPU kernel change and copy the result rows into this file.
Use physical NPUs 6 and 7 only.

### Komodo And Cannoe FP16

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=6 \
python scripts/benchmark_komodo_npu_ab.py \
  --cases quick --dtype fp16 --device 0 --warmup 2 --iters 5

CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=7 \
python scripts/benchmark_komodo_npu_ab.py \
  --cases quick --dtype fp16 --device 0 --warmup 2 --iters 5 --cannoe
```

For Qwen projection claims, run the relevant `qwen3_6_*` case set on both
physical devices and report mean, min/max, and paired change versus the previous
ledger row. For every Cannoe/Cannoe Ascend C kernel change, including narrow
micro-optimizations, also run the full Qwen3 27B GPTQ FP16 projection gate so
`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, and `down_proj` are all
checked for end-to-end module regression:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=0 \
python scripts/benchmark_qwen3_27b_gptq_fp16.py \
  --path cannoe --device npu:0 --tokens 1 --warmup 10 --iters 50 \
  --json-output /tmp/cannoe_qwen3_27b_full_gate.json
```

2026-05-22 CANN 9.1 beta down-projection prepack retune update: the 2026-05-19
`prepack_tile_n=320` rule is no longer used by the bound plain-native Cannoe
path. A fresh Qwen3 27B down-only sweep measured `tile_n=320` at `0.2852 ms`,
while `tile_n=1024` measured `0.2772 ms` and `tile_n=1536` measured
`0.2768 ms`, all at `239.2 MB` peak. `tile_n=4096` was similar speed
(`0.2773 ms`) but raised peak memory to `355.4 MB`. Plain-native Cannoe now
inherits the parent `1024` tile for this shape; the old `320` rule remains only
for non-plain planned/fused experiments until that path has a validated runtime
winner.

### Dense-Cache Guard

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=6 \
GPTQMODEL_TEST_NPU_DEVICE=npu:0 PYTHONWARNINGS=ignore \
pytest -q tests/test_npu_support.py::test_npu_komodo_gptq_group16_does_not_cache_dense_fallback

CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=7 \
GPTQMODEL_TEST_NPU_DEVICE=npu:0 PYTHONWARNINGS=ignore \
pytest -q tests/test_npu_support.py::test_npu_komodo_gptq_group16_does_not_cache_dense_fallback
```

### Correctness Coverage For Kernels Without Timing Rows

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=6 \
GPTQMODEL_TEST_NPU_DEVICE=npu:0 PYTHONWARNINGS=ignore \
pytest -q \
  tests/test_npu_support.py::test_npu_paroquant_uses_komodo_native_int4_after_rotation \
  tests/test_npu_support.py::test_npu_torch_gguf_q4_0_uses_native_int4 \
  tests/test_npu_support.py::test_npu_torch_qqq_does_not_cache_full_runtime_weight \
  tests/test_npu_support.py::test_npu_exllamav3_torch_does_not_cache_full_runtime_weight
```

Mirror the same command with `ASCEND_RT_VISIBLE_DEVICES=7` before marking a
kernel "covered" for the current change.

## Update Rules

- Latest snapshot rows should be overwritten only when a newer run uses the same
  case set, dtype, warmup/iters, and source-drop/cache policy.
- Historical rows are append-only.
- If a result changes by more than 5%, record it as a regression or improvement
  and include the prior row it compares against.
- Never mix old physical-device runs with new claims. New claims in this repo
  must use PCI bus order and physical NPUs 6 and 7.
- If a kernel only has correctness coverage, say so explicitly. Do not imply a
  speedup until the benchmark matrix has a timing row for that kernel.
