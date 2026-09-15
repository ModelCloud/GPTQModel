# Packed Marlin prefill/decode split

## Result

GPTQModel has a purpose-built large-M W4A16 Marlin kernel that consumes the
checkpoint's Marlin-packed INT4 weights and permuted group scales directly. It
does not construct, retain, or cache a dense FP16/BF16 weight matrix.

The automatic route is enabled by default and is intentionally conservative:

- the dispatch key is `(compute capability, SM count, dtype, logical K, logical N, M)`;
- the promoted table currently targets only 124-SM `sm_80` boards;
- the ordinary Marlin path remains authoritative for decode, padding tails,
  unsupported quantization contracts, unprofiled shapes, and other GPUs; and
- explicit configs `1..4` remain available for offline tuning, but automatic
  dispatch uses only configs 1 and 2 in the offline promotion table.

In particular, measurements from the local 124-SM boards are not applied to a
108-SM A100 merely because both report compute capability 8.0.

No environment variable is needed. To opt out before loading the model:

```bash
export GPTQMODEL_MARLIN_PACKED_PREFILL=0
```

Optional controls are:

```bash
export GPTQMODEL_MARLIN_PACKED_PREFILL_MIN_ROWS=1024
export GPTQMODEL_MARLIN_PACKED_PREFILL_CONFIG=0  # 0=auto, 1..4=manual tile
export GPTQMODEL_MARLIN_PACKED_PREFILL_STATS=1   # opt-in route counters
```

Route counters can be inspected and reset without parsing logs:

```python
from gptqmodel.nn_modules.qlinear.marlin import (
    get_marlin_packed_prefill_route_stats,
    reset_marlin_packed_prefill_route_stats,
)

reset_marlin_packed_prefill_route_stats()
model(**inputs)
print(get_marlin_packed_prefill_route_stats(reset=True))
```

The snapshot reports `auto_hits`, `auto_misses`, `auto_hit_rate`, miss reasons,
and counts grouped by hardware, dtype, M/K/N, and selected config. Statistics
are opt-in so the Python counter lock is absent from normal timed inference.

## Kernel design

Ordinary Marlin is optimized to keep tiny-M decode busy. It stripes work across
K and uses workspace locks/reduction when more than one CTA contributes to an
output tile. That scheduling is valuable when M does not expose enough output
tiles.

`MarlinPrefill` is a separately generated CUDA kernel family. For large M:

- one CTA owns one M/N output tile;
- that CTA traverses the complete K dimension;
- packed INT4 words are loaded and dequantized into registers immediately
  before tensor-core MMA;
- no dense weight is written to global memory;
- no cross-CTA K reduction or workspace lock is needed; and
- a partial final M tile is bounds checked.

Four launch shapes remain compiled for BF16 and FP16; this change does not
modify their math:

| Config | Threads | M tile | N tile | K stage |
|---:|---:|---:|---:|---:|
| 1 | 128 | 64 | 128 | 64 |
| 2 | 128 | 64 | 256 | 64 |
| 3 | 256 | 32 | 512 | 64 |
| 4 | 64 | 64 | 128 | 64 |

The native contract is full-K symmetric GPTQ U4B8, group size 128, no
activation order, and no zero point. The Python policy additionally requires
logical K/N to need no padding before selecting an automatic route. Native
launch-geometry and shared-memory checks remain the final safety gate.

## Real projection inventory

The offline scan reads the model `config.json` semantics rather than assuming
that every family has Llama dimensions. Counts below are for TP=1.

| Model | K | N | Count | Roles |
|:---|---:|---:|---:|:---|
| Llama 8B | 4096 | 1024 | 64 | k/v |
| Llama 8B | 4096 | 4096 | 64 | q/o |
| Llama 8B | 4096 | 14336 | 64 | gate/up |
| Llama 8B | 14336 | 4096 | 32 | down |
| Llama 70B | 8192 | 1024 | 160 | k/v |
| Llama 70B | 8192 | 8192 | 160 | q/o |
| Llama 70B | 8192 | 28672 | 160 | gate/up |
| Llama 70B | 28672 | 8192 | 80 | down |
| Qwen3 8B | 4096 | 1024 | 72 | k/v |
| Qwen3 8B | 4096 | 4096 | 72 | q/o |
| Qwen3 8B | 4096 | 12288 | 72 | gate/up |
| Qwen3 8B | 12288 | 4096 | 36 | down |
| Qwen3 32B | 5120 | 1024 | 128 | k/v |
| Qwen3 32B | 5120 | 8192 | 64 | q |
| Qwen3 32B | 8192 | 5120 | 64 | o |
| Qwen3 32B | 5120 | 25600 | 128 | gate/up |
| Qwen3 32B | 25600 | 5120 | 64 | down |

The N=1024 projections were measured but not promoted: their packed configs
were unstable or slower over the M sweep.

## Automatic route table

All rows below also require `(major, minor, SMs) = (8, 0, 124)`. Only the
requalified Llama-3.2-1B points remain exact; the larger-model entries are M
intervals. Bounds are inclusive.

### FP16

| K | N | M range | Config |
|---:|---:|:---|---:|
| 2048 | 8192 | 1024 | 1 |
| 2048 | 8192 | 2048 | 2 |
| 4096 | 4096 | 4097-8192 | 2 |
| 4096 | 12288 | 1025-8192 | 2 |
| 4096 | 14336 | 1025-8192 | 2 |
| 12288 | 4096 | 8000-8192 | 2 |
| 14336 | 4096 | 6144-8192 | 2 |
| 5120 | 8192 | 2049-8192 | 2 |
| 5120 | 25600 | 1024-8192 | 2 |
| 8192 | 5120 | 4096-8192 | 2 |
| 25600 | 5120 | 4097-8192 | 2 |
| 8192 | 8192 | 3072-8192 | 2 |
| 28672 | 8192 | 3072-8192 | 2 |

### BF16

| K | N | M range | Config |
|---:|---:|:---|---:|
| 2048 | 8192 | 1024 | 1 |
| 2048 | 8192 | 2048 | 2 |
| 4096 | 4096 | 2049-4096 | 1 |
| 4096 | 4096 | 4097-8192 | 2 |
| 4096 | 12288 | 1025-8192 | 2 |
| 4096 | 14336 | 1025-8192 | 2 |
| 12288 | 4096 | 6144-8192 | 2 |
| 14336 | 4096 | 6144-8192 | 2 |
| 5120 | 8192 | 2049-8192 | 2 |
| 5120 | 25600 | 1024-8192 | 2 |
| 8192 | 5120 | 2049-4096 | 1 |
| 8192 | 5120 | 4097-8192 | 2 |
| 25600 | 5120 | 4097-8192 | 2 |
| 8192 | 8192 | 2049-8192 | 2 |
| 28672 | 8192 | 2049-8192 | 2 |

The Python and CUDA copies of this offline table are compared exactly by a
unit test. This prevents route statistics from claiming a hit that differs
from the native config-0 selector.

The FP16 and BF16 `(K,N,M,config)=(2048,2048,2048,1)` points are also absent.
A fresh two-GPU high-sample rerun used seven paired rounds per GPU, 100 timed
iterations per round, and 50 warmups. FP16 produced a `1.0048x` one-sided 95%
lower bound and a `1.0000x` minimum raw result; BF16 produced `1.0234x` and
`1.0217x`. Both fail the promotion policy and now fall back to ordinary
Marlin.

The complete BF16 `(K,N)=(8192,28672)` interval is intentionally absent. A
two-GPU, five-round-per-GPU recheck gave `M=8192` a one-sided 95% lower bound
of only `1.0434x` and a minimum raw result of `1.0526x`, failing both promotion
paths. Earlier `M=3072` reruns also ranged from `1.046x` to `1.059x`, showing
the run-to-run sensitivity. Excluding only that M value would imply unsupported
continuity, so the shape stays on ordinary Marlin for every M until a repeated
discrete table or a fully revalidated interval clears the stricter gate below.

## Projection scan

The validation used two allocator-leased, initially idle GPUs:

| Property | GPU 0 | GPU 1 |
|:---|:---|:---|
| Name | NVIDIA PG506-230 | NVIDIA PG506-232 |
| Compute capability | 8.0 | 8.0 |
| SM count | 124 | 124 |
| Memory | 96 GiB | 96 GiB |
| Dtypes | FP16 and BF16 | FP16 and BF16 |

Software was Torch `2.13.0+cu130`, CUDA runtime `13.0`, and driver
`610.43.02`. The benchmark records the resolved Marlin C++/NVCC flags,
architecture environment, and source revision alongside the GPU metadata. It
uses preallocated outputs, warmup, CUDA events, an alternating launch order,
and 60 samples per point (3 rounds x 20 iterations).

The resolved build used `-O3 -std=c++17 -DENABLE_BF16` and
`-D_GLIBCXX_USE_CXX11_ABI=1`; NVCC additionally used `--threads 8`,
`--optimize=3`, `-Xptxas -O3,-dlcm=ca`, `-lineinfo`, fatbin compression, and
the project's diagnostic suppressions. Architecture flags emitted both PTX
and SASS for compute capability 8.0.

All four compiled configs were scanned on both GPUs at
`M={512,513,1024,1025,2048,2049,4096,4097,8191,8192}`. Configs 1 and 2 were
also scanned at interval-interior points
`M={768,1000,1023,1536,2000,3072,4000,6144,8000}`. The repeat produced 3,400
boundary/config timings and 1,836 interval-interior timings. After withdrawing
the two slow square points, all 190 sampled points covered by the current 28
routes pass the cross-GPU promotion gate. Their minimum per-GPU geometric-mean
speedups range from `1.0812x` to `1.2458x`. These are projection results, not
end-to-end estimates.

### Promotion gate

New or re-promoted points must be measured on at least two distinct physical
GPU UUIDs with at least three paired rounds per GPU. Results are grouped by
compute capability, SM count, dtype, M, K, N, and config; unlike hardware is
never pooled. A point passes only when correctness is finite and either:

- the one-sided 95% lower confidence bound of balanced, paired, log speedups is
  at least `1.05x`, while every GPU's paired geometric mean is also at least
  `1.05x`; or
- every raw artifact is at least `1.07x`, preserving a margin for run-to-run
  noise.

An interval may be published only if its boundaries, interior samples, and M
tails pass independently. Failing one point withdraws the interval unless an
explicitly measured discrete table is used instead.

### Previous 1B baseline

The earlier Llama-3.2-1B report is retained as the motivation for widening the
route table. Its selected `K=2048, N=8192` projection improved by 1.075x at
`M=1024` and 1.180x at `M=2048`, while end-to-end prefill improved by only
1.008x and 1.030x respectively. The expanded route table targets the resulting
projection-coverage bottleneck; it does not change the packed kernel math.

## End-to-end prefill and route coverage

Four real W4G128, symmetric, `desc_act=False` checkpoints were tested in BF16
with eager attention. Each row is 15 alternating A/B trials after two warmups.
The paired ratio is ordinary Marlin time divided by packed-prefill time.

| Model | Prompt M | Auto hits / projections | Paired prefill | Feature wins | Decode paired | Token IDs |
|:---|---:|---:|---:|---:|---:|:---|
| Llama 3.1 8B | 2048 | 64 / 224 | 1.0416x | 15/15 | 1.0021x | equal |
| Llama 3.1 8B | 2049 | 128 / 224 | 1.0413x | 15/15 | 0.9874x | equal |
| Llama 3.1 8B | 3072 | 128 / 224 | 1.0444x | 15/15 | 1.0065x | equal |
| Qwen3 8B | 2048 | 72 / 252 | 1.0369x | 15/15 | 0.9928x | equal |
| Qwen3 8B | 2049 | 144 / 252 | 1.0358x | 15/15 | 0.9987x | equal |
| Qwen3 8B | 3072 | 144 / 252 | 1.0392x | 15/15 | 0.9960x | equal |
| Qwen3 32B | 2048 | 128 / 448 | 1.0481x | 15/15 | 0.9923x | equal |
| Qwen3 32B | 2049 | 256 / 448 | 1.0285x | 15/15 | 1.0024x | equal |
| Qwen3 32B | 3072 | 256 / 448 | 1.0502x | 15/15 | 1.0083x | equal |
| Llama 70B | 2048 | 0 / 560 | 1.0002x | 9/15 | 1.0024x | equal |
| Llama 70B | 2049 | 240 / 560 | 1.0341x | 15/15 | 1.0075x | equal |
| Llama 70B | 3072 | 240 / 560 | 1.0351x | 15/15 | 0.9992x | equal |

Llama 8B and Qwen3 8B exceed the 3% end-to-end target at every measured M.
Qwen3 32B falls below it only at `M=2049`. Llama 70B has no promoted route at
`M=2048`, so that row remains effectively unchanged; its `M=2049` and `M=3072`
runs exceed 3%. Every untimed M=1 route probe reported `decode` and used
ordinary Marlin, so the short decode timing spread above compares identical
native routes.

### Qwen3 32B compatible checkpoint

The public
[`JunHowie/Qwen3-32B-GPTQ-Int4`](https://huggingface.co/JunHowie/Qwen3-32B-GPTQ-Int4)
checkpoint, with `bits=4`, `group_size=128`, `sym=true`, and `desc_act=false`,
was loaded end to end in BF16. GPTQModel selected 448 `MarlinLinear`
projections; no synthetic model shell or config-only substitute was used. Its
results are included in the table above.

A separate 15-pair, 32-token decode run measured `1.0079x`; all 14,336
projection calls reported `decode`, packed-prefill hits stayed at zero, and all
generated token IDs matched. This validates checkpoint compatibility and route
coverage on 124-SM `sm_80`; it does not supply 108-SM A100 evidence.

## Correctness and fallbacks

- Every projection/config/M result was finite. Worst packed-versus-ordinary
  differences in the full sweep were `0.00390625` for FP16 and `0.03125` for
  BF16.
- The full boundary scan produced 68 dense-reference records across the two
  GPUs. Worst max absolute error against FP32 dequantization was `0.002316` for
  FP16 and `0.016973` for BF16.
- K/N padding tails used logical `K=N=288` (runtime padded to `384x320`) at
  `M={64,65,513}`, two seeds, and both dtypes. All 12 forced-packed cases
  matched ordinary Marlin and the dense reference. Automatic mode reported
  `contract_miss` and stayed on ordinary Marlin.
- Unit tests cover route boundaries, table overlap, dtype misses, sm_90 and
  sm_86 misses, and the critical `(8,0,108)` selector miss. This is not a
  substitute for a real 108-SM A100 run.
- M=1 is rejected before table lookup even when the configured minimum is one.

## Current vLLM comparison

The upstream implementation was rechecked on vLLM `main` on 2026-08-20. It
has generic small/large-batch Marlin thread configurations, a generated kernel
selector, and an M-splitting loop. It does not currently contain GPTQModel's
separate `MarlinPrefill` family, `use_packed_prefill` switch, or a dtype/SM/K/N/M
promotion table. In other words, upstream has continued to improve general
Marlin, but it does not provide a ready-made equivalent of this packed-prefill
route to copy. See vLLM's
[`marlin.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/marlin.cu)
and
[`generate_kernels.py`](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/generate_kernels.py).

## Reproduce

The matrix scanner derives projection shapes directly from the four model
configs, performs the strict physical-GPU idle preflight before importing
Torch, records hardware/build metadata, and writes an auditable JSON artifact:

```bash
python -m gpu_allocator.cli run -n 1 -t 3600 \
  --reason marlin-packed-prefill-matrix -- \
  env PYTHONPATH=. python scripts/benchmark_marlin_packed_prefill_matrix.py \
  --model-config /path/to/model-a/config.json \
  --model-config /path/to/model-b/config.json \
  --model-config /path/to/model-c/config.json \
  --model-config /path/to/model-d/config.json \
  --dtype both \
  --m-values 512,513,1024,1025,2048,2049,4096,4097,8191,8192 \
  --configs 1,2,3,4 \
  --warmup 10 --iters 20 --rounds 3 \
  --json-out /tmp/marlin_prefill_matrix.json
```

Repeat the selector candidates at the sampled interval interiors used for
promotion:

```bash
python -m gpu_allocator.cli run -n 1 -t 3600 \
  --reason marlin-packed-prefill-interiors -- \
  env PYTHONPATH=. python scripts/benchmark_marlin_packed_prefill_matrix.py \
  --model-config /path/to/model-a/config.json \
  --model-config /path/to/model-b/config.json \
  --model-config /path/to/model-c/config.json \
  --model-config /path/to/model-d/config.json \
  --dtype both \
  --m-values 768,1000,1023,1536,2000,3072,4000,6144,8000 \
  --configs 1,2 \
  --dense-reference-m 0 \
  --warmup 10 --iters 20 --rounds 3 \
  --json-out /tmp/marlin_prefill_interiors.json
```

Run the same matrix and interior scans on a second physical GPU, keeping dtype,
shapes, build, and sampling parameters identical. Then apply the fail-closed
offline gate to both JSON artifacts:

```bash
python scripts/analyze_marlin_packed_prefill_promotion.py \
  /tmp/gpu-a-matrix.json /tmp/gpu-b-matrix.json \
  --min-gpus 2 --min-rounds-per-gpu 3 \
  --confidence-speedup 1.05 --raw-speedup 1.07 \
  --json-out /tmp/marlin_prefill_promotion.json
```

The scanner records the physical UUID and each round's median; the analyzer
balances round counts across UUIDs so repeating one GPU cannot dominate the
confidence calculation.

The end-to-end script now applies the same strict idle gate and records a
separate, untimed prefill/decode route-stat probe so counter overhead is not
included in the A/B timing:

```bash
python -m gpu_allocator.cli run -n 1 -t 3600 \
  --reason marlin-prefill-e2e -- \
  env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/benchmark_marlin_prefill_decode.py \
  --model /path/to/w4g128-checkpoint \
  --dtype bf16 \
  --prompt-lengths 2048,2049,3072 \
  --decode-tokens 4 \
  --model-warmup 2 --model-runs 15 \
  --skip-layer \
  --json-out /tmp/marlin_prefill_e2e.json
```

Do not replace the allocator lease or UUID-based idle gate with an assumed
fixed CUDA index; PCI order and device inventory can change.

## Rejected designs

- A persistent BF16 weight cache was rejected because it duplicated all
  quantized projections and added 1.8125 GiB even for the original 1B model.
- A conventional kernel that expanded a 128x128 INT4 tile into shared memory
  achieved only about 0.23x weighted Marlin throughput.
- Two experimental direct-grid tile shapes showed intermittent corruption and
  were removed before the production family was generated.
- N=1024 and smaller-M routes remain ordinary Marlin when the interval scan did
  not demonstrate a stable benefit.
