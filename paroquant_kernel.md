# ParoQuant Triton mega-kernel optimization log

## Goal and current status

This work targets ParoQuant W4A16 inference throughput on NVIDIA Ampere `sm_80`, for both single-token decode
and multi-token prefill. The retained first pass fuses the existing ParoQuant channel scaling, learned rotations,
AWQ INT4 dequantization, Tensor Core GEMM, and optional bias into one Triton launch.

The first pass is commit `d7384585` and is published in draft PR #39. Follow-up passes keep the same fused operation
while specializing decode partner width, decode warp count, and prefill row tiles only for measured dtype/shape
regimes.

The optimization is intentionally exposed through the explicit `BACKEND.PAROQUANT_TRITON` backend with priority
zero. It does not change automatic backend selection or the serialized ParoQuant checkpoint format. Unsupported
devices, shapes, dtypes, autograd calls, cold CUDA graph captures, and runtime launch failures retain the existing
CUDA AWQ or legacy Triton routes.

## Operation and fusion contract

For an input `x`, the fused path preserves this operation:

```text
x_scaled = x * channel_scales
x_rotated = learned_pairwise_rotations(x_scaled, pairs, theta)
out = awq_w4a16_gemm(x_rotated, qweight, qzeros, scales) + bias
```

The previous production path enqueued four serial kernels per module call:

```text
channel scale / rotation -> rotated activation temporary -> AWQ GEMM -> bias
```

The retained path builds immutable gather metadata once per module and performs all five phases inside one native
launch. Rotation results remain local to the Triton program and feed the dequantized `tl.dot` directly, so no
rotated activation is written to global memory. The standard schedule has no grid-wide dependency or persistent
workspace. The measured decode and FP16 prefill split-K schedules use FP32 partials and counters across their
CTAs, then complete the ordered reduction, bias, and output write inside the last CTA. Eager execution caches
stream-owned scratch; a warmed CUDA graph capture creates private graph-pool scratch and captures its initial
counter reset.

## Retained implementation

- `build_paroquant_rotation_lookup()` expands compact checkpoint `pairs` and `theta` tensors into global partner,
  cosine, and signed-sine gather tables. It verifies that each rotation round is a permutation of every 128-channel
  group. The module derives a local partner table for decode and split-K prefill while retaining global `int32`
  indices for regular prefill.
- `paroquant_rotation_gemm_kernel` applies channel scaling and either one or eight rotation rounds, unpacks and
  dequantizes group-128 AWQ INT4 weights, accumulates with Tensor Cores, and adds bias.
- `paroquant_rotation_gemm_splitk_kernel` assigns each of the 16 K=128 groups to an independent CTA for the
  measured decode and FP16 prefill projections. Each CTA stores an FP32 partial and atomically publishes
  completion; the last CTA reduces partials in fixed K order, adds bias, writes output, and resets the tile
  counter. Partial storage is compact per BMxBN output tile instead of reserving the full input row count for
  every tile. Scratch is cached by runtime device, CUDA stream, row count, and output width for eager calls. Each
  warmed graph capture allocates private scratch from its graph pool and records a counter reset before the split
  launch. Triton 3.7 reuses the compatible process-local compiled launcher after the first normal JIT call; all
  other Triton ABIs retain JIT dispatch. Warm contiguous decode calls pass the caller activation pointer and
  explicit row count directly to that compiled launcher, avoiding an otherwise-identical two-dimensional input
  view. Same-stream warm calls also reuse the last validated eager scratch entry without rebuilding its four-field
  dictionary key.
- Decode uses `BLOCK_N=128`, `BLOCK_K=128`, eight warps, and two stages. FP16 with `krot=8` uses `BLOCK_M=2` for
  M=1/2 and `BLOCK_M=4` for M=3-8; BF16 or `krot=1` uses `BLOCK_M=8`. BF16 `krot=8, K>=384` issues the first
  partner load while channel scaling is in flight and uses K-loop factor one; at K>=1024 it also issues the packed
  weight load before the rotation rounds. Shorter BF16 K retains factor two. FP16 and `krot=1` use local `int16`
  partners; BF16 with `krot=8` uses local `int8` partners.
- Prefill with `N<=512` defaults to `BLOCK_M=8`. Exact `K=2048, krot=8` FP16/BF16 uses `BLOCK_M=32` only
  at N=512/M=497-992, N=384/M=657-1312, N=256/M=993-1984, or N=128/M=1985-3968 on a runtime-probed
  124-SM device, where BM8 needs at least three CTA waves and BM32 needs one. All other device inventories and
  small-N shapes retain BM8.
- Wider FP16 `K=2048, krot=8` uses `BLOCK_M=32` only when the runtime SM count shows that BM16 needs exactly
  twice as many CTA waves as BM32. N=640-1920 is capped at one BM32 wave and N=2048 at four BM32 waves; the
  measured N=2176-4096 extension is restricted to its first BM32 wave on a 124-SM target. Other wider prefill uses
  `BLOCK_M=16`. On that measured 124-SM target, the retained wide BM32 tiles use 16 warps and one stage; other
  devices and prefill regimes retain eight warps and two stages. All prefill uses `BLOCK_N=128`, `BLOCK_K=128`,
  and global `int32` partners. Exact `K=2048, N=512, krot=8` unrolls the K-block loop by two for FP16 and four for
  BF16; other prefill retains factor one. BF16 `krot=8, K>=1024, N<=512` also issues the first partner and packed
  weight loads before completing the rotations. Only the N<=2048 BM32 tile rule is portable across runtime SM
  counts; its 16-warp schedule and the N=2176-4096 bands are restricted to the 124-SM target.
- Measured FP16 `K=2048, krot=8` prefill bands use split-K16 with `BLOCK_M=32`, `BLOCK_N=128`, four warps, and one
  stage on the runtime-probed 124-SM target. The retained row bands are M=9-992 at N=512, M=9-256 at
  N=1920/2048, M=97-192 at N=2560, M=81-160 at N=3072, and M=49-96 at N=4096. Other widths, rows, dtypes,
  K values, rotation counts, SM inventories, and devices retain regular prefill or the established selector
  fallback.
- An explicit rotation FMA is used for `krot=8, K>=1024` decode and for the same rotation/K regime at prefill
  `N<=512`. Wider prefill, shorter K, and `krot=1` keep the default lowering.
- FP16 and BF16 are supported. BF16 follows the existing CUDA rotation contract by using FP16 rotation
  intermediates before the BF16 dot product.
- Rotation metadata is lazy, non-persistent module state. The cache also supports tensors created under
  `torch.inference_mode()` where version counters are unavailable.
- The steady-state selector uses queued per-call CUDA events and caches the p50 winner per shape. Cold graph capture
  stays on the established route. A warmed split-K plan with a validated compiled launcher captures private
  scratch; different graph captures receive different allocations, while a cold or unsupported capture retains
  the standard mega-kernel.

The specialization gate is based on the input tensor's runtime device capability, never on a fixed CUDA index. The
current gate requires:

```text
architecture          sm_80
mode                  inference only
activation dtype      FP16 or BF16
layout                contiguous
quantization          W4A16, group size 128
rotation rounds       1 or 8
K                     divisible by 128 and <= 2048
N                     divisible by 8
prefill N             <= 4096; N>2048 only for measured FP16 first-wave bands on 124 SMs
split-K decode        FP16/BF16, K=2048, krot=8, M=1-8, measured widths, 124 SMs
split-K prefill       FP16, K=2048, krot=8, measured M/N bands, 124 SMs
split-K graph         warmed compatible Triton 3.7 compiled launcher; cold/unsupported capture falls back
```

Decode may use wider `N`; the measured 2048-to-8192 gate projection remains profitable. All failed eligibility
checks preserve the existing dispatch.

## Reproduction environment

The benchmark runner exposed one physical GPU to each process with `CUDA_VISIBLE_DEVICES`, so each worker used
logical `cuda:0`. Device properties were probed at runtime.

| Physical GPU | PCI bus | UUID | GPU | CC | SMs | Memory |
|---:|:---|:---|:---|:---:|---:|---:|
| 3 | `00000000:69:00.0` | `GPU-471ecdd7-a171-4d5c-d61f-a1802dc76e4c` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 4 | `00000000:A0:00.0` | `GPU-14ab23f1-a785-e9df-bbb5-215547154e3c` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 5 | `00000000:A5:00.0` | `GPU-3a4bf14f-fa28-df88-f6e8-00ef6b13d473` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 6 | `00000000:DE:00.0` | `GPU-737e2423-874a-23a4-1126-dfbe3e77c294` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 7 | `00000000:E4:00.0` | `GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |

```text
NVIDIA driver          610.43.02
PyTorch                2.13.0+cu130
CUDA runtime           13.0
Triton                 3.7.1
Quantization           symmetric W4A16, group size 128
Rotation rounds        8
Accumulation           FP32
Benchmark samples      100 warmups, 2,000 per-call CUDA-event samples
```

Representative benchmark shards were run concurrently on physical GPUs 3, 4, and 5:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/benchmark_paroquant_triton_ab.py \
  --device 0 --dtype fp16 --warmup 100 --iters 2000 --case-id decode_q_proj --case-id prefill_k_proj
CUDA_VISIBLE_DEVICES=4 python scripts/benchmark_paroquant_triton_ab.py \
  --device 0 --dtype fp16 --warmup 100 --iters 2000 --case-id decode_gate_proj
CUDA_VISIBLE_DEVICES=5 python scripts/benchmark_paroquant_triton_ab.py \
  --device 0 --dtype fp16 --warmup 100 --iters 2000 --case-id prefill_q_proj
```

The same sharding was repeated with `--dtype bf16`. Each process used an isolated Triton cache directory.

## Retained performance

Times below are full module calls measured with one CUDA event pair per sample after warmup. Throughput is module
input tokens per second, so prefill reports 128 tokens per call. Speedup is baseline mean divided by candidate mean.

### FP16

| Regime | Shape MxKxN | GPU | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tok/s |
|:---|:---|---:|---:|---:|---:|---:|
| Decode Q | 1x2048x2048 | 3 | 201.728 / 207.325 / 224.256 | 163.840 / 169.383 / 180.224 | 1.2240x | 4,823.4 -> 5,903.8 |
| Decode gate | 1x2048x8192 | 4 | 195.584 / 213.222 / 215.040 | 158.720 / 173.506 / 178.176 | 1.2289x | 4,690.0 -> 5,763.5 |
| Prefill Q | 128x2048x2048 | 5 | 197.632 / 202.081 / 214.016 | 158.720 / 163.841 / 184.320 | 1.2334x | 633,408.5 -> 781,247.6 |
| Prefill K | 128x2048x512 | 3 | 197.632 / 203.154 / 212.992 | 160.768 / 166.095 / 177.152 | 1.2231x | 630,062.6 -> 770,644.0 |

### BF16

| Regime | Shape MxKxN | GPU | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tok/s |
|:---|:---|---:|---:|---:|---:|---:|
| Decode Q | 1x2048x2048 | 3 | 185.344 / 187.567 / 199.680 | 156.672 / 161.051 / 173.056 | 1.1646x | 5,331.4 -> 6,209.2 |
| Decode gate | 1x2048x8192 | 4 | 193.536 / 205.958 / 209.920 | 164.864 / 174.918 / 182.272 | 1.1775x | 4,855.4 -> 5,717.0 |
| Prefill Q | 128x2048x2048 | 5 | 188.416 / 192.507 / 202.752 | 171.008 / 178.891 / 198.656 | 1.0761x | 664,909.5 -> 715,518.4 |
| Prefill K | 128x2048x512 | 3 | 187.392 / 193.031 / 204.800 | 159.744 / 163.217 / 176.128 | 1.1827x | 663,105.4 -> 784,232.5 |

The FP16 prefill-Q allocator peak falls from 5,242,880 bytes to 524,288 bytes because the 5 MiB rotated activation
temporary is eliminated. First-pass lazy lookup metadata costs 196,608 bytes for `K=2048, krot=8`. The decode
specialization adds 32,768 bytes for FP16 (`int16`) or 16,384 bytes for BF16 `krot=8` (`int8`), bringing current
metadata to 229,376 or 212,992 bytes respectively. Other measured baseline-to-candidate peak allocations are
40,960 to 4,096 bytes for decode-Q, 151,552 to 16,384 bytes for decode-gate, and 1,703,936 to 131,072 bytes for
prefill-K.

Numerical comparisons use the existing CUDA path plus a dense dequantized reference. Candidate-versus-baseline
mean absolute error is 0.00566-0.01347 for FP16 and 0.00003-0.01019 for BF16. The corresponding maximum absolute
differences are 0.5-1.0 for FP16 and 0.0625-8.0 for BF16; the large BF16 maxima occur at large-magnitude outputs while
the aggregate error remains low.

## Profiler attribution

Nsight Systems captured 50 warmed forwards per route:

| Route | Regime | Launches/forward | Dominant kernel median under profiler | CUDA launch API |
|:---|:---|---:|---:|:---|
| Existing | decode Q | 4 | GEMM 16.720 us; rotation 4.608 us; reduce 3.008 us; bias 1.888 us | 1.568 ms total, 7.843 us mean |
| Existing | prefill Q | 4 | GEMM 27.376 us; rotation 5.792 us; reduce 5.056 us; bias 5.216 us | 1.834 ms total |
| Mega | decode Q | 1 | 123.520 us | exactly 50 launches |
| Mega | prefill Q | 1 | 179.136 us | exactly 50 launches |

The fused raw kernel is longer than the old GEMM because every output CTA performs the learned input rotation. The
end-to-end route still wins by removing three launches, launch gaps, the global rotated-activation round trip, and
the large temporary allocation.

Nsight Compute classifies the retained source as latency/resource limited rather than DRAM limited:

| Regime | Grid/block | Registers | Shared memory | Waves/SM | Theoretical/achieved occupancy | Compute SOL | Memory SOL | DRAM SOL |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Decode Q | 16 / 128 | 128 | 34.816 KiB | 0.03 | 25.0% / 6.25% | 1.66% | 2.32% | 0.52% |
| Prefill Q | 128 / 256 | 104 | 37.888 KiB | 0.52 | 25.0% / 12.90% | 24.7% | 28.9% | 0.55% |

`cuobjdump --dump-resource-usage` reports zero local memory and zero stack for both specializations, so there are no
register spills. Decode has only 16 CTAs and is dominated by grid latency; prefill has about half a wave per SM and
is constrained by the combined register/shared-memory resource envelope.

## Follow-up kernel pass

The follow-up measurements use warmed CUDA graph replay to isolate the GPU specialization from Python validation,
plan selection, allocation, and multi-launch fallback behavior. Global-`int32` and local-`int16` decode kernels were
alternated in both orders for 4,000 samples each. Prefill configurations were interleaved for 2,000 samples each.
All compared outputs are bit-identical.

### Decode local partner indices

| Dtype/regime | GPU | Global int32 p50/mean/p95 (us) | Local int16 p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|---:|
| FP16 decode Q | 5 | 112.640 / 201.244 / 471.040 | 110.592 / 198.548 / 470.016 | 1.0185x | 1.0136x |
| BF16 decode Q | 5 | 113.664 / 183.248 / 470.016 | 111.616 / 179.210 / 467.968 | 1.0183x | 1.0225x |
| FP16 decode gate | 4 | 114.688 / 302.288 / 3024.896 | 112.640 / 295.817 / 3017.728 | 1.0182x | 1.0219x |
| BF16 decode gate | 4 | 115.712 / 186.046 / 484.352 | 113.664 / 180.909 / 480.256 | 1.0180x | 1.0284x |

GPU 3 and GPU 5 decode-Q repeats preserve the same 1.8-2.8% p50 improvement across both dtypes. One GPU 3 FP16
repeat had a noisy mean/p95 reversal under concurrent load; the other five repeat/dtype combinations improved mean,
and every comparison was bit-identical. Decode resource use remains 128 registers, 33,792 bytes of dynamic shared
memory, and zero local memory/spills.

### Small-N prefill row tile

For `M=128, K=2048, N=512`, changing `BLOCK_M` from 16 to 8 doubles the grid from 32 to 64 CTAs and lowers the
compiled FP16 register count from 128 to 106 while keeping eight warps and 128x128 K/N tiles.

| Dtype | GPU | BM16 p50/mean/p95 (us) | BM8 p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|---:|
| FP16 | 3 | 118.784 / 118.859 / 119.808 | 92.160 / 91.928 / 92.160 | 1.2889x | 1.2930x |
| BF16 | 4 | 113.664 / 171.201 / 204.800 | 93.184 / 138.001 / 128.000 | 1.2198x | 1.2406x |

GPU 5 repeats produced the same p50 values and improved mean by 1.42x FP16 and 1.36x BF16. The specialization is
restricted to `N<=512`: at `N=2048`, BM8 regressed p50 from 167.936 to 190.464 us FP16 and from 166.912 to
194.560 us BF16, so wide prefill retains BM16.

A final full-module selector audit ran concurrently on GPUs 3/4/5 with NUMA-local CPU pinning and the autotuner
enabled. Under that stressed host schedule the selector chose established CUDA AWQ or dense plans instead of the
mega-kernel. This is expected fallback behavior and is not used as incremental kernel timing evidence; it confirms
that an environment where the mega-kernel does not win is not forced onto the specialized route.

## GPU 7-only continuation

For the final continuation phase, GPU work was restricted to physical GPU 7 (`CUDA_VISIBLE_DEVICES=7`). The paired
configuration runner alternates CUDA-graph replays in both orders, performs eager and replay warmup, and requires
exact output equality before timing. Final retained comparisons use 4,000 samples per variant.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 128 --k 2048 --n 2048 --krot 8 --eager-warmup 50 --graph-warmup 50 --iters 4000 \
  --variant bm16:16:128:8:2:global32 --variant bm32:32:128:8:2:global32
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 --eager-warmup 50 --graph-warmup 50 --iters 4000 \
  --variant local16:4:128:4:2:local16 --variant local8:4:128:4:2:local8
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE=0 \
  python scripts/benchmark_paroquant_triton_ab.py --device 0 --dtype fp16 --warmup 100 --iters 2000 \
  --case-id decode_q_proj --case-id prefill_q_proj --case-id prefill_k_proj
```

### BF16 decode 8-bit local partners

Group-local partner offsets are bounded by 127. For BF16 decode with eight rotation rounds, storing those offsets as
`int8` instead of `int16` reduces the partner table from 32 to 16 KiB for `K=2048`. Input dtype is part of the lazy
metadata cache key, so FP16 and BF16 transitions rebuild the correct non-persistent table.

| Shape/dtype | int16 p50/mean/p95 (us) | int8 p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|
| `1x2048x2048`, BF16, `krot=8` | 110.592 / 114.187 / 129.024 | 109.568 / 113.047 / 126.976 | 1.0093x | 1.0101x |

The generated resource envelope is unchanged: grid 16, block 128, 128 registers/thread, 33,792 bytes dynamic shared
memory, 0.03 waves/SM, and about 6.25% active warps. NCU reports compute/DRAM SOL of 1.73%/0.51% for int16 and
1.75%/0.51% for int8, confirming that the path remains latency-limited rather than bandwidth-saturated.

### FP16 wide-prefill structural-tail tile

At `M=128, K=N=2048, krot=8`, BM16 launches 128 CTAs on the runtime-probed 124-SM device, leaving a four-CTA
tail. BM32 launches 64 longer CTAs and removes that tail. The specialization is deliberately exact-shape and FP16
only because neighboring row counts and BF16 do not benefit.

| Shape/dtype | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|
| `128x2048x2048`, FP16, `krot=8` | 167.936 / 171.047 / 194.560 | 152.576 / 156.063 / 177.152 | 1.1007x | 1.0960x |

NCU records BM16 as grid 128, block 256, 128 registers/thread, and 36,864 bytes dynamic shared memory. BM32 is grid
64, block 256, 160 registers/thread, and 40,960 bytes dynamic shared memory. Both remain latency/resource limited;
NCU replay duration is perturbed and does not reproduce the event-timed win, so it is used only for resources and
SOL classification.

### Decode warp-saturation pass

The next GPU 7-only pass revisited decode warp count after the retained local-partner specializations changed
Triton code generation. The configuration benchmark performs eager warmup, CUDA-graph replay warmup, alternating
AB/BA timing, and exact output comparison. The final single-token rows below are repeat-two results from two
independent 4,000-sample runs; the eight-row results are independent 4,000-sample comparisons.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 1 --k 2048 --n 2048 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm4_w4:4:128:4:2:local16 --variant bm4_w8:4:128:8:2:local16
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm4_w4:4:128:4:2:local8 --variant bm4_w8:4:128:8:2:local8 \
  --variant bm8_w8:8:128:8:2:local8
```

| Dtype/rows | Old BM4/W4 p50/mean/p95 (us) | Retained p50/mean/p95 (us) | Retained config | p50 speedup | mean speedup |
|:---|---:|---:|:---|---:|---:|
| FP16, M=1 | 110.592 / 113.804 / 128.000 | 95.232 / 97.557 / 109.568 | BM4/W8 | 1.1613x | 1.1665x |
| BF16, M=1 | 109.568 / 112.223 / 126.976 | 91.136 / 92.409 / 104.448 | BM8/W8 | 1.2022x | 1.2144x |
| FP16, M=8 | 112.640 / 115.653 / 130.048 | 96.256 / 99.162 / 111.616 | BM4/W8 | 1.1702x | 1.1663x |
| BF16, M=8 | 112.640 / 115.444 / 130.048 | 90.112 / 92.967 / 104.448 | BM8/W8 | 1.2500x | 1.2418x |

The eight-warp win repeated for `K=256/512/1024/2048` and `N=512/2048/8192`. For `krot=1`, BM8/W8 improved
single-token FP16 p50/mean by 1.1667x/1.1766x and BF16 by 1.1600x/1.1643x; M=8 also improved both dtypes. Those
cross-checks support the narrow dtype/rotation dispatch rule rather than one exact projection shape.

Nsight Compute first collected `SpeedOfLight`, then the latency-targeted `LaunchStats`, `Occupancy`,
`SchedulerStats`, and `WarpStateStats` sections. These are the actual SOL commands; report replay duration is
explanatory evidence rather than normal timing evidence.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. ncu --profile-from-start off --section SpeedOfLight --csv \
  --launch-count 1 --force-overwrite --export artifacts/paroquant_megakernel_20260723/gpu7_only/ncu_continue/bf16_old_w4_sol \
  python scripts/benchmark_paroquant_triton_configs.py --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 \
  --eager-warmup 20 --graph-warmup 10 --iters 10 --variant old:4:128:4:2:local8 \
  --profile-variant old --profile-launches 1
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. ncu --profile-from-start off --section SpeedOfLight --csv \
  --launch-count 1 --force-overwrite --export artifacts/paroquant_megakernel_20260723/gpu7_only/ncu_continue/bf16_new_w8_sol \
  python scripts/benchmark_paroquant_triton_configs.py --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 \
  --eager-warmup 20 --graph-warmup 10 --iters 10 --variant new:8:128:8:2:local8 \
  --profile-variant new --profile-launches 1
```

| NCU metric | Old BM4/W4 | Retained BM8/W8 |
|:---|---:|---:|
| SOL replay duration (us) | 189.15 | 136.77 |
| Compute / memory / DRAM throughput | 1.76% / 2.33% / 0.51% | 3.79% / 3.82% / 0.70% |
| Grid / block threads | 16 / 128 | 16 / 256 |
| Registers/thread | 128 | 105 |
| Dynamic shared memory/block | 33.79 KiB | 34.82 KiB |
| Waves/SM | 0.03 | 0.06 |
| Theoretical / achieved occupancy | 25.00% / 6.25% | 25.00% / 12.50% |
| Active / eligible / issued warps per scheduler | 1.00 / 0.14 / 0.14 | 2.00 / 0.42 / 0.30 |
| Scheduler cycles with no eligible warp | 86.25% | 70.01% |
| Warp cycles per issued instruction | 7.28 | 6.68 |

The low SOL values keep the classification latency-bound. The retained schedule lowers registers, doubles active
warps for the one-block-per-active-SM launch, and materially reduces scheduler starvation without changing the
16-CTA grid or numerical operation.

### K-loop overlap pass

Triton's `tl.range(..., loop_unroll_factor=...)` hint exposes adjacent K blocks to compiler scheduling without
changing accumulation order. The paired benchmark now accepts an optional seventh variant field for this factor.
Every comparison below completed exact output equality before timing. Final results are repeat-two measurements
with 4,000 CUDA-graph samples per variant.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant u1:8:128:8:2:local8:1 --variant u2:8:128:8:2:local8:2
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 128 --k 2048 --n 512 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant u1:8:128:8:2:global32:1 --variant u2:8:128:8:2:global32:2
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 128 --k 2048 --n 512 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant u1:8:128:8:2:global32:1 --variant u4:8:128:8:2:global32:4
```

| Dtype/regime | Factor 1 p50/mean/p95 (us) | Retained p50/mean/p95 (us) | Factor | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|---:|
| BF16 decode Q | 91.136 / 93.771 / 105.472 | 87.040 / 90.002 / 100.352 | 2 | 1.0471x | 1.0419x |
| FP16 prefill K | 92.160 / 95.193 / 106.496 | 90.112 / 93.148 / 104.448 | 2 | 1.0227x | 1.0220x |
| BF16 prefill K | 93.184 / 97.065 / 107.520 | 89.088 / 93.241 / 103.424 | 4 | 1.0460x | 1.0410x |

BF16 decode factor two also improved `K=256/512/1024`, `N=512/8192`, and `M=8`. The prefill factors repeated at
M=21/64/128. Production keeps exact gates because FP16 decode and `krot=1` regressed, BF16 `N=2048` prefill
regressed by about 21%, and the shortest FP16 K cases were flat.

The actual validation NCU command for retained BF16 decode is:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. ncu --profile-from-start off --section SpeedOfLight --csv \
  --launch-count 1 --force-overwrite \
  --export artifacts/paroquant_megakernel_20260723/gpu7_only/ncu_continue/bf16_unroll2_sol \
  python scripts/benchmark_paroquant_triton_configs.py --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 \
  --eager-warmup 20 --graph-warmup 10 --iters 10 --variant u2:8:128:8:2:local8:2 \
  --profile-variant u2 --profile-launches 1
```

| NCU metric | Factor 1 | Retained factor 2 |
|:---|---:|---:|
| SOL replay duration (us) | 136.77 | 132.35 |
| Compute / memory / DRAM throughput | 3.79% / 3.82% / 0.70% | 3.91% / 3.96% / 0.74% |
| Registers/thread | 105 | 128 |
| Dynamic shared memory/block | 34.82 KiB | 34.82 KiB |
| Achieved occupancy | 12.50% | 12.50% |
| Issued warps per scheduler | 0.30 | 0.31 |
| Scheduler cycles with no eligible warp | 70.01% | 69.15% |
| Warp cycles per issued instruction | 6.68 | 6.51 |

Unrolling raises the static register allocation, but the 16-CTA grid already places at most one block on each
active SM. Achieved occupancy is therefore unchanged while the compiler shortens dependency exposure and reduces
cycles per issued instruction.

### Final GPU 7 full-module throughput

The final module comparison forces the eligible mega-kernel plan, uses 100 warmups and 2,000 CUDA-event samples,
and compares against the production `ParoLinear` route on the same GPU.

| Dtype/regime | Shape | Existing mean (us) | Mega mean (us) | Mean speedup | Existing -> mega tokens/s |
|:---|:---|---:|---:|---:|---:|
| FP16 decode Q | `1x2048x2048` | 209.18 | 172.11 | 1.215x | 4,780.7 -> 5,810.1 |
| FP16 prefill Q | `128x2048x2048` | 206.68 | 172.30 | 1.200x | 619,312 -> 742,907 |
| FP16 prefill K | `128x2048x512` | 210.22 | 173.13 | 1.214x | 608,884 -> 739,315 |
| BF16 decode Q | `1x2048x2048` | 191.80 | 171.50 | 1.118x | 5,213.7 -> 5,830.8 |
| BF16 prefill Q | `128x2048x2048` | 193.04 | 172.83 | 1.117x | 663,070 -> 740,605 |
| BF16 prefill K | `128x2048x512` | 194.56 | 167.83 | 1.159x | 657,883 -> 762,667 |

### Two-row FP16 decode tile

The eight-warp FP16 `krot=8` path was retested with smaller row tiles after the warp and K-loop changes altered
Triton code generation. BM2 is retained only for one- and two-row decode; BM4 remains in place for M=3-8, while
BF16 and `krot=1` retain BM8. Every paired comparison required exact output equality before timing.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 1 --k 2048 --n 2048 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm4:4:128:8:2:local16:1 --variant bm2:2:128:8:2:local16:1
```

| Dtype/rows | BM4/BM8 p50/mean/p95 (us) | BM2 p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|
| FP16 M=1, repeat 1 | 95.232 / 97.961 / 110.592 | 93.184 / 96.310 / 108.544 | 1.0220x | 1.0171x |
| FP16 M=1, repeat 2 | 95.232 / 97.203 / 110.592 | 93.184 / 95.447 / 108.544 | 1.0220x | 1.0184x |
| FP16 M=2 | 95.232 / 99.798 / 110.592 | 93.184 / 97.902 / 108.544 | 1.0220x | 1.0194x |

The gain repeats at `K=1024/2048` and `N=512/2048/8192`; `K=256/512` is neutral at event resolution. M=3 is
also neutral, so the production boundary stops at M=2 instead of adding an extra M tile. BF16 BM2/BM4 and FP16
BM1 regress or remain slower and are not selected.

NCU classifies retained FP16 BM2 as latency-bound: grid 16, block 256, 108 registers/thread, 33.28 KiB dynamic
shared memory, 12.52% achieved occupancy, 0.37 eligible warps per scheduler, and 73.76% scheduler cycles with no
eligible warp. Long-scoreboard waits account for 46.6% of the average 7.66 warp cycles per issued instruction.

### Explicit rotation FMA pass

Triton's default lowering kept one multiply and one add separate in each rotation update. An explicit
`tl.fma(a, cos, paired * sin)` keeps the FP32 coefficient contract and produces bit-identical output while reducing
the instruction stream. The configuration benchmark accepts an eighth `EXPLICIT_FMA` variant field so both
lowerings can be compiled and timed in the same process.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 1 --k 2048 --n 2048 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant muladd:2:128:8:2:local16:1:0 --variant fma:2:128:8:2:local16:1:1
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. ncu --profile-from-start off --section InstructionStats \
  --launch-count 1 --force-overwrite \
  --export artifacts/paroquant_megakernel_20260723/gpu7_only/ncu_continue/fp16_bm2_w8_fma_instructions \
  python scripts/benchmark_paroquant_triton_configs.py --dtype fp16 --m 1 --k 2048 --n 2048 --krot 8 \
  --eager-warmup 20 --graph-warmup 10 --iters 10 --variant fma:2:128:8:2:local16:1:1 \
  --profile-variant fma --profile-launches 1
```

The specialization is retained for `krot=8`, `K>=1024` decode at every supported N, and for the same K/rotation
regime at prefill `N<=512`. It remains disabled for `krot=1`, `K<1024`, and prefill `N>=1024`, where paired medians
were neutral or the tail moved in the wrong direction.

| Dtype/regime | Mul+add p50/mean/p95 (us) | Explicit FMA p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|
| FP16 decode M=1, repeat 1 | 93.184 / 95.357 / 108.544 | 91.136 / 93.274 / 106.496 | 1.0225x | 1.0223x |
| FP16 decode M=1, repeat 2 | 93.184 / 96.218 / 108.544 | 92.160 / 94.686 / 106.496 | 1.0111x | 1.0162x |
| BF16 decode M=1 | 87.040 / 90.244 / 100.352 | 86.016 / 88.670 / 98.304 | 1.0119x | 1.0178x |
| FP16 decode M=8 | 96.256 / 98.814 / 111.616 | 92.160 / 94.632 / 107.520 | 1.0444x | 1.0442x |
| BF16 decode M=8 | 87.040 / 92.463 / 101.376 | 84.992 / 90.205 / 99.328 | 1.0241x | 1.0250x |
| FP16 prefill, `128x2048x512` | 103.424 / 97.877 / 104.448 | 100.352 / 95.502 / 101.376 | 1.0306x | 1.0249x |
| BF16 prefill, `128x2048x512` | 90.112 / 96.531 / 104.448 | 87.040 / 93.083 / 101.376 | 1.0353x | 1.0370x |

Decode FMA gains also repeat at `N=512/8192`. At `K=1024`, FP16/BF16 decode improve p50 by 1.8%/1.9%, and
small-N prefill improves by 1.9%/5.7%. BF16 `K=512` improves, but the common K gate remains 1024 because FP16 is
flat there and the shorter shapes have event-resolution tails. Both dtypes are neutral at prefill `N=1024/2048`.

Instruction Statistics reports 2,956,672 executed instructions for mul+add and 2,798,976 for explicit FMA, a 5.3%
reduction. The resource envelope remains 108 registers/thread and 33.28 KiB shared memory. A separate SOL replay
shortens from 147.39 to 145.38 us; paired CUDA-event measurements above remain the performance decision evidence.

### Latest full-module selection check

The post-BM2/FMA module check used 100 warmups and 2,000 samples on physical GPU 7. Host-side outliers make these
means noisier than the paired raw-kernel measurements, but all six workloads still select and win with the
mega-kernel. Accuracy remains within the previously recorded dense/existing tolerances.

| Dtype/regime | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| FP16 decode Q | 196.61 / 215.52 / 215.04 | 164.86 / 181.26 / 182.27 | 1.189x | 4,640 -> 5,517 |
| FP16 prefill Q | 195.58 / 213.92 / 211.97 | 165.89 / 188.31 / 185.34 | 1.136x | 598,352 -> 679,743 |
| FP16 prefill K | 197.63 / 234.09 / 224.26 | 167.94 / 194.40 / 223.23 | 1.204x | 546,807 -> 658,439 |
| BF16 decode Q | 188.42 / 207.00 / 206.85 | 164.86 / 177.55 / 183.30 | 1.166x | 4,831 -> 5,632 |
| BF16 prefill Q | 192.51 / 195.45 / 205.82 | 166.91 / 169.72 / 180.22 | 1.152x | 654,896 -> 754,202 |
| BF16 prefill K | 191.49 / 257.56 / 228.35 | 166.91 / 183.71 / 185.34 | 1.402x | 496,972 -> 696,750 |

### BF16 first-partner latency-hiding pass

Source correlation attributed the largest remaining decode stall to the first consumer of each rotation's partner
lookup. Prefetching an entire next rotation increased live state and was previously rejected. The retained narrower
mechanism moves only rotation zero's partner load ahead of channel scaling, so that one independent load can overlap
the scale load and multiply without changing any value or arithmetic order. The configuration benchmark accepts a
ninth `PREFETCH_FIRST` field for paired experiments.

The prefetch changes BF16 code generation enough that the old factor-two decode K-loop unroll is no longer useful.
Production therefore combines first-partner prefetch with factor one for BF16 `krot=8, K>=384`. K=256 stays on the
old schedule because repeated means reversed despite favorable event buckets. FP16 and `krot=1` remain unchanged.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant retained_u2:8:128:8:2:local8:2:1:0 \
  --variant first_u1:8:128:8:2:local8:1:1:1
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 128 --k 2048 --n 512 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant retained_u4:8:128:8:2:global32:4:1:0 \
  --variant first_u4:8:128:8:2:global32:4:1:1
```

Every paired run asserted exact candidate-versus-prior output equality before graph capture and timing.

| BF16 regime | Prior p50/mean/p95 (us) | Prefetch p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|
| Decode M=1, repeat 1 | 86.016 / 88.513 / 99.328 | 80.896 / 83.628 / 93.184 | 1.0633x | 1.0584x |
| Decode M=1, reversed order | 86.016 / 87.073 / 98.304 | 80.896 / 82.324 / 93.184 | 1.0633x | 1.0577x |
| Decode M=8 | 86.016 / 88.813 / 99.328 | 80.896 / 83.546 / 93.184 | 1.0633x | 1.0630x |
| Decode N=512 | 83.968 / 87.710 / 97.280 | 79.872 / 82.919 / 92.160 | 1.0513x | 1.0578x |
| Decode N=8192 | 86.016 / 88.481 / 100.352 | 81.920 / 83.970 / 95.232 | 1.0500x | 1.0537x |
| Decode K=1024 | 47.104 / 49.816 / 54.272 | 45.056 / 47.234 / 51.200 | 1.0455x | 1.0547x |
| Decode K=1536 | 65.536 / 69.442 / 75.776 | 62.464 / 65.667 / 71.680 | 1.0492x | 1.0575x |
| Prefill M=21, K=2048, N=512 | 84.992 / 87.517 / 98.304 | 81.920 / 84.448 / 95.232 | 1.0375x | 1.0363x |
| Prefill M=128, K=2048, N=512 | 87.040 / 88.495 / 100.352 | 83.968 / 85.635 / 97.280 | 1.0366x | 1.0334x |
| Prefill M=512, K=2048, N=512 | 237.568 / 240.801 / 275.456 | 230.400 / 232.986 / 267.264 | 1.0311x | 1.0335x |
| Prefill M=1024, K=2048, N=512 | 393.216 / 393.475 / 394.240 | 379.904 / 380.977 / 380.928 | 1.0350x | 1.0328x |
| Prefill M=128, K=1024, N=512 | 54.272 / 51.376 / 55.296 | 53.248 / 50.501 / 54.272 | 1.0192x | 1.0173x |

Small-N prefill retains its existing loop factor and enables the prefetch only for BF16 `krot=8, K>=1024,
N<=512`. It repeated from M=21 through M=1024 and at N=256. N=1024 regressed from
113.664/116.865/132.096 us to 122.880/126.188/142.336 us, and N=2048 regressed from
165.888/167.036/167.936 us to 175.104/175.776/176.128 us, so wider prefill keeps the prior lowering.

The final NCU comparison used the following section-first commands; normal CUDA-event timing above remains the
performance decision evidence.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. ncu --profile-from-start off --section SpeedOfLight \
  --launch-count 1 --force-overwrite \
  --export artifacts/paroquant_megakernel_20260723/gpu7_only/ncu_prefetch_first_bf16_baseline_sol \
  python scripts/benchmark_paroquant_triton_configs.py --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 \
  --eager-warmup 20 --graph-warmup 10 --iters 10 \
  --variant retained:8:128:8:2:local8:2:1:0 --variant first_u1:8:128:8:2:local8:1:1:1 \
  --profile-variant retained --profile-launches 1
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. ncu --profile-from-start off --section SpeedOfLight \
  --launch-count 1 --force-overwrite \
  --export artifacts/paroquant_megakernel_20260723/gpu7_only/ncu_prefetch_first_u1_bf16_candidate_sol \
  python scripts/benchmark_paroquant_triton_configs.py --dtype bf16 --m 1 --k 2048 --n 2048 --krot 8 \
  --eager-warmup 20 --graph-warmup 10 --iters 10 \
  --variant retained:8:128:8:2:local8:2:1:0 --variant first_u1:8:128:8:2:local8:1:1:1 \
  --profile-variant first_u1 --profile-launches 1
```

| NCU metric | Prior BF16 U2 | Prefetch BF16 U1 |
|:---|---:|---:|
| SOL replay duration (us) | 129.888 | 126.240 |
| Compute / memory / DRAM throughput | 3.77% / 4.01% / 0.75% | 3.94% / 4.17% / 0.76% |
| Registers/thread | 128 | 101 |
| Dynamic shared memory/block | 34.82 KiB | 34.82 KiB |
| Theoretical / achieved occupancy | 25.00% / 12.50% | 25.00% / 12.50% |
| Eligible / issued warps per scheduler | 0.39 / 0.30 | 0.41 / 0.31 |
| Scheduler cycles with no eligible warp | 70.33% | 69.09% |
| Warp cycles per issued instruction | 6.78 | 6.43 |
| Long-scoreboard stalls per active issue | 3.229 | 3.027 |

The final BF16 full-module run used 100 warmups and 2,000 samples. Both workloads selected the mega-kernel.
Decode candidate-versus-dense maximum error stayed equal to the existing route at 6.0; prefill stayed equal at
8.0. The permanent randomized-pair test also requires bit-exact output against the prior schedule.

| Regime | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| BF16 decode Q | 193.54 / 202.36 / 209.92 | 169.98 / 177.87 / 187.39 | 1.138x | 4,941.6 -> 5,622.1 |
| BF16 prefill K | 193.54 / 203.15 / 209.92 | 172.03 / 180.11 / 188.42 | 1.128x | 630,082 -> 710,678 |

### FP16 wide-prefill runtime wave pass

The retained FP16 BM32 projection was previously restricted to exactly 128 rows. Mapping neighboring rows exposed
the actual mechanism: on the runtime-probed 124-SM GPU, BM16 changes from 112 CTAs at M=112 to 128 CTAs at M=113,
while BM32 remains at or below one wave through M=224. At M=225, BM32 itself changes from 112 to 128 CTAs and
becomes slower. The same sawtooth repeats in later bands. Production computes both wave counts from the runtime SM
count and selects BM32 only when BM16 uses exactly twice as many waves, with a measured cap of four BM32 waves.
Later bands remain restricted to the measured 124-SM inventory; other runtime inventories retain only the original
one-wave rule. The gate covers exact
`FP16, K=2048, N in {640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048}, krot=8` shapes.
N<2048 retains only its measured one-wave band; N=2048 retains up to four waves on the measured 124-SM target.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 192 --k 2048 --n 2048 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

Every paired run asserted exact BM32-versus-BM16 output equality before graph capture and timing. The permanent
boundary accuracy cases use independent randomized coefficients at both ends of all four retained GPU 7 bands and
require bit-exact output.

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 112 | 118.784 / 121.290 / 138.240 | 152.576 / 155.204 / 177.152 | 1 | 1 | reject BM32 |
| 113 | 166.912 / 169.076 / 193.536 | 152.576 / 153.868 / 176.128 | 2 | 1 | retain BM32 |
| 128 | 167.936 / 171.047 / 194.560 | 152.576 / 156.063 / 177.152 | 2 | 1 | retain BM32 |
| 192 | 168.960 / 172.312 / 196.608 | 152.576 / 155.496 / 177.152 | 2 | 1 | retain BM32 |
| 224 | 169.984 / 172.018 / 196.608 | 152.576 / 154.797 / 177.152 | 2 | 1 | retain BM32 |
| 225 | 169.984 / 171.554 / 195.584 | 290.816 / 293.218 / 336.896 | 2 | 2 | reject BM32 |
| 256 | 257.024 / 258.668 / 258.048 | 292.864 / 294.984 / 293.888 | 3 | 2 | reject BM32 |

The later-band continuation tested both sides of every boundary through four BM32 waves. All exact 2:1 wave ratios
win; every adjacent non-2:1 ratio loses and stays on BM16.

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 368 | 262.144 / 262.981 / 263.168 | 295.936 / 296.031 / 296.960 | 3 | 2 | reject BM32 |
| 369 | 323.584 / 323.561 / 325.632 | 295.936 / 296.122 / 296.960 | 4 | 2 | retain BM32 |
| 384 | 323.584 / 325.370 / 325.632 | 295.936 / 297.919 / 296.960 | 4 | 2 | retain BM32 |
| 480 | 327.680 / 329.113 / 328.704 | 299.008 / 299.967 / 300.032 | 4 | 2 | retain BM32 |
| 481 | 327.680 / 329.118 / 329.728 | 435.200 / 436.420 / 436.224 | 4 | 3 | reject BM32 |
| 608 | 410.624 / 410.928 / 411.648 | 440.320 / 440.487 / 441.344 | 5 | 3 | reject BM32 |
| 609 | 471.040 / 471.742 / 473.088 | 440.320 / 440.812 / 441.344 | 6 | 3 | retain BM32 |
| 640 | 475.136 / 475.397 / 477.184 | 441.344 / 441.482 / 442.368 | 6 | 3 | retain BM32 |
| 736 | 482.304 / 482.219 / 483.328 | 445.440 / 445.747 / 446.464 | 6 | 3 | retain BM32 |
| 737 | 549.888 / 550.245 / 550.912 | 578.560 / 579.027 / 579.584 | 7 | 4 | reject BM32 |
| 864 | 562.176 / 562.546 / 563.200 | 584.704 / 585.431 / 585.728 | 7 | 4 | reject BM32 |
| 865 | 623.616 / 623.899 / 624.640 | 585.728 / 585.656 / 585.728 | 8 | 4 | retain BM32 |
| 896 | 626.688 / 626.646 / 627.712 | 586.752 / 586.733 / 587.776 | 8 | 4 | retain BM32 |
| 992 | 636.928 / 637.502 / 638.976 | 592.896 / 592.787 / 593.920 | 8 | 4 | retain BM32 |
| 993 | 700.416 / 700.505 / 701.440 | 722.944 / 722.999 / 723.968 | 9 | 5 | reject BM32 |

The N=1024 continuation shifts the first wave transition to M=241. Paired 4,000-sample runs again required exact
BM32/BM16 output equality. BM32 wins only while it removes the second BM16 wave, so production retains M=241-480
and leaves both the neighboring one-versus-one and two-versus-two regimes on BM16. Later N=1024 wave bands remain
disabled because they have not been measured.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 256 --k 2048 --n 1024 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 240 | 118.784 / 120.114 / 119.808 | 152.576 / 153.573 / 153.600 | 1 | 1 | reject BM32 |
| 241 | 166.912 / 169.640 / 193.536 | 152.576 / 155.086 / 177.152 | 2 | 1 | retain BM32 |
| 256 | 166.912 / 169.183 / 193.536 | 152.576 / 154.017 / 176.128 | 2 | 1 | retain BM32 |
| 480 | 169.984 / 172.988 / 196.608 | 152.576 / 156.156 / 178.176 | 2 | 1 | retain BM32 |
| 481 | 169.984 / 171.392 / 196.608 | 291.840 / 294.727 / 337.920 | 2 | 2 | reject BM32 |

The N=1536 continuation moves the first transition to M=161. Its paired 4,000-sample runs were also bit-exact and
retain BM32 only through M=320. The M=160 and M=321 neighbors show that the same runtime wave predicate rejects
BM32 once it cannot halve the wave count.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 256 --k 2048 --n 1536 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 160 | 119.808 / 122.192 / 138.240 | 152.576 / 156.045 / 177.152 | 1 | 1 | reject BM32 |
| 161 | 167.936 / 170.493 / 194.560 | 152.576 / 155.295 / 177.152 | 2 | 1 | retain BM32 |
| 256 | 169.984 / 171.815 / 196.608 | 152.576 / 154.886 / 177.152 | 2 | 1 | retain BM32 |
| 320 | 169.984 / 170.763 / 171.008 | 152.576 / 153.971 / 153.600 | 2 | 1 | retain BM32 |
| 321 | 256.000 / 256.585 / 257.024 | 291.840 / 293.311 / 293.888 | 3 | 2 | reject BM32 |

N=768 moves the same transition to M=321-640. The paired runs were bit-exact, and M=320/641 again reject BM32
at the neighboring equal-wave-count regimes.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 480 --k 2048 --n 768 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 320 | 118.784 / 120.270 / 120.832 | 152.576 / 153.783 / 155.648 | 1 | 1 | reject BM32 |
| 321 | 166.912 / 169.627 / 193.536 | 152.576 / 155.483 / 177.152 | 2 | 1 | retain BM32 |
| 480 | 169.984 / 170.545 / 171.008 | 152.576 / 153.552 / 153.600 | 2 | 1 | retain BM32 |
| 640 | 169.984 / 173.413 / 197.632 | 153.600 / 156.081 / 178.176 | 2 | 1 | retain BM32 |
| 641 | 169.984 / 170.959 / 171.008 | 290.816 / 292.655 / 292.864 | 2 | 2 | reject BM32 |

N=640 is the first aligned width above the small-N BM8 policy. Its five output-column tiles move the same FP16
BM16-to-BM32 one-wave transition to M=385-768. All 4,000-sample paired runs were bit-exact before timing.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 385 --k 2048 --n 640 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 384 | 118.784 / 121.020 / 126.976 | 152.576 / 155.053 / 162.816 | 1 | 1 | reject BM32 |
| 385 | 164.864 / 168.147 / 191.488 | 152.576 / 155.263 / 176.128 | 2 | 1 | retain BM32 |
| 768 | 168.960 / 170.070 / 169.984 | 152.576 / 153.909 / 153.600 | 2 | 1 | retain BM32 |
| 769 | 168.960 / 169.906 / 169.984 | 290.816 / 291.655 / 291.840 | 2 | 2 | reject BM32 |

A separate 2,000-sample BF16 screen rejected broadening the dtype gate. BM32 regressed mean from 171.473 to
173.783 us at M=385; at M=768 its p50 was identical and mean improved only 0.2%, below the selector margin. The
M=384/769 neighbors regressed by 45.9%/88.9% mean. BF16 therefore retains BM16 across N=640.

N=896 fills the next 128-aligned projection gap. Seven output-column tiles move the first FP16 2:1 wave band to
M=273-544. Every 4,000-sample paired run required bit-exact output before timing.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 273 --k 2048 --n 896 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 272 | 142.336 / 184.240 / 370.688 | 176.128 / 210.378 / 380.928 | 1 | 1 | reject BM32 |
| 273 | 165.888 / 168.987 / 192.512 | 152.576 / 155.321 / 176.128 | 2 | 1 | retain BM32 |
| 544 | 168.960 / 170.685 / 171.008 | 152.576 / 154.044 / 153.600 | 2 | 1 | retain BM32 |
| 545 | 168.960 / 170.938 / 195.584 | 290.816 / 294.374 / 336.896 | 2 | 2 | reject BM32 |

The M=272 boundary run was host-noisy but still rejected BM32 at p50, mean, and p95; its earlier 2,000-sample
screen also rejected BM32 by 21.9% mean. A BF16 2,000-sample screen kept BM16: BM32 regressed mean by 1.3% at
M=273 and improved only 0.2% with essentially flat p50/p95 at M=544. The M=272/545 BF16 neighbors regressed by
45.9%/89.2%.

N=1152 uses nine output-column tiles and moves the first FP16 2:1 wave band to M=209-416. All paired boundary
runs required bit-exact output and used 4,000 timing samples.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 209 --k 2048 --n 1152 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 208 | 118.784 / 121.351 / 138.240 | 152.576 / 155.413 / 177.152 | 1 | 1 | reject BM32 |
| 209 | 165.888 / 169.123 / 192.512 | 152.576 / 155.765 / 177.152 | 2 | 1 | retain BM32 |
| 416 | 168.960 / 172.602 / 196.608 | 152.576 / 155.672 / 177.152 | 2 | 1 | retain BM32 |
| 417 | 168.960 / 169.606 / 169.984 | 290.816 / 291.524 / 291.840 | 2 | 2 | reject BM32 |

The BF16 screen again stayed on BM16. BM32 regressed p50/mean/p95 by 1.2%/1.1%/1.1% at M=209 and improved only
0.6%/0.3%/0.0% at M=416, below the selector margin. Its M=208/417 neighbors regressed mean by 46.8%/89.0%.

N=1408 uses eleven output-column tiles and moves the first FP16 2:1 wave band to M=177-352. Every paired
boundary run asserted bit-exact output and used 4,000 timing samples.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 177 --k 2048 --n 1408 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 176 | 118.784 / 120.449 / 137.216 | 152.576 / 154.411 / 176.128 | 1 | 1 | reject BM32 |
| 177 | 166.912 / 168.435 / 192.512 | 152.576 / 153.839 / 176.128 | 2 | 1 | retain BM32 |
| 352 | 169.984 / 171.324 / 171.008 | 152.576 / 154.044 / 153.600 | 2 | 1 | retain BM32 |
| 353 | 254.976 / 256.593 / 257.024 | 291.840 / 293.539 / 292.864 | 3 | 2 | reject BM32 |

The BF16 screen stayed on BM16. BM32 regressed mean by 0.6% at M=177 and improved only 0.4% at M=352, below
the selector margin; the M=176/353 neighbors regressed mean by 46.4%/26.9%.

N=1664 uses thirteen output-column tiles and moves the first FP16 2:1 wave band to M=145-288. Every paired
boundary run asserted bit-exact output and used 4,000 timing samples.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 145 --k 2048 --n 1664 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 144 | 118.784 / 121.189 / 138.240 | 152.576 / 154.776 / 176.128 | 1 | 1 | reject BM32 |
| 145 | 166.912 / 168.714 / 192.512 | 152.576 / 154.370 / 176.128 | 2 | 1 | retain BM32 |
| 288 | 169.984 / 171.747 / 196.608 | 152.576 / 155.196 / 177.152 | 2 | 1 | retain BM32 |
| 289 | 169.984 / 170.517 / 171.008 | 291.840 / 293.446 / 292.864 | 2 | 2 | reject BM32 |

The BF16 screen stayed on BM16. BM32 regressed mean by 0.8% at M=145 and improved only 0.3% at M=288, below
the selector margin; the M=144/289 neighbors regressed mean by 46.7%/90.0%.

N=1920 uses fifteen output-column tiles and moves the first FP16 2:1 wave band to M=129-256. Every paired
boundary run asserted bit-exact output and used 4,000 timing samples.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 129 --k 2048 --n 1920 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 128 | 118.784 / 122.862 / 138.240 | 152.576 / 156.559 / 177.152 | 1 | 1 | reject BM32 |
| 129 | 166.912 / 169.472 / 193.536 | 152.576 / 154.406 / 176.128 | 2 | 1 | retain BM32 |
| 256 | 168.960 / 170.875 / 174.080 | 152.576 / 154.302 / 156.672 | 2 | 1 | retain BM32 |
| 257 | 256.000 / 257.238 / 257.024 | 292.864 / 294.543 / 293.888 | 2 | 2 | reject BM32 |

The BF16 screen stayed on BM16. BM32 regressed mean by 0.4% at M=129 and improved only 0.3% at M=256, below
the selector margin; the M=128/257 neighbors regressed mean by 46.5%/27.0%.

N=1280 places the first transition at M=193-384. Exact paired output and both adjacent losing boundaries follow
the same pattern.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 256 --k 2048 --n 1280 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 192 | 119.808 / 127.620 / 146.432 | 152.576 / 160.997 / 178.176 | 1 | 1 | reject BM32 |
| 193 | 166.912 / 169.487 / 193.536 | 152.576 / 154.706 / 176.128 | 2 | 1 | retain BM32 |
| 256 | 168.960 / 170.992 / 195.584 | 152.576 / 154.372 / 177.152 | 2 | 1 | retain BM32 |
| 384 | 169.984 / 170.666 / 171.008 | 152.576 / 153.629 / 153.600 | 2 | 1 | retain BM32 |
| 385 | 254.976 / 255.328 / 256.000 | 291.840 / 292.186 / 292.864 | 3 | 2 | reject BM32 |

N=1792 completes the measured 256-aligned projection widths through N=2048. Its first transition is M=129-256;
exact paired output and the M=128/257 neighbors again validate the runtime wave rule.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 192 --k 2048 --n 1792 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm16:16:128:8:2:global32:1:0:0 --variant bm32:32:128:8:2:global32:1:0:0
```

| FP16 rows | BM16 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM16 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 128 | 118.784 / 121.997 / 138.240 | 152.576 / 156.257 / 177.152 | 1 | 1 | reject BM32 |
| 129 | 165.888 / 166.267 / 167.936 | 152.576 / 152.764 / 153.600 | 2 | 1 | retain BM32 |
| 192 | 168.960 / 170.785 / 195.584 | 152.576 / 154.215 / 176.128 | 2 | 1 | retain BM32 |
| 256 | 169.984 / 170.996 / 171.008 | 152.576 / 153.773 / 153.600 | 2 | 1 | retain BM32 |
| 257 | 169.984 / 170.363 / 171.008 | 289.792 / 290.304 / 290.816 | 2 | 2 | reject BM32 |

The full-module audits forced the eligible mega-kernel plan, used 100 warmups and 2,000 CUDA-event samples, and
retained the dense and existing-route accuracy comparisons.

| Regime | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| FP16 prefill M=192 | 201.728 / 204.913 / 216.064 | 175.104 / 180.371 / 191.488 | 1.136x | 936,982 -> 1,064,470 |
| FP16 prefill M=256, N=1024 | 202.240 / 204.781 / 218.112 | 181.248 / 187.045 / 197.632 | 1.095x | 1,250,119 -> 1,368,656 |
| FP16 prefill M=480, N=1024 | 198.656 / 201.836 / 216.064 | 176.128 / 180.320 / 196.608 | 1.119x | 2,378,174 -> 2,661,931 |
| FP16 prefill M=256, N=1536 | 201.728 / 207.404 / 217.088 | 176.128 / 179.181 / 191.488 | 1.158x | 1,234,306 -> 1,428,727 |
| FP16 prefill M=320, N=1536 | 195.584 / 198.470 / 209.920 | 175.104 / 180.184 / 187.392 | 1.101x | 1,612,333 -> 1,775,962 |
| FP16 prefill M=480, N=768 | 197.632 / 200.407 / 214.016 | 176.128 / 179.147 / 191.488 | 1.119x | 2,395,132 -> 2,679,368 |
| FP16 prefill M=640, N=768 | 196.608 / 198.987 / 210.944 | 175.104 / 179.814 / 191.488 | 1.107x | 3,216,286 -> 3,559,225 |
| FP16 prefill M=385, N=640 | 200.704 / 217.581 / 219.136 | 179.200 / 195.975 / 198.656 | 1.110x | 1,769,456 -> 1,964,540 |
| FP16 prefill M=768, N=640 | 198.656 / 222.507 / 217.088 | 177.152 / 190.702 / 197.632 | 1.167x | 3,451,577 -> 4,027,224 |
| FP16 prefill M=273, N=896 | 203.776 / 242.357 / 450.560 | 180.224 / 205.399 / 280.576 | 1.180x | 1,126,436 -> 1,329,123 |
| FP16 prefill M=544, N=896 repeat | 193.536 / 202.550 / 209.920 | 174.080 / 181.244 / 191.488 | 1.118x | 2,685,756 -> 3,001,476 |
| FP16 prefill M=209, N=1152 | 198.656 / 202.594 / 214.016 | 174.080 / 181.014 / 191.488 | 1.119x | 1,031,621 -> 1,154,607 |
| FP16 prefill M=416, N=1152 | 194.560 / 198.431 / 207.872 | 173.056 / 176.985 / 186.368 | 1.121x | 2,096,444 -> 2,350,480 |
| FP16 prefill M=177, N=1408 | 205.824 / 209.930 / 220.160 | 180.224 / 188.636 / 197.632 | 1.113x | 843,139 -> 938,317 |
| FP16 prefill M=352, N=1408 | 200.704 / 210.986 / 218.112 | 183.296 / 192.852 / 201.728 | 1.094x | 1,668,357 -> 1,825,229 |
| FP16 prefill M=145, N=1664 | 202.752 / 207.650 / 224.256 | 177.152 / 185.004 / 207.872 | 1.122x | 698,290 -> 783,766 |
| FP16 prefill M=288, N=1664 repeat | 204.800 / 223.415 / 241.664 | 182.272 / 195.699 / 205.824 | 1.142x | 1,289,082 -> 1,471,648 |
| FP16 prefill M=129, N=1920 | 196.608 / 204.540 / 211.968 | 176.128 / 181.882 / 186.368 | 1.125x | 630,684 -> 709,253 |
| FP16 prefill M=256, N=1920 | 202.752 / 212.477 / 218.112 | 180.224 / 188.116 / 195.584 | 1.130x | 1,204,834 -> 1,360,863 |
| FP16 prefill M=256, N=1280 | 199.680 / 203.398 / 216.064 | 177.152 / 192.190 / 196.608 | 1.058x | 1,258,615 -> 1,332,012 |
| FP16 prefill M=384, N=1280 | 191.488 / 194.394 / 204.800 | 172.032 / 175.917 / 185.344 | 1.105x | 1,975,368 -> 2,182,847 |
| FP16 prefill M=192, N=1792 | 206.848 / 212.962 / 242.688 | 182.272 / 191.528 / 216.064 | 1.112x | 901,570 -> 1,002,462 |
| FP16 prefill M=256, N=1792 | 191.488 / 194.240 / 206.848 | 176.128 / 183.157 / 192.512 | 1.061x | 1,317,957 -> 1,397,706 |

Existing-versus-dense maximum error was 0.5, mega-versus-dense maximum error was 1.0, and
existing-versus-mega maximum/mean error was 1.0/0.005810. The schedule-only BM16/BM32 comparisons remain
bit-identical, so the row-tile optimization itself adds no numerical error. At N=1024, both full-module endpoints
kept existing-versus-mega maximum error at 1.0; mean error was 0.005867 at M=256 and 0.006023 at M=480. N=1536
also kept maximum error at 1.0, with mean error 0.006138 at M=256 and 0.006123 at M=320. One maximum-latency outlier
landed in the M=256 baseline and one in the M=320 candidate; the p50 and p95 improvements agree with the raw pass.
At N=768, maximum error remained 1.0 and mean error was 0.006092/0.005898 at M=480/640. The M=640 candidate mean
includes one 2.442 ms maximum sample; its p50 and p95 still improve by 12.3% and 9.2%. Peak allocation falls from
8,601,600 to 737,280 bytes at M=480 and from 11,468,800 to 983,040 bytes at M=640. At N=640, the normal selector
chose `prefill_megakernel` at both endpoints. Existing-versus-mega maximum/mean error was 1.0/0.006203 at M=385
and 1.0/0.006157 at M=768. Peak allocation falls from 6,012,416 to 493,056 bytes and from 11,993,088 to 983,040
bytes. The M=385 baseline/candidate means include isolated 4.279/7.853 ms host-side maxima, while p50 and p95
improve by 12.0%/9.3%. At N=896, the normal selector also chose `prefill_megakernel` at both endpoints.
Existing-versus-mega maximum/mean error was 1.0/0.005798 at M=273 and 1.0/0.005962 at M=544. Peak allocation
falls from 5,521,408 to 489,472 bytes and from 11,001,856 to 974,848 bytes. The first M=544 selector audit had a
0.45% p95 noise reversal; a 200-warmup, 4,000-sample repeat improved p50/mean/p95 by 10.1%/10.5%/8.8%. At N=1152,
the normal selector chose `prefill_megakernel` at both endpoints. Existing-versus-mega maximum error remained 1.0,
with mean error 0.006275 at M=209 and 0.006683 at M=416. Peak allocation falls from 5,190,144 to 481,792 bytes
and from 10,330,112 to 958,464 bytes. At N=1408, the normal selector chose the mega-kernel at both endpoints.
Existing-versus-mega maximum error remained 1.0, with mean error 0.006176 at M=177 and 0.006001 at M=352. Peak
allocation falls from 5,211,136 to 498,688 bytes and from 11,214,848 to 991,232 bytes. At N=1664, the normal
selector chose the mega-kernel at both endpoints. Existing-versus-mega maximum error remained 1.0, with mean
error 0.006027 at M=145 and 0.005943 at M=288. Peak allocation falls from 4,937,216 to 482,816 bytes and from
9,805,824 to 958,464 bytes. At N=1920, the normal selector chose the mega-kernel at both endpoints.
Existing-versus-mega maximum error remained 1.0, with mean error 0.006344 at M=129 and 0.006119 at M=256. Peak
allocation falls from 4,986,880 to 495,616 bytes and from 10,551,296 to 983,040 bytes. At N=1280, the M=256
candidate contains one 9.370 ms maximum sample but still improves p50/p95 by 12.7%/9.9%; the M=384 p50/mean/p95
all improve by 9.5-11.3%. Existing-versus-mega maximum error is 0.5 at M=256 and 1.0 at M=384, with mean error
0.005718/0.006245. Peak allocation falls from 6,946,816 to 655,360 bytes and 10,420,224 to 983,040 bytes. N=1792
contains isolated 7.558/6.314 ms candidate maxima at M=192/256, but p50 improves 13.5%/8.7% and p95 improves
11.0%/6.9%; mean still improves 11.2%/6.1%. Existing-versus-mega maximum/mean error is 0.5/0.006039 at M=192 and
1.0/0.005646 at M=256. Peak allocation falls from 6,979,584 to 688,128 bytes and 9,306,112 to 917,504 bytes.

At M=640, BM32 makes the raw mega-kernel 7.7% faster but the established route remains faster overall. With the
normal autotuner enabled, the module selects `cuda_awq`, returns bit-identical output, and does not force the losing
mega-kernel. This preserves the full-operator selection outcome while improving the explicit/forced kernel path.

### Large-M N=512 one-wave tile

The small-N BM8 schedule accumulates enough CTAs to need three waves at M=497 on the 124-SM target. BM32 keeps
exact `K=2048, N=512, krot=8` in one wave through M=992, reducing duplicated rotation work while preserving the
existing dtype-specific unroll, FMA, and BF16 first-partner schedules. Production probes the input device at runtime
and selects BM32 only for FP16/BF16 M=497-992 on a 124-SM device. Other SM counts, shapes, and dtypes retain BM8.

The paired configuration runner required bit-exact output before timing. These commands were repeated at every
reported row boundary:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 497 --k 2048 --n 512 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm8:8:128:8:2:global32:2:1:0 --variant bm32:32:128:8:2:global32:2:1:0
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 497 --k 2048 --n 512 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm8:8:128:8:2:global32:4:1:1 --variant bm32:32:128:8:2:global32:4:1:1
```

| FP16 rows | BM8 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM8 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 496 | 109.568 / 110.774 / 125.952 | 153.600 / 155.695 / 178.176 | 2 | 1 | reject BM32 |
| 497 | 180.224 / 181.905 / 208.896 | 153.600 / 155.028 / 177.152 | 3 | 1 | retain BM32 |
| 512 | 180.224 / 181.941 / 182.272 | 153.600 / 154.533 / 154.624 | 3 | 1 | retain BM32 |
| 992 | 215.040 / 218.272 / 237.568 | 154.624 / 156.446 / 169.984 | 4 | 1 | retain BM32 |
| 993 | 277.504 / 277.555 / 278.528 | 292.864 / 292.886 / 293.888 | 5 | 2 | reject BM32 |
| 1024 | 278.528 / 279.261 / 279.552 | 293.888 / 294.081 / 294.912 | 5 | 2 | reject BM32 |

| BF16 rows | BM8 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM8 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 496 | 161.792 / 164.334 / 188.416 | 161.792 / 163.454 / 187.392 | 2 | 1 | reject: flat p50 |
| 497 | 229.376 / 232.885 / 266.240 | 161.792 / 163.816 / 187.392 | 3 | 1 | retain BM32 |
| 512 | 230.400 / 231.512 / 231.424 | 161.792 / 162.399 / 162.816 | 3 | 1 | retain BM32 |
| 992 | 317.440 / 318.645 / 318.464 | 161.792 / 162.257 / 162.816 | 4 | 1 | retain BM32 |
| 993 | 378.880 / 378.858 / 379.904 | 309.248 / 309.813 / 310.272 | 5 | 2 | reject end to end |

Forced full-module comparisons used `GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE=0`, 100 warmups, and 2,000 CUDA-event
samples, except the disclosed BF16 M=512 confirmation repeat, which used 200 warmups and 4,000 samples.

| Dtype/rows | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| FP16 M=512 | 209.408 / 221.365 / 245.760 | 181.248 / 190.463 / 219.136 | 1.162x | 2,312,925 -> 2,688,187 |
| FP16 M=992 | 203.776 / 207.498 / 220.160 | 180.224 / 183.511 / 197.632 | 1.131x | 4,780,775 -> 5,405,684 |
| BF16 M=512 repeat | 190.464 / 194.759 / 205.824 | 173.056 / 176.990 / 189.440 | 1.100x | 2,628,891 -> 2,892,820 |
| BF16 M=992 | 192.512 / 201.147 / 207.872 | 178.176 / 182.649 / 194.560 | 1.101x | 4,931,707 -> 5,431,187 |

The first BF16 M=512 run improved p50/mean from 192.512/210.314 to 176.128/200.338 us but contained one
candidate-tail excursion that moved p95 from 219.136 to 287.744 us. The longer repeat above removed the reversal
and improved all three statistics. At BF16 M=993, the raw two-wave BM32 win does not survive full dispatch:
186.368/194.829/202.752 us for the established route beats 310.272/310.306/310.272 us for the mega-kernel. The
production cutoff therefore remains at the one-wave M=992 boundary.

The BM8/BM32 schedule comparison is bit-exact for both dtypes at M=497 and M=992, so this tile change itself adds
no numerical error. Against the separate established route, FP16 existing-versus-mega maximum/mean absolute error
is 0.5/0.006306 at M=512 and 1.0/0.006104 at M=992. At BF16 M=512, existing/mega maximum error versus dense is
2.0/4.0 and existing-versus-mega maximum/mean error is 4.0/0.009338; M=992 existing-versus-mega error is
4.0/0.009888. Peak allocation falls from 6,815,744 to 524,288 bytes at M=512 for either dtype, from 13,205,504 to
1,015,808 bytes for FP16 M=992, and from 13,467,648 to 1,015,808 bytes for BF16 M=992.

### Large-M N=256 one-wave tile

At N=256, the same small-N wave transition moves to M=993-1984. BM8 needs three to four waves while BM32 remains
within one on the 124-SM target. Production uses the same runtime-probed predicate but explicitly gates the width,
so unmeasured small-N projections and other SM inventories remain on BM8. Both dtype schedules keep loop factor
one; FP16 keeps explicit FMA and BF16 keeps explicit FMA plus first-partner prefetch.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 993 --k 2048 --n 256 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm8:8:128:8:2:global32:1:1:0 --variant bm32:32:128:8:2:global32:1:1:0
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 993 --k 2048 --n 256 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm8:8:128:8:2:global32:1:1:1 --variant bm32:32:128:8:2:global32:1:1:1
```

Every paired run required bit-exact BM8/BM32 output before timing.

| FP16 rows | BM8 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM8 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 992 | 109.568 / 110.546 / 114.688 | 152.576 / 154.296 / 159.744 | 2 | 1 | reject BM32 |
| 993 | 183.296 / 186.592 / 212.992 | 152.576 / 154.933 / 176.128 | 3 | 1 | retain BM32 |
| 1984 | 214.016 / 217.216 / 247.808 | 152.576 / 155.420 / 177.152 | 4 | 1 | retain BM32 |
| 1985 | 280.576 / 281.409 / 282.624 | 290.816 / 290.890 / 291.840 | 5 | 2 | reject BM32 |

| BF16 rows | BM8 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM8 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 992 | 110.592 / 112.839 / 128.000 | 174.080 / 177.766 / 201.728 | 2 | 1 | reject BM32 |
| 993 | 180.224 / 183.504 / 206.848 | 174.080 / 176.974 / 198.656 | 3 | 1 | retain BM32 |
| 1984 | 218.112 / 220.145 / 251.904 | 174.080 / 176.523 / 201.728 | 4 | 1 | retain BM32 |
| 1985 | 279.552 / 279.958 / 280.576 | 332.800 / 332.433 / 332.800 | 5 | 2 | reject BM32 |

The normal selector audit used 20 autotune warmups, 100 autotune samples, then 100 module warmups and 2,000
CUDA-event samples. It selected `prefill_megakernel` at both retained endpoints for both dtypes.

| Dtype/rows | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| FP16 M=993 | 196.096 / 200.150 / 214.016 | 174.080 / 177.552 / 192.512 | 1.127x | 4,961,291 -> 5,592,732 |
| FP16 M=1984 | 195.584 / 199.298 / 211.968 | 175.104 / 178.869 / 193.536 | 1.114x | 9,954,965 -> 11,091,901 |
| BF16 M=993 | 189.440 / 206.675 / 207.872 | 176.128 / 188.253 / 202.752 | 1.098x | 4,804,634 -> 5,274,826 |
| BF16 M=1984 | 187.392 / 223.106 / 212.992 | 176.128 / 197.841 / 208.896 | 1.128x | 8,892,652 -> 10,028,260 |

The BF16 M=993 lower edge was repeated with 200 warmups and 4,000 samples after an initial noisy mean/p95
reversal. The repeat improved existing-to-mega p50/mean/p95 from 187.392/191.445/205.824 to
174.080/180.621/202.752 us, confirming the retained boundary. The selector-audit means contain isolated host-side
maximum-latency outliers, but all p50 and p95 comparisons still favor the mega-kernel.

The permanent randomized endpoint tests require exact BM32/BM8 equality, so this scheduling change adds no
numerical error. In the selector audit, FP16 existing/mega dense maximum error remains 1.0 at both endpoints;
existing-versus-mega maximum/mean error is 1.0/0.006374 at M=993 and 1.0/0.006039 at M=1984. BF16 M=993 keeps
existing/mega dense maximum error at 4.0 and existing-versus-mega error at 4.0/0.010254. At BF16 M=1984,
existing/mega dense maximum error is 4.0/8.0 and existing-versus-mega error is 8.0/0.009888. Peak allocation falls
from 8,892,928 to 508,416 bytes and 17,276,928 to 1,015,808 bytes at the FP16 endpoints; BF16 falls from
8,643,072 to 508,416 bytes and 17,268,736 to 1,015,808 bytes.

### Large-M N=128 one-wave tile

At N=128, one output tile moves the three-to-one wave transition to M=1985 and keeps BM32 in one wave through
M=3968 on the 124-SM target. Production explicitly gates N=128 alongside the other measured small-N widths; all
other widths and SM inventories retain BM8.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 1985 --k 2048 --n 128 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm8:8:128:8:2:global32:1:1:0 --variant bm32:32:128:8:2:global32:1:1:0
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 1985 --k 2048 --n 128 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm8:8:128:8:2:global32:1:1:1 --variant bm32:32:128:8:2:global32:1:1:1
```

All boundary runs were bit-exact before timing.

| FP16 rows | BM8 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM8 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 1984 | 107.520 / 110.649 / 124.928 | 152.576 / 155.914 / 177.152 | 2 | 1 | reject BM32 |
| 1985 | 182.272 / 184.790 / 210.944 | 152.576 / 154.222 / 176.128 | 3 | 1 | retain BM32 |
| 3968 | 210.944 / 212.085 / 215.040 | 152.576 / 153.485 / 153.600 | 4 | 1 | retain BM32 |
| 3969 | 279.552 / 280.024 / 281.600 | 289.792 / 290.052 / 290.816 | 5 | 2 | reject BM32 |

| BF16 rows | BM8 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM8 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 1984 | 108.544 / 109.876 / 125.952 | 173.056 / 175.503 / 200.704 | 2 | 1 | reject BM32 |
| 1985 | 179.200 / 182.401 / 207.872 | 173.056 / 176.744 / 200.704 | 3 | 1 | retain BM32 |
| 3968 | 212.992 / 215.020 / 245.760 | 174.080 / 175.543 / 201.728 | 4 | 1 | retain BM32 |
| 3969 | 276.480 / 277.029 / 277.504 | 330.752 / 331.221 / 331.776 | 5 | 2 | reject BM32 |

The same normal-selector protocol selected `prefill_megakernel` at both endpoints for both dtypes.

| Dtype/rows | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| FP16 M=1985 | 194.560 / 198.254 / 212.992 | 165.888 / 169.460 / 185.344 | 1.170x | 10,012,404 -> 11,713,699 |
| FP16 M=3968 | 191.488 / 195.086 / 208.896 | 164.864 / 168.549 / 183.296 | 1.157x | 20,339,712 -> 23,542,063 |
| BF16 M=1985 | 187.392 / 191.686 / 202.752 | 174.080 / 174.440 / 175.104 | 1.099x | 10,355,497 -> 11,379,275 |
| BF16 M=3968 | 183.296 / 186.277 / 198.656 | 175.104 / 175.107 / 175.104 | 1.064x | 21,301,567 -> 22,660,487 |

The permanent endpoint tests require exact BM32/BM8 equality, so the tile change adds no numerical error. FP16
existing/mega dense and existing-versus-mega maximum error remains 1.0 at both endpoints; mean difference is
0.006149 at M=1985 and 0.006210 at M=3968. BF16 M=1985 keeps existing/mega dense and cross-route maxima at 4.0
with mean difference 0.010010. At BF16 M=3968, existing/mega dense maxima are 4.0/8.0 and cross-route maximum/mean
error is 8.0/0.010620. Peak allocation falls from 12,960,256 to 508,416 bytes and 25,919,488 to 1,015,808 bytes
at the FP16 endpoints; BF16 falls from 16,261,120 to 508,416 bytes and 33,554,432 to 1,015,808 bytes.

### Large-M N=384 one-wave tile

N=384 fills the remaining aligned small-N projection width between the retained N=256 and N=512 schedules. Three
output-column tiles put BM8's first three-wave row at M=657, while BM32 remains within one wave through M=1312 on
the runtime-probed 124-SM target. Production gates that exact range for FP16/BF16 `K=2048, krot=8`; neighboring
rows, other widths, and other SM inventories retain BM8.

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 657 --k 2048 --n 384 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm8:8:128:8:2:global32:1:1:0 --variant bm32:32:128:8:2:global32:1:1:0
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype bf16 --m 657 --k 2048 --n 384 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 4000 \
  --variant bm8:8:128:8:2:global32:1:1:1 --variant bm32:32:128:8:2:global32:1:1:1
```

All 4,000-sample boundary runs required bit-exact BM32/BM8 output before timing.

| FP16 rows | BM8 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM8 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 656 | 111.616 / 114.198 / 129.024 | 152.576 / 156.669 / 177.152 | 2 | 1 | reject BM32 |
| 657 | 182.272 / 186.086 / 211.968 | 152.576 / 155.192 / 177.152 | 3 | 1 | retain BM32 |
| 1312 | 217.088 / 218.197 / 219.136 | 153.600 / 154.241 / 154.624 | 4 | 1 | retain BM32 |
| 1313 | 217.088 / 218.639 / 249.856 | 290.816 / 293.340 / 336.896 | 5 | 2 | reject BM32 |

| BF16 rows | BM8 p50/mean/p95 (us) | BM32 p50/mean/p95 (us) | BM8 waves | BM32 waves | Result |
|---:|---:|---:|---:|---:|:---|
| 656 | 111.616 / 114.672 / 130.048 | 174.080 / 177.782 / 201.728 | 2 | 1 | reject BM32 |
| 657 | 180.224 / 181.113 / 181.248 | 174.080 / 174.263 / 174.080 | 3 | 1 | retain BM32 |
| 1312 | 218.112 / 220.191 / 221.184 | 174.080 / 175.905 / 179.200 | 4 | 1 | retain BM32 |
| 1313 | 218.112 / 219.058 / 219.136 | 331.776 / 333.227 / 332.800 | 5 | 2 | reject BM32 |

The normal selector audit used 20 autotune warmups, 100 autotune samples, then 100 module warmups and 2,000
CUDA-event samples. It selected `prefill_megakernel` at both retained endpoints for both dtypes.

| Dtype/rows | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| FP16 M=657 | 218.112 / 226.844 / 237.568 | 171.008 / 206.768 / 209.920 | 1.097x | 2,896,268 -> 3,177,480 |
| FP16 M=1312 | 202.752 / 206.368 / 223.232 | 171.008 / 174.474 / 192.512 | 1.183x | 6,357,583 -> 7,519,757 |
| BF16 M=657 | 188.416 / 203.236 / 209.920 | 174.080 / 175.298 / 175.104 | 1.159x | 3,232,689 -> 3,747,903 |
| BF16 M=1312 | 188.416 / 194.353 / 206.848 | 175.104 / 174.960 / 175.104 | 1.111x | 6,750,598 -> 7,498,873 |

The FP16 M=657 mega-kernel mean includes one disclosed 13.872 ms host-side maximum sample; its p50, mean, and
p95 still improve, and all other endpoint comparisons improve all three statistics. The permanent randomized
endpoint tests require exact BM32/BM8 equality, so the scheduling change itself adds no numerical error. FP16
existing/mega dense maximum error is 0.5 at M=657 and 1.0 at M=1312; cross-route maximum/mean error is
0.5/0.006287 and 1.0/0.006462. BF16 existing/mega dense maximum error remains 4.0 at both endpoints; cross-route
maximum/mean error is 4.0/0.008301 and 4.0/0.010010. Peak allocation falls from 7,232,512 to 504,832 bytes and
14,442,496 to 1,007,616 bytes at the FP16 endpoints; BF16 falls from 7,459,840 to 504,832 bytes and 14,442,496
to 1,007,616 bytes.

### Wide FP16 BM32 16-warp schedule

After completing the 128-aligned width sweep, paired CUDA-graph measurements showed that the retained wide FP16
BM32 tiles benefit consistently from 16 warps and one Triton pipeline stage. The production gate is deliberately
narrow: exact `K=2048`, `krot=8`, FP16, a 128-aligned N=640-4096 width, a row count already selected for BM32 by
the runtime wave rule, and a runtime-probed 124-SM device. N>2048 is further limited to its measured first-wave
band. Small-N BM32, BF16, BM16, other shapes, and other SM counts keep eight warps and two stages. The wrapper
reuses the SM count already read for wave selection, so this specialization adds no second device-property query.

The retained run used only physical GPU 7:

```text
CUDA_VISIBLE_DEVICES     7
GPU                      NVIDIA PG506-230
UUID                     GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28
PCI bus                  00000000:E4:00.0
compute capability       8.0
SM count                 124
memory                   98,304 MiB
driver                   610.43.02
PyTorch / CUDA           2.13.0+cu130 / 13.0
Triton                   3.7.1
JIT launch               BM32, BN128, BK128, W16, one stage, sm_80
```

Each screen used AB/BA-alternating CUDA-graph events and required an exact W8S2/W16S1 output match before timing.
N=1920 used 4,000 samples; the other width screens used 2,000 samples.

| N / rows | W8S2 p50/mean/p95 (us) | W16S1 p50/mean/p95 (us) | Mean speedup |
|---:|---:|---:|---:|
| 640 / 385 | 152.576 / 155.554 / 177.152 | 143.360 / 145.291 / 165.888 | 1.071x |
| 768 / 321 | 152.576 / 155.736 / 177.152 | 143.360 / 146.121 / 165.888 | 1.066x |
| 896 / 273 | 152.576 / 158.662 / 171.008 | 143.360 / 148.862 / 160.768 | 1.066x |
| 1024 / 241 | 152.576 / 157.179 / 177.152 | 143.360 / 147.720 / 165.888 | 1.064x |
| 1152 / 209 | 152.576 / 153.778 / 176.128 | 142.336 / 143.863 / 144.384 | 1.069x |
| 1280 / 193 | 152.576 / 158.492 / 177.152 | 142.336 / 148.403 / 165.888 | 1.068x |
| 1408 / 177 | 152.576 / 157.469 / 177.152 | 143.360 / 147.660 / 165.888 | 1.066x |
| 1536 / 161 | 152.576 / 156.881 / 177.152 | 143.360 / 147.095 / 165.888 | 1.067x |
| 1664 / 145 | 152.576 / 160.113 / 177.152 | 143.360 / 149.964 / 165.888 | 1.068x |
| 1792 / 129 | 152.576 / 154.759 / 177.152 | 143.360 / 144.976 / 165.888 | 1.067x |
| 1920 / 129 | 152.576 / 153.523 / 153.600 | 143.360 / 144.032 / 143.360 | 1.066x |
| 2048 / 128 | 152.576 / 156.616 / 177.152 | 143.360 / 146.379 / 165.888 | 1.070x |

The later retained N=2048 bands also improved: mean speedup was 1.067x at M=369, 1.068x at M=609, 1.068x at
M=865, and 1.070x at M=992. A final 4,000-sample paired production-wrapper repeat removed non-paired system drift:

| N / rows | Production W8S2 p50/mean/p95 (us) | Production W16S1 p50/mean/p95 (us) | p50 / mean speedup |
|---:|---:|---:|---:|
| 640 / 385 | 152.576 / 152.519 / 152.576 | 142.336 / 142.979 / 143.360 | 1.072x / 1.067x |
| 1920 / 129 | 152.576 / 154.701 / 176.128 | 143.360 / 144.914 / 165.888 | 1.064x / 1.067x |
| 1920 / 256 | 152.576 / 152.997 / 153.600 | 143.360 / 143.314 / 144.384 | 1.064x / 1.068x |
| 2048 / 128 | 152.576 / 152.433 / 152.576 | 142.336 / 142.963 / 143.360 | 1.072x / 1.066x |

The normal module selector continued to choose `prefill_megakernel` for the first-wave production cases. These
4,000-sample module measurements include the normal dispatch path and compare it with the established ParoQuant
route:

| Shape | Existing p50/mean/p95 (us) | W16 mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| M=129, N=1920 | 199.680 / 204.177 / 215.040 | 181.248 / 185.444 / 194.560 | 1.101x | 631,803 -> 695,626 |
| M=256, N=1920 | 200.704 / 207.057 / 235.520 | 181.248 / 185.625 / 209.920 | 1.115x | 1,236,377 -> 1,379,124 |
| M=128, N=2048 | 199.680 / 203.976 / 217.088 | 182.272 / 189.023 / 200.704 | 1.079x | 627,525 -> 677,165 |

Existing-versus-mega maximum/mean errors remained 1.0/0.006344, 1.0/0.006157, and 1.0/0.006210,
respectively. The scheduling-only W8S2/W16S1 comparisons were bit-exact. Peak forward allocation fell from
4,986,880 to 495,616 bytes, 9,895,936 to 983,040 bytes, and 5,242,880 to 524,288 bytes. The normal selector chose
`cuda_awq` at N=2048/M=992, so the faster raw mega-kernel does not override that end-to-end decision.

Focused Nsight Compute captures used the same N=1920/M=129 input and one profiled launch per report. W16S1 lowers
replay duration and register pressure while exposing enough additional warps to hide the long dependency chain:

| Metric | W8S2 | W16S1 |
|:---|---:|---:|
| Replay duration | 212.672 us | 185.312 us |
| Threads / CTA | 256 | 512 |
| Registers / thread | 160 | 120 |
| Dynamic shared memory / CTA | 40.960 KiB | 40.960 KiB |
| Active warps / SM cycle | 7.912 | 15.981 |
| Achieved occupancy | 12.36% | 24.97% |
| Eligible warps / scheduler cycle | 0.555 | 1.145 |
| Issue-active warps / scheduler cycle | 0.38 | 0.50 |
| SM throughput | 23.00% | 30.23% |
| Compute-memory throughput | 26.48% | 36.49% |

Concurrent probes kept the optimization bounded:

- N=512 BM32 must remain W8S2. W16S1 regressed FP16 by about 2% and BF16 by 6-7% at M=497/992.
- BF16 decode row tiles BM1/2/4 all lost to retained BM8; one, two, and three stages were effectively flat.
  BN64 and BN256 also lost to BN128.
- A direct-global first paired-activation load regressed BF16 decode by about 3.1%; an aligned-N unmasked path
  regressed by about 5%. Both source experiments were fully reverted.
- Extending N=2048 BM32 to a fifth wave improves the raw mega-kernel at M=1105/1216, but the normal selector's
  `cuda_awq` route remains much faster. The four-wave cap stays in place.

### Extended FP16 first-wave widths through N=4096

The same FP16 BM32/W16/S1 schedule remains profitable for every 128-aligned width from N=2176 through N=4096
when BM32 fits in its first CTA wave and BM16 needs exactly two waves. This extension is deliberately narrower
than the portable N<=2048 wave rule: it requires exact `K=2048`, `krot=8`, FP16, a runtime-probed 124-SM device,
and the first-wave row band. For `t=N/128` and `q=floor(124/t)`, the retained inclusive row range is
`16q+1 <= M <= 32q`. BF16, other SM counts, unaligned widths, neighboring rows, and configured prefill maximums
below the requested N preserve the established route.

Every endpoint screen ran on physical GPU 7 with 1,000 AB/BA-alternating CUDA-graph samples and required exact
BM16/W8/S2 versus BM32/W16/S1 equality before timing:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/benchmark_paroquant_triton_configs.py \
  --dtype fp16 --m 97 --k 2048 --n 2560 --krot 8 --eager-warmup 100 --graph-warmup 100 --iters 1000 \
  --variant bm16:16:128:8:2:global32 --variant bm32:32:128:16:1:global32
```

| N | Retained M band | Lower-edge raw mean speedup | Upper-edge raw mean speedup |
|---:|:---|---:|---:|
| 2176 | 113-224 | 1.1732x | 1.1858x |
| 2304 | 97-192 | 1.1652x | 1.1861x |
| 2432 | 97-192 | 1.1658x | 1.1759x |
| 2560 | 97-192 | 1.1745x | 1.1835x |
| 2688 | 81-160 | 1.1561x | 1.1794x |
| 2816 | 81-160 | 1.1704x | 1.1873x |
| 2944 | 81-160 | 1.1738x | 1.1830x |
| 3072 | 81-160 | 1.1870x | 1.1887x |
| 3200 | 65-128 | 1.1545x | 1.1851x |
| 3328 | 65-128 | 1.1670x | 1.1803x |
| 3456 | 65-128 | 1.1752x | 1.1900x |
| 3584 | 65-128 | 1.1743x | 1.1834x |
| 3712 | 65-128 | 1.1758x | 1.1837x |
| 3840 | 65-128 | 1.1740x | 1.1856x |
| 3968 | 65-128 | 1.1760x | 1.1790x |
| 4096 | 49-96 | 1.1650x | 1.1748x |

Normal production-wrapper measurements used 200 warmups and 4,000 CUDA-event samples on physical GPU 7. The
selector chose `prefill_megakernel` for every representative retained case below:

| Shape | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s | Peak allocation bytes |
|:---|---:|---:|---:|---:|---:|
| M=97, N=2560 | 203.776 / 209.861 / 227.328 | 182.272 / 188.539 / 200.704 | 1.113x | 462,210 -> 514,481 | 4,867,072 -> 496,640 |
| M=192, N=2560 | 201.728 / 205.438 / 217.088 | 183.296 / 187.473 / 201.728 | 1.096x | 934,589 -> 1,024,145 | 9,633,792 -> 983,040 |
| M=81, N=3072 | 197.632 / 201.468 / 214.016 | 178.176 / 184.836 / 196.608 | 1.090x | 402,048 -> 438,227 | 4,810,752 -> 497,664 |
| M=160, N=3072 | 272.384 / 278.609 / 310.272 | 237.568 / 252.823 / 304.128 | 1.102x | 574,282 -> 632,855 | 9,502,720 -> 983,040 |
| M=96, N=4096 | 198.656 / 202.504 / 216.064 | 177.152 / 182.016 / 195.584 | 1.113x | 474,064 -> 527,426 | 7,471,104 -> 786,432 |

Existing-versus-mega maximum/mean absolute error was 1.0/0.006393, 1.0/0.006042, 1.0/0.005470,
1.0/0.006145, and 1.0/0.006390 in table order; dense-reference errors remained inside the already validated
FP16 envelope. N=4096/M=49 is inside the raw first-wave band but the normal selector retained `decode_fused`;
that fallback's 235.376 us mean beat the existing route's 408.926 us mean, so it is not attributed to the
mega-kernel and no plan is forced.

The following continuation probes were rejected and fully reverted:

- W16 first-partner prefetch, K-loop unroll factors two/four, local `int16` partners, and explicit wide-prefill FMA
  were neutral or 2.5-8.8% slower.
- Nsight Compute exposed 3.8-way shared-load and 4.9-way shared-store bank conflicts, but transposed rotation
  scratch and row-grouped gather layouts (groups 2/4/8/16) were flat or up to 11% slower.
- `maxnreg` caps from 128 down to 64 all regressed. A direct packed-load experiment changed 394 of 247,680
  outputs (maximum difference 0.5), so it was rejected on accuracy before performance timing.
- W16 BM16 lost. A raw W4 BM16 result at N=1920/M=257 did not survive the end-to-end selector comparison, where
  `cuda_awq` remained substantially faster.
- Interleaving large modules distorted allocator timing. Repeating N=2560-4096 with sequential module construction
  produced the retained production results above.

### BF16 packed-weight latency hiding

The next GPU 7-only Nsight Compute capture showed BF16 decode remained latency-bound: only 0.42 warps per
scheduler were eligible, schedulers had no eligible warp for 69.07% of cycles, and long-scoreboard stalls consumed
3.02 of the 6.44 warp cycles per issued instruction. DRAM throughput was only 0.76%, so increasing memory bandwidth
was not the target. The retained schedule issues each K-block's packed-weight load before the eight rotation rounds
and consumes it afterward, overlapping its latency with rotation work.

The production gate follows the measured boundary: BF16, `krot=8`, and K>=1024 for decode; prefill additionally
requires N<=512, matching the existing first-partner prefetch regime. FP16, `krot=1`, shorter K, and wider BF16
prefill preserve the prior load order. The paired benchmark runner now exposes this flag independently and requires
exact eager and CUDA-graph output before timing.

All raw comparisons used physical GPU 7, AB/BA-alternating CUDA-graph events, and exact baseline/candidate output.
The main decode result was repeated in reverse declaration order with 8,000 samples:

| BF16 decode shape | Retained p50/mean/p95 (us) | Weight-prefetch p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=2048, 4K | 80.896 / 82.447 / 93.184 | 79.872 / 81.734 / 92.160 | 1.0128x | 1.0087x |
| M=1, K=2048, N=2048, 8K repeat | 80.896 / 81.183 / 80.896 | 79.872 / 80.443 / 80.896 | 1.0128x | 1.0092x |
| M=8, K=2048, N=2048 | 80.896 / 84.512 / 94.208 | 79.872 / 83.455 / 93.184 | 1.0128x | 1.0127x |
| M=1, K=2048, N=512 | 78.848 / 80.816 / 91.136 | 78.848 / 80.142 / 91.136 | 1.0000x | 1.0084x |
| M=1, K=2048, N=8192 | 81.920 / 86.393 / 95.232 | 80.896 / 85.475 / 94.208 | 1.0127x | 1.0107x |
| M=1, K=1024, N=2048 | 44.032 / 46.527 / 51.200 | 44.032 / 46.107 / 50.176 | 1.0000x | 1.0091x |

The same early load improves both retained small-N prefill row tiles:

| BF16 prefill shape | Retained p50/mean/p95 (us) | Weight-prefetch p50/mean/p95 (us) | p50 speedup | mean speedup |
|:---|---:|---:|---:|---:|
| M=128, K=2048, N=512, BM8 | 83.968 / 87.064 / 97.280 | 82.944 / 86.448 / 96.256 | 1.0123x | 1.0071x |
| M=128, K=2048, N=256, BM8 | 87.040 / 88.203 / 100.352 | 86.016 / 87.209 / 99.328 | 1.0119x | 1.0114x |
| M=497, K=2048, N=512, BM32 | 161.792 / 164.565 / 187.392 | 160.768 / 163.337 / 186.368 | 1.0064x | 1.0075x |

Nsight Compute confirms that the load moved into useful rotation latency. The candidate increases registers but
does not change the shared-memory footprint or the grid-limited active-warp count:

| Metric | Retained | Weight prefetch |
|:---|---:|---:|
| Replay duration | 126.53 us | 121.73 us |
| Registers / thread | 101 | 126 |
| Dynamic shared memory / CTA | 34.82 KiB | 34.82 KiB |
| Achieved occupancy | 12.53% | 12.42% |
| Warp cycles / issued instruction | 6.44 | 6.25 |
| Long-scoreboard cycles / issued instruction | 3.02 | 2.79 |
| Issued warps / scheduler | 0.31 | 0.32 |
| Executed instructions | 2,974,368 | 2,978,592 |

Normal production-module timing used 200 warmups and 4,000 CUDA-event samples. Both workloads selected the
mega-kernel and kept the established dense/cross-route numerical envelope:

| Regime | Existing p50/mean/p95 (us) | Mega p50/mean/p95 (us) | Mean speedup | Existing -> mega tokens/s |
|:---|---:|---:|---:|---:|
| BF16 decode, M=1/N=2048 | 193.54 / 196.73 / 207.87 | 169.98 / 173.31 / 185.34 | 1.135x | 5,083.2 -> 5,770.1 |
| BF16 prefill, M=128/N=512 | 194.56 / 198.57 / 210.94 | 182.27 / 185.60 / 199.68 | 1.070x | 644,593 -> 689,659 |

Decode existing/mega maximum error versus dense was 6.0 for both routes, with cross-route maximum/mean error
0.0625/0.000031. Prefill existing/mega maximum error versus dense was 8.0 for both routes, with cross-route
maximum/mean error 4.0/0.010193.

The boundary probes explain the narrow gate:

- BF16 K=512 was flat at p50/p95 and regressed mean from 29.965 to 30.149 us.
- FP16 decode regressed p50/mean/p95 from 92.160/94.174/106.496 to 94.208/96.751/109.568 us.
- BF16 `krot=1` regressed p50/mean from 52.224/50.530 to 56.320/54.820 us.
- Wide FP16 BM32 improved mean only 0.3%, below the retained threshold, so it keeps the prior load order.
- Wide-prefill BN64, BN256, and BM64 schedules were exact but 1.50-1.98x slower than retained BM32/BN128/W16.

### BF16 decode one-launch split-K schedule

After packed-weight prefetch, BF16 decode was limited by grid size rather than bandwidth. At M=1/K=2048/N=2048,
the standard schedule launched only 16 CTAs on 124 SMs (0.06 waves/SM). Nsight Compute reported 12.42% achieved
occupancy, 2.00 active warps and 0.42 eligible warps per scheduler, while DRAM throughput was only 0.79%.

The retained split-K schedule launches one CTA per K=128 quantization group and output tile. CTAs write FP32
partials to caller-owned scratch, publish completion through an acquire/release counter, and the last CTA reduces
all 16 partials in fixed K order, adds bias, writes BF16 output, and resets the counter. This remains one native
Triton kernel launch. Scratch is isolated by CUDA stream and actual row count; CUDA graph capture uses the standard
mega-kernel because replaying shared counters across graph instances is not yet proven safe.

The production gate is deliberately limited to the measured 124-SM `sm_80` target, BF16, `krot=8`, K=2048,
M=1-8, N in {512, 2048, 8192}, and eager inference. FP16, other K/N values, other runtime SM counts, and graph
capture preserve the prior schedule. `GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_DECODE_SPLITK=0` disables the path.

Raw AB/BA CUDA-graph measurements isolate the native launch:

| BF16 decode shape | Standard p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | Mean speedup |
|:---|---:|---:|---:|
| M=1, K=2048, N=512 | 88.064 / 84.308 / 91.136 | 15.360 / 15.090 / 16.384 | 5.587x |
| M=1, K=2048, N=2048 | 79.872 / 83.291 / 93.184 | 19.456 / 20.204 / 22.528 | 4.123x |
| M=1, K=2048, N=8192 | 80.896 / 84.457 / 94.208 | 40.960 / 42.077 / 47.104 | 2.007x |

The initial integration through the fully validated public wrapper used two identically populated production
modules with alternating eager AB/BA order. Physical GPU 7 used 100 warmups and 4,000 samples per path:

| Shape | Standard p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s | Scratch |
|:---|---:|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 192.512 / 191.948 / 209.920 | 167.936 / 162.009 / 184.320 | 1.146x / 1.185x | 5,209.8 -> 6,172.5 | 32.0 KiB |
| M=1, K=2048, N=2048 | 205.824 / 197.897 / 220.160 | 166.912 / 160.553 / 186.368 | 1.233x / 1.233x | 5,053.1 -> 6,228.5 | 128.1 KiB |
| M=1, K=2048, N=8192 | 206.848 / 197.723 / 222.208 | 171.008 / 162.737 / 189.440 | 1.210x / 1.215x | 5,057.6 -> 6,144.9 | 512.2 KiB |
| M=2, K=2048, N=2048 | 192.512 / 185.100 / 205.824 | 165.888 / 162.397 / 181.248 | 1.160x / 1.140x | 10,805.0 -> 12,315.5 | 256.1 KiB |
| M=4, K=2048, N=2048 | 192.512 / 185.437 / 205.824 | 164.864 / 158.905 / 181.248 | 1.168x / 1.167x | 21,570.6 -> 25,172.3 | 512.1 KiB |
| M=8, K=2048, N=2048 | 193.536 / 194.505 / 211.968 | 165.888 / 167.519 / 190.464 | 1.167x / 1.161x | 41,130.1 -> 47,755.6 | 1,024.1 KiB |

The paired benchmark checks a BF16 dense reference before timing. Across the six rows above, the split path never
increased dense-reference maximum error; cross-schedule mismatches affected 0-12 values, maximum absolute
difference was 2.0, and mean difference was at most 0.000810. A separate 32-input randomized sweep at
M=1/K=N=2048 found no dense maximum-error increase, at most 0.002075 additional mean error, and at most four
changed BF16 units in any element. Committed tests enforce 1% relative/2.0 absolute closeness, mean difference
<=0.003, and a dense maximum/mean envelope no wider than +2.0/+0.003.

Nsight Compute confirms that split-K converts the underfilled launch into useful parallelism:

| Metric | Standard | Split-K16 |
|:---|---:|---:|
| Replay duration | 121.73 us | 19.68 us |
| Grid CTAs | 16 | 256 |
| Waves / SM | 0.06 | 0.52 |
| Registers / thread | 126 | 64 |
| Dynamic shared memory / CTA | 34.82 KiB | 34.82 KiB |
| Achieved occupancy | 12.42% | 24.97% |
| Active warps / scheduler | 2.00 | 3.98 |
| Eligible warps / scheduler | 0.42 | 0.78 |
| Issued warps / scheduler | 0.32 | 0.41 |
| Memory / DRAM throughput | 4.32% / 0.79% | 27.42% / 4.91% |

#### Cached compiled-launch continuation

Once split-K reduced device work, Python/Triton dispatch became the dominant interval. Alternating 2,000-sample
layer attribution at M=1/K=N=2048 measured:

| Dispatch layer | GPU-event p50/mean/p95 (us) | Host p50/mean/p95 (us) |
|:---|---:|---:|
| Full module | 174.080 / 179.543 / 196.608 | 149.865 / 154.897 / 171.016 |
| Mega-kernel method | 144.384 / 152.361 / 165.888 | 122.083 / 129.028 / 141.209 |
| Public validated split wrapper | 114.688 / 119.602 / 130.048 | 92.186 / 96.734 / 106.052 |
| Module-validated direct JIT launch plus allocation | 82.944 / 86.608 / 96.256 | 61.226 / 64.114 / 72.147 |
| Same launch with benchmark-only reused output | 70.656 / 71.770 / 83.968 | 51.183 / 50.687 / 60.645 |

The output allocation remains per-call to preserve PyTorch tensor ownership. The retained module path instead
skips duplicate public-wrapper validation after its stricter production gate. That change is bit-exact and improves
paired module mean by 1.112x-1.128x across N=512/2048/8192 and M=8.

Triton still repeated specialization binding and cache lookup on every warm launch. On Triton 3.7 only, the module
now retains the `CompiledKernel` returned by the first normal JIT launch and calls its current-stream launcher
directly on later eager calls. It preserves Triton launch metadata/hooks and the CUDA device guard. The internal ABI
is gated to Triton 3.7 with feature inspection; older/newer Triton, debug/instrumented JIT modes, active pre-run
hooks, ABI errors, and `GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_DECODE_COMPILED_LAUNCH=0` use the normal JIT
launcher. The process-local compiled object is cleared by `_apply()` and excluded from pickle/state serialization.
CUDA graph capture still uses the standard mega-kernel.

Against the module-validated JIT launcher, the compiled launcher improves paired mean by 1.245x-1.297x at
M=1/N=512/2048/8192. The final repository benchmark uses the same dense accuracy gates, 100 warmups, 4,000 AB/BA
samples, and identically populated modules:

| Shape | Standard p50/mean/p95 (us) | Final split-K16 p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 202.752 / 195.366 / 215.040 | 126.976 / 121.235 / 145.408 | 1.597x / 1.611x | 5,118.6 -> 8,248.4 |
| M=1, K=2048, N=2048 | 205.824 / 194.466 / 221.184 | 126.976 / 118.943 / 143.360 | 1.621x / 1.635x | 5,142.3 -> 8,407.4 |
| M=1, K=2048, N=8192 | 191.488 / 186.260 / 207.872 | 125.952 / 119.092 / 140.288 | 1.520x / 1.564x | 5,368.8 -> 8,396.8 |
| M=2, K=2048, N=2048 | 194.560 / 189.970 / 207.872 | 128.000 / 125.253 / 144.384 | 1.520x / 1.517x | 10,528.0 -> 15,967.7 |
| M=4, K=2048, N=2048 | 191.488 / 185.262 / 205.824 | 124.928 / 119.566 / 140.288 | 1.533x / 1.549x | 21,591.1 -> 33,454.3 |
| M=8, K=2048, N=2048 | 194.560 / 189.919 / 207.872 | 128.000 / 126.124 / 143.360 | 1.520x / 1.506x | 42,123.1 -> 63,429.9 |

The launcher changes no device code: final dense and cross-schedule accuracy is identical to the first split-K
table. Repeated JIT/compiled launches are bit-exact, the compiled cache is stream-independent while scratch remains
stream-owned, and a warmed module can be pickled with the process-local launcher omitted and rebuilt on demand.

A synchronized 10,000-launch stress loop left every counter at zero. Focused execution covers repeated launches,
all retained M/N shapes, a dense reference, two concurrent CUDA streams with distinct scratch, graph capture and
replay fallback, the environment toggle, dtype fallback, and a non-124-SM runtime inventory. Compute Sanitizer
memcheck, racecheck, and synccheck passes on the final source reported zero errors, hazards, and warnings.

Two follow-ups were rejected before integration: prefetching the first coefficient vector increased raw latency by
about 37%, while an earlier packed-weight source-order variant was neutral. The retained split kernel keeps the
same rotation/dequantization source order as the standard BF16 schedule.

#### Empty-hook and warmed-dispatch continuation

The cached Triton 3.7 launcher was still passing its default empty `HookChain` objects into every launch. Although
the chains contained no callbacks, the generated launcher entered Python twice and rebuilt launch metadata.
Alternating 7,000-sample direct-launch timing at M=1/K=N=2048 measured
51.200/52.490/58.368 us p50/mean/p95 with the empty chains and 47.104/48.161/54.272 us when they were represented
as inactive, a 1.087x p50 gain. The retained path checks both chains on every call: empty Triton 3.7 chains use
`None`, while a real profiler or user callback receives the original `LazyDict` metadata and enter/exit calls.

Stream-owned scratch selection now uses the same Triton runtime stream handle that launches the compiled kernel.
On this runtime the raw handle query costs about 0.152 us versus 4.639 us for constructing a public
`torch.cuda.Stream`; missing/incompatible Triton driver APIs fall back to the public PyTorch path. Concurrent-stream
tests still allocate distinct scratch, and graph capture remains on the standard mega-kernel. Warmed forward
dispatch also reuses the exact cached plan, directly calls the selected mega-kernel, and shares one flattened input
with the adapter path. A paired 7,000-sample prototype improved full-module p50 by 1.017x without changing cold
autotune, training/grad behavior, or the existing two-level fallback and cache demotion.

A block-row/warp sweep found one additional device-side exception. Only M=1/N=8192 retained a 4-row/4-warp CTA:
with inactive hooks it improved the direct compiled launch from 59.392/60.246/66.560 to
57.344/57.891/64.512 us p50/mean/p95 (1.036x/1.041x p50/mean). The same schedule was flat or noisy at the other
five production gates, which keep the established 8-row/8-warp specialization.

The final paired eager benchmark used physical GPU 7, BF16, dense-reference checks before timing, 100 warmups, and
4,000 alternating AB/BA samples:

| Shape | Standard p50/mean/p95 (us) | Final split-K16 p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 188.416 / 184.926 / 204.800 | 108.544 / 105.991 / 122.880 | 1.736x / 1.745x | 5,407.6 -> 9,434.7 |
| M=1, K=2048, N=2048 | 197.632 / 188.200 / 209.920 | 107.520 / 101.853 / 120.832 | 1.838x / 1.848x | 5,313.5 -> 9,818.1 |
| M=1, K=2048, N=8192 | 190.464 / 190.205 / 209.920 | 111.616 / 107.572 / 122.880 | 1.706x / 1.768x | 5,257.5 -> 9,296.1 |
| M=2, K=2048, N=2048 | 187.392 / 180.323 / 197.632 | 106.496 / 100.355 / 118.784 | 1.760x / 1.797x | 11,091.2 -> 19,929.2 |
| M=4, K=2048, N=2048 | 187.392 / 179.962 / 199.680 | 106.496 / 99.782 / 118.784 | 1.760x / 1.804x | 22,226.9 -> 40,087.4 |
| M=8, K=2048, N=2048 | 187.392 / 181.486 / 199.680 | 106.496 / 101.469 / 119.808 | 1.760x / 1.789x | 44,080.5 -> 78,841.9 |

Relative to the prior compiled-launch artifact, split-K p50 falls from 124.928-128.000 us to
106.496-111.616 us. The dense and cross-schedule accuracy table is unchanged: maximum cross error is 2.0, maximum
mean cross error is 0.000810, and neither dense maximum nor mean error exceeds the committed envelope.

#### Live-buffer and host-grid continuation

Post-commit Python profiling found that registered-buffer lookup had become the next host bottleneck. Fifty
thousand warmed calls attributed about 10.9 us to rotation metadata and 7.8 us to scale/bias dtype checks; most of
that time was repeated `nn.Module.__getattr__` resolution for `pairs`, `theta`, `qweight`, `qzeros`, `scales`, and
`bias`. The retained path reads these tensors from the module's live `_buffers` mapping on every call. It therefore
avoids stale aliases while still observing buffer replacement, `.data`/storage changes, and in-place version
updates. The rotation key now also covers `channel_scales`; a change clears the typed rotation cache before launch.

The compiled launcher also replaced two host-side `triton.cdiv` calls with identical integer ceiling arithmetic.
The former calls passed through Triton's constexpr argument-unwrapping machinery despite operating only on Python
integers. At M=1/K=N=2048, an 8,000-sample repeat moved module p50/mean/p95 from
101.376/95.474/115.712 us after live-buffer lookup to 91.136/89.317/103.424 us after the host-grid change. Finally,
scratch selection returns its already validated raw stream handle to the compiled launcher. That removed a second
driver query; p50 remained in the same event bucket while mean improved from 89.317 to 87.458 us.

The final paired eager benchmark again used physical GPU 7, BF16 dense-reference checks before timing, 100 warmups,
and 4,000 alternating AB/BA samples:

| Shape | Standard p50/mean/p95 (us) | Final split-K16 p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 177.152 / 178.430 / 192.512 | 90.112 / 87.981 / 102.400 | 1.966x / 2.028x | 5,604.4 -> 11,366.1 |
| M=1, K=2048, N=2048 | 180.224 / 176.268 / 194.560 | 89.088 / 85.470 / 102.400 | 2.023x / 2.062x | 5,673.2 -> 11,699.9 |
| M=1, K=2048, N=8192 | 181.248 / 179.520 / 197.632 | 93.184 / 89.673 / 105.472 | 1.945x / 2.002x | 5,570.4 -> 11,151.7 |
| M=2, K=2048, N=2048 | 177.152 / 177.108 / 190.464 | 89.088 / 91.732 / 103.424 | 1.989x / 1.931x | 11,292.5 -> 21,802.5 |
| M=4, K=2048, N=2048 | 177.152 / 176.221 / 189.440 | 89.088 / 86.565 / 100.352 | 1.989x / 2.036x | 22,698.7 -> 46,207.9 |
| M=8, K=2048, N=2048 | 177.152 / 175.507 / 190.464 | 89.088 / 88.498 / 102.400 | 1.989x / 1.983x | 45,582.2 -> 90,397.1 |

Compared with the preceding empty-hook continuation, p50 falls another 15.7-20.2%. The numerical table remains
unchanged: maximum cross error is 2.0 and maximum mean cross error is 0.000810. A dedicated warm-launch regression
replaces scale, bias, and channel-scale buffers in FP16, mutates theta in place, requires runtime BF16 conversion,
and compares the split result with the independent standard mega-kernel before checking counter reset.

#### Zero-scratch native-launch continuation

After the integer-grid cleanup, cProfile still attributed 53 ms across 4,000 warm launches to Triton's generated
`CudaLauncher.__call__`, including repeated Python checks and two zero-sized scratch-allocation calls before its
native `cuda_utils.launch`. The split-K specialization has no compiler-managed global or profile scratch. On the
exact Triton 3.7 ABI, the compiled-launch capability check now verifies both scratch sizes are zero and verifies
every native-launch field before calling the generated C launcher directly. Other Triton ABIs or launchers with
compiler scratch retain the normal JIT fallback. Empty and active launch hooks keep the existing behavior, and a
current-device check bypasses the Python device context only when the weight's device is already active.

An in-process paired prototype at M=1/K=N=2048 was bit-exact and reduced p50 from about 94-95 us to 86-88 us.
The final six-shape benchmark used physical GPU 7, BF16 dense-reference checks before timing, 100 warmups, and
4,000 alternating standard/split samples:

| Shape | Standard p50/mean/p95 (us) | Direct split-K16 p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 177.152 / 175.002 / 190.464 | 84.992 / 83.914 / 96.256 | 2.084x / 2.085x | 5,714.2 -> 11,916.9 |
| M=1, K=2048, N=2048 | 182.272 / 182.248 / 196.608 | 86.016 / 83.283 / 98.304 | 2.119x / 2.188x | 5,487.0 -> 12,007.2 |
| M=1, K=2048, N=8192 | 181.248 / 183.263 / 195.584 | 90.112 / 89.860 / 100.352 | 2.011x / 2.039x | 5,456.6 -> 11,128.5 |
| M=2, K=2048, N=2048 | 180.224 / 178.379 / 191.488 | 87.040 / 85.722 / 98.304 | 2.071x / 2.081x | 11,212.1 -> 23,331.3 |
| M=4, K=2048, N=2048 | 178.176 / 176.780 / 189.440 | 84.992 / 83.247 / 97.280 | 2.096x / 2.124x | 22,627.0 -> 48,049.6 |
| M=8, K=2048, N=2048 | 180.224 / 179.583 / 191.488 | 88.064 / 85.195 / 98.304 | 2.047x / 2.108x | 44,547.5 -> 93,902.2 |

Relative to the preceding artifact, p50 falls another 3.3-5.7% at every production gate. Mean improves 2.6-6.6%
at five gates; N=8192 moves from 89.673 to 89.860 us, a reported 0.2% noisy reversal, while its p50 and p95 both
improve. The dense and cross-schedule accuracy table is unchanged: maximum cross error is 2.0 and maximum mean
cross error is 0.000810. The raw artifact is
`artifacts/paroquant_megakernel_20260723/splitk_direct_launcher_candidate_ab.json`.

#### Caller-shaped output continuation

The common model call passes a contiguous three-dimensional `[batch, sequence, K]` activation, but the module
previously allocated the split-K result as `[M, N]` and returned a final reshape view. The internal split prepare
and compiled helpers now accept the caller's output dimensions and allocate the owning result at that shape.
Adapters, other execution plans, unsupported split gates, and the measured N=8192 exception retain the existing
two-dimensional result plus final reshape. Repeated results are distinct allocations with no `_base`, so this
removes a view without introducing output aliasing.

A same-process 5,000-sample baseline/candidate comparison was bit-exact and improved p50 by 3.7-5.1% and mean by
1.7-5.4% across M=1/N=512 and M=1/2/4/8/N=2048. N=8192 instead regressed by one or two event buckets in four
8,000-sample repeats, so it explicitly keeps the prior output path. cProfile confirms the retained shapes execute
one reshape per call instead of two. The final paired eager benchmark used physical GPU 7, 300 warmups, BF16 dense
checks before timing, and 4,000 alternating standard/split samples:

| Shape | Standard p50/mean/p95 (us) | Shaped split-K16 p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 178.176 / 178.157 / 189.440 | 81.920 / 78.380 / 91.136 | 2.175x / 2.273x | 5,613.0 -> 12,758.4 |
| M=1, K=2048, N=2048 | 178.176 / 177.884 / 196.608 | 80.896 / 77.064 / 93.184 | 2.203x / 2.308x | 5,621.6 -> 12,976.2 |
| M=1, K=2048, N=8192 | 180.224 / 176.744 / 190.464 | 90.112 / 86.622 / 98.304 | 2.000x / 2.040x | 5,657.9 -> 11,544.4 |
| M=2, K=2048, N=2048 | 181.248 / 178.921 / 193.536 | 82.944 / 80.199 / 93.184 | 2.185x / 2.231x | 11,178.1 -> 24,937.8 |
| M=4, K=2048, N=2048 | 178.176 / 174.307 / 188.416 | 79.872 / 76.829 / 90.112 | 2.231x / 2.269x | 22,948.1 -> 52,063.4 |
| M=8, K=2048, N=2048 | 177.152 / 176.073 / 187.392 | 79.872 / 77.033 / 89.088 | 2.218x / 2.286x | 45,435.8 -> 103,851.0 |

The first full-table attempt observed a transient whole-device slowdown only in its opening N=512 case: both
standard and split p50 rose to 1,236.992 and 728.064 us. An immediate 8,000-sample N=512 repeat returned to
183.296/179.154/195.584 us standard and 83.968/79.372/93.184 us split, and the complete longer-warmup repeat above
was stable. Both artifacts remain recorded rather than discarding the anomaly. The final dense/cross accuracy table
is unchanged: maximum cross error is 2.0 and maximum mean cross error is 0.000810.

#### Graph-owned split-K continuation

Warmed CUDA graphs previously captured the standard mega-kernel because the eager counter/partial cache could not
be shared safely by multiple graph replays. A capture now uses split-K only when the exact compiled-launch plan is
already cached and still passes its Triton ABI/instrumentation checks. Each capture allocates new partials and
zeroed counters in its CUDA graph-private allocator pool. The zero becomes a captured reset node, the split kernel
resets counters again on completion, and no graph scratch enters the eager stream cache. Cold captures, disabled
compiled launch, unsupported Triton ABIs, and failed capability checks retain the standard mega-kernel.

The regression test captures two graphs from the same module, requires distinct partial pointers, alternates three
replays, launches both graphs concurrently on separate streams, checks exact equality with eager split-K, and
verifies every graph and eager counter is zero. It then clears the compiled plan and requires a cold capture to
return the bit-exact standard result. The production benchmark holds no Python graph-scratch references and
completed thousands of exact replays, exercising normal graph-pool lifetime management.

`scripts/benchmark_paroquant_splitk.py --cuda-graph` now reproduces the paired graph measurement. The final GPU 7
run used dense and eager-split accuracy checks before capture, 300 replay warmups, and 4,000 alternating graph
replays:

| Shape | Standard graph p50/mean/p95 (us) | Split-K graph p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 78.848 / 78.931 / 78.848 | 15.360 / 15.188 / 15.360 | 5.133x / 5.197x | 12,669.3 -> 65,841.6 |
| M=1, K=2048, N=2048 | 79.872 / 79.882 / 79.872 | 20.480 / 21.029 / 21.504 | 3.900x / 3.799x | 12,518.5 -> 47,552.5 |
| M=1, K=2048, N=8192 | 80.896 / 81.277 / 81.920 | 38.912 / 38.908 / 38.912 | 2.079x / 2.089x | 12,303.6 -> 25,701.7 |
| M=2, K=2048, N=2048 | 79.872 / 80.312 / 79.872 | 20.480 / 20.904 / 21.504 | 3.900x / 3.842x | 24,902.8 -> 95,674.6 |
| M=4, K=2048, N=2048 | 79.872 / 80.011 / 79.872 | 21.504 / 21.180 / 21.504 | 3.714x / 3.778x | 49,993.3 -> 188,854.0 |
| M=8, K=2048, N=2048 | 79.872 / 80.197 / 79.872 | 21.504 / 21.865 / 22.528 | 3.714x / 3.668x | 99,754.5 -> 365,886.0 |

The six graph outputs reproduce the unchanged dense/cross accuracy table: maximum cross error is 2.0 and maximum
mean cross error is 0.000810. A separate 8,000-replay M=1/N=2048 repeat measured
79.872/80.231/80.896 us standard versus 20.480/21.000/21.504 us split.

#### Direct caller-input continuation

After graph integration, cProfile attributed 33 ms across 10,000 eager M=1/N=2048 calls to the input reshape and
additional time to repeating warmed classification and plan-key construction. The validated Triton C launcher
consumes a tensor data pointer plus explicit M/K values; its device kernel does not depend on the Python tensor
rank. A narrow warmed route now passes the original contiguous activation and its explicit row count directly to
the cached launcher. It requires evaluation/inference mode, no adapter, BF16, K=2048, krot=8, M=1-8,
N in {512, 2048, 8192}, the cached decode-mega plan, and an existing compatible compiled launcher. Cold/JIT calls,
noncontiguous inputs, training/autograd, adapters, other shapes/dtypes, and failed compiled launches retain the
established flattening and two-level fallback.

The regression wraps the real compiled launch and requires two three-dimensional calls to receive the exact caller
tensor object plus explicit M=1, while keeping distinct owning, bit-exact outputs. Existing tests still cover
active hooks, current-device mismatch, live-buffer replacement/mutation, JIT disablement, forced launch failure,
two eager streams, two concurrent graph replays, and cold graph fallback. The direct input is also captured
normally: a six-shape graph repeat remained exact and measured the same 15-39 us replay bands.

An alternating same-process comparison used identical modules, 300 warmups, and 5,000 samples. The baseline
retained the caller-shaped output optimization but constructed the input view; the candidate was bit-exact at all
six gates:

| Shape | Input-view p50/mean/p95 (us) | Direct-input p50/mean/p95 (us) | p50 / mean speedup |
|:---|---:|---:|---:|
| M=1, K=2048, N=512 | 88.064 / 98.085 / 100.352 | 80.896 / 90.336 / 93.184 | 1.089x / 1.086x |
| M=1, K=2048, N=2048 | 86.016 / 92.863 / 98.304 | 80.896 / 87.184 / 93.184 | 1.063x / 1.065x |
| M=1, K=2048, N=8192 | 93.184 / 97.610 / 103.424 | 90.112 / 92.524 / 99.328 | 1.034x / 1.055x |
| M=2, K=2048, N=2048 | 86.016 / 91.203 / 97.280 | 80.896 / 85.868 / 92.160 | 1.063x / 1.062x |
| M=4, K=2048, N=2048 | 84.992 / 93.357 / 97.280 | 78.848 / 88.979 / 90.112 | 1.078x / 1.049x |
| M=8, K=2048, N=2048 | 83.968 / 93.948 / 96.256 | 78.848 / 90.145 / 91.136 | 1.065x / 1.042x |

The smallest N=8192 p50 gain was repeated with 500 warmups and 10,000 samples:
88.064/98.817/99.328 us input-view versus 87.040/95.249/97.280 us direct-input, or 1.012x p50 and 1.037x mean.
Post-change cProfile reduced total traced time for 10,000 M=1/N=2048 calls from 0.754 to 0.688 seconds and removed
all 10,000 input reshape calls.

The reproducible production benchmark then ran the unchanged dense/cross checks, 300 warmups, and 4,000
alternating standard/split samples:

| Shape | Standard p50/mean/p95 (us) | Direct-input split p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 180.224 / 188.502 / 198.656 | 74.752 / 84.422 / 88.064 | 2.411x / 2.233x | 5,305.0 -> 11,845.2 |
| M=1, K=2048, N=2048 | 181.248 / 184.138 / 196.608 | 73.728 / 73.410 / 86.016 | 2.458x / 2.508x | 5,430.7 -> 13,622.2 |
| M=1, K=2048, N=8192 | 181.248 / 179.437 / 194.560 | 86.016 / 83.869 / 93.184 | 2.107x / 2.139x | 5,573.0 -> 11,923.3 |
| M=2, K=2048, N=2048 | 180.224 / 186.011 / 192.512 | 74.752 / 72.665 / 82.944 | 2.411x / 2.560x | 10,752.1 -> 27,523.6 |
| M=4, K=2048, N=2048 | 181.248 / 181.705 / 193.536 | 74.752 / 75.313 / 83.968 | 2.425x / 2.413x | 22,013.7 -> 53,111.6 |
| M=8, K=2048, N=2048 | 183.296 / 195.034 / 198.656 | 76.800 / 77.765 / 88.064 | 2.387x / 2.508x | 41,018.5 -> 102,874.0 |

The dense/cross table is unchanged: maximum cross error is 2.0 and maximum mean cross error is 0.000810.

#### Validated last-scratch continuation

The direct caller-input route has already validated every split-K invariant and queried the current raw stream.
Eager calls now compare that device/stream/M/N tuple with the last scratch entry before constructing a cache key
and probing the general per-stream dictionary. A miss still uses and refreshes the dictionary, so alternating or
concurrent streams retain distinct allocations. Capture-state detection runs first: graph-private scratch never
enters or replaces the eager last-hit entry. The alias is cleared by `_apply()`/rotation-cache reset and omitted
from serialized module state; it does not allocate or own additional CUDA storage.

A switchable in-process comparison used six alternating blocks of 1,500 samples per route. The last hit reduced
p50 by exactly one 1.024 us event bucket at all six production gates:

| Shape | Keyed scratch p50/mean (us) | Last scratch p50/mean (us) | p50 / mean speedup |
|:---|---:|---:|---:|
| M=1, K=2048, N=512 | 75.776 / 83.236 | 74.752 / 84.189 | 1.014x / 0.989x |
| M=1, K=2048, N=2048 | 75.776 / 81.822 | 74.752 / 80.688 | 1.014x / 1.014x |
| M=1, K=2048, N=8192 | 89.088 / 99.865 | 88.064 / 98.300 | 1.012x / 1.016x |
| M=2, K=2048, N=2048 | 75.776 / 82.821 | 74.752 / 82.024 | 1.014x / 1.010x |
| M=4, K=2048, N=2048 | 74.752 / 82.767 | 73.728 / 84.220 | 1.014x / 0.983x |
| M=8, K=2048, N=2048 | 75.776 / 86.513 | 74.752 / 82.696 | 1.014x / 1.046x |

The two noisy mean reversals were repeated rather than hidden. Ten alternating blocks of 2,500 samples moved
M=1/N=512 from 74.752/82.184 us keyed to 73.728/80.641 us last-hit (1.014x/1.019x p50/mean). M=4/N=2048 was
effectively flat: 74.752/84.257 versus 74.752/84.117 us, with block-mean medians 85.082 versus 83.434 us.

The final production run used the unchanged dense/cross accuracy gate, 500 warmups, and 5,000 alternating
standard/split samples:

| Shape | Standard p50/mean/p95 (us) | Last-scratch split p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 181.248 / 192.172 / 196.608 | 75.776 / 77.756 / 87.040 | 2.392x / 2.471x | 5,203.7 -> 12,860.8 |
| M=1, K=2048, N=2048 | 185.344 / 194.796 / 201.728 | 75.776 / 76.556 / 87.040 | 2.446x / 2.544x | 5,133.6 -> 13,062.3 |
| M=1, K=2048, N=8192 | 183.296 / 196.069 / 209.920 | 87.040 / 88.994 / 101.376 | 2.106x / 2.203x | 5,100.2 -> 11,236.7 |
| M=2, K=2048, N=2048 | 183.296 / 191.836 / 215.040 | 77.824 / 83.369 / 94.208 | 2.355x / 2.301x | 10,425.6 -> 23,989.9 |
| M=4, K=2048, N=2048 | 182.272 / 186.947 / 196.608 | 75.776 / 75.508 / 87.040 | 2.405x / 2.476x | 21,396.4 -> 52,974.7 |
| M=8, K=2048, N=2048 | 184.320 / 198.111 / 199.680 | 77.824 / 80.300 / 90.112 | 2.368x / 2.467x | 40,381.4 -> 99,626.8 |

A six-shape graph repeat remained exact at 15.343-38.931 us mean. The regression requires the eager last entry to
keep its original storage before and after two private graph captures and excludes the alias from serialization.

#### FP16 one-launch split-K decode continuation

The same launch-underfill mechanism also limited FP16 decode, but the original split kernel deliberately converted
both Tensor Core operands to BF16. The retained kernel now specializes its dot operands by caller dtype: BF16 keeps
the established BF16 conversion and numerical contract, while FP16 keeps the rotated activation and dequantized
weight in FP16. The counter protocol, fixed-order FP32 reduction, output allocation, graph-private scratch, direct
caller-input path, and validated Triton 3.7 native launcher are otherwise shared.

The FP16 production gate is intentionally exact rather than range-based: K=2048, `krot=8`, the measured 124-SM
`sm_80` runtime, and M/N in {(1, 512), (1, 2048), (1, 8192), (2, 2048), (4, 2048), (8, 2048)}. Other FP16
shapes retain the standard mega-kernel. FP16 uses its existing local `int16` partner table; BF16 keeps local
`int8`. Compiled-launch keys now include dtype so a warmed BF16 specialization can never be reused for FP16, while
the dtype-independent FP32 partials and int32 counters remain safely reusable on the same stream.

A corrected raw configuration harness now launches the requested split tile, warp count, stages, and split factor
directly; the previous split variant silently called the production wrapper and therefore ignored those requested
geometry fields. Exact/cross-reference checks run before every timed candidate. The retained FP16 schedules are:

| Shape | Split row tile / warps | Standard p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | Mean speedup |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 8 / 8 | 90.112 / 94.181 / 104.448 | 14.336 / 14.955 / 17.408 | 6.30x |
| M=1, K=2048, N=2048 | 2 / 4 | 92.160 / 95.511 / 106.496 | 17.408 / 17.562 / 19.456 | 5.44x |
| M=1, K=2048, N=8192 | 2 / 4 | 93.184 / 95.467 / 106.496 | 34.816 / 36.081 / 39.936 | 2.65x |
| M=2, K=2048, N=2048 | 4 / 4 | 92.160 / 96.735 / 106.496 | 17.408 / 18.057 / 20.480 | 5.36x |
| M=4, K=2048, N=2048 | 4 / 4 | 93.184 / 94.921 / 107.520 | 17.408 / 17.714 / 20.480 | 5.36x |
| M=8, K=2048, N=2048 | 8 / 8 | 93.184 / 94.536 / 107.520 | 20.480 / 20.480 / 23.552 | 4.62x |

The reproducible full-module eager benchmark used physical GPU 7, 500 warmups, 6,000 alternating samples, and
dense plus standard/split accuracy checks before timing:

| Shape | Standard p50/mean/p95 (us) | FP16 split p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 191.488 / 196.659 / 207.872 | 76.800 / 78.914 / 91.136 | 2.493x / 2.492x | 5,085 -> 12,672 |
| M=1, K=2048, N=2048 | 193.536 / 192.543 / 206.848 | 77.824 / 76.792 / 90.112 | 2.487x / 2.507x | 5,194 -> 13,022 |
| M=1, K=2048, N=8192 | 193.536 / 201.893 / 208.896 | 86.016 / 81.518 / 97.280 | 2.250x / 2.477x | 4,953 -> 12,267 |
| M=2, K=2048, N=2048 | 193.536 / 198.363 / 209.920 | 77.824 / 81.494 / 93.184 | 2.487x / 2.434x | 10,083 -> 24,542 |
| M=4, K=2048, N=2048 | 192.512 / 198.420 / 209.920 | 77.824 / 79.225 / 92.160 | 2.474x / 2.505x | 20,159 -> 50,489 |
| M=8, K=2048, N=2048 | 194.560 / 192.427 / 207.872 | 78.848 / 74.837 / 90.112 | 2.468x / 2.571x | 41,574 -> 106,900 |

The CUDA-graph repeat used 300 warmups and 5,000 alternating replays:

| Shape | Standard graph p50/mean/p95 (us) | FP16 split graph p50/mean/p95 (us) | p50 speedup | Split tokens/s |
|:---|---:|---:|---:|---:|
| M=1, K=2048, N=512 | 90.112 / 90.198 / 90.112 | 15.360 / 15.547 / 16.384 | 5.867x | 64,321 |
| M=1, K=2048, N=2048 | 92.160 / 91.913 / 92.160 | 18.432 / 18.061 / 18.432 | 5.000x | 55,368 |
| M=1, K=2048, N=8192 | 93.184 / 93.285 / 93.184 | 36.864 / 36.652 / 36.864 | 2.528x | 27,284 |
| M=2, K=2048, N=2048 | 92.160 / 92.126 / 92.160 | 18.432 / 19.195 / 18.432 | 5.000x | 104,196 |
| M=4, K=2048, N=2048 | 91.136 / 91.527 / 91.136 | 18.432 / 18.917 / 19.456 | 4.944x | 211,448 |
| M=8, K=2048, N=2048 | 92.160 / 92.686 / 93.184 | 21.504 / 21.249 / 21.504 | 4.286x | 376,481 |

Neither route increased maximum error versus the dense FP16 reference; the dense maximum remained 0.25 or 0.5
at every shape. Standard/split mean absolute differences were 0.000526-0.001139, maximum difference was at most
0.5, and 6-167 output values differed depending on shape. The full 329-test ParoQuant suite passed on physical GPU
7 immediately before commit. Compute Sanitizer memcheck and synccheck passed the six FP16 parity shapes, FP16 dense
envelope, and dtype-isolation case with zero errors; racecheck passed representative M=1/M=4 shapes plus
dtype-isolated cache reuse with zero hazards.

#### FP16 irregular-row split-K continuation

The first FP16 split-K gate covered the six common power-of-two and projection shapes. A follow-up screen measured
every missing N=2048 decode row count rather than extrapolating the launch schedule. M=3 retains a 4-row/4-warp
CTA; M=5-7 use an 8-row/4-warp CTA. M=8 deliberately keeps its previously validated 8-warp schedule. The
production FP16 N=2048 gate therefore now covers every M=1-8 row count, while N=512/8192 remain single-row only.

Exact raw-kernel measurements on physical GPU 7 used 50 eager warmups, 200 graph warmups, 5,000 alternating
samples, and standard/split accuracy checks before timing:

| Shape | Retained split tile / warps | Standard p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | Mean speedup |
|:---|---:|---:|---:|---:|
| M=3, K=2048, N=2048 | 4 / 4 | 91.136 / 94.341 / 106.496 | 17.408 / 17.637 / 20.480 | 5.349x |
| M=5, K=2048, N=2048 | 8 / 4 | 92.160 / 95.846 / 107.520 | 19.456 / 19.949 / 22.528 | 4.805x |
| M=6, K=2048, N=2048 | 8 / 4 | 92.160 / 96.115 / 107.520 | 19.456 / 20.148 / 22.528 | 4.771x |
| M=7, K=2048, N=2048 | 8 / 4 | 93.184 / 96.498 / 107.520 | 19.456 / 20.576 / 23.552 | 4.690x |

The full-module eager benchmark used 500 warmups and 6,000 alternating samples:

| Shape | Standard p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | p50 / mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=3, K=2048, N=2048 | 194.560 / 187.805 / 207.872 | 79.872 / 73.519 / 92.160 | 2.436x / 2.555x | 15,974 -> 40,806 |
| M=5, K=2048, N=2048 | 198.656 / 193.166 / 210.944 | 81.920 / 76.938 / 93.184 | 2.425x / 2.511x | 25,885 -> 64,988 |
| M=6, K=2048, N=2048 | 197.632 / 193.633 / 208.896 | 80.896 / 75.042 / 91.136 | 2.443x / 2.580x | 30,987 -> 79,955 |
| M=7, K=2048, N=2048 | 197.632 / 200.138 / 211.968 | 81.920 / 82.817 / 94.208 | 2.413x / 2.417x | 34,976 -> 84,524 |

The graph-owned path used 300 warmups and 5,000 alternating replays:

| Shape | Standard graph p50/mean/p95 (us) | Split-K graph p50/mean/p95 (us) | p50 / mean speedup | Split tokens/s |
|:---|---:|---:|---:|---:|
| M=3, K=2048, N=2048 | 91.136 / 91.748 / 92.160 | 18.432 / 18.695 / 19.456 | 4.944x / 4.908x | 160,474 |
| M=5, K=2048, N=2048 | 92.160 / 92.839 / 93.184 | 20.480 / 20.562 / 21.504 | 4.500x / 4.515x | 243,170 |
| M=6, K=2048, N=2048 | 92.160 / 92.578 / 93.184 | 20.480 / 20.651 / 21.504 | 4.500x / 4.483x | 290,540 |
| M=7, K=2048, N=2048 | 92.160 / 92.628 / 93.184 | 21.504 / 21.194 / 21.504 | 4.286x / 4.371x | 330,290 |

The dense-reference maximum error remained unchanged at 0.5 or 1.0 for every new shape. Cross-route mean error
was 0.000594-0.000913 and maximum error was at most 0.5. Focused route and FP16/BF16 parity coverage passed 45
tests before the full-suite accuracy gate. Compute Sanitizer memcheck and synccheck passed all ten measured FP16
split shapes with zero errors; racecheck passed representative new M=3/M=7 shapes with zero hazards. The final
pre-commit run passed all 351 ParoQuant tests, including dense envelopes at FP16/BF16 M=1/3/7 and warmed compiled
module execution at every new FP16 row count.

#### FP16 projection-width split-K continuation

The next decode pass measured every previously uncovered FP16 row at the small K/V projection width (`N=512`) and
wide gate projection width (`N=8192`). The production FP16 split-K gate now covers the complete Cartesian product
`M=1-8`, `K=2048`, `N in {512, 2048, 8192}`, and `krot=8` on the runtime-probed 124-SM `sm_80` target. The
existing architecture, SM-count, dtype, shape, capture, stream, and fallback checks remain unchanged.

The measured small-width schedule is BM8/W8 for every row count. Wide `N=8192` uses BM2/W4 for M=1-2, BM4/W4 for
M=3-4, and BM8/W8 for M=5-8. The N=2048 schedules retained by the preceding passes are unchanged. Accuracy checks
ran before every configuration timing. Selected raw CUDA-graph measurements on physical GPU 7 were:

| Shape | Split BM/warps | Standard p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | Mean speedup |
|:---|---:|---:|---:|---:|
| M=2, K=2048, N=512 | 8 / 8 | 90.112 / 97.046 / 104.448 | 14.336 / 15.004 / 16.384 | 6.468x |
| M=3, K=2048, N=512 | 8 / 8 | 103.424 / 101.188 / 104.448 | 16.384 / 16.077 / 17.408 | 6.294x |
| M=4, K=2048, N=512 | 8 / 8 | 90.112 / 96.976 / 104.448 | 15.360 / 15.432 / 16.384 | 6.284x |
| M=5, K=2048, N=512 | 8 / 8 | 103.424 / 99.299 / 104.448 | 16.384 / 15.787 / 17.408 | 6.290x |
| M=6, K=2048, N=512 | 8 / 8 | 103.424 / 97.172 / 104.448 | 16.384 / 15.968 / 17.408 | 6.086x |
| M=7, K=2048, N=512 | 8 / 8 | 90.112 / 95.959 / 104.448 | 15.360 / 15.696 / 17.408 | 6.114x |
| M=8, K=2048, N=512 | 8 / 8 | 90.112 / 95.662 / 104.448 | 15.360 / 16.059 / 17.408 | 5.957x |
| M=2, K=2048, N=8192 | 2 / 4 | 93.184 / 95.991 / 108.544 | 35.840 / 36.997 / 41.984 | 2.595x |
| M=3, K=2048, N=8192 | 4 / 4 | 106.496 / 101.054 / 107.520 | 41.984 / 39.771 / 43.008 | 2.541x |
| M=4, K=2048, N=8192 | 4 / 4 | 93.184 / 95.520 / 107.520 | 36.864 / 37.773 / 43.008 | 2.529x |
| M=5, K=2048, N=8192 | 8 / 8 | 111.616 / 116.206 / 130.048 | 39.936 / 41.265 / 46.080 | 2.816x |
| M=6, K=2048, N=8192 | 8 / 8 | 112.640 / 117.308 / 131.072 | 39.936 / 42.018 / 47.104 | 2.792x |
| M=7, K=2048, N=8192 | 8 / 8 | 113.664 / 116.455 / 130.048 | 40.960 / 41.919 / 47.104 | 2.778x |
| M=8, K=2048, N=8192 | 8 / 8 | 112.640 / 116.737 / 131.072 | 41.984 / 42.651 / 48.128 | 2.737x |

The final full-module eager run used 500 warmups and 6,000 alternating samples per route. It exercised the native
compiled launcher and included the dense and standard/split accuracy gates before timing:

| Shape | Standard p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | Mean speedup | Standard -> split tokens/s |
|:---|---:|---:|---:|---:|
| M=2, K=2048, N=512 | 193.536 / 197.573 / 210.944 | 79.872 / 79.265 / 93.184 | 2.493x | 10,123 -> 25,232 |
| M=3, K=2048, N=512 | 192.512 / 197.559 / 206.848 | 78.848 / 79.350 / 92.160 | 2.490x | 15,185 -> 37,807 |
| M=4, K=2048, N=512 | 195.584 / 191.716 / 208.896 | 81.920 / 79.415 / 94.208 | 2.414x | 20,864 -> 50,368 |
| M=5, K=2048, N=512 | 194.560 / 202.369 / 214.016 | 81.920 / 81.812 / 97.280 | 2.474x | 24,707 -> 61,116 |
| M=6, K=2048, N=512 | 195.584 / 199.219 / 212.992 | 81.920 / 80.300 / 98.304 | 2.481x | 30,118 -> 74,720 |
| M=7, K=2048, N=512 | 193.536 / 190.835 / 208.896 | 80.896 / 77.011 / 95.232 | 2.478x | 36,681 -> 90,896 |
| M=8, K=2048, N=512 | 193.536 / 194.244 / 208.896 | 81.920 / 80.084 / 95.232 | 2.425x | 41,185 -> 99,895 |
| M=2, K=2048, N=8192 | 197.632 / 204.201 / 215.040 | 90.112 / 90.189 / 102.400 | 2.264x | 9,794 -> 22,176 |
| M=3, K=2048, N=8192 | 196.608 / 192.773 / 209.920 | 90.112 / 81.617 / 100.352 | 2.362x | 15,562 -> 36,757 |
| M=4, K=2048, N=8192 | 197.632 / 206.275 / 232.448 | 90.112 / 90.411 / 114.688 | 2.282x | 19,392 -> 44,243 |
| M=5, K=2048, N=8192 | 215.040 / 213.354 / 230.400 | 93.184 / 85.967 / 103.424 | 2.482x | 23,435 -> 58,162 |
| M=6, K=2048, N=8192 | 215.040 / 203.374 / 242.688 | 93.184 / 82.351 / 109.568 | 2.470x | 29,502 -> 72,859 |
| M=7, K=2048, N=8192 | 215.040 / 212.063 / 230.400 | 92.160 / 80.678 / 102.400 | 2.629x | 33,009 -> 86,765 |
| M=8, K=2048, N=8192 | 216.064 / 208.912 / 230.400 | 94.208 / 79.779 / 103.424 | 2.619x | 38,294 -> 100,277 |

The graph-owned path used 300 warmups and 5,000 alternating replays:

| Shape | Standard graph p50/mean/p95 (us) | Split-K graph p50/mean/p95 (us) | Mean speedup | Split tokens/s |
|:---|---:|---:|---:|---:|
| M=2, K=2048, N=512 | 90.112 / 90.145 / 90.112 | 15.360 / 15.626 / 16.384 | 5.769x | 127,995 |
| M=3, K=2048, N=512 | 90.112 / 90.122 / 90.112 | 15.360 / 15.511 / 16.384 | 5.810x | 193,412 |
| M=4, K=2048, N=512 | 90.112 / 90.028 / 90.112 | 15.360 / 15.752 / 16.384 | 5.715x | 253,940 |
| M=5, K=2048, N=512 | 90.112 / 90.313 / 90.112 | 15.360 / 15.832 / 16.384 | 5.704x | 315,811 |
| M=6, K=2048, N=512 | 90.112 / 89.964 / 90.112 | 16.384 / 16.043 / 16.384 | 5.608x | 374,004 |
| M=7, K=2048, N=512 | 90.112 / 90.131 / 90.112 | 16.384 / 16.163 / 16.384 | 5.576x | 433,076 |
| M=8, K=2048, N=512 | 90.112 / 89.971 / 90.112 | 16.384 / 16.239 / 16.384 | 5.541x | 492,654 |
| M=2, K=2048, N=8192 | 93.184 / 93.344 / 93.184 | 36.864 / 37.259 / 37.888 | 2.505x | 53,678 |
| M=3, K=2048, N=8192 | 93.184 / 93.170 / 93.184 | 37.888 / 37.527 / 37.888 | 2.483x | 79,942 |
| M=4, K=2048, N=8192 | 93.184 / 92.975 / 93.184 | 37.888 / 37.895 / 38.912 | 2.453x | 105,554 |
| M=5, K=2048, N=8192 | 111.616 / 112.437 / 113.664 | 40.960 / 41.091 / 41.984 | 2.736x | 121,680 |
| M=6, K=2048, N=8192 | 112.640 / 112.746 / 113.664 | 41.984 / 42.026 / 43.008 | 2.683x | 142,770 |
| M=7, K=2048, N=8192 | 112.640 / 113.164 / 113.664 | 41.984 / 42.195 / 43.008 | 2.682x | 165,898 |
| M=8, K=2048, N=8192 | 112.640 / 113.184 / 113.664 | 41.984 / 42.412 / 43.008 | 2.669x | 188,628 |

The dense-reference maximum error stayed exactly unchanged at 0.25-1.0. Standard/split maximum difference was at
most 0.5 and mean absolute difference was 0.000467-0.001090. Per-shape scratch is 64-256 KiB for N=512 and
1.0-4.0 MiB for N=8192; no new scratch layout or lifetime rule was introduced.

The focused accuracy gate passed 101 tests, including every M=1-8/N=512/2048/8192 split shape in both FP16 and
BF16 and five warmed production-path cases at the new widths. Compute Sanitizer memcheck and synccheck passed all
24 FP16 split shapes with zero errors. Racecheck passed representative M=2/N=512, M=3/N=8192, and M=8/N=8192
cases with zero hazards/errors/warnings. The final pre-commit suite passed all 407 ParoQuant tests with 16 warnings
on physical GPU 7.

The exact benchmark payloads are intentionally uncommitted:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_widths_module_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_widths_graph_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{2,3,4,5,6,7,8}_n512*_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{2,3,4,5,6,7,8}_n8192*_screen.json
```

#### FP16 N=8192 split-K output-tile continuation

Nsight Compute attribution on the retained FP16 M=8/N=8192 split-K16 kernel showed a latency/tail-bound launch
rather than a DRAM-bound one. The profiled launch used 64 registers/thread and 34.82 KiB dynamic shared memory,
reached 42.95% achieved occupancy versus 50% theoretical occupancy, and issued 2.06 waves/SM: two full waves plus
a 32-CTA tail. SM and memory throughput were 47.14% and 47.84%, but DRAM throughput was only 7.53%. Schedulers had
no eligible warp for 43.13% of cycles. This motivated reusing each rotated activation tile across twice as much
output instead of reducing the split factor.

The split-factor control confirmed that split-K16 remains necessary. At M=2/4/8, split-K16 p50 was
35.840/36.864/40.960 us, while split-K8 measured 40.960/40.960/49.152 us; factors four and two were slower again.
The candidate therefore keeps split 16 and widens only the output tile from BN128/stage-2 to BN256/stage-1. The
selector is restricted to FP16, M=5-8, K=2048, N=8192, and the existing 124-SM `sm_80` production gate. FP16
M=2-4, BF16, all other widths, and every fallback retain BN128/stage-2.

Accuracy ran before each screen. A 500-eager/300-graph warmup and 6,000-sample alternating confirmation produced:

| Shape | BN128 p50/mean/p95 (us) | BN256 p50/mean/p95 (us) | p50 / mean improvement |
|:---|---:|---:|---:|
| M=5, K=2048, N=8192 | 39.936 / 40.474 / 46.080 | 38.912 / 39.490 / 45.056 | 2.6% / 2.5% |
| M=6, K=2048, N=8192 | 39.936 / 40.225 / 40.960 | 38.912 / 39.300 / 39.936 | 2.6% / 2.4% |
| M=7, K=2048, N=8192 | 39.936 / 40.459 / 40.960 | 38.912 / 39.456 / 39.936 | 2.6% / 2.5% |
| M=8, K=2048, N=8192 | 40.960 / 40.959 / 41.984 | 39.936 / 40.109 / 40.960 | 2.5% / 2.1% |

BN128 and BN256 were bit-identical to each other at every retained shape. Relative to the standard mega-kernel,
both had maximum absolute difference 0.5 and mean difference 0.000585-0.000629 in the paired raw confirmation.
BN256 lost at M=2-4 and is deliberately excluded there.

The complete eager module remains dominated by Python launch work but still improved over the standard route by
2.57-2.67x on mean, reaching 66,655-108,910 tokens/s. The graph-owned complete path preserved the device gain:

| Shape | Prior BN128 graph p50/mean (us) | BN256 graph p50/mean/p95 (us) | Mean improvement | BN256 tokens/s |
|:---|---:|---:|---:|---:|
| M=5, K=2048, N=8192 | 40.960 / 41.091 | 39.936 / 40.105 / 40.960 | 2.5% | 124,672 |
| M=6, K=2048, N=8192 | 41.984 / 42.026 | 39.936 / 40.465 / 40.960 | 3.9% | 148,276 |
| M=7, K=2048, N=8192 | 41.984 / 42.195 | 40.960 / 40.611 / 40.960 | 3.9% | 172,365 |
| M=8, K=2048, N=8192 | 41.984 / 42.412 | 41.984 / 41.614 / 41.984 | 1.9% | 192,242 |

The production accuracy run kept dense-reference maximum error exactly unchanged at 0.5. Standard/BN256 maximum
difference was 0.5 and mean absolute difference was 0.000629-0.000816. Scratch partial storage remains the same
size because halving output-tile count offsets doubling BN. Counter storage stays conservatively sized for BN128,
so the dtype-agnostic scratch cache remains safe when FP16 BN256 and BF16 BN128 reuse the same allocation.

All 48 FP16/BF16 M=1-8 by N=512/2048/8192 split-K parity cases passed, as did both warmed FP16 production
boundaries and explicit FP16-to-BF16 N=8192 scratch reuse. The complete suite passed all 415 tests with 16 warnings.
Compute Sanitizer memcheck and synccheck passed M=5/M=8 plus the cross-dtype scratch test with zero errors;
racecheck passed the warmed M=8/N=8192 production path with zero hazards/errors/warnings.

The same continuation rejected a BM64/stage-1 raw prefill candidate at M=384/N=2048: it improved the fused
mega-kernel from 277.504 to 228.352 us p50, but the complete module selected the established CUDA AWQ route at
about 202.752 us. No prefill routing changed.

The exact payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m{2,3,4,5,6,7,8}_n8192_bn256_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{5,6,7,8}_n8192_bn256_confirm.json
artifacts/paroquant_megakernel_20260723/fp16_split_factor_m{2,4,8}_n8192_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_bn256_module.json
artifacts/paroquant_megakernel_20260723/fp16_split_bn256_graph.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_profile.ncu-rep
artifacts/paroquant_megakernel_20260723/prefill_n2048_m384_bm64_screen.json
```

#### FP16 N=8192 rotated-activation reuse

A second Nsight Compute pass on the retained FP16 M=8 BN256 kernel refined the next bottleneck. The 512-CTA grid
formed 2.06 waves/SM, used 110 registers/thread and 67.58 KiB dynamic shared memory, and was limited to two
blocks/SM (25% theoretical, 23.69% achieved occupancy). Compute and memory throughput were only 29.26% and
31.93%; DRAM was 7.48%. Schedulers had no eligible warp for 58.44% of cycles, with long scoreboard accounting for
about 30.7% of issue spacing. Source attribution placed the largest samples after the rotation metadata loads.
L1/L2 hit rates were 63.61%/58.64%, there were no spills, and the main excess traffic was partial reduction plus
Tensor Core shared-memory staging. This favored reusing rotated activations over making the BN256 tile still wider.

The retained kernel processes two sequential logical BN128 output tiles per CTA. It preloads both packed-weight
tiles, rotates the activation once, then performs two dequantize/dot/reduction phases. The logical partial and
counter layout remains BN128, so FP16 and BF16 can continue to share dtype-agnostic scratch safely. The selector is
restricted to FP16 M=5-8, K=2048, N=8192, split-K16. Its one-K-block-per-split invariant is explicit in the host
schedule; M=1-4, BF16, other dimensions, and other split factors retain the original one-output-tile kernel.

Accuracy ran before timing. The M=5-7 screens used 6,000 alternating graph samples, and the final production M=8
screen repeated the comparison through the cleaned benchmark interface:

| Shape | BN256 p50/mean/p95 (us) | Two BN128 tiles p50/mean/p95 (us) | p50 / mean improvement |
|:---|---:|---:|---:|
| M=5, K=2048, N=8192 | 38.912 / 39.768 / 45.056 | 37.888 / 38.393 / 44.032 | 2.6% / 3.5% |
| M=6, K=2048, N=8192 | 38.912 / 39.326 / 39.936 | 37.888 / 37.957 / 38.912 | 2.6% / 3.5% |
| M=7, K=2048, N=8192 | 38.912 / 39.876 / 43.008 | 37.888 / 38.681 / 41.984 | 2.6% / 3.0% |
| M=8, K=2048, N=8192 | 39.936 / 40.896 / 46.080 | 38.912 / 39.718 / 45.056 | 2.6% / 2.9% |

The two-output-tile candidate was bit-identical to BN256 at every row count. Relative to the standard mega-kernel,
maximum absolute difference remained 0.5 and mean absolute difference remained 0.000585-0.000629.

The complete graph-owned route improved at every retained shape versus the prior BN256 artifact:

| Shape | BN256 graph p50/mean (us) | Two-tile graph p50/mean/p95 (us) | Mean improvement | Two-tile tokens/s |
|:---|---:|---:|---:|---:|
| M=5, K=2048, N=8192 | 39.936 / 40.105 | 38.912 / 38.709 / 39.936 | 3.5% | 129,170 |
| M=6, K=2048, N=8192 | 39.936 / 40.465 | 38.912 / 39.066 / 39.936 | 3.5% | 153,584 |
| M=7, K=2048, N=8192 | 40.960 / 40.611 | 38.912 / 39.467 / 40.960 | 2.8% | 177,362 |
| M=8, K=2048, N=8192 | 41.984 / 41.614 | 39.936 / 39.842 / 40.960 | 4.3% | 200,795 |

The host-dispatch-dominated eager route stayed in the same 93-94 us p50 band. M=5-7 improved modestly versus the
prior artifact, while M=8 reversed by one 1.024 us event bucket and had a noisier mean; no eager-host improvement
is claimed. The kernel is retained on the paired raw and complete CUDA-graph wins.

Wider BN512 tiles, split-K8, and four-warp BN256 variants all regressed. Loading channel scales earlier left p50
flat and worsened mean (40.866 -> 41.123 us); marking rotation metadata `evict_last` regressed
39.936/40.229/45.056 to 40.960/41.161/46.080 us. All four experiments were reverted. The final 416-test GPU 7
suite passed, including all 48 FP16/BF16 M=1-8 by N=512/2048/8192 parity cases, dense-error envelopes, compiled
launch reuse, CUDA graphs, concurrent streams, and FP16/BF16 scratch reuse. Compute Sanitizer memcheck and
synccheck passed M=5/M=8 plus cross-dtype scratch reuse with zero errors; racecheck passed M=8 with zero
hazards/errors/warnings.

The post-change detailed NCU report measured 80 registers/thread, 34.82 KiB dynamic shared memory, 29.79%
achieved occupancy, and 1.38 waves/SM. Replay duration fell from BN256's 49.06 us to 44.19 us, while compute and
memory throughput rose to 34.80% and 38.22%. The grid remained latency-limited: long-scoreboard samples were still
the largest stall class, and registers limited residency to three CTAs/SM. A narrowly applied `maxnreg=78` cap
reduced profiled duration again to 43.17 us and raised achieved occupancy slightly to 30.03%, without changing the
three-CTA residency limit.

The register cap was screened independently at every retained row count after accuracy:

| Shape | Uncapped p50/mean/p95 (us) | maxnreg=78 p50/mean/p95 (us) | p50 / mean improvement |
|:---|---:|---:|---:|
| M=5, K=2048, N=8192 | 36.864 / 37.687 / 43.008 | 35.840 / 36.599 / 40.960 | 2.8% / 2.9% |
| M=6, K=2048, N=8192 | 37.888 / 38.090 / 39.936 | 36.864 / 36.821 / 37.888 | 2.7% / 3.3% |
| M=7, K=2048, N=8192 | 37.888 / 38.219 / 39.936 | 36.864 / 36.902 / 37.888 | 2.7% / 3.4% |
| M=8, K=2048, N=8192 | 37.888 / 38.544 / 39.936 | 36.864 / 37.106 / 37.888 | 2.7% / 3.7% |

Caps 68-76 also improved the uncapped kernel but did not beat 78 consistently. A 64-register cap regressed to
39.936/40.619/46.080 us and is rejected. The production selector returns the cap only for the same exact
FP16/M=5-8/K=2048/N=8192/split-K16 schedule; every fallback returns no register cap.

The complete graph-owned route retained the improvement:

| Shape | Uncapped graph p50/mean (us) | maxnreg=78 graph p50/mean/p95 (us) | Mean improvement | Final tokens/s |
|:---|---:|---:|---:|---:|
| M=5, K=2048, N=8192 | 38.912 / 38.709 | 37.888 / 37.613 / 38.912 | 2.8% | 132,934 |
| M=6, K=2048, N=8192 | 38.912 / 39.066 | 37.888 / 37.744 / 38.912 | 3.4% | 158,967 |
| M=7, K=2048, N=8192 | 38.912 / 39.467 | 37.888 / 37.949 / 38.912 | 3.8% | 184,457 |
| M=8, K=2048, N=8192 | 39.936 / 39.842 | 37.888 / 38.379 / 39.936 | 3.7% | 208,449 |

Accuracy remained identical to the uncapped two-tile kernel and within the same dense/standard envelope. The
complete 416-test suite passed again on GPU 7. Memcheck and synccheck passed M=5/M=8 plus cross-dtype scratch reuse
with zero errors; racecheck passed M=8 with zero hazards/errors/warnings. The configuration benchmark now exposes
an explicit optional register cap and reestablishes empty atomic counters after NCU kernel replay before graph
validation.

The exact payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m{5,6,7}_n8192_bn128x2_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_two_n_tiles_production_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_two_n_tiles_module.json
artifacts/paroquant_megakernel_20260723/fp16_split_two_n_tiles_graph.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_bn512_split_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_bn256_w4_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_earlyscale_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_bn256_evictlast_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_bn256_profile.ncu-rep
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_bn256_memory_profile.ncu-rep
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_two_n_tiles_profile.ncu-rep
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_two_n_tiles_maxr78_profile.ncu-rep
artifacts/paroquant_megakernel_20260723/fp16_split_m{5,6,7}_n8192_two_n_tiles_maxr78_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_two_n_tiles_maxnreg_{screen,bracket,fine}.json
artifacts/paroquant_megakernel_20260723/fp16_split_two_n_tiles_maxr78_graph.json
```

#### FP16 N=8192 paired completion counters

The next output-reuse screens kept the two-logical-BN128 schedule. Two sequential BN256 tiles were exact but
regressed the M=8 raw kernel from 36.864/37.637/43.008 us to 51.200/51.973/59.392 us
p50/mean/p95; a 96-register cap recovered only 46.080/46.639/53.248 us. Raising the two-tile pipeline depth also
lost: stage one measured 36.864/37.813/41.984 us, versus 38.912/39.260/44.032 at stage two and
38.912/40.068/45.056 at stage three.

Reusing one rotation across four logical BN128 tiles preserved the current split-K result exactly but increased
M=8 latency from 36.864/37.773/43.008 us to 46.080/47.164/54.272 us. Register caps could not close the gap:
the best tested BN128x4 result was 41.984/42.634/48.128 us at 78 registers. Four BN64 tiles with four warps and
the same cap reached 41.984/42.491/44.032 us, still slower than the current two-tile kernel. All four-tile source
and routing changes were removed.

The retained change instead combines synchronization for the existing two logical output tiles. Every CTA now
stores both FP32 partial tiles before one release/acquire completion atomic. The last split CTA acquires all prior
writes, reduces both tiles in the unchanged split-0-through-15 order, writes both outputs, and resets one counter.
This halves completion atomic adds and pair-counter resets without changing the partial layout, reduction order,
scratch ownership, graph behavior, or dtype reuse. The selector remains exact to FP16 M=5-8, K=2048, N=8192,
split-K16 on the existing 124-SM `sm_80` route. M=1-4, BF16, other dimensions, split factors, devices, and CPU
fallbacks retain their prior kernels.

Accuracy ran before every timed screen. A paired 4,000-sample row confirmation selected `maxnreg=76`:

| Shape | Prior per-tile counters p50/mean/p95 (us) | Paired counter p50/mean/p95 (us) | p50 / mean improvement |
|:---|---:|---:|---:|
| M=5, K=2048, N=8192 | 35.840 / 36.921 / 41.984 | 34.816 / 35.597 / 39.936 | 2.9% / 3.6% |
| M=6, K=2048, N=8192 | 36.864 / 37.260 / 40.960 | 34.816 / 35.817 / 38.912 | 5.6% / 3.9% |
| M=7, K=2048, N=8192 | 36.864 / 37.145 / 41.984 | 35.840 / 36.164 / 40.960 | 2.8% / 2.6% |
| M=8, K=2048, N=8192 | 36.864 / 38.167 / 43.008 | 35.840 / 36.653 / 40.960 | 2.8% / 4.0% |

Every paired-counter output was bit-identical to the prior split-K kernel. Relative to the standard mega-kernel,
maximum absolute difference stayed 0.5 and raw mean absolute difference stayed 0.000585-0.000629. The complete
dense comparison also remained unchanged: dense-reference maximum error was 0.5 and standard/split cross mean
absolute difference was 0.000629-0.000816.

The complete graph-owned route improved at all four production rows:

| Shape | Prior graph p50/mean (us) | Paired-counter graph p50/mean/p95 (us) | Mean improvement | Final tokens/s |
|:---|---:|---:|---:|---:|
| M=5, K=2048, N=8192 | 37.888 / 37.613 | 35.840 / 36.168 / 36.864 | 3.8% | 138,245 |
| M=6, K=2048, N=8192 | 37.888 / 37.744 | 35.840 / 36.418 / 36.864 | 3.5% | 164,752 |
| M=7, K=2048, N=8192 | 37.888 / 37.949 | 36.864 / 36.782 / 37.888 | 3.1% | 190,313 |
| M=8, K=2048, N=8192 | 37.888 / 38.379 | 36.864 / 36.921 / 37.888 | 3.8% | 216,681 |

The detailed M=8 Nsight Compute replay used the same 512-CTA grid and 256-thread blocks. The retained cap produced
76 registers/thread with no local spills, 34.82 KiB dynamic shared memory, 37.5% theoretical and 29.95% achieved
occupancy, and 1.38 waves/SM. Replay duration was 42.91 us, versus 43.17 us in the prior maxnreg=78 basic report
and 44.19 us in the uncapped detailed report. Compute and memory throughput reached 35.75% and 39.76%; DRAM
remained only 8.54%. L1/TEX and L2 throughput were 52.02% and 15.49%, with 64.19%/57.54% hit rates.

The final pre-commit accuracy suite passed all 420 ParoQuant tests with 16 warnings on physical GPU 7. This
includes explicit M=5/6/7/8 N=8192 production calls, FP16/BF16 parity and dense-error envelopes, compiled reuse,
concurrent streams, live-buffer updates, private CUDA graph captures, and dtype-agnostic scratch reuse. Compute
Sanitizer memcheck and synccheck passed M=5/M=8 plus cross-dtype scratch reuse with zero errors; racecheck passed
M=8 with zero hazards/errors/warnings. Ruff and `git diff --check` were clean. The configuration benchmark now
exposes the paired-counter choice explicitly.

The exact payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_bn256x2_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_two_n_tiles_maxr78_stage_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_bn128x4{,_maxnreg}_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_four_tiles_resource_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_counter{,_maxnreg_fine}_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{5,6,7,8}_n8192_pair_counter_rows.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_counter_r76_profile.{json,ncu-rep}
artifacts/paroquant_megakernel_20260723/fp16_split_pair_counter_r76_graph.json
```

#### FP16 paired-counter second-weight lifetime

The retained paired-counter kernel previously loaded both packed BN128 weight tiles before eight activation
rotations. A compile-time benchmark switch showed that keeping only the first tile in flight across rotation and
loading the second tile immediately afterward shortens its live range without changing arithmetic. The production
constant disables only this second-tile prefetch in the exact paired-counter schedule; all one-tile/two-tile
fallbacks and the first tile's latency hiding are unchanged.

Accuracy ran before timing, and late/prefetched outputs were bit-identical. The reversed-order 5,000-sample raw
confirmation measured:

| Shape | Prefetch both p50/mean/p95 (us) | Late second tile p50/mean/p95 (us) | Mean improvement |
|:---|---:|---:|---:|
| M=5, K=2048, N=8192 | 34.816 / 36.094 / 40.960 | 34.816 / 35.810 / 39.936 | 0.8% |
| M=6, K=2048, N=8192 | 34.816 / 35.981 / 40.960 | 34.816 / 35.633 / 39.936 | 1.0% |
| M=7, K=2048, N=8192 | 35.840 / 36.232 / 40.960 | 34.816 / 35.902 / 40.960 | 0.9% |
| M=8, K=2048, N=8192 | 35.840 / 36.151 / 40.960 | 35.840 / 35.959 / 40.960 | 0.5% |

The first M=8 screen independently repeated the mean gain, 36.369 -> 36.051 us. The complete graph-owned route
also improved mean latency at every row:

| Shape | Prior paired p50/mean/p95 (us) | Late-load p50/mean/p95 (us) | Mean improvement | Final tokens/s |
|:---|---:|---:|---:|---:|
| M=5, K=2048, N=8192 | 35.840 / 36.168 / 36.864 | 35.840 / 35.928 / 37.888 | 0.7% | 139,168 |
| M=6, K=2048, N=8192 | 35.840 / 36.418 / 36.864 | 35.840 / 35.820 / 36.864 | 1.6% | 167,503 |
| M=7, K=2048, N=8192 | 36.864 / 36.782 / 37.888 | 35.840 / 36.344 / 36.864 | 1.2% | 192,605 |
| M=8, K=2048, N=8192 | 36.864 / 36.921 / 37.888 | 36.864 / 36.547 / 37.888 | 1.0% | 218,894 |

M=5 graph p95 moved one 1.024 us bucket in the losing direction even though its raw p95 improved by one bucket
and both raw and complete means improved; the reversal is reported rather than hidden. Dense/cross accuracy was
unchanged at every shape.

Detailed M=8 NCU replay fell from 42.91 to 41.79 us. The launch stayed spill-free at 76 registers/thread,
34.82 KiB dynamic shared memory, 37.5% theoretical/29.98% achieved occupancy, and 1.38 waves/SM. Compute and
memory throughput rose from 35.75%/39.76% to 36.76%/40.89%; L1/TEX and L2 throughput reached 52.90% and 17.02%.

Before commit, all 420 ParoQuant tests passed with 16 warnings on physical GPU 7. Memcheck and synccheck again
passed M=5/M=8 plus cross-dtype scratch reuse with zero errors, and racecheck passed M=8 with zero
hazards/errors/warnings. Ruff and `git diff --check` were clean.

The exact payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_second_weight_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{5,6,7,8}_n8192_pair_second_weight_rows.json
artifacts/paroquant_megakernel_20260723/fp16_split_pair_counter_late_second_graph.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_late_second_profile.{json,ncu-rep}
```

#### FP16 paired-counter terminal reset

The paired-counter kernel used an atomic exchange to reset a completion counter after the last split-K CTA had
reduced both output tiles. That final atomic is unnecessary: observing `prior_count == SPLIT_K - 1` means every
other CTA has completed its release/acquire atomic add after publishing its two partial tiles, and no CTA in the
launch accesses that counter again. The last CTA can therefore reset the counter with a plain store after writing
the final outputs. Reuse by a later launch is ordered on the same CUDA stream, while concurrent streams and private
CUDA graph captures already own separate scratch buffers. A compile-time benchmark switch retains the old atomic
reset for direct comparison; production uses the plain terminal store only in the exact paired-counter schedule.

Accuracy ran before each timing screen. Atomic- and plain-reset outputs were bit-identical, and the full comparison
remained within the established envelope at every production row: dense-reference and standard/split maximum
absolute differences were 0.5, while standard/split mean absolute difference was
0.000629/0.000816/0.000783/0.000780 for M=5/6/7/8. The clean reversed-order 5,000-sample raw confirmation measured:

| Shape | Atomic reset p50/mean/p95 (us) | Plain reset p50/mean/p95 (us) | Mean improvement |
|:---|---:|---:|---:|
| M=5, K=2048, N=8192 | 34.816 / 35.706 / 39.936 | 33.792 / 35.019 / 39.936 | 1.9% |
| M=6, K=2048, N=8192 | 34.816 / 35.181 / 39.936 | 33.792 / 34.550 / 38.912 | 1.8% |
| M=7, K=2048, N=8192 | 34.816 / 36.415 / 40.960 | 34.816 / 35.808 / 39.936 | 1.7% |
| M=8, K=2048, N=8192 | 35.840 / 36.455 / 40.960 | 34.816 / 35.887 / 40.960 | 1.6% |

The initial 6,000-sample M=8 screen also favored the plain reset at p50 and mean, but contained unrelated system
outliers; the reversed-order screen above is the raw decision run. The complete graph-owned route confirmed lower
mean latency at all four rows:

| Shape | Atomic reset p50/mean/p95 (us) | Plain reset p50/mean/p95 (us) | Mean improvement | Final tokens/s |
|:---|---:|---:|---:|---:|
| M=5, K=2048, N=8192 | 35.840 / 35.928 / 37.888 | 34.816 / 35.110 / 35.840 | 2.3% | 142,411 |
| M=6, K=2048, N=8192 | 35.840 / 35.820 / 36.864 | 35.840 / 35.353 / 35.840 | 1.3% | 169,718 |
| M=7, K=2048, N=8192 | 35.840 / 36.344 / 36.864 | 35.840 / 35.627 / 36.864 | 2.0% | 196,478 |
| M=8, K=2048, N=8192 | 36.864 / 36.547 / 37.888 | 35.840 / 36.087 / 36.864 | 1.3% | 221,686 |

Nsight Compute replay did not reproduce the steady-state win. The plain-reset detailed M=8 replay took 43.14 us,
versus 41.79 us for the earlier late-load atomic-reset report, while retaining 76 registers/thread, 34.82 KiB
dynamic shared memory, no spills, and 30.19% achieved occupancy. A fresh back-to-back basic replay similarly
measured 42.34 us/30.29% achieved occupancy for atomic reset and 43.97 us/29.93% for plain reset. This disagreement
is reported explicitly. The plain reset is retained because both repeated raw screens and the complete
graph-replay workload consistently improved, which better represents inference than serialized profiler replay;
the result should be rechecked if counter placement or launch ordering changes.

Before commit, all 420 ParoQuant tests passed with 16 warnings on physical GPU 7. This includes the full
dense-error envelope, every M=5/6/7/8 N=8192 production call, concurrent streams, private CUDA graph scratch, and
cross-dtype scratch reuse. Compute Sanitizer memcheck and synccheck passed M=5/M=8 plus cross-dtype reuse with zero
errors. Racecheck separately passed the M=8 production path, concurrent streams, and graph-owned scratch with zero
hazards/errors/warnings. Ruff and `git diff --check` were clean. The configuration benchmark exposes the reset
choice explicitly while preserving atomic reset as the default for older variant strings.

The exact payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_reset_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{5,6,7,8}_n8192_pair_reset_rows.json
artifacts/paroquant_megakernel_20260723/fp16_split_pair_plain_reset_graph.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_plain_reset_profile.{json,ncu-rep}
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_{atomic,plain}_reset_basic.ncu-rep
```

#### Post-reset partial-cache policy (rejected)

The next accuracy-first GPU 7 screen tried bypassing L1 for the FP32 partial scratch. Atomic/plain-reset output
accuracy was unchanged for every variant, but all `.cg` policies lost against default caching at M=8, K=2048,
N=8192 over 3,000 alternating graph samples:

| Partial scratch policy | p50/mean/p95 (us) | Mean change versus default |
|:---|---:|---:|
| Default | 34.816 / 35.338 / 39.936 | baseline |
| `.cg` stores | 34.816 / 35.554 / 39.936 | -0.6% |
| `.cg` reduction loads | 34.816 / 35.808 / 40.960 | -1.3% |
| `.cg` stores and loads | 34.816 / 35.858 / 40.960 | -1.5% |

The production cache policy remains unchanged. The exact rejected screen is
`artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_partial_cache_screen.json`.

#### Post-reset pair-counter spacing (rejected)

Padding paired completion counters initially looked promising. Stride two fits the existing 64-element N=8192
counter allocation exactly and beat dense stride one in the accuracy-first 5,000-sample direct confirmations:

| Shape | Stride one mean (us) | Stride two mean (us) | Direct mean improvement |
|:---|---:|---:|---:|
| M=5, K=2048, N=8192 | 34.919 | 34.641 | 0.8% |
| M=6, K=2048, N=8192 | 34.753 | 34.616 | 0.4% |
| M=7, K=2048, N=8192 | 35.033 | 34.966 | 0.2% |
| M=8, K=2048, N=8192 | 35.802 | 35.585 | 0.6% |

However, the complete graph-owned production route reversed at every row versus the published dense-counter
baseline: M=5/6/7/8 means moved from 35.110/35.353/35.627/36.087 us to
35.223/35.482/35.713/36.272 us, regressions of 0.2-0.5%. Larger stride 4-32 variants also failed to beat stride
two in the initial screen. Counter spacing is therefore rejected, production remains dense, and scratch allocation
does not grow.

The exact rejected payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_counter_stride_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{5,6,7,8}_n8192_pair_counter_stride_confirm.json
artifacts/paroquant_megakernel_20260723/fp16_split_pair_counter_stride2_graph.json
```

#### Post-reset final-acquire ordering (rejected)

Replacing all 16 acquire/release counter increments with release-only increments plus one acquire atomic in the
last CTA preserved bit-identical output but did not reduce arrival cost. At M=8, K=2048, N=8192 over 6,000
alternating graph samples, the existing schedule measured 34.816/35.366/39.936 us p50/mean/p95, while the
release-then-acquire schedule measured 34.816/35.451/39.936 us. Production retains acquire/release on each arrival.
The exact rejected payload is
`artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_final_acquire_screen.json`.

#### Post-reset local-partial reuse (rejected)

The final CTA already holds the second output tile's FP32 accumulator when it publishes scratch. Reusing that value
for its fixed-order reduction removes one of 32 partial-tile reloads per pair and remains bit-identical, but extends
the accumulator's register lifetime across the counter arrival. At the retained 76-register cap, M=8 mean latency
regressed from 35.049 to 43.798 us. A bounded cap sweep found no recovery: the best reuse candidate, maxnreg=80,
measured 44.190 us mean versus the screen's 35.787 us baseline, and maxnreg=78/84/88/96 were slower still.
Production continues to reload the local partial from scratch.

The exact rejected payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_reuse_second_partial_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n8192_pair_reuse_second_partial_registers.json
```

#### FP16 N=8192 paired counters across all decode rows

The paired-counter kernel added for M=5-8 also removes duplicated rotations and half of the completion counters at
M=1-4, but the narrower four-warp row tiles need much looser register caps. Accuracy-first cap sweeps selected the
following exact FP16/K=2048/N=8192/split-K16 schedule on the existing runtime-probed 124-SM `sm_80` gate:

| Rows | Row tile / warps / stages | Retained maxnreg |
|---:|---:|---:|
| M=1 | BM2 / W4 / S1 | 128 |
| M=2 | BM2 / W4 / S1 | 160 |
| M=3 | BM4 / W4 / S1 | 160 |
| M=4 | BM4 / W4 / S1 | 120 |

M=5-8 retain BM8/W8/S1/maxnreg=76. BF16, other widths/K/split factors, other devices, and every unsupported path
retain their existing one-tile schedules and fallbacks. The paired grid falls from 1,024 to 512 CTAs without
changing the partial-scratch layout or allocation size.

Every paired result was bit-identical to its prior one-tile split-K result before timing. The balanced
6,000-sample direct confirmations measured:

| Shape | Prior one-tile p50/mean/p95 (us) | Paired p50/mean/p95 (us) | p50 / mean improvement |
|:---|---:|---:|---:|
| M=1, K=2048, N=8192 | 34.816 / 35.411 / 35.840 | 31.744 / 32.085 / 32.768 | 8.8% / 9.4% |
| M=2, K=2048, N=8192 | 35.840 / 36.797 / 41.984 | 31.744 / 33.082 / 36.864 | 11.4% / 10.1% |
| M=3, K=2048, N=8192 | 36.864 / 37.201 / 38.912 | 32.768 / 33.218 / 34.816 | 11.1% / 10.7% |
| M=4, K=2048, N=8192 | 36.864 / 37.306 / 41.984 | 33.792 / 34.505 / 38.912 | 8.3% / 7.5% |

The complete graph-owned production route, with 300 warmups and 6,000 samples per row, reached:

| Shape | Split-K16 p50/mean/p95 (us) | Final tokens/s |
|:---|---:|---:|
| M=1, K=2048, N=8192 | 32.768 / 32.908 / 33.792 | 30,388 |
| M=2, K=2048, N=8192 | 32.768 / 33.216 / 33.792 | 60,212 |
| M=3, K=2048, N=8192 | 33.792 / 33.830 / 34.816 | 88,679 |
| M=4, K=2048, N=8192 | 34.816 / 35.474 / 36.864 | 112,758 |

The ordinary compiled eager-module route reached 11,583/23,596/34,282/47,693 tokens/s for M=1/2/3/4, with
86.330/84.761/87.511/83.870 us mean latency. Graph and eager benchmarks both ran dense and cross-schedule checks
before timing. Across the final production graph cases, dense-reference maximum error remained 0.5 and cross
maximum error was 0.25-0.5; cross mean absolute error was 0.000668/0.000636/0.000768/0.000735 for M=1/2/3/4.

Nsight Compute confirms that this is a launch-width and reuse win rather than spilling hidden by event timing. At
M=3, the old one-tile kernel used 1,024 CTAs, 96 registers/thread, 33.79 KiB dynamic shared memory, and 22.97%
achieved occupancy; replay duration was 46.85 us. The paired maxnreg=160 kernel used 512 CTAs, 160
registers/thread, the same 33.79 KiB dynamic shared memory, and 14.77% achieved occupancy, but reduced replay
duration to 41.82 us (10.7%). Both reports recorded zero local-load and local-store bytes. The M=1 maxnreg=128
endpoint used 512 CTAs, 128 registers/thread, 33.28 KiB dynamic shared memory, and 22.46% achieved occupancy.

Before commit, all 425 ParoQuant tests passed with 16 warnings on physical GPU 7. This includes every
FP16/BF16 M=1-8/N=8192 parity case, dense-error envelopes, explicit M=1-8 warmed compiled paths, graph replay,
concurrent streams, scratch reuse, and fallback gates. Compute Sanitizer memcheck and synccheck passed the new
M=1/M=3/M=4 paths with zero errors; racecheck passed the same paths with zero hazards/errors/warnings. Ruff and
`git diff --check` were clean.

The exact retained payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m4_n8192_pair_counter{,_high}_register_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,2,3}_n8192_pair_counter_high_register_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,2,3,4}_n8192_pair_counter_confirm.json
artifacts/paroquant_megakernel_20260723/fp16_split_pair_counter_all_rows_graph.json
artifacts/paroquant_megakernel_20260723/fp16_split_pair_counter_m1_m4_eager.json
artifacts/paroquant_megakernel_20260723/fp16_split_m1_n8192_pair_r128_basic.{json,ncu-rep}
artifacts/paroquant_megakernel_20260723/fp16_split_m3_n8192_pair_r160_resource.{json,ncu-rep}
artifacts/paroquant_megakernel_20260723/fp16_split_m3_n8192_single_tile_resource.{json,ncu-rep}
```

#### N=2048 paired/stage continuation (rejected)

The paired-counter schedule was also tested at FP16 M=4/K=N=2048 after exact-output validation. The retained
one-tile/stage-2 kernel measured 17.408/17.836/18.432 us p50/mean/p95, while the best paired candidate
(`maxnreg=160`) measured 19.456/19.499/20.480 us. Halving the counter count therefore regressed mean latency by
9.3% at this narrower width and was rejected.

A separate one-tile stage/register continuation did not produce a robust replacement for stage two:

| Shape | Retained stage-2 p50/mean/p95 (us) | Stage-1/maxnreg=96 p50/mean/p95 (us) | Outcome |
|:---|---:|---:|:---|
| M=1, K=N=2048 | 16.384 / 17.206 / 19.456 | 17.408 / 17.210 / 19.456 | p50 regression |
| M=4, K=N=2048 | 17.408 / 17.986 / 20.480 | 17.408 / 17.812 / 20.480 | flat tails; 1.0% mean only |
| M=6, K=N=2048 | 19.456 / 19.859 / 22.528 | 19.456 / 19.797 / 22.528 | 0.3% mean only |
| M=8, K=N=2048 | 20.480 / 20.591 / 23.552 | 21.504 / 22.434 / 25.600 | clear regression |

Production therefore keeps one output tile, two stages, and no register cap at N=2048. The rejected payloads are:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m4_n2048_pair_counter_register_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m4_n2048_one_tile_stage_register_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,4,6,8}_n2048_one_tile_s1_r96_confirm.json
```

#### FP16 paired-counter first-weight lifetime

The paired N=8192 kernel previously kept its first packed BN128 weight tile live across all eight activation
rotations. Delaying that load until rotation completes reduces its live range. The effect depends on the row tile:
M=1-5 improve, while M=6-8 are neutral or slightly slower and retain prefetch. Register caps were re-screened
after shortening the live range, producing this exact FP16/K=2048/N=8192/split-K16 schedule:

| Rows | Row tile / warps / stages | First weight | Retained maxnreg |
|---:|:---|:---|---:|
| M=1 | BM2 / W4 / S1 | late | 136 |
| M=2 | BM2 / W4 / S1 | late | 168 |
| M=3 | BM4 / W4 / S1 | late | 144 |
| M=4 | BM4 / W4 / S1 | late | 120 |
| M=5 | BM8 / W8 / S1 | late | 76 |
| M=6-8 | BM8 / W8 / S1 | prefetched | 76 |

The selector remains behind the existing runtime-probed 124-SM `sm_80`, FP16, K=2048, N=8192, split-K16 gate.
BF16, other shapes, other devices, and every fallback keep their previous generated kernels.

Every selected result was bit-exact against the prior paired schedule. The benchmark runner now has an
`--exact-pair` gate that fails before timing on any element mismatch. Reversed-order 8,000-sample confirmations
measured:

| Shape | Prior prefetch p50/mean/p95 (us) | Selected late-load p50/mean/p95 (us) | p50 / mean improvement |
|:---|---:|---:|---:|
| M=1, K=2048, N=8192 | 31.744 / 31.746 / 32.768 | 30.720 / 31.038 / 31.744 | 3.2% / 2.2% |
| M=2, K=2048, N=8192 | 31.744 / 32.084 / 32.768 | 31.744 / 31.711 / 31.744 | 0.0% / 1.2% |
| M=3, K=2048, N=8192 | 32.768 / 32.762 / 32.768 | 31.744 / 32.045 / 32.768 | 3.1% / 2.2% |
| M=4, K=2048, N=8192 | 33.792 / 34.184 / 34.816 | 32.768 / 33.086 / 33.792 | 3.0% / 3.2% |
| M=5, K=2048, N=8192 | 33.792 / 34.263 / 34.816 | 33.792 / 34.109 / 34.816 | 0.0% / 0.4% |

The complete graph-owned production path ran the unchanged dense/cross accuracy gate before 300 warmups and
6,000 samples:

| Shape | Final graph p50/mean/p95 (us) | Final tokens/s |
|:---|---:|---:|
| M=1, K=2048, N=8192 | 31.744 / 31.844 / 32.768 | 31,403 |
| M=2, K=2048, N=8192 | 31.744 / 32.418 / 32.768 | 61,695 |
| M=3, K=2048, N=8192 | 32.768 / 33.326 / 33.792 | 90,021 |
| M=4, K=2048, N=8192 | 33.792 / 34.413 / 34.816 | 116,234 |
| M=5, K=2048, N=8192 | 34.816 / 35.213 / 35.840 | 141,993 |

Relative to the preceding published graph run, M=1-4 mean latency improved by 3.2%, 2.4%, 1.5%, and 3.0%.
The ordinary compiled eager route reached 11,282/22,585/34,438/45,979/61,556 tokens/s for M=1/2/3/4/5.
Across M=1-5, dense-reference maximum error remained 0.5-1.0, cross maximum error remained 0.25-1.0, and cross
mean absolute error remained 0.000636-0.000767.

Nsight Compute corroborated the strongest single-row result. M=1 replay duration fell from 40.74 to 39.17 us
(3.9%) with the same 512-CTA grid and 33.28 KiB dynamic shared memory. Registers/thread moved from 128 to 136,
achieved occupancy from 22.15% to 15.00%, and compute/DRAM throughput from 26.51%/8.97% to 26.71%/9.33%.
At M=3, the basic-report replay was neutral within profiler variation (41.31 versus 41.38 us), while event timing
repeated the retained win; registers/thread fell from 160 to 144 and achieved occupancy rose from 14.69% to
15.21%. Explicit local-memory metrics reported zero local-load and local-store bytes for both selected endpoints.

Before commit, all 425 ParoQuant tests passed with 16 warnings in 44.36 seconds on physical GPU 7. This includes
FP16/BF16 decode and prefill parity, dense-error envelopes, selector fallbacks, warmed compiled calls, CUDA graph
replay, concurrent streams, scratch ownership/reuse, and live-buffer updates. Compute Sanitizer memcheck and
synccheck passed M=1/M=3/M=5 with zero errors; racecheck passed the same paths with zero
hazards/errors/warnings. Ruff and `git diff --check` were clean.

The exact payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,2,3,4,5}_n8192_pair_first_weight_lifetime.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,2,3,4,5}_n8192_pair_late_first_register_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,2,3,4,5}_n8192_pair_late_first_confirm.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,2,3,4,5}_n8192_pair_late_first_exact.json
artifacts/paroquant_megakernel_20260723/fp16_split_pair_late_first_production_{graph_m1_m5,graph_m6_m8,eager_m1_m5}.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,3}_n8192_pair_{prefetch_first,late_first}_r*_basic.{json,ncu-rep}
artifacts/paroquant_megakernel_20260723/fp16_split_m{1,3}_n8192_pair_late_first_r*_local.ncu-rep
```

#### FP16 prefill split-K16

The next GPU 7 pass split the 16 K=128 groups across CTAs for underfilled FP16 prefill instead of duplicating all
16 groups inside each output CTA. The retained one-output-tile schedule uses BM32/BN128, four warps, one stage,
local `int16` partner offsets, and the existing fixed-order FP32 last-CTA reduction. A runtime-probed 124-SM
`sm_80` gate limits it to measured K=2048/krot=8 bands. Regular prefill and every non-target fallback are
unchanged.

The split scratch layout now reserves `BLOCK_M * BLOCK_N` elements per output tile and split rather than
`M * BLOCK_N`. This makes the allocation independent of the full row span repeated across every M tile. At
M=129/N=1920 it falls from about 75.6 MiB to 18.8 MiB; at M=992/N=512 it falls from about 958 MiB to 32 MiB.
Decode keeps dtype-reusable stream scratch by reserving its maximum eight-row tile. The public validator,
benchmark allocator, eager cache, and private graph-capture allocator all use the same compact contract.

Every configuration screen checked output accuracy before timing. Representative 3,000-sample raw paired results
against the exact production regular mega-kernel were:

| Shape | Regular p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | Mean speedup |
|:---|---:|---:|---:|
| M=9, K=2048, N=512 | 84.992 / 88.694 / 98.304 | 27.648 / 28.585 / 31.744 | 3.103x |
| M=128, K=2048, N=512 | 88.064 / 94.440 / 101.376 | 39.936 / 41.730 / 45.056 | 2.263x |
| M=497, K=2048, N=512 | 154.624 / 157.971 / 178.176 | 88.064 / 90.388 / 102.400 | 1.748x |
| M=992, K=2048, N=512 | 179.200 / 169.662 / 182.272 | 142.336 / 144.488 / 164.864 | 1.174x |
| M=32, K=2048, N=1920 | 118.784 / 122.945 / 137.216 | 33.792 / 34.534 / 38.912 | 3.560x |
| M=129, K=2048, N=1920 | 154.624 / 150.224 / 157.696 | 92.160 / 92.545 / 94.208 | 1.623x |
| M=256, K=2048, N=1920 | 156.672 / 155.605 / 180.224 | 136.192 / 140.224 / 157.696 | 1.110x |
| M=97/192, K=2048, N=2560 | 147.325 / 154.121 mean | 99.066 / 138.591 mean | 1.487x / 1.112x |
| M=81/160, K=2048, N=3072 | 148.475 / 156.216 mean | 95.298 / 141.186 mean | 1.558x / 1.107x |
| M=49/96, K=2048, N=4096 | 149.245 / 153.195 mean | 89.031 / 117.988 mean | 1.676x / 1.298x |

The complete compiled eager module, including dispatch, allocation, bias, and output shaping, measured:

| Shape | Regular p50/mean/p95 (us) | Split-K16 p50/mean/p95 (us) | Mean speedup |
|:---|---:|---:|---:|
| M=128, K=2048, N=512 | 201.728 / 224.868 / 340.992 | 110.592 / 125.567 / 130.048 | 1.791x |
| M=497, K=2048, N=512 | 225.280 / 232.729 / 296.960 | 92.160 / 112.632 / 159.744 | 2.066x |
| M=129, K=2048, N=1920 | 211.968 / 218.863 / 272.384 | 97.280 / 111.186 / 159.744 | 1.968x |
| M=96, K=2048, N=4096 | 190.464 / 192.888 / 259.072 | 117.760 / 129.485 / 125.952 | 1.490x |

Private-scratch CUDA graph replay improved the same four means by 2.140x, 1.788x, 1.599x, and 1.305x. The full
autotuner retained `prefill_megakernel` over CUDA AWQ and all legacy candidates at both standard projection
checks: M=128/N=2048 improved the existing backend by 1.854x mean and M=128/N=512 by 1.889x.

Dense accuracy ran before both eager and graph timing. Across the four production checks, standard/split dense
maximum error was 0.5-1.0, split dense mean error was 0.005486-0.006404, cross maximum error was 0.5-1.0, and
cross mean absolute error was 0.000708-0.000764. Repeated split outputs were bit-exact. The first focused
production/gate/scratch run passed 65 tests, then the full suite passed all 459 tests with 16 warnings in
53.84 seconds on physical GPU 7. Compute Sanitizer memcheck and synccheck passed the new multi-M-tile production
path with zero errors; racecheck reported zero hazards, errors, and warnings. Ruff and `git diff --check` were
clean.

Nsight Compute reports 1,200 CTAs at M=129/N=1920, 128 registers/thread, 40.96 KiB dynamic shared memory, 25%
theoretical and 22.18% achieved occupancy, 2.42 waves/SM, and 99.94 us replay duration. Explicit local-load and
local-store metrics were both zero. The remaining device-side opportunity is CTA-wave tail balance and
rotation/output-tile reuse rather than spill removal.

The exact retained payloads remain local:

```text
artifacts/paroquant_megakernel_20260723/fp16_prefill_m{9,64,128,497,992}_n512_split16*.json
artifacts/paroquant_megakernel_20260723/fp16_prefill_m{32,129,256}_n1920_split16*.json
artifacts/paroquant_megakernel_20260723/fp16_prefill_m{97,192}_n2560_split16.json
artifacts/paroquant_megakernel_20260723/fp16_prefill_m{81,160}_n3072_split16.json
artifacts/paroquant_megakernel_20260723/fp16_prefill_m{49,96}_n4096_split16*.json
artifacts/paroquant_megakernel_20260723/fp16_prefill_splitk_production_{eager,graph}.json
artifacts/paroquant_megakernel_20260723/fp16_prefill_splitk_autotune_routes.json
artifacts/paroquant_megakernel_20260723/fp16_prefill_m129_n1920_split16_basic.ncu-rep
```

## Validation completed

- `ruff check` passed for every changed Python path.
- `git diff --check` passed.
- `pytest -q tests/kernels/test_paroquant.py` passed independently and concurrently on physical GPUs 6 and 7:
  20 passed, 16 warnings on each device for the final follow-up source.
- CPU-only focused tests passed; CUDA-specific cases skipped when CUDA was hidden.
- FP16 and BF16, `krot=1` and `krot=8`, decode and prefill, M/N tails, a non-default stream, inference-created
  tensors, architecture fallback, autograd fallback, warmed CUDA graph capture/replay, and non-persistent metadata
  state were exercised.
- Compute Sanitizer memcheck passed decode and prefill on physical GPU 5 with zero errors.
- Final follow-up Compute Sanitizer memcheck passed all eight FP16/BF16, decode/prefill, `krot=1/8` parity cases on
  physical GPU 7 with zero errors.
- The GPU 7-only continuation source passed all 21 focused tests (16 warnings). Compute Sanitizer memcheck passed
  the eight FP16/BF16 decode/prefill parity cases plus the exact retained BM32 regression test with zero errors.
- The decode warp-saturation source passed all 26 focused tests (16 warnings) on physical GPU 7. The expanded
  suite adds M=8 decode parity for FP16/BF16 and `krot=1/8`; Compute Sanitizer memcheck passed all 13 selected
  mega-kernel cases with zero errors.
- The retained K-loop factors passed the same 26-test GPU 7 suite (16 warnings), including exact routing gates for
  winning and rejected regimes. Compute Sanitizer memcheck again passed all 13 selected mega-kernel cases with
  zero errors.
- The BM2/FMA continuation passed all 26 focused tests (16 warnings) on physical GPU 7. The route tests cover the
  M=1/2/3, K, N, and krot specialization boundaries. Compute Sanitizer memcheck passed the same 13 selected
  mega-kernel cases with zero errors.
- The BF16 first-partner source passed all 28 focused tests (16 warnings) on physical GPU 7. Two added randomized
  pair-schedule cases require exact output equality against the prior decode and prefill schedules. Compute
  Sanitizer memcheck passed all 15 selected mega-kernel cases with `ERROR SUMMARY: 0 errors`.
- The runtime wave-boundary source passed all 38 focused tests (16 warnings) on physical GPU 7. The added route
  cases cover 124-, 108-, and 80-SM inventories, and the M=113/128/224 BM32 cases require bit-exact equality
  against BM16. Compute Sanitizer memcheck passed all 17 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The four-band wave continuation passed all 58 focused tests (16 warnings) on physical GPU 7. Eight retained
  lower/upper band endpoints require bit-exact BM32/BM16 equality, while route tests cover every winning and losing
  GPU 7 boundary through M=993. Compute Sanitizer memcheck passed all 23 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=1024 one-wave continuation passed all 65 focused tests (16 warnings) on physical GPU 7. Both retained
  N=1024 endpoints require bit-exact BM32/BM16 equality, and route tests keep the neighboring and unmeasured later
  bands on BM16. Compute Sanitizer memcheck passed all 25 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=1536 one-wave continuation passed all 72 focused tests (16 warnings) on physical GPU 7. Both retained
  N=1536 endpoints require bit-exact BM32/BM16 equality, with explicit losing-boundary and later-band route checks.
  Compute Sanitizer memcheck passed all 27 selected mega-kernel cases with `ERROR SUMMARY: 0 errors`.
- The N=768 one-wave continuation passed all 79 focused tests (16 warnings) on physical GPU 7. Both retained N=768
  endpoints require bit-exact BM32/BM16 equality, and the later 4:2 wave band remains explicitly disabled. Compute
  Sanitizer memcheck passed all 29 selected mega-kernel cases with `ERROR SUMMARY: 0 errors`.
- The N=1280 one-wave continuation passed all 86 focused tests (16 warnings) on physical GPU 7. Both retained
  N=1280 endpoints require bit-exact BM32/BM16 equality, and its later 4:2 wave band remains explicitly disabled.
  Compute Sanitizer memcheck passed all 31 selected mega-kernel cases with `ERROR SUMMARY: 0 errors`.
- The N=1792 one-wave continuation passed all 93 focused tests (16 warnings) on physical GPU 7. Both retained
  N=1792 endpoints require bit-exact BM32/BM16 equality, and its later 4:2 wave band remains explicitly disabled.
  Compute Sanitizer memcheck passed all 33 selected mega-kernel cases with `ERROR SUMMARY: 0 errors`.
- The large-M N=512 continuation passed all 103 focused tests (16 warnings) on physical GPU 7. Four FP16/BF16
  endpoint cases require bit-exact BM32/BM8 equality, while route tests cover both band neighbors, the dtype gate,
  and 124-/108-/80-SM inventories. Compute Sanitizer memcheck passed all 37 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The large-M N=256 continuation passed all 113 focused tests (16 warnings) on physical GPU 7. Four new FP16/BF16
  endpoint cases require bit-exact BM32/BM8 equality, while routing tests cover M=992/993/1984/1985 and retain BM8
  on 108-/80-SM inventories. Compute Sanitizer memcheck passed all 41 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The large-M N=128 continuation passed all 123 focused tests (16 warnings) on physical GPU 7. Four new FP16/BF16
  endpoint cases require bit-exact BM32/BM8 equality, while routing tests cover M=1984/1985/3968/3969 and retain
  BM8 on 108-/80-SM inventories. Compute Sanitizer memcheck passed all 45 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The large-M N=384 continuation passed all 133 focused tests (16 warnings) on physical GPU 7. Four new FP16/BF16
  endpoint cases require bit-exact BM32/BM8 equality, while routing tests cover M=656/657/1312/1313 and retain
  BM8 on 108-/80-SM inventories. Compute Sanitizer memcheck passed all 49 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=640 one-wave continuation passed all 141 focused tests (16 warnings) on physical GPU 7. Both retained
  endpoints require bit-exact BM32/BM16 equality; routing tests cover M=384/385/768/769, the BF16 fallback, and the
  disabled later 4:2 band. Compute Sanitizer memcheck passed all 51 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=896 one-wave continuation passed all 148 focused tests (16 warnings) on physical GPU 7. Both retained
  endpoints require bit-exact BM32/BM16 equality; routing tests cover M=272/273/544/545, the BF16 fallback, and the
  disabled later 4:2 band. Compute Sanitizer memcheck passed all 53 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=1152 one-wave continuation passed all 155 focused tests (16 warnings) on physical GPU 7. Both retained
  endpoints require bit-exact BM32/BM16 equality; routing tests cover M=208/209/416/417, the BF16 fallback, and the
  disabled later 4:2 band. Compute Sanitizer memcheck passed all 55 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=1408 one-wave continuation passed all 162 focused tests (16 warnings) on physical GPU 7. Both retained
  endpoints require bit-exact BM32/BM16 equality; routing tests cover M=176/177/352/353, the BF16 fallback, and the
  disabled later 4:2 band. Compute Sanitizer memcheck passed all 57 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=1664 one-wave continuation passed all 169 focused tests (16 warnings) on physical GPU 7. Both retained
  endpoints require bit-exact BM32/BM16 equality; routing tests cover M=144/145/288/289, the BF16 fallback, and the
  disabled later 4:2 band. Compute Sanitizer memcheck passed all 59 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=1920 one-wave continuation passed all 176 focused tests (16 warnings) on physical GPU 7. Both retained
  endpoints require bit-exact BM32/BM16 equality; routing tests cover M=128/129/256/257, the BF16 fallback, and the
  disabled later 4:2 band. Compute Sanitizer memcheck passed all 61 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The wide FP16 16-warp continuation passed all 177 focused tests (16 warnings) on physical GPU 7. All 31 retained
  wide BM32 endpoints require exact equality to BM16, and route tests require W16S1 only on the 124-SM target while
  a 108-SM inventory retains W8S2. Compute Sanitizer memcheck passed all 61 selected mega-kernel cases with
  `ERROR SUMMARY: 0 errors`.
- The N=2176-4096 extension passed the complete 290-test ParoQuant kernel suite (16 warnings) on physical GPU 7.
  All 63 retained wide BM32 endpoints require exact equality to BM16; 64 route-boundary cases cover every extended
  neighbor, and launch/eligibility tests preserve BF16, non-124-SM, configured-limit, and outside-band fallbacks.
  Compute Sanitizer memcheck passed all 93 selected parity, wide, small-N, and prefetch cases with
  `ERROR SUMMARY: 0 errors`.
- The packed-weight prefetch source passed the complete 291-test ParoQuant kernel suite (16 warnings) on physical
  GPU 7. New decode and prefill route tests cover the dtype, krot, K=896/1024, and small-N boundaries; randomized
  BF16 decode/prefill tests require exact output against the prior load schedule. Compute Sanitizer memcheck passed
  all 93 selected parity, wide, small-N, and combined-prefetch cases with `ERROR SUMMARY: 0 errors`.
- The final BF16 split-K plus cached-dispatch source passed the complete 301-test ParoQuant kernel suite
  (16 warnings) on physical GPU 7. Ten focused cases cover six production M/N shapes, deterministic JIT/compiled
  reuse, a BF16 dense-reference envelope, real concurrent-stream execution, graph capture/replay fallback,
  compiled-launch kill-switch and ABI fallback, non-persistent serialization, the dtype/SM gates, and counter
  reset. A torch-profiler smoke test retained the named kernel event. A 10,000-call compiled-launch stress loop was
  bit-exact and left every counter at zero. Compute Sanitizer memcheck and synccheck passed all ten with
  `ERROR SUMMARY: 0 errors`; racecheck reported `0 hazards displayed (0 errors, 0 warnings)`.
- The empty-hook, raw-stream, cached-plan, and N=8192 row-tile continuation again passed all 301 tests
  (16 warnings) on physical GPU 7. The focused split-K cases additionally require active enter/exit hooks to
  receive non-null launch metadata, the public stream-query fallback to retain stream ownership, and a forced
  mega-kernel failure to demote the cached plan through the existing fallback. The final six-shape benchmark ran
  its dense and cross-schedule accuracy gates before timing, and a torch-profiler smoke test retained the named
  `paroquant_rotation_gemm_splitk_kernel` event. Compute Sanitizer memcheck and synccheck passed all ten focused
  cases with `ERROR SUMMARY: 0 errors`; racecheck reported
  `0 hazards displayed (0 errors, 0 warnings)`.
- The live-buffer, host-grid, and single-query stream continuation passed all 302 tests (16 warnings) on physical
  GPU 7. The eleventh split-K case replaces scale, bias, and channel-scale buffers in FP16, mutates theta in place,
  requires BF16 conversion, compares with the standard mega-kernel, and verifies counter reset. The final
  six-shape benchmark ran the unchanged dense/cross accuracy gates before timing. Compute Sanitizer memcheck and
  synccheck passed all 11 focused cases with `ERROR SUMMARY: 0 errors`; racecheck reported
  `0 hazards displayed (0 errors, 0 warnings)`.
- The zero-scratch direct-launch source passed all 302 tests (16 warnings) on physical GPU 7. Its 11 split-K
  cases include active launch hooks, live-buffer mutation, concurrent streams, graph fallback, and a forced
  current-device mismatch that exercises the retained device context. Compute Sanitizer memcheck and synccheck
  passed all 11 with `ERROR SUMMARY: 0 errors`; racecheck reported
  `0 hazards displayed (0 errors, 0 warnings)`.
- The caller-shaped output source passed all 302 tests (16 warnings) on physical GPU 7. The added regression
  requires two three-dimensional outputs to be bit-exact, owning, and backed by distinct storage. Compute
  Sanitizer memcheck and synccheck passed all 11 split-K tests with `ERROR SUMMARY: 0 errors`; racecheck reported
  `0 hazards displayed (0 errors, 0 warnings)`.
- The graph-owned split-K source passed all 302 tests (16 warnings) on physical GPU 7. The graph regression covers
  two private captures, concurrent-stream exact replays, counter reset, and cold compiled-plan fallback. Compute
  Sanitizer memcheck and synccheck passed all 11 split-K tests with `ERROR SUMMARY: 0 errors`; racecheck reported
  `0 hazards displayed (0 errors, 0 warnings)`.
- The direct caller-input source passed all 302 tests (16 warnings) on physical GPU 7. The regression requires the
  compiled launcher to receive the original three-dimensional activation and explicit row count, with bit-exact
  owning outputs. The six eager and graph benchmark cases ran the unchanged dense/cross accuracy gates before
  timing. Compute Sanitizer memcheck and synccheck passed all 11 split-K tests with
  `ERROR SUMMARY: 0 errors`; racecheck reported `0 hazards displayed (0 errors, 0 warnings)`.
- The validated last-scratch source passed all 302 tests (16 warnings) on physical GPU 7. The graph regression
  requires private capture storage to remain distinct from the eager last entry and confirms the alias is absent
  from serialized state. Compute Sanitizer memcheck and synccheck passed all 11 split-K tests with
  `ERROR SUMMARY: 0 errors`; racecheck reported `0 hazards displayed (0 errors, 0 warnings)`.
- The FP16 split-K continuation passed all 329 ParoQuant tests (16 warnings) on physical GPU 7 immediately before
  commit. The suite covers the initial six measured FP16 production gates, the then-unmeasured M=3 and rejected
  width boundaries, FP16/BF16 dense-error envelopes, exact warmed reuse, dtype-isolated compiled launchers,
  stream-owned scratch, two graph captures, and every established fallback. Compute Sanitizer memcheck and
  synccheck passed eight selected FP16 and cache-isolation cases with `ERROR SUMMARY: 0 errors`; racecheck passed
  three representative cases with `0 hazards displayed (0 errors, 0 warnings)`.
- The irregular-row FP16 continuation added exact launch/gate and parity coverage for M=3/5/6/7. Before the final
  full-suite gate, 45 focused tests passed on physical GPU 7. The final pre-commit suite passed all 351 tests
  (16 warnings), including real compiled module calls and expanded FP16/BF16 dense envelopes. Compute Sanitizer
  memcheck and synccheck passed all ten measured FP16 split shapes with `ERROR SUMMARY: 0 errors`; racecheck
  passed M=3 and M=7 with `0 hazards displayed (0 errors, 0 warnings)`.
- Compute Sanitizer synccheck passed both regimes with zero errors.
- Compute Sanitizer racecheck reported zero errors and zero hazards.

## Qwen3-8B current-versus-main projection benchmark

After completing and publishing the FP16 prefill split-K16 experiment, the complete native Qwen3-8B linear shape
set was benchmarked on physical GPU 7. The baseline was `origin/main` at
`fe75dd9b2c909d11bd966b6c9f3230f135b6320c`; the candidate was the published ParoQuant branch at
`bbfb2e6829d46aae735ef3441a86ddd3ae606cc5`. The local Qwen3-8B configuration has 36 layers, hidden size 4096,
intermediate size 12288, 32 attention heads, eight key/value heads, and head dimension 128. Its unique projection
shapes and per-layer multiplicities are:

| projection group | K | N | calls/layer |
|---|---:|---:|---:|
| q_proj + o_proj | 4096 | 4096 | 2 |
| k_proj + v_proj | 4096 | 1024 | 2 |
| gate_proj + up_proj | 4096 | 12288 | 2 |
| down_proj | 12288 | 4096 | 1 |

`scripts/benchmark_paroquant_qwen3_8b.py` generates deterministic packed W4 group-128, `krot=8` buffers without
materializing a dense K-by-N source. Quant scales are sampled in `[0.01, 0.05]`, bias is scaled by 0.1, and inputs
and runtime tensors use BF16. Every case is checked against `ParoLinear` before timing and the repeated candidate
call must be bit-exact. Acceptance requires `rtol=0.02`, `atol=0.25`, and mean absolute error divided by reference
mean magnitude no greater than 0.5%. The worst observed normalized mean error was 0.1742% on main and 0.1484% on
the candidate; the maximum absolute error was 1.0 on both. Every candidate CUDA-AWQ case was bit-exact to the
reference; the nonzero candidate maximum came from a BF16 dense M=512 case.

Each revision ran in three fresh processes with 50 timed-path warmups, 500 CUDA-event samples, and the production
autotuner configured for 10 warmups, 20 samples, and a 5% selection margin. The tables use the median of the three
per-process statistics. Hardware was the GPU-7 `NVIDIA PG506-230`, UUID
`724ea08e-67c3-c0ce-29bb-e6c48e7dde28`, sm_80, 124 SMs, 96 GiB, with Torch 2.13.0+cu130, CUDA 13.0, and
Triton 3.7.1.

| projection | M | main plan | current plan | main/current p50 us | p50 speedup | mean speedup |
|---|---:|---|---|---:|---:|---:|
| q/o | 1 | dense | cuda_awq | 241.664 / 197.632 | 1.223x | 1.171x |
| k/v | 1 | dense | cuda_awq | 245.760 / 195.584 | 1.257x | 1.222x |
| gate/up | 1 | prefill_fused | cuda_awq | 227.328 / 197.632 | 1.150x | 1.172x |
| down | 1 | dense | cuda_awq | 263.168 / 210.944 | 1.248x | 1.198x |
| q/o | 8 | dense / prefill_fused | cuda_awq | 229.376 / 197.632 | 1.161x | 1.193x |
| k/v | 8 | dense | cuda_awq | 244.736 / 196.608 | 1.245x | 1.277x |
| gate/up | 8 | prefill_fused | cuda_awq | 229.376 / 198.656 | 1.155x | 1.104x |
| down | 8 | dense | cuda_awq | 257.024 / 209.920 | 1.224x | 1.179x |
| q/o | 128 | dense | cuda_awq | 245.760 / 209.920 | 1.171x | 1.214x |
| k/v | 128 | dense | cuda_awq | 244.736 / 194.560 | 1.258x | 1.324x |
| gate/up | 128 | dense | cuda_awq | 260.096 / 221.184 | 1.176x | 1.219x |
| down | 128 | dense | cuda_awq | 258.048 / 219.136 | 1.178x | 1.206x |
| q/o | 512 | dense | dense | 263.168 / 272.384 | 0.966x | 0.976x |
| k/v | 512 | dense | cuda_awq | 242.688 / 207.872 | 1.167x | 1.141x |
| gate/up | 512 | dense | dense | 329.728 / 330.752 | 0.997x | 0.998x |
| down | 512 | dense | dense | 347.136 / 346.112 | 1.003x | 1.000x |

Weighting those projections by their per-layer multiplicity and then by 36 layers gives:

| M | main/current p50 linear-stack ms | p50 speedup | main/current projected p50 linear tok/s | mean speedup |
|---:|---:|---:|---:|---:|
| 1 | 60.936 / 50.135 | 1.215x | 16.4 / 19.9 | 1.190x |
| 8 | 59.904 / 50.246 | 1.192x | 133.5 / 159.2 | 1.187x |
| 128 | 63.332 / 52.937 | 1.196x | 2021.1 / 2418.0 | 1.243x |
| 512 | 72.659 / 70.853 | 1.025x | 7046.6 / 7226.3 | 1.024x |

These are projected linear-only stack rates, not end-to-end model TPS; attention, normalization, KV-cache work,
sampling, and framework scheduling are intentionally excluded. Native Qwen3-8B has K=4096 or K=12288, beyond
the mega-kernel's measured `K<=2048` envelope. The current production runtime therefore selects its CUDA-AWQ
fallback for the profitable M<=128 cases and k/v M=512, while the other M=512 projections use dense. This result
is a full current-ParoQuant-runtime comparison against main at Qwen3-8B shapes, but it is not evidence that the
mega-kernel itself accelerates these wider-K projections. The unchanged dense M=512 routes remain within noise;
the weighted M=512 gain comes from k/v selecting CUDA-AWQ.

Immediately before the benchmark/log commit, all 459 tests in `tests/kernels/test_paroquant.py` passed on physical
GPU 7 (16 warnings, 44.02 seconds). Across the six full benchmark processes, all 96 revision/shape/row cases
passed the dense-reference, normalized-error, and repeat-determinism gates before timing. An expanded
`tests/test_paroquant.py` plus kernel run reached 579 passes but also exposed two non-numerical test-fixture
failures: the live-buffer test's allocator-address assertion passed on isolated rerun and in the clean kernel-suite
rerun, while one processor-test dummy lacks the `is_embeddings_module` argument used by production and already
present in the corresponding `origin/main` test fixture. The uncommitted comparison changes touch only this log
and the standalone benchmark, so that pre-existing processor fixture mismatch was not folded into this kernel
benchmark commit.

## Qwen3-8B large-K decode mega-kernel continuation

The Qwen benchmark above showed that the published branch still fell back to CUDA-AWQ at every M=1/8 native
Qwen width because the mega-kernel was capped at K=2048. A GPU-7-only continuation measured split-K schedules
for the exact BF16 Qwen3-8B decode shapes and extends the default cap to K=12288 without broadening the portable
gate. The route still requires sm_80, 124 SMs, group size 128, `krot=8`, contiguous BF16 input, inference mode,
and exactly M=1 or M=8. Other large-K shapes, dtypes, row counts, architectures, and SM counts retain their
established fallback. Setting `GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_MAX_K=2048` also disables the new route.

The retained schedules are:

| M | K | N | split K | BM / warps | output tiles/CTA | max registers | paired counter |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 1024 | 32 | 4 / 4 | 1 | uncapped | no |
| 1 | 4096 | 4096 | 32 | 4 / 4 | 1 | 128 | no |
| 1 | 4096 | 12288 | 32 | 4 / 4 | 2 | 128 | yes |
| 1 | 12288 | 4096 | 96 | 1 / 4 | 2 | 128 | yes |
| 8 | 4096 | 1024 / 4096 | 32 | 8 / 8 | 1 | uncapped | no |
| 8 | 4096 | 12288 | 32 | 8 / 8 | 2 | uncapped | no |
| 8 | 12288 | 4096 | 32 | 8 / 8 | 1 | uncapped | no |

Large-K is BF16-only, so its scratch uses the exact measured BM instead of the K=2048 decode allocation's
cross-dtype `max(BM, 8)` rule. This reduces M=1 partial scratch from 4 to 2 MiB for q/o, 12 to 6 MiB for gate/up,
and 12 to 1.5 MiB for down while preserving stream ownership and private graph-capture allocations. The split
factor is propagated through scratch sizing, the warmed Triton launcher, its direct compiled launch, and graph
capture instead of relying on the older split-16 constant.

Raw CUDA-graph timing used 20 eager warmups, 50 graph warmups, 2,000 alternating CUDA-event samples, and an
accuracy check before timing. The retained device schedules measured:

| M | projection | regular / retained p50 us | raw speedup |
|---:|---|---:|---:|
| 1 | q/o, 4096->4096 | 155.648 / 36.864 | 4.222x |
| 1 | k/v, 4096->1024 | 150.528 / 17.408 | 8.647x |
| 1 | gate/up, 4096->12288 | 197.632 / 80.896 | 2.443x |
| 1 | down, 12288->4096 | 583.680 / 81.920 | 7.125x |
| 8 | q/o, 4096->4096 | 167.424 / 43.008 | 3.893x |
| 8 | k/v, 4096->1024 | 150.528 / 22.528 | 6.682x |
| 8 | gate/up, 4096->12288 | 198.656 / 103.424 | 1.921x |
| 8 | down, 12288->4096 | 578.560 / 112.640 | 5.137x |

The complete production benchmark was then repeated three times in fresh processes with the same methodology as
the main comparison: 50 warmups, 500 CUDA-event samples, 10/20 autotune warmups/samples, and a 5% selection
margin. The baseline remains `origin/main@fe75dd9b2c909d11bd966b6c9f3230f135b6320c`; the candidate was the
working tree based on `1dfcc133c441cd8072104fef2fad1501a5826764`. Every M=1/8 projection selected
`decode_megakernel`; M=128/512 kept the established CUDA-AWQ or dense routes.

| projection | M | main/current p50 us | p50 speedup | mean speedup |
|---|---:|---:|---:|---:|
| q/o | 1 | 241.664 / 98.304 | 2.458x | 2.653x |
| k/v | 1 | 245.760 / 94.208 | 2.609x | 2.763x |
| gate/up | 1 | 227.328 / 105.472 | 2.155x | 2.362x |
| down | 1 | 263.168 / 109.568 | 2.402x | 2.492x |
| q/o | 8 | 229.376 / 103.424 | 2.218x | 2.387x |
| k/v | 8 | 244.736 / 93.184 | 2.626x | 2.723x |
| gate/up | 8 | 229.376 / 104.448 | 2.196x | 2.397x |
| down | 8 | 257.024 / 110.592 | 2.324x | 2.540x |

Weighting the four projection groups by Qwen3-8B's per-layer call counts and 36 layers gives the updated full
current-versus-main result:

| M | main/current p50 linear-stack ms | p50 speedup | main/current projected p50 linear tok/s | mean speedup |
|---:|---:|---:|---:|---:|
| 1 | 60.936 / 25.399 | 2.399x | 16.4 / 39.4 | 2.570x |
| 8 | 59.904 / 25.657 | 2.335x | 133.5 / 311.8 | 2.501x |
| 128 | 63.332 / 53.748 | 1.178x | 2021.1 / 2381.5 | 1.270x |
| 512 | 72.659 / 71.553 | 1.015x | 7046.6 / 7155.5 | 1.032x |

These remain linear-only projections rather than end-to-end model TPS. Relative to the first large-K production
schedule, the register caps and paired output reductions improved the weighted M=1 mean by 3.6% and raw graph
latency by up to about 24%; the three-repeat eager p50 aggregate moved by -1.6%. That p50 reversal is reported
explicitly because the retained choice targets sustained and CUDA-graph decode throughput, not a hidden single
latency statistic.

Full-set accuracy is unchanged from the earlier comparison. On the new large-K decode route, worst normalized
mean error versus `ParoLinear` was 0.0092%, maximum absolute error was 0.25, and every repeated output was
bit-exact. The full 16-case candidate matrix remained inside `rtol=0.02`, `atol=0.25`, and the 0.5% normalized
mean-error limit; its worst normalized mean error was still the unrelated dense M=512 q/o case at 0.1484%.

Nsight Compute 2025.3.1 profiled the M=1, K=N=4096 split-32 kernel before and after the retained 128-register cap.
The initial kernel was latency/occupancy constrained rather than DRAM-bound:

| metric | uncapped | maxnreg=128 |
|---|---:|---:|
| registers/thread | 150 | 128 |
| theoretical / achieved occupancy | 18.75% / 17.33% | 25.00% / 22.98% |
| scheduler cycles with no eligible warp | 63.98% | 58.66% |
| compute / DRAM throughput | 32.04% / 7.78% | 33.31% / 7.98% |
| profiled duration | 47.65 us | 46.46 us |

The large-K change passed 21 focused production/config/graph tests and then all 485 tests in
`tests/kernels/test_paroquant.py` on physical GPU 7 (16 warnings, 54.54 seconds). Compute Sanitizer memcheck
passed the two paired-output production graph cases with `ERROR SUMMARY: 0 errors`. A separate
`--leak-check full` run also had no access failures but returned 99 after reporting 155,189,257 bytes held by
PyTorch's process-exit CUDA caching allocator; the meaningful memcheck rerun with leak reporting disabled passed.

## Qwen eager prepared-launch continuation

Nsight Systems then isolated why the large-K graph kernels did not translate fully to eager module time. The
GPU-7 trace used five warmups and 20 timed M=1 calls per Qwen projection. Its final steady clusters put the
q/o split-32 kernel near 38 us, k/v near 16 us, and the paired gate/up and down kernels near 86 us, while eager
calls arrived about 140 us apart under tracing. A 200-call `cProfile` run outside Nsight measured about
80-90 us of Python time per module call. Repeated scratch lookup, rotation-source checks, launch-configuration
selection, Triton specialization binding, output allocation, and the native submission all remained on that
warm path.

The retained decode-only prepared launch binds the invariant grid, launch configuration, compiler metadata,
static weight/rotation pointers, and stream-owned scratch once after the normal validated Triton launch. Its
module cache is keyed by device, raw stream, rows, and dtype. Before every fast submission it still checks the
selected plan, compiled-kernel identity, live qweight/scale/zero/bias identities, and the mutation versions of
pairs, theta, and channel scales. CUDA graph capture bypasses the eager prepared state and keeps its private
allocator-owned scratch; another eager stream receives separate scratch and a separate closure. Dynamic Triton
launch hooks, current-device context switching, serialization clearing, instance/class overrides, and the
existing fallback demotion remain intact. The prepared call allocates the caller's final output shape directly,
so the wide-Qwen fast path also avoids its final view.

The final closure form reduced a 500-call k/v CPU profile to 24 ms including `nn.Module` dispatch, or about
48 us/call; 22 ms was inside `forward`, or about 44 us/call. The actual native launch accounted for about
10-12 us of that profile. Extending the same prepared-state lookup to FP16 prefill was rejected because it added
host checks to a compute-dominated path without removing the full module dispatch. Prefill therefore continues
through the byte-for-byte prior compiled-launch branch.

The full deterministic Qwen3-8B matrix was rerun in three fresh GPU-7 processes with 50 warmups, 500 CUDA-event
samples, and 10/20/5% autotune settings. Every case again ran its dense/reference and repeat-determinism checks
before timing. The M=1/8 per-projection medians versus `origin/main@fe75dd9b` are:

| projection | M | main/prepared p50 us | p50 speedup | mean speedup |
|---|---:|---:|---:|---:|
| q/o | 1 | 241.664 / 76.800 | 3.147x | 3.259x |
| k/v | 1 | 245.760 / 64.512 | 3.810x | 3.781x |
| gate/up | 1 | 227.328 / 78.848 | 2.883x | 3.096x |
| down | 1 | 263.168 / 79.872 | 3.295x | 3.500x |
| q/o | 8 | 229.376 / 77.824 | 2.947x | 3.135x |
| k/v | 8 | 244.736 / 64.512 | 3.794x | 3.714x |
| gate/up | 8 | 229.376 / 104.448 | 2.196x | 2.412x |
| down | 8 | 257.024 / 110.592 | 2.324x | 2.539x |

Weighting the complete four-projection set by Qwen3-8B's per-layer multiplicities and 36 layers gives:

| M | main/prepared p50 linear-stack ms | p50 speedup | main/prepared projected p50 linear tok/s | mean speedup |
|---:|---:|---:|---:|---:|
| 1 | 60.936 / 18.727 | 3.254x | 16.4 / 53.4 | 3.380x |
| 8 | 59.904 / 21.750 | 2.754x | 133.5 / 367.8 | 2.919x |
| 128 | 63.332 / 53.674 | 1.180x | 2021.1 / 2384.8 | 1.220x |
| 512 | 72.659 / 71.664 | 1.014x | 7046.6 / 7144.5 | 1.002x |

Relative to published large-K commit `aecbb133`, the prepared path improves weighted p50 by 1.356x at M=1 and
1.180x at M=8. Gate/up and down M=8 are kernel-bound and remain in their prior p50 buckets; the host-dominated
q/o and k/v routes improve by 1.28-1.46x. M=128/512 do not enter the new branch. Their p50 values are effectively
flat versus `aecbb133`; noisier per-process means moved down despite identical code and are reported rather than
attributed to this change. These remain projected linear-only rates, not end-to-end model TPS.

After the implementation and benchmark log were finalized, all 485 tests in
`tests/kernels/test_paroquant.py` passed on physical GPU 7 (16 warnings, 82.09 seconds). This includes the full
FP16/BF16 dense-error envelopes, all eight production Qwen large-K accuracy cases, graph-private scratch replay,
concurrent-stream isolation, live-buffer mutation, hook/device-context behavior, and the untouched FP16 prefill
compiled route.

## Combined native rotation and CUDA-AWQ dispatch

The next GPU-7 Nsight Systems capture profiled 40 warmed M=128 calls for each Qwen3-8B projection after the
selector chose CUDA-AWQ. Every projection still submitted rotation, packed GEMM, FP32 split-K reduction, and
bias as four CUDA kernels. The traced device work was roughly 55 us for k/v, 96 us for q/o, and 241-245 us for
gate/up and down, while the complete q/o and k/v calls remained near 200 us. Across 640 launches,
`cudaLaunchKernel` had an 8.386 us median API duration. The small q/o and k/v projections were therefore
submission-bound; gate/up and down were primarily kernel-bound.

The retained path adds `gptqmodel_paroquant::rotate_awq_gemm`. It launches the existing tuned rotation and then
calls the already-loaded `gptqmodel_awq::gemm_forward` through a typed native dispatcher handle. The AWQ source
is not duplicated in the rotation extension, the CUDA launch sequence and FP32 reduction remain unchanged, and
optional bias is applied with the same ATen add inside the native call. Python resolves the rotation launch
configuration and live tensors as before, but submits the complete route through one torch op instead of a
rotation op followed by an AWQ op. A lock-protected one-time AWQ readiness check avoids registry work on the hot
path. Training/grad-enabled calls, identity rotations, CPU/NPU inputs, unavailable extensions, and combined-op
failures retain the established separate fallback.

Controlled bias-free BF16 AB blocks used one synchronization per sample so host submission remained visible.
Each route ran three 300-sample blocks after 20 warmups. The median block results were:

| projection | separate p50/mean/p95 us | combined p50/mean/p95 us | p50 speedup | mean speedup |
|---|---:|---:|---:|---:|
| q/o, M=128 | 217.088 / 219.235 / 227.328 | 158.720 / 160.154 / 168.960 | 1.368x | 1.369x |
| k/v, M=128 | 187.392 / 189.577 / 199.680 | 130.048 / 132.151 / 141.312 | 1.441x | 1.435x |
| gate/up, M=128 | 340.992 / 342.340 / 352.256 | 284.672 / 285.536 / 293.888 | 1.198x | 1.199x |
| down, M=128 | 339.968 / 341.081 / 351.232 | 286.720 / 288.440 / 296.960 | 1.186x | 1.183x |

The combined output was bit-exact to the separate route for FP16 and BF16, with and without bias, at
M=1/8/128/512. CUDA graph replay and two concurrent streams also produced bit-exact, independently allocated
outputs. A forced native-op exception returned to the separate rotation plus AWQ path.

The full deterministic Qwen3-8B harness then ran in three fresh current processes and three fresh processes from
the newly fetched `origin/main@0b0405f24facb073ee69b00a03291fdab9779408`. Each process used 50 warmups,
500 CUDA-event samples, and 10/20/5% autotune settings. All 96 revision/shape/row cases ran dense-reference and
repeat-determinism gates before timing. The median M=128 and CUDA-AWQ M=512 results were:

| projection | M | main/current p50 us | p50 speedup | mean speedup |
|---|---:|---:|---:|---:|
| q/o | 128 | 253.952 / 157.696 | 1.610x | 1.521x |
| k/v | 128 | 252.928 / 144.384 | 1.752x | 1.662x |
| gate/up | 128 | 266.240 / 221.184 | 1.204x | 1.255x |
| down | 128 | 261.120 / 220.160 | 1.186x | 1.255x |
| k/v | 512 | 248.832 / 156.672 | 1.588x | 1.544x |

The other M=512 projections remained dense and essentially flat. Weighting all four projection groups by their
Qwen3-8B per-layer multiplicities and 36 layers gives:

| M | main/current p50 linear-stack ms | p50 speedup | main/current projected p50 linear tok/s | mean speedup |
|---:|---:|---:|---:|---:|
| 1 | 61.415 / 18.801 | 3.267x | 16.3 / 53.2 | 3.387x |
| 8 | 63.037 / 21.824 | 2.889x | 126.9 / 366.6 | 3.006x |
| 128 | 65.065 / 45.601 | 1.427x | 1967.3 / 2807.0 | 1.422x |
| 512 | 73.839 / 67.424 | 1.095x | 6934.0 / 7593.7 | 1.085x |

Relative to published commit `41cd6f11`, the new path improves weighted p50 by 1.177x at M=128 and 1.063x at
M=512. M=1/8 continue through the prepared split-K mega-kernel and remain in the same p50 buckets. These remain
projected linear-only rates, not end-to-end model TPS.

Immediately before commit, all 491 tests in `tests/kernels/test_paroquant.py` passed on physical GPU 7
(16 warnings, 48.95 seconds). The suite includes the existing dense-reference envelopes and all eight large-K
Qwen decode cases plus the new FP16/BF16 bias/no-bias bit-exact checks, graph replay, concurrent-stream
ownership, and forced combined-op fallback.

## Vectorized FP32 reduction and fused bias

Full Nsight Compute captures on physical GPU 7 then separated the CUDA-AWQ M=128 GEMM from its FP32 split-K
reducer. The q/o GEMM used 1,024 two-warp CTAs, 72 registers/thread, and 9.98 KiB static shared memory. It ran
for 77.984 us with 46.86% SM throughput, 68.26% compute-memory throughput, 22.95% achieved occupancy, and an
85.16% L2 hit rate. K/v had only 256 CTAs and ran for 46.240 us with 19.70% SM throughput and 6.38% achieved
occupancy. The separate scalar reducers were much shorter but still material at the complete-call scale:

| projection | scalar reducer grid | duration | DRAM throughput | registers/thread |
|---|---:|---:|---:|---:|
| q/o, M=128 | 2,048 x 256 threads | 10.720 us | 32.13% | 30 |
| k/v, M=128 | 512 x 256 threads | 5.664 us | 15.36% | 30 |

The retained reducer processes four adjacent FP32 partials per thread with aligned `float4` loads and packed
FP16/BF16 stores. It preserves the split-index accumulation order independently for every output element. The
same reports measured:

| projection | vec4 reducer grid | scalar/vec4 duration | kernel speedup | vec4 DRAM throughput |
|---|---:|---:|---:|---:|
| q/o, M=128 | 512 x 256 threads | 10.720 / 7.488 us | 1.432x | 46.15% |
| k/v, M=128 | 128 x 256 threads | 5.664 / 5.312 us | 1.066x | 16.26% |

The combined ParoQuant dispatch now calls a dedicated `gptqmodel_awq::gemm_forward_bias` entry point. Existing
AWQ callers keep the original `gemm_forward` schema and behavior. When FP32 accumulation and the vector reducer
are active, the reducer first rounds each split sum to the requested FP16/BF16 output, then performs the same
packed output-dtype bias add and final rounding that the former ATen add performed. No-bias calls use the same
vector reducer without reading a bias tensor. Non-vectorizable shapes, disabled fusion, non-FP32 accumulation,
and the existing PyTorch reduction path retain their prior scalar/add fallbacks. The diagnostic environment
switches used by the tests are `GPTQMODEL_AWQ_DISABLE_VEC4_SPLITK_REDUCE` and
`GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE_BIAS`.

A one-call `torch.profiler` capture after warmup reported exactly three `cudaLaunchKernel` submissions:
6.880 us for rotation, 77.055 us for the packed GEMM, and 5.344 us for the vector reducer. There was no separate
bias kernel. Eight alternating 300-sample BF16 M=128 blocks compared vector reduction with a separate bias
launch against vector reduction with bias fused:

| projection | separate/fused p50 us | p50 speedup | separate/fused mean us | mean speedup |
|---|---:|---:|---:|---:|
| q/o | 137.472 / 120.832 | 1.138x | 147.476 / 131.832 | 1.119x |
| k/v | 126.976 / 112.640 | 1.127x | 136.640 / 118.700 | 1.151x |
| gate/up | 216.064 / 207.872 | 1.039x | 216.783 / 209.014 | 1.037x |
| down | 219.136 / 213.504 | 1.026x | 219.780 / 214.312 | 1.026x |

Every paired output was bit-exact. The focused accuracy matrix also covered FP16/BF16, bias/no-bias, the scalar
reducer fallback, the separate-bias fallback, CUDA graph replay, and two concurrent streams with independently
allocated outputs.

The complete Qwen3-8B matrix was rerun in three fresh GPU-7 processes with the same 50 warmups, 500 CUDA-event
samples, and 10/20/5% autotune settings. All 48 current-process projection/row cases passed their
dense-reference and repeat-determinism gates before timing. Relative to `c30e0511`, this pass changes only the
CUDA-AWQ-selected prefill cases:

| M | c30/latest p50 linear-stack ms | p50 speedup | c30/latest mean linear-stack ms | mean speedup |
|---:|---:|---:|---:|---:|
| 1 | 18.801 / 18.653 | 1.008x | 19.253 / 19.333 | 0.996x |
| 8 | 21.824 / 21.750 | 1.003x | 22.123 / 22.292 | 0.992x |
| 128 | 45.601 / 41.988 | 1.086x | 47.731 / 42.884 | 1.113x |
| 512 | 67.424 / 66.060 | 1.021x | 68.738 / 67.859 | 1.013x |

M=1/8 use the unchanged decode mega-kernel, so their small median/mean movements are measurement noise rather
than an attributed effect. `origin/main` was fetched again immediately after the run and remained at
`0b0405f24facb073ee69b00a03291fdab9779408`. Against the three existing fresh-process runs from that exact main
revision, the latest weighted Qwen3-8B projection result is:

| M | main/latest p50 linear-stack ms | p50 speedup | main/latest projected p50 linear tok/s | mean speedup |
|---:|---:|---:|---:|---:|
| 1 | 61.415 / 18.653 | 3.292x | 16.3 / 53.6 | 3.373x |
| 8 | 63.037 / 21.750 | 2.898x | 126.9 / 367.8 | 2.983x |
| 128 | 65.065 / 41.988 | 1.550x | 1967.3 / 3048.5 | 1.582x |
| 512 | 73.839 / 66.060 | 1.118x | 6934.0 / 7750.5 | 1.099x |

These remain projected 36-layer linear-only rates, not end-to-end model TPS.

Immediately before commit, all 497 tests in `tests/kernels/test_paroquant.py` and
`tests/kernels/test_awq_cuda_fp32_reduce.py` passed on physical GPU 7 after a clean AWQ rebuild (16 warnings,
96.43 seconds). This includes the dense-reference error envelopes, FP16/BF16 exact reducer comparisons, forced
scalar reduction, forced separate-bias fallback, CUDA graph replay, and concurrent-stream coverage. An expanded run including
`tests/test_extension_load_api.py` finished with 509 passed and one unrelated pre-existing assertion: GrassHopper
has been in the extension registry since `19ee351b`, while the untouched expected `load()` result still omits it.

## Coalesced AWQ partial stores

The next full Nsight Compute pass attributed 262,144 excessive global sectors in the BF16 M=128 q/o GEMM to the
FP32 split-partial writeback. Each thread owns adjacent accumulator pairs, but the established scalar loop issued
the two values as separate 32-bit stores. Each instruction therefore used only 16 of every 32-byte sector. The
retained path converts and writes adjacent FP32, FP16, or BF16 values as one aligned pair. It does not change the
MMA, dequantization, accumulation, or split-reduction order.

On physical GPU 7, an otherwise identical full NCU capture measured:

| q/o M=128 GEMM | scalar stores | packed stores | change |
|---|---:|---:|---:|
| duration | 77.984 us | 76.096 us | 1.025x |
| excessive global sectors | 262,144 | 0 | eliminated |
| executed instructions | 21,848,064 | 21,815,296 | -32,768 |
| registers/thread | 72 | 72 | unchanged |
| static shared memory | 9.98 KiB | 9.98 KiB | unchanged |
| achieved occupancy | 22.95% | 23.13% | effectively unchanged |

The report is
`artifacts/paroquant_megakernel_20260723/qwen3_m128_qo_awq_packed_store_full_gpu7.ncu-rep`.
Because fresh-process module clocks moved by more than this small kernel delta, the end-to-end decision used a
same-process switch that was removed before commit. Eight 500-sample ABBA blocks alternated the scalar and packed
kernel specializations with identical BF16 M=128 tensors. Every paired output was bit-exact:

| projection | scalar/packed block-median p50 us | p50 speedup | scalar/packed block-median mean us | mean speedup |
|---|---:|---:|---:|---:|
| q/o | 109.568 / 109.568 | 1.000x | 113.677 / 111.870 | 1.016x |
| k/v | 104.448 / 103.424 | 1.010x | 105.324 / 105.646 | 0.997x |
| gate/up | 208.896 / 202.752 | 1.030x | 209.204 / 203.388 | 1.029x |
| down | 214.016 / 212.992 | 1.005x | 214.426 / 213.580 | 1.004x |

Weighting those direct combined calls by Qwen3-8B projection multiplicity improves both p50 and mean by 1.015x.
Immediately before commit, all 497 tests in `tests/kernels/test_paroquant.py` and
`tests/kernels/test_awq_cuda_fp32_reduce.py` passed on physical GPU 7 (16 warnings, 53.27 seconds). This includes
the independent dense-weight references as well as FP16/BF16 reducer, combined-dispatch, graph, and stream tests.

## Reused-weight M32 AWQ prefill tile

At M=128, the retained M16xN128 AWQ kernel launched two 64-thread CTAs for every pair of 16-row tiles. Both CTAs
loaded and dequantized the same K32xN128 weight tile. The new M32xN128 specialization uses four warps: two row
groups by two 64-column groups. Its 128 threads cooperatively load one 32x128 activation tile and one 32x128
dequantized weight tile, then the two row groups reuse that shared weight tile. Total MMA warp work is unchanged,
but CTA count and packed-weight/dequantization work are halved.

The route is limited to exact M=128 Qwen3-8B projections, group size 128, split-K4, FP16/BF16, and a
runtime-probed sm_80 device with 124 SMs. All other shapes, split factors, group sizes, architectures, and SM
inventories instantiate the original M16 kernel. The M16 compile-time mapping produces the same 21,815,296
executed q/o instructions as the preceding packed-store commit. `GPTQMODEL_AWQ_DISABLE_M32_QWEN_PREFILL=1`
provides an explicit measured fallback and lets the accuracy test compare both schedules in one process.

The final source used physical GPU 7 (`NVIDIA PG506-230`, sm_80, 124 SMs, 98,304 MiB), PyTorch 2.13.0+cu130,
CUDA 13.0, and the repository AWQ JIT flags:

```text
C++:  -O3 -std=c++17 -DENABLE_BF16 -D_GLIBCXX_USE_CXX11_ABI=1
NVCC: -O3 -std=c++17 -DENABLE_BF16 -D_GLIBCXX_USE_CXX11_ABI=1
      --threads 8 --optimize=3 -Xptxas -O3,-dlcm=ca -lineinfo
      -Xfatbin -compress-all -diag-suppress=179,39,177
```

Full Nsight Compute captures show the intended work reduction:

| BF16 M=128 GEMM | M16 q/o | M32 q/o | M16 gate/up | M32 gate/up |
|---|---:|---:|---:|---:|
| duration | 76.93 us | 64.26 us | 198.02 us | 155.10 us |
| grid / block threads | 1,024 / 64 | 512 / 128 | 3,072 / 64 | 1,536 / 128 |
| executed instructions | 21,815,296 | 16,709,632 | 67,620,864 | 50,128,896 |
| registers/thread | 72 | 72 | 72 | 72 |
| static shared memory | 9.98 KiB | 11.26 KiB | 9.98 KiB | 11.26 KiB |
| achieved occupancy | 23.20% | 24.54% | 36.28% | 37.63% |

The reports are `qwen3_m128_qo_awq_m16_specialized_full_gpu7.ncu-rep`,
`qwen3_m128_qo_awq_m32_full_gpu7.ncu-rep`, `qwen3_m128_gateup_awq_m16_template_full_gpu7.ncu-rep`, and
`qwen3_m128_gateup_awq_m32_full_gpu7.ncu-rep` under `artifacts/paroquant_megakernel_20260723/`.

Eight 500-sample ABBA blocks used the production fallback switch around complete rotation-plus-AWQ-plus-bias
calls. Every M16/M32 output was bit-exact:

| dtype | projection | M16/M32 p50 us | p50 speedup | M16/M32 mean us | mean speedup |
|---|---|---:|---:|---:|---:|
| BF16 | q/o | 125.952 / 124.928 | 1.008x | 127.930 / 127.395 | 1.004x |
| BF16 | k/v | 115.712 / 114.688 | 1.009x | 117.797 / 116.879 | 1.008x |
| BF16 | gate/up | 202.752 / 162.816 | 1.245x | 203.179 / 163.471 | 1.243x |
| BF16 | down | 211.968 / 176.128 | 1.203x | 212.442 / 176.551 | 1.203x |
| FP16 | q/o | 120.320 / 119.808 | 1.004x | 123.600 / 124.090 | 0.996x |
| FP16 | k/v | 108.544 / 108.544 | 1.000x | 112.104 / 109.868 | 1.020x |
| FP16 | gate/up | 201.216 / 161.280 | 1.248x | 202.026 / 173.729 | 1.163x |
| FP16 | down | 207.360 / 172.544 | 1.202x | 207.294 / 173.382 | 1.196x |

Weighting Qwen3-8B projection multiplicities gives BF16 p50/mean speedups of 1.122x/1.119x and FP16 speedups of
1.122x/1.095x. The reproducible runner is `scripts/benchmark_paroquant_awq_m32_ab.py`; the final BF16 and FP16
artifacts are `qwen3_m128_awq_m32_final_bf16_ab_gpu7.json` and
`qwen3_m128_awq_m32_final_fp16_ab_gpu7.json`.

The requested full BF16 branch-versus-main comparison used five current repeats and three repeats of main
`0b0405f24facb073ee69b00a03291fdab9779408`, each with 50 warmups and 500 CUDA-event samples for all four
Qwen3-8B linear shapes. These are projected 36-layer linear-only times and throughput, not end-to-end model TPS:

| M | main/current p50 36-layer us | p50 speedup | main/current projected tok/s | main/current mean 36-layer us | mean speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 61,415.4 / 18,690.0 | 3.286x | 16.3 / 53.5 | 65,207.4 / 18,877.0 | 3.454x |
| 8 | 63,037.4 / 21,676.0 | 2.908x | 126.9 / 369.1 | 66,499.2 / 21,893.7 | 3.037x |
| 128 | 65,065.0 / 37,601.3 | 1.730x | 1,967.3 / 3,404.1 | 67,850.4 / 37,906.3 | 1.790x |
| 512 | 73,838.6 / 66,465.8 | 1.111x | 6,934.0 / 7,703.2 | 74,566.9 / 66,809.1 | 1.116x |

The aggregate artifact is `artifacts/paroquant_megakernel_20260723/qwen3_8b_m32_vs_main_0b0405f2.json`.
The four new FP16/BF16 dense-reference cases require the M32 output to be bit-exact to M16 FP32 reduction, never
increase maximum dense error, and reduce mean dense error versus the legacy reduction. Representative BF16
gate/up passed memcheck and synccheck with zero errors and racecheck with zero hazards/errors/warnings; BF16 down
also passed memcheck with zero errors. Immediately before commit, all 501 tests in
`tests/kernels/test_paroquant.py` and `tests/kernels/test_awq_cuda_fp32_reduce.py` passed on physical GPU 7
(16 warnings, 48.00 seconds).

## Bank-conflict-free M32 activation stores

The retained M32 tile initially assigned each four-lane vector-load quartet to adjacent activation rows. With the
40-element padded shared row, an eight-lane 128-byte transaction then addressed the same bank set twice. Full NCU
source counters attributed every excessive shared wavefront in the M32 kernel to that activation store. The new
M32-only bijection assigns lane `x` to row `(x >> 3) + (x & 4)` and K-vector `x & 3`, pairing rows four apart
inside each transaction. The global load and shared destination use the same mapping, so the logical activation
tile and MMA order do not change. M16 retains its established row layout and every other specialization is
unchanged.

The final-source BF16 reports on physical GPU 7 show the intended conflict removal without changing the 72
register/thread or 11.264 KiB shared-memory resource envelope:

| M32 M=128 GEMM | q/o before | q/o remapped | gate/up before | gate/up remapped |
|---|---:|---:|---:|---:|
| duration | 64.256 us | 63.296 us | 155.104 us | 153.344 us |
| executed instructions | 16,709,632 | 16,318,464 | 50,128,896 | 48,955,392 |
| excessive shared wavefronts | 262,144 | 0 | 786,432 | 0 |
| shared-store bank conflicts | 262,144 | 0 | 820,006 | 39,177 |
| achieved occupancy | 24.54% | 24.34% | 37.63% | 37.45% |

This is a 1.015x q/o and 1.011x gate/up kernel speedup. The unchanged M16 layout also benefited from the isolated
address expressions: its q/o capture moved from 76.928 to 76.064 us and from 21,815,296 to 21,616,640 executed
instructions, while retaining the same 262,144 shared-conflict wavefronts. The final reports are
`qwen3_m128_qo_awq_m32_row4_finalsource_full_gpu7.ncu-rep`,
`qwen3_m128_gateup_awq_m32_row4_finalsource_full_gpu7.ncu-rep`, and
`qwen3_m128_qo_awq_m16_preserved_full_gpu7.ncu-rep` under
`artifacts/paroquant_megakernel_20260723/`.

Eight 500-sample ABBA blocks around complete rotation-plus-AWQ-plus-bias calls remained bit-exact in both dtypes:

| dtype | projection | M16/M32 p50 us | p50 speedup | M16/M32 mean us | mean speedup |
|---|---|---:|---:|---:|---:|
| BF16 | q/o | 128.000 / 128.000 | 1.000x | 135.031 / 135.262 | 0.998x |
| BF16 | k/v | 117.760 / 117.760 | 1.000x | 125.813 / 124.085 | 1.014x |
| BF16 | gate/up | 202.240 / 160.768 | 1.258x | 203.439 / 161.334 | 1.261x |
| BF16 | down | 211.456 / 173.056 | 1.222x | 212.464 / 173.856 | 1.222x |
| FP16 | q/o | 116.736 / 116.736 | 1.000x | 118.149 / 117.932 | 1.002x |
| FP16 | k/v | 107.008 / 106.496 | 1.005x | 108.543 / 107.655 | 1.008x |
| FP16 | gate/up | 200.704 / 159.744 | 1.256x | 201.020 / 160.416 | 1.253x |
| FP16 | down | 205.824 / 171.520 | 1.200x | 206.082 / 171.364 | 1.203x |

The final weighted M16/M32 speedups are 1.123x/1.124x BF16 p50/mean and 1.125x/1.125x FP16. Memcheck and
synccheck each report zero errors on the exact final source; racecheck reports zero hazards, errors, and warnings.
The final ABBA artifacts are `qwen3_m128_awq_m32_row4_finalsource_bf16_repeat_gpu7.json` and
`qwen3_m128_awq_m32_row4_finalsource_fp16_gpu7.json`. Immediately before commit, all 501 tests in
`tests/kernels/test_paroquant.py` and `tests/kernels/test_awq_cuda_fp32_reduce.py` passed on physical GPU 7
(16 warnings, 46.95 seconds).

## BF16 gate/up packed-weight L2 caching

After removing the M32 activation-store conflicts, full NCU still attributed a material part of gate/up issue
latency to long-scoreboard stalls. The retained specialization loads packed weights with cache-global (`.cg`)
semantics so the 48 MiB gate/up weight stream bypasses L1 and uses L2. It is compile-time gated to BF16 and the
exact Qwen3-8B M=128, K=4096, N=12288, group-size-128, split-4 M32 route on the measured sm_80/124-SM target.
FP16, q/o, k/v, down, M16, all other shapes, and all other devices keep the ordinary packed-weight load.

Full NCU on physical GPU 7 measured:

| BF16 gate/up M32 M=128 | ordinary load | cache-global load |
|---|---:|---:|
| duration | 153.344 us | 151.936 us |
| executed instructions | 48,955,392 | 48,955,392 |
| registers/thread | 72 | 72 |
| shared memory | 11.264 KiB | 11.264 KiB |
| long-scoreboard cycles / issue-active cycle | 1.874 | 1.817 |
| L2 throughput | 28.75% | 36.30% |
| excessive shared wavefronts | 0 | 0 |

This is a 1.009x isolated-kernel speedup with unchanged instructions and resource use. Stable complete-call screens
placed the cache-global gate/up p50 near 159.744 us versus 160.768 us for the ordinary load; a later run measured
157.696 us, but unrelated GPU residency made the whole-suite timing distribution bimodal, so the fixed-clock NCU
result is the primary retention evidence. Applying `.cg` broadly was rejected: q/o regressed from 63.296 to
64.640 us in NCU, and FP16 was neutral/slower. The final ordinary-load q/o fallback retained 16,318,464
instructions, 72 registers/thread, 11.264 KiB shared memory, and zero excessive shared wavefronts.

The final focused accuracy gate passed all four M32 BF16/FP16 gate/up and down cases, including exact equality to
M16 FP32 reduction and the dense-reference error envelope. Memcheck and synccheck each report zero errors on the
exact BF16 gate/up launch; racecheck reports zero hazards, errors, and warnings. The primary artifacts are
`qwen3_m128_gateup_awq_m32_k4096_ldcg_full_gpu7.ncu-rep`,
`qwen3_m128_gateup_awq_m32_row4_finalsource_full_gpu7.ncu-rep`,
`qwen3_m128_qo_awq_m32_cache_fallback_full_gpu7.ncu-rep`, and
`qwen3_m128_awq_m32_bf16_gateup_ldcg_final_gpu7.json` under
`artifacts/paroquant_megakernel_20260723/`. Immediately before commit, all 501 tests in
`tests/kernels/test_paroquant.py` and `tests/kernels/test_awq_cuda_fp32_reduce.py` passed on physical GPU 7
(16 warnings, 47.41 seconds).

## Post-experiment Qwen3-8B branch-versus-main benchmark

After the previously running host experiment released the benchmark window, the full deterministic BF16
Qwen3-8B shape harness was rerun on physical GPU 7. The candidate was the published ParoQuant branch at
`af3b93b0a9dbe57f223c78e5e089f2f4f17b6660`; the freshly fetched baseline was
`origin/main@0b0405f24facb073ee69b00a03291fdab9779408`. Each revision ran in three fresh processes with 50 warmups
and 500 CUDA-event samples for q/o, k/v, gate/up, and down at M=1/8/128/512. All 96
revision/projection/row/repeat cases passed their dense-reference, normalized-error, and repeat-determinism gates
before timing.

The table uses the median of the three fresh-process statistics and weights projection multiplicity across all
36 Qwen3-8B layers. Throughput remains a projected linear-only rate, not end-to-end model TPS:

| M | main/current p50 36-layer us | p50 speedup | main/current projected tok/s | main/current mean 36-layer us | mean speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 61,599.7 / 18,653.2 | 3.302x | 16.2 / 53.6 | 64,049.9 / 19,585.9 | 3.270x |
| 8 | 62,152.7 / 21,749.8 | 2.858x | 128.7 / 367.8 | 65,642.1 / 22,158.1 | 2.962x |
| 128 | 64,290.8 / 37,232.6 | 1.727x | 1,991.0 / 3,437.8 | 68,086.7 / 38,381.0 | 1.774x |
| 512 | 73,396.2 / 66,502.7 | 1.104x | 6,975.8 / 7,698.9 | 74,844.8 / 68,655.0 | 1.090x |

At M=128, current/main p50 speedups were 1.749x q/o, 1.929x k/v, 1.658x gate/up, and 1.515x down. The unchanged
dense M=512 q/o route was 0.963x in this run, while the current CUDA-AWQ k/v route was 1.739x; their
multiplicity-weighted stack plus the essentially flat gate/up and down routes produced the aggregate 1.104x.
The aggregate artifact is
`artifacts/paroquant_megakernel_20260723/qwen3_8b_af3b93b0_vs_main_0b0405f2_post_experiment.json`.
Immediately before this log commit, all 501 tests in `tests/kernels/test_paroquant.py` and
`tests/kernels/test_awq_cuda_fp32_reduce.py` passed on physical GPU 7 (16 warnings, 48.97 seconds).

## M=512 q/o reused-weight AWQ tile

The M32 AWQ tile also wins at the exact Qwen3-8B M=512 q/o shape. Relative to M16, it halves the CTA count while
letting two 16-row warp groups reuse each dequantized K32xN128 weight tile. The retained extension is deliberately
narrow: M=512, K=4096, N=4096, group size 128, split-K4, FP16/BF16, and a runtime-probed sm_80 device with 124
SMs. M=512 k/v remains on M16, gate/up and down retain their established dense selections, the M=128 routes are
unchanged, and every other shape, architecture, or SM inventory keeps the existing fallback. The BF16
cache-global packed-weight specialization remains restricted to M=128 gate/up.

Four 50-warmup/500-sample ABBA pairs around complete rotation-plus-AWQ-plus-bias calls were bit-exact between M16
and M32 in both dtypes:

| q/o M=512 | M16 p50/mean/p95 us | M32 p50/mean/p95 us | p50 speedup | mean speedup |
|---|---:|---:|---:|---:|
| BF16 | 263.168 / 262.668 / 266.240 | 210.944 / 211.583 / 213.504 | 1.248x | 1.241x |
| FP16 | 260.096 / 260.022 / 262.144 | 209.920 / 210.773 / 211.968 | 1.239x | 1.234x |

Full Nsight Compute on the BF16 q/o GEMM confirms that reuse, rather than a resource-envelope change, produces
the gain:

| BF16 q/o M=512 GEMM | M16 | M32 | change |
|---|---:|---:|---:|
| duration | 246.59 us | 193.47 us | 1.275x |
| grid / block threads | 4,096 / 64 | 2,048 / 128 | CTA count halved |
| executed instructions | 86,466,560 | 65,273,856 | -24.5% |
| registers/thread | 72 | 72 | unchanged |
| static shared memory | 9.98 KiB | 11.26 KiB | +1.28 KiB |
| achieved occupancy | 37.37% | 38.15% | effectively unchanged |
| L2 hit rate | 94.79% | 92.44% | -2.35 points |

The report is
`artifacts/paroquant_megakernel_20260723/qwen3_m512_qo_awq_m16_m32_full_gpu7.ncu-rep`.
Three fresh-process production-harness repeats selected `cuda_awq` for candidate q/o at M=512 and passed every
dense-reference, normalized-error, and repeat-determinism gate. Against the preceding published branch median,
q/o moved from 278.528 to 206.848 us p50 (1.347x) and from 302.301 to 208.130 us mean (1.452x); the complete
projection-weighted 36-layer M=512 stack improved from 66,502.7 to 61,599.7 us p50 (1.080x), raising projected
linear-only throughput from 7,698.9 to 8,311.7 token/s. Against freshly fetched
`origin/main@0b0405f24facb073ee69b00a03291fdab9779408`, the same stack is 1.192x faster at p50 and 1.211x faster
on mean:

| M | main/current p50 36-layer us | p50 speedup | main/current projected tok/s | main/current mean 36-layer us | mean speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 61,599.7 / 18,911.2 | 3.257x | 16.2 / 52.9 | 64,049.9 / 19,343.5 | 3.311x |
| 8 | 62,152.7 / 21,823.5 | 2.848x | 128.7 / 366.6 | 65,642.1 / 22,275.7 | 2.947x |
| 128 | 64,290.8 / 37,269.5 | 1.725x | 1,991.0 / 3,434.4 | 68,086.7 / 37,743.1 | 1.804x |
| 512 | 73,396.2 / 61,599.7 | 1.192x | 6,975.8 / 8,311.7 | 74,844.8 / 61,821.9 | 1.211x |

The aggregate artifacts are `qwen3_8b_m512_qo_m32_vs_af3b93b0.json` and
`qwen3_8b_m512_qo_m32_vs_main_0b0405f2.json`. The expanded FP16/BF16 dense-reference test requires the new route
to remain bit-exact to M16 FP32 reduction and inside the independent dense-weight error envelope. Representative
BF16 M=512 q/o passed memcheck, initcheck, and synccheck with zero errors and racecheck with zero hazards, errors,
or warnings on physical GPU 7. The expanded pre-commit suite passed all 503 tests in
`tests/kernels/test_paroquant.py` and `tests/kernels/test_awq_cuda_fp32_reduce.py` on physical GPU 7
(16 warnings, 47.56 seconds).

## Deferred M=512 q/o asynchronous activation staging

A successful exact M=512 q/o prototype staged each thread's aligned 16-byte activation fragment with Ampere
`cp.async.cg` before performing the independent packed-weight load and dequantization work. The copy waits before
the existing producer/consumer barrier, so shared-memory contents, Tensor Core inputs, FP32 accumulation order,
split-K reduction order, output dtype, and bias ordering are unchanged. Bypassing L1 is useful here because the
same activation rows are shared across output tiles through L2 while the dequantized-weight path remains the
dominant L1/shared-memory consumer.

The prototype was gated to M=512, K=N=4096, group size 128, split four, FP16/BF16, and a runtime-probed sm_80
device with 124 SMs. It could not run for M=128, k/v, gate/up, down, another shape, architecture, or SM inventory.
After these measurements, the target workload was explicitly narrowed to M in {1, 2, 4, 8, 16, 32}; M128/M512
are no longer optimization targets. The M512-only source and test changes were therefore reverted rather than
committed. The published generic M32 path and all fallbacks remain unchanged.

Four 50-warmup/500-sample ABBA pairs around the complete rotation-plus-AWQ-plus-bias call remained bit-exact:

| q/o M=512 | sync p50/mean/p95 us | async `.cg` p50/mean/p95 us | p50 speedup | mean speedup |
|---|---:|---:|---:|---:|
| BF16 | 210.432 / 210.932 / 212.992 | 196.608 / 205.113 / 199.680 | 1.070x | 1.028x |
| FP16 | 209.408 / 209.688 / 211.968 | 197.632 / 198.071 / 200.192 | 1.060x | 1.059x |

The BF16 mean includes rare samples above p95 during external GPU contention; an earlier four-pair `.cg` reversal
measured 211.754/199.392 us mean (1.062x) with 211.456/199.168 us p50. Fixed-clock Nsight Compute confirms the
mechanism independently of those tails:

| BF16 q/o M=512 GEMM | synchronous M32 | async `.cg` M32 | change |
|---|---:|---:|---:|
| duration | 194.176 us | 176.992 us | 1.097x |
| executed instructions | 65,273,856 | 52,854,784 | -19.0% |
| registers/thread | 72 | 80 | +8 |
| static shared memory | 11.264 KiB | 11.264 KiB | unchanged |
| achieved occupancy | 38.14% | 33.78% | -4.36 points |
| L2 hit rate | 92.36% | 94.27% | +1.91 points |

Three fresh-process full Qwen3-8B runs for current and three for
`main@0b0405f24facb073ee69b00a03291fdab9779408` passed dense-reference, normalized-error, and deterministic-replay
gates before timing. The new change moved current M=512 q/o from the preceding branch's 206.848 to 196.608 us
p50 (1.052x) and 208.130 to 196.405 us mean (1.060x). The projection-weighted M=512 36-layer stack moved
61,599.7 to 60,936.2 us p50 (1.011x), raising projected linear-only throughput from 8,311.7 to 8,402.2 token/s.
Against fresh main, the complete current stack is 1.198x faster at M=512 p50 and 1.204x on mean:

| M | main/current p50 36-layer us | p50 speedup | main/current projected tok/s | main/current mean 36-layer us | mean speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 61,415.4 / 18,874.4 | 3.254x | 16.3 / 53.0 | 63,796.1 / 19,885.1 | 3.208x |
| 8 | 61,857.8 / 21,676.0 | 2.854x | 129.3 / 369.1 | 63,951.7 / 22,656.4 | 2.823x |
| 128 | 64,475.1 / 37,269.5 | 1.730x | 1,985.3 / 3,434.4 | 66,516.3 / 39,253.5 | 1.695x |
| 512 | 73,027.6 / 60,936.2 | 1.198x | 7,011.0 / 8,402.2 | 74,361.2 / 61,768.2 | 1.204x |

The preserved prototype artifacts are `qwen3_m512_qo_awq_sync_async_activation_cg16_final_{bf16,fp16}_ab_gpu7.json`,
`qwen3_m512_qo_awq_sync_async_activation_cg16_full_gpu7.ncu-rep`,
`qwen3_8b_async_cg_current_vs_5403f9bb_sync.json`, and
`qwen3_8b_async_cg_current_vs_main_0b0405f2.json`.

## Exact low-M Qwen split-K coverage

The active workload was narrowed to batch/row M in {1, 2, 4, 8, 16, 32}. A fresh BF16 Qwen3-8B baseline on
physical GPU 7 exposed a dispatch hole: only M=1/8 used the large-K mega-kernel, while M=2/4/16/32 fell back to
CUDA-AWQ. The retained change admits exactly the six requested rows for the four Qwen projection shapes:
K=4096/N in {1024, 4096, 12288} and K=12288/N=4096. The architecture check remains runtime-probed sm_80 with
124 SMs; other dtypes, shapes, row counts, architectures, and SM inventories retain autotuned fallbacks.
The measured device was an NVIDIA PG506-230, UUID `GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28`, with 96 GiB,
driver 610.43.02, compute capability 8.0, and 124 SMs. The software stack was PyTorch 2.13.0+cu130 and Triton
3.7.1; no CUDA extension build flags changed, and the new specializations were generated by the existing Triton
sm_80 JIT path.

M=2/4 reuse the existing split-32 schedule. Autotuning selects it for all four projections. At M=16 it selects
split-K for q/o and k/v while retaining CUDA-AWQ for gate/up and down. M=32 q/o adds a measured schedule with
BM32, eight warps, two sequential BN128 output tiles per CTA, paired completion counters, first-weight prefetch,
one stage, and a 128-register cap. K/v retains BM8; gate/up and down retain CUDA-AWQ. Cached non-mega plans bypass
split-K capability probing so the rejected fallback shapes do not pay a hot-path host-submission penalty.

Three fresh-process runs of the preceding commit and three candidate runs were executed sequentially on physical
GPU 7 with 50 warmups and 500 CUDA-event samples per case. Each run checked the ParoLinear reference, exact repeat
determinism, output shape/dtype, and normalized mean error before timing. The weighted stack is two q/o, two k/v,
two gate/up, and one down call per layer across 36 layers:

| M | preceding/candidate p50 36-layer us | p50 speedup | preceding/candidate projected tok/s | preceding/candidate mean 36-layer us | mean speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 18,800.6 / 18,874.4 | 0.996x | 53.2 / 53.0 | 19,799.7 / 19,226.5 | 1.030x |
| 2 | 33,988.6 / 21,381.1 | 1.590x | 58.8 / 93.5 | 36,047.3 / 21,632.8 | 1.666x |
| 4 | 33,804.3 / 21,528.6 | 1.570x | 118.3 / 185.8 | 35,684.3 / 22,207.6 | 1.607x |
| 8 | 21,676.0 / 21,970.9 | 0.987x | 369.1 / 364.1 | 22,068.4 / 22,346.6 | 0.988x |
| 16 | 34,172.9 / 25,030.7 | 1.365x | 468.2 / 639.2 | 37,174.2 / 25,670.8 | 1.448x |
| 32 | 34,799.6 / 27,002.9 | 1.289x | 919.6 / 1,185.1 | 37,050.8 / 27,594.0 | 1.343x |

M=1/8 execute unchanged schedules; their p50 movements are at most two 1.024-us event buckets and serve as the
noise control. The new projection-level p50 speedups are 1.701/2.095/1.293/1.373x at M=2 for q/o, k/v, gate/up,
and down; 1.671/2.145/1.265/1.337x at M=4; 1.671/2.031x at M=16 q/o and k/v; and 1.518/1.773x at M=32 q/o and
k/v. Cached gate/up and down fallback p50 remains equal or faster in the paired median.

All 24 forced split-K accuracy cases passed on GPU 7, including the slower shapes that production autotuning
rejects. Across the three production matrices, maximum absolute error was 0.25, maximum mean absolute error was
0.001640, maximum normalized mean error was 0.00925%, and every repeated output was bit-identical. The static
factor/launch/output selector suite passed 63 cases. Compute Sanitizer memcheck on the retained M=32,
K=N=4096 schedule reported zero errors. The final pre-commit run passed all 538 tests in
`tests/kernels/test_paroquant.py` on physical GPU 7 (16 warnings, 50.63 seconds).

Full Nsight Compute on M=32 q/o confirms the retained schedule launches 512 CTAs of 256 threads, uses
128 registers/thread and 40.96 KiB dynamic shared memory, reaches 22.82% achieved occupancy, executes 17.61
million instructions with zero local-memory spills, and hits L2 at 75.50%. Its 2.06 waves/SM and 29.12% SM
throughput identify tail imbalance and shared-memory bank conflicts as the next device-side limits rather than
DRAM bandwidth (4.34%).

The paired aggregate is
`artifacts/paroquant_megakernel_20260723/qwen3_8b_low_m_target_final_paired_vs_5403f9bb_bf16_gpu7.json`.
The profile is
`artifacts/paroquant_megakernel_20260723/qwen3_m32_qo_split32_pair_prefetch_full_gpu7.ncu-rep`.

## Cached low-M fallback submission and branch-versus-main comparison

The M=16/32 wide Qwen projections still correctly select CUDA-AWQ: forced complete-module measurements put
gate/up mega-kernel p50 at 200.704/386.048 us versus 63.488/73.728 us for the direct CUDA plan, and down at
201.728/387.072 us versus 129.024/135.168 us. Once autotuning has cached `cuda_awq`, the retained inference-only
fast return now submits that same fused rotation-plus-AWQ op before re-entering the general mega-kernel router.
Adapters, training/autograd, uncached plans, native-op failures, and all other paths retain the established flow.

Three fresh processes from `896a2d16` and three candidate processes ran sequentially on physical GPU 7 with
50 warmups and 500 CUDA-event samples. The four paths changed by this follow-up all improved in the paired median:

| Projection | M | preceding/candidate p50 us | p50 speedup | preceding/candidate mean us | mean speedup |
|:---|---:|---:|---:|---:|---:|
| gate/up | 16 | 135.168 / 128.000 | 1.056x | 137.964 / 135.662 | 1.017x |
| gate/up | 32 | 139.264 / 133.120 | 1.046x | 141.259 / 139.332 | 1.014x |
| down | 16 | 139.264 / 133.120 | 1.046x | 146.827 / 140.212 | 1.047x |
| down | 32 | 139.264 / 134.144 | 1.038x | 152.316 / 143.262 | 1.063x |

The unchanged M=16 q/o and k/v measurements were noisier in that six-process sequence, so the projection-weighted
M=16 mean moved backward even though both modified fallback means improved. The weighted p50 improved 1.021x at
M=16 and 1.024x at M=32. The controlled artifact is
`artifacts/paroquant_megakernel_20260723/qwen3_8b_cached_cuda_fast_vs_896a2d16_gpu7.json`.

After this experiment, the full BF16 Qwen3-8B matrix was rerun in three fresh candidate processes and compared
with three fresh `main@0b0405f24facb073ee69b00a03291fdab9779408` processes. Every process checked the ParoLinear
reference, output shape/dtype, normalized mean error, and exact repeated output before its 50 warmups and
500 timed samples per projection. The requested active set remains exactly M in {1, 2, 4, 8, 16, 32}; no
M=128/512 measurement participates:

| M | main/current p50 36-layer us | p50 speedup | main/current projected tok/s | main/current mean 36-layer us | mean speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 61,747.2 / 18,874.4 | 3.271x | 16.2 / 53.0 | 63,141.3 / 19,888.6 | 3.175x |
| 2 | 61,673.5 / 21,491.7 | 2.870x | 32.4 / 93.1 | 62,867.3 / 21,980.0 | 2.860x |
| 4 | 61,341.7 / 21,676.0 | 2.830x | 65.2 / 184.5 | 62,753.8 / 22,419.2 | 2.799x |
| 8 | 62,079.0 / 21,970.9 | 2.826x | 128.9 / 364.1 | 64,387.5 / 22,509.7 | 2.860x |
| 16 | 61,820.9 / 23,998.5 | 2.576x | 258.8 / 666.7 | 63,837.1 / 25,774.6 | 2.477x |
| 32 | 62,189.6 / 26,210.3 | 2.373x | 514.6 / 1,220.9 | 64,190.8 / 27,975.7 | 2.295x |

These are projection-weighted linear-only estimates for 36 Qwen3-8B layers, not end-to-end model TPS. The complete
comparison artifact is
`artifacts/paroquant_megakernel_20260723/qwen3_8b_low_m_current_cached_cuda_fast_vs_main_0b0405f2_gpu7.json`.

Before this follow-up commit, all 539 tests in `tests/kernels/test_paroquant.py` passed on physical GPU 7
(16 warnings, 48.84 seconds). That run includes all 24 forced Qwen large-K accuracy cases, deterministic repeat,
CUDA graph, concurrent-stream, live-buffer, FP16/BF16, and cached-routing coverage. The production matrices kept
the preceding maximum absolute/mean/normalized-mean error envelopes of 0.25, 0.001640, and 0.00925%.

## M=2/4 Qwen q/o row-tile continuation

The next device-side pass changed only exact BF16 Qwen q/o `M in {2, 4}, K=N=4096, split-K32` launches from
BM8/eight warps to BM4/four warps. K/v, gate/up, down, M=1/8/16/32, other dtypes, and all non-Qwen shapes keep
their preceding schedules. A 5,000-sample paired CUDA-graph run alternated both independently compiled kernels
over the same input, quantized buffers, and scratch on physical GPU 7:

| M | BM8/W8 p50/mean/p95 us | BM4/W4 p50/mean/p95 us | p50/mean speedup |
|---:|---:|---:|---:|
| 2 | 40.960 / 41.561 / 47.104 | 37.888 / 38.505 / 44.032 | 1.081x / 1.079x |
| 4 | 40.960 / 41.744 / 43.008 | 38.912 / 38.794 / 38.912 | 1.053x / 1.076x |

The BM4/W4 and BM8/W8 split-K outputs were bit-identical for both rows. Both also passed the dense non-split
reference gate; the maximum/mean differences were 2.0/0.000641 at M=2 and 2.0/0.000406 at M=4. The complete
production reference uses the tighter ParoLinear thresholds below.

Three fresh final-source processes and three preceding-commit processes then ran the complete six-row Qwen matrix.
The changed q/o p50 stayed in the same 1.024-us event bucket at both rows, while q/o mean improved from
82.870 to 80.982 us at M=2 (1.023x) and 81.189 to 79.909 us at M=4 (1.016x). Projection-weighted across the
complete 36-layer linear stack, M=2 improved 21,491.7 to 21,454.8 us p50 and 21,980.0 to 21,847.6 us mean;
M=4 improved 21,676.0 to 21,602.3 us p50 and 22,419.2 to 21,999.3 us mean. Unchanged rows moved in both
directions under process-level noise and are not attributed to this specialization. The controlled and whole-stack
artifacts are:

```text
artifacts/paroquant_megakernel_20260723/qwen3_m2_q_bm4w4_controlled_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_m4_q_bm4w4_controlled_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_low_m_m2m4_q_bm4w4_vs_117c3091_gpu7.json
```

The exact final source was also compared with the same three fresh
`main@0b0405f24facb073ee69b00a03291fdab9779408` processes used above. Every candidate process passed the
ParoLinear reference, normalized-error, output shape/dtype, and exact-repeat gates before timing:

| M | main/current p50 36-layer us | p50 speedup | main/current projected tok/s | main/current mean 36-layer us | mean speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 61,747.2 / 18,948.1 | 3.259x | 16.2 / 52.8 | 63,141.3 / 19,214.5 | 3.286x |
| 2 | 61,673.5 / 21,454.8 | 2.875x | 32.4 / 93.2 | 62,867.3 / 21,847.6 | 2.878x |
| 4 | 61,341.7 / 21,602.3 | 2.840x | 65.2 / 185.2 | 62,753.8 / 21,999.3 | 2.853x |
| 8 | 62,079.0 / 22,044.7 | 2.816x | 128.9 / 362.9 | 64,387.5 / 22,429.6 | 2.871x |
| 16 | 61,820.9 / 24,440.8 | 2.529x | 258.8 / 654.6 | 63,837.1 / 25,507.2 | 2.503x |
| 32 | 62,189.6 / 26,431.5 | 2.353x | 514.6 / 1,210.7 | 64,190.8 / 27,293.7 | 2.352x |

These remain projected linear-only rates, not model TPS. The comparison is
`artifacts/paroquant_megakernel_20260723/qwen3_8b_low_m_m2m4_q_bm4w4_vs_main_0b0405f2_gpu7.json`.
Immediately before commit, all 542 tests in `tests/kernels/test_paroquant.py` passed on physical GPU 7. This
includes all 24 production Qwen accuracy shapes, graph, stream, live-buffer, deterministic-repeat, FP16/BF16,
fallback, and selector coverage.

## M=2 q/o paired-output continuation

Nsight Compute 2025.3.1 profiled the retained M=2 q/o split-K32 kernel on physical GPU 7 with the following
targeted command; `cudaProfilerStart/Stop` in the benchmark isolated one warmed Triton launch:

```text
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=.:scripts ncu --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --kernel-name regex:paroquant_rotation_gemm_splitk_kernel --launch-count 1 \
  python scripts/benchmark_paroquant_triton_configs.py \
  --device 0 --dtype bf16 --m 2 --k 4096 --n 4096 --krot 8 ...
```

The raw report measured 48.29 us duration, 31.82% SM throughput, 38.23% aggregate memory throughput, and only
7.69% DRAM throughput. The 128-thread CTA used 150 registers/thread and 33.79 KiB dynamic shared memory, limiting
theoretical/achieved occupancy to 18.75%/17.19%. Schedulers had no eligible warp in 64.04% of cycles, and the
1,024-CTA grid ended after 2.75 waves/SM. This classified the kernel as latency/occupancy-bound rather than
DRAM-bound. The source report is
`artifacts/paroquant_megakernel_20260723/qwen3_m2_q_bm4w4_latency_gpu7.ncu-rep`.

Halving the grid to 512 CTAs with two sequential BN128 outputs was slower unless the paired-counter kernel used a
144-register cap and prefetched its first packed-weight tile. A 10,000-sample alternating CUDA-graph repeat
preserved exact output and improved the retained single-output kernel:

| M=2 q/o | single output | paired output | speedup |
|:---|---:|---:|---:|
| p50 us | 37.888 | 36.864 | 1.028x |
| mean us | 42.143 | 40.813 | 1.033x |
| p95 us | 39.936 | 38.912 | 1.026x |

The exact production-module comparison independently warmed the single-output and paired-output compiled launchers
over identical buffers, then alternated 10,000 complete calls AB/BA on the same stream. It measured
79.872/85.831/345.088 us versus 77.824/84.368/344.064 us p50/mean/p95, or 1.026x/1.017x p50/mean. Both variants
experienced the same external long-tail launch gaps; the candidate improved rather than hid that tail. Outputs
were bit-identical in the accepted run and both differed from ParoLinear by at most 0.125.

Three separate candidate processes produced M=2 q/o p50 values of 35.840, 36.864, and 75.776 us, but concurrent
host/GPU contention also produced 364-368 us p95 tails and changed the autotuned plans of otherwise untouched
M=1/4/8/16/32 shapes. That six-row matrix is retained locally but excluded from whole-stack acceptance. The
paired same-process complete-call result above is the retention evidence. The specialization is restricted to
BF16 `M=2, K=N=4096, split-K32`; M=4, k/v, gate/up, down, FP16, other shapes, devices, and fallbacks are unchanged.
The controlled artifacts are:

```text
artifacts/paroquant_megakernel_20260723/qwen3_m2_q_pair144_repeat_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_m2_q_pair144_module_ab_gpu7.json
```

Immediately before commit, all 545 tests in `tests/kernels/test_paroquant.py` passed on physical GPU 7. The run
includes every production Qwen accuracy shape, graph, concurrent-stream, live-buffer, deterministic-repeat,
FP16/BF16, fallback, launch-selector, and output-selector check.

## Rejected experiments

- After retaining BM4/W4 for M=2/4 q/o, a controlled output sweep kept BN128 and two stages. Stage one/three were
  flat or worse, eight warps regressed 10.8-13.2%, BN64 regressed 7.5-8.4%, and BN256 regressed 21.1-22.2%.
  A 128-register cap improved the raw kernel by one 1.024-us event bucket, but three complete-call processes
  moved q/o mean from 80.982 to 81.779 us at M=2 and from 79.909 to 82.436 us at M=4. The cap was rejected at
  the production boundary.
- Two-output CTAs improved the M=16 q/o raw kernel from 70.656/71.156 to 69.632/69.698 us p50/mean, but the
  three-process complete call was flat at p50 and regressed mean from 81.680 to 85.164 us. The same schedule
  regressed M=16 k/v from 28.672 to 30.720 us raw p50. M=32 k/v regressed from 43.008 to 44.032 us; paired
  counters with 96/112/128-register caps also lost.
- Output-reuse and register-cap sweeps at the remaining unchanged rows produced no candidate. At M=1 q/o, the
  retained 128-register single-output schedule measured 36.864 us versus 39.936 us for two outputs; at M=1 k/v,
  two outputs regressed 18.432 to 21.504 us. At M=8, q/o regressed 44.032 to 45.056 us and k/v regressed
  22.528 to 23.552 us. The tested caps also lost.
- Split-K16 at M=2/4 was slower than split-K32 for both q/o and k/v. Q/o raw p50 rose from 37.888 to
  45.056-55.296 us, while k/v rose from 20.480 to 22.528-24.576 us. Both split factors stayed within their
  dense-reference gates, so the rejection is performance-only.
- Using all 96 K groups as independent splits for the M=2/4 down projection also lost. Split-K32 versus split-K96
  W8 p50/mean was 107.520/120.606 versus 120.832/130.017 us at M=2 and 109.568/108.371 versus
  122.880/122.630 us at M=4. W4 and paired-output split-K96 variants were slower still; split-K32 remains.
- Extending BM4/W4 beyond exact M=2/4 q/o was rejected. M=2 k/v regressed to 66.560 us and M=4 k/v remained
  65.536 us; gate/up regressed to 103.424/109.568 us, and down no longer retained the mega-kernel plan.
- M=16 q/o BM16 reached 80.896 us versus the retained roughly 77-79 us. Four warps reached 87.040 us for q/o
  and 74.752 us for k/v; split-K16 reached the same 80.896/74.752 us. All retain BM8/W8 split-K32.
- M=32 k/v BM32 and BM16 reached 81.920 and 78.848 us. Split-K16 appeared faster in an isolated fresh process,
  but controlled alternating timing put split-K32 at 79.872/82.345/88.064 us and split-K16 at
  82.944/84.356/90.112 us p50/mean/p95. Four warps similarly reversed from a noisy fresh-process screen to
  83.968/85.569/93.184 us versus 76.800/82.082/87.040 us for eight warps. One stage regressed to 575.488 us.
- A streamed four-BN128 M=32 q/o CTA enlarged the live/resource envelope and regressed to 115.712 us. Independent
  counters for the two outputs reached 91.136 us. An atomic paired-counter reset reached 87.040 us p50 but
  88.809 us mean and did not improve the retained schedule. All prototypes were removed after exact-output checks.
- NCU source attribution localized the M=32 q/o shared conflicts to 16 compiler-generated Tensor Core staging
  instructions (`STS.128` and `LDSM`), each with 32,768 excessive wavefronts, rather than the explicit split-K
  scratch path. Prefetching both paired weights increased p50 from 87.040 to 89.088 us. Pairing two BN256 tiles
  halved the grid from 512 to 256 CTAs but enlarged the resource envelope and regressed p50 to 113.664 us.
  Both candidates were reverted; the retained schedule still prefetches only the first of two BN128 tiles.
- Applying BM16/BM32 row tiles to every M=16/32 projection slowed M=16 k/v and did not make the wide gate/up or
  down schedules beat CUDA-AWQ. The retained native row tile is restricted to M=32, K=N=4096.
- M=32 q/o BM32 with four warps regressed p50 to 110.592 us. BN64 lost selection to CUDA-AWQ, BN512 regressed to
  129.024 us, and the single BN256 tile reached 89.088 us but remained slower than two sequential BN128 tiles.
  The paired BN128 schedule reached 88.064 us; first-weight prefetch reduced it to 87.040 us.
- Two stages regressed the paired M=32 q/o schedule from 87.040 to 94.208 us. Register caps of 112/96 did not
  improve its p50 and worsened mean/tail timing, while 144 registers regressed p50 to 99.328 us. The retained
  schedule uses one stage and 128 registers.
- Capping the asynchronous M32 kernel to the synchronous path's seven-block/72-register envelope forced spills
  and regressed BF16 p50 from 210.944 us synchronous to 241.152 us async (0.875x). The retained async kernel uses
  80 registers/thread because its lower occupancy is more than offset by copy/dequantization overlap.
- Splitting each 16-byte activation copy into two 8-byte `cp.async.ca` operations preserved exact output but
  reduced the complete-call BF16 gain to 211.456/210.432 us (1.005x). A two-stage activation-only pipeline added
  2.5 KiB shared memory but also lost, at 210.432/219.136 us (0.960x). One 16-byte single-stage copy is retained.
- Caching asynchronous activation copies at all levels (`cp.async.ca`) improved BF16 p50 from 210.432 to
  200.704 us (1.048x), but cache-global copies reached 211.456/199.168 us (1.062x) and improved fixed-clock
  duration another 5.024 us. The measured M=512 specialization therefore uses `.cg`.

- Exact M=512 q/o row tiles wider than M32 did not produce a stable end-to-end win. M64 reduced executed GEMM
  instructions from 65.27 to 53.23 million and halved the grid, but raised registers from 72 to 77/thread and
  lowered achieved occupancy from 38.27% to 34.20%; fixed-clock NCU improved only 192.992 to 192.544 us.
  Three production repeats put q/o p50 at 206.848/204.800 us for M32/M64, but a longer focused run reversed to
  0.995x p50 and 0.989x mean. M128 regressed M64's 205.824 us to 300.032 us, and a half-warp M64 metadata shuffle
  regressed M32's 209.920 us to 215.040 us. All candidates were exact to M32 before timing and were reverted.
  The artifacts are `qwen3_m512_qo_awq_m32_m64_full_gpu7.ncu-rep`,
  `qwen3_8b_m512_qo_m64_candidate_repeat{1,2,3}.json`,
  `qwen3_m512_awq_m64_m128_bf16_ab_gpu7.json`, and
  `qwen3_m512_awq_m32_m64_shuffle_bf16_ab_gpu7.json`.
- Reducing exact M=512 q/o from split four to split two preserved determinism and stayed inside the dense-reference
  envelope, with 0.25 maximum and 0.000012 mean absolute cross-split drift. It nevertheless moved direct p50 from
  204.800 to 205.824 us, while fixed-clock GEMM duration regressed from 194.176 to 211.104 us and achieved
  occupancy fell from 38.14% to 32.60%. The fixed-order split-four contract remains unchanged. Results are in
  `qwen3_m512_qo_awq_split4_split2_bf16_ab_gpu7.json` and
  `qwen3_m512_qo_awq_split4_split2_full_gpu7.ncu-rep`.
- Wider fixed-order M=512 reducer blocks were too small and unstable to retain. A four-way 256/512-thread screen
  initially moved complete-call p50 from 208.896 to 207.872 us, but a focused reversal measured
  204.800/205.824 us. Fixed-clock reducer duration improved only 26.848 to 26.592 us despite occupancy moving
  from 67.48% to 70.49%. The 256-thread reducer is retained. Reproducible results are
  `qwen3_m512_qo_awq_reduce_threads_bf16_gpu7.json`,
  `qwen3_m512_qo_awq_reduce_256_512_bf16_focused_gpu7.json`, and
  `qwen3_m512_qo_awq_reduce_threads_256_512_full_gpu7.ncu-rep`.
- Giving each M=512 reducer thread two separated `float4` output vectors halved its CTA grid while preserving the
  exact split-add order and warp-contiguous accesses. Four 50-warmup/500-sample BF16 ABBA pairs nevertheless
  regressed complete-call p50 from 210.432 to 211.968 us, mean from 211.055 to 212.338 us, and p95 from
  212.992 to 214.016 us. The one-vector-per-thread reducer remains in place; the rejected run is
  `qwen3_m512_qo_awq_vec4_vec8_bf16_ab_gpu7.json`.
- Compiling M=512, K=N=4096, group 128, and split four into a shape-specialized M32 kernel reduced executed
  instructions from 65.27 to 45.60 million, but unconstrained NVCC allocation grew from 72 to 80 registers/thread,
  reduced occupancy from 38.09% to 34.02%, and regressed fixed-clock duration from 194.528 to 218.720 us. Holding
  both generic and static kernels to the same seven-block/72-register launch envelope made p50 exactly flat at
  210.944 us; mean improved only 0.3% while p95 regressed from 213.504 to 214.016 us. Integer address work was
  hidden under the memory/scoreboard floor, so the specialization was reverted. Results are
  `qwen3_m512_qo_awq_generic_static_metrics_gpu7.ncu-rep` and
  `qwen3_m512_qo_awq_generic_static_both_lb7_bf16_ab_gpu7.json`.
- Folding the BF16 gate/up split-K reduction into whichever CTA increments a per-tile completion counter last
  reduced the fused specialization from 72 to 64 registers/thread and added only four shared-memory bytes
  (11,264 to 11,268 bytes). All four FP16/BF16 gate/up and down M32 dense-reference cases passed, and the
  bias-bearing gate/up benchmark remained bit-exact to M16. The reduction tail nevertheless erased the M32 gain:
  four 50-warmup/500-sample ABBA pairs measured current separate-reducer M16/M32 p50 at
  199.680/157.696 us, while the last-CTA build measured 203.264/208.896 us. The candidate was only 0.973x versus
  its same-run M16 control and was reverted. Reproducible JSON is
  `qwen3_m128_awq_m32_lastcta_{disabled,candidate}_bf16_ab_gpu7.json`.
- Combining each reducer thread's two BF16 bias loads and two BF16 output stores into 64-bit operations eliminated
  all 196,608 excessive NCU sectors and reduced executed instructions from 995,328 to 946,176 without changing
  32-register occupancy. The complete ten-case AWQ/ParoQuant reducer suite passed, including exact BF16 reduction
  and the four Qwen dense-reference cases. Lower transaction count did not lower latency: fixed-clock reducer
  duration regressed from 16.61 to 16.86 us, and four 50-warmup/1,000-sample ABBA blocks moved gate/up M32 p50
  from 159.744 to 160.768 us. The 64-bit load/store pair was reverted.
- Replacing the retained BF16 gate/up cache-global (`.cg`) packed-weight load with the streaming (`.cs`) policy
  preserved exact output but regressed fixed-clock NCU duration from 151.936 to 156.256 us. Long-scoreboard stalls
  rose from 1.82 to 1.84 cycles per issued instruction and L2 throughput fell from 36.30% to 28.10%. The L2-only
  cache-global policy is retained.
- An exact BF16 gate/up M64 tile halved row-CTA count and packed-weight/dequantization work, reducing executed
  instructions from 48.96 million to 39.92 million, but its two independent loads per warp raised long-scoreboard
  stalls from 1.82 to 3.25 and regressed NCU duration from 151.936 to 168.576 us. Restricting weight production to
  four warps reduced instructions further to 25.98 million, but the other four warps waited at each K barrier:
  barrier and long-scoreboard stalls reached 6.12 and 5.98 cycles per issued instruction and duration regressed to
  189.120 us. Both M64 variants passed the four-case exact/dense accuracy gate before timing and were reverted.
- Launching the four M32 row CTAs consecutively for each gate/up N tile was exact, but did not improve shared-L2
  temporal locality in practice. Fixed-clock NCU duration regressed from 151.936 to 156.992 us, long-scoreboard
  stalls rose from 1.82 to 1.91 cycles per issued instruction, L2 throughput fell from 36.30% to 34.64%, and the
  grid-remap address work added 18,432 executed instructions. The original N-tile-first grid order is retained.
- Extending the cache-global packed-weight load to exact BF16 down was exact but regressed fixed-clock NCU
  duration from 174.720 to 178.560 us. Long-scoreboard stalls rose from 2.11 to 2.13 cycles per issued instruction
  and L2 hit rate fell from 77.39% to 69.47%. Down retains its ordinary load; the `.cg` specialization remains
  limited to BF16 gate/up.
- Reordering activation-load lane quartets in the original M16 tile paired shared rows four apart and eliminated
  all 262,144 excessive shared-store wavefronts without changing output, but the q/o NCU duration regressed from
  76.096 to 77.120 us. M16 therefore retains its original lane order; the separately measured M32-only remap above
  is retained.
- Prefetching all four packed M32 weight words before the independent activation and zero/scale work was exact,
  but reduced the weighted BF16 M16/M32 p50 ratio from 1.123x to 1.121x and down-projection ratio from 1.222x to
  1.200x. Prefetching only the first word still reduced weighted p50/mean to 1.121x/1.120x. A preliminary
  two-word buffer was rejected by the focused accuracy gate before timing because it did not cover all four
  per-thread loads. The original load schedule is retained.
- Loading each M32 K-group's scales and zero packs once with one half-warp, staging them in 320 bytes of shared
  memory, and preloading the next group after the existing B-store barrier was exact but serialized the producer
  warp and added shared traffic. Gate/up regressed from 160.768 to 222.720 us and the weighted BF16 M16/M32 p50
  ratio fell from 1.123x to 0.950x, so the cached per-warp metadata loads are retained.
- Adding a 15-CTA launch bound reduced the q/o GEMM from 72 to 64 registers/thread and raised theoretical
  occupancy from 43.75% to 46.88%, but it serialized the load/dequant path. Long-scoreboard stalls reached 70.9%
  and duration regressed from 76.096 to 137.536 us. The uncapped launch is retained.
- The existing AWQ N=64 tile doubled k/v M=128 CTA count from 256 to 512, but it did not convert the measured
  underfill into a stable module win. N=128/N=64 measured 129.024/128.000 us p50 while mean moved
  131.216/131.878 us in the wrong direction. The established N=128 tile was restored.
- A two-element reducer increased q/o and k/v CTA counts, but full Nsight Compute put q/o at 8.832 us and k/v at
  5.504 us versus 7.488 and 5.312 us for vec4. The four-element reducer is retained.
- CUDA-AWQ split factors were rescreened after combined dispatch. M=128 q/o split 8 appeared 7.4% faster in one
  sweep, but an eight-block 500-sample AB reversal put split 4/split 8 at 156.160/157.184 us p50 and
  158.363/159.373 us mean. Gate/up split 2 and down split 8 retained only 2.6% and 1.8% p50 gains while changing
  FP32 reduction ordering. Three independent dense-reference seeds kept the same 0.25-0.5 maximum error and
  changed mean error only in the fifth or sixth decimal place, but the small speedups did not justify a
  shape-specific numerical contract. The default split 4 remains unchanged.
- Generalizing prepared-state lookup to regular FP16 prefill added roughly 2-12 us to the four checked production
  shapes instead of removing dispatch work. An exact `aecbb133` worktree on the same GPU measured split-K p50
  values of 110.592, 92.160, 104.448, and 117.760 us at M128/N512, M497/N512, M129/N1920, and M96/N4096. The
  generalized candidate reached 114.688, 103.424, 114.688, and 119.808 us. Decode-only preparation restores the
  original prefill launcher; a follow-up run measured 111.616, 98.304, 108.544, and 117.760 us amid matching
  movement in the paired standard route.
- M=128, K=N=4096 split-32 reached 271.360 us raw versus about 210-221 us for the complete CUDA-AWQ route, so
  large-K prefill remains on the established fallback. The default cap changed only to admit exact M=1/8 BF16
  Qwen shapes.
- `maxnreg=128` improved M=1 q/o, gate/up, and down but regressed k/v and every tested M=8 shape. M=8 q/o moved
  from 44.032 to 49.152 us raw, and M=8 gate/up moved from 103.424 to 104.448 us; those schedules remain uncapped.
- For M=1, K=N=4096, stages one and two tied at 36.864 us with the register cap. BLOCK_N=64 and two-output
  variants lost to the retained single-output BLOCK_N=128 kernel. Pairing output counters also lost at N=4096.
- M=8, K=12288, N=4096 split-96 one-output, two-output, and paired variants measured 125.952, 146.432, and
  145.408 us p50 versus 114.688 us for retained split-32. The wider split remains M=1-only.
- Artificial config screens with much larger legacy quant scales reached 0.006612 mean absolute split drift at
  M=8/K=12288. The production-scale Qwen harness was substantially tighter, with normalized mean error no greater
  than 0.0092%, and all retained routes passed the dense-reference envelope before timing.
- Nsight Compute on FP16 M=1/N=8192 reported 31.88% SM throughput, 8.10% DRAM throughput, 25% theoretical
  occupancy, 60.04% scheduler cycles with no eligible warp, and a 32-CTA tail after two full waves. Stages one,
  two, and three were exact but shared the same 34.816 us p50; BM1 and W8 variants lost. Marking repeated
  activation/rotation metadata loads `evict_last` regressed p50 from 34.816 to 35.840 us and was reverted.
- The N=640/M=896-1184 BM64/W16 prefill tile beat the current fused BM16 kernel by 13.3-14.8% raw, reaching
  228.352 us p50, but the complete module selected `cuda_awq` at 189.440-194.560 us p50. The mega-kernel route
  therefore remains disabled in this later band.
- Inlining the validated native launch removed one Python closure/call and improved isolated M=1/N=2048 block
  medians by one 1.024 us bucket, but the full ten-shape FP16/BF16 table was mixed and several gates regressed by
  the same bucket. Skipping the current-device query when only one CUDA device was visible was also flat in paired
  blocks. Both host shortcuts were reverted.
- Later narrow-prefill BM64 schedules improved the raw fused kernel but still lost to the established complete
  route. At N=640/M=1185, BM16, BM32/W16, BM64/W16, and BM128/W16 measured about 319.5, 277.5, 228.4, and
  399.4 us p50. BM64/W16/stage-1 was the best raw candidate, but its 228.4 us floor remained above the established
  selector before module overhead, so no new prefill band was enabled.
- FP16 split factors four and eight lost to split 16. At M=1/N=2048 their raw p50 values were 27.648 and
  21.504 us versus 19.456 us; at N=8192 they were 65.536 and 54.272 us versus 37.888 us. BLOCK_N=128 also beat
  the tested narrower and wider output tiles.
- Letting the last split CTA retain and reuse its own partial preserved the fixed reduction order but increased
  N=2048 split p50 from 19.456 to 23.552 us. A symmetric-zero specialization was exact for valid packed zeros but
  worsened split p50 from 19.456 to 20.480 us at N=2048 and 39.936 to 43.008 us at N=8192. Both were reverted.
- Four-warp BF16 split geometries sometimes improved raw device mean, but repeated same-process full-module
  comparisons were mixed. M=1 BM2/W4 and BM8/W8 both measured 77.824 us module p50, with BM2 slightly worse on
  mean; M=2 BM4/W4 moved repeated block means in both directions. BF16 keeps its previously validated schedules.
- A prepared compiled launcher for regular prefill did not remove measurable dispatch cost: controlled
  M=129/N=1920 timing put the full module, internal method, wrapper, and direct compiled path in the same
  143.360 us p50 bucket. It was reverted.
- Feature-detected direct calls to PyTorch's bound CUDA capture-state and current-device queries did not produce a
  stable six-shape win. An eight-block four-way M=1/N=2048 comparison found no p50 gain from the direct capture
  query; the direct device query moved p50 by one 1.024 us bucket. A subsequent six-shape public/direct-device
  comparison left p50 flat at four gates and improved N=8192/M=1 and N=2048/M=8 by 1.2% and 1.4%, but mean
  regressed 2.4% at N=512/M=1 and 2.3% at N=2048/M=2. Both private shortcuts were reverted, and the public
  PyTorch capture/device guards remain.
- Replacing the rotation metadata tuple with cached tensor identity/version checks reduced an eight-repeat
  6,000-call host-loop median only from 52.605 to 52.188 us/call. Paired CUDA-event timing was flat/noisy, so the
  more complex cache state was rejected; live buffer pointers and versions continue to be checked directly.
- Allocating the N=8192 split result directly at the caller's three-dimensional shape regressed p50 in four
  8,000-sample repeats: baseline/candidate p50 pairs were 93.184/95.232, 93.184/94.208, 91.136/93.184, and
  90.112/91.136 us. Mean improved once and regressed three times. N=8192 retains the final reshape.
- A cached shape-only output template makes `empty_like` faster than `new_empty` in isolation, but the required
  dictionary lookup reduces the median advantage to about 0.49 us (3.214 -> 2.723 us). That sub-event-bucket gain
  does not justify persistent CUDA tensor cache state by itself and was not implemented.
- A global 4-row/4-warp split-K CTA reduced graph-batched device time but did not reliably improve the complete
  compiled launch at N=512/2048 or M=2-8. Those gates retain the established 8-row/8-warp schedule; only the
  independently paired M=1/N=8192 gain is enabled.
- Decode BM2 did not improve BF16 p50 and slightly regressed mean; BM1 regressed p50/mean/p95. A later eight-warp
  retest retains BM2 only for FP16 `krot=8` at M=1/2; BM4 remains the FP16 M=3-8 decode row tile.
- Local `int8` partners regressed FP16 decode from 110.592 / 113.232 / 128.000 us to
  112.640 / 115.265 / 131.072 us p50/mean/p95 on GPU 7. They also regressed BF16 `krot=1` by 1.9-3.0%, so the
  compact table is restricted to BF16 `krot=8`.
- Prefill BM4 at `M=128, N=512` regressed BF16 p50 from 93.184 to 118.784 us. BM8 remains the small-N default
  outside the retained 124-SM N=512/M=497-992, N=384/M=657-1312, N=256/M=993-1984, and N=128/M=1985-3968
  BM32 bands.
- Prefill BM32 was flat/slower for BF16 `M=128, N=2048`, and at FP16 `M=64` it regressed p50 from 118.784 to
  152.576 us. FP16 M=112/225/256 and every later adjacent non-2:1 wave ratio also regressed. For N>=640, BM32 is
  retained only for exact `K=2048`, 128-aligned N=640-4096, `krot=8` FP16 shapes where the runtime calculation
  finds twice as many BM16 waves. N<2048 is capped at the measured first BM32 wave; N=2048 is capped at four waves
  on the 124-SM target; N>2048 is restricted to the measured first BM32 wave on that target. The separate
  N in {128, 256, 384, 512} FP16/BF16 bands use their measured BM8-to-BM32 wave rule.
- N=1024's second raw 4:2 wave band (M=737-992) is bit-exact and improves the mega-kernel by 8.8-9.6%, but loses
  end to end. At M=864 the established route's 214.016/220.228/232.448 us p50/mean/p95 beat the mega-kernel's
  297.984/298.265/299.008 us; at M=992, 208.896/212.001/225.280 beat 300.032/300.596/301.056 us. The N=1024 cap
  remains one BM32 wave. BM64 reduces the M=864 raw kernel further to 253.952 us with 8 warps and 228.352 us with
  16 warps/one stage, but still misses the established route. Explicit FMA is neutral at 228.352 us and 32 warps
  regresses to 262.144 us, so the larger tile is also rejected.
- A prepared internal launcher skipped repeated public-wrapper shape and metadata validation after module gating.
  Its outputs were bit-identical, but concurrent ABBA measurements left most medians in the same event bucket and
  moved repeated mean/p95 results in both directions. The sub-resolution dispatch-only change was reverted.
- Caching the mega-kernel cosine and signed-sine lookups in FP16 reduced metadata from 192 to 128 KiB, but changed
  the native rotation's FP32 coefficient contract. FP16 prefill-K candidate-versus-baseline mean absolute error rose
  from about 0.0059 to 0.2061 and the maximum reached 1.5, so the experiment was reverted before profiling.
- Local `int16` partner indices reduced prefill registers in one specialization, but regressed prefill-K p50 by
  6.7-8.6% across FP16/BF16. They are retained only for decode; prefill continues to use global `int32` indices.
- `BLOCK_M=8` is a large win at prefill `N=512` but regresses the `N=2048` projection by 13.4-16.6% p50. The row
  tile therefore remains shape-specific instead of replacing the wider prefill configuration.
- `BLOCK_N=32/64/256` sweeps created either more duplicated rotation work or a larger resource envelope and did not
  produce a stable p50/mean/p95 improvement. After retaining eight warps, repeated `BLOCK_N=64/256` and one/three
  stage checks still lost or remained flat, so `BLOCK_N=128` remains unchanged. Two stages remain the default;
  only the separately measured 124-SM wide FP16 BM32 schedule uses one.
- The earlier global/int16-partner sweep did not retain eight decode warps. Repeating the experiment after the
  local-partner code-generation changes produced the retained 16-25% gains above. Raising the block again to
  16 warps regressed p50/mean by about 24-27% in both dtypes and was rejected.
- Packing FP32 cosine/sine bit patterns into one 64-bit load preserved exact output but regressed BF16 p50 by 1.1%
  and FP16 p50/mean by 4.5%/4.7%; integer unpack work outweighed the removed pointer/load instruction.
- Unsigned 8-bit local partner offsets kept FP16 p50 flat in two 4,000-sample repeats and improved mean only
  0.19-0.25%; BF16 was flat/slightly slower. The sub-resolution result was reverted.
- An interleaved FP32 cosine/sine AoS layout preserved output but regressed BF16 decode from about 100 to 136 us.
  The contiguous split arrays remain in place.
- Outer-loop `num_stages=2/3` hints left p50/p95 unchanged and moved means by only about 0.2%; disabling LICM
  regressed BF16 and was neutral for FP16. Both compiler-hint experiments were reverted.
- Under FP16 BM2, `BLOCK_N=64/256` regressed p50 by about 24%/17%. Removing guaranteed-true K-tail predicates was
  neutral at p50 and worse on mean. Bypassing L1 with `.cg` regressed p50 by about 12%.
- Flattening the nested K/rotation loop and disabling accumulator multibuffering regressed FP16 decode p50 by
  about 19% and 16%, respectively. The default loop structure remains in place.
- Reversing the explicit FMA to `fma(paired, sin, a * cos)` improved single-token timing by another 1-2%, but
  changed 4-10% of output elements on independent FP16/BF16 seeds and also changed M=8/prefill output. The retained
  FMA direction is the bit-identical lowering.
- Moving coefficient loads before the gather regressed BF16 by about 40%; merely loading sine before cosine also
  regressed BF16. Explicitly widening partner indices before broadcast was neutral/slightly worse.
- Removing the partner mask, changing its fallback to scalar zero, or marking the partner table `evict_last`
  perturbed code generation and regressed one or both dtypes. Prefetching the next rotation's partner vector
  increased live state and regressed FP16/BF16 p50 by about 11%/23%.
- Prefetching the small zero/scale vectors across the rotation loop increased their live ranges and regressed
  FP16/BF16 decode p50 by about 16%/15%. Loads remain adjacent to dequantization.
- Reordering all lookup tables by K group preserved exact output, but regressed FP16 decode from
  92.160/95.464/106.496 us to 104.448/109.174/121.856 us and left BF16 p50/p95 flat with a worse mean.
- First-partner prefetch is not enabled for FP16: it regressed decode from 92.160/95.743/105.472 us to
  101.376/105.805/116.736 us. BF16 wide prefill is also excluded by the N=1024/2048 regressions above.
- BF16 K=256 decode did not produce a stable p50/mean/p95 win across ordering repeats, so the contiguous retained
  prefetch range starts at K=384.
- Holding both rotation-zero and rotation-one partner vectors increased live state and regressed BF16 decode from
  80.896/84.546/94.208 us to 90.112/94.866/105.472 us. Only the retained first-partner vector stays live across
  channel scaling.
- Partially unrolling the inner rotation loop by two preserved exact output but regressed BF16 decode from
  80.896/83.190/94.208 us to 83.968/86.851/98.304 us. Splitting rotation zero out of the loop was neutral in one
  ordering and slightly slower in the reverse ordering, so the compact loop remains.
- Moving the first partner load before the activation, or moving both partner and channel-scale loads before the
  activation, left p50/p95 unchanged at 80.896/94.208 us and moved mean by less than 0.1%. The retained
  activation/partner/scale source order remains.
- Capping BF16 decode registers with Triton's `maxnreg` preserved exact output but introduced spills. The retained
  80.896/81.368/81.920 us p50/mean/p95 became 81.920/82.532/82.944 at 96 registers,
  84.992/85.426/86.016 at 88, 82.944/83.564/83.968 at 80, and 102.400/103.665/103.424 at 72. The uncapped launch
  remains in place.
- Splitting each rotated K=128 group into two ordered K=64 Tensor Core dot calls preserved exact output but added
  dot/setup cost. BF16 decode regressed from 80.896/82.986/94.208 to 82.944/85.359/96.256 us. FP16 remained flat
  at 92.160/106.496 us p50/p95 and mean moved from 95.539 to 95.674 us. The single K=128 dot remains in place.
- Caching BF16 channel scales as FP16 preserved exact multiplication inputs and output, but regressed decode from
  80.896/82.723/94.208 to 81.920/83.242/94.208 us. The existing BF16 metadata plus in-kernel conversion remains.
- Enabling first-partner prefetch, explicit FMA, or both for `krot=1` was exact but sub-resolution. All BF16
  variants remained at 45.056/52.224 us p50/p95 and all FP16 variants at 43.008/50.176 us; means were flat or
  slightly worse. The lighter one-rotation path keeps both switches disabled.
- K-loop factors two/four regress FP16 decode, both `krot=1` dtypes, and BF16 wide prefill. Only the measured BF16
  `krot=8, K<384` decode and exact small-N prefill regimes retain unrolling.
- The regular non-split mega-kernel loses to the established route at `K=4096/8192`. Broad large-K dispatch
  remains rejected; only exact BF16 Qwen M in {1, 2, 4, 8, 16, 32} split-K schedules bypass the `K<=2048` gate,
  and production autotuning retains CUDA-AWQ whenever the measured mega-kernel is slower.
- Prefill with `N=8192` loses to the existing AWQ path because duplicated rotations and the fused resource envelope
  outweigh launch savings. Prefill therefore remains capped at `N<=4096`, with N>2048 limited to the measured
  FP16 first-wave bands on the 124-SM target; decode retains the profitable wider-N route.

## Local artifacts

Profiler and benchmark artifacts are intentionally not committed. The current workspace paths are:

```text
artifacts/paroquant_megakernel_20260723/final_exact_fp16/
artifacts/paroquant_megakernel_20260723/final_exact_bf16/
artifacts/paroquant_megakernel_20260723/baseline_decode_gpu3.nsys-rep
artifacts/paroquant_megakernel_20260723/baseline_prefill_gpu4.nsys-rep
artifacts/paroquant_megakernel_20260723/candidate_decode_gpu3.nsys-rep
artifacts/paroquant_megakernel_20260723/candidate_prefill_gpu4.nsys-rep
artifacts/paroquant_megakernel_20260723/ncu/decode_q_gpu3_sol.ncu-rep
artifacts/paroquant_megakernel_20260723/ncu/decode_q_gpu3_occupancy.ncu-rep
artifacts/paroquant_megakernel_20260723/ncu/prefill_q_gpu4_sol.ncu-rep
artifacts/paroquant_megakernel_20260723/ncu/prefill_q_gpu4_occupancy.ncu-rep
artifacts/paroquant_megakernel_20260723/partner_graph_repeat/
artifacts/paroquant_megakernel_20260723/prefill_config/
artifacts/paroquant_megakernel_20260723/prefill_q_config/
artifacts/paroquant_megakernel_20260723/followup_final/
artifacts/paroquant_megakernel_20260723/gpu7_only/
artifacts/paroquant_megakernel_20260723/split16_decode_bf16.ncu-rep
artifacts/paroquant_megakernel_20260723/splitk_module_ab.json
artifacts/paroquant_megakernel_20260723/splitk_raw_ab.json
artifacts/paroquant_megakernel_20260723/splitk_module_unchecked_ab.json
artifacts/paroquant_megakernel_20260723/splitk_module_compiled_ab.json
artifacts/paroquant_megakernel_20260723/splitk_module_hookfast_ab.json
artifacts/paroquant_megakernel_20260723/splitk_module_launchfast_ab.json
artifacts/paroquant_megakernel_20260723/splitk_module_livebuffer_final_ab.json
artifacts/paroquant_megakernel_20260723/splitk_direct_launcher_candidate_ab.json
artifacts/paroquant_megakernel_20260723/splitk_shaped_output_candidate_ab.json
artifacts/paroquant_megakernel_20260723/splitk_shaped_output_final_ab.json
artifacts/paroquant_megakernel_20260723/splitk_shaped_output_n512_repeat_ab.json
artifacts/paroquant_megakernel_20260723/splitk_shaped_output_final_repeat_ab.json
artifacts/paroquant_megakernel_20260723/splitk_graph_m1n2048_candidate_ab.json
artifacts/paroquant_megakernel_20260723/splitk_graph_final_ab.json
artifacts/paroquant_megakernel_20260723/splitk_direct_input_final_ab.json
artifacts/paroquant_megakernel_20260723/splitk_direct_input_graph_check_ab.json
artifacts/paroquant_megakernel_20260723/splitk_runtime_api_baseline_ab.json
artifacts/paroquant_megakernel_20260723/splitk_runtime_api_candidate_ab.json
artifacts/paroquant_megakernel_20260723/splitk_current_device_candidate_ab.json
artifacts/paroquant_megakernel_20260723/splitk_current_device_final_ab.json
artifacts/paroquant_megakernel_20260723/splitk_last_scratch_final_ab.json
artifacts/paroquant_megakernel_20260723/splitk_last_scratch_graph_check_ab.json
artifacts/paroquant_megakernel_20260723/fp16_split_m1_n512_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m1_n2048_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m1_n8192_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m2_n2048_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m4_n2048_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m8_n2048_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_production_module_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_production_graph.json
artifacts/paroquant_megakernel_20260723/fp16_split_m1_n8192_profile.ncu-rep
artifacts/paroquant_megakernel_20260723/fp16_split_n8192_stage_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_n8192_metadata_evictlast.json
artifacts/paroquant_megakernel_20260723/prefill_n640_m896_bm64_screen.json
artifacts/paroquant_megakernel_20260723/prefill_n640_m1024_bm64_screen.json
artifacts/paroquant_megakernel_20260723/prefill_n640_m1184_bm64_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m3_n2048_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m5_n2048_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m6_n2048_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m7_n2048_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_irregular_module_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_irregular_graph_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_widths_module_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_widths_graph_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{2,3,4,5,6,7,8}_n512*_screen.json
artifacts/paroquant_megakernel_20260723/fp16_split_m{2,3,4,5,6,7,8}_n8192*_screen.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_main_repeat{1,2,3}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_current_repeat{1,2,3}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_current_vs_main.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_large_k_tuned_full_repeat{1,2,3}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_large_k_tuned_vs_main.json
artifacts/paroquant_megakernel_20260723/qwen_k{4096,12288}_*_split*_screen.json
artifacts/paroquant_megakernel_20260723/qwen_k4096_m1_n4096_split32*_gpu7.ncu-rep
artifacts/paroquant_megakernel_20260723/qwen3_large_k_m1_eager_timeline_gpu7.nsys-rep
artifacts/paroquant_megakernel_20260723/qwen3_8b_prepared_minimal_repeat{1,2,3}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_prepared_minimal_vs_main.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_prepared_minimal_vs_aecbb133.json
artifacts/paroquant_megakernel_20260723/fp16_prefill_{aecbb133,prepared_launch,decode_only_prepared}_eager_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_m128_cuda_awq_timeline_gpu7.nsys-rep
artifacts/paroquant_megakernel_20260723/qwen3_m128_combined_cached_nested_dispatch_ab_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_m128_split_candidate_{repeat_ab,dense_accuracy}_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_combined_dispatch_current_repeat{1,2,3}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_main_0b0405f2_repeat{1,2,3}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_combined_dispatch_vs_{main_0b0405f2,41cd6f11}.json
artifacts/paroquant_megakernel_20260723/qwen3_m128_{qo,kv}_awq_{gemm,reduce,vec2_reduce,vec4_reduce}_full_gpu7.ncu-rep
artifacts/paroquant_megakernel_20260723/qwen3_8b_{scalar_reduce,vec4_reduce,vec4_bias_current}_repeat{1,2,3}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_vec4_bias_vs_{c30e0511,main_0b0405f2}.json
artifacts/paroquant_megakernel_20260723/qwen3_m128_gateup_awq_m32_{row4_finalsource,k4096_ldcg}_full_gpu7.ncu-rep
artifacts/paroquant_megakernel_20260723/qwen3_m128_qo_awq_m32_{k4096_ldcg,cache_fallback}_full_gpu7.ncu-rep
artifacts/paroquant_megakernel_20260723/qwen3_m128_awq_m32_bf16_gateup_ldcg_final_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_m512_awq_m32_qo_narrow_{bf16,fp16}_ab_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_m512_qo_awq_m16_m32_full_gpu7.ncu-rep
artifacts/paroquant_megakernel_20260723/qwen3_8b_m512_qo_m32_candidate_repeat{1,2,3}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_m512_qo_m32_vs_{af3b93b0,main_0b0405f2}.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_low_m_{prev_5403f9bb,target_final}_paired_bf16_gpu7_r*.json
artifacts/paroquant_megakernel_20260723/qwen3_8b_low_m_target_final_paired_vs_5403f9bb_bf16_gpu7.json
artifacts/paroquant_megakernel_20260723/qwen3_m32_qo_split32_pair_prefetch_full_gpu7.ncu-rep
```

## Next optimization targets

The active batch-M target set is now exactly {1, 2, 4, 8, 16, 32}; M128/M512 are excluded from new optimization
and acceptance decisions. Split-K removes the dominant FP16/BF16 K=2048 decode underfill and extends the same
mechanism to exact Qwen3-8B BF16 K=4096/12288 small-M shapes. Decode-only prepared submission closes much of the measured
eager-versus-graph gap without changing prefill, graph capture, live-buffer, stream, or fallback behavior.
Further GPU work must remain on physical GPU 7 unless the user changes that constraint. The next passes should
change one mechanism at a time:

1. Profile the retained M=16/32 q/o and k/v split-K schedules independently, since the streamed four-BN128,
   paired BN256, second-weight-prefetch, narrower split, and alternate row/warp schedules are measured losses.
2. Sweep only the exact M=2/4 q/o output schedule around the retained BM4/W4 row geometry; changes must improve
   controlled complete-call timing as well as the raw CUDA-graph kernel.
3. Keep M=16/32 wide gate/up and down on their specialized cached CUDA-AWQ path unless a new native schedule
   beats that complete-call baseline; the current mega-kernel schedules are measured losses.
4. Determine whether a native output-allocation/launch entry point can reduce the remaining small-M host-submission
   floor without returning aliased outputs or bypassing live-buffer and stream ownership.
5. Determine whether pre-zeroed capture-stream scratch can safely remove the graph counter-reset node without
   sharing mutable state across independently replayable graphs; retain the reset when ownership is ambiguous.

Every retained follow-up must improve paired raw-kernel p50, preserve the full-operator selection outcome, be
cross-checked on FP16 and BF16 with a narrow dtype gate when only one wins, and keep the explicit backend gate and
all fallbacks intact. Mean and tail results must be reported, including noisy reversals, rather than hidden behind a
single summary statistic.
