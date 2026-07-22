# Fused EoRA inference on Marlin: profiling and optimization log

## Goal and current status

This work targets eager, decode-like and small-M EoRA/LoRA adapter inference
on the GPTQ Marlin backend. The correctness contract is:

```text
out = marlin(x, qweight, scales) + (x @ lora_A) @ lora_B
```

The first retained optimization is commit `f0ddd939` (`Optimize fused EoRA
Marlin inference`) on branch `paroquant-update-0721`. It replaces the three
library kernels used by the adapter tail with one cooperative CUDA kernel on
eligible `sm_80` devices. A portable PyTorch path remains available for every
unsupported shape, dtype, device, compressed-LoRA mode, CUDA graph capture, or
runtime launch failure.

The second retained pass is commit `a69f4410` (`Reduce EoRA Marlin launch
overhead and scratch`) and is published in PR #25. The third retained
pass, commit `e9391ef7` (`Fuse Marlin and EoRA native dispatch`), links the
EoRA kernel into each dtype-specific Marlin extension and enqueues Marlin plus
EoRA from one native operator.

The fourth retained pass, commit `cb1276be` (`Streamline prepared EoRA Marlin
dispatch`), specializes that native operator for immutable,
post-initialized GPTQ W4A16 module state, sends contiguous 1-D/2-D/3-D inputs
directly through native dispatch, and reuses transient Marlin FP32 reduction
scratch as the EoRA workspace.

The fifth retained pass specializes the CUDA adapter kernel for the common
ranks 64 and 128 while preserving the runtime-rank implementation for every
other valid adapter. On the corrected Torch 2.13 environment, contiguous 2-D
full-forward p50 reached 35.84-69.63 us for FP16 and 35.84-68.61 us for BF16
across the six measured shapes.

The sixth retained pass specializes the same common ranks for single-row
decode. It removes per-CTA row division and folds the workspace row bound
without changing the existing multi-row or generic-rank kernels.

The seventh retained pass splits the common single-row up projection into one
fully coalesced warp per 32 ranks: two warps for rank 64 and four for rank 128.
The warps compute independent FP32 partials and reduce through the existing
shared buffer. Rank-128 EoRA kernel median falls from 14.880 to 10.400 us and
rank-64 falls from 10.416 to 8.608 us, with unchanged 32-register usage,
1.02 KiB shared memory, and allocator-visible VRAM. Final full-forward p50 is
34.82-69.63 us for FP16 and 35.84-68.61 us for BF16. Subsequent scratch
packing reduces the original 1,992-2,760 KiB small-M peaks to 136-365.5 KiB,
and the common prepared path owns no persistent per-layer EoRA workspace.

The eighth retained pass specializes the rank-128 single-row down reduction.
Adjacent lanes accumulate neighboring input positions for one rank, then an
in-warp shuffle combines each pair without the generic shared-memory write,
read, and block barrier. Across 200 exact-source launches, the FP16 kernel
median falls from 10.368 to 10.240 us on physical GPU 4 and the BF16 median
falls from 10.399 to 10.208 us on physical GPU 5. Nsight Compute replay time
falls from 23.100 to 21.760 us, while registers, shared memory, spills,
allocator-visible VRAM, and every fallback remain unchanged.

The ninth retained pass gives the rank-128 single-row down phase up to 256 of
the output projection's already-launched CTAs instead of capping participation
at 128. It is active only when the output is wide enough to provide the extra
blocks; rank 64, runtime ranks, multi-row calls, and the 4096-column attention
projection retain the 128-CTA cap. For the 4096-to-11008 MLP up projection,
the FP16 kernel median falls from 13.024 to 10.944 us on physical GPU 4 and the
BF16 median falls from 12.928 to 11.008 us on physical GPU 5. Repeated matched
full-forward FP16 p50 falls from 44.032 to 41.984 us, BF16 p50 falls from
45.056 to 43.008 us, and allocator-visible VRAM remains unchanged.

The tenth retained pass increases fused-grid parallelism for strongly
contracting rank-128 single-row projections. When `K >= 2N`, the output tile
narrows from 32 to 16 columns, so the 11008-to-4096 MLP down projection uses
256 CTAs for both phases instead of 128. Its FP16 kernel median falls from
16.543 to 13.152 us on physical GPU 4 and BF16 falls from 16.496 to 13.184 us
on physical GPU 5. Matched full-forward p50 improves from 50.176 to 46.080 us
FP16 and from 51.200 to 48.128 us BF16. Multi-row, rank-64, generic-rank, and
non-contracting projections retain their prior launch geometry and VRAM.

The eleventh retained pass packs Marlin's FP32 global-reduction scratch for
`M <= 8`. Those kernels consume only four alternating fragment groups, so the
temporary slice stride and allocation now cover eight live rows instead of a
padded 16. Full-call peak allocation falls from 264 to 136 KiB for decode
attention and MLP down, from 709.5 to 365.5 KiB for MLP up, and from 320 to
192 KiB at M=8. M=16 and larger paths are unchanged. A matched BF16 Nsight
Systems source toggle measures the Marlin median at 14.976 us before and
14.912 us after, confirming that the memory reduction has no kernel-latency
cost.

The twelfth retained pass compacts those four fragment groups by the actual
live output rows for `M < 8` and gives the dominant M=1 path a direct
single-row mapping. Decode full-call peak allocation falls again from 136 to
24 KiB, while MLP-up falls from 365.5 to 64.5 KiB. Relative to the original
implementation, the three M=1 production shapes now use 96.78-99.13% less
transient allocator memory. A final BF16 Nsight Systems capture measures the
Marlin kernel at 14.848 us, slightly below the published 14.912 us median, so
the additional compaction also has no latency cost. M=8 and larger paths are
unchanged.

The thirteenth retained pass decouples rank-128 square-attention launch width
from its 32-column up-projection tiling. The original 128 CTAs still own every
output column, while 64 additional CTAs participate only in the down
projection and naturally receive empty output ranges after the cooperative
barrier. A 128/160/192/224/256-CTA Nsight Systems sweep selects 192: the BF16
EoRA kernel median falls from 10.528 to 9.920 us (-5.78%). Nsight Compute replay
duration falls from 22.11 to 19.42 us and achieved occupancy rises from 12.34%
to 17.25%, with unchanged 32-register use, 1.02 KiB shared memory, zero spills,
and allocator-visible VRAM. The policy is compile-time gated to single-row
rank 128 and runtime gated to non-contracting square projections; rank 64,
MLP up/down, multi-row, generic-rank, and every fallback route are unchanged.

The fourteenth retained pass combines the dominant single-row rank-128
4096-to-4096 Marlin and EoRA path into one cooperative GPU launch. Its 124
256-thread CTAs complete Marlin, synchronize the grid, reuse dead FP32 Marlin
reduction scratch for the LoRA down projection, synchronize again, and add the
LoRA up projection to the base output. A final BF16 Nsight Systems capture
measures one 23.328 us GPU kernel instead of a 26.495 us two-kernel span
(-11.95%) and reduces the profiled host range from 68.881 to 59.689 us
(-13.34%). Peak allocation remains 24 KiB. The specialization is runtime
gated to tested 124-SM `sm_80` devices with cooperative launch support; every
other shape, rank, architecture, and device retains the established fallback.

The fifteenth retained pass exposes memory-level parallelism inside the
mega-kernel's LoRA-down phase. Eight independent FP32 accumulation chains
interleave its otherwise serial down-weight loads before the same lane-pair
and cross-CTA reductions. BF16 Nsight Systems kernel median falls from 23.328
to 20.832 us (-10.70%), while FP16 finishes at 20.576 us. Nsight Compute shows
long-scoreboard delay falling from 4.33 to 1.71 cycles per issued instruction
and measured memory throughput rising from 340.12 to 446.96 GB/s. Register
use, shared memory, spills, allocator peak, launch count, and all fallback
code remain unchanged.

The sixteenth retained profiling pass removes two residual tails inside that
single launch. First, otherwise-idle warps 4-7 in blocks 0-3 compute the four
LoRA-up tiles left over by the 124-block grid instead of running those tiles
as a serial second pass. Second, the LoRA-down projection maps each CTA to a
32-rank tile so every warp reads adjacent LoRA-A weights and the cross-CTA
atomic count falls from 15,872 to 3,968. Against an exact-source baseline,
the final raw kernel median falls from 20.544 to 19.456 us FP16 (-5.30%) and
from 20.896 to 19.632 us BF16 (-6.05%). Peak allocation remains 24 KiB;
registers, dynamic shared memory, launch count, spills, and fallback dispatch
are unchanged.

The seventeenth retained pass overlaps LoRA-down with the exact 114-CTA
Marlin stripe schedule. Eight role-specialized CTAs compute two K slices for
each 32-rank tile and retain their partials in shared memory until Marlin
finishes. They then publish into dead Marlin reduction scratch without
workspace clearing or atomics. A 122-CTA launch also removes two idle tail
CTAs while six upper-warp tiles cover the remaining output columns. Across
1,000-launch Nsight Systems captures, raw-kernel p50 falls from 19.552 to
18.784 us FP16 (-3.93%) and from 19.936 to 19.168 us BF16 (-3.85%). Peak
allocation stays at 24 KiB, the operation remains one launch, and all
unsupported configurations retain their prior fallback.

## Historical reproduction environment

The passes above through the initial mega-kernel work used PCI bus ordering
and exposed only physical GPUs 4 and 5. The later GPU-2/3 audit and current
optimization environment are recorded with their corresponding results
below.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=/root/GPT-QModel-Ultra-2
export TORCH_CUDA_ARCH_LIST=8.0
export PATH=/root/vm314t/bin:$PATH
```

The live inventory was queried at runtime rather than embedded in dispatch:

| Physical GPU | Logical device | PCI bus | GPU | CC | SMs | Memory |
|---:|---:|:---|:---|:---:|---:|---:|
| 4 | 0 | `00000000:A0:00.0` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 5 | 1 | `00000000:A5:00.0` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |

```text
Driver                 610.43.02
Python interpreter     /root/vm314t/bin/python
Python                 3.14.5 free-threaded build
PyTorch                2.13.0+cu130
CUDA runtime/toolkit   13.0 / 13.0.88
Triton                 3.7.1
Transformers           5.14.1
torchvision            0.28.0+cu130
Nsight Systems         2024.6.2
Nsight Compute         2025.3.1
```

The first four retained passes and their historical tables were originally
measured with PyTorch 2.12.0+cu130 in the same environment. After the
environment moved to PyTorch 2.13.0+cu130, the exact fourth-pass source was
rebuilt and remeasured before comparing the fifth-pass candidate. All
baseline-to-candidate results below therefore use matched Torch 2.13 builds.

The representative production benchmark uses synthetic packed Marlin W4A16
weights with 4 bits, group size 128, symmetric quantization, no activation
ordering, FP16 or BF16 compute, and a dense FP32 LoRA reference. Final timings
use 500 warmups and 3,000 synchronized CUDA-event samples. Nsight captures use
shorter bounded ranges as stated with each artifact.

## Baseline profile before optimization

The formal baseline was captured before changing the kernel, for decode
`M=1, K=4096, N=4096, rank=128, FP16`.

```bash
nsys profile --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=artifacts/eora_marlin_20260721/baseline_marlin_addmm_decode_r128_nsys \
  python scripts/benchmark_eora_marlin_fused.py \
  --device cuda:0 --dtype fp16 --scope marlin --variants addmm \
  --case-pattern decode_attn_r128 --warmup 30 --iters 20 --profile
```

Nsight Systems showed four GPU launches per layer:

| Stage | Calls/layer | Mean GPU time | Source/operation |
|:---|---:|---:|:---|
| Quantized base | 1 | 15.33 us | Marlin W4A16 kernel |
| LoRA down | 1 | 4.82 us | cuBLAS GEMV for `x @ A` |
| LoRA down reduction | 1 | 3.07 us | cuBLASLt split-K reduction |
| LoRA up and add | 1 | 6.03 us | tensor-core GEMM with beta=1 |

The three adapter kernels consumed only about 13.9 us of raw GPU time, but
their CUDA launch and Python/operator dispatch cost dominated eager layer
latency. Nsight Compute also showed that the down GEMV launched 256 blocks for
only 0.26 waves, achieved 23.85% occupancy, and reached about 171 GB/s. The
operation was too small to saturate either compute or memory bandwidth.

An existing scalar `lora_up_add` CUDA experiment was also profiled. Its up
kernel took about 15.8 us versus 6.0 us for the library up/add GEMM and did not
remove the two down-projection kernels. It remains opt-in and is not selected
by the optimized production route.

## Retained cooperative kernel

The retained `lora_fused_add` operator uses one cooperative grid:

1. Zero an FP32 `[rows, rank]` workspace.
2. Synchronize the grid.
3. Compute `x @ lora_A` across up to 128 CTAs per row with FP32 accumulation.
4. Reduce per-block partials in shared memory before issuing FP32 atomics.
5. Synchronize the grid.
6. Compute `down @ lora_B`, add the existing Marlin output, and store in the
   original FP16 or BF16 dtype.

The original measured launch used 128 blocks by 256 threads for the
representative rank-128 decode shape. The final square-attention specialization
launches 192 blocks: the original 128 retain 32-column up-projection ownership,
and 64 down-only blocks increase reduction parallelism without overlapping
output stores. Earlier block-local reduction lowered the fused kernel median
from 18.75 us to 18.46 us by reducing common-rank atomic traffic by 2-6x.

Production preparation caches the dtype-specific integrated Marlin+EoRA
operator and LoRA tensors at module post-initialization. The common GPTQ
W4A16, group-128, no-bias, no-activation-order, full-K, FP32-reduction path
with at least two quantization groups also caches a narrow prepared dispatcher.
That dispatcher derives dimensions
from the already prepared tensors, allocates output directly in the input's
1-D/2-D/3-D shape, and omits optional operands and invariant flags from the
boxed call. The validated integrated operator remains selected for every
other supported Marlin configuration.

The hot forward path invokes the selected operator once; native code enqueues
both GPU kernels without returning through Python between launches. The
generic integrated workspace grows only when an eligible call has more rows
than previously observed. The common prepared path instead lends EoRA the
already-live transient Marlin FP32 reduction scratch after Marlin finishes on
the same stream.

Automatic selection currently requires:

- a live CUDA device with compute capability exactly 8.0;
- FP16 or BF16 dense LoRA tensors;
- contiguous input and Marlin output;
- flattened rows at most 16 by default;
- rank at most 512 by default; and
- no active CUDA stream capture.

`GPTQMODEL_EORA_MARLIN_COOPERATIVE=0` disables only the cooperative route.
`GPTQMODEL_EORA_MARLIN_FUSED=0` disables all optional EoRA-Marlin fusion.
Architecture and cooperative-launch properties, SM count, occupancy, device,
and stream are queried at runtime; no fixed CUDA index or product string is
used for dispatch.

## Current performance

The following values are full packed-Marlin-plus-adapter forward p50 latency,
not isolated kernel time. The sweep uses contiguous 2-D linear inputs, as used
by flattened inference engines. Each value uses 500 warmups and 3,000 samples
under PyTorch 2.13. A separate `[1, 1, 4096]` transformer-shaped FP16 benchmark
measures 38.91 us integrated versus 150.53 us fallback.

### FP16 on physical GPU 4

| Case | M | K | N | Rank | Fallback | Cooperative | Reduction |
|:---|---:|---:|---:|---:|---:|---:|---:|
| decode attention | 1 | 4096 | 4096 | 64 | 145.41 us | 31.74 us | 78.17% |
| decode attention | 1 | 4096 | 4096 | 128 | 150.53 us | 30.72 us | 79.59% |
| decode MLP up | 1 | 4096 | 11008 | 128 | 149.50 us | 37.89 us | 74.66% |
| decode MLP down | 1 | 11008 | 4096 | 128 | 153.60 us | 39.94 us | 74.00% |
| batch 8 attention | 8 | 4096 | 4096 | 128 | 151.55 us | 53.25 us | 64.86% |
| rows 16 attention | 16 | 4096 | 4096 | 128 | 151.55 us | 69.63 us | 54.05% |

### BF16 on physical GPU 5

| Case | M | K | N | Rank | Fallback | Cooperative | Reduction |
|:---|---:|---:|---:|---:|---:|---:|---:|
| decode attention | 1 | 4096 | 4096 | 64 | 144.38 us | 33.79 us | 76.60% |
| decode attention | 1 | 4096 | 4096 | 128 | 144.38 us | 30.72 us | 78.72% |
| decode MLP up | 1 | 4096 | 11008 | 128 | 143.36 us | 37.89 us | 73.57% |
| decode MLP down | 1 | 11008 | 4096 | 128 | 148.48 us | 41.98 us | 71.73% |
| batch 8 attention | 8 | 4096 | 4096 | 128 | 148.48 us | 54.27 us | 63.45% |
| rows 16 attention | 16 | 4096 | 4096 | 128 | 144.38 us | 68.61 us | 52.48% |

The cooperative columns are exact thirteenth-pass sequential measurements.
The fallback columns remain the most recent unchanged-route measurements. The
final sweep uses the documented 500 warmups and 3,000 CUDA-event samples;
whole-forward values still move by one or two 1.024 us event ticks between
runs. Direct matched Nsight captures isolate the two scratch-layout passes:
the eight-row packing moved Marlin from 14.976 to 14.912 us, and final M=1
live-row compaction measures 14.848 us. The final attention pass is attributed by its
direct 10.528-to-9.920 us EoRA kernel comparison rather than the noisier eager
forward range. The ninth pass targets wide rank-128 expansion, and the tenth
targets strong rank-128 contraction; their matched source toggles and profiler
medians below isolate each incremental gain.

The optimized eager path issues two kernels from one native operator: Marlin
plus the fused adapter. Its adapter kernel performs more raw GPU work than the
three highly tuned library kernels combined, but removes two CUDA launches,
two adapter operator dispatches, and the Python round trip between Marlin and
EoRA. During CUDA graph capture, where launch overhead is amortized,
production deliberately uses the library path with lower raw GPU work.

## Nsight profiler results

For `M=1, K=N=4096, rank=128, FP16`, the fourth-pass runtime-rank fused
operator reported the following Nsight Compute metrics before specialization:

| Metric | Value |
|:---|---:|
| Grid / block | 128 / 256 threads |
| Kernel median in Nsight Systems | 18.30 us |
| Registers per thread | 32 |
| Static shared memory per block | 1.02 KiB |
| Local-memory spills | 0 |
| Achieved occupancy | 6.46% |
| Compute throughput | 3.49% |
| DRAM throughput | 2.21% |
| Memory throughput | 53.99 GB/s |
| L1/TEX hit rate | 28.39% |
| L2 hit rate | 36.44% |
| Long-scoreboard stall share | about 49% |

The low occupancy is primarily a workload-size effect: the grid has roughly
one block per live SM and two global synchronization phases. The subsequent
tuning therefore targeted launch/dispatch overhead and memory dependency
latency rather than adding generic occupancy machinery.

A fresh exact-revision Nsight Systems baseline measured a 10.848 us median gap
from the end of Marlin to the start of cooperative EoRA and a 44.384 us median
two-kernel span. Enqueuing both kernels from the integrated native operator
reduces those values to 1.280 us and 34.272 us respectively. The integrated
trace reports 14.592 us Marlin and 18.304 us EoRA kernel medians, confirming
that the improvement comes from dispatch removal rather than skipped or
changed GPU work.

The fourth-pass bounded trace compares the exact `e9391ef7` native-dispatch
baseline with the final prepared W4A16 dispatcher over 49 warmed calls after
discarding the first profiled iteration:

| Nsight Systems median | `e9391ef7` | Prepared dispatch | Change |
|:---|---:|---:|---:|
| CPU NVTX range | 85.083 us | 69.463 us | -18.36% |
| Range start to first CUDA launch | 46.158 us | 34.847 us | -24.50% |
| Marlin launch API | 9.127 us | 7.915 us | -13.28% |
| Marlin kernel | 14.784 us | 14.656 us | -0.87% |
| Inter-kernel GPU gap | 1.280 us | 1.376 us | +0.096 us |
| EoRA kernel | 18.272 us | 18.304 us | +0.032 us |
| Two-kernel GPU span | 34.368 us | 34.335 us | -0.033 us |

The instrumented CUDA-event p50 moves from 81.696 to 68.160 us. The production
3,000-sample FP16 rank-128 p50 moves from the prior pass's 47.10 us to 37.89 us.
The effectively unchanged GPU kernels and span confirm that the additional
gain is host-side preparation and boxed-dispatch reduction.

The fifth-pass comparison rebuilt both the exact `cb1276be` source and the
fixed-rank candidate with PyTorch 2.13. Over 49 warmed calls, after discarding
the first profiled iteration, Nsight Systems reports:

| Nsight Systems median | Runtime rank | Static rank 128 | Change |
|:---|---:|---:|---:|
| CPU NVTX range | 73.941 us | 75.263 us | +1.79% |
| Range start to first CUDA launch | 37.542 us | 38.433 us | +2.37% |
| Marlin launch API | 8.476 us | 8.155 us | -3.79% |
| Marlin kernel | 14.624 us | 14.752 us | +0.88% |
| Inter-kernel GPU gap | 1.280 us | 1.376 us | +0.096 us |
| EoRA kernel | 18.304 us | 17.024 us | -6.99% |
| Two-kernel GPU span | 34.272 us | 33.184 us | -3.18% |

The CPU range is profiler-noisy and did not predict production timing. A
separate tail-only capture gives a cleaner EoRA median of 18.128 us before and
16.704 us after specialization, a 1.424 us or 7.85% reduction. Synchronized
production p50 improves in all 12 dtype/shape cases, while peak allocation and
numerical error are unchanged.

The next exact Torch 2.13 NCU pair explains the fixed-rank gain and identifies
the remaining single-row opportunity. Runtime rank 128 executes 805,469
instructions; the rank-128 specialization executes 640,856, a 20.4% reduction,
with the same 32 registers per thread and effectively unchanged occupancy.
NCU replay duration falls from 38.688 to 37.696 us. Source/SASS mapping then
showed every CTA still evaluating `blockIdx / blocks_per_row` even though a
single-row launch has `row=0` and `row_block=blockIdx`.

The sixth-pass compile-time single-row specialization removes that division
only for ranks 64 and 128. The generic-rank and multi-row instantiations retain
their previous code. Its matched full-forward Nsight Systems comparison is:

| Nsight Systems median | Static rank | Static rank + M=1 | Change |
|:---|---:|---:|---:|
| CPU NVTX range | 75.263 us | 72.218 us | -4.05% |
| Range start to first CUDA launch | 38.433 us | 35.798 us | -6.86% |
| Marlin launch API | 8.155 us | 8.517 us | +4.44% |
| Marlin kernel | 14.752 us | 14.592 us | -1.08% |
| Inter-kernel GPU gap | 1.376 us | 1.312 us | -0.064 us |
| EoRA kernel | 17.024 us | 15.232 us | -10.53% |
| Two-kernel GPU span | 33.184 us | 31.232 us | -5.88% |

Tail-only Nsight gives 16.704 versus 14.880 us, a 10.92% reduction. NCU
reports 640,856 versus 513,382 executed instructions (-19.89%), 37.696 versus
33.216 us replay duration (-11.89%), unchanged 32-register usage, and memory
throughput increasing from 56.47 to 64.06 GB/s. Long-scoreboard and barrier
stalls now account for a larger fraction of each remaining instruction; they
remain the next kernel bottlenecks rather than integer row mapping.

The seventh-pass full-warp up projection directly attacks that scoreboard
bottleneck without splitting a warp's contiguous output tile. Each warp still
loads 32 adjacent FP16/BF16 weights for a given rank, but separate warps own
independent 32-rank slices. The existing 256-float down-phase shared buffer is
dead after the second grid barrier, so it holds the warp partials without new
shared memory or VRAM. The selected points are two warps at rank 64 and four
warps at rank 128.

Matched tail-only Nsight Systems medians are:

| Rank | Sixth pass | Full-warp up | Change |
|---:|---:|---:|---:|
| 64 | 10.416 us | 8.608 us | -17.36% |
| 128 | 14.880 us | 10.400 us | -30.11% |

The final integrated rank-128 capture keeps Marlin and the native launch gap
effectively unchanged while shortening the adapter and complete GPU span:

| Nsight Systems median | Sixth pass | Full-warp up | Change |
|:---|---:|---:|---:|
| Marlin kernel | 14.592 us | 14.656 us | +0.064 us |
| Inter-kernel GPU gap | 1.312 us | 1.312 us | unchanged |
| EoRA kernel | 15.232 us | 10.720 us | -29.62% |
| Two-kernel GPU span | 31.232 us | 26.752 us | -14.34% |

Final Nsight Compute counters explain the gain. Total instructions remain
nearly flat because the same GEMV work plus a small shared reduction is still
performed, but more independent warps can run while loads are outstanding:

| NCU metric | Sixth pass | Full-warp up | Change |
|:---|---:|---:|---:|
| Replay duration | 33.216 us | 23.100 us | -30.45% |
| Executed instructions | 513,382 | 517,337 | +0.77% |
| Issued instructions | 534,865 | 535,702 | +0.16% |
| Memory throughput | 64.06 GB/s | 92.16 GB/s | +43.86% |
| Registers/thread | 32 | 32 | unchanged |
| Achieved occupancy | 8.37% | 12.58% | +50.30% |
| Up-loop long-scoreboard samples | 599 | 50 | -91.65% |

Static shared memory remains 1.02 KiB and NCU reports zero local-memory spill
requests. The unchanged instruction count and sharply lower scoreboard count
confirm latency hiding, rather than skipped arithmetic, as the source of the
speedup.

The eighth-pass rank-128 down reduction removes another synchronization point
from that specialized single-row kernel. The previous mapping gave each rank
two lanes separated by 128 thread indices, stored both partials in shared
memory, synchronized the complete CTA, then loaded and combined them. The new
mapping gives each rank a pair of adjacent lanes. Even and odd lanes each load
a contiguous 16-value rank segment at neighboring input positions, and the
even lane combines the pair with `__shfl_down_sync` before the existing
atomic add. The generic-rank, rank-64, and multi-row implementations are
unchanged.

Matched 200-launch Nsight Systems captures and exact-source Nsight Compute
give:

| Metric | Full-warp up | Adjacent-lane down | Change |
|:---|---:|---:|---:|
| FP16 kernel median, physical GPU 4 | 10.368 us | 10.240 us | -1.23% |
| BF16 kernel median, physical GPU 5 | 10.399 us | 10.208 us | -1.84% |
| NCU replay duration, FP16 | 23.100 us | 21.760 us | -5.80% |
| Executed instructions | 517,337 | 546,116 | +5.56% |
| Issued instructions | 535,702 | 564,977 | +5.46% |
| Memory throughput | 92.16 GB/s | 97.78 GB/s | +6.10% |
| Long-scoreboard cycles per issued instruction | 19.61 | 18.02 | -8.11% |
| Barrier cycles per issued instruction | 12.47 | 8.66 | -30.55% |
| Registers/thread | 32 | 32 | unchanged |
| Achieved occupancy | 12.58% | 12.47% | -0.11 points |

The simpler synchronization wins despite the modest instruction increase.
Static shared memory remains 1.02 KiB, local-memory spill requests remain
zero, and the full-forward CUDA-event medians tie the current tables at their
1.024 us reporting granularity. Both dtypes improve in the direct kernel
capture without an end-to-end or VRAM regression.

The ninth pass targets wide rank-128 single-row projections. Every output CTA
must already launch for the up projection, but only the first 128 CTAs used to
share the down projection. For `N=11008`, 344 output CTAs are available. A
bounded sweep selected 256 participating CTAs:

| Down-phase CTA cap | FP16 kernel median | Change from 128 |
|---:|---:|---:|
| 128 | 13.024 us | baseline |
| 192 | 11.200 us | -14.00% |
| 256 | 10.912 us | -16.21% |
| all 344 | 11.008 us | -15.47% |

The exact final 256-cap source measured 10.944 us over 200 launches, 15.97%
below the exact 128-cap baseline. A matched BF16 replay on physical GPU 5
measured 12.928 to 11.008 us (-14.85%). The 256 cap balances shorter per-CTA
down loops against extra atomic contributors; using every output CTA was
slightly slower.

Matched exact-source NCU reports explain the FP16 gain:

| NCU metric | 128 down CTAs | 256 down CTAs | Change |
|:---|---:|---:|---:|
| Replay duration | 23.17 us | 20.96 us | -9.54% |
| Memory throughput | 168.82 GB/s | 186.60 GB/s | +10.53% |
| Barrier cycles per issued instruction | 29.85 | 20.95 | -29.82% |
| Long-scoreboard cycles per issued instruction | 15.82 | 15.10 | -4.55% |
| Registers/thread | 32 | 32 | unchanged |
| Achieved occupancy | 33.42% | 32.74% | -0.68 points |

The launch grid, four grid-wide phases, 1.02 KiB static shared allocation, and
output work are unchanged. The extra CTAs only divide the existing down GEMV
work more finely, so the pass adds no launch, workspace, persistent state, or
allocator-visible memory.

The tenth pass applies the same parallelism lesson to the opposite projection
shape. The 11008-to-4096 MLP down projection previously launched only 128
32-column CTAs, so the 256-CTA down-phase cap could not be reached. For the
single-row rank-128 specialization only, strongly contracting projections
with `K >= 2N` now use a narrower output tile. A bounded FP16 sweep on physical
GPU 4 selected 16 columns:

| Output columns per CTA | Grid CTAs | FP16 kernel median | Change from 32 |
|---:|---:|---:|---:|
| 32 | 128 | 16.543 us | baseline |
| 24 | 171 | 14.640 us | -11.50% |
| 20 | 205 | 13.600 us | -17.79% |
| 18 | 228 | 13.440 us | -18.76% |
| 16 | 256 | 13.152 us | -20.50% |
| 12 | 342 | 15.232 us | -7.92% |

Sixteen columns produces exactly 256 CTAs, so every launched block
participates in both phases. Wider tiles underfill the selected down cap;
narrower tiles launch blocks that are idle during the down phase and add more
up-reduction work. The exact final BF16 capture on physical GPU 5 improves
from 16.496 to 13.184 us (-20.08%). Matched full-forward source toggles move
FP16 p50 from 50.176 to 46.080 us (-8.16%) and BF16 p50 from 51.200 to
48.128 us (-6.00%).

Exact-source FP16 NCU reports confirm that the additional blocks improve
latency hiding and useful memory parallelism:

| NCU metric | 32-column tile | 16-column tile | Change |
|:---|---:|---:|---:|
| Replay duration | 35.74 us | 25.54 us | -28.54% |
| Waves per SM | 0.13 | 0.26 | +100% |
| DRAM throughput | 109.43 GB/s | 153.16 GB/s | +39.97% |
| Long-scoreboard cycles per issued instruction | 22.18 | 20.14 | -9.20% |
| Barrier cycles per issued instruction | 5.05 | 8.55 | +69.31% |
| Registers/thread | 32 | 32 | unchanged |
| Achieved occupancy | 12.73% | 25.71% | +12.98 points |

The higher barrier share is outweighed by twice the grid-level parallelism.
Static shared memory remains 1.02 KiB, local-memory spill requests remain
zero, and the route adds no allocation. Rank 64, runtime ranks, multi-row
calls, attention, expansion projections, non-Ampere devices, and library
fallbacks retain their existing launch geometry.

## Memory result

For the representative rank-128 tail, allocator-visible transient memory is:

| Route | Per-call peak above steady state |
|:---|---:|
| Library fallback | 512 bytes |
| Cooperative | 0 bytes |

The generic cooperative module owns an FP32 workspace of
`rows * rank * 4` bytes. It starts at one row, or 512 bytes for rank 128, and
grows lazily to the largest eligible row count. The common prepared W4A16
module now owns no EoRA workspace: after the Marlin kernel completes, the
native wrapper passes its transient FP32 `c_tmp` allocation to EoRA on the
same stream. Native allocation expands that tensor when needed so it always
covers the active `rows * rank` floats. This removes 512 persistent bytes per
rank-128 adapted layer at the default initial row count while the
allocator-visible decode full-call peak is now 24 KiB.

Marlin's FP32 reduction scratch previously used the global upper bound
`64 * SMs * max_thread_n`, even for `M <= 16`. The retained allocation now
matches the actual small-batch launch bound:

```text
(M if M <= 8 else round_up(M, 16))
    * min(N, SMs * max_thread_n_for_live_config) * sizeof(float)
```

The live architecture-specific small-batch configuration table determines
`max_thread_n`; larger-M paths retain the original conservative bound. A
forced packed-prefill launch at exactly M=16 reserves all `N` columns because
that route selects a different configuration family. This is transient
allocator memory, not a new persistent buffer. For `M <= 8`, the kernel's
FP32 reduction loop visits four alternating `int4` fragment groups. Packing
those groups first halved their per-slice stride. The final pass maps their
even and odd fragment lanes into actual output rows, so `M < 8` reserves only
the rows that can reach the output. M=1 has a direct mapping that avoids the
generic row-pair branch. Reduction arithmetic and output layout are unchanged.

| Case | Previous full peak | Final full peak | Reduction |
|:---|---:|---:|---:|
| decode attention, M=1 | 1,992 KiB | 24 KiB | 98.80% |
| decode MLP up, M=1 | 2,005.5 KiB | 64.5 KiB | 96.78% |
| decode MLP down, M=1 | 2,760 KiB | 24 KiB | 99.13% |
| attention, M=8 | 2,048 KiB | 192 KiB | 90.62% |
| attention, M=16 | 2,112 KiB | 384 KiB | 81.82% |

## Correctness and safety completed

- Seventy-one focused EoRA tests passed, including the validated and prepared
  operators, FP16, BF16, ranks 37, 64, 128, and 320, expanding
  `K=97, N=130` and strongly contracting `K=257, N=97` shapes, single-row and
  multi-row inputs, 3-D inputs, both requested GPUs, and a non-default stream.
- All 31 Marlin/JIT regression tests passed, including clean FP16 and BF16 JIT
  builds and forwards from the exact final source. The added regression runs
  every `M=1..8` value and compares live-row FP32 reduction against the
  established FP16/BF16 reduction path.
- All comparisons use a dense FP32 LoRA reference; the final benchmark sweep
  had worst maximum absolute error below 0.020.
- Compute Sanitizer memcheck reported `ERROR SUMMARY: 0 errors` for the exact
  final source across all six full-module shapes: FP16 exercised both the
  prepared cooperative and ordinary Marlin routes on physical GPU 4, and
  BF16 exercised prepared cooperative dispatch on physical GPU 5. This covers
  M=1/8/16 and both projection directions. The forced M=16 packed-prefill
  route and earlier standalone bias/activation-order cases are also covered.
- A separate exact-source memcheck ran the new FP16 and BF16 regression for
  every compacted row count `M=1..8` on physical GPU 4 and also reported
  `ERROR SUMMARY: 0 errors`.
- Fresh FP16 and BF16 JIT rebuilds for `sm_80` compiled and executed
  successfully from the final source on physical GPU 4; the BF16 binary was
  additionally exercised by the final benchmark and sanitizer on physical
  GPU 5. Forced M=16 packed prefill measured 80.90 us FP16 and 72.70 us BF16
  p50, with 384 KiB peak allocation in both cases.
- A pre-warmed CUDA graph capture/replay test confirmed that the prepared
  integrated operator receives zero calls during capture and replay is
  bitwise identical to the library fallback.
- Compute Sanitizer also exercised the final cooperative kernel for both
  dtypes, prepared and validated entry points, ranks 37, 64, 128, and 320,
  rows 1 and 3, and both non-divisible projection directions: all 64
  parameterizations passed with
  `ERROR SUMMARY: 0 errors`.
- Real full-module 1-D, 2-D, and 3-D prepared dispatches retained their input
  prefixes, output dtype, nonzero adapter updates, and dense-FP32-reference
  error below 0.0064.
- A real 3-D group-size-64 module selected the validated generic integrated
  operator, retained its 512-byte workspace, applied a nonzero adapter update,
  and matched the FP32 reference with 0.00680 maximum absolute error.
- Ruff and `git diff --check` passed.
- A direct same-module comparison proved that both the integrated and library
  routes apply a nonzero LoRA update and match the dense FP32 reference. The
  benchmark now populates path-less merged-weight buffers before module
  post-initialization, matching the real checkpoint-loading lifecycle.
- A broader related test selection passed 31 of 32 tests. The unrelated
  failure is a stale adapter-config assertion expecting four serialized fields
  while the current `Lora.to_dict()` returns nine.

## Artifacts

Profiler reports and benchmark JSON are local, untracked artifacts under
`artifacts/eora_marlin_20260721/`:

```text
baseline_marlin_addmm_decode_r128_nsys.nsys-rep
baseline_down_gemv_decode_r128_ncu.ncu-rep
final_block_reduce_marlin_cooperative_decode_r128_nsys.nsys-rep
final_block_reduce_fused_decode_r128_ncu.ncu-rep
final_block_reduce_marlin_sweep_fp16_gpu4.json
final_block_reduce_marlin_sweep_bf16_gpu5.json
final_block_reduce_tail_memory_fp16_gpu4.json
final_exact_ctmp_marlin_cooperative_decode_r128_nsys.nsys-rep
noop_reshape_marlin_cooperative_decode_r128_nsys.nsys-rep
final_round2_prepared_decode_r128_nsys.nsys-rep
final_round2_all_fp16_gpu4.json
final_round2_all_bf16_gpu5.json
final_round2_3d_decode_r128_fp16_gpu4.json
forced_packed_m16_scratch_fp16_gpu4.json
forced_packed_m16_scratch_bf16_gpu5.json
round3_baseline_decode_r128_nsys.nsys-rep
integrated_dispatch_decode_r128_nsys.nsys-rep
integrated_dispatch_corrected_all_fp16_gpu4.json
integrated_dispatch_corrected_all_bf16_gpu5.json
integrated_dispatch_3d_decode_r128_fp16_gpu4.json
integrated_dispatch_forced_packed_m16_fp16_gpu4.json
integrated_dispatch_forced_packed_m16_bf16_gpu5.json
integrated_dispatch_memcheck_fp16_gpu4.json
round4_baseline_native_dispatch_decode_r128_nsys.nsys-rep
round4_baseline_native_dispatch_torch_trace.json
round4_final_all_fp16_gpu4.json
round4_final_all_bf16_gpu5.json
round4_release_all_fp16_gpu4.json
round4_release_all_bf16_gpu5.json
round4_release_3d_decode_r128_fp16_gpu4.json
round4_reuse_ctmp_3d_decode_r128_fp16_gpu4.json
round4_final_packed_prefill16_fp16_gpu4.json
round4_final_packed_prefill16_bf16_gpu5.json
round4_final_prepared_memcheck_gpu4.log
round4_final_prepared_packed_memcheck_gpu4.log
round4_final_narrow_scratch_floor_memcheck_gpu4_v2.log
round4_final_prepared_native_dispatch_decode_r128_nsys_v2.nsys-rep
round4_release_prepared_native_dispatch_decode_r128_nsys_v3.nsys-rep
round4_release_prepared_memcheck_gpu4.log
round4_release_prepared_packed_memcheck_gpu4.log
round6_baseline_all_fp16_gpu4_torch213.json
round6_static_rank_candidate_repeat_all_fp16_gpu4_torch213.json
round6_baseline_all_bf16_gpu5_torch213.json
round6_static_rank_candidate_all_bf16_gpu5_torch213.json
round6_static_rank_candidate_3d_fp16_gpu4_torch213.json
round6_baseline_tail_nsys_torch213.nsys-rep
round6_static_rank_candidate_tail_nsys_torch213.nsys-rep
round6_baseline_full_nsys_torch213.nsys-rep
round6_static_rank_candidate_full_nsys_torch213.nsys-rep
round7_runtime_rank_r128_fp16_gpu4_torch213_ncu.ncu-rep
round7_static_rank_r128_fp16_gpu4_ncu.ncu-rep
round7_static_single_row_r128_fp16_gpu4_ncu.ncu-rep
round7_static_single_row_candidate_tail_nsys_torch213.nsys-rep
round7_static_single_row_candidate_full_nsys_torch213.nsys-rep
round7_static_single_row_candidate_repeat_all_fp16_gpu4_torch213.json
round7_static_single_row_candidate_repeat_all_bf16_gpu5_torch213.json
round7_static_single_row_candidate_3d_fp16_gpu4_torch213.json
round7_static_single_row_memcheck_gpu4.log
round8_warp_down_cache_candidate_tail_nsys_torch213.nsys-rep
round8_shared_x_cache_candidate_tail_nsys_torch213.nsys-rep
round8_fullwarp_split_candidate_tail_nsys_torch213.nsys-rep
round8_four_fullwarps_candidate_tail_nsys_torch213.nsys-rep
round8_eight_fullwarps_candidate_tail_nsys_torch213.nsys-rep
round8_rank64_head_tail_nsys_torch213.nsys-rep
round8_rank64_two_fullwarps_candidate_tail_nsys_torch213.nsys-rep
round8_rank64_four_fullwarps_candidate_tail_nsys_torch213.nsys-rep
round8_rank64_head_bf16_gpu5_tail_nsys_torch213.nsys-rep
round8_rank64_two_fullwarps_bf16_gpu5_tail_nsys_torch213.nsys-rep
round8_fullwarp_final_full_nsys_torch213.nsys-rep
round8_fullwarp_final_r128_fp16_gpu4_ncu.ncu-rep
round8_fullwarp_final_all_fp16_gpu4_torch213.json
round8_fullwarp_final_all_bf16_gpu5_torch213.json
round8_fullwarp_final_3d_fp16_gpu4_torch213.json
round8_fullwarp_final_memcheck_gpu4.log
round8_fullwarp_final_integrated_memcheck_gpu4.log
round8_fullwarp_final_integrated_memcheck_gpu4.json
round9_fullwarp_head_tail_200_nsys_torch213.nsys-rep
round9_rank128_down_pair_shfl_down_candidate_tail_200_nsys_torch213.nsys-rep
round9_fullwarp_head_bf16_gpu5_tail_200_nsys_torch213.nsys-rep
round9_down_pair_shuffle_final_bf16_gpu5_tail_200_nsys_torch213.nsys-rep
round9_down_pair_shuffle_final_r128_fp16_gpu4_ncu.ncu-rep
round9_down_pair_shuffle_candidate_r128_fp16_gpu4_torch213.json
round9_down_pair_shuffle_candidate_r128_bf16_gpu5_torch213.json
round9_down_pair_shuffle_final_memcheck_gpu4.log
round9_down_pair_shuffle_final_integrated_memcheck_gpu4.log
round9_down_pair_shuffle_final_integrated_memcheck_gpu4.json
round10_warp_x_broadcast_candidate_tail_200_nsys_torch213.nsys-rep
round10_named_up_barrier_candidate_r128_tail_200_nsys_torch213.nsys-rep
round10_named_up_barrier_candidate_r64_tail_200_nsys_torch213.nsys-rep
round10_native_cache_baseline_full_200_nsys_torch213.nsys-rep
round10_thread_local_cache_candidate_full_200_nsys_torch213.nsys-rep
round10_cache_switch_baseline_a_r128_fp16_gpu4_torch213.json
round10_cache_switch_candidate_a_r128_fp16_gpu4_torch213.json
round10_cache_switch_baseline_b_r128_fp16_gpu4_torch213.json
round10_cache_switch_candidate_b_r128_fp16_gpu4_torch213.json
round10_mlp_up_down128_baseline_tail_200_nsys_torch213.nsys-rep
round10_mlp_up_down192_candidate_tail_200_nsys_torch213.nsys-rep
round10_mlp_up_down256_candidate_tail_200_nsys_torch213.nsys-rep
round10_mlp_up_down384_candidate_tail_200_nsys_torch213.nsys-rep
round10_wide_rank128_down256_final_tail_200_nsys_torch213.nsys-rep
round10_wide_rank128_down128_baseline_mlp_up_bf16_gpu5_tail_200_nsys_torch213.nsys-rep
round10_wide_rank128_down256_final_mlp_up_bf16_gpu5_tail_200_nsys_torch213.nsys-rep
round10_wide_rank128_down128_baseline_mlp_up_fp16_gpu4_ncu.ncu-rep
round10_wide_rank128_down256_final_mlp_up_fp16_gpu4_ncu.ncu-rep
round10_wide_rank128_full_baseline_a_mlp_up_fp16_gpu4_torch213.json
round10_wide_rank128_full_candidate_a_mlp_up_fp16_gpu4_torch213.json
round10_wide_rank128_full_baseline_b_mlp_up_fp16_gpu4_torch213.json
round10_wide_rank128_full_candidate_b_mlp_up_fp16_gpu4_torch213.json
round10_wide_rank128_full_baseline_mlp_up_bf16_gpu5_torch213.json
round10_wide_rank128_full_candidate_mlp_up_bf16_gpu5_torch213.json
round10_wide_rank128_down256_final_all_fp16_gpu4_torch213.json
round10_wide_rank128_down256_final_all_bf16_gpu5_torch213.json
round10_wide_rank128_down256_final_memcheck_gpu4.log
round10_wide_rank128_down256_final_integrated_memcheck_gpu4.log
round10_wide_rank128_down256_final_integrated_memcheck_gpu4.json
round11_pairrank_128thread_candidate_mlp_up_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_up_half_atomic_candidate_mlp_up_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_static_mlp_up_shape_candidate_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_mlp_down_tile32_baseline_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_mlp_down_tile24_candidate_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_mlp_down_tile20_candidate_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_mlp_down_tile18_candidate_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_mlp_down_tile16_candidate_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_mlp_down_tile12_candidate_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_mlp_down_tile32_baseline_bf16_gpu5_tail_200_nsys_torch213.nsys-rep
round11_narrow_rank128_tile16_final_mlp_down_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round11_narrow_rank128_tile16_final_mlp_down_bf16_gpu5_tail_200_nsys_torch213.nsys-rep
round11_mlp_down_tile32_baseline_fp16_gpu4_ncu.ncu-rep
round11_narrow_rank128_tile16_final_mlp_down_fp16_gpu4_ncu.ncu-rep
round11_narrow_rank128_full_baseline_a_mlp_down_fp16_gpu4_torch213.json
round11_narrow_rank128_full_candidate_a_mlp_down_fp16_gpu4_torch213.json
round11_narrow_rank128_full_candidate_b_mlp_down_fp16_gpu4_torch213.json
round11_narrow_rank128_full_baseline_mlp_down_bf16_gpu5_torch213.json
round11_narrow_rank128_full_candidate_mlp_down_bf16_gpu5_torch213.json
round11_narrow_rank128_tile16_final_all_fp16_gpu4_torch213.json
round11_narrow_rank128_tile16_final_all_bf16_gpu5_torch213.json
round11_narrow_rank128_tile16_final_memcheck_gpu4.log
round11_narrow_rank128_tile16_final_integrated_memcheck_gpu4.log
round11_narrow_rank128_tile16_final_integrated_memcheck_gpu4.json
round12_current_head_full_attn_r128_fp16_gpu4_50_nsys_torch213.nsys-rep
round12_current_head_prepared_dispatch_fp16_gpu4_torch_trace.json
round12_multirow_pair_baseline_m8_r128_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round12_multirow_pair_candidate_m8_r128_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round12_rank64_pair2_baseline_attn_fp16_gpu4_ncu.ncu-rep
round12_rank64_pair2_candidate_attn_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round12_rank64_ilp4_candidate_attn_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round12_attn_tile32_baseline_v3_r128_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round12_attn_tile16_candidate_r128_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round12_attn_tile24_candidate_r128_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round12_packed_m8_scratch_baseline_toggle_all_fp16_gpu4_torch213.json
round12_packed_m8_scratch_candidate_toggle_all_fp16_gpu4_torch213.json
round12_packed_m8_scratch_baseline_toggle_all_bf16_gpu5_torch213.json
round12_packed_m8_scratch_candidate_toggle_all_bf16_gpu5_torch213.json
round12_packed_m8_scratch_baseline_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round12_packed_m8_scratch_candidate_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round12_packed_m8_scratch_final_all_fp16_gpu4_torch213.json
round12_packed_m8_scratch_final_all_bf16_gpu5_torch213.json
round12_packed_m8_scratch_final_all_paths_fp16_gpu4_memcheck.log
round12_packed_m8_scratch_final_integrated_bf16_gpu5_memcheck.log
round13_attn_tile30_candidate_r128_fp16_gpu4_tail_200_nsys_torch213.nsys-rep
round13_live_row_scratch_candidate_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round13_live_row_scratch_candidate_repeat_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round13_live_row_m1fast_candidate_repeat_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round13_live_row_scratch_final_3000_all_fp16_gpu4_torch213.json
round13_live_row_scratch_final_3000_all_bf16_gpu5_torch213.json
round13_live_row_scratch_final_all_paths_fp16_gpu4_memcheck.log
round13_live_row_scratch_final_integrated_bf16_gpu5_memcheck.log
round13_live_row_scratch_final_rows1_8_gpu4_memcheck.log
round14_current_head_attn_r128_bf16_gpu5_ncu.ncu-rep
round14_attn_down160_candidate_gil_full_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round14_attn_down192_candidate_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round14_attn_down224_candidate_gil_full_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round14_attn_down256_candidate_attn_r128_bf16_gpu5_nsys_torch213.nsys-rep
round14_attn_down192_final_attn_r128_bf16_gpu5_ncu.ncu-rep
round14_attn_down128_control_gil_full_attn_r128_fp16_gpu4_nsys_torch213.nsys-rep
round14_attn_down192_final_gil_full_attn_r128_fp16_gpu4_nsys_torch213.nsys-rep
round14_down192_final_sequential_all_fp16_gpu4_torch213.json
round14_down192_final_sequential_all_bf16_gpu5_torch213.json
round14_down192_final_integrated_memcheck_fp16_gpu4.log
round14_down192_final_integrated_memcheck_bf16_gpu5.log
```

## Continued experiment log

Each candidate includes a matched correctness check and synchronized
full-Marlin latency comparison.
Candidates that only improve profiler-instrumented kernel duration, regress
the production forward path, or trade correctness for speed will be rejected.

| Status | Candidate | Rationale | Result |
|:---|:---|:---|:---|
| retained | Cache cooperative launch limits per live device and dtype | Avoid repeated runtime occupancy/property queries | FP16 rank-128 p50 110.59 -> 107.52 us; repeated queries removed |
| retained | Cache Marlin dynamic shared-memory attributes | Remove a repeated CUDA runtime call | `cudaFuncSetAttribute` disappeared from the warmed profile; p50 about 107.52 -> 105.47 us |
| retained | Exact small-M Marlin FP32 scratch bound | Reduce transient VRAM | representative peak 1,992 -> 264 KiB; sanitizer clean |
| retained | Guarded null pointers for absent Marlin operands | Remove zero-length tensor allocations | five -> two `aten::empty` calls; cooperative p50 106.50 -> 104.45 us |
| retained | Skip no-op and duplicate reshapes/views | Close the inter-kernel host gap | gap 26.896 -> 11.472 us; 3-D p50 98.304 -> 93.184 us |
| retained | Prepared internal EoRA entry point | Avoid repeating post-init invariants | about 1.0 us matched; prior-pass Nsight gap 10.640 us |
| retained | One native Marlin+EoRA dispatch | Remove the remaining Python/operator launch gap | gap 10.848 -> 1.280 us; FP16 rank-128 p50 81.92 -> 47.10 us |
| retained | Prepared common W4A16 native dispatch | Remove immutable Marlin validation, optional operands, and boxed arguments | NVTX median 85.083 -> 69.463 us; FP16 rank-128 p50 47.10 -> 37.89 us |
| retained | Native 1-D/2-D/3-D prepared shapes | Avoid Python flatten/restore dispatch | transformer-shaped p50 61.44 -> 41.98 us |
| retained | Reuse Marlin `c_tmp` for EoRA | Remove redundant per-layer adapter scratch | rank-128 persistent workspace 512 -> 0 bytes on the common path |
| retained | Populate benchmark merged-LoRA buffers | Exercise the real checkpoint lifecycle and prevent a zero-adapter reference | both routes now prove a nonzero update; current tables regenerated |
| retained | Compile-time rank 64/128 EoRA kernels | Remove runtime division, remainder, and address math in the common loops | tail kernel 18.128 -> 16.704 us; all 12 FP16/BF16 full-forward cases improve |
| retained | Compile-time single-row rank 64/128 kernels | Eliminate invariant row division and workspace bounds in decode | tail kernel 16.704 -> 14.880 us; M=1 full-forward improves or ties with unchanged VRAM |
| retained | Full-warp 32-rank up-projection slices | Hide long-scoreboard latency while preserving fully coalesced weight loads | rank 128 kernel 14.880 -> 10.400 us; rank 64 10.416 -> 8.608 us; no extra memory |
| retained | Adjacent-lane rank-128 down reduction | Replace the generic shared-memory reduction and CTA barrier with a pair shuffle | FP16 10.368 -> 10.240 us; BF16 10.399 -> 10.208 us; no extra memory |
| retained | Up to 256 rank-128 M=1 down CTAs on wide outputs | Divide the down GEMV across more already-launched output CTAs | MLP-up FP16 13.024 -> 10.944 us; BF16 12.928 -> 11.008 us; no extra launch or memory |
| retained | 16-column tiles for strong rank-128 M=1 contractions | Launch enough output CTAs to reach the selected 256-CTA down cap | MLP-down FP16 16.543 -> 13.152 us; BF16 16.496 -> 13.184 us; no extra memory |
| retained | Pack M<=8 Marlin FP32 reduction scratch | Remove unused padded fragment rows | M=1 peaks 264/709.5 -> 136/365.5 KiB; M=8 320 -> 192 KiB; Marlin median 14.976 -> 14.912 us |
| retained | Pack Marlin FP32 scratch by live row for M<8 | Remove scratch for fragment rows that cannot reach the output | M=1 peaks 136/365.5 -> 24/64.5 KiB; Marlin median 14.912 -> 14.848 us; M>=8 unchanged |
| rejected | Four independent up-projection accumulator chains | Address long-scoreboard stalls | kernel median regressed 18.46 -> 18.72 us; relative full-forward gain also slipped |
| rejected | Two-lane-per-output up reduction | Shorten serial rank dependency chains | kernel median regressed 18.50 -> 22.50 us |
| rejected | Warp-shuffle cache of the up-projection down row | Remove redundant reads of the 128-element intermediate | kernel median regressed 14.880 -> 40.160 us |
| rejected | Warp-broadcast each down input value | Reduce duplicate input loads across rank pairs | uniform-tail implementation was correct but regressed 10.240 -> 14.048 us |
| rejected | Counted named barrier for up warps | Avoid waiting on inactive down-only warps during the up reduction | rank 128 tied at 10.240 us and rank 64 regressed 8.608 -> 8.640 us |
| rejected | Thread-local native metadata-cache mirrors | Remove four warmed mutex-protected cache reads per prepared forward | same-binary source toggles were inconsistent below timer/host noise; GPU work was unchanged |
| rejected | Shared input cache in the down projection | Remove repeated global input loads across ranks | kernel median regressed 14.880 -> 15.424 us |
| rejected | Eight up-projection warps at rank 128 | Increase latency-hiding warps beyond the selected four | kernel median regressed 10.400 -> 10.496 us |
| rejected | Four up-projection warps at rank 64 | Increase latency-hiding warps beyond the selected two | kernel median regressed 8.608 -> 8.640 us |
| rejected | 128-thread adjacent-rank block | Give each rank pair a smaller CTA while preserving vectorized down loads | MLP-up kernel regressed 10.944 -> 12.896 us |
| rejected | FP16/BF16 atomics directly into output | Eliminate the shared up-partial reduction | correct but less accurate; MLP-up kernel regressed 10.944 -> 15.392 us |
| rejected | Static 4096-to-11008 template | Fold the common MLP-up dimensions at compile time | median changed 10.944 -> 10.880 us while means tied; binary expansion was unjustified |
| rejected | 128-thread cooperative EoRA block | Reduce per-CTA resource use and expose more blocks | kernel median regressed 18.272 -> 21.696 us; tail p50 57.34 -> 59.39 us |
| rejected | 64 down-projection CTAs instead of 128 | Reduce contention and excess CTAs | tail p50 57.34 -> 60.42 us; kernel about 18.3 -> 23.8 us |
| rejected | Block-partial workspace instead of down atomics | Remove global atomic accumulation | correct with no persistent VRAM, but full-forward p50 37.89 -> 48.13 us |
| rejected | Direct prepared M=1 Marlin launcher | Bypass generic native selection | CPU range 69.463 -> 71.937 us; GPU span unchanged |
| rejected | Six-argument/default-schema prepared call | Reduce boxed arguments | p50 remained 36.864 us |
| rejected | Split prepared Python branch | Avoid conditional work on the hot route | steady p50 and mean did not improve |
| rejected | Paired-rank vector loads and half atomics | Increase load width and reduce FP32 atomic traffic | correct, but matched Torch 2.13 tail p50 58.37 -> 60.42 us |
| rejected | Private direct CUDA allocation path | Reduce output-allocation dispatcher overhead | isolated operator CPU fell 28.899 -> 22.925 us, but full range tied at 69.463/69.403 us |
| rejected | Runtime `rows == 1` branch in one kernel | Skip row division without more instantiations | kernel improved only 16.704 -> 16.544 us, while decode full-forward regressed about 2.05 us and M=8 was perturbed |
| rejected | Extend adjacent-pair rank-128 down mapping to multi-row | Remove the generic multi-row block reduction | M=8 FP16 regressed 41.344 -> 42.560 us; M=16 BF16 regressed 55.760 -> 61.439 us |
| rejected | Adjacent-pair rank-64 down mapping | Remove rank-64 block barrier | FP16 regressed 11.872 -> 12.576 us despite a BF16 improvement |
| rejected | Four rank-64 down accumulator chains | Increase independent memory work per lane | FP16 regressed 11.872 -> 12.959 us |
| rejected | Prepared-Python early shortcut | Reduce wrapper branching before native dispatch | repeated source toggles were inconsistent and showed no stable gain |
| rejected | Narrow attention output tiles | Add CTAs to the 4096-to-4096 rank-128 launch | 32/30/16/24-column medians were 13.408/13.552/13.472/14.176 us |
| rejected | Cooperative execution during CUDA graph capture | Replace the library graph tail after eager-kernel tuning | M=1 tied or regressed; M=8/M=16 regressed 27.648/29.696 -> 53.248/69.632 us |
| retained | Per-module generic EoRA workspace | Preserve stream safety outside the prepared path | generic state keeps 512 bytes at rank 128/M=1; prepared calls reuse their own same-stream `c_tmp` |

### 2026-07-21: cache cooperative launch limits

A matched Torch operator profile showed that every fused call repeated two
`cudaDeviceGetAttribute` calls and one occupancy calculation. Including the
associated runtime work, these consumed roughly 5 us of CPU time per layer.
The maximum cooperative grid is now queried once and cached by the process for
the actual logical CUDA device and scalar type. The runtime device guard is
entered before the first query, and no physical index or SM count is assumed.

After a clean JIT rebuild, a 1,000-iteration FP16 run on physical GPU 4 gave:

```text
Route          Before cache p50   After cache p50
-------------  -----------------  ---------------
fallback                160.77 us        160.77 us
cooperative             110.59 us        107.52 us
```

The post-change operator profile no longer contained the per-call occupancy
or device-attribute rows. The result is retained.

### 2026-07-21: reject four-way up-loop ILP

Four independent FP32 accumulators were tested in the up-projection rank loop
to hide long-scoreboard latency. The absolute full-forward p50 appeared lower,
but the fallback in the same run sped up by a similar amount: relative
cooperative improvement slipped from 33.1% with cached launch limits to 32.7%.
The matched Nsight Systems kernel median also increased from 18.46 to 18.72 us.
The accumulator experiment was reverted.

### 2026-07-21: cache Marlin launch attributes

The warmed Marlin operator profile contained one
`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` call per
forward. The requested limit is now cached by the actual logical device and
selected kernel pointer, and is updated only if a later launch requests a
larger value. The profile row disappeared after warmup. A representative
matched run moved cooperative p50 from about 107.52 to 105.47 us; the change is
retained for both ordinary and packed-prefill launches.

### 2026-07-21: exact small-M FP32 reduction scratch

The original Marlin wrapper allocated for the largest global `thread_n` and
all resident SMs even when the selected small-batch kernel could address fewer
columns. The new bound scans only the live architecture's eligible small-M
configuration family and caps resident columns at `N`. For M=1, N=4096 this
reduces the FP32 scratch itself to 256 KiB and the measured full-call peak to
264 KiB. N=11008 measures 709.5 KiB including output storage. The forced
packed-prefill M=16 boundary conservatively reserves all columns because its
kernel selection differs from the ordinary small-M family; final clean FP16
and BF16 builds both measured 384 KiB and passed the dense-reference check.

Compute Sanitizer covered M=1/8/16, K/N of 4096 and 11008 in both projection
directions, the forced M=16 packed-prefill route, plus the unchanged M>16
activation-order path, with zero errors. The larger-M allocation formula
remains unchanged.

### 2026-07-21: remove empty operand placeholders

The common symmetric Marlin path created zero-length CUDA tensors for absent
bias, global scale, and activation-order scratch. These were used only to
obtain pointers ignored behind existing feature flags. They are now undefined
tensors with guarded null pointers; real bias, activation-order, zero-point,
and global-scale operands keep their original validation and allocation.

The Torch operator trace fell from five to two `aten::empty` calls per forward.
Marlin operator CPU time in the instrumented 100-call trace fell from 75.818 to
59.598 us/call. Bias and activation-order sanitizer cases passed with zero
errors.

### 2026-07-21: remove redundant reshape dispatch

Five `reshape/view` pairs were emitted per layer even when input and output
were already 2-D. The fast path now passes 2-D tensors through directly and
retains the original flatten/restore behavior for N-D inputs. Since the native
EoRA operator mutates the supplied output, its returned 2-D alias no longer
needs to be reshaped again.

For N-D inputs, `MarlinLinear.forward` now flattens once before the base and
adapter paths and restores shape once at return. Previously the base path
flattened and restored before EoRA immediately flattened both tensors again.

The 100-call trace went from 500 reshapes plus 500 views to zero for the 2-D
decode case. Bounded Nsight captures measured the Marlin-to-EoRA median idle
gap at 26.896 us before and 11.472 us after. Full 2-D and 3-D module outputs
were bitwise identical when fed identical flattened data. The additional
single-flatten N-D change reduced `[1, 1, 4096]` cooperative p50 from 98.304 to
93.184 us in the matched low-overhead harness; that pass's CLI reported
96.26 us. 1-D, 2-D, and 3-D outputs were bitwise identical.

### 2026-07-21: prepared internal operator

Module post-initialization already fixes LoRA shapes, dtype, device,
contiguity, and workspace ownership. Production now caches an internal
prepared operator that omits those repeated native checks. The original
validated operator remains the public path used by direct callers. Alternating
same-process runs measured 78.848 versus 77.824 us and 76.800 versus 75.776 us
for validated versus prepared calls, respectively. Dedicated tests cover both
entry points. That pass's bounded Nsight capture measured a 10.640 us median
Marlin-to-EoRA gap and a 44.064 us two-kernel GPU span.

### 2026-07-21: reject two-lane up reduction

A half-warp-coalesced experiment assigned two lanes to each output column and
combined their rank partials with a warp shuffle. It halved each thread's
serial rank loop but split memory transactions and added reduction work.
Nsight Systems showed the fused-kernel median worsening from about 18.50 to
22.50 us. The scalar, fully coalesced column loop was restored.

### 2026-07-21: one native Marlin+EoRA dispatch

The remaining eager bottleneck was no longer either GPU kernel. A fresh
50-call exact-revision trace measured a 10.848 us median idle gap between the
Marlin and prepared EoRA launches. The dtype-specific Marlin JIT extensions
now link the existing EoRA CUDA source and expose an integrated prepared
operator. Marlin and EoRA are still separate kernels with the same stream and
numerical contracts, but native code enqueues them back-to-back without a
second Python/Torch dispatcher transition.

For `M=1, K=N=4096, rank=128, FP16`, the median inter-kernel gap falls to
1.280 us and GPU span falls from 44.384 to 34.272 us. The synchronized
full-forward p50 falls from the round-three 81.92 us baseline to 47.10 us.
CUDA graph capture still selects the lower-work library route before entering
the integrated operator. Unsupported devices, dtypes, shapes, ranks, and
compressed adapters retain the same fallback.

### 2026-07-21: correct merged-LoRA benchmark lifecycle

The integrated-dispatch audit explicitly compared the adapter update against
the base-only output and exposed a benchmark setup flaw. A path-less `Lora`
uses the repository's merged-weight lifecycle: `MarlinLinear` deep-copies the
adapter, registers module-owned A/B buffers, and expects checkpoint loading to
populate those buffers before `post_init()`. The synthetic benchmark had left
them zero and later replaced only the old cooperative state with the original
standalone tensors. This preserved launch shapes and timing, but meant the
library comparison did not prove a nonzero adapter update.

The builder now copies synthetic A/B weights into the registered module
buffers before post-initialization, exactly as checkpoint loading does. A
same-module check measures a 0.01172 maximum LoRA update for both routes; the
integrated and library outputs are not bitwise equal, as expected from their
different reduction order, and both have 0.00414 maximum absolute error
against the FP32 reference. The harness now also fails if a nonzero reference
adapter produces a zero measured update. Every current performance table was
regenerated with the corrected lifecycle.

### 2026-07-21: prepared common W4A16 native dispatch

After the two kernel launches were already adjacent, Torch and Nsight traces
showed that immutable Marlin argument construction and validation had become
the main remaining eager cost. Module post-initialization already proves the
common GPTQ conditions: `uint4b8`, group size 128, symmetric zero-point-free
weights with at least two groups, no activation order or bias, full K, and
FP32 reduction. A new
dtype-specific prepared entry point now encodes those invariants instead of
receiving the generic operator's 23 boxed arguments on every token. It derives
M, K, N, group count, device properties, and output shape directly from eight
prepared inputs. All other formats and configurations continue to use the
validated integrated operator.

The Python forward path checks the cached state before doing general dtype,
flattening, packed-routing, and output-reshape setup. Contiguous 1-D, 2-D, and
3-D tensors are passed directly to native code, which allocates the output in
the corresponding prefix shape and treats the data as flattened rows only
inside the kernels. Non-contiguous and otherwise ineligible inputs fall
through to the existing general path.

Against the exact `e9391ef7` baseline, the final 49-call Nsight median drops
from 85.083 to 69.463 us and host lead to the first CUDA launch drops from
46.158 to 34.847 us. Marlin remains 14.6-14.8 us, EoRA remains 18.3 us, and
the two-kernel GPU span remains 34.4 us. The 3,000-sample production FP16
rank-128 p50 drops from 47.10 to 37.89 us; BF16 also reaches 37.89 us. A real
transformer-shaped `[1, 1, 4096]` call drops from the prior integrated
61.44 us to 41.98 us.

### 2026-07-21: reuse Marlin reduction scratch for EoRA

Marlin's `c_tmp` is live until the native wrapper returns and is no longer
needed after the Marlin launch completes. The prepared wrapper now passes that
same FP32 tensor to the EoRA launch on the current stream. Stream ordering
guarantees that EoRA observes completed Marlin writes before it zeros and uses
the prefix as `[rows, rank]` workspace. Native allocation takes the maximum of
Marlin's required scratch and the active EoRA `rows * rank` elements, so
environment overrides and uncommon narrow outputs cannot under-allocate it.

A targeted `M=16, K=256, N=64, rank=128` stress case requires 2,048 EoRA
workspace floats while the ordinary Marlin formula needs only 1,024. The
prepared allocation expanded to the larger bound, matched the FP32 reference
within 0.000964, and passed Compute Sanitizer with zero errors. Modules with a
single quantization group remain on the validated generic operator because
Marlin canonicalizes that scale mode to channelwise (`group_size=-1`).

This removes the prepared module's persistent EoRA allocation, from 512 bytes
to zero at rank 128 and the default initial row count. The full-call peak is
unchanged because the much larger Marlin output and `c_tmp` allocations
already dominate. Real 1-D/2-D/3-D correctness checks, default and forced
packed-prefill runs, clean FP16/BF16 JIT builds, CUDA graph bypass, and
Compute Sanitizer all pass with the shared transient scratch.

### 2026-07-21: reject a 128-thread cooperative block

The cooperative EoRA block size was reduced from 256 to 128 threads for ranks
up to 128 to test whether lower per-CTA resource use and additional scheduling
freedom helped the small decode grid. It did not: the matched standalone tail
p50 regressed from 57.34 to 59.39 us, and the bounded Nsight kernel median
regressed from 18.272 to 21.696 us. The 256-thread launch was fully restored;
no part of this candidate is retained.

### 2026-07-21: Torch 2.13 rebaseline and rejected follow-ups

The environment update required rebuilding both Marlin dtype extensions and
the standalone EoRA extension. The exact `cb1276be` source was restored and
measured first, then each candidate was built from the same Torch 2.13 base.
This avoided attributing runtime/compiler changes to a CUDA source change.

The next experiments targeted the two remaining areas visible in Nsight:
down-projection atomics and prepared-dispatch host cost. Reducing down CTAs
from 128 to 64 regressed the tail p50 from 57.34 to 60.42 us and the kernel
from about 18.3 to 23.8 us. Replacing atomics with a block-partial workspace
was correct and did not add persistent module memory, but regressed full
forward from 37.89 to 48.13 us. Paired-rank vector loads with half-precision
atomics also regressed the matched Torch 2.13 tail from 58.37 to 60.42 us.

On the host side, a dedicated prepared M=1 Marlin launch path increased the
instrumented CPU range from 69.463 to 71.937 us without changing GPU span.
Splitting the Python prepared branch and reducing boxed arguments through
schema defaults produced no synchronized gain. Calling a private direct CUDA
allocator lowered an isolated operator's CPU time from 28.899 to 22.925 us,
but the complete range was effectively tied at 69.463 versus 69.403 us and the
private API was not suitable for production. A direct pybind-only dispatcher
was also abandoned after module initialization proved incompatible with the
free-threaded Python 3.14 runtime. Every one of these changes was reverted.

### 2026-07-21: specialize common adapter ranks

Nsight Compute had identified long dependency chains and integer address
work in the fused tail. Ranks 64 and 128 dominate the target adapters and are
fixed at module post-initialization, so the CUDA dispatcher now selects a
compile-time specialization for those two ranks and retains the generic
runtime-rank kernel for all other values. The compiler can eliminate runtime
rank division and remainder in the lane mapping, fold rank strides into
addressing, and use a fixed up-projection loop bound. Device, dtype, shape,
cooperative-capacity, and CUDA-graph gates remain runtime-probed and unchanged.

Matched Torch 2.13 production p50 changes are:

| Case | FP16 before | FP16 after | BF16 before | BF16 after |
|:---|---:|---:|---:|---:|
| decode attention, rank 64 | 36.86 us | 35.84 us | 36.86 us | 35.84 us |
| decode attention, rank 128 | 37.89 us | 35.84 us | 37.89 us | 36.86 us |
| decode MLP up, rank 128 | 44.03 us | 43.01 us | 46.08 us | 45.06 us |
| decode MLP down, rank 128 | 51.20 us | 49.15 us | 51.20 us | 50.18 us |
| batch 8 attention, rank 128 | 55.30 us | 53.25 us | 56.32 us | 53.25 us |
| rows 16 attention, rank 128 | 72.70 us | 69.63 us | 72.70 us | 68.61 us |

The final FP16 `[1, 1, 4096]` rank-128 case is 40.96 us. Peak allocation is
unchanged at 264 KiB for decode attention/down, 709.5 KiB for MLP up, 320 KiB
for M=8, and 384 KiB for M=16. Dense-FP32-reference maximum absolute error is
unchanged, with the worst measured case below 0.020. Generic ranks 37 and 320,
both common ranks, both dtypes, both entry points, CUDA graph fallback, and
sanitizer coverage all pass.

### 2026-07-21: matched NCU after publishing rank specialization

After commit `1dffefeb` was pushed and PR #25 updated, a fresh full Nsight
Compute capture profiled the exact published rank-128 specialization. A
second report loaded the retained Torch 2.13 runtime-rank binary directly, so
both sides use the same interpreter, CUDA runtime, driver, device, launch, and
50-pass NCU collection.

| NCU metric | Runtime rank | Static rank 128 | Change |
|:---|---:|---:|---:|
| Replay duration | 38.688 us | 37.696 us | -2.56% |
| Executed instructions | 805,469 | 640,856 | -20.44% |
| Issued instructions | 829,718 | 657,179 | -20.80% |
| Memory throughput | 55.11 GB/s | 56.47 GB/s | +2.47% |
| Registers/thread | 32 | 32 | unchanged |
| Achieved occupancy | 6.49% | 6.45% | unchanged |

The lower instruction count leaves memory dependencies and grid barriers as a
larger share of the shorter kernel. Source/SASS correlation also exposed a
runtime integer division used to derive `row` and `row_block` in every CTA.
That division is invariant for the target decode regime because `rows == 1`.

### 2026-07-21: specialize common single-row decode

A first experiment put a uniform `rows == 1` branch inside every kernel. Its
tail median improved only from 16.704 to 16.544 us, while full-forward decode
regressed by about 2.05 us and the M=8 path was perturbed. It was fully
reverted. The retained design instead creates single-row instantiations only
for ranks 64 and 128. Generic ranks and every multi-row launch select the
existing instantiations. For the single-row kernels, compile-time constants
fold `rows * rank`, set `row=0`, set `row_block=blockIdx`, and remove the
reciprocal/division sequence entirely.

Matched fifth-pass to sixth-pass production p50 is:

| Case | FP16 rank-only | FP16 + M=1 | BF16 rank-only | BF16 + M=1 |
|:---|---:|---:|---:|---:|
| decode attention, rank 64 | 35.84 us | 35.84 us | 35.84 us | 35.84 us |
| decode attention, rank 128 | 35.84 us | 34.82 us | 36.86 us | 36.86 us |
| decode MLP up, rank 128 | 43.01 us | 41.98 us | 45.06 us | 43.01 us |
| decode MLP down, rank 128 | 49.15 us | 47.10 us | 50.18 us | 49.15 us |
| batch 8 attention, rank 128 | 53.25 us | 53.25 us | 53.25 us | 53.25 us |
| rows 16 attention, rank 128 | 69.63 us | 69.63 us | 68.61 us | 68.61 us |

Tail-only Nsight measures 14.880 us, 10.92% below the rank-only 16.704 us.
The full two-kernel span falls from 33.184 to 31.232 us. NCU executed
instructions fall another 19.89%, from 640,856 to 513,382, and NCU replay
duration falls 11.89%, from 37.696 to 33.216 us. The transformer-shaped FP16
case improves from 40.96 to 38.91 us. All allocator peaks and numerical errors
are unchanged.

Focused coverage now explicitly crosses rows 1 and 3, ranks 37/64/128/320,
FP16/BF16, and validated/prepared calls: all 32 kernel combinations pass both
pytest and Compute Sanitizer, with 39 total focused tests passing and
`ERROR SUMMARY: 0 errors`.

### 2026-07-21: parallelize the up projection with full warps

Source-correlated NCU sampling after single-row specialization attributed 599
long-scoreboard samples to the up-weight loop and 486 to the down projection's
input load. Two down-side attempts were measured first. Caching each warp's
intermediate values in registers and broadcasting them with shuffles expanded
the kernel from 14.880 to 40.160 us. Cooperatively staging the repeated input
values in shared memory added a block barrier and regressed the kernel to
15.424 us. Both changes were fully reverted.

The retained up-side design differs from the earlier rejected two-lane output
split. A full warp still owns 32 adjacent output columns, so every rank step is
a fully coalesced 64-byte FP16/BF16 load. Independent warps process separate
32-rank slices, write FP32 partials into the existing down-phase shared buffer,
and one warp combines them after a block barrier. This creates latency-hiding
work without increasing global traffic, register count, shared memory, or
allocator memory.

The bounded FP16 warp-count sweep on physical GPU 4 selected the following
points:

| Rank | One warp | Two warps | Four warps | Eight warps | Selected |
|---:|---:|---:|---:|---:|:---|
| 64 | 10.416 us | 8.608 us | 8.640 us | not tested | two |
| 128 | 14.880 us | 13.152 us | 10.400 us | 10.496 us | four |

A matched BF16 rank-64 capture on physical GPU 5 confirms 10.432 to 8.640 us,
a 17.18% kernel reduction. This resolves the noisier full-forward CUDA-event
sample, whose 1.024 us timer ticks moved in the opposite direction during one
sweep. The larger M=1 projections show the end-to-end gain clearly: FP16 MLP
up/down improve from 41.98/47.10 to 37.89/43.01 us, and BF16 improves from
43.01/49.15 to 38.91/44.03 us. Multi-row paths are byte-for-byte unchanged and
their timings remain 53.25 us at M=8 and 68.61-69.63 us at M=16.

The exact final source passes 39 focused EoRA tests, all 29 Marlin JIT tests,
fresh FP16/BF16 builds, the six-shape integrated FP16 memcheck, and the
32-combination cooperative-kernel memcheck. Both sanitizer runs report
`ERROR SUMMARY: 0 errors`; all measured allocator peaks and the worst dense
FP32 reference error below 0.020 remain unchanged.

### 2026-07-21: remove the rank-128 down block barrier

After the full-warp up-projection pass, exact-source NCU still reported 19.61
long-scoreboard and 12.47 barrier stall cycles per issued instruction. The
single-row rank-128 down projection used two lanes per rank, but those lanes
were 128 thread indices apart. All 256 threads therefore wrote FP32 partials
to shared memory, crossed a CTA-wide barrier, and the first 128 threads loaded
and summed the two values.

The retained specialization instead maps adjacent lane pairs to ranks. One
lane accumulates even input positions and its neighbor accumulates odd input
positions. Within every warp, each parity group reads a contiguous 32-byte
rank segment; the even lane receives its neighbor's FP32 partial with one
warp shuffle and performs the same atomic accumulation. Every lane reaches
the shuffle with a full mask. The change is compile-time gated to
`static_single_row && static_rank == 128`; rank 64, runtime ranks, multi-row
calls, non-Ampere devices, and all library fallbacks retain their established
paths.

An intermediate symmetric XOR shuffle and the final one-way down shuffle both
measured 10.240 us FP16 median over 200 launches. The one-way form had the
lower mean, 10.539 versus 10.565 us, and was selected because only the even
lane consumes the exchanged value. Against exact commit `a98abeaa`, the final
FP16 median is 1.23% lower on physical GPU 4. A fresh exact-source BF16 capture
on physical GPU 5 measures 10.208 us, 1.84% below the 10.399 us baseline.

NCU replay duration improves 5.80%, from 23.100 to 21.760 us. Barrier stalls
fall 30.55%, long-scoreboard stalls fall 8.11%, and measured memory throughput
rises 6.10%. Registers remain 32 per thread, the static shared allocation
remains 1.02 KiB, and there are no local spills. Full-forward medians and all
allocator peaks remain unchanged at the reporting resolution.

The final source passed all 39 focused EoRA tests and all 29 Marlin/JIT tests,
including clean FP16 and BF16 `sm_80` compilation. Compute Sanitizer covered
32 validated/prepared, dtype, rank, and row combinations plus all six
integrated Marlin benchmark shapes. Both runs finished with
`ERROR SUMMARY: 0 errors`, and the integrated dense-FP32-reference maximum
absolute error remained below 0.0078.

### 2026-07-21: scale the rank-128 down phase for wide projections

Source-correlated NCU after the adjacent-lane pass still attributed 91 sampled
long-scoreboard stalls to the down-weight loop, versus 39 in the up-weight
loop. The up-reduction barrier accounted for another 41 barrier samples. Three
alternatives were measured and reverted before changing the launch policy:

- Broadcasting each input value across a warp initially exposed an unsafe
  full-mask shuffle in a divergent tail loop. The corrected uniform-loop form
  passed all eight focused rank-128 cases, but regressed the kernel from 10.240
  to 14.048 us.
- A counted named barrier let only the up-projection warps participate. It tied
  rank 128 at 10.240 us and regressed rank 64 from 8.608 to 8.640 us.
- Thread-local mirrors removed four warmed mutex-protected metadata-cache
  reads from prepared native dispatch. Same-binary source toggles moved
  sub-microsecond p50 values in both directions and changed no GPU work, so the
  cache mirrors were not retained.

The retained change leaves the 344-block MLP-up launch grid intact and raises
only the single-row static-rank-128 down-participation cap from 128 to 256.
Attention `N=4096` provides only 128 output blocks and is therefore unchanged.
Rank 64, generic ranks, and all multi-row kernels retain the old cap. The
128/192/256/all-block sweep selected 256 because its 10.912 us intermediate
median was lower than 11.200 us at 192 and 11.008 us with all 344 blocks. The
exact narrowed final source measured 10.944 us against the exact 13.024 us
baseline over 200 FP16 launches.

Two complete source-toggle pairs on physical GPU 4 repeated the FP16
full-forward p50 improvement from 44.032 to 41.984 us (-4.65%). A matched
physical-GPU-5 BF16 pair measured 45.056 to 43.008 us (-4.55%). Host-side
means and upper percentiles remained noisy, so the direct kernel captures are
the primary attribution: FP16 improves 15.97% and BF16 improves 14.85%.

Exact-source NCU reports 23.17 to 20.96 us replay duration, 29.85 to 20.95
barrier stalls per issued instruction, and 168.82 to 186.60 GB/s memory
throughput. Register use remains 32 per thread, static shared memory remains
1.02 KiB, and no new allocation is introduced. Final six-shape FP16 and BF16
sweeps retain 264/709.5/264/320/384 KiB peaks for attention, MLP up, MLP down,
M=8, and M=16 respectively; worst dense-FP32-reference error remains below
0.0191.

The exact final source passes all 39 focused EoRA tests and all 29 Marlin/JIT
tests under `/root/vm314t/bin/python` with Torch 2.13.0+cu130. Compute
Sanitizer passes all 32 cooperative-kernel parameterizations and all six
integrated Marlin shapes with `ERROR SUMMARY: 0 errors`. Every GPU command
uses `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5`; logical devices
0 and 1 map to physical GPUs 4 and 5.

### 2026-07-21: double CTAs for strong rank-128 contractions

The next profile-and-measure loop first tested three ways to shorten the
already-optimized MLP-up kernel. A 128-thread adjacent-rank mapping passed all
eight targeted rank-128 correctness cases but regressed the kernel from
10.944 to 12.896 us. Writing up-projection atomics directly into the FP16 or
BF16 output also passed shape checks, but increased numerical error and
regressed the kernel to 15.392 us. Finally, compile-time specialization for
`K=4096, N=11008` moved one 200-launch median by a single 64 ns profiler tick,
from 10.944 to 10.880 us, while the means were effectively identical at
11.9408 and 11.9393 us. None justified its performance, accuracy, or binary
size tradeoff, and all three were reverted.

Profiling the opposite 11008-to-4096 projection showed a structural limit:
32 output columns per CTA produced only 128 blocks, leaving half of the
selected 256-CTA rank-128 down capacity unused. A 24/20/18/16/12-column sweep
measured 14.640/13.600/13.440/13.152/15.232 us against the 16.543 us baseline.
The retained 16-column point launches exactly 256 CTAs. It is gated by the
compile-time single-row/rank-128 specialization and runtime `K >= 2N`, so no
other kernel instantiation or projection family changes.

On physical GPU 4, the exact final FP16 kernel median is 13.152 us, 20.50%
below the 16.543 us baseline. On physical GPU 5, BF16 improves from 16.496 to
13.184 us (-20.08%). Matched full-Marlin source toggles repeat the end-to-end
effect: FP16 p50 moves from 50.176 to 46.080 us (-8.16%), and BF16 moves from
51.200 to 48.128 us (-6.00%). Allocator-visible peak remains 264 KiB for the
MLP-down case.

NCU attributes the FP16 gain to increased grid parallelism. Replay duration
falls from 35.74 to 25.54 us, waves per SM double from 0.13 to 0.26, achieved
occupancy rises from 12.73% to 25.71%, and measured DRAM throughput rises
from 109.43 to 153.16 GB/s. Long-scoreboard stalls fall from 22.18 to 20.14
cycles per issued instruction. Registers remain 32 per thread, static shared
memory remains 1.02 KiB, and local-memory spills remain zero.

The exact retained source passes all 71 focused EoRA tests and all 29
Marlin/JIT tests. The expanded matrix explicitly covers both non-divisible
expansion (`K=97, N=130`) and strong contraction (`K=257, N=97`) for two
dtypes, two row counts, four ranks, and both validated/prepared entry points.
Compute Sanitizer passes all 64 combinations plus the six integrated Marlin
shapes with `ERROR SUMMARY: 0 errors`. Final FP16 and BF16 all-shape sweeps
keep maximum dense-FP32-reference error below 0.020 and the established
264/709.5/264/320/384 KiB allocator peaks. All commands use
`/root/vm314t/bin/python` and expose only PCI-ordered physical GPUs 4 and 5.

### 2026-07-21: post-contraction dispatch audit and rejected follow-ups

A fresh Nsight Systems capture of the published tenth pass profiled 49 warmed
FP16 rank-128 attention forwards on physical GPU 4. The median native sequence
was 14.752 us of Marlin, a 1.408 us inter-kernel gap, and 13.824 us of EoRA.
The CPU range was 68 us, including a 46.007 us lead to the first GPU kernel.
CUDA launch API medians were 7.734 us for the Marlin launch and 4.609 us for
the cooperative EoRA launch. This confirms that the old Python round trip and
double-digit inter-kernel bubble remain eliminated; host/operator setup is now
larger than the 29.984 us two-kernel GPU span.

A Torch operator trace reached the same attribution from a different angle.
Under profiler instrumentation, the outer prepared forward took 114 us total
with 64.6 us self time, the integrated native operator took 37.06 us, its two
`aten::empty` allocations accounted for 10.76 us, and the two GPU kernels
accounted for 29.28 us. The following bounded follow-ups were correct but did
not improve production latency and were reverted:

- Extending the adjacent-pair rank-128 down mapping to every row changed M=8
  FP16 from 41.344 to 42.560 us and M=16 BF16 from 55.760 to 61.439 us.
- Rank-64 NCU reported 17.41 us replay duration, 319,144 executed
  instructions, 16.7 barrier and 15.5 long-scoreboard stall cycles per issued
  instruction, 32 registers, 1.02 KiB static shared memory, and no spills. A
  barrier-free adjacent-pair mapping regressed FP16 from 11.872 to 12.576 us;
  four independent down chains regressed it further to 12.959 us.
- An early prepared-Python shortcut produced inconsistent A/B results and no
  stable improvement, so the wrapper remains shared with fallback handling.
- Attention output tiles of 16, 24, and 30 columns measured 13.472, 14.176,
  and 13.552 us against 13.408 us at the retained 32 columns. The 30-column
  point increased the grid from 128 to 137 CTAs but still regressed 1.07%.

### 2026-07-21: pack M<=8 Marlin reduction scratch

The remaining allocator peak came from padding every small-M FP32 reduction
slice to 16 rows. The `m_block_size_8` Marlin kernel actually advances its
fragment loop by two, visiting only four `int4` groups. The retained change
packs those groups with half the old group stride, uses an eight-row per-slice
offset, and makes both generic and prepared native allocators reserve eight
rows for `M <= 8`. The M=16 and larger code, packed-prefill exception, output
layout, reduction arithmetic, and all architecture fallbacks are unchanged.

Matched 500-warmup, 3,000-sample source toggles measured:

| Case | FP16 before | FP16 packed | BF16 before | BF16 packed | Peak before | Peak packed |
|:---|---:|---:|---:|---:|---:|---:|
| attention rank 64, M=1 | 35.84 us | 35.84 us | 36.86 us | 35.84 us | 264 KiB | 136 KiB |
| attention rank 128, M=1 | 35.84 us | 35.84 us | 35.84 us | 37.89 us | 264 KiB | 136 KiB |
| MLP up rank 128, M=1 | 41.98 us | 40.96 us | 37.89 us | 38.91 us | 709.5 KiB | 365.5 KiB |
| MLP down rank 128, M=1 | 39.94 us | 39.94 us | 41.98 us | 40.96 us | 264 KiB | 136 KiB |
| attention rank 128, M=8 | 52.22 us | 52.22 us | 54.27 us | 54.27 us | 320 KiB | 192 KiB |
| attention rank 128, M=16 | 69.63 us | 69.63 us | 68.61 us | 68.61 us | 384 KiB | 384 KiB |

The BF16 whole-forward samples move in both directions by one or two 1.024 us
event ticks. A matched Nsight Systems source toggle therefore isolated the
changed kernel on physical GPU 5:

| Nsight Systems median | 16-row scratch | Packed 8-row scratch | Change |
|:---|---:|---:|---:|
| Profiled full-forward p50 | 66.69 us | 65.76 us | -1.39% |
| Marlin kernel | 14.976 us | 14.912 us | -0.43% |
| Unchanged EoRA kernel | 13.792 us | 14.048 us | +1.86% noise |

The isolated Marlin result and tied FP16 production medians establish that
the 40.00-48.48% incremental peak-memory reduction has no material latency
cost. Relative to the original implementation, final full-call reductions are
81.78-95.07% for M<=8 and 81.82% for M=16.

The exact retained source was rebuilt for FP16 and BF16 under Torch
2.13.0+cu130 and passed all 29 Marlin/JIT tests in 262.99 seconds plus all 71
focused EoRA tests in 42.65 seconds. Compute Sanitizer then exercised all six
shapes through both FP16 prepared-cooperative and ordinary Marlin dispatch on
physical GPU 4, followed by all six BF16 prepared-cooperative shapes on
physical GPU 5. Both runs report `ERROR SUMMARY: 0 errors`; maximum absolute
error against the dense FP32 reference remains 0.01902. Every command used
`/root/vm314t/bin/python` with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5`.

### 2026-07-21: pack M<8 Marlin scratch by live output row

After the eight-row pass, M=1 still reserved all eight rows in every global
reduction slice even though only one can reach the output. The retained
follow-up makes the small-M slice height equal to `prob_m`. For `M=2..7`, a
thread's row-pair index selects the even and odd fragment lanes and stores
each live row as contiguous `float2` column pairs. M=1 uses only the threads
that own its even fragment lane and bypasses the generic row-pair predicates.
M=8 stays on the prior vectorized eight-row path, while M>=16, packed prefill,
all reduction arithmetic, and non-target fallbacks are unchanged.

Before retaining the change, the generic live-row implementation was profiled
twice on physical GPU 5. It reduced the rank-128 attention peak from 136 to
24 KiB and produced full-profile p50 values of 64.45 and 65.34 us versus
65.76 us for the published eight-row source. Its Marlin median was 15.104 us
in both captures, 0.192 us above the published 14.912 us. The direct M=1
mapping recovered that small cost:

| BF16 Nsight Systems median | Published 8 rows | Generic live rows | Final M=1 mapping |
|:---|---:|---:|---:|
| Profiled full-forward p50 | 65.76 us | 64.45 / 65.34 us | 62.91 us |
| Marlin kernel | 14.912 us | 15.104 / 15.104 us | 14.848 us |
| Peak allocation | 136 KiB | 24 KiB | 24 KiB |

The final 500-warmup, 3,000-sample production sweep measured:

| Case | FP16 p50 | BF16 p50 | Final peak |
|:---|---:|---:|---:|
| attention rank 64, M=1 | 36.86 us | 36.86 us | 24 KiB |
| attention rank 128, M=1 | 36.86 us | 34.82 us | 24 KiB |
| MLP up rank 128, M=1 | 35.84 us | 37.89 us | 64.5 KiB |
| MLP down rank 128, M=1 | 39.94 us | 41.98 us | 24 KiB |
| attention rank 128, M=8 | 54.27 us | 53.25 us | 192 KiB |
| attention rank 128, M=16 | 69.63 us | 68.61 us | 384 KiB |

This is an incremental 82.35% full-call peak reduction for each measured M=1
shape relative to the published eight-row pass. Relative to the original
implementation, attention, MLP up, and MLP down now use 98.80%, 96.78%, and
99.13% less transient allocator memory. M=8 remains 90.62% below the original
peak and M=16 remains 81.82% below it.

The new regression compares FP32 scratch reduction with the established
FP16/BF16 reduction path for every `M=1..8`. The full final-source suites pass
31 Marlin/JIT tests and 71 fused-EoRA tests. Compute Sanitizer reports zero
errors for all six FP16 shapes through both cooperative and ordinary Marlin,
all six BF16 cooperative shapes, and the separate two-dtype M=1..8 test.
Maximum absolute error against the dense FP32 LoRA reference is 0.01902.

The exact FP16 and BF16 JIT source hashes are `a5c51d23d4f2bf0f` and
`8953da26e59e3b21`. Both were built by `/root/vm314t/bin/python` with Torch
2.13.0+cu130, CUDA 13.0, `-gencode=arch=compute_80,code=sm_80`, `-O3`,
`-Xptxas -O3,-dlcm=ca`, and `-lineinfo`. Every GPU command used
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5`.

### 2026-07-21: decouple square-attention down and up grid widths

Profiling resumed from exact published commit `83617750`. A 200-call BF16
Nsight Systems trace on physical GPU 5 separated the remaining host and device
costs before changing source:

| Current-head median | Time |
|:---|---:|
| CPU NVTX range | 68.601 us |
| Range start to Marlin launch API | 34.115 us |
| Range start to Marlin GPU kernel | 44.509 us |
| Marlin launch API | 8.206 us |
| Cooperative EoRA launch API | 5.160 us |
| Marlin kernel | 14.848 us |
| Inter-kernel GPU gap | 1.408 us |
| EoRA kernel | 10.528 us |
| Two-kernel GPU span | 26.848 us |

This confirmed that native prepared dispatch had already reduced the
inter-kernel bubble to one low-microsecond gap, while CPU lead remained the
largest eager cost. Two host/allocation follow-ups were measured and reverted:

- Caching `OpOverloadPacket.default` moved the first CPU range from 68.601 to
  68.270 us, but a repeat regressed it to 72.469 us. GPU work was unchanged.
- Combining output and reduction scratch into one allocation tied scratch
  lifetime to the returned output, retained the 24 KiB measured peak, and
  regressed warmed BF16 p50 from 38.91 to roughly 44-45 us.

Baseline NCU showed that the 128-block EoRA grid achieved only 12.34%
occupancy and 0.13 waves per SM. The 32-column up phase needs exactly those
128 blocks for 4,096 output columns, but the rank-128 down phase can use more
parallelism. The retained policy therefore separates `blocks_per_row`, which
continues to own output tiles, from `launch_blocks_per_row` and `down_blocks`.
Extra blocks help the down projection, cross the cooperative grid barrier, and
then receive a `col_begin` at or beyond `out_features`, so they cannot write
output. The gate requires compile-time single-row rank 128 plus a runtime
square, non-contracting projection with at least 128 output blocks.

The complete BF16 raw-kernel sweep used 50 warmups and 200 measured forwards:

| Down / launch CTAs | EoRA median | Change from 128 | Decision |
|---:|---:|---:|:---|
| 128 | 10.528 us | baseline | published source |
| 160 | 10.240 us | -2.74% | rejected |
| 192 | 9.920 us | -5.78% | retained |
| 224 | 9.952 us | -5.47% | rejected |
| 256 | 10.239 us | -2.75% | rejected |

The 192-CTA point is also the smallest of the tied-nearby 192/224 choices,
avoiding 32 unnecessary down-only blocks. Nsight profiling used the corrected
`/root/vm314t/bin/python` and Torch 2.13.0+cu130. Full-process Nsight injection
required `PYTHON_GIL=1` to avoid a Python 3.14t launcher deadlock before CUDA;
this profiler-only setting did not change the compiled kernel or normal
free-threaded benchmark runs.

An exact current-environment FP16 control on physical GPU 4 changed only the
launch constant. Its 128-CTA EoRA median was 10.143 us versus 10.112 us at the
retained 192 CTAs (-0.31%), effectively tied at profiler resolution. The
synchronized 500-sample p50 moved from 32.768 to 30.720 us. Because the Marlin
median also shifted from 14.400 to 14.944 us between captures, the BF16 sweep
and matched NCU occupancy result remain the primary selection evidence; the
FP16 pair establishes that the shared policy does not regress the second
dtype/device.

Matched full-set NCU attributes the gain to the intended occupancy increase:

| NCU metric | 128 CTAs | Final 192 CTAs | Change |
|:---|---:|---:|---:|
| Replay duration | 22.11 us | 19.42 us | -12.17% |
| Waves per SM | 0.13 | 0.19 | +46.15% |
| Achieved occupancy | 12.34% | 17.25% | +4.91 points |
| Compute throughput | 4.29% | 5.35% | +1.06 points |
| Memory throughput | 96.22 GB/s | 109.54 GB/s | +13.84% |
| L2 hit rate | 40.75% | 42.87% | +2.12 points |
| Long-scoreboard stall cycles/instruction | 17.93 | 16.60 | -7.42% |
| Executed instructions | 547,476 | 600,820 | +9.74% |
| Registers / static shared / spills | 32 / 1.02 KiB / 0 | 32 / 1.02 KiB / 0 | unchanged |

The final sequential 500-warmup, 3,000-sample production sweep measured:

| Case | FP16 p50 | BF16 p50 | Peak |
|:---|---:|---:|---:|
| attention rank 64, M=1 | 31.74 us | 33.79 us | 24 KiB |
| attention rank 128, M=1 | 30.72 us | 30.72 us | 24 KiB |
| MLP up rank 128, M=1 | 37.89 us | 37.89 us | 64.5 KiB |
| MLP down rank 128, M=1 | 39.94 us | 41.98 us | 24 KiB |
| attention rank 128, M=8 | 53.25 us | 54.27 us | 192 KiB |
| attention rank 128, M=16 | 69.63 us | 68.61 us | 384 KiB |

Whole-forward means and upper percentiles remained host-noisy, so the direct
Nsight kernel sweep is the primary incremental attribution. Allocator peaks
are identical to the live-row scratch pass. Dense-FP32-reference maximum
absolute error remains 0.007788 FP16 and 0.01902 BF16.

The exact final source passed all 71 focused EoRA tests in 31.97 seconds and
all 31 Marlin/JIT tests in 44.70 seconds. Compute Sanitizer exercised all six
FP16 shapes through both cooperative and ordinary Marlin dispatch on physical
GPU 4 and all six BF16 cooperative shapes on physical GPU 5; both report
`ERROR SUMMARY: 0 errors`. Final FP16 and BF16 JIT fingerprints are
`8403f700b53ad0a5` and `293456164ae727ac`. The installed runtime inventory at
this pass is Python 3.14.5t, Torch 2.13.0+cu130, CUDA 13.0, Transformers
5.14.1, and Triton 3.7.1. Every GPU command exposed only PCI-ordered physical
GPUs 4 and 5.

### 2026-07-21: reject cooperative CUDA-graph replay

After publishing the 192-CTA attention pass, graph capture was re-audited
because its 9.920 us fused adapter kernel is now less raw work than the
original three-kernel library tail. Direct capture of the prepared integrated
operator succeeded on this CUDA 13 / Torch 2.13 stack, preserved cooperative
state, and produced the expected reduction-order agreement. An initial
rank-128 attention run moved replay p50 from 32.768 to 31.744 us, but paired
same-process measurements showed that most of that apparent gain was GPU
clock state.

A capture-gate override then compared both graph routes in one process for
every production shape, with 300 alternating warmups and 1,500 paired samples:

| FP16 graph case | Integrated p50 | Library p50 | Result |
|:---|---:|---:|:---|
| attention rank 64, M=1 | 30.720 us | 29.696 us | integrated slower |
| attention rank 128, M=1 | 27.648 us | 27.648 us | tie |
| MLP up rank 128, M=1 | 33.792 us | 33.792 us | tie |
| MLP down rank 128, M=1 | 37.888 us | 37.888 us | tie |
| attention rank 128, M=8 | 53.248 us | 27.648 us | integrated slower |
| attention rank 128, M=16 | 69.632 us | 29.696 us | integrated slower |

The largest difference between library and integrated outputs was 0.0078125,
consistent with their already-validated FP32 accumulation-order difference.
The eager `torch.cuda.is_current_stream_capturing()` query measured only 0.533
us per call, so skipping it for M=1 would exchange a small host saving for a
rank-64 graph regression and a more complex policy. Both attempted gate
changes were reverted. Production continues to use the library adapter route
during graph capture and the faster integrated route for eligible eager calls.

### 2026-07-21: fuse rank-128 decode attention into one cooperative launch

Profiling resumed from published commit `8975d821` in the corrected
`/root/vm314t` environment. Three independent BF16 runs on physical GPU 5,
each with 500 warmups and 3,000 synchronized CUDA-event samples, all measured
34.82 us p50 for `M=1, K=N=4096, rank=128`. A 200-forward Nsight Systems
capture then established the exact native-dispatch baseline:

| BF16 baseline median | Time |
|:---|---:|
| CPU NVTX range | 68.881 us |
| Range start to Marlin launch API | 34.621 us |
| Marlin launch API | 7.921 us |
| Gap to EoRA launch API | 1.543 us |
| EoRA launch API | 5.070 us |
| Marlin GPU kernel | 14.944 us |
| Inter-kernel GPU gap | 1.408 us |
| EoRA GPU kernel | 10.080 us |
| Two-kernel GPU span | 26.495 us |

Two bounded feasibility experiments preceded the combined kernel. Returning
the prepared output without the intermediate tuple tied warmed p50 and moved
the 1.543 us API gap only to 1.533 us, so that source was reverted. A
256-thread Marlin specialization reduced its isolated kernel from 14.944 to
13.824 us while preserving the 124-block grid, 96 registers per thread,
166,912 bytes of dynamic shared memory, and zero spills. Whole-forward timing
was noisy by one event tick, so the standalone specialization was also
reverted; it established that the 256-thread geometry needed by a combined
rank-128 LoRA phase was feasible.

The retained specialization is a generated `MarlinEoraRank128` instantiation.
It launches one cooperative block per runtime-reported SM, with 256 threads
per block. After the ordinary Marlin phase completes, a grid barrier makes the
base output and reduction writes visible. The kernel then reuses Marlin's
dead FP32 reduction scratch as the 128-float EoRA workspace, computes the LoRA
down projection across all CTAs, crosses a second grid barrier, and adds the
LoRA up projection in coalesced 32-column tiles. On the tested 124-SM boards,
four CTAs cover a short second output-tile pass.

Dispatch is deliberately narrow: `M=1`, `K=N=4096`, rank 128, compute
capability 8.0, exactly 124 runtime-reported SMs, and cooperative-launch
support. It is reached only through the already-validated prepared GPTQ
W4A16/group-128/no-bias path. Unsupported ranks, dimensions, row counts,
architectures, devices, and CUDA graph capture continue through ordinary
Marlin followed by the existing prepared EoRA kernel. No CUDA index is used
as a capability proxy.

The final exact-source BF16 profile on physical GPU 5 contains exactly one
`MarlinEoraRank128` launch for every measured forward:

| Nsight Systems median | Two-kernel baseline | Mega-kernel | Change |
|:---|---:|---:|---:|
| GPU launches / forward | 2 | 1 | -50.00% |
| CPU NVTX range | 68.881 us | 59.689 us | -13.34% |
| Range start to first launch API | 34.621 us | 31.761 us | -8.26% |
| Launch API time | 12.991 us total | 8.431 us | -35.10% |
| Inter-launch / inter-kernel gaps | 1.543 / 1.408 us | 0 / 0 us | eliminated |
| GPU work span | 26.495 us | 23.328 us | -11.95% |

The final kernel uses a `124 x 256` launch, 96 registers per thread, 166,912
bytes of dynamic shared memory, 167,936 bytes of executed shared memory, and
zero local memory or spills. The normal free-threaded final sweeps used 500
warmups and 3,000 samples sequentially on physical GPUs 4 and 5:

| Case | FP16 p50 | BF16 p50 | Peak |
|:---|---:|---:|---:|
| attention rank 64, M=1 | 33.79 us | 33.79 us | 24 KiB |
| attention rank 128, M=1 | 30.72 us | 30.72 us | 24 KiB |
| MLP up rank 128, M=1 | 35.84 us | 37.89 us | 64.5 KiB |
| MLP down rank 128, M=1 | 39.94 us | 41.98 us | 24 KiB |
| attention rank 128, M=8 | 53.25 us | 53.25 us | 192 KiB |
| attention rank 128, M=16 | 69.63 us | 68.61 us | 384 KiB |

The target FP16 p50 is identical to the prior published 30.72 us. The target
BF16 p50 is also identical to the older published long-run result, while it is
11.77% below all three fresh 34.82 us pre-change controls from this profiling
round. Three initial post-change BF16 repetitions measured 28.67, 28.67, and
29.70 us. Because eager host timing varies with clock and runtime state, the
direct Nsight launch count and GPU span are the primary attribution. All
non-target rows take source-identical fallback kernels; one- or two-tick
movements in their full-call measurements are treated as host noise.

The final source passes the combined 104-test focused suite: 71 EoRA tests and
33 Marlin/JIT tests, including real FP16 and BF16 4096-to-4096 adapter modules
compared with an ordinary quantized base plus dense FP32 LoRA reference.
Maximum absolute error in the complete matrix is 0.01902. Compute Sanitizer
reports `ERROR SUMMARY: 0 errors` for the final FP16 mega-kernel on physical
GPU 4, the BF16 mega-kernel on physical GPU 5, and a BF16 rank-64 fallback on
physical GPU 5. Generated-source validation, Ruff, and `git diff --check` also
pass.

Final FP16 and BF16 JIT fingerprints are `92b824fdeaeda4ff` and
`97f12b34818c0638`. Both use Python 3.14.5t from `/root/vm314t`, Torch
2.13.0+cu130, CUDA 13.0, Transformers 5.14.1, Triton 3.7.1,
`-gencode=arch=compute_80,code=sm_80`, `-O3`,
`-Xptxas -O3,-dlcm=ca`, and `-lineinfo`. Nsight full-process injection alone
used `PYTHON_GIL=1`; normal benchmarks retained the free-threaded runtime.
Every GPU command used
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5`.

### 2026-07-21: expose LoRA-down memory-level parallelism in the mega-kernel

Nsight Compute on the published one-chain mega-kernel identified a latency
bottleneck rather than a bandwidth limit. The kernel reached only 12.50%
theoretical occupancy because Marlin's 166,912-byte dynamic shared-memory
allocation permits one 256-thread block per SM. It used only 12.99% compute
and 13.93% DRAM throughput, with a warp eligible to issue on 14.72% of
scheduler cycles. Barrier and long-scoreboard stalls consumed 4.72 and 4.33
cycles per issued instruction respectively. Source-correlated PC sampling
placed 491 of the 669 long-scoreboard samples directly on the two LoRA-down
loads.

The retained change gives that down loop eight independent FP32 accumulator
chains. Each thread still owns the same rank and K positions, the same number
of values is read, and the same lane-pair shuffle and cross-CTA atomics reduce
the result. The independent chains let Ampere overlap outstanding `x` and
LoRA-A loads despite the kernel's intentionally low Marlin-limited occupancy.
No allocation or launch is added.

A bounded BF16 Nsight Systems sweep on physical GPU 5 selected eight chains:

| Down chains | Kernel p50 | Kernel mean | Kernel p95 | Change vs. one |
|---:|---:|---:|---:|---:|
| 1 | 23.328 us | 23.888 us | 26.592 us | baseline |
| 2 | 21.888 us | 22.355 us | 24.703 us | -6.17% |
| 4 | 21.056 us | 21.507 us | 23.840 us | -9.74% |
| 8 | 20.832 us | 21.280 us | 23.615 us | -10.70% |

The eight-chain endpoint remains measurably below four in p50, mean, and p95.
Its direct Nsight Compute comparison explains the improvement:

| NCU metric | One chain | Eight chains | Change |
|:---|---:|---:|---:|
| Replay duration | 31.74 us | 24.16 us | -23.88% |
| Warp cycles / issued instruction | 13.73 | 10.67 | -22.29% |
| Long-scoreboard stall cycles / issued instruction | 4.33 | 1.71 | -60.51% |
| Barrier stall cycles / issued instruction | 4.72 | 4.33 | -8.26% |
| Measured memory throughput | 340.12 GB/s | 446.96 GB/s | +31.41% |
| DRAM throughput | 13.93% | 18.30% | +31.37% |
| Compute throughput | 12.99% | 16.18% | +24.56% |
| BF16 registers / dynamic shared / spills | 96 / 166,912 B / 0 | 96 / 166,912 B / 0 | unchanged |

The final 200-forward Nsight Systems medians are 20.576 us FP16 on physical
GPU 4 and 20.832 us BF16 on physical GPU 5. FP16 uses 94 registers per thread;
BF16 uses 96. Both retain the `124 x 256` cooperative launch, 166,912 bytes of
dynamic shared memory, 167,936 bytes of executed shared memory, and zero local
memory. Peak allocator use remains 24 KiB for the target call.

The final normal free-threaded 500-warmup, 3,000-sample sweeps measured:

| Case | FP16 p50 | BF16 p50 | Peak |
|:---|---:|---:|---:|
| attention rank 64, M=1 | 35.84 us | 35.84 us | 24 KiB |
| attention rank 128, M=1 | 29.70 us | 31.74 us | 24 KiB |
| MLP up rank 128, M=1 | 35.84 us | 37.89 us | 64.5 KiB |
| MLP down rank 128, M=1 | 39.94 us | 41.98 us | 24 KiB |
| attention rank 128, M=8 | 53.25 us | 53.25 us | 192 KiB |
| attention rank 128, M=16 | 69.63 us | 68.61 us | 384 KiB |

The eager BF16 full-call median moved by one 1.024 us event tick in both
directions across repeated same-binary runs, while the 200-sample raw-kernel
improvement is stable in p50, mean, and p95. The direct profiles therefore
remain the primary attribution. Every non-target row follows unchanged code.

Both exact FP16 and BF16 sources compiled successfully for `sm_80`. The
affected real-module regression passes in both dtypes, and the combined suite
passes 103 tests with only the redundant force-rebuild smoke test deselected;
that smoke test had passed in the immediately preceding 104-test run. Compute
Sanitizer reports `ERROR SUMMARY: 0 errors` for the final eight-chain FP16
kernel on physical GPU 4 and BF16 kernel on physical GPU 5. The full sweep's
maximum dense-FP32-reference error remains 0.01902.

Final FP16 and BF16 JIT fingerprints are `0492552cadf7373e` and
`afe131d6ad40313f`. The builds use `/root/vm314t/bin/python`, Torch
2.13.0+cu130, CUDA 13.0, `-gencode=arch=compute_80,code=sm_80`, `-O3`,
`-Xptxas -O3,-dlcm=ca`, and `-lineinfo`. Every GPU test, benchmark, sanitizer
run, and profile exposed only PCI-ordered physical GPUs 4 and 5.

### 2026-07-22: remove the LoRA-up tail and coalesce LoRA-down work

This pass began from published revision `5c9b90dd` on
`paroquant-update-0721` and PR #25. The formal workload is the prepared GPTQ
W4A16 mega-kernel at `M=1, K=N=4096, rank=128`, group size 128, symmetric
quantization, no activation ordering, no bias, and FP32 reduction. Every
profile used one cooperative `124 x 256` launch per forward. FP16 ran on
physical GPU 4/logical `cuda:0`; BF16 ran on physical GPU 5/logical `cuda:1`.
Both are runtime-probed PG506-230 `sm_80` devices with 124 SMs and 98,304 MiB.

The exact starting source was rebuilt under `/root/vm314t/bin/python` with
Python 3.14.5t, Torch 2.13.0+cu130, CUDA 13.0, Nsight Systems 2024.6.2, and
Nsight Compute 2025.3.1. Before source changes, the real-module dense-update
test passed for both dtypes and the benchmark agreed with its dense FP32 LoRA
reference:

| Exact-source Nsys baseline | FP16, GPU 4 | BF16, GPU 5 |
|:---|---:|---:|
| Raw kernel p50 | 20.544 us | 20.896 us |
| Raw kernel mean | 21.081 us | 21.305 us |
| Raw kernel p95 | 23.808 us | 23.616 us |
| Minimum / maximum | 20.064 / 25.024 us | 20.448 / 25.920 us |
| Launches per forward | 1 | 1 |
| Peak allocation | 24 KiB | 24 KiB |
| Maximum absolute error | 0.005573 | 0.01007 |

The baseline Nsight Systems reports are
`round16_latest_mega_fp16_gpu4_nsys.nsys-rep` and
`round16_latest_mega_bf16_gpu5_nsys.nsys-rep` under
`artifacts/eora_marlin_20260721/`. Each contains 200 measured forwards after
500 warmups. The baseline NVTX range p50 was 60.621 us FP16 and 58.542 us
BF16 under full-process Nsys injection. The corresponding pre-launch, launch
API, and API-to-kernel medians were 33.524/7.670/2.340 us FP16 and
31.405/7.975/2.184 us BF16. These instrumented host ranges are not substituted
for direct GPU duration.

Full-set Nsight Compute replay established that this was no longer a
bandwidth-saturation problem. BF16 used 96 registers per thread, 166.91 KiB
dynamic shared memory, no local-memory spills, and one resident block per SM.
Theoretical and achieved occupancy were 12.50% and 12.33%. Only 18.89% of
scheduler cycles had at least one eligible warp, while DRAM and compute
throughput were 17.76% and 15.87%. Warp cycles per issued instruction were
10.58; barrier and long-scoreboard stalls contributed 4.36 and 1.66 cycles
per instruction. Source sampling attributed 1,020 of 1,922 samples to
barriers, with the first cooperative grid synchronization waiting on the
Marlin stripe tail. FP16 showed the same structure at 94 registers per thread,
12.46% achieved occupancy, and zero spills.

The source inspection exposed a second structural tail after the final grid
synchronization: 128 32-column LoRA-up tiles were mapped onto only 124 CTAs.
Blocks 0-3 therefore executed a second full tile after every other block had
finished, while their upper four warps were idle during both passes. The
retained up mapping gives warps 0-3 the primary tile and warps 4-7 the
secondary tile in blocks 0-3. All eight warps write disjoint shared partials,
one block barrier exposes them, and threads 0-63 reduce and write the two
tiles. The FP32 rank grouping and addition order within each output remain
unchanged.

That change alone produced a reproducible raw-kernel improvement:

| Up-tail result | FP16, GPU 4 | BF16, GPU 5 repeat |
|:---|---:|---:|
| Raw kernel p50 | 19.808 us | 20.031 us |
| Raw kernel mean | 20.335 us | 20.591 us |
| Raw kernel p95 | 22.960 us | 23.328 us |
| p50 change vs. baseline | -3.58% | -4.14% |

The BF16 full-counter replay moved from 24.93 to 23.74 us. Scheduler cycles
with an eligible warp rose from 18.89% to 19.13%, warp cycles per issued
instruction fell from 10.58 to 10.43, and barrier stall cost fell from 4.36
to 4.13 cycles per instruction. Registers, shared memory, theoretical
occupancy, and spills were unchanged.

The remaining LoRA-down phase originally assigned two threads to each rank in
every CTA. That preserved broad grid parallelism but generated 124 atomics per
rank and only partially coalesced the rank-weight loads. The retained mapping
assigns each CTA one of four 32-rank tiles and one strided K slice. Each warp
loads 32 adjacent LoRA-A values from a single K row, the eight warp partials
reduce through already-allocated shared scratch, and one warp emits 32
atomics. Total atomics fall from `124 * 128 = 15,872` to
`124 * 32 = 3,968`; no allocation, launch, or phase boundary is added.

Two independent BF16 Nsys captures measured 19.632/20.062 us and
19.632/20.162 us p50/mean. The stable identical medians and 0.10 us mean
spread distinguish the change from event-timing and clock noise. FP16
cross-validation measured 19.456 us p50 and 19.891 us mean.

| Raw mega-kernel | Baseline FP16 | Final FP16 | Change | Baseline BF16 | Final BF16 | Change |
|:---|---:|---:|---:|---:|---:|---:|
| p50 | 20.544 us | 19.456 us | -5.30% | 20.896 us | 19.632 us | -6.05% |
| mean | 21.081 us | 19.891 us | -5.64% | 21.305 us | 20.062 us | -5.83% |
| p95 | 23.808 us | 22.306 us | -6.31% | 23.616 us | 22.304 us | -5.56% |
| peak allocation | 24 KiB | 24 KiB | unchanged | 24 KiB | 24 KiB | unchanged |

The final reports are
`round16_parallel_up_tail_down_tile32_candidate_fp16_gpu4_nsys.nsys-rep`,
`round16_parallel_up_tail_down_tile32_candidate_bf16_gpu5_nsys.nsys-rep`, and
`round16_parallel_up_tail_down_tile32_repeat_bf16_gpu5_nsys.nsys-rep`. The
normal synchronized target p50 remains quantized at 30.72 us in the final
sequential matrix, so the raw 200-launch distributions are the primary
incremental attribution. Nsys-instrumented host range p50 was 61.457 us FP16
and 59.809 us in the clean BF16 repeat; native dispatch and its one-launch
contract were not changed by this pass.

The final combined BF16 full-set NCU report is
`round16_parallel_up_tail_down_tile32_bf16_gpu5_ncu.ncu-rep`:

| NCU metric | Starting kernel | Final kernel | Change |
|:---|---:|---:|---:|
| Replay duration | 24.93 us | 23.68 us | -5.01% |
| Scheduler cycles with eligible warp | 18.89% | 19.51% | +0.62 points |
| Eligible warps / scheduler | 0.25 | 0.26 | +4.00% |
| Warp cycles / issued instruction | 10.58 | 10.23 | -3.31% |
| Barrier stall cycles / instruction | 4.36 | 4.04 | -7.34% |
| DRAM throughput | 17.76% | 18.69% | +0.93 points |
| Measured DRAM bandwidth | 433.20 GB/s | 456.05 GB/s | +5.27% |
| Compute throughput | 15.87% | 16.96% | +1.09 points |
| Achieved occupancy | 12.33% | 12.52% | +0.19 points |
| Registers / dynamic shared / spills | 96 / 166.91 KiB / 0 | 96 / 166.91 KiB / 0 | unchanged |

The tiled reduction adds shared-reduction instructions, so executed SASS
instructions rise from 2,313,940 to 2,358,094. The lower latency despite that
increase, plus unchanged low DRAM utilization, confirms that coalescing and
reduced atomic serialization are the benefit. Source samples attributed 547
of 1,519 final samples to barriers versus 1,020 of 1,922 initially. Sampling
counts are statistical; the direct stall-cycle and duration metrics above are
the selection criteria.

Rejected experiments remain out of the source:

| BF16 experiment | Raw p50 | Raw mean | Raw p95 | Decision |
|:---|---:|---:|---:|:---|
| Exact starting kernel | 20.896 us | 21.305 us | 23.616 us | baseline |
| LoRA-down ILP 8 to 16 | 20.799 us | 21.313 us | 23.904 us | reject: mean and tail regress |
| Remove only terminal up barrier | 20.832 us | 21.315 us | 23.840 us | reject: no stable mean gain |
| Parallel upper-warps up tail | 20.031 us | 20.591 us | 23.328 us | retain |
| Add 32-rank tiled down | 19.632 us | 20.062 us | 22.304 us | retain |
| Replace with 16-rank tiled down | 19.839 us | 20.378 us | 23.338 us | reject: extra input broadcasts |

The final sequential 500-warmup, 3,000-sample matrix preserves every fallback
shape and allocator footprint:

| Case | FP16 p50 | BF16 p50 | Peak | FP16 / BF16 max abs |
|:---|---:|---:|---:|---:|
| attention rank 64, M=1 | 36.86 us | 33.79 us | 24 KiB | 0.003836 / 0.008408 |
| attention rank 128, M=1 | 30.72 us | 30.72 us | 24 KiB | 0.008028 / 0.01488 |
| MLP up rank 128, M=1 | 35.84 us | 37.89 us | 64.5 KiB | 0.003901 / 0.01095 |
| MLP down rank 128, M=1 | 39.94 us | 40.96 us | 24 KiB | 0.007788 / 0.01902 |
| attention rank 128, M=8 | 53.25 us | 53.25 us | 192 KiB | 0.005325 / 0.01099 |
| attention rank 128, M=16 | 69.63 us | 68.61 us | 384 KiB | 0.005060 / 0.01186 |

Validation on the retained 32-rank source:

- `test_marlin_eora_rank128_attention_mega_kernel_matches_dense_update`:
  2 passed in 9.98 seconds, FP16 and BF16.
- Broader `tests/test_marlin_jit.py` run with only the redundant force-rebuild
  smoke test deselected: 32 passed, 1 deselected in 27.77 seconds.
- Compute Sanitizer memcheck on physical GPU 4 FP16 and physical GPU 5 BF16:
  `ERROR SUMMARY: 0 errors` for both final mega-kernels.
- `generate_kernels.py --check`: passed.
- `git diff --check`: passed.
- Final JIT fingerprints: FP16 `f768a7b973a2d7d1`, BF16
  `659178aeb9dbbb7f`.

All GPU commands began with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5`. Profiler reports,
SQLite exports, and benchmark JSON live under
`artifacts/eora_marlin_20260721/` and remain deliberately untracked.

### 2026-07-22: naive versus native/fused throughput and Nsight audit

This audit compares the current kernel source at revision `779878b2` against
the actual eager inference fallback, rather than comparing only successive
mega-kernel candidates. Final measurements used only PCI-ordered physical GPU
2 (`0000:64:00.0`, logical `cuda:0`, FP16) and physical GPU 3
(`0000:69:00.0`, logical `cuda:1`, BF16). Both runtime-probed devices are
PG506-230 `sm_80` GPUs with 124 SMs and 98,304 MiB. The software stack was
driver 610.43.02, `/root/vm314t/bin/python`, Torch 2.13.0+cu130, CUDA 13.0,
Nsight Systems 2024.6.2, and Nsight Compute 2025.3.1. GPUs 6 and 7 were not
used. Results collected before the switch to GPUs 2 and 3 were excluded from
the tables below.

The layer microbenchmark uses synthetic GPTQ W4A16 weights, group size 128,
symmetric quantization, no activation ordering, no bias, and an FP32 dense
LoRA reference. Timed runs used 500 warmups and 3,000 per-call CUDA-event
samples. Sustained throughput is the median of five independent 1,000-call
stream-timing passes. The final runs were pinned to idle NUMA-node-0 CPU 4
with `OMP_NUM_THREADS=1` because unrelated host activity raised the load
average above 30 and was visibly inserting false GPU queue gaps in unpinned
runs. `PYTHON_GIL=1` was retained to match the previous Torch 2.13 profiles.
For `M=1`, row/s is a per-layer token-row rate, not end-to-end model tokens/s.

The benchmark routes have precise meanings:

- `fallback` is ordinary Marlin followed by `Lora.apply`: a library LoRA-down
  `matmul`, its split-K reduction, and a library `addmm` LoRA-up/add.
- `addmm` is an implementation-path control. At full Marlin scope it is the
  same module route as `fallback`; at tail scope it reaches the same three
  library kernels through `apply_eora_marlin_fused_lora` and therefore adds
  Python/helper overhead. It is not a separate GPU kernel implementation.
- `cuda_up_add` is ordinary Marlin plus the library LoRA-down and native
  `lora_up_add`. Nsight shows that this still uses four full-path launches.
- `cooperative` is integrated native dispatch. Exact `M=1, K=N=4096,
  rank=128` uses one cooperative Marlin-EoRA mega-kernel. The other eligible
  shapes use a tightly dispatched two-launch `Marlin + fused EoRA tail`
  native bundle. At tail scope it is the standalone one-launch cooperative
  EoRA kernel.

The benchmark was extended during this audit to emit repeated aggregate
stream time, call/s, row/s, raw repeat samples, CPU affinity, and GIL state.
It also permits one profiled variant across multiple cases so Nsight Systems
can use repeated CUDA profiler capture ranges. Per-call p50/p95 remains in
the JSON, but sustained row/s below comes from the repeated aggregate timing
because it includes real eager launch-feed gaps.

#### Full Marlin plus EoRA throughput

| dtype | case | naive stream | integrated stream | speedup | naive row/s | integrated row/s | integrated p50 |
|:---|:---|---:|---:|---:|---:|---:|---:|
| FP16 | attention r64, M=1 | 146.24 us | 27.13 us | 5.39x | 6,838 | 36,863 | 35.84 us |
| FP16 | attention r128, M=1 | 148.66 us | 22.44 us | 6.62x | 6,727 | 44,559 | 31.74 us |
| FP16 | MLP up r128, M=1 | 156.31 us | 34.86 us | 4.48x | 6,397 | 28,687 | 39.94 us |
| FP16 | MLP down r128, M=1 | 153.87 us | 36.82 us | 4.18x | 6,499 | 27,159 | 46.08 us |
| FP16 | attention r128, M=8 | 157.48 us | 51.76 us | 3.04x | 50,799 | 154,565 | 62.46 us |
| FP16 | attention r128, M=16 | 158.13 us | 72.79 us | 2.17x | 101,186 | 219,820 | 80.90 us |
| BF16 | attention r64, M=1 | 158.23 us | 27.80 us | 5.69x | 6,320 | 35,968 | 34.82 us |
| BF16 | attention r128, M=1 | 151.71 us | 21.46 us | 7.07x | 6,592 | 46,596 | 30.72 us |
| BF16 | MLP up r128, M=1 | 150.08 us | 36.67 us | 4.09x | 6,663 | 27,271 | 43.01 us |
| BF16 | MLP down r128, M=1 | 152.58 us | 38.58 us | 3.96x | 6,554 | 25,922 | 48.13 us |
| BF16 | attention r128, M=8 | 162.66 us | 51.36 us | 3.17x | 49,183 | 155,773 | 61.44 us |
| BF16 | attention r128, M=16 | 162.80 us | 72.48 us | 2.25x | 98,281 | 220,754 | 69.63 us |

The target route comparison makes clear that native up/add alone is not the
source of the win:

| Target full path | FP16 stream / call/s | BF16 stream / call/s |
|:---|---:|---:|
| Naive Marlin + library LoRA | 148.66 us / 6,727 | 151.71 us / 6,592 |
| Full-scope `addmm` control | 159.63 us / 6,265 | 149.71 us / 6,680 |
| Marlin + native CUDA up/add | 162.15 us / 6,167 | 161.94 us / 6,175 |
| Integrated one-launch mega | 22.44 us / 44,559 | 21.46 us / 46,596 |

The `addmm` control and fallback execute the same full module path, so their
small difference is run-order noise. The native up/add path is slower than
the library baseline for every tested FP16 shape and all but a statistically
equivalent BF16 prefill result. It should remain a fallback capability, not
the preferred dispatch on these `sm_80` shapes.

#### Standalone EoRA tail throughput and allocation

| dtype | case | library tail | native up/add tail | cooperative tail | cooperative speedup | library / cooperative peak |
|:---|:---|---:|---:|---:|---:|---:|
| FP16 | attention r64, M=1 | 55.99 us | 86.65 us | 51.87 us | 1.08x | 0.5 / 0 KiB |
| FP16 | attention r128, M=1 | 55.15 us | 80.74 us | 45.78 us | 1.20x | 0.5 / 0 KiB |
| FP16 | MLP up r128, M=1 | 53.93 us | 80.37 us | 45.76 us | 1.18x | 0.5 / 0 KiB |
| FP16 | MLP down r128, M=1 | 59.59 us | 88.93 us | 49.80 us | 1.20x | 0.5 / 0 KiB |
| FP16 | attention r128, M=8 | 58.61 us | 85.92 us | 47.61 us | 1.23x | 2 / 0 KiB |
| FP16 | attention r128, M=16 | 60.90 us | 82.52 us | 58.21 us | 1.05x | 4 / 0 KiB |
| BF16 | attention r64, M=1 | 54.65 us | 81.82 us | 55.86 us | 0.98x | 0.5 / 0 KiB |
| BF16 | attention r128, M=1 | 64.13 us | 86.71 us | 46.12 us | 1.39x | 0.5 / 0 KiB |
| BF16 | MLP up r128, M=1 | 57.96 us | 87.34 us | 48.31 us | 1.20x | 0.5 / 0 KiB |
| BF16 | MLP down r128, M=1 | 61.19 us | 85.06 us | 46.90 us | 1.30x | 0.5 / 0 KiB |
| BF16 | attention r128, M=8 | 61.21 us | 83.59 us | 46.76 us | 1.31x | 2 / 0 KiB |
| BF16 | attention r128, M=16 | 58.36 us | 83.74 us | 53.14 us | 1.10x | 4 / 0 KiB |

The standalone cooperative tail provides modest eager gains and eliminates
the measured transient LoRA-down allocation. Its BF16 rank-64 result is a
2.2% regression and should be treated as parity. The much larger full-path
gain comes from integrated native dispatch and removal of Marlin-to-LoRA
launch gaps, not from replacing only the up projection.

Full-call incremental allocated-memory peaks are unchanged between naive and
integrated routes: 24 KiB for attention/MLP-down decode, 64.5 KiB for MLP-up,
192 KiB at M=8, and 384 KiB at M=16. These are allocated-byte deltas after
warmup, not CUDA reserved-memory totals. All variants matched the dense FP32
LoRA update. The largest full-path absolute errors were 0.008028 FP16 and
0.01902 BF16; the largest isolated-tail errors were 0.001932 FP16 and
0.007808 BF16.

#### Nsight Systems launch and gap attribution

The focused BF16 target captures contain 200 forwards after 500 warmups. The
host NVTX values below include Nsight injection overhead and are not used as
the normal-run latency result. Kernel work, GPU span, and gaps come directly
from correlated CUDA runtime/kernel records inside each NVTX range.

| BF16 target route | launches | summed kernel p50 | GPU span p50 | internal gaps p50 | host NVTX p50 | between-forward GPU gap |
|:---|---:|---:|---:|---:|---:|---:|
| Naive Marlin + library LoRA | 4 | 30.048 us | 92.607 us | 62.527 us | 206.013 us | 118.176 us |
| Marlin + native up/add | 4 | 38.368 us | 130.687 us | 92.159 us | 226.361 us | 98.559 us |
| Integrated mega | 1 | 19.936 us | 19.936 us | 0 us | 60.475 us | 43.168 us |

The naive four launches are Marlin (15.136 us p50), LoRA-down GEMV
(4.640 us), its split-K reduction (3.008 us), and LoRA-up/add GEMV
(7.232 us). Native up/add replaces the 7.232 us library up kernel with a
15.744 us custom kernel while leaving four launches, explaining its
regression. The mega-kernel does more than erase 62.527 us of internal idle
time: its 19.936 us raw duration is also 33.7% below the naive 30.048 us sum
because intermediate traffic and redundant work are removed.

The repeated all-shape BF16 captures show which routes are true mega-kernels
and which are native two-launch bundles:

| case | naive launches / span / gaps | integrated launches / span / gaps |
|:---|---:|---:|
| attention r64, M=1 | 4 / 89.440 / 61.920 us | 2 / 25.344 / 1.376 us |
| attention r128, M=1 | 4 / 90.687 / 60.479 us | 1 / 19.936 / 0 us |
| MLP up r128, M=1 | 4 / 92.608 / 53.856 us | 2 / 35.456 / 1.376 us |
| MLP down r128, M=1 | 3 / 96.896 / 60.575 us | 2 / 39.072 / 1.344 us |
| attention r128, M=8 | 3 / 61.216 / 36.479 us | 2 / 53.600 / 1.344 us |
| attention r128, M=16 | 3 / 60.928 / 33.984 us | 2 / 70.592 / 1.376 us |

For prefill M=16, the integrated two-kernel route has more raw GPU span than
the library route, but its normal eager throughput is still 2.25x higher
because native dispatch removes much larger host/launch gaps. This supports
the existing CUDA-graph gate: once graph replay removes launch overhead, the
library kernels can have less raw GPU work for larger M.

The FP16 GPU-2 target capture independently measured one launch per forward
and a 19.552 us raw-kernel p50 (20.014 us mean), confirming that an earlier
un-pinned 28 us stream result was host scheduling noise rather than a kernel
regression.

#### Nsight Compute status of the final BF16 mega-kernel

The full-set one-kernel replay on physical GPU 3 recorded:

| metric | final value |
|:---|---:|
| NCU replay duration | 23.424 us |
| Grid / block | 124 CTAs / 256 threads |
| Registers per thread | 96 |
| Dynamic shared memory | 166.912 KiB |
| Shared-memory residency limit | 1 block/SM |
| Achieved occupancy | 12.53% |
| Local loads / stores and spilling requests | 0 / 0 / 0 |
| DRAM bandwidth / peak throughput | 461.03 GB/s / 18.86% |
| SM throughput | 17.14% |
| Eligible warps per scheduler | 0.263 |
| Warp cycles per issued instruction | 10.231 |
| Barrier stall cycles per instruction | 4.137 |
| Long-scoreboard stall cycles per instruction | 1.670 |
| Executed SASS instructions | 2,358,664 |

The one-launch objective is achieved, and the kernel remains neither DRAM-
nor compute-saturated. Shared-memory capacity fixes residency at one CTA per
SM; low eligible-warp rate and barrier stalls remain the first place to look
for incremental gains. Any next mega-kernel change should target phase-tail
imbalance or useful work between existing grid barriers without increasing
registers/shared memory, adding atomics, or reintroducing a launch.

Final validation used the same GPU-2/3 visibility. The focused integrated
dispatch and dense-update selection in `tests/test_marlin_jit.py` passed 7
tests with 26 deselected in 16.61 seconds. `ruff check` on the benchmark and
`git diff --check` also passed.

Final benchmark JSON is under `artifacts/eora_marlin_20260722_gpu23/` in
`final_pinned_cpu4_marlin_all_{fp16_gpu2,bf16_gpu3}_w500_i3000_t1000x5.json`
and `final_pinned_cpu4_tail_all_{fp16_gpu2,bf16_gpu3}_w500_i3000_t1000x5.json`.
Focused target timelines are
`nsys_full_r128_{naive,native_upadd,mega}_bf16_gpu3.nsys-rep`; standalone-tail
timelines are `nsys_tail_r128_*_bf16_gpu3.nsys-rep`; repeated all-shape
captures are `nsys_full_all_shapes_{naive,integrated}_bf16_gpu3.[1-6].nsys-rep`.
The FP16 target timeline is `nsys_full_r128_mega_fp16_gpu2.nsys-rep`, and the
full-counter report is `ncu_full_r128_mega_bf16_gpu3.ncu-rep`. SQLite exports
and all profiler artifacts remain deliberately untracked.

### 2026-07-22: overlap LoRA-down with Marlin role CTAs

This pass used only PCI-ordered physical GPUs 2 and 3. Every GPU command set
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=2,3`; no test or
profile used GPUs 4-7. The interpreter was `/root/vm314t/bin/python` with
PyTorch `2.13.0+cu130`.

| Physical GPU | Logical device | PCI bus | dtype | GPU | CC | SMs | Memory |
|---:|---:|:---|:---|:---|:---:|---:|---:|
| 2 | 0 | `0000:64:00.0` | FP16 | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 3 | 1 | `0000:69:00.0` | BF16 | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |

#### Bottleneck and retained design

The prior 124-CTA kernel waited for Marlin to finish before every CTA started
LoRA-down. It then cleared 128 FP32 workspace values and performed 3,968
cross-CTA `atomicAdd` operations. Nsight Compute source correlation confirmed
that the first cooperative join dominated the fused tail: the baseline report
attributed 585 PC samples to barrier stalls, 304 to long scoreboard, and 179
to wait. The first post-Marlin grid join accounted for 298 samples and the
second for 97.

For this fixed `M=1, K=N=4096, rank=128` specialization, the exact Marlin
stripe schedule needs 114 CTAs. The retained schedule assigns CTAs 114-121 to
LoRA-down before the first grid join. Each of the eight workers owns one of
two K slices for a 32-rank tile and keeps its reduced 32-value partial in
shared memory. After Marlin completes, the workers publish disjoint partials
to the now-dead `C_tmp`, then all 122 CTAs run the existing parallel LoRA-up
phase. The up phase sums the two partials per rank. This keeps the one-launch
contract, adds no allocation, and removes both workspace initialization and
all LoRA-down atomics.

Reducing the launch from 124 to 122 CTAs removes two blocks with no Marlin or
LoRA-down role. Because 4,096 columns require 128 32-column output tiles, the
otherwise-idle upper warps in blocks 0-5 compute the six remaining tiles.
The runtime gate remains restricted to tested 124-SM `sm_80` devices with
cooperative-launch support; other shapes and devices keep the existing path.

#### Nsight Systems latency

Each raw result below is an exact 1,000-launch capture after warmup. These
kernel durations avoid Python and host-launch noise.

| dtype | implementation | grid | p50 | mean | p95 |
|:---|:---|---:|---:|---:|---:|
| FP16 | committed atomic baseline | 124 | 19.552 us | 20.014 us | 22.624 us |
| FP16 | role-specialized down | 124 | 18.848 us | 19.193 us | 21.472 us |
| FP16 | final role-specialized down | 122 | 18.784 us | 19.164 us | 21.311 us |
| BF16 | committed atomic baseline | 124 | 19.936 us | 20.368 us | 22.880 us |
| BF16 | role-specialized down | 124 | 19.231 us | 19.596 us | 21.951 us |
| BF16 | final role-specialized down | 122 | 19.168 us | 19.538 us | 21.792 us |

Relative to the committed baseline, the final FP16 p50/mean/p95 improve by
3.93%/4.25%/5.80%; BF16 improves by 3.85%/4.07%/4.76%. The final kernel uses
one `122 x 256` cooperative launch, 166.912 KiB dynamic shared memory, 95
FP16 or 96 BF16 registers per thread, and no local memory.

A matched normal benchmark alternated the exact cached baseline binary with
the new source under the same host load:

| dtype | exact baseline | final | latency change | throughput change |
|:---|---:|---:|---:|---:|
| FP16 | 25.00 us / 39,998 calls/s | 24.05 us / 41,581 calls/s | -3.8% | +4.0% |
| BF16 | 24.13 us / 41,447 calls/s | 22.67 us / 44,118 calls/s | -6.1% | +6.4% |

Allocator-visible peak remains 24 KiB. The maximum absolute error against
the dense update was 0.005573 FP16 and 0.01007 BF16.

#### Rejected candidates

| candidate | measured result | disposition |
|:---|:---|:---|
| Let all CTAs overlap LoRA-down into disjoint workspace | Correct only after accounting for live `C_tmp` aliasing; sustained matched runs regressed from Marlin/down contention | Rejected |
| Add 128 transient floats, a persistent argument, or a lock-based tail | No durable latency gain and added storage or synchronization complexity | Rejected |
| Launch only 114 CTAs | FP16 p50 19.808 us and mean 20.216 us versus 19.552/20.014 us baseline | Rejected |
| Warp-broadcast `eora_x` | FP16 p50 22.848 us, 16.9% slower | Rejected |
| Reduce dynamic shared memory to 49 KiB, 192 CTAs | FP16 p50 23.072 us | Rejected |
| Reduce dynamic shared memory to 49 KiB, 124 CTAs | 19.487 us p50 but 22.784 us p95; tie/noise over 1,000 launches | Reverted |

The key correctness lesson from the overlap experiments is that `C_tmp`
still contains live Marlin reduction fragments until the first cooperative
join. LoRA workers may retain partials in per-CTA shared memory during that
interval, but must not publish into `C_tmp` early.

#### Final Nsight Compute attribution

The final BF16 full-counter replay recorded 26.18 us; replay duration is
perturbed and is not directly comparable with the raw Nsight Systems values.

| metric | final 122-CTA kernel |
|:---|---:|
| Grid / block | 122 CTAs / 256 threads |
| Registers / dynamic shared memory | 96 / 166.912 KiB |
| Achieved occupancy | 12.27% |
| Local loads / stores and spills | 0 / 0 / 0 |
| DRAM / compute throughput | 16.89% / 14.43% |
| Executed SASS instructions | 2,214,702 |
| Eligible warps per scheduler | 0.216 |
| Warp cycles per issued instruction | 12.219 |
| Barrier / long-scoreboard / wait stalls | 5.931 / 1.576 / 1.225 cycles per instruction |

Executed instructions fall 6.10% from the 2,358,664-instruction baseline.
Source correlation attributes 422 of 606 fused-region barrier samples to the
first grid join, 139 to the up-phase block barrier, and 41 to the second grid
join. Twenty-one of 23 long-scoreboard samples land on the final output add.
The NCU barrier ratio rises because 114 CTAs now wait for eight down workers
or the Marlin tail while the total instruction denominator shrinks; the
unperturbed Nsight Systems captures nevertheless show the lower end-to-end
kernel latency. The next safe target is first-join tail balance without
increasing registers/shared memory or restoring atomics.

#### Validation and artifacts

- `pytest -q tests/test_marlin_jit.py -k 'integrated_eora or marlin_eora_rank128'`:
  7 passed, 26 deselected in 85.85 seconds, including clean FP16 and BF16 JIT
  rebuilds from the final source.
- `gptqmodel_ext/marlin/generate_kernels.py --check`: passed.
- Compute Sanitizer memcheck: `ERROR SUMMARY: 0 errors` for FP16 on GPU 2 and
  BF16 on GPU 3.
- Compute Sanitizer synccheck: `ERROR SUMMARY: 0 errors` for FP16 on GPU 2.
- Racecheck exited with 11 hazards in the pre-existing Marlin `cp.async`
  pipeline, including ordinary non-EoRA Marlin. It did not attribute a hazard
  to the new post-Marlin code, so racecheck is recorded as inconclusive rather
  than a pass.
- `git diff --check`: passed before final documentation cleanup.

Profiler outputs remain untracked under
`artifacts/eora_marlin_20260722_gpu23/`. The final raw timelines are
`nsys_idle8_grid122_{fp16_gpu2,bf16_gpu3}_1000.sqlite`; the final counter
report is `ncu_idle8_grid122_bf16_gpu3.ncu-rep`. Matched normal results are
`final_idle8_grid122_{fp16_gpu2,bf16_gpu3}_w500_i3000_t1000x5.json`, with
`paired_old_baseline_{fp16_gpu2,bf16_gpu3}_w500_i3000_t1000x5.json` as the
exact cached-source comparison.

### 2026-07-22: complete-K tiles and one-barrier lock-tail handoff

This continuation again used only PCI-ordered physical GPUs 2 and 3. Every
GPU command set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=2,3`; GPUs 4-7 were not used. The interpreter was
`/root/vm314t/bin/python`, with PyTorch `2.13.0+cu130`, CUDA 13.0, and
`TORCH_CUDA_ARCH_LIST=8.0`.

#### Phase-tail diagnosis

A bounded 40-launch timing probe carried per-CTA arrival timestamps through
the first cooperative join and copied them out only after the join. LoRA-down
was not the critical path: its latest worker arrived a median 2.048 us before
the latest Marlin CTA, with a 2.176 us mean lead and a worst observed 1.024 us
lag. Marlin CTA arrival spread was 5.120 us median, 5.709 us mean, and 8.192 us
maximum. CTA 3 was the last Marlin block in 26/40 launches and CTA 17 in 8/40,
identifying the boundary-crossing Marlin stripes as the dominant first-join
tail.

An attempted timestamp probe that wrote into Marlin locks before the join
deadlocked. Those locks remain live until every dependent Marlin stripe has
finished; unused lock slots may be reused, but active lock values may not be
overwritten for instrumentation or adapter scratch.

#### Retained schedule and handoff

The eight LoRA-down workers now each own one complete-K, 16-rank tile instead
of one of two K slices for a 32-rank tile. The thread mapping retains
coalesced rank-contiguous LoRA-A reads and the same per-thread K work. A
balanced 16-way shared reduction produces one FP32 value per rank.

Each worker converts its 16 totals to the compute dtype and publishes them in
the unused tail of the existing Marlin lock workspace before the cooperative
join. This fixed Marlin shape consumes lock indices 0-31; the 128 FP16/BF16
totals consume indices 32-95 when viewed as 32-bit lock words. The operator
already guarantees 124 lock words on the required 124-SM device. Compile-time
bounds now prove that the handoff fits. After the single grid join, LoRA-up
loads one value per rank directly from that lock tail.

This removes the former FP32 `C_tmp` publication, the second grid join, and
one addition per rank in LoRA-up. It adds no allocation and preserves the
single-launch contract. Materializing LoRA-down in FP16/BF16 matches the
ordinary two-GEMM LoRA path, whose down result has the compute dtype before
the up projection.

#### Raw Nsight Systems latency

The table uses matched 1,000-launch captures after 500 warmups. Values are raw
GPU kernel durations, independent of profiler-wrapped Python timing.

| dtype | implementation | p50 | mean | p95 | registers |
|:---|:---|---:|---:|---:|---:|
| FP16 | prior 32-rank/two-slice baseline | 18.784 us | 19.164 us | 21.311 us | 95 |
| FP16 | complete-K 16-rank, FP32 `C_tmp` handoff | 18.144 us | 18.552 us | 20.866 us | 94 |
| FP16 | final one-join lock-tail handoff | 16.928 us | 17.265 us | 19.423 us | 94 |
| BF16 | prior 32-rank/two-slice baseline | 19.168 us | 19.538 us | 21.792 us | 96 |
| BF16 | complete-K 16-rank, FP32 `C_tmp` handoff | 18.592 us | 19.008 us | 21.346 us | 96 |
| BF16 | final one-join lock-tail handoff | 16.960 us | 17.315 us | 19.616 us | 96 |

Relative to the preceding retained source, final p50/mean/p95 improve by
9.88%/9.91%/8.86% for FP16 and 11.52%/11.38%/9.99% for BF16. The final
kernel remains one `122 x 256` cooperative launch with 166.912 KiB dynamic
shared memory and zero local memory.

Longer non-profiler runs used 500 warmups, 3,000 event samples, and five
1,000-call throughput repeats. Under the current shared-host load, sustained
medians were 20.09 us / 49,779 calls/s for FP16 and 19.75 us / 50,641 calls/s
for BF16. Allocator-visible peak remained 24 KiB. Maximum absolute error
against the dense FP32 update remained 0.005573 FP16 and 0.01007 BF16.

#### Nsight Compute attribution

Full-counter replay perturbs absolute duration, so the matched raw Nsight
Systems captures above remain the performance decision. The counters explain
the improvement:

| metric | prior | 16-rank FP32 handoff | final lock-tail |
|:---|---:|---:|---:|
| Replay duration | 26.18 us | 25.15 us | 23.65 us |
| Executed SASS instructions | 2,214,702 | 2,177,993 | 2,166,213 |
| Eligible warps/scheduler | 0.216 | 0.230 | 0.250 |
| Warp cycles/issued instruction | 12.219 | 11.774 | 10.760 |
| Barrier stall cycles/instruction | 5.931 | 6.706 | 4.897 |
| Long-scoreboard stall cycles/instruction | 1.576 | 1.684 | 1.520 |
| Wait stall cycles/instruction | 1.225 | 1.228 | 1.092 |
| DRAM throughput | 16.89% | 17.58% | 18.72% |
| Compute throughput | 14.43% | 14.80% | 15.67% |

The final report still shows only one full wave, low eligible-warp rate, and
barrier stalls as the largest stall class. The phase probe shows that the next
high-confidence target is Marlin stripe-tail balance or a safe per-output
handoff; adding more LoRA-down throughput alone cannot shorten the current
critical path.

#### Rejected candidates in this pass

| candidate | FP16 raw result | reason rejected |
|:---|:---|:---|
| 128 aligned Marlin CTAs plus 8 down CTAs, 49 KiB shared memory | 20.480 us p50 / 21.000 us mean | Twelve second-wave CTAs and co-resident contention outweighed stripe alignment |
| Independent 128-thread cooperative groups for the two up tiles | 19.488 us p50 / 19.873 us mean | Group synchronization overhead exceeded any tail reduction |
| Four complete-K 32-rank down workers | 23.232 us p50 / 23.505 us mean | Doubling each worker's down work moved LoRA-down onto the critical path |
| FP16 Marlin cross-CTA reduction to enable scratch removal | 17.408 us p50 / 17.629 us mean | Regressed p50/mean and doubled max absolute error to 0.01159 |
| Half-warp shuffle down reduction | 39.680 us p50 / 39.810 us mean | Transposed mapping changed contiguous rank loads into 256-byte-strided K loads |

Directly storing the reduced value instead of round-tripping through an extra
shared slot was retained as a simplification. It held FP16 p50 at 16.928 us,
reduced mean from 17.278 to 17.265 us, and reduced p95 from 19.487 to
19.423 us. BF16 remained effectively unchanged.

#### Validation and artifacts

- Focused integrated-dispatch and mega-kernel tests: 7 passed, 26 deselected
  for FP16 and BF16.
- Compute Sanitizer memcheck: zero errors for FP16 on physical GPU 2 and BF16
  on physical GPU 3.
- Compute Sanitizer synccheck: zero errors for FP16 on physical GPU 2.
- Both FP16 and BF16 final sources JIT-compiled successfully with the lock-tail
  bounds assertions.

Profiler artifacts remain untracked under
`artifacts/eora_marlin_20260722_gpu23/`. Raw final timelines are
`nsys_candidate_tile16_locktail_direct_{fp16_gpu2,bf16_gpu3}_1000.sqlite`;
the final BF16 counter report is
`ncu_candidate_tile16_locktail_direct_bf16_gpu3.ncu-rep`. Longer benchmark
JSON files are
`final_tile16_locktail_direct_{fp16_gpu2,bf16_gpu3}_w500_i3000_t1000x5.json`.

### 2026-07-22: output-owned Marlin lock wavefront

This continuation used only PCI-ordered physical GPUs 2 and 3. Every GPU
command set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=2,3`; GPUs 4-7 were not used. Physical GPU 2 was logical
`cuda:0` for FP16 and physical GPU 3 was logical `cuda:1` for BF16. Both are
124-SM NVIDIA PG506-230 `sm_80` devices with 98,304 MiB. The interpreter was
`/root/vm314t/bin/python`, with PyTorch `2.13.0+cu130` and
`TORCH_CUDA_ARCH_LIST=8.0`.

#### Stripe dependency diagnosis

For the fixed `M=1, K=N=4096` kernel, Marlin has 32 K tiles by 32 output
tiles, or 1,024 work tiles. The former 122-CTA cooperative grid gave the
generic Marlin scheduler nine tiles per stripe. Blocks 0-112 therefore owned
nine tiles, block 113 owned seven, and blocks 114-121 did no Marlin work. The
important detail was that 28 of the Marlin stripes crossed an output boundary.
Those CTAs had to complete a global reduction for one output before computing
their second segment, while as many as five CTAs serialized on an output lock.

An offline dependency model showed that redistributing the two short
eight-tile stripes among 114 Marlin CTAs could not remove this structure: all
placements retained 28 boundary crossers and a maximum lock-chain length of
five. The next candidate therefore changed the fused-only schedule rather
than perturbing the generic stripe start.

The retained kernel launches exactly one full 124-CTA cooperative wave:

- 116 CTAs own complete output slices and never cross an output boundary;
- eight CTAs retain one complete-K, 16-rank LoRA-down tile each;
- 20 output slices use four Marlin CTAs and 12 use three, covering all 1,024
  tiles exactly;
- from low K to high K, four-CTA slices receive `10/9/7/6` tiles and
  three-CTA slices receive `12/11/9` tiles.

Marlin visits each output lock from high K to low K. Giving the first lock
visitor the smallest K segment lets its global reduction overlap the next
contributor's tensor-core work. The lowest-K, final writer receives the most
work because it can hide that work behind the preceding lock chain. This
output-owned wavefront also lets the compiler remove the generic follow-on
slice machinery from the fused instantiation.

The specialization remains behind `MARLIN_EORA_FUSED`. Ordinary Marlin,
prefill, non-target shapes, other architectures, CPU execution, and fallback
dispatch retain the existing scheduler.

#### Raw Nsight Systems latency

All rows below are matched 1,000-launch captures after 500 warmups. They are
raw GPU kernel durations and exclude profiler-inflated Python/CUDA-event time.

| dtype | schedule | p50 | mean | p95 | registers |
|:---|:---|---:|---:|---:|---:|
| FP16 | prior 122-CTA crossing stripes | 16.928 us | 17.261 us | 19.392 us | 94 |
| FP16 | 124-CTA output-owned, equal tiles | 15.520 us | 16.022 us | 18.752 us | 78 |
| FP16 | final output-owned wavefront | 14.847 us | 15.272 us | 17.632 us | 78 |
| BF16 | prior 122-CTA crossing stripes | 16.960 us | 17.315 us | 19.616 us | 96 |
| BF16 | final output-owned wavefront | 14.976 us | 15.419 us | 17.823 us | 80 |

Relative to the preceding retained source, final p50/mean/p95 improve by
12.29%/11.52%/9.08% for FP16 and 11.70%/10.95%/9.14% for BF16. The kernel
remains one cooperative launch, now `124 x 256`, with 166.912 KiB dynamic
shared memory, one resident CTA per SM, and zero local-memory spills.

The longer shared-host runs used 500 warmups, 3,000 event samples, and five
1,000-call throughput repeats. Sustained medians were 24.16 us / 41,399 calls/s
for FP16 and 22.78 us / 43,906 calls/s for BF16. Those eager measurements were
noisier than the raw timelines under current host load and are recorded for
reproducibility rather than used for the schedule decision. Allocator-visible
peak remained 24 KiB. Maximum absolute error against the dense FP32 update was
0.004135 for FP16 and 0.01007 for BF16.

#### Final Nsight Compute attribution

Full-counter replay perturbs absolute duration, so raw Nsight Systems remains
the performance gate. The matched BF16 counter reports nevertheless show the
compiler and scheduling effects:

| metric | prior | final wavefront |
|:---|---:|---:|
| Replay duration | 23.65 us | 23.33 us |
| Executed SASS instructions | 2,166,213 | 1,975,739 |
| Registers/thread | 96 | 80 |
| Local spilling requests | 0 | 0 |
| Achieved occupancy | 12.01% | 13.20% |
| Waves/SM | 0.98 | 1.00 |
| Eligible warps/scheduler | 0.25 | 0.23 |
| Warp cycles/issued instruction | 10.76 | 12.20 |
| Barrier stall cycles/instruction | 4.897 | 5.925 |
| Long-scoreboard stall cycles/instruction | 1.520 | 1.671 |
| Wait stall cycles/instruction | 1.092 | 0.965 |
| DRAM throughput | 18.72% | 18.98% |
| Compute throughput | 15.67% | 14.53% |

Executed instructions fall 8.79%, registers fall 16.67%, and replay duration
falls 1.35%. The normalized barrier ratio rises because the instruction
denominator is smaller and all 124 CTAs now participate in the exact full
wave. Source correlation attributes 575 of 797 barrier samples to the first
grid join. The unperturbed raw timelines demonstrate that the shorter,
better-balanced schedule wins despite that replay ratio.

#### Post-wavefront phase probe

A temporary 100-launch FP16 probe captured the global timer immediately before
the first join. It stored only quantized max/min timestamps in lock indices
96-103, which are beyond both the live Marlin locks (0-31) and the LoRA-down
handoff (32-95). The probe was removed from the retained source.

LoRA-down minus Marlin latest arrival was 0.000 us median, +0.184 us mean,
-1.024 us minimum, and +3.072 us maximum; down was later in 18/100 launches.
Marlin arrival spread was 5.120 us median, 5.243 us mean, and 8.192 us maximum.
Down-worker spread was 1.024 us median, 1.587 us mean, and 6.144 us maximum.
The latest Marlin IDs were 80, 104, 107, 110, and 113: the expected 12-tile
lowest-K final writers. Down IDs 122 and 123 were most often the latest adapter
workers. The two phases are now balanced closely enough that optimizing only
one side is unlikely to reduce median latency.

An earlier version of this probe wrote per-CTA timestamps into `C_tmp` before
the join. Late Marlin global reductions still owned that scratch and corrupted
some records. Phase probes must use storage proven dead or unused at the exact
instrumentation boundary, not merely storage that is dead after the barrier.

#### Rejected candidates in this pass

| candidate | FP16 raw result | reason rejected |
|:---|:---|:---|
| Reposition two 8-tile stripes among 114 CTAs | model-only | Retained 28 boundary crossers and five-contributor lock chains for every placement |
| More aggressive `11/9/7/5` and `13/11/8` wavefront | 15.168 us p50 / 15.547 us mean / 17.856 us p95 | Crossed the overlap optimum; only the worst single outlier improved |
| Same-rank half-warp shuffle before an eight-value shared reduction | 15.184 us p50 / 15.564 us mean / 17.824 us p95 | Shuffle, predicate, and store overhead exceeded the eight removed shared loads |

The rejected shuffle retained coalesced LoRA-A loads; it is distinct from the
earlier rejected half-warp mapping that introduced 256-byte-strided loads.

#### Validation and artifacts

- `pytest -q tests/test_marlin_jit.py -k 'integrated_eora or marlin_eora_rank128'`:
  7 passed, 26 deselected for the final FP16/BF16 source.
- `gptqmodel_ext/marlin/generate_kernels.py --check`: passed.
- Compute Sanitizer memcheck: zero errors for FP16 on physical GPU 2 and BF16
  on physical GPU 3.
- Compute Sanitizer synccheck: zero errors for FP16 on physical GPU 2.
- `git diff --check`: passed.

Profiler outputs remain untracked under
`artifacts/eora_marlin_20260722_gpu23/`. Final raw timelines are
`nsys_candidate_wavefront_10_9_7_6_12_11_9_{fp16_gpu2,bf16_gpu3}_1000.sqlite`.
The final BF16 counter report is
`ncu_candidate_wavefront_10_9_7_6_12_11_9_bf16_gpu3.ncu-rep`. Longer benchmark
JSON files are
`final_output_owned_wavefront_{fp16_gpu2,bf16_gpu3}_w500_i3000_t1000x5.json`.
