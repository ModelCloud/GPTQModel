# Qwen3.8-Flash-Next P32 inference on H100: expert phase 2

This phase accelerates the exact Qwen3.8-Flash-Next routed-expert geometry. It
does not change quantization, checkpoint bytes, P32 contiguous-window storage,
codebooks, or decoded arithmetic. The production route is deliberately narrow:
physical NVIDIA H100, FP16, M1 through M16, and the exact expert shapes below.

## Exact model geometry

The model has 512 routed experts, selects 10 experts per token, and also has a
shared expert. Each expert uses these projections:

| Projection | Weight `[N, K]` |
|---|---:|
| Gate | `[640, 2560]` |
| Up | `[640, 2560]` |
| Down | `[2560, 640]` |

The 640-wide dimension satisfies P32's K16/N16 contract but not the current
Hopper TMA/WGMMA kernel's stricter K256/N256 geometry. Padding to 768 would add
work and storage traffic. Instead, the runtime uses the existing exact P32
segmented implementation compiled natively for SM90, groups gate and up into
one launch, and routes down directly without materializing the generic planar
payload.

## Accepted schedule and safety gates

Gate/up uses `(32, 32)` independent split counts at W2, W2.5, W3, and W3.5 for
M1/M2/M4/M8/M16. The split is intentionally uniform across rates. The older
SM80 compact split-40 specialization corrupted the second grouped child when
compiled for SM90 at W3, while the generic split-32 path was exact at every
tested rate and row count. That invalid specialization is not exposed on H100.

Production dispatch fails closed unless all of these conditions hold:

- physical device name is `NVIDIA H100` with compute capability 9.0;
- input is FP16 and the checkpoint is V2B2 P32 with transition bits 4--7;
- M is 1, 2, 4, 8, or 16 for grouped gate/up, and no larger than 16 for direct
  gate/up or down;
- grouped dimensions are exactly `K=2560`, widths `(640, 640)`, and split
  counts `(32, 32)`; direct dimensions are `(K,N)=(2560,640)` or `(640,2560)`.

Other shapes retain their previous route. CUDA graph replay executes only the
captured device work and requires no host-side synchronization or autotuning.

## H100 speed and accuracy

The committed benchmark compares the prior planar V2B2-P32 CUDA decoder with
the native narrow SM90 route. It uses CUDA Graph replay and CUDA events with 30
warmups, 300 samples, and 50 replays per sample in a
candidate/control/candidate sandwich on an idle 132-SM H100.

| Rate | Gate/up M1 | M2 | M4 | M8 | M16 |
|---|---:|---:|---:|---:|---:|
| W2 | 2.653x | 3.211x | 2.993x | 2.752x | 7.877x |
| W2.5 | 2.730x | 8.752x | 8.104x | 7.926x | 8.020x |
| W3 | 8.363x | 8.981x | 8.224x | 8.005x | 7.940x |
| W3.5 | 8.048x | 7.988x | 8.136x | 7.880x | 7.923x |

| Rate | Down M1 | M2 | M4 | M8 | M16 |
|---|---:|---:|---:|---:|---:|
| W2 | 2.339x | 3.072x | 3.036x | 2.680x | 7.666x |
| W2.5 | 2.527x | 9.108x | 8.610x | 8.092x | 7.700x |
| W3 | 7.327x | 8.704x | 8.451x | 8.110x | 7.712x |
| W3.5 | 7.978x | 8.947x | 8.700x | 7.969x | 7.703x |

Across all 40 cells, the geometric-mean speedup is **6.235x**, the minimum is
**2.339x**, and the maximum is **9.108x**. Gate/up alone is **6.267x** and down
is **6.203x** by geometric mean. Every cell was bitwise repeatable.

The largest candidate-versus-planar errors were `1.7731e-7` mean and
`1.4305e-6` maximum for gate/up, and `6.2540e-8` mean and `5.9605e-7` maximum
for down. These are far below the local `4e-3` mean and `0.046875` maximum
gates.

The sampled full-K oracle disables TF32 and evaluates FP32 and FP64 references.
Worst FP64 mean errors were `9.6976e-8` planar versus `1.5557e-7` candidate for
gate/up, and `3.8208e-8` planar versus `5.3143e-8` candidate for down. The
candidate changes operation order and is slightly farther from FP64 than the
planar path, but the absolute drift is sub-micro and also slightly better than
the sampled FP32 anchor in the worst aggregate. No accuracy gate was relaxed.

## Nsight Compute and SASS evidence

A matched W3 M1 gate/up profile compares one prior planar child kernel with the
new grouped kernel. Nsight instrumentation changes absolute timing, so CUDA
Graph event timing above remains authoritative.

| Metric | Planar split-K child | Native grouped SM90 |
|---|---:|---:|
| Kernel duration | 57.440 us | 9.632 us |
| Grid | 800 CTAs | 192 CTAs |
| Block size | 256 | 128 |
| Executed instructions | 8.218M | 1.733M |
| Registers/thread | 48 | 56 |
| Shared memory/block | 15.92 KiB | 7.87 KiB |
| Waves/SM | 1.21 | 0.16 |
| Active warps | 53.86% | 9.15% |
| Local spill requests | 0 | 0 |

The extension contains a native `sm_90` cubin. Its selected specialization has
56 registers, 7,872 bytes of shared memory, zero local storage, and no local
load/store instructions. Static SASS inspection counted 1,856 instructions,
including 64 FP32 FFMAs; there are no HMMA/HGMMA instructions. The speedup is
therefore from eliminating redundant planar decode/reduction work, grouping
gate/up, shrinking the grid and instruction count, and avoiding extra launches,
not from claiming tensor-core throughput that this narrow specialization does
not use.

Profiler reports:

- `/root/qvq-profiler-artifacts/qwen38-flash-next-phase2/w3-gate-up-m1-planar.ncu-rep`
- `/root/qvq-profiler-artifacts/qwen38-flash-next-phase2/w3-gate-up-m1-narrow.ncu-rep`

## Reproduction

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<H100-UUID> \
  python scripts/benchmark_qvq_p32_qwen38_flash_next_experts_h100.py \
  --warmup 30 --samples 300 --replays-per-sample 50 \
  --output /root/qvq-results/qwen38-flash-next-experts-h100-phase2-production.json
```
