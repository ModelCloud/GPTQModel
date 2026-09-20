# Qwen3.8-Flash-Next P32 inference on H100: attention phase 1

This phase starts the QVQ P32 contiguous-window inference work for the exact
Qwen3.8-Flash-Next projection geometry.  It does not change quantization,
checkpoint bytes, the P32 representation, or post-quantization arithmetic.
It adds physical-H100 schedules for the hybrid attention projections and a
deterministic ordered split-K path for the full-attention output projection.

## Exact model geometry

The local `/monster/data/model/Qwen3.8-Flash-Next` checkpoint supplies these
attention weights:

| Projection | Weight `[N, K]` |
|---|---:|
| Full-attention Q | `[12288, 2560]` |
| Full-attention K/V | `[512, 2560]` each |
| Full-attention O | `[2560, 6144]` |
| Linear-attention QKV | `[10240, 2560]` |
| Linear-attention Z | `[6144, 2560]` |
| Linear-attention output | `[2560, 6144]` |

The model is a 512-expert MoE.  Its expert MLP width is 640, which is not a
multiple of the current Hopper P32 TMA/WGMMA kernel's 256-column tile.  This
phase therefore covers attention only; expert MLP support is a separate
kernel-shape problem rather than an implicit padded benchmark.

## Accepted schedules

The grouped kernels retain independent deterministic FP32 reductions for each
child rather than deriving a split from the concatenated output width:

| Group | W2 | W2.5 | W3 | W3.5 |
|---|---:|---:|---:|---:|
| Full Q/K/V `(12288, 512, 512)` | `(1,1,1)` | `(5,10,10)` | `(1,1,1)` | `(2,10,10)` |
| Linear QKV/Z `(10240, 6144)` | `(1,1)` | `(1,1)` | `(2,2)` | `(2,2)` |

The full-attention O projection uses ordered split counts 12, 6, 12, and 24
for W2, W2.5, W3, and W3.5 respectively.  The policy is fail-closed to a
132-SM H100 at M1/M2/M4/M8/M16.  It deliberately does not select the atomic
split path: each partial is accumulated in FP32 and a second kernel reduces
partials left-to-right, making graph replay repeatable.

An experimental W2.5 linear QKV/Z split `(5,5)` regressed the robust graph
replay benchmark by 5.9%, so it was rejected and remains `(1,1)`.

## H100 speed and accuracy

The committed benchmark uses CUDA Graph replay and CUDA events with 30
warmups, 300 samples, and 50 replays per sample.  Each cell is a
candidate/control/candidate sandwich.  The device was an idle physical
132-SM NVIDIA H100, and the tested O projection was `K=6144, N=2560`.

| Rate | M1 | M2 | M4 | M8 | M16 |
|---|---:|---:|---:|---:|---:|
| W2 | 2.816x | 2.840x | 2.842x | 2.820x | 3.202x |
| W2.5 | 2.612x | 2.599x | 2.624x | 2.606x | 2.918x |
| W3 | 2.652x | 2.687x | 2.681x | 2.673x | 3.003x |
| W3.5 | 2.473x | 2.507x | 2.481x | 2.502x | 2.782x |

The 20-cell geometric-mean speedup is **2.710x**; the minimum is **2.473x**.
All cells were bitwise repeatable.  Candidate-versus-split-1 mean absolute
error was `7.65e-6` to `8.78e-6`, and maximum absolute error was `3.67e-5` to
`5.53e-5`, comfortably below the local `4e-3` mean and `0.046875` maximum
gates.

A separate sampled FP64 oracle covered 256 output columns and 16 rows over the
complete K dimension.  TF32 was disabled and FP32 matmul precision was
`highest`.

| Rate | Split-1 MAE vs FP64 | Ordered MAE vs FP64 | Ordered closer / 4096 |
|---|---:|---:|---:|
| W2 | `8.898e-6` | `8.043e-7` | 3945 |
| W2.5 | `8.781e-6` | `1.520e-6` | 3803 |
| W3 | `8.986e-6` | `8.131e-7` | 3960 |
| W3.5 | `9.011e-6` | `4.341e-7` | 3995 |

The optimized path is therefore faster and closer to the FP64 oracle.  This
is an accuracy improvement from a deterministic reduction order, not a speed
trade for a looser oracle.

## Nsight Compute and SASS evidence

Matched W3 profiles compare split 1 with ordered split 12:

| Metric | Split 1 | Ordered split 12 |
|---|---:|---:|
| Main-kernel duration | 47.712 us | 16.064 us |
| Grid | 40 CTAs | 480 CTAs |
| Registers/thread | 60 | 59 |
| Achieved occupancy | 7.806% | 26.007% |
| Eligible warps/scheduler | 0.376 | 0.864 |
| L1 throughput | 14.37% | 45.52% |
| L2 throughput | 4.76% | 15.63% |

The ordered reducer measured 4.128 us under profiler instrumentation.  SASS
inspection found no local-memory spills.  The main kernel executes more
instructions after splitting, but latency falls because 480 independent CTAs
fill the H100 instead of leaving most SM capacity idle.  CUDA-event graph
timing is authoritative for the end-to-end numbers above; Nsight timing is
used to explain the mechanism.

## Reproduction

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<H100-UUID> \
  python scripts/benchmark_qvq_p32_qwen38_flash_next_h100.py \
  --output /root/qvq-results/qwen38-flash-next-attention-output-phase1.json
```

Profiler reports used for the W3 comparison:

- `/root/qvq-profiler-artifacts/qwen38-flash-next-phase1/w3-o-split1.ncu-rep`
- `/root/qvq-profiler-artifacts/qwen38-flash-next-phase1/w3-o-ordered-split12.ncu-rep`
