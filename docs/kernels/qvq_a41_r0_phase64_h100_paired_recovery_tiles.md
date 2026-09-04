# Phase 64: H100 paired recovery output tiles

Phase 64 shares the exact recovery-high dependency tree between two output
tiles that differ only in the final 32-point Hadamard bit. The physical
NVIDIA H100 benefits once all 16 logical rows are present: the isolated fused
recovery/precondition chain improves **1.0900x** at M16, and the complete
Llama 3.2 1B MLP improves **1.0187x geometrically across W2--W3.5 at M16**.

The smaller row counts do not have enough blocks to absorb the additional
per-thread work. Production therefore selects the paired kernel only for
M16. M1/M2/M4/M8 retain the exact Phase-63 kernel.

## Exact paired-tree math

The Phase-63 selected-output tree evaluates one output tile `t` from 32
low-stage values:

```text
for stage = 0..4:
    sign = -1 if bit(stage, t) else +1
    values[j] = R(values[2*j] + sign * values[2*j + 1])
```

Here `R` is the existing operation that rounds finite values to FP16 while
retaining an overflowing intermediate in FP32. Output tiles `t` and `t+16`
have identical signs for stages zero through three and opposite signs only
at stage four. Phase 64 therefore evaluates:

```text
values = [x0, x1, ..., x31]

for stage = 0..3:
    sign = -1 if bit(stage, t) else +1
    values[j] = R(values[2*j] + sign * values[2*j + 1])

output(t)      = R(values[0] + values[1])
output(t + 16) = R(values[0] - values[1])
```

The transformation is byte-exact because it preserves the same balanced,
ascending-stage subtrees and the same rounding point at every node. For two
outputs, selected recovery changes from 64 loads and 62 rounded operations
to 32 loads and 32 rounded operations.

Each CTA carries both columns independently through the unchanged recovery
scale/bias, rounded FP16 SiLU, product, down scale/divisor, and packed-FP16
low transform. The final packed-FP16 high transform is unchanged.

## Launch and memory behavior

| Property | Phase 63 | Phase 64 M16 |
|:--|--:|--:|
| recovery/precondition middle blocks per row | 32 | **16** |
| threads per block | 256 | 256 |
| outputs per thread | 1 | **2** |
| selected-tree loads per two outputs | 64 | **32** |
| selected-tree rounded operations per two outputs | 62 | **32** |
| recovery FP32 workspace | `2*M*8192*4` bytes | unchanged |
| precondition FP16 workspace | `M*8192*2` bytes | unchanged |
| persistent VRAM added | 0 | 0 |

No checkpoint tensor, quantization result, grouped P32 payload, selector,
alternative bank, split-K schedule, or deterministic reduction order changes.

## Same-binary isolated sweep

Timing used 30 warmups, 300 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample after three spaced 0% utilization / 0 MiB H100 admission
samples. Both paths were compiled into one extension. Every output was bit
exact. `Better` compares the paired kernel directly with Phase 63.

| M | M/K/N gate/up x2 | Phase 63 | Paired tiles | vs Phase 63 | Better |
|--:|:--|--:|--:|--:|:--:|
| 1 | 1/2048/8192 x2 | 7.621 us | 7.923 us | 0.9620x | No |
| 2 | 2/2048/8192 x2 | 7.813 us | 8.001 us | 0.9765x | No |
| 4 | 4/2048/8192 x2 | 7.851 us | 8.151 us | 0.9632x | No |
| 8 | 8/2048/8192 x2 | 8.703 us | 8.636 us | 1.0079x | Yes |
| 16 | 16/2048/8192 x2 | 10.743 us | **9.856 us** | **1.0900x** | **Yes** |

M8 is too close to noise for production promotion. The M16-only gate keeps
the robust win and prevents every observed small-row regression.

## Nsight Compute and emitted SASS

Nsight Compute 2026.2.1 collected 19 hardware-counter replay passes on the
physical H100 for both M16 kernels. The profiler was launched with:

```text
ncu --profile-from-start off \
  --kernel-name regex:<exact-kernel-name> \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --section InstructionStats --section MemoryWorkloadAnalysis \
  -- python scripts/profile_qvq_phase64_paired_recovery_tiles.py \
  phase63|paired --m 16 --idle-memory-mib 0
```

| M16 metric | Phase 63 | Paired tiles | Change |
|:--|--:|--:|--:|
| grid | 32 x 16 blocks | 16 x 16 blocks | -50% |
| NCU duration | 6.98 us | **6.27 us** | **1.113x** |
| executed warp instructions | 2,595,361 | **1,597,009** | **-38.47%** |
| registers/thread | 42 | **40** | -2 |
| user static shared memory/block | 512 B | 1,024 B | +512 B |
| achieved occupancy | 43.06% | 23.86% | lower grid |
| active warps/scheduler | 6.80 | 3.80 | lower grid |
| eligible warps/scheduler | 1.56 | 0.63 | lower grid |
| warp cycles/issued instruction | 13.13 | **10.21** | -22.2% |
| DRAM throughput | 7.13% | 7.95% | +0.82 point |

CUDA `cuobjdump` independently reports zero stack/local bytes for both
kernels. The paired kernel contains 1,240 static instructions per CTA versus
1,088 for Phase 63, as expected because one CTA produces twice as many
outputs. Halving the grid reduces the dynamic instruction count despite the
larger CTA program. Profiler reports and full SASS remain outside Git under
`/root/qvq-profiler-artifacts/phase64-paired-recovery`.

## Complete Llama 3.2 1B MLP

Marlin and Machete execute W4 and are figurative latency/efficiency baselines,
not equal-rate comparisons. Effective TFLOP/s uses the dense-equivalent
gate/up/down operation count. `Better` is the strict median comparison with
the committed Phase-63 artifact. M1--M8 use unchanged production code, so
their sub-percent cross-run movements are telemetry rather than code deltas.

| W | M | M/K/N: gate/up x2; down | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 63 | Better |
|--:|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 1 | 1/2048/8192 x2; 1/8192/2048 | 44.209 | 2.277 | 0.656x | 1.133x | 1.0018x | Yes |
| 2 | 2 | 2/2048/8192 x2; 2/8192/2048 | 44.460 | 4.528 | 0.700x | 1.127x | 1.0022x | Yes |
| 2 | 4 | 4/2048/8192 x2; 4/8192/2048 | 44.804 | 8.987 | 0.701x | 1.130x | 1.0039x | Yes |
| 2 | 8 | 8/2048/8192 x2; 8/8192/2048 | 45.890 | 17.549 | 0.638x | 1.106x | 1.0002x | Yes |
| 2 | 16 | 16/2048/8192 x2; 16/8192/2048 | 47.107 | 34.191 | 0.690x | 1.080x | **1.0186x** | **Yes** |
| 2.5 | 1 | 1/2048/8192 x2; 1/8192/2048 | 45.489 | 2.213 | 0.637x | 1.101x | 0.9995x | No |
| 2.5 | 2 | 2/2048/8192 x2; 2/8192/2048 | 45.700 | 4.405 | 0.681x | 1.096x | 1.0011x | Yes |
| 2.5 | 4 | 4/2048/8192 x2; 4/8192/2048 | 46.423 | 8.674 | 0.676x | 1.091x | 0.9977x | No |
| 2.5 | 8 | 8/2048/8192 x2; 8/8192/2048 | 47.140 | 17.083 | 0.621x | 1.076x | 0.9991x | No |
| 2.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | 48.287 | 33.355 | 0.673x | 1.054x | **1.0188x** | **Yes** |
| 3 | 1 | 1/2048/8192 x2; 1/8192/2048 | 44.957 | 2.239 | 0.645x | 1.114x | 1.0016x | Yes |
| 3 | 2 | 2/2048/8192 x2; 2/8192/2048 | 45.324 | 4.442 | 0.686x | 1.106x | 0.9972x | No |
| 3 | 4 | 4/2048/8192 x2; 4/8192/2048 | 45.700 | 8.811 | 0.687x | 1.108x | 1.0006x | Yes |
| 3 | 8 | 8/2048/8192 x2; 8/8192/2048 | 46.645 | 17.265 | 0.627x | 1.088x | 0.9957x | No |
| 3 | 16 | 16/2048/8192 x2; 16/8192/2048 | 47.635 | 33.812 | 0.683x | 1.068x | **1.0183x** | **Yes** |
| 3.5 | 1 | 1/2048/8192 x2; 1/8192/2048 | 45.959 | 2.190 | 0.631x | 1.090x | 1.0010x | Yes |
| 3.5 | 2 | 2/2048/8192 x2; 2/8192/2048 | 46.392 | 4.340 | 0.671x | 1.080x | 1.0019x | Yes |
| 3.5 | 4 | 4/2048/8192 x2; 4/8192/2048 | 46.617 | 8.637 | 0.673x | 1.086x | 1.0039x | Yes |
| 3.5 | 8 | 8/2048/8192 x2; 8/8192/2048 | 47.294 | 17.028 | 0.619x | 1.073x | 1.0013x | Yes |
| 3.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | 48.619 | 33.127 | 0.669x | 1.046x | **1.0193x** | **Yes** |

Geometric means across all 20 cells are **1.0041x versus Phase 63**,
**2.9283x versus ordinary per-module QVQ**, **1.0924x versus Machete W4**,
and **0.6626x versus Marlin W4**. The four changed M16 cells all win; their
geometric means are **1.0187x versus Phase 63** and **1.0620x versus Machete
W4**.

## Promotion gates

- Twenty-four focused operator cases pass for both kernels across
  M1/M8/M16, scale modes 3/4, padded/unpadded output, eager execution, and
  CUDA Graph replay; candidate outputs are bit exact.
- The real Llama 3.2 one-layer logits and cached-generation test uses a
  16-token logits input and positively observes paired-path telemetry, while
  four-token prefill/one-token decode exercise the retained small-row path.
- The formal 20-cell MLP benchmark checks exact output for every rate/M pair
  and positively verifies the M16 production telemetry.
- Dispatch remains behind the existing physical-H100, SM90, exact-SiLU,
  N=8192, and grouped-runtime legality gates, with the additional `rows==16`
  condition. H200 and every unsupported shape retain the prior path.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase64_h100/paired_recovery_tiles.json`
- `artifacts/a41_phase64_h100/production_mlp_paired_recovery_tiles_vs_phase63.json`
