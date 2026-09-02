# QVQ A41/R0 Phase 10: H100 exact SiLU/precondition fusion

Phase 10 fuses the exact FP16 SiLU activation into the low stage of the
physical-H100 N=8192 SwiGLU/down-input precondition.  This deletes one CUDA
Graph node and the activated-gate global-memory materialization without
changing the FP16 activation or product boundaries.

The isolated SiLU-plus-precondition chain improves by **1.3515x** geometric
mean.  The complete W2--W3.5 Llama 3.2 1B MLP improves in all 20 M/rate cells
and is **1.0212x** faster than accepted Phase 9, with a new latency range of
**65.77--70.96 us**.

This is an execution-only change.  Canonical P32 payloads, quantized weights,
scales, output recovery, the down P32 kernel, Hadamard butterfly order, and
non-H100 fallbacks are unchanged.

## Exact math and rounding contract

Before Phase 10, production evaluates:

```text
recovered gate FP16
    -> standalone torch SiLU in FP32
    -> round activated gate to FP16
    -> write/read activated gate
    -> FP16 gate*up product
    -> down scale and low Hadamard stages
    -> high Hadamard stages
```

The fused low-stage specialization evaluates:

```text
recovered gate FP16
    -> convert gate to FP32
    -> gate / (1 + exp(-gate))
    -> round activated gate to FP16
    -> FP16 gate*up product
    -> down scale and low Hadamard stages
    -> high Hadamard stages
```

For each gate value \(g\), the helper is:

\[
a=\operatorname{fp16}_{rn}
  \left(\frac{\operatorname{fp32}(g)}
  {1+\exp(-\operatorname{fp32}(g))}\right).
\]

The existing product boundary remains:

\[
p=\operatorname{fp16}_{rn}
  (\operatorname{fp32}(a)\operatorname{fp32}(u)).
\]

The division form is deliberate.  On the H100 PyTorch/CUDA stack used for
promotion, it matched `torch.nn.functional.silu` for every one of the 63,488
finite FP16 bit patterns.  The superficially equivalent
`g * sigmoid(g)` form had one mismatch and was rejected.  A half-arithmetic
implementation had 10,161 mismatches and was also rejected.

The CUDA kernel has only two fixed low-stage specializations:
`FuseSilu=false` and `FuseSilu=true`.  It does not multiply templates by
rate, M, bias, or dtype.

## Production legality and fallback

Production enables fused SiLU only after the existing Phase-8 gate has
already proven all of the following:

- physical NVIDIA H100;
- compute capability 9.0;
- two-member gate/up group;
- output width 8192;
- exact supported SiLU callable;
- exact multiblock precondition path.

Recognized activation functions are `torch.nn.functional.silu`, non-inplace
`torch.nn.SiLU`, and Transformers `SiLUActivation`.  Inplace SiLU, wrapper
lambdas, GELU, H200, and unsupported shapes retain the original activation
and precondition path.  Telemetry records
`h100_fused_silu_precondition_low_launches` independently.

## Launches, traffic, and VRAM

| Property | Phase 9 | Phase 10 |
|:--|--:|--:|
| CUDA Graph nodes | 3 | 2 |
| standalone SiLU | yes | no |
| low precondition | unfused | owns exact SiLU |
| high precondition | half2 | unchanged half2 |
| persistent VRAM added | 0 | 0 |

Phase 10 removes the transient activated-gate tensor:

\[
2MN = 2M(8192)\ \text{bytes}.
\]

That is 16 KiB at M1 and 256 KiB at M16.  The existing multiblock workspace
is unchanged, so no persistent VRAM is added.

## Measured SASS and CUDA Graph attribution

Nsight Compute 2026.2.1 captured one M1 launch for standalone PyTorch SiLU,
the unfused low stage, the fused low stage, and the unchanged `half2` high
stage.  These are executed warp-instruction counts from generated SASS, not
source-level estimates.

| Kernel | Executed instructions | Registers/thread | Local/shared spills |
|:--|--:|--:|--:|
| standalone PyTorch SiLU | 6,080 | 32 | 0 / 0 |
| unfused low | 48,896 | 24 | 0 / 0 |
| fused-SiLU low | 54,528 | 24 | 0 / 0 |
| unchanged half2 high | 968 | 40 | 0 / 0 |
| unfused three-kernel chain | 55,944 | -- | 0 |
| fused two-kernel chain | 55,496 | -- | 0 |

The full chain removes only 448 executed instructions, or **0.80%**.  This is
therefore primarily a launch and materialization optimization.  The fused
low stage does more work locally, increasing its instructions by 5,632, but
it absorbs a standalone 6,080-instruction kernel.

The SASS explains the added low-stage work.  Relative to the unfused low
stage, the fused specialization adds the expected FP32 activation sequence,
including 2,560 more `FFMA`, 512 `MUFU`, and 512 `FCHK` instructions.  It
keeps 24 registers/thread and introduces no spill.

Both low stages are tiny-grid latency/dependency bound rather than HBM bound.
The NCU speed-of-light pass measured only 0.67% DRAM throughput for unfused
low and 0.61% for fused low.  Fused low has 0.134 eligible warps/cycle and
6.88 long-scoreboard cycles per issued instruction, versus 0.169 and 4.34 for
unfused low.  Optimizing the fused activation's local dependency chain alone
is therefore unlikely to reproduce the end-to-end gain; deleting the
boundary is the important result.

Nsight Systems 2026.4.1 captured one warmed CUDA Graph replay with node-level
tracing:

| Chain | Node durations | Node count | Graph kernel span |
|:--|:--|--:|--:|
| unfused | SiLU 1.472 us; low 1.760 us; high 1.792 us | 3 | 5.248 us |
| fused | low 1.632 us; high 1.664 us | 2 | 3.584 us |

The node-level graph span improves by **1.464x**.  CUDA-event measurements
below remain the performance authority; profiler timings are supporting
attribution because profiler replay perturbs short kernels.

## Isolated SiLU plus precondition benchmark

Both paths are compiled into the same extension and timed in one process.
Times are warmed CUDA Graph replays measured by CUDA events: 30 warmups, 100
samples, and 50 replays per sample.  Every result is FP16 bit-exact.

| M | Unfused p50 us | Fused p50 us | Speedup | Better |
|---:|---:|---:|---:|:---:|
| 1 | 6.125 | 4.382 | 1.398x | Yes |
| 2 | 6.142 | 4.563 | 1.346x | Yes |
| 4 | 6.211 | 4.660 | 1.333x | Yes |
| 8 | 6.379 | 4.724 | 1.350x | Yes |
| 16 | 6.764 | 5.078 | 1.332x | Yes |

Geometric-mean speedup is **1.3515x**.

## Complete Llama 3.2 1B MLP

The formal matrix uses 20 warmups, 50 CUDA-event samples, and 20 CUDA Graph
replays per sample.  It includes gate/up P32, recovery, exact SiLU, FP16
product, down precondition, and down P32/recovery.  `vs` is comparator latency
divided by QVQ latency, so a value below one means the W4 comparator remains
faster.  `Better` compares with the accepted Phase-9 artifact.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 9 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 69.099 | 1.457 | 0.430x | 0.735x | 1.0131x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.874 | 2.923 | 0.462x | 0.733x | 1.0198x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.266 | 5.813 | 0.463x | 0.736x | 1.0264x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.062 | 11.494 | 0.427x | 0.720x | 1.0178x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.770 | 24.489 | 0.501x | 0.772x | 1.0237x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 69.723 | 1.444 | 0.426x | 0.729x | 1.0156x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 69.959 | 2.878 | 0.454x | 0.722x | 1.0191x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 70.366 | 5.722 | 0.455x | 0.724x | 1.0192x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.957 | 11.349 | 0.422x | 0.711x | 1.0225x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 66.573 | 24.193 | 0.495x | 0.762x | 1.0258x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 70.455 | 1.429 | 0.422x | 0.721x | 1.0140x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 70.350 | 2.862 | 0.452x | 0.718x | 1.0193x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 70.867 | 5.682 | 0.452x | 0.719x | 1.0138x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.912 | 11.356 | 0.422x | 0.712x | 1.0262x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 66.763 | 24.124 | 0.493x | 0.760x | 1.0310x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 69.615 | 1.446 | 0.427x | 0.730x | 1.0187x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 69.952 | 2.878 | 0.454x | 0.722x | 1.0193x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 70.149 | 5.740 | 0.457x | 0.727x | 1.0281x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.557 | 11.414 | 0.424x | 0.715x | 1.0200x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 66.469 | 24.231 | 0.495x | 0.763x | 1.0312x | Yes |

Geometric means:

- **1.0212x** versus accepted Phase 9;
- **2.0404x** versus ordinary per-module QVQ;
- **0.4509x** versus Marlin W4;
- **0.7314x** versus Machete W4.

The W4 comparisons are figurative dense-equivalent efficiency baselines.
They do not claim equal quantization quality or equal compressed decode work.

## Correctness and validation

- Exhaustive comparison covers all 63,488 finite FP16 gate bit patterns.
- 60 random kernel cases cover fuse/no-fuse, scalar/half2 high, M=1/2/4/8/16,
  three seeds, and exact repeatability.
- Four CUDA Graph/contract-guard combinations pass.
- Narrow activation recognition rejects unsupported and inplace callables.
- Runtime lifecycle/fallback/uninstall telemetry passes.
- A real one-layer Llama test passes logits tolerance, cached generation, and
  fused-launch telemetry.
- The focused H100 suite reports 65 passed tests.

Compilation used no more than four Ninja jobs, one NVCC host thread, and one
CUDA 13 split-compile partition:

```text
MAX_JOBS=4
NINJAFLAGS=-j4
CMAKE_BUILD_PARALLEL_LEVEL=4
NVCC_THREADS=1
GPTQMODEL_QVQ_NVCC_THREADS=1
GPTQMODEL_NVCC_SPLIT_COMPILE=1
```

Artifacts and drivers:

- `artifacts/a41_phase10_h100/fused_silu_precondition_experiment.json`
- `artifacts/a41_phase10_h100/production_mlp_fused_silu_vs_baselines.json`
- `scripts/benchmark_qvq_phase10_silu_precondition.py`
- `scripts/profile_qvq_phase10_silu_precondition.py`

Nsight reports, source SASS exports, and raw metric exports are kept outside
Git under `/root/qvq-profiler-artifacts/phase10-fused-silu/`.

## Next experiment

Phase 11 subsequently promoted exact `half2` vectorization of the
precondition low stage. Its bit-1 lane exchange and warp-local packed
butterflies are documented in
`qvq_a41_r0_phase11_h100_half2_precondition_low.md`. Further isolated SiLU
algebra remains a lower priority: Phase 10 shows that removing the
launch/materialized boundary matters more than shaving the activation
dependency chain.
