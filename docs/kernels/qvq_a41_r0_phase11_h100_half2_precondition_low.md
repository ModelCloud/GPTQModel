# QVQ A41/R0 Phase 11: H100 half2 precondition low stage

Phase 11 vectorizes the first eight stages of the exact N=8192
SwiGLU/down-input Hadamard precondition on the physical H100. Adjacent FP16
columns enter one `half2` after the bit-1 butterfly, remain packed through
four warp-local stages, and use shared memory only for the final three
cross-warp stages.

The retained kernel removes **33.1%** of the low-stage executed warp
instructions and improves the isolated fused-SiLU precondition in all five M
rows by **1.0515x** geometric mean. The complete W2--W3.5 Llama 3.2 1B MLP
improves in all 20 M/rate cells and is **1.0111x** faster than accepted Phase
10, with a new latency range of **64.95--70.54 us**.

This is an execution-only change. Canonical P32 payloads, quantized weights,
scales, the exact SiLU boundary, down P32 decode, output recovery, tensor
layouts, CUDA Graph node count, operation-local workspace, and all non-H100
fallbacks are unchanged.

## Exact math and lane ownership

For every butterfly pair `(a, b)`, the required stage is:

\[
s=\operatorname{fp16}_{rn}(\operatorname{fp32}(a)+\operatorname{fp32}(b)),
\]

\[
d=\operatorname{fp16}_{rn}(\operatorname{fp32}(a)-\operatorname{fp32}(b)).
\]

The retained design keeps 256 threads, so each thread still computes one
input column and Phase 10's exact SiLU/product/scale setup retains its original
eight-warp parallelism. Adjacent lanes exchange their rounded input with a
fixed XOR shuffle. Even lanes then construct:

```cpp
half2 pair = __halves2half2(a, b);
half2 swapped = __lowhigh2highlow(pair);
half2 sum = __hadd2(pair, swapped);
half2 difference = __hsub2(pair, swapped);
half2 packed = __lows2half2(sum, difference);
```

The low lane of `sum` is `rn(a+b)` and the low lane of `difference` is
`rn(a-b)`, so `packed` is exactly the original bit-1 output. There is no
reassociation and no extra or missing narrowing boundary.

The next four pair-index butterflies, corresponding to original column bits
2, 4, 8, and 16, stay in registers. Even physical lanes use masked XOR
shuffles to exchange packed values. The lower member executes `hadd2`; the
upper member executes `hsub2(lower, upper)`, preserving the original
lower-minus-upper orientation.

Only the remaining original bits 32, 64, and 128 cross warp boundaries. The
128 packed values are written once to shared memory, and the first 128 threads
execute those final three stages. The result is stored to the same contiguous
`[M,8192]` FP16 workspace used before Phase 11.

## Candidate history

Instruction reduction alone was not the promotion gate. Three exact designs
were compiled, profiled, and benchmarked:

| Candidate | Design | Executed instructions | Full-MLP result | Decision |
|:--|:--|--:|:--|:--|
| v1 | 128 threads; each thread serially initializes two columns | 28,032 (-48.6%) | 1.0009x; 8/20 regressions | rejected |
| v2 | 256-thread setup, then 128-thread shared-memory butterfly | 40,576 (-25.6%) | 1.0037x; 17/20 improve | rejected |
| v3 | 256-thread setup, warp-local packed stages, three shared stages | 36,480 (-33.1%) | 1.0111x; 20/20 improve | promoted |

The first candidate cut the most instructions but halved useful setup warps
and reduced eligible warps from roughly 0.15 to 0.07 per cycle. The second
restored setup parallelism but still paid seven shared-memory/barrier stages.
The promoted design retains the setup grid and removes four of those shared
stages. This is why its end-to-end result is stronger despite not having the
lowest raw instruction count.

The extension contains only four fixed low-stage specializations: scalar or
packed low, each with fused SiLU disabled or enabled. Templates are not
multiplied by rate, M, dtype, bias, or codebook.

## Matched SASS and scheduler analysis

Nsight Compute 2026.2.1 captured one M1 scalar-low and one M1 retained
packed-low launch from the same binary. The reports, raw metric exports, and
source-correlated SASS exports are stored outside Git under
`/root/qvq-profiler-artifacts/phase11-half2-low/`.

These are measured executed warp-instruction counts, not estimates from CUDA
source:

| Metric/opcode | Phase-10 scalar low | Phase-11 packed low | Change |
|:--|--:|--:|--:|
| total executed warp instructions | 54,528 | 36,480 | **-33.1%** |
| `HADD2` | 4,864 | 3,328 | -31.6% |
| `FADD` | 3,584 | 0 | -100% |
| `F2FP` | 4,352 | 0 | -100% |
| `HFMA2` | 0 | 1,536 | packed replacement |
| `SHFL` | 0 | 1,280 | new register exchange |
| `STS` | 3,584 | 768 | -78.6% |
| `LDS` | 3,584 | 640 | -82.1% |
| `BAR` | 2,304 | 1,024 | -55.6% |
| registers/thread | 24 | 19 | -20.8% |
| static shared memory | 528 B | 512 B | -16 B |
| local/shared spills | 0 / 0 | 0 / 0 | unchanged |
| NCU duration | 3.968 us | 3.808 us | -4.0% |

The packed kernel trades scheduler headroom for a much shorter instruction
stream. Active warps remain nearly unchanged (11.96% to 11.86%), but eligible
warps/cycle falls from 0.15 to 0.10 and long-scoreboard latency rises from
7.21 to 11.11 cycles per issued instruction. This explains why a 33.1%
instruction reduction becomes a 5.15% isolated geometric-mean latency win,
not a 33% latency win. DRAM throughput remains negligible (0.59% versus
0.61%); this is a dependency/launch-latency kernel, not an HBM-bound kernel.

## Launches, traffic, and VRAM

| Property | Phase 10 | Phase 11 |
|:--|--:|--:|
| low-stage threads/block | 256 | 256 |
| low-stage warps/block | 8 | 8 |
| low-stage blocks/row | 32 | 32 |
| high stage | half2 | unchanged half2 |
| CUDA Graph nodes | 2 | 2 |
| global workspace | `2*M*8192` bytes | unchanged |
| persistent VRAM added | 0 | 0 |

The existing transient workspace is 16 KiB at M1 and 256 KiB at M16. Phase
11 changes only how the low kernel fills it. It adds no checkpoint bytes, no
persistent cache, and no additional global-memory allocation.

## Isolated fused-SiLU precondition benchmark

Both paths are compiled into the same extension and timed in one process.
Times are warmed CUDA Graph replays measured by CUDA events: 30 warmups, 300
samples, and 50 replays per sample. Every output is FP16 bit-exact.

| M | Scalar-low p50 us | half2-low p50 us | Speedup | Better |
|---:|---:|---:|---:|:---:|
| 1 | 4.614 | 4.330 | 1.066x | Yes |
| 2 | 4.557 | 4.402 | 1.035x | Yes |
| 4 | 4.651 | 4.628 | 1.005x | Yes |
| 8 | 4.845 | 4.537 | 1.068x | Yes |
| 16 | 5.094 | 4.693 | 1.085x | Yes |

Geometric-mean speedup is **1.0515x**.

## Complete Llama 3.2 1B MLP

The formal matrix uses 30 warmups, 100 CUDA-event samples, and 30 CUDA Graph
replays per sample. It includes gate/up P32, recovery, exact SiLU, FP16
product, down precondition, and down P32/recovery. `vs` is comparator latency
divided by QVQ latency, so a value below one means the W4 comparator remains
faster. `Better` compares against the committed Phase-10 production artifact;
`No` would be recorded as a regression.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 10 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.167 | 1.477 | 0.435x | 0.752x | 1.0137x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 67.784 | 2.970 | 0.465x | 0.735x | 1.0161x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 68.609 | 5.869 | 0.461x | 0.732x | 1.0096x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.361 | 11.610 | 0.425x | 0.722x | 1.0101x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.954 | 24.796 | 0.504x | 0.776x | 1.0126x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 69.110 | 1.457 | 0.429x | 0.742x | 1.0089x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 69.054 | 2.916 | 0.457x | 0.722x | 1.0131x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.546 | 5.790 | 0.455x | 0.722x | 1.0118x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.397 | 11.439 | 0.419x | 0.712x | 1.0079x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.868 | 24.452 | 0.497x | 0.765x | 1.0107x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 69.251 | 1.454 | 0.428x | 0.740x | 1.0174x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 69.327 | 2.904 | 0.455x | 0.719x | 1.0147x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.890 | 5.761 | 0.453x | 0.718x | 1.0140x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.545 | 11.416 | 0.418x | 0.710x | 1.0052x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 66.066 | 24.379 | 0.495x | 0.763x | 1.0106x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.939 | 1.460 | 0.430x | 0.744x | 1.0098x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 69.120 | 2.913 | 0.456x | 0.721x | 1.0120x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.724 | 5.775 | 0.454x | 0.720x | 1.0061x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.114 | 11.486 | 0.421x | 0.714x | 1.0063x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.678 | 24.523 | 0.498x | 0.767x | 1.0120x | Yes |

Geometric means:

- **1.0111x** versus accepted Phase 10;
- **2.0584x** versus ordinary per-module QVQ;
- **0.4520x** versus Marlin W4;
- **0.7345x** versus Machete W4.

The W4 comparisons are figurative dense-equivalent efficiency baselines.
They do not claim equal quantization quality or equal compressed decode work.

## Correctness and validation

- 120 random cases cover M=1/2/4/8/16, three seeds, scalar/half2 high,
  fused/unfused SiLU, scalar/half2 low, and ten exact repetitions.
- Eight scalar/packed/fused/high combinations pass CUDA Graph replay and
  argument guards.
- The packed fused path matches the reference for every one of the 63,488
  finite FP16 gate bit patterns.
- Runtime lifecycle/fallback/uninstall telemetry passes.
- A real Llama 3.2 layer passes logits tolerance, cached generation, exact
  token output, and packed-low launch telemetry.
- The focused H100 validation reports 131 passed tests.

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

- `artifacts/a41_phase11_h100/half2_precondition_low_experiment.json`
- `artifacts/a41_phase11_h100/production_mlp_half2_low_vs_baselines.json`
- `scripts/benchmark_qvq_phase11_half2_precondition_low.py`
- `scripts/profile_qvq_phase11_half2_precondition_low.py`

The measured artifacts identify production source commit `e68634d4`. The
Nsight reports and raw/source exports remain outside Git because the binary
reports are large.

## Next experiment

Phase 12 tested exact cooperative-grid and row-local completion-counter
low/high fusion. Both regressed every M, so neither was promoted. The accepted
alternative keeps the first five paired-recovery low butterflies warp-local,
removes 30.5% of that stage's executed instructions, and improves all 20
complete-MLP rate/M cells. See
`qvq_a41_r0_phase12_h100_warp_recovery_low.md` for the exact math, rejected
candidate data, matched SASS analysis, and production benchmark.
