# Phase 68: H100 packed gate/up recovery

Phase 68 collapses the identical gate and up recovery topology into the two
lanes of native FP16-pair arithmetic.  It is promoted only for the physical
NVIDIA H100 M16 fused Llama-style MLP path.  The checkpoint format,
quantization result, P32 decoder, reduction order, and M1--M8 paths do not
change.

## Algebra and exactness

Phase 67 evaluates two independent five-stage recovery trees.  For each tree
node it adds or subtracts FP32 values, rounds the result to FP16, and converts
that rounded value back to FP32 for the next stage.  Gate and up select the
same two output tiles and follow the same add/subtract decisions, so their
independent values can occupy the low and high lanes of one `half2`.

For gate values \(g_i\) and up values \(u_i\), Phase 68 first evaluates the
same stage-zero FP32 arithmetic and packs its independently rounded results:

\[
p_i = \operatorname{half2}\left(
  R_{16}(g_{2i}\mathbin{\pm}g_{2i+1}),
  R_{16}(u_{2i}\mathbin{\pm}u_{2i+1})
\right).
\]

All following operands are exactly representable FP16 values.  Adding or
subtracting two FP16 values is exact in FP32 before the FP16 rounding step.
Consequently, lane-wise `__hadd2` and `__hsub2` produce exactly the same two
rounded results as the former independent FP32-add/FP16-convert sequences:

\[
\operatorname{hadd2}((g_0,u_0),(g_1,u_1))
=
(R_{16}(g_0+g_1),R_{16}(u_0+u_1)).
\]

The packed path is used only when conservative bounds prove that neither
tree can overflow.  Each projection computes four striped non-negative
partial sums and merges them as a balanced tree.  The longest FP32 summation
path has ten additions.  With unit roundoff \(u=2^{-24}\), the standard
positive-sum error bound is

\[
\gamma_{10}=\frac{10u}{1-10u}<5.97\times10^{-7}.
\]

If the computed sum is at most 64000, the exact sum is therefore below

\[
\frac{64000}{1-\gamma_{10}}<64000.039.
\]

Even after five worst-case upward FP16 roundings,

\[
64000.039(1+2^{-11})^5 < 64157 < 65504.
\]

NaN or infinity fails the ordered bound comparison.  If either projection is
unsafe, both projections execute the original Phase-64 overflow-preserving
trees.  Choosing that original tree for an otherwise safe sibling changes no
result.  Thus the optimization preserves every FP16 bit and all existing
rounding boundaries.

## Rejected precursor

A four-stripe bound applied independently to the existing scalar gate and up
trees reduced executed instructions by 3.00% and improved the isolated stage
in three of three runs.  Complete-MLP measurement was neutral: 2/4 rates won
and the geometric ratio was 1.000016x.  That precursor is not present in the
production binary.  Its SASS result motivated packing the shared tree itself.

## Isolated H100 measurement

Each run used three spaced 0% utilization / 0 MiB admission samples, 30
warmups, 300 CUDA-event samples, and 50 warmed CUDA Graph replays per sample.
All outputs are bit exact to Phase 67.

| Run | M/K/N gate/up x2 | Phase 67 | Packed gate/up | vs Phase 67 | Better last |
|--:|:--|--:|--:|--:|:--:|
| 1 | 16/2048/8192 x2 | 9.779 us | **9.419 us** | **1.0382x** | Yes |
| 2 | 16/2048/8192 x2 | 9.722 us | **9.294 us** | **1.0460x** | Yes |
| 3 | 16/2048/8192 x2 | 9.798 us | **9.412 us** | **1.0410x** | Yes |

The median isolated speedup is **1.0410x**.

## SASS and hardware counters

Nsight Compute 2026.2.1 collected 40 replay passes on the production binary.
`cuobjdump --dump-sass` and `--dump-resource-usage` were used on that exact
JIT extension.  The common bounded path replaces separate gate/up conversion
and butterfly streams with packed FP16 operations.  The fallback increases
static code size but is not dynamically executed for the benchmark input.

| M16 middle-kernel metric | Phase 67 | Phase 68 | Change |
|:--|--:|--:|--:|
| duration | 6.272 us | **6.176 us** | **1.0155x** |
| executed instructions | 1,535,402 | **1,181,304** | **-23.06%** |
| registers/thread | 45 | 72 | +27 |
| allocated registers/thread | 48 | 72 | +24 |
| static shared memory/block | 1.024 KiB | 1.024 KiB | unchanged |
| stack/local spilling requests | 0 | 0 | unchanged |
| achieved occupancy | 24.35% | 23.73% | -0.62 points |
| active warps/scheduler | 3.802 | 3.826 | +0.6% |
| eligible warps/scheduler | 0.623 | 0.479 | -23.1% |
| warp cycles/issued instruction | 10.37 | 13.73 | +32.4% |
| DRAM throughput | 7.96% | 8.12% | effectively unchanged |

The higher register count does not reduce the useful two-wave launch geometry:
M16 launches 256 blocks on a 132-SM H100.  Active warps remain effectively
unchanged, and there are no spills.

Profiler reports and extracted SASS remain outside Git under
`/root/qvq-profiler-artifacts/phase68-packed-gate-up`.

## Complete Llama 3.2 1B MLP

Timing uses 30 warmups, 300 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample after the strict H100 admission gate.  Marlin and Machete
execute W4 and are figurative latency/efficiency baselines.  Effective TFLOP/s
uses the dense-equivalent gate/up/down operation count.  `vs last` compares
with the committed Phase-67 benchmark; all four rows improve.

| W | M | M/K/N: gate/up x2; down | QVQ | Eff. TFLOP/s | vs Marlin W4 | vs Machete W4 | vs last | Better last |
|--:|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 16 | 16/2048/8192 x2; 16/8192/2048 | **46.838 us** | 34.387 | 0.697x | **1.074x** | **1.0048x** | Yes |
| 2.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | **48.053 us** | 33.517 | 0.680x | **1.046x** | **1.0057x** | Yes |
| 3 | 16 | 16/2048/8192 x2; 16/8192/2048 | **47.257 us** | 34.082 | 0.691x | **1.064x** | **1.0039x** | Yes |
| 3.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | **48.297 us** | 33.348 | 0.676x | **1.041x** | **1.0035x** | Yes |

Geometric results are **1.00612x** versus the same-binary Phase-67 control,
**1.00448x** versus the last committed benchmark, **1.05626x** versus Machete
W4, and **0.68607x** versus Marlin W4.

## Promotion gates

- Fifty-three focused CUDA cases pass, including scale modes 3/4, bias/no
  bias, paired/unpaired control paths, eager execution, CUDA Graph replay, and
  a forced high-magnitude fallback.
- The packed result is bit exact to the accepted recovery path.
- The real Llama 3.2 layer/logits/cached-generation test passes and observes
  the new production telemetry.
- Dispatch is restricted to physical NVIDIA H100, M16, the existing exact
  fused two-child N=8192 gate/up path, and the bounded-recovery gate.
- M1--M8, H200, unsupported shapes, checkpoint storage, persistent VRAM, and
  P32 decoder/reduction behavior are unchanged.

Artifacts:

- `artifacts/a41_phase68_h100/packed_gate_up_run1.json`
- `artifacts/a41_phase68_h100/packed_gate_up_run2.json`
- `artifacts/a41_phase68_h100/packed_gate_up_run3.json`
- `artifacts/a41_phase68_h100/production_mlp_packed_gate_up_vs_phase67.json`

Benchmark driver:

- `scripts/benchmark_qvq_phase68_packed_gate_up_recovery.py`
