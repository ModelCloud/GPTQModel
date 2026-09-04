# QVQ A41/R0 Phase 17: direct padded down input

Phase 17 removes the separate allocation, zero-fill, and valid-row copy
between the fused SwiGLU/down-precondition transform and the Hopper down P32
kernel. For `M < 16`, the precondition high stage now writes the final
`16x8192` input buffer directly: transformed values occupy rows `[0,M)` and
the same kernel writes exact FP16 zero to rows `[M,16)`. The M16 specialization
is unchanged.

On the physical 132-SM NVIDIA H100, the affected boundary improves **1.6649x**
geometric mean for M1/M2/M4/M8. Complete Llama 3.2 1B MLP improves **1.0420x**
across all 16 affected rate/M cells, and every affected cell is positive.

## Exact row math

Let `T` be the exact Phase-16 fused SwiGLU, down scale, and Hadamard
precondition result:

\[
T_r=H\left(\operatorname{FP16}(\operatorname{SiLU}(G_r))
              \odot U_r\odot SU_{down}\right),\qquad 0\le r<M.
\]

The old runtime constructed the down input `P` in two operations:

\[
P=0_{16\times8192},\qquad P[0:M,:]=T.
\]

The Phase-17 high-stage kernel writes the same definition directly:

\[
P_r=\begin{cases}
T_r,&0\le r<M,\\
0,&M\le r<16.
\end{cases}
\]

The down matrix multiplication has no cross-row dependency:

\[
Y_r=P_rQ_{down}.
\]

Therefore zero padding may be produced by the precondition kernel without
changing any valid row, P32 decode, WGMMA accumulation, split-K order, or
output recovery. Runtime slices `Y[0:M,:]` before recovery, just as the old
path did after its separately padded input. Tests require the complete
`16x8192` FP16 input buffer to be bit-exact, including an all-zero tail.

## CUDA pipeline

Before:

```text
precondition low (M rows)
  -> precondition high (M rows)
  -> allocate/fill 16x8192 zeros
  -> copy Mx8192 valid rows
  -> down P32 on M16
```

Phase 17:

```text
precondition low (M rows)
  -> precondition high (16 rows: M transformed + 16-M zero)
  -> down P32 on M16
```

Only the measured H100 fused-MLP path with `M < 16` requests direct padding.
M16 dispatches the pre-existing no-padding compile-time specialization, so it
does not pay a row predicate.

## Isolated CUDA Graph A/B

The A/B uses 30 warmups, 200 CUDA-event samples, and 50 CUDA Graph replays per
sample. Both paths produce the complete padded matrix and are FP16 bit-exact.

| M | Separate padding us | Direct padding us | Speedup | Bytes no longer simultaneously retained | Better |
|---:|---:|---:|---:|---:|:---:|
| 1 | 7.100 | 4.211 | 1.686x | 16 KiB | Yes |
| 2 | 7.210 | 4.304 | 1.675x | 32 KiB | Yes |
| 4 | 7.407 | 4.511 | 1.642x | 64 KiB | Yes |
| 8 | 7.348 | 4.435 | 1.657x | 128 KiB | Yes |
| 16 | 4.793 | 4.705 | 1.019x | 0 | Yes (identical control) |

M16 invokes identical production code in both controls; its small measured
difference is timing variance rather than a direct-padding effect.

## Nsight Systems launch evidence

Nsight Systems 2026.4.1 traced one warmed M1 CUDA Graph replay with CUDA graph
node tracing enabled. The application verifies H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, PCI `00000000:44:00.0`, before
capture. Raw reports remain outside Git under:

```text
/root/qvq-profiler-artifacts/phase17-direct-padding/
```

The exported Nsight CSV calls CUDA-visible ordinal zero `NVIDIA H200 (0)`;
that label follows the host's global ordinal-zero name even though visibility
was restricted to the H100 UUID. The application-side UUID/PCI assertion and
`nvidia-smi` both resolve the executed device to physical GPU 1, NVIDIA H100.

| Graph node | Separate duration | Direct duration | Change |
|:--|--:|--:|:--|
| precondition low | 1.664 us | 1.504 us | same kernel |
| precondition high | 1.504 us, grid `4x1` | 1.792 us, grid `4x16` | writes valid and zero rows |
| FP16 zero fill | 1.152 us | removed | one kernel removed |
| device-to-device valid-row copy | 1.024 us | removed | one memory node removed |
| visible nodes | 4 | 2 | **-2 nodes** |

Profiler durations include instrumentation and are not the benchmark timing.
The trace is used to verify launch topology: the direct high stage does more
row work, while the fill and copy disappear completely.

## Transient and persistent storage

The old path retained an `Mx8192` FP16 transform output at the same time as a
new `16x8192` padded input. The new path retains only the final padded buffer.
The removed transient live bytes are:

\[
2M\cdot8192=16M\ \text{KiB}.
\]

That is 16, 32, 64, and 128 KiB for M1, M2, M4, and M8. M16 is unchanged.
Checkpoint format, grouped P32 payload, persistent cache size, and persistent
VRAM are unchanged.

## Complete Llama 3.2 1B MLP

The formal post-commit artifact executes production SHA `71a3f947` on the
physical H100. `vs` is comparator latency divided by QVQ latency, so a value
below one means the W4 comparator is faster. `Better` is the strict comparison
with the committed Phase-16 artifact; `No` records a measured regression.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 16 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 64.997 | 1.549 | 0.449x | 0.786x | 1.0383x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 64.899 | 3.102 | 0.483x | 0.779x | 1.0420x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 65.587 | 6.139 | 0.480x | 0.779x | 1.0425x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 66.303 | 12.146 | 0.444x | 0.770x | 1.0404x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.487 | 24.976 | 0.506x | 0.796x | 0.9986x | No |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 65.919 | 1.527 | 0.443x | 0.775x | 1.0419x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 65.907 | 3.055 | 0.475x | 0.767x | 1.0431x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 66.425 | 6.062 | 0.474x | 0.769x | 1.0410x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 67.033 | 12.014 | 0.439x | 0.762x | 1.0433x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.297 | 24.666 | 0.500x | 0.786x | 0.9976x | No |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 63.041 | 1.597 | 0.463x | 0.810x | 1.0416x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 62.985 | 3.196 | 0.497x | 0.802x | 1.0444x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 63.431 | 6.348 | 0.497x | 0.805x | 1.0457x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 64.137 | 12.556 | 0.459x | 0.796x | 1.0430x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 62.358 | 25.828 | 0.523x | 0.823x | 0.9977x | No |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 65.694 | 1.532 | 0.445x | 0.777x | 1.0387x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 66.096 | 3.046 | 0.474x | 0.765x | 1.0427x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 66.552 | 6.050 | 0.473x | 0.767x | 1.0443x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 67.082 | 12.005 | 0.439x | 0.761x | 1.0396x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.311 | 24.661 | 0.500x | 0.786x | 0.9956x | No |

The 16 affected M1-M8 cells improve **16/16** with **1.0420x** geometric
mean. The all-20-cell geometric means are **1.0329x** versus Phase 16,
**2.1424x** versus ordinary per-module QVQ, **0.4726x** versus Marlin W4,
and **0.7828x** versus Machete W4. M16 runs unchanged production source, so
its four `No` cells are independent-run telemetry rather than code
regressions. The W4 comparisons remain figurative dense-equivalent throughput
baselines, not equal-work or equal-quality claims.

## Correctness and promotion gates

- 134 focused CUDA precondition tests pass on the physical H100.
- All five M values prove valid rows bit-exact to the pre-padding path, zero
  tail rows, repeatability, and CUDA Graph stability.
- The 50-test grouped runtime suite passes, including the real one-layer Llama
  logits/cache lifecycle and direct-padding telemetry.
- The complete MLP artifact passes its dense/reference and exactness checks.
- Formal measurements used strict three-sample 0%-utilization, zero-MiB idle
  admission and CUDA-event timing around warmed CUDA Graph replay.
- Builds use at most four Ninja jobs, one NVCC host thread, and one CUDA split
  compile partition.

Artifacts:

- `artifacts/a41_phase17_h100/direct_padding_experiment.json`
- `artifacts/a41_phase17_h100/production_mlp_direct_padding_vs_phase16.json`

## Rejected decoder experiments

Phase 17 first tested three representation/algebra candidates against the
accepted Phase-16 W3 decoder. A pair-index lane-private level table regressed
W3 M1 to **0.827x**. A doubled-product PGC form improved isolated gate/up by
about 1.3% but was flat in the complete MLP because high extraction replaced
the removed multiply with shift/sign-extension work. Explicit lane-base
hoisting measured **0.990x**. All three candidates were fully reverted before
the direct-padding production commit.

## Next experiment

Phase 18 implements the analogous direct-padded shared input Hadamard for both
grouped gate/up and QKV. See
`docs/kernels/qvq_a41_r0_phase18_h100_direct_input_padding.md`.
