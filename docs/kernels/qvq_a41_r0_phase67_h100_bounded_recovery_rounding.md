# Phase 67: H100 bounded recovery rounding

Phase 67 removes repeated overflow tests from the exact recovery-high tree
when one conservative bound proves that every intermediate result fits in
FP16. The optimization is promoted only for the physical NVIDIA H100 M16
fused MLP path. It improves the complete MLP at all four QVQ rates in the
same-binary comparison while M1--M8 retain the Phase-64 implementation.

## Exactness argument

The accepted recovery tree rounds every internal result with

```text
R(x) = fp16(x) when the rounded value is finite, otherwise x in FP32.
```

For the 32 inputs to one five-stage recovery subtree, Phase 67 computes

\[
B=\sum_{i=0}^{31}|x_i|.
\]

When \(B\le 64000\), the magnitude at every exact tree node is bounded by
the sum of the magnitudes below that node. FP16 round-to-nearest can increase
a normal finite magnitude by at most a factor of \(1+2^{-11}\) at each
stage. Therefore the largest possible rounded five-stage result is below

\[
64000(1+2^{-11})^5 < 64157 < 65504.
\]

The subnormal absolute error is negligible relative to this margin. Thus no
rounded node can become infinity, so `R(x)` is exactly the direct
FP32-to-FP16-to-FP32 conversion at every node. NaN or infinity propagates
into `B` and fails the ordered comparison. Inputs above the bound use the
original overflow-preserving tree unchanged.

The paired M16 specialization evaluates the bound once for each shared
32-value tree and retains Phase 64's exact output-tile pairing. No butterfly,
rounding point, scale, bias, SiLU boundary, product, or down-input transform
is reordered.

## Isolated same-binary measurements

Each run used three spaced 0% utilization / 0 MiB admission samples, 30
warmups, 300 CUDA-event samples, and 50 warmed CUDA Graph replays per sample.
The table reports the median timing and speedup across three independent
runs. `Wins` counts strict candidate wins among those runs.

| M | M/K/N gate/up x2 | Phase 64 | Bounded | vs Phase 64 | Wins |
|--:|:--|--:|--:|--:|--:|
| 1 | 1/2048/8192 x2 | 7.745 us | 7.597 us | 1.0181x | 3/3 |
| 2 | 2/2048/8192 x2 | 7.898 us | 7.748 us | 1.0186x | 3/3 |
| 4 | 4/2048/8192 x2 | 8.002 us | 7.990 us | 1.0014x | 2/3 |
| 8 | 8/2048/8192 x2 | 8.860 us | 8.656 us | 1.0211x | 3/3 |
| 16 | 16/2048/8192 x2 | 9.930 us | 9.809 us | 1.0124x | 3/3 |

The isolated experiment wins broadly, but its whole-MLP effect is small.
An initial all-M production trial was 0.9996x geometrically versus a
same-binary Phase-64 control. Production is consequently restricted to M16,
where all four rates won in the high-sample complete-MLP measurement.

## Nsight Compute and emitted code

Nsight Compute 2026.2.1 collected 19 hardware-counter replay passes for the
M16 middle kernel. Both specializations were selected explicitly within the
same binary. `cuobjdump --dump-resource-usage` independently reports zero
stack and zero local memory for both.

| M16 metric | Phase 64 | Phase 67 | Change |
|:--|--:|--:|--:|
| duration | 6.30 us | **6.27 us** | 1.005x |
| executed instructions | 1,598,895 | **1,537,450** | **-3.84%** |
| registers/thread | 40 | 45 | +5 |
| static shared memory/block | 1.02 KiB | 1.02 KiB | unchanged |
| achieved occupancy | 24.36% | 24.35% | unchanged |
| active warps/scheduler | 3.81 | 3.80 | unchanged |
| eligible warps/scheduler | 0.62 | 0.62 | unchanged |
| warp cycles/issued instruction | 10.37 | 10.37 | unchanged |
| DRAM throughput | 7.99% | 7.96% | unchanged |

The result matches the intended mechanism: fewer dynamically executed
finite-test/select instructions with no spill, memory-bandwidth, occupancy,
or scheduler regression. Profiler reports remain outside Git under
`/root/qvq-profiler-artifacts/phase67-bounded-recovery`.

## Complete Llama 3.2 1B MLP

Timing uses 30 warmups, 300 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample after the strict H100 admission gate. Marlin and Machete
execute W4 and are figurative latency/efficiency baselines, not equal-rate
comparisons. Effective TFLOP/s uses the dense-equivalent gate/up/down
operation count.

`vs last` and `Better last` compare with the committed Phase-64 artifact, as
requested. `vs same-bin` compares the candidate with Phase 64 compiled into
and timed in this exact process. M1--M8 execute identical production code in
both same-binary arms, so their strict differences are timing noise rather
than kernel changes. M16 is the only changed production path.

| W | M | M/K/N: gate/up x2; down | QVQ us | Eff. TFLOP/s | vs Marlin W4 | vs Machete W4 | vs last | Better last | vs same-bin | Better same-bin |
|--:|--:|:--|--:|--:|--:|--:|--:|:--:|--:|:--:|
| 2 | 1 | 1/2048/8192 x2; 1/8192/2048 | 44.310 | 2.272 | 0.657x | 1.125x | 0.9977x | No | 1.0008x | Yes |
| 2 | 2 | 2/2048/8192 x2; 2/8192/2048 | 44.579 | 4.516 | 0.700x | 1.118x | 0.9973x | No | 1.0004x | Yes |
| 2 | 4 | 4/2048/8192 x2; 4/8192/2048 | 45.151 | 8.918 | 0.696x | 1.116x | 0.9923x | No | 0.9978x | No |
| 2 | 8 | 8/2048/8192 x2; 8/8192/2048 | 45.989 | 17.511 | 0.638x | 1.096x | 0.9978x | No | 1.0014x | Yes |
| 2 | 16 | 16/2048/8192 x2; 16/8192/2048 | 47.065 | 34.221 | 0.693x | 1.076x | 1.0009x | Yes | **1.0017x** | **Yes** |
| 2.5 | 1 | 1/2048/8192 x2; 1/8192/2048 | 45.449 | 2.215 | 0.640x | 1.097x | 1.0009x | Yes | 1.0041x | Yes |
| 2.5 | 2 | 2/2048/8192 x2; 2/8192/2048 | 45.817 | 4.394 | 0.681x | 1.088x | 0.9975x | No | 1.0007x | Yes |
| 2.5 | 4 | 4/2048/8192 x2; 4/8192/2048 | 46.506 | 8.658 | 0.676x | 1.084x | 0.9982x | No | 0.9987x | No |
| 2.5 | 8 | 8/2048/8192 x2; 8/8192/2048 | 47.213 | 17.057 | 0.622x | 1.068x | 0.9984x | No | 0.9993x | No |
| 2.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | 48.325 | 33.329 | 0.675x | 1.048x | 0.9992x | No | **1.0018x** | **Yes** |
| 3 | 1 | 1/2048/8192 x2; 1/8192/2048 | 45.085 | 2.233 | 0.645x | 1.106x | 0.9971x | No | 0.9979x | No |
| 3 | 2 | 2/2048/8192 x2; 2/8192/2048 | 45.280 | 4.446 | 0.689x | 1.101x | 1.0010x | Yes | 1.0024x | Yes |
| 3 | 4 | 4/2048/8192 x2; 4/8192/2048 | 45.532 | 8.843 | 0.690x | 1.107x | 1.0037x | Yes | 1.0004x | Yes |
| 3 | 8 | 8/2048/8192 x2; 8/8192/2048 | 46.568 | 17.293 | 0.630x | 1.082x | 1.0017x | Yes | 0.9985x | No |
| 3 | 16 | 16/2048/8192 x2; 16/8192/2048 | 47.444 | 33.948 | 0.688x | 1.067x | 1.0040x | Yes | **1.0012x** | **Yes** |
| 3.5 | 1 | 1/2048/8192 x2; 1/8192/2048 | 46.024 | 2.187 | 0.632x | 1.083x | 0.9986x | No | 1.0009x | Yes |
| 3.5 | 2 | 2/2048/8192 x2; 2/8192/2048 | 46.028 | 4.374 | 0.678x | 1.083x | 1.0079x | Yes | 1.0010x | Yes |
| 3.5 | 4 | 4/2048/8192 x2; 4/8192/2048 | 46.672 | 8.627 | 0.673x | 1.080x | 0.9988x | No | 0.9989x | No |
| 3.5 | 8 | 8/2048/8192 x2; 8/8192/2048 | 47.529 | 16.943 | 0.618x | 1.061x | 0.9951x | No | 1.0011x | Yes |
| 3.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | 48.465 | 33.232 | 0.673x | 1.045x | 1.0032x | Yes | **1.0020x** | **Yes** |

At M16 the candidate wins **4/4** same-binary cells with a geometric
**1.00168x** speedup. It also wins 3/4 cells versus the older committed
artifact, geometrically **1.00182x**. Across all 20 cells, including the 16
unchanged-code measurements, QVQ is **1.0862x** Machete W4 and **0.6643x**
Marlin W4 geometrically.

## Promotion gates

- Forty-eight focused operator cases cover both kernels, paired/unpaired
  tiles, scale modes 3/4, padded/unpadded output, eager execution, and CUDA
  Graph replay with bit-exact output.
- A high-magnitude test forces the overflow-preserving fallback and requires
  exact FP16 bits with finite final output.
- The real Llama 3.2 layer/logits/cached-generation test passes and positively
  observes M16 bounded-rounding telemetry.
- No checkpoint format, quantization result, persistent VRAM, reduction order,
  non-H100 dispatch, or M1--M8 production path changes.
- Compilation was capped at four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase67_h100/bounded_recovery_rounding.json`
- `artifacts/a41_phase67_h100/bounded_recovery_rounding_run2.json`
- `artifacts/a41_phase67_h100/bounded_recovery_rounding_run3.json`
- `artifacts/a41_phase67_h100/production_mlp_bounded_recovery_vs_phase64.json`
