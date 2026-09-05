# Qwen3.8-27B H100 large-M MLP optimization

## Scope

This phase targets the complete Qwen3.8-27B MLP at native inference row
counts `M = 1024, 2048, 4096` on the physical NVIDIA H100.  The model shapes
are:

```text
gate: M x 5120 x 17408
up:   M x 5120 x 17408
down: M x 17408 x 5120
```

All QVQ weights remain canonical V2B2-P32.  The implementation adds no dense,
FP16, or FP8 persistent weight cache and does not change checkpoint storage.

## Exact MLP contract

For Qwen's folded intermediate axis, gate and up have no output Hadamard and
down has no input Hadamard.  The execution order is therefore:

```text
t = H_5120(x * SU_gate_up)
g32 = t @ Q_gate
u32 = t @ Q_up
g16 = fp16(g32 * SV_gate + bias_gate)
u16 = fp16(u32 * SV_up + bias_up)
a16 = fp16(SiLU(g16))
p16 = fp16(a16 * u16)
d16 = fp16(p16 * SU_down)
y32 = d16 @ Q_down
y16 = fp16(H_5120(y32) * SV_down + bias_down)
```

The fused large-M SwiGLU precondition kernel retains each shown FP16 rounding
boundary explicitly.  It removes intermediate stores and reloads but does not
reassociate arithmetic.

## Production changes

1. Qwen gate/up uses the existing row-reuse-eleven schedule and pads each
   supported power-of-two `M` by `M/32`, making the row count divisible by
   `11 * 16` with only 3.125% extra rows.
2. Two adjacent N64 consumers share the same TMA-staged activation rows in an
   N128 CTA for Qwen gate/up.
3. Gate/up FP32 recovery, FP16 SiLU, FP16 product, and down `SU` multiplication
   are fused without deleting any semantic rounding boundary.
4. Qwen down uses the same N128 staging and row-reuse-eleven schedule.
5. Qwen's composite `H_5120 = H_40 ⊗ H_128` implementation exposes
   128-wide rows to the generic power-of-two CUDA kernel.  The previous fixed
   1024-thread block made 896 threads participate only in block barriers.
   The H100 path now launches 128 threads for an H128 row.  Butterfly ownership,
   ordering, shared layout, and arithmetic are unchanged.

## NCU and SASS evidence

Profiles were captured on the physical H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348` with NCU 2026.2.1.0.  Formal
latencies use warmed CUDA Graph replay measured by CUDA events; NCU uses an
eager single-launch mapping trace because profiling a whole graph would mix
unrelated nodes.

### H128 composite Hadamard

| Metric | Main, 1024 threads | Candidate, 128 threads | Change |
| --- | ---: | ---: | ---: |
| Kernel latency | 389.15 us | 71.26 us | 5.46x faster |
| Executed instructions | 207,749,120 | 67,829,760 | -67.35% |
| Achieved occupancy | 89.97% | 94.41% | +4.44 points |
| Eligible warps/scheduler | 1.77 | 5.39 | 3.05x higher |
| Compute throughput | 50.69% | 91.23% | +40.54 points |
| Registers/thread | 25 | 25 | unchanged |

The SASS change is structural rather than algebraic: useful HADD/FADD work is
unchanged while predicate, branch, synchronization, and address instructions
from dead lanes collapse.  Examples include branch execution falling from
29.98M to 9.34M and barrier-arrival bookkeeping falling from 27.03M combined
`BSSY`/`BSYNC` executions to 6.39M.

### Qwen down row reuse

| Metric | N128 reuse-8 | N128 reuse-11 | Change |
| --- | ---: | ---: | ---: |
| Kernel latency | 765.28 us | 621.18 us | 1.232x faster |
| Executed instructions | 326,403,606 | 278,141,616 | -14.79% |
| Registers/thread | 153 | 168 | expected accumulator growth |
| Achieved occupancy | 14.02% | 13.98% | effectively unchanged |

The remaining P32 SASS is dominated by address and decode plumbing (`R2UR`,
`IMAD`, `VIADD`, `PRMT`, and `LOP3`) rather than HBM traffic.  Further decoder
work should target representation-level address reuse, not another increase in
CTA shared-memory footprint; row-reuse eleven already approaches Hopper's
per-block shared-memory ceiling.

## Complete MLP benchmark

Candidate SHA: `c23277bd7e677264a6cfc6bfa43c73a6d42c2da0`.

Baseline SHA: `e509f61a92bd84eaa1a12f231273bfc836d315b8`.

The candidate and baseline were both measured with 10 graph warmups, 30
samples, and 20 graph replays per sample.  The Marlin and Machete columns are
W4 baselines.  A ratio below one in those columns means QVQ is still slower
than that W4 kernel.

| Weight | M | QVQ candidate | Main QVQ | vs main | vs Marlin W4 | vs Machete W4 | Mean abs error | Max abs error | Better than last | Regression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| W2 | 1024 | 2383.520 us | 4260.710 us | 1.788x | 0.708x | 0.414x | 3.539e-09 | 5.960e-08 | Yes | No |
| W2 | 2048 | 4634.757 us | 8595.326 us | 1.855x | 0.712x | 0.425x | 3.536e-09 | 5.960e-08 | Yes | No |
| W2 | 4096 | 9238.190 us | 17274.010 us | 1.870x | 0.720x | 0.424x | 3.540e-09 | 5.960e-08 | Yes | No |
| W2.5 | 1024 | 2387.238 us | 4363.133 us | 1.828x | 0.707x | 0.414x | 3.542e-09 | 5.960e-08 | Yes | No |
| W2.5 | 2048 | 4646.032 us | 8769.218 us | 1.887x | 0.711x | 0.424x | 3.540e-09 | 5.960e-08 | Yes | No |
| W2.5 | 4096 | 9374.946 us | 17667.705 us | 1.885x | 0.709x | 0.417x | 3.538e-09 | 5.960e-08 | Yes | No |
| W3 | 1024 | 2424.637 us | 4362.566 us | 1.799x | 0.696x | 0.407x | 3.530e-09 | 5.960e-08 | Yes | No |
| W3 | 2048 | 4683.590 us | 8772.794 us | 1.873x | 0.705x | 0.421x | 3.543e-09 | 5.960e-08 | Yes | No |
| W3 | 4096 | 9505.275 us | 17680.280 us | 1.860x | 0.699x | 0.412x | 3.543e-09 | 5.960e-08 | Yes | No |
| W3.5 | 1024 | 2363.018 us | 4674.520 us | 1.978x | 0.714x | 0.418x | 3.528e-09 | 5.960e-08 | Yes | No |
| W3.5 | 2048 | 4603.043 us | 9411.468 us | 2.045x | 0.717x | 0.428x | 3.527e-09 | 5.960e-08 | Yes | No |
| W3.5 | 4096 | 9341.557 us | 18980.951 us | 2.032x | 0.712x | 0.419x | 3.537e-09 | 5.960e-08 | Yes | No |

Geometric-mean speedup versus main is **1.890x**, equivalent to **47.09% lower
latency**.  The minimum cell speedup is **1.788x**, so all measured W2-W3.5 and
M1024-M4096 cells exceed the requested 1.5x target.  The best cell reaches
**2.045x**.

## Correctness and graph safety

- All 12 benchmark cells passed the dense-P32 Torch oracle gate.
- Maximum absolute error across every cell is `5.960e-08`, far below `2e-3`.
- Formal timing captures and replays the complete MLP in a CUDA Graph.
- 42 focused Hadamard accuracy, overflow, dtype, padding, and graph tests pass.
- 14 folded-SwiGLU tests, including M32 and M128 graph replay, pass.
