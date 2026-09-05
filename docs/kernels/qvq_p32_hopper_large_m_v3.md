# QVQ P32 Hopper large-M optimization, third pass

## Scope and baseline

This pass starts from merged `origin/main` commit `f74ec0cc`, which contains
PR #112. It targets the complete Llama 3.2 1B MLP on the physical 132-SM
NVIDIA H100. All timings use warmed CUDA Graph replay and CUDA events. Marlin
and Machete use W4 and remain figurative latency/efficiency baselines for the
lower-rate QVQ W2-W3.5 kernels.

The MLP contains two `M x 2048 x 8192` gate/up projections, exact FP16
recovery and SwiGLU/down preconditioning, and one `M x 8192 x 2048` down
projection.

## Phase 1: reuse one decoded P32 fragment across M128

The merged kernel owns four independent M16 activation tiles per CTA. For one
P32 fragment `Q_f`, it computes

```text
A_f = decode_p32(Q_f)
C_0 += A_f @ B_0
C_1 += A_f @ B_1
C_2 += A_f @ B_2
C_3 += A_f @ B_3
```

and repeats the complete compressed-weight decode for the next M64 slab. The
new kernel retains eight independent FP32 accumulators and eight TMA-staged
M16 activation tiles:

```text
A_f = decode_p32(Q_f)
for r in 0..7:
    C_r += A_f @ B_r
```

Thus the exact same decoded register fragment feeds M128 rather than M64.
There is no change to the P32 checkpoint, selector stream, codebook, input
transform, K traversal, FP32 accumulation within an output element, or FP16
recovery boundary. Runtime dispatch is restricted to the measured physical
H100 Llama gate/up geometry, unsplit children, `M >= 128`, and `M % 128 == 0`.
Other shapes retain reuse-4 or the established fallback.

The named low-level operator is
`p32_window_m128_tma_grouped_reuse8`. The Python wrapper rejects split-K
children and invalid row geometry before launch. Production telemetry records
`h100_reuse8_gate_up_launches` independently.

## Correctness and CUDA Graph safety

The real `2048 -> 8192 + 8192` low-level test covers W2, W2.5, W3, and W3.5.
Reuse-8 must be bit-exact to reuse-4, and the captured/replayed CUDA Graph
must remain bit-exact. The runtime test separately checks H100 dispatch,
telemetry, and the absence of plain fallback. All five focused tests pass.

Across the full production benchmark the maximum absolute error against the
dense FP32 P32 Torch oracle is `1.073e-6`, below the `2e-3` acceptance bound.

## H100 production benchmark

The executable kernel is commit `3d670c08`. `Better than last` is the strict
median comparison with the fresh merged-main artifact. Ratios above one in
the baseline columns mean QVQ is faster.

| Weight | MLP MKN (gate/up; down) | Merged main | Reuse-8 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last |
|---|---|---:|---:|---:|---:|---:|:---:|
| W2 | 128x2048x8192; 128x8192x2048 | 122.352 us | 115.046 us | 1.064x | 0.691x | 0.638x | Yes |
| W2 | 512x2048x8192; 512x8192x2048 | 410.857 us | 381.370 us | 1.077x | 0.434x | 0.307x | Yes |
| W2 | 4096x2048x8192; 4096x8192x2048 | 3156.976 us | 2968.358 us | 1.064x | 0.471x | 0.308x | Yes |
| W2.5 | 128x2048x8192; 128x8192x2048 | 123.860 us | 112.987 us | 1.096x | 0.703x | 0.649x | Yes |
| W2.5 | 512x2048x8192; 512x8192x2048 | 422.785 us | 388.721 us | 1.088x | 0.426x | 0.301x | Yes |
| W2.5 | 4096x2048x8192; 4096x8192x2048 | 3204.343 us | 2918.758 us | 1.098x | 0.479x | 0.313x | Yes |
| W3 | 128x2048x8192; 128x8192x2048 | 122.546 us | 116.418 us | 1.053x | 0.683x | 0.630x | Yes |
| W3 | 512x2048x8192; 512x8192x2048 | 420.238 us | 385.756 us | 1.089x | 0.430x | 0.303x | Yes |
| W3 | 4096x2048x8192; 4096x8192x2048 | 3183.490 us | 2975.374 us | 1.070x | 0.470x | 0.307x | Yes |
| W3.5 | 128x2048x8192; 128x8192x2048 | 123.464 us | 114.809 us | 1.075x | 0.692x | 0.639x | Yes |
| W3.5 | 512x2048x8192; 512x8192x2048 | 433.610 us | 396.394 us | 1.094x | 0.418x | 0.295x | Yes |
| W3.5 | 4096x2048x8192; 4096x8192x2048 | 3283.414 us | 3028.471 us | 1.084x | 0.461x | 0.302x | Yes |

All twelve cells improve. The geometric-mean speedup over merged main is
`1.0792x`; the Marlin and Machete geometric ratios are `0.5180x` and
`0.3897x`, respectively.

## Exact-commit Nsight Compute and SASS audit

The W3 M512 gate/up kernel at commit `3d670c08` was collected with Nsight
Compute 2026.2.1 using SpeedOfLight, LaunchStats, Occupancy, SchedulerStats,
WarpStateStats, InstructionStats, and MemoryWorkloadAnalysis. The matched
merged-main report used the same command and physical H100.

| Metric | M64 reuse-4 | M128 reuse-8 | Change |
|---|---:|---:|---:|
| NCU duration | 178.30 us | 154.24 us | 1.156x |
| Executed warp instructions | 102.93 M | 62.01 M | -39.76% |
| Grid blocks | 1024 | 512 | -50% |
| Registers/thread | 91 | 137 | +46 |
| Achieved occupancy | 26.63% | 13.81% | -12.82 points |
| Eligible warps/scheduler | 1.54 | 0.63 | -0.91 |
| DRAM throughput | 6.24% | 7.39% | +1.15 points |

The emitted dynamic opcode mix demonstrates algebraic/representation reuse,
not a timing-only effect. `IMAD` falls from 19.93M to 9.99M, `PRMT` from
12.68M to 6.34M, `LOP3` from 11.20M to 5.64M, shared halfword/word loads from
8.39M each to 4.19M each, and 64-bit funnel shifts from 4.19M to 2.10M.
Tensor-core `HGMMA` remains exactly 4.19M because useful matrix work is
unchanged.

The higher register footprint and lower occupancy are real costs, but the
40% instruction reduction wins. The next phase should attempt to recover
scheduler readiness without giving up M128 decode reuse, preferably by
sharing the same eight input tiles across more independent N64 consumer
warpgroups. Any wider-CTA candidate must stay within H100 shared-memory and
register limits and is rejected on spills or full-MLP regression.

## Phase 2: coalesced reuse-8 accumulator stores

The original reuse-8 implementation retained the generic RS-WGMMA scalar
store mapping. Each consumer lane owns eight permuted FP32 accumulator values,
so direct global stores produce nearly twice the ideal number of sectors. The
reuse-4 kernel already had an exact shared-memory transpose, but its compile-
time gate excluded `RowTilesPerCta == 8`.

After the final WGMMA stage, the reuse-8 kernel now reclaims dead input TMA
storage, scatters each 16x64 accumulator tile into canonical shared-memory
order, and writes aligned `float4` vectors. The transpose changes no value or
arithmetic order. W2.5 remains excluded because its measured decoder depth
makes the extra barriers unprofitable.

| Weight | MLP MKN (gate/up; down) | Reuse-8 | Coalesced | Speedup | vs merged main | vs Marlin W4 | vs Machete W4 | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| W2 | 128x2048x8192; 128x8192x2048 | 115.046 us | 111.314 us | 1.034x | 1.099x | 0.709x | 0.657x | Yes |
| W2 | 512x2048x8192; 512x8192x2048 | 381.370 us | 379.795 us | 1.004x | 1.082x | 0.430x | 0.311x | Yes |
| W2 | 4096x2048x8192; 4096x8192x2048 | 2968.358 us | 2887.635 us | 1.028x | 1.093x | 0.484x | 0.316x | Yes |
| W2.5 | 128x2048x8192; 128x8192x2048 | 112.987 us | 112.474 us | 1.005x | 1.101x | 0.702x | 0.650x | Yes |
| W2.5 | 512x2048x8192; 512x8192x2048 | 388.721 us | 386.990 us | 1.004x | 1.092x | 0.422x | 0.305x | Yes |
| W2.5 | 4096x2048x8192; 4096x8192x2048 | 2918.758 us | 2915.539 us | 1.001x | 1.099x | 0.480x | 0.313x | Yes |
| W3 | 128x2048x8192; 128x8192x2048 | 116.418 us | 113.401 us | 1.027x | 1.081x | 0.696x | 0.645x | Yes |
| W3 | 512x2048x8192; 512x8192x2048 | 385.756 us | 381.948 us | 1.010x | 1.100x | 0.428x | 0.309x | Yes |
| W3 | 4096x2048x8192; 4096x8192x2048 | 2975.374 us | 2946.670 us | 1.010x | 1.080x | 0.475x | 0.310x | Yes |
| W3.5 | 128x2048x8192; 128x8192x2048 | 114.809 us | 111.954 us | 1.026x | 1.103x | 0.705x | 0.653x | Yes |
| W3.5 | 512x2048x8192; 512x8192x2048 | 396.394 us | 392.659 us | 1.010x | 1.104x | 0.416x | 0.301x | Yes |
| W3.5 | 4096x2048x8192; 4096x8192x2048 | 3028.471 us | 2978.757 us | 1.017x | 1.102x | 0.470x | 0.307x | Yes |

All twelve strict medians are lower, although W2.5 executes unchanged code
and its sub-percent movement is cross-run telemetry. Geometric-mean speedup
is `1.0144x` versus Phase 1 and `1.0948x` versus merged main.

The exact-commit `5c6632c7` W3 M512 Nsight Compute/SASS comparison is:

| Metric | Reuse-8 direct store | Reuse-8 coalesced | Change |
|---|---:|---:|---:|
| NCU duration | 154.24 us | 146.50 us | 1.053x |
| Executed warp instructions | 62.01 M | 63.63 M | +2.62% |
| L2 compression/store input sectors | 2.135 M | 1.087 M | -49.1% |
| Registers/thread | 137 | 139 | +2 |
| Achieved occupancy | 13.81% | 13.95% | +0.14 point |
| Eligible warps/scheduler | 0.63 | 0.69 | +0.06 |
| DRAM throughput | 7.39% | 7.27% | -0.12 point |

The opcode audit shows the decoder math is essentially unchanged; the small
instruction increase is the shared transpose and vector-store plumbing. The
win comes from eliminating conflicting/scattered output transactions, not
from reducing arithmetic. No spills are reported.

## Phase 3: paired, bounded FP16 recovery-low

The Phase-2 trace identified the first eight gate/up output-Hadamard stages as
the next largest removable boundary: two projection-major grids consumed
about 52 microseconds at M512. Gate and up have identical butterfly geometry,
but previously repeated CTA address/control/barrier work independently.

One Phase-3 CTA now owns the matching gate and up 256-column tiles. It first
computes conservative per-projection L1 bounds. When both bounds are at most
64000, no intermediate in the eight-stage tile transform can overflow FP16,
so gate and up occupy the low/high lanes of one `half2` value:

```text
packed[c] = half2(gate[c], up[c])
packed[c], packed[c xor bit] =
    half2_add_sub(packed[c], packed[c xor bit])
```

Every lane still receives the same independently rounded FP16 sum or
difference as the prior scalar FP32-add-then-convert sequence. The 1504-value
margin below FP16 maximum also absorbs floating reduction error. If either
bound is unsafe, the entire CTA executes the original scalar FP32,
`round_fp16_unless_overflow` path. The output workspace layout and the later
recovery/SwiGLU/down-precondition kernel are unchanged.

The focused physical-H100 tests cover M32 and M512 exactness, CUDA Graph
capture/replay, and a deliberately high-magnitude M32 case that forces the
overflow fallback. All three pass with exact FP16 bit equality.

The production artifact was generated from executable commit `8f6eb84e` with
CUDA Graph replay and CUDA-event timing after a three-sample idle-H100 gate.
`Better than last` compares strict medians with Phase 2.

| Weight | MLP MKN (gate/up; down) | Phase 2 | Phase 3 | Speedup | vs merged main | vs Marlin W4 | vs Machete W4 | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| W2 | 128x2048x8192; 128x8192x2048 | 111.314 us | 107.290 us | 1.038x | 1.140x | 0.740x | 0.682x | Yes |
| W2 | 512x2048x8192; 512x8192x2048 | 379.795 us | 369.216 us | 1.029x | 1.113x | 0.449x | 0.317x | Yes |
| W2 | 4096x2048x8192; 4096x8192x2048 | 2887.635 us | 2794.330 us | 1.033x | 1.130x | 0.487x | 0.307x | Yes |
| W2.5 | 128x2048x8192; 128x8192x2048 | 112.474 us | 109.856 us | 1.024x | 1.127x | 0.723x | 0.666x | Yes |
| W2.5 | 512x2048x8192; 512x8192x2048 | 386.990 us | 373.018 us | 1.037x | 1.133x | 0.445x | 0.313x | Yes |
| W2.5 | 4096x2048x8192; 4096x8192x2048 | 2915.539 us | 2804.438 us | 1.040x | 1.143x | 0.485x | 0.305x | Yes |
| W3 | 128x2048x8192; 128x8192x2048 | 113.401 us | 109.011 us | 1.040x | 1.124x | 0.728x | 0.671x | Yes |
| W3 | 512x2048x8192; 512x8192x2048 | 381.948 us | 372.019 us | 1.027x | 1.130x | 0.446x | 0.314x | Yes |
| W3 | 4096x2048x8192; 4096x8192x2048 | 2946.670 us | 2834.234 us | 1.040x | 1.123x | 0.480x | 0.302x | Yes |
| W3.5 | 128x2048x8192; 128x8192x2048 | 111.954 us | 107.578 us | 1.041x | 1.148x | 0.738x | 0.680x | Yes |
| W3.5 | 512x2048x8192; 512x8192x2048 | 392.659 us | 377.283 us | 1.041x | 1.149x | 0.440x | 0.310x | Yes |
| W3.5 | 4096x2048x8192; 4096x8192x2048 | 2978.757 us | 2869.104 us | 1.038x | 1.144x | 0.475x | 0.299x | Yes |

All twelve cells improve. The geometric-mean speedup is `1.0355x` over Phase
2 and `1.1337x` cumulatively over merged PR-112 main. The Marlin and Machete
W4 geometric ratios are `0.5394x` and `0.4003x`. Maximum and maximum-row-mean
absolute error against the dense P32 Torch oracle are `1.073e-6` and
`1.585e-7`, respectively.

### Exact-commit Nsight Compute and SASS audit

Matched M512 reports were collected from commit `8f6eb84e` with Nsight
Compute 2026.2.1 using the same seven profiling sections as Phases 1 and 2.
The plain and paired kernels coexist in that executable, so compiler flags,
inputs, and the physical H100 are identical.

| Metric | Projection-major scalar | Paired bounded FP16 | Change |
|---|---:|---:|---:|
| Grid blocks | 1024 | 512 | -50% |
| NCU duration | 51.07 us | 32.51 us | 1.571x |
| Executed instructions | 38.27 M | 22.61 M | -40.93% |
| Registers/thread | 18 | 28 | +10 |
| Static shared memory/block | 1.06 KiB | 2.19 KiB | +1.13 KiB |
| Achieved occupancy | 90.54% | 91.06% | +0.52 point |
| Eligible warps/scheduler | 2.97 | 2.59 | -0.38 |
| DRAM throughput | 38.03% | 59.30% | +21.27 points |

Source-correlated emitted SASS confirms algebraic packing. Dynamic
`F2FP.F16.F32.PACK_AB` falls from 3.93M to 0.66M, `HADD2.F32` unpack work
from 3.93M to 0.79M, `FSETP`/`FSEL` overflow checks from 2.62M each to 0.52M
each, and shuffle butterflies from 1.31M to 0.66M. The safe path replaces
that scalar representation with 0.85M native packed FP16 add/sub operations.
The added absolute-bound reduction accounts for 1.31M shuffle-down and part
of the retained FP32-add stream. Reports remain outside Git at
`/tmp/qvq_v3_phase3_{plain,packed}_low_w3_m512.ncu-rep`.

## Phase 4: reuse-8 for the unsplit down projection

At M512 and above the Llama 8192-to-2048 down projection no longer uses
split-K, but the single-child wrapper still selected reuse-4. Phase 4 selects
the already exact reuse-8 implementation for unsplit single children when M
is at least 512 and divisible by 128. M128 stays on reuse-4 because reducing
its down grid from 64 to only 32 blocks would underfill the 132-SM H100.

The operation is the same Phase-1 algebra applied to a one-segment payload:
one decoded P32 fragment supplies eight independent M16 WGMMA row tiles. It
does not alter checkpoint packing, K accumulation, reduction, or output
recovery. The all-rate production down-shape test requires reuse-8 to be bit
exact to reuse-4 and verifies CUDA Graph replay at M512.

A same-session W3 boundary sweep measured every requested intermediate row
bucket before promotion:

| M | MLP MKN (gate/up; down) | Reuse-4 down | Reuse-8 down | Speedup | Better than last |
|---:|---|---:|---:|---:|:---:|
| 512 | 512x2048x8192; 512x8192x2048 | 373.802 us | 371.251 us | 1.0069x | Yes |
| 1024 | 1024x2048x8192; 1024x8192x2048 | 728.554 us | 720.122 us | 1.0117x | Yes |
| 2048 | 2048x2048x8192; 2048x8192x2048 | 1449.098 us | 1429.728 us | 1.0135x | Yes |
| 4096 | 4096x2048x8192; 4096x8192x2048 | 2832.893 us | 2798.778 us | 1.0122x | Yes |

The committed all-rate artifact uses executable commit `060b9554` and the
same strict idle admission, CUDA Graph replay, and CUDA-event protocol as the
earlier phases.

| Weight | MLP MKN (gate/up; down) | Phase 3 | Phase 4 | Speedup | vs merged main | vs Marlin W4 | vs Machete W4 | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| W2 | 128x2048x8192; 128x8192x2048 | 107.290 us | 107.254 us | 1.000x | 1.141x | 0.740x | 0.680x | Yes |
| W2 | 512x2048x8192; 512x8192x2048 | 369.216 us | 363.238 us | 1.016x | 1.131x | 0.457x | 0.322x | Yes |
| W2 | 4096x2048x8192; 4096x8192x2048 | 2794.330 us | 2739.302 us | 1.020x | 1.152x | 0.497x | 0.318x | Yes |
| W2.5 | 128x2048x8192; 128x8192x2048 | 109.856 us | 109.987 us | 0.999x | 1.126x | 0.722x | 0.663x | No |
| W2.5 | 512x2048x8192; 512x8192x2048 | 373.018 us | 370.003 us | 1.008x | 1.143x | 0.448x | 0.316x | Yes |
| W2.5 | 4096x2048x8192; 4096x8192x2048 | 2804.438 us | 2770.259 us | 1.012x | 1.157x | 0.492x | 0.314x | Yes |
| W3 | 128x2048x8192; 128x8192x2048 | 109.011 us | 109.501 us | 0.996x | 1.119x | 0.725x | 0.666x | No |
| W3 | 512x2048x8192; 512x8192x2048 | 372.019 us | 368.525 us | 1.009x | 1.140x | 0.450x | 0.317x | Yes |
| W3 | 4096x2048x8192; 4096x8192x2048 | 2834.234 us | 2774.938 us | 1.021x | 1.147x | 0.491x | 0.314x | Yes |
| W3.5 | 128x2048x8192; 128x8192x2048 | 107.578 us | 108.288 us | 0.993x | 1.140x | 0.733x | 0.673x | No |
| W3.5 | 512x2048x8192; 512x8192x2048 | 377.283 us | 364.938 us | 1.034x | 1.188x | 0.455x | 0.320x | Yes |
| W3.5 | 4096x2048x8192; 4096x8192x2048 | 2869.104 us | 2736.461 us | 1.048x | 1.200x | 0.498x | 0.318x | Yes |

The code changes only M512 and M4096 in this matrix; all eight changed cells
improve, with a `1.0212x` geometric mean. M128's three `No` cells and one
sub-percent `Yes` are unchanged-code run variation. Across the complete
matrix Phase 4 is `1.0131x` over Phase 3 and `1.1485x` cumulatively over
merged PR-112 main. Maximum dense-oracle error remains `1.073e-6`.

### Exact-commit Nsight Compute and SASS audit

The W3 M512 reuse-4 and reuse-8 down kernels coexist in commit `060b9554` and
were profiled with identical inputs and sections on the physical H100.

| Metric | Reuse-4 down | Reuse-8 down | Change |
|---|---:|---:|---:|
| Grid blocks | 256 | 128 | -50% |
| NCU duration | 98.24 us | 93.92 us | 1.046x |
| Executed instructions | 50.61 M | 31.48 M | -37.79% |
| Registers/thread | 93 | 147 | +54 |
| Achieved occupancy | 15.01% | 7.78% | -7.23 points |
| Eligible warps/scheduler | 0.84 | 0.41 | -0.43 |
| DRAM throughput | 6.24% | 6.48% | +0.24 point |

The emitted SASS has unchanged 2.10M WGMMA instructions. Decoder and address
work is nearly halved: `IMAD` 9.91M to 5.02M, `PRMT` 6.33M to 3.17M,
`LOP3` 5.51M to 2.79M, shared word and halfword loads 4.19M each to 2.10M,
and 64-bit funnel shifts 2.10M to 1.05M. Thus the win is decoded-fragment
reuse despite lower occupancy, not changed tensor work. Reports remain at
`/tmp/qvq_v3_phase4_down_reuse{4,8}_w3_m512.ncu-rep`.

## Phase 5: packed multiblock shared-input transform at large M

The grouped gate/up path must compute the shared 2048-wide input transform
before P32. Large M still used one 1024-thread CTA per row, with scalar
half-to-float arithmetic and a full-row shared-memory butterfly. The exact
decode multiblock transform already factorizes the same ascending transform
into eight independent 256-column low tiles followed by three tile-index high
stages. Phase 5 extends that implementation through M4096.

For logical M above 16 the operator returns a multiple-of-64 row allocation,
updates only logical rows in its low grid, and has the high grid write exact
zero to padded rows. Native `half2` add/sub retains both FP16 lanes' original
rounding boundary. No butterfly is reassociated or reordered.

Focused M32, M129, and M512 tests require exact FP16 bits versus the original
transform, exact-zero padding, and exact CUDA Graph replay. The existing
prescale-overflow test also remains exact and finite.

| Weight | MLP MKN (gate/up; down) | Phase 4 | Phase 5 | Speedup | vs merged main | vs Marlin W4 | vs Machete W4 | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| W2 | 128x2048x8192; 128x8192x2048 | 107.254 us | 103.923 us | 1.032x | 1.177x | 0.764x | 0.704x | Yes |
| W2 | 512x2048x8192; 512x8192x2048 | 363.238 us | 353.501 us | 1.028x | 1.162x | 0.469x | 0.332x | Yes |
| W2 | 4096x2048x8192; 4096x8192x2048 | 2739.302 us | 2653.184 us | 1.032x | 1.190x | 0.512x | 0.322x | Yes |
| W2.5 | 128x2048x8192; 128x8192x2048 | 109.987 us | 106.458 us | 1.033x | 1.163x | 0.746x | 0.687x | Yes |
| W2.5 | 512x2048x8192; 512x8192x2048 | 370.003 us | 360.288 us | 1.027x | 1.173x | 0.460x | 0.326x | Yes |
| W2.5 | 4096x2048x8192; 4096x8192x2048 | 2770.259 us | 2684.922 us | 1.032x | 1.193x | 0.506x | 0.319x | Yes |
| W3 | 128x2048x8192; 128x8192x2048 | 109.501 us | 106.397 us | 1.029x | 1.152x | 0.746x | 0.687x | Yes |
| W3 | 512x2048x8192; 512x8192x2048 | 368.525 us | 361.469 us | 1.020x | 1.163x | 0.459x | 0.325x | Yes |
| W3 | 4096x2048x8192; 4096x8192x2048 | 2774.938 us | 2697.293 us | 1.029x | 1.180x | 0.504x | 0.317x | Yes |
| W3.5 | 128x2048x8192; 128x8192x2048 | 108.288 us | 105.088 us | 1.030x | 1.175x | 0.756x | 0.696x | Yes |
| W3.5 | 512x2048x8192; 512x8192x2048 | 364.938 us | 354.000 us | 1.031x | 1.225x | 0.469x | 0.332x | Yes |
| W3.5 | 4096x2048x8192; 4096x8192x2048 | 2736.461 us | 2662.349 us | 1.028x | 1.233x | 0.510x | 0.321x | Yes |

All twelve cells improve. Phase 5 is `1.0292x` over Phase 4 and `1.1821x`
cumulatively over merged PR-112 main. Marlin and Machete W4 geometric ratios
are `0.5620x` and `0.4177x`; they remain figurative W4 baselines. Maximum and
maximum-row-mean dense-oracle absolute error remain `1.073e-6` and
`1.585e-7`.

### Exact-commit Nsight Compute and SASS audit

Commit `1dd7ca29` contains both the original and multiblock input transforms.
Matched M512 captures on the physical H100 report:

| Metric | One-block scalar | Multiblock low | Multiblock high | Combined change |
|---|---:|---:|---:|---:|
| NCU duration | 19.90 us | 7.62 us | 4.03 us | 1.708x |
| Executed instructions | 14.30 M | 3.62 M | 0.11 M | -73.88% |
| Registers/thread | 25 | 16 | 24 | no increase |
| Achieved occupancy | 88.41% | 82.43% | 17.02% | staged grids |
| Eligible warps/scheduler | 4.21 | 2.14 | 0.10 | staged grids |
| DRAM throughput | 4.35% | 11.38% | 21.37% | higher useful rate |

The SASS representation changes materially. The low stage's leading useful
work is 229K native `HADD2`, 164K packed multiply-add, and 164K shuffle
butterflies; the high tail needs only 24.6K packed adds and 24.6K packed
multiply-adds. The original kernel instead executes 1.82M branches, 1.18M
predicate comparisons, 0.79M barrier-sync operations, and 0.67M scalar
half-unpack adds. Reports remain at
`/tmp/qvq_v3_phase5_input_{oneblock,multiblock_low,multiblock_high}_m512.ncu-rep`.
