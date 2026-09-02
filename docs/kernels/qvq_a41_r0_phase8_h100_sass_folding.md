# QVQ A41/R0 Phase 8: H100 SASS folding analysis

This analysis profiles the exact Phase-8 gate/up recovery, SwiGLU, and
down-input precondition path at commit `1a30c8d6` on the physical 132-SM
`NVIDIA H100` (compute capability 9.0).  The objective is to find math that
can be folded without moving or deleting any of the FP16 rounding boundaries
that define the production result.

The result is that these transforms are **latency and instruction bound, not
HBM-bandwidth bound**.  The safest first fold is to pack the two independent
FP32-to-FP16 conversions produced by every recovery butterfly into one Hopper
`F2FP.F16.F32.PACK_AB` instruction.  The next larger opportunity is to keep
two adjacent FP16 columns in one `half2` throughout the precondition
butterfly.  Fusing SiLU into the precondition low stage is useful primarily
because it deletes a launch and a materialized tensor, not because SiLU has a
large arithmetic cost.

No production benchmark was run for this document.  Nsight Compute durations
below are replay/instrumentation data and must not be compared with the CUDA
event timings in the Phase-8 benchmark matrix.

## Capture scope

The profiled MLP geometry is the Llama 3.2 1B gate/up site at `M=1`,
`K=2048`, `N=8192`, followed by the `N=8192` down precondition.  The kernels
are independent of P32 rate, so one capture covers W2, W2.5, W3, and W3.5.

The Nsight Compute capture used targeted sections rather than a full replay:

```text
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:<phase8-kernel-regex>' --launch-count 2 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --section InstructionStats --section MemoryWorkloadAnalysis \
  --force-overwrite -o <report> -- <M1 benchmark driver>
```

The standalone PyTorch SiLU launch was timed with Nsight Systems before its
targeted Nsight Compute capture:

```text
nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true \
  -o /tmp/qvq_phase8_silu_trace_v1 <8192-element FP16 SiLU driver>
```

The reports and raw CSV exports are preserved outside the worktree at:

```text
/root/qvq-profiler-artifacts/phase8-sass/
```

## Resource and issue profile

`cuobjdump --dump-resource-usage` reports no local-memory or shared-memory
spills for any Phase-8 transform kernel.

| Kernel | Grid x block at M1 | Registers/thread | Shared memory/block | Executed warp instructions | NCU duration us | Combined memory throughput | DRAM throughput | Active warps | Eligible warps/cycle | Issue active |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| recovery low | 64 x 256 | 28 | 2.080 KiB | 107,520 | 3.584 | 9.194% | 0.845% | 12.243% | 0.181 | 16.121% |
| recovery high | 8 x 64 | 48 | 1.024 KiB | 21,440 | 7.872 | 4.448% | 1.232% | 3.061% | 0.116 | 11.633% |
| precondition low | 32 x 256 | 24 | 1.552 KiB | 48,896 | 3.520 | 9.257% | 0.666% | 12.928% | 0.172 | 15.035% |
| precondition high | 4 x 64 | 40 | 1.024 KiB | 3,984 | 4.640 | 7.030% | 0.240% | 2.746% | 0.099 | 9.909% |
| PyTorch SiLU | 8 x 128 | 32 | 1.024 KiB | 6,080 | 3.360 | 9.651% | 0.364% | 6.105% | 0.077 | 7.739% |

The dominant sampled stall is `long_scoreboard`: 3.527 cycles/instruction for
recovery low, 5.680 for recovery high, 4.025 for precondition low, 7.180 for
precondition high, and 4.122 for SiLU.  However, DRAM reaches only 0.24--1.23%
of peak and the high stages expose just 8 and 4 blocks.  This is dependency
latency in tiny grids, not HBM saturation.  Increasing memory traffic or
merely changing cache policy cannot solve it.

Nsight Systems measured the standalone 8192-element FP16 SiLU at a median of
`1.536 us` over 15 launches (`1.536--2.208 us`).  Nsight Compute reports
`3.360 us` because of profiling overhead; that value is not a production
latency.

## Executed SASS composition

The counts below are executed warp instructions from the targeted M1 capture.

| Kernel | Main executed opcodes | Interpretation |
|:--|:--|:--|
| recovery low | `LEA 9,728`; `F2FP 7,680`; `HADD2 7,680`; `FSETP 7,680`; `FSEL 7,680`; `LDS 7,168`; `STS 7,168`; `BSYNC 7,168`; `FADD 6,656` | Overflow-preserving FP16 emulation and shared-memory butterfly dominate. |
| recovery high | `F2FP 4,096`; `HADD2 3,584`; `FSETP 3,584`; `FSEL 3,584`; `FADD 3,072`; `LDG 1,536`; `FMUL 512`; `STG 512` | Scalar narrowing/overflow guards are 14,848 of 21,440 instructions, or 69.3%. |
| precondition low | `HADD2 4,608`; `F2FP 4,096`; `LEA 3,840`; `LDS 3,584`; `STS 3,584`; `BSYNC 3,584`; `FADD 3,328`; `BAR 2,304` | FP16 unpack/repack plus shared butterfly and synchronization dominate. |
| precondition high | `HADD2 1,280`; `FADD 1,280`; `F2FP 640`; `LDG 256`; `STG 256` | The transform core is 3,200 of 3,984 instructions, or 80.3%. |
| PyTorch SiLU | `FFMA 2,560`; `MUFU 512`; `SHF 288`; `HFMA2 256`; `HADD2 256`; `FADD 256`; `F2FP 128`; `LDG 32`; `STG 32` | Arithmetic is compact already; the independent launch is the larger problem. |

`HADD2.F32` in these sequences is also used to unpack a half lane into FP32;
its presence does not mean that the current kernels are already performing
two independent butterfly pairs per thread.

## Fold 1: paired overflow-preserving narrowing

Recovery must reproduce the following scalar operation after every FP32
butterfly result:

\[
R(x)=
\begin{cases}
\operatorname{fp32}(\operatorname{fp16}_{rn}(x)), &
  \text{if the narrowed value is finite},\\
x, & \text{otherwise}.
\end{cases}
\]

The current code evaluates `R(a+b)` and `R(a-b)` separately.  SASS shows a
four-instruction narrowing/guard chain for each result:

```text
F2FP.F16.F32.PACK_AB dst, RZ, value
HADD2.F32              narrowed, -RZ, dst.H0_H0
FSETP.GEU              overflow, |narrowed|, +INF
FSEL                    result, value, narrowed, overflow
```

Hopper's conversion accepts two independent FP32 sources.  A minimal CUDA
13.3 SM90 compilation of `__floats2half2_rn(a, b)` emitted:

```text
F2FP.F16.F32.PACK_AB R0, R5, R2
HADD2.F32 R13, -RZ, R0.H0_H0
HADD2.F32 R0,  -RZ, R0.H1_H1
FSETP.GEU ... |R13|, +INF
FSETP.GEU ... |R0|,  +INF
FSEL ... a ...
FSEL ... b ...
```

Therefore two scalar conversions can share one `F2FP` while their finite
tests and fallbacks remain independent:

```cpp
half2 packed = __floats2half2_rn(sum, difference);
float narrowed_sum = __low2float(packed);
float narrowed_difference = __high2float(packed);
sum = isfinite(narrowed_sum) ? narrowed_sum : sum;
difference = isfinite(narrowed_difference) ? narrowed_difference : difference;
```

This is byte-exact, including finite FP16 values, signed zero, infinities,
and the existing overflow fallback.  It changes neither addition order nor a
rounding boundary.

In recovery high, 3,584 `F2FP` instructions belong to guarded rounds and 512
to final scalar FP16 stores.  Pairing the butterfly sum/difference and pairing
adjacent independent epilogue rounds has an upper bound of 1,792 fewer
executed instructions: **8.36% of the entire high kernel**.  Recovery low has
the same repeated scalar pattern, although its predication and shared-memory
work make a static percentage less reliable; its actual delta must be read
from a new capture.

This is the first implementation candidate because it is local, exact, uses
no extra memory, and does not reduce the number of active blocks.

## Fold 2: two-column `half2` precondition butterfly

The precondition path stores FP16 after every butterfly.  Unlike recovery, it
does not need to retain an overflowing FP32 result.  The high-stage compiler
currently handles one column per thread.  For each butterfly pair it performs
two half-to-float unpacks, two scalar add/sub operations, and one packed
float-to-half conversion:

\[
2\ HADD2.F32 + 2\ FADD + 1\ F2FP.
\]

Those counts exactly explain the high-stage transform core:

```text
1,280 HADD2 + 1,280 FADD + 640 F2FP = 3,200 instructions.
```

Instead, one thread can own two adjacent within-tile columns as a `half2`.
For every high-stage tile pair:

```cpp
half2 a = values2[tile];
half2 b = values2[peer];
values2[tile] = __hadd2(a, b);
values2[peer] = __hsub2(a, b);
```

Each half lane performs the same IEEE FP16 add/sub and round as the scalar
reference.  Two column-pairs then need two packed half operations rather than
ten unpack/scalar/repack instructions.  Loads and stores can also become
32-bit operations.

The risk is launch geometry: high would fall from 64 to 32 threads per block.
It keeps four blocks per row and the same total mathematical outputs, but the
already tiny grid has fewer warps.  Promotion therefore depends on CUDA-event
latency, not instruction count alone.

The low stage can use the same two-column ownership for butterfly bits 2
through 128.  Bit 1 mixes the two lanes inside a `half2`, so it needs one
explicit lane swap/shuffle before producing packed sum/difference.  The
padded shared-memory mapping must be redesigned in half2 units and checked
for bank conflicts.  High-stage `half2` is the lower-risk prototype and
should precede low-stage conversion.

## Fold 3: SiLU into precondition low

The current path materializes:

```text
FP16 recovered gate
  -> standalone PyTorch SiLU
  -> FP16 activated gate allocation
  -> precondition-low load
```

At M1 that intermediate causes 16 KiB of gate reads plus 16 KiB of activated
gate writes and costs another kernel launch.  Precondition low can instead
load the recovered FP16 gate, execute the same SiLU instruction sequence,
round the result to FP16 in a register, and continue with the existing
FP16-rounded gate/up product:

```text
gate half
  -> exact PyTorch-compatible SiLU math
  -> half round (unchanged boundary)
  -> multiply by up half
  -> half round (unchanged boundary)
  -> down SU / normalization / low Hadamard
```

The maximum directly observed standalone saving is approximately the Nsight
Systems median of `1.536 us`, plus avoiding the temporary allocation/traffic.
The fused kernel must reproduce PyTorch's CUDA 13.3 `FFMA`/`MUFU` reciprocal
refinement sequence; substituting a different sigmoid expression is not an
exact implementation.  This should be attempted after the local paired-round
fold because it changes the public kernel input and runtime coordination.

## What should not be folded yet

A complete recovery-high to SiLU to precondition-low kernel looks attractive
but crosses the factorization boundary.  Recovery high owns one fixed local
column across 32 tiles, while precondition low owns all 256 local columns of
one tile.  A one-CTA fusion serializes the 32 tiles and recreates the Phase-7
under-parallelization.  Recomputing recovery high independently for each
output tile duplicates high-transform work, while a cluster/distributed
shared-memory transpose adds synchronization to a path whose prior wider
cluster experiments regressed.

Similarly, splitting the high-stage 64-thread block into two 32-thread blocks
does not create more total warps and introduces duplicated state or a new
reduction boundary.  The SASS evidence favors reducing dependency-chain
instructions first.

## Experiment order and gates

| Priority | Experiment | Expected mechanism | Required promotion gates |
|--:|:--|:--|:--|
| 1 | Pair guarded recovery rounds | One `F2FP.PACK_AB` for two FP32 results | Byte-exact recovery/MLP, exact repeatability and graph replay; fewer executed `F2FP`; positive CUDA-event full-MLP result |
| 2 | `half2` precondition high | Two columns per thread; packed FP16 add/sub and 32-bit loads/stores | Byte-exact precondition; no new bank conflict or spills; positive isolated and full-MLP timing |
| 3 | Fuse exact SiLU into precondition low | Delete one launch and activated-gate materialization | Byte-exact versus PyTorch SiLU lifecycle, graph safety, positive full-MLP timing |
| 4 | `half2` precondition low | Packed bits 2--128 plus explicit bit-1 lane exchange | Byte-exact; source/SASS proof; bank-conflict and latency improvement |
| 5 | Cross-stage transpose/cluster fusion | Eliminate workspace and another launch | Only after the preceding local folds; must beat the non-cluster path in CUDA-event timing |

For every promoted experiment, the formal H100 benchmark must report the full
W2/W2.5/W3/W3.5 by M=1/2/4/8/16 matrix, MKN, ratios versus Marlin W4 and
Machete W4, and `Better` versus the immediately preceding committed result.
