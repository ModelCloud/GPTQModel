# BM64/BN32 SASS and algebraic reduction waves 27–32

These waves target the exact lossless window representation on the historical
F6 seed-7 snapshot. The checkpoint is
`/root/qvq-results/calibration-fisher-frontier-wave14-v1/llama32-1b-f6_yaqa125x_seed7`.
No weights, trellis words, bank IDs, or decoded values were changed. All runs
used A100-class SM80 devices and explicit UUID assignment through the queue
dispatcher.

The eight physical assignments were:

| Queue worker | GPU UUID | Device |
|---:|---|---|
| 0 | `GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2` | PG506-230 |
| 1 | `GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855` | PG506-232 |
| 2 | `GPU-8be4c651-4058-83df-154b-291f1b86add8` | PG506-230 |
| 3 | `GPU-471ecdd7-a171-4d5c-d61f-a1802dc76e4c` | PG506-230 |
| 4 | `GPU-14ab23f1-a785-e9df-bbb5-215547154e3c` | PG506-230 |
| 5 | `GPU-3a4bf14f-fa28-df88-f6e8-00ef6b13d473` | PG506-230 |
| 6 | `GPU-737e2423-874a-23a4-1126-dfbe3e77c294` | PG506-230 |
| 7 | `GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28` | PG506-230 |

## Exact algebraic A/B wave 28

Wave 28 compared the production BM64/BN32 scalar decode with two exact source
rewrites: factored tile addresses and predicated bank XOR. It covered eight
real projections and M = 1, 16, 128, 256, 512, and 2048. Every one of the 32
reports completed and every local teacher/window gate passed.

The pooled median full-layer speedup over the production window was:

| Address mode | Bank mode | M=1 | M=16 | M=128 | M=256 | M=512 | M=2048 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | multiply | 0.959× | 0.963× | 0.968× | 0.991× | 1.166× | 1.291× |
| factored | multiply | 0.979× | 0.932× | 0.986× | 0.967× | 1.182× | 1.292× |
| baseline | predicated | 0.976× | 0.969× | 0.965× | 0.973× | 1.163× | 1.288× |
| factored | predicated | 0.962× | 0.996× | 0.944× | 0.945× | 1.161× | 1.295× |

The address and bank rewrites are exact but do not approach 3×. The raw
reports are in
[wave 28 results](results/bm64bn32-algebra-wave28/).

## Launch and decode frontier wave 29

Wave 29 held BM64/BN32/split-K=1 fixed and swept the launch configuration and
an exact pair-level codebook fold on layer-0 `down_proj`. The pair LUT maps
each post-bank 16-bit state directly to its two FP16 outputs, eliminating the
integer mix and codebook-index arithmetic. It uses 256 KiB of FP16 values and
does not alter the serialized P32 representation.

All 72 cases across the eight configurations passed both the FP32-teacher and
production-window local gates. The per-configuration medians were:

| Decode | Warps | Stages | M=1 | M=16 | M=128 | M=512 | M=2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar | 2 | 2 | 0.954× | 0.905× | 0.915× | 1.259× | 1.250× |
| scalar | 4 | 1 | 1.012× | 0.982× | 0.923× | 1.198× | 1.260× |
| scalar | 4 | 3 | 0.989× | 0.959× | 0.914× | 1.228× | **1.286×** |
| scalar | 8 | 1 | 0.983× | 0.978× | 0.959× | 1.148× | 1.166× |
| pair LUT | 2 | 2 | 0.970× | 1.000× | 0.977× | 1.090× | 1.041× |
| pair LUT | 4 | 1 | 0.840× | 1.010× | 1.044× | 1.071× | 1.025× |
| pair LUT | 4 | 3 | 0.946× | 0.969× | 0.935× | 1.083× | 1.036× |
| pair LUT | 8 | 1 | 0.813× | 0.918× | 0.946× | 0.939× | 0.972× |

The scalar four-warp/three-stage arm is the best repeatable launch choice in
this wave. It remains a 1.29× layer result, so it does not support a 3× claim.
The raw reports are in
[wave 29 results](results/bm64bn32-frontier-wave29/).

## Nsight wave 30: scalar winner

Wave 30 profiled the scalar BM64/BN32/warp-4/stage-3 arm at M=2048 for the
same eight projections. The command collected instruction classes, pipeline
activity, occupancy, launch resources, memory workload, scheduler state, and
source counters. The median across projections was:

| Metric | Median |
|---|---:|
| Registers/thread | 64 |
| Dynamic shared memory/CTA | 8,192 bytes |
| Theoretical occupancy | 50.0% |
| Achieved occupancy | 45.6% |
| Integer SASS instructions | 14,722,879,488 |
| Memory SASS instructions | 3,231,711,232 |
| FP16 SASS instructions | 536,870,912 |
| HMMA instructions | 16,777,216 |
| Long-scoreboard stall | 13.9% |
| Barrier stall | 11.7% |
| Math-pipe throttle | 14.1% |
| MIO throttle | 5.4% |

The static `nvdisasm` view of a representative 64-register scalar cubin had
392 SASS instructions, including 96 IMAD, 24 IADD3, 64 LOP3, 35 right-shift
operations, 19 global loads, 4 global stores, and 4 HMMA instructions. The
dynamic profile is the authoritative execution count; static counts describe
one compiled shape and are included to identify the SASS mix.

The dominant opportunity is integer/address/decode work rather than Tensor
Core arithmetic. The raw Nsight reports and CSVs are in
[wave 30 results](results/bm64bn32-ncu-wave30/).

## Nsight wave 31: pair-LUT fold

Wave 31 profiled the exact pair-LUT arm with the same geometry. It reduced
integer execution but moved the critical path to random LUT loads:

| Metric | Scalar | Pair LUT | Change |
|---|---:|---:|---:|
| Registers/thread | 64 | 68 | +4 |
| Achieved occupancy | 45.6% | 41.3% | −4.3 points |
| Integer SASS instructions | 14.72B | 10.03B | −31.9% |
| Memory SASS instructions | 3.23B | 3.23B | unchanged |
| HMMA instructions | 16.78M | 16.78M | unchanged |
| Long-scoreboard stall | 13.9% | 50.5% | +36.6 points |
| Barrier stall | 11.7% | 9.1% | −2.6 points |
| Math-pipe throttle | 14.1% | 1.1% | −13.0 points |
| MIO throttle | 5.4% | 15.4% | +10.0 points |

This is a useful negative result: algebraic reduction alone is insufficient
when it replaces cheap arithmetic with a random 256 KiB table lookup. The raw
reports and CSVs are in
[wave 31 results](results/bm64bn32-ncu-wave31/).

## Cache-qualified pair LUT wave 32

Wave 32 tested `.ca` and `.cg` cache qualifiers for the exact pair LUT across
the eight projections and all nine row counts. All 72 cases passed the local
gates. The pooled medians were:

| Cache policy | M=1 | M=16 | M=128 | M=512 | M=2048 |
|---|---:|---:|---:|---:|---:|
| `.ca` | 0.953× | 1.008× | 0.994× | 1.008× | 1.033× |
| `.cg` | 0.966× | 0.963× | 0.963× | 0.871× | 0.827× |

Neither policy is a speedup path. The raw reports are in
[wave 32 results](results/bm64bn32-pairlut-cache-wave32/).

## Resident window-word waves 33–34

The exact resident-word mode loads the two BN32 window tiles and their bank
bytes once per K16 iteration, then gathers each state from the local tile
image. It uses the same standard window bytes and the same decode arithmetic;
the only change is where the words are staged.

Wave 33 covered the eight projections and all nine row counts. All 72 cases
passed both local gates. The pooled full-layer speedup over production window
was:

| M | 1 | 2 | 4 | 8 | 16 | 32 | 128 | 512 | 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Resident words | 0.962× | 0.961× | 0.935× | 1.030× | 1.022× | 1.023× | 0.960× | 1.148× | 1.249× |

Wave 34 profiled the same resident-word arm at M=2048. Compared with the
scalar winner, it achieved the intended instruction/resource changes:

| Metric | Scalar | Resident words | Change |
|---|---:|---:|---:|
| Registers/thread | 64 | 56 | −8 |
| Achieved occupancy | 45.6% | 51.7% | +6.1 points |
| Integer SASS instructions | 14.72B | 12.73B | −13.6% |
| Memory SASS instructions | 3.23B | 3.33B | +3.1% |
| HMMA instructions | 16.78M | 16.78M | unchanged |
| Long-scoreboard stall | 13.9% | 15.9% | +2.0 points |
| Barrier stall | 11.7% | 17.4% | +5.7 points |
| Math-pipe throttle | 14.1% | 8.6% | −5.5 points |
| MIO throttle | 5.4% | 13.3% | +8.0 points |

The lower register footprint and lower integer count do not translate into a
3× result because gather, memory, and synchronization costs become visible.
The resident-word mode is therefore an exact resource-reduction reference,
not the current production candidate. Raw reports are in
[wave 33 results](results/bm64bn32-resident-wave33/) and
[wave 34 results](results/bm64bn32-resident-ncu-wave34/).

## Resident launch frontier waves 35–36

Wave 35 swept resident-word launches over all eight mapped GPUs, using the
eight projections `l0/l1 × q/gate/up/down` and all nine row counts. All 72
cases passed both local gates and the resident output matched the standard
window reference within the recorded FP16 boundary. The best resident arm was
`BM64/BN32`, four warps, two stages, at about 1.27× at M=2048; it remained
slower than the scalar production winner for several projections and low-M
shapes. The representative fused-vs-window medians were:

| Arm | M=1 | M=16 | M=128 | M=512 | M=2048 |
|---|---:|---:|---:|---:|---:|
| w2/s1, l0 q | 1.043× | 0.993× | 0.964× | 0.988× | 1.148× |
| w2/s2, l1 q | 0.970× | 0.988× | 0.980× | 0.938× | 1.230× |
| w4/s1, l0 gate | 0.922× | 0.996× | 0.990× | 1.119× | 1.203× |
| w4/s2, l1 gate | 0.886× | 0.920× | 1.037× | 1.185× | 1.267× |
| w4/s3, l0 up | 0.944× | 0.913× | 0.955× | 1.137× | 1.264× |
| w8/s1, l1 up | 0.935× | 0.932× | 0.970× | 1.052× | 1.140× |
| w8/s2, l0 down | 0.942× | 1.010× | 0.935× | 1.097× | 1.092× |
| w8/s3, l1 down | 0.857× | 0.964× | 0.956× | 1.092× | 1.101× |

Wave 36 profiled the strongest resident launch, w4/s2 at M=2048, across the
same eight projections. The medians were:

| Metric | Resident w4/s2 |
|---|---:|
| Registers/thread | 56 |
| Dynamic shared memory/CTA | 8,192 bytes |
| Achieved occupancy | 51.6% |
| Integer SASS instructions | 12.19B |
| Memory SASS instructions | 3.33B |
| FP16 SASS instructions | 536.87M |
| HMMA instructions | 16.78M |
| Long-scoreboard stall | 16.3% |
| Barrier stall | 19.5% |
| Math-pipe throttle | 8.1% |
| MIO throttle | 14.2% |

Relative to the scalar w4/s3 profile, resident w4/s2 lowers registers from
64 to 56 and integer execution by roughly 17%, while increasing barrier and
MIO pressure. Its measured runtime still does not approach 3×. Raw JSON,
CSV, and profiler logs are in [wave 35 results](results/bm64bn32-resident-frontier-wave35/)
and [wave 36 results](results/bm64bn32-resident-ncu-wave36/).

## BM128 cross-row reuse wave 37

Wave 37 tested `BM128/BN32`, four warps, two stages, standard exact window
payloads, across eight projections and all nine row counts. This is a
cross-row reuse ceiling test: the decoder and weight tile are shared by twice
as many activation rows per CTA, while the P32 bits, state transitions,
codebook, transforms, and FP32 accumulation remain unchanged. All 72/72 cases
passed both local gates and no decoded output differed from the production
window beyond the recorded FP16 boundary.

The pooled fused full-layer speedup versus production window was:

| M | 1 | 2 | 4 | 8 | 16 | 32 | 128 | 512 | 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM128/BN32 | 0.974× | 0.960× | 0.949× | 0.977× | 0.949× | 0.943× | 0.911× | 1.220× | 1.398× |

The best individual M=2048 result was 1.43× on layer-0 down projection, but
the pooled result is below the existing BM64/BN32 scalar arm. BM128 therefore
does not provide the required 3× path: larger row tiles improve amortization
at high M, but the larger accumulator and CTA footprint impose enough layout
and scheduling cost to erase the gain. The complete per-projection JSON and
logs are in [wave 37 results](results/bm128-wave37/).

Wave 38 profiled the BM128 arm at M=2048 across the same eight projections.
The pooled Nsight medians were:

| Metric | BM64 scalar w4/s3 | BM128 w4/s2 | Change |
|---|---:|---:|---:|
| Registers/thread | 64 | 87 | +23 |
| Dynamic shared memory/CTA | 8,192 | 16,384 | 2× |
| Achieved occupancy | 45.6% | 30.3% | −15.3 points |
| Integer SASS instructions | 14.72B | 7.37B | −49.9% |
| Memory SASS instructions | 3.23B | 1.75B | −45.7% |
| HMMA instructions | 16.78M | 16.78M | unchanged |
| Long-scoreboard stall | 13.9% | 23.2% | +9.3 points |
| Barrier stall | 11.7% | 9.6% | −2.1 points |
| Math-pipe throttle | 14.1% | 6.5% | −7.6 points |
| MIO throttle | 5.4% | 9.6% | +4.2 points |

The instruction reduction is real reuse, but it is purchased with 87 registers,
16 KiB shared memory, lower occupancy, and higher long-scoreboard pressure.
This explains why the runtime gain is only 1.398× pooled at M=2048. Raw
profiles are in [wave 38 results](results/bm128-ncu-wave38/).

## Exact CUDA transform fold wave 39

Wave 39 added an exact transform-mode control to the layer scorecard. With
`cuda-fused`, the existing CUDA Hadamard primitive absorbs the FP32 SU
pre-scale and SV post-scale around the BM64/BN32 operator; `separate` remains
the historical reference path. Direct standalone checks on the Llama 3.2 1B
shapes showed bitwise-equal input and output transform results. The P32 window
payload and decoded values are unchanged.

The fair comparison used CUDA-fused transforms for both production window and
BM64/BN32, across eight projections and all nine row counts. All 72/72 cases
passed both local gates. The pooled fused full-layer speedup was:

| M | 1 | 2 | 4 | 8 | 16 | 32 | 128 | 512 | 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM64/BN32 w4/s3 | 0.536× | 0.530× | 0.530× | 0.539× | 0.544× | 0.847× | 1.163× | 1.539× | 1.787× |

This exposes the remaining shape boundary clearly: the direct decode/MMA
kernel is much slower for M≤16, crosses the window near M=128, and reaches
1.787× pooled at M=2048. The local fold removes roughly 0.7 ms from each
2048-row transform in the layer-0 down example, but it benefits both arms and
does not by itself reach 3×. Complete scorecards are in
[wave 39 results](results/bm64bn32-fused-transform-wave39/).

Wave 40 ran the matching Nsight profile on all eight projections at M=2048.
The profiler is intentionally bracketed around the decode/MMA call by
`--profile-fused`, so the transform fold does not alter the measured inner
kernel. Its pooled metrics reproduce the BM64 resource profile:

| Metric | CUDA-fused transform arm |
|---|---:|
| Registers/thread | 64 |
| Dynamic shared memory/CTA | 8,192 bytes |
| Achieved occupancy | 45.6% |
| Integer SASS instructions | 14.72B |
| Memory SASS instructions | 3.23B |
| HMMA instructions | 16.78M |
| Long-scoreboard stall | 13.8% |
| Barrier stall | 11.7% |
| Math-pipe throttle | 14.1% |
| MIO throttle | 5.3% |

This confirms that SU/Hadamard/SV folding improves boundary work without
changing BM64 decode/MMA resource use. It does not hide the decoder under
MMA; the next 3× experiment must fuse the transform work into the CTA or
reduce the decode/MMA critical path. Raw profiles are in
[wave 40 results](results/bm64bn32-transform-ncu-wave40/).

## Scalar level-table cache wave 43

Wave 43 applied `.ca` to the 512-byte scalar PGC16 level-table loads in the
standard BM64/BN32 w4/s3 kernel. This is an exact SASS cache-policy test; all
72/72 projection/row-count cases passed the local gates. It did not improve
the kernel:

| M | 1 | 2 | 4 | 8 | 16 | 32 | 128 | 512 | 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Scalar `.ca` | 0.979× | 0.954× | 0.934× | 0.962× | 0.966× | 0.962× | 0.966× | 1.170× | 1.284× |

The default scalar load remains the selected exact path. Raw results are in
[wave 43 results](results/bm64bn32-scalar-ca-wave43/).

## BM64 register-cap wave 44

Wave 44 tested Triton `maxnreg` values 0, 48, 56, 64, 72, 80, and 96 (with a
repeat at 56) on layer-0 `down_proj`, all nine row counts, and all eight GPUs.
Every one of the 72 cases passed the exact local gates. The representative
full-layer speedups versus production window were:

| Cap | M=1 | M=16 | M=128 | M=512 | M=2048 |
|---:|---:|---:|---:|---:|---:|
| 0, uncapped | 0.925× | 0.977× | 0.964× | 1.240× | 1.290× |
| 48 | 0.948× | 1.110× | 0.949× | 1.199× | 1.283× |
| 56 | 0.922× | 1.097× | 0.974× | 1.221× | 1.286× |
| 64 | 1.008× | 0.914× | 0.941× | 1.148× | 1.288× |
| 72 | 0.975× | 0.996× | 0.966× | 1.165× | 1.281× |
| 80 | 0.841× | 1.014× | 1.004× | 1.186× | 1.241× |
| 96 | 1.020× | 0.869× | 1.271× | 1.067× | 1.208× |

The cap does not improve the uncapped kernel. The 48-register arm raised the
M=2048 inner latency from about 1.87 ms to 2.27 ms, consistent with spill or
compiler scheduling cost; larger caps also regressed. Raw reports are in
[wave 44 results](results/bm64bn32-maxnreg-wave44/).

## Exact shift/mask address fold wave 45

Wave 45 replaced runtime divide/modulo forms in the standard scalar decode
path with equivalent shifts, masks, and a predicated final-word wrap. The
change preserves the exact state extraction and PGC16 mapping. All 72/72
projection/row-count cases passed the local gates, but the SASS rewrite was
slower:

| M | 1 | 2 | 4 | 8 | 16 | 32 | 128 | 512 | 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Shift/mask fold | 0.982× | 0.949× | 0.979× | 0.958× | 0.953× | 0.958× | 0.974× | 1.136× | 1.247× |

The compiler already strength-reduced the relevant constant arithmetic, while
the explicit predication changed scheduling and address generation. The
default scalar source remains selected. Raw results are in
[wave 45 results](results/bm64bn32-bitfold-wave45/).

## Decision and next target

The current exact dispatch remains production window for M < 512 and the
BM64/BN32 direct decode/MMA kernel for M >= 512. The best measured exact
large-prefill layer arm in these waves is 1.29× over production window; the
previous full-model repeat remains the stronger 1.423× M=2048 result.

The profiling evidence prioritizes three directions for a genuine 3× attempt:

1. stage window words and bank metadata once per output tile, then reuse them
   across multiple activation-row tiles without reloading or rematerializing;
2. lower the 64-register tile footprint while keeping the BM64 reuse level;
3. fuse SU, input Hadamard, decode/MMA, output Hadamard, and SV so transform
   traffic and launch boundaries no longer dominate the full linear layer.

The pair-LUT path is retained as an algebraic reference and rejected as a
runtime candidate. No quality exception or gate relaxation was used.
