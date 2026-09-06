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
