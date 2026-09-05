# Hopper large-M P32 v4

This phase starts from merged PR #114 (`9dcaf07e`) and targets another
1.25x full-MLP speedup on the physical 132-SM NVIDIA H100. All timings use
warmed CUDA Graph replay measured by CUDA events; CPU and container scheduling
are outside the timed region.

## Reuse-11 gate/up kernel

The merged gate/up kernel decodes one P32 weight tile and reuses it for eight
M16 activation tiles, or 128 rows. At logical M512 that produces four row CTA
waves. The H100 shared-memory limit cannot hold twelve double-buffered input
tiles, but it can hold eleven:

```text
11 * M16 * K256 * sizeof(fp16) * 2 stages = 180,224 bytes
P32 trellis, selectors, levels, and barriers  = ~40 KiB
total dynamic shared memory                  = ~221 KiB
```

The runtime therefore pads only the grouped gate/up inner operand from 512 to
528 rows and launches three M176 CTA waves:

```text
merged:    512 / 128 = 4 row waves
reuse-11:  528 / 176 = 3 row waves
```

The extra sixteen zero rows add 3.125% tensor-core work, while the P32 decode
and address stream is executed 25% fewer times. The first 512 outputs retain
the same K traversal, WGMMA instruction sequence, FP32 accumulation order, and
FP16 recovery boundary. Padded rows are discarded before recovery.

The specialization is deliberately narrow: physical NVIDIA H100, grouped
Llama 2048-to-8192 gate/up, unsplit children, and logical M512 or M4096. Other
devices, shapes, rates, and row counts retain the merged production path.

At M4096, padding to M4224 similarly replaces 32 reuse-8 row waves with 24
reuse-11 waves:

```text
merged:    4096 / 128 = 32 row waves
reuse-11:  4224 / 176 = 24 row waves
```

The same 3.125% zero-row tensor work therefore removes 25% of repeated P32
decode/address CTAs. The runtime slices back to the 4096 logical rows before
the fused gate/up recovery and down precondition, so neither model-visible
shape nor the FP32/FP16 rounding contract changes.

## Rejected decode lookup

An exact 1 MiB table mapping `(bank, 16-bit state)` directly to two packed
FP16 levels removed affine PGC arithmetic but changed regular shared-memory
loads into data-dependent cache lookups. W3/M512 full MLP regressed from
337.982 to 665.511 microseconds. The experiment was removed without a commit.

## Rejected one-stage reuse-16

A single-stage M256 kernel fit in shared memory and was exact and graph-safe,
but exposed TMA latency. W3/M512 regressed from 337.982 to 389.771 microseconds.
It was removed without a production commit.

## M512 result

| Weight | Gate/up M×K×N | Down M×K×N | Merged main | Reuse-11 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| W2 | 512×2048×8192 | 512×8192×2048 | 331.784 µs | 319.615 µs | 1.038× | 0.515× | 0.370× | Yes |
| W2.5 | 512×2048×8192 | 512×8192×2048 | 340.222 µs | 335.828 µs | 1.013× | 0.490× | 0.352× | Yes |
| W3 | 512×2048×8192 | 512×8192×2048 | 337.982 µs | 325.754 µs | 1.038× | 0.505× | 0.363× | Yes |
| W3.5 | 512×2048×8192 | 512×8192×2048 | 335.194 µs | 328.096 µs | 1.022× | 0.501× | 0.361× | Yes |

Geometric speedup over merged main is **1.028x**. Maximum absolute error over
the four complete-MLP cases is `1.0133e-6`, below the locked `2e-3` limit.
The low-level and runtime tests require bit-exact output and CUDA Graph replay.

Artifact:
`artifacts/qvq_hopper_large_m/v4_reuse11_m512_candidate.json`.

This is the first v4 win, not the complete 1.25x target. Nsight Compute and
source-correlated SASS were captured from exact post-merge revision
`ae90283c` with Nsight Compute hardware-counter replay.

| Metric | Merged reuse-8 | Reuse-11 | Change |
| --- | ---: | ---: | ---: |
| Gate/up kernel duration | 147.104 µs | 127.168 µs | -13.6% |
| Executed warp instructions | 63,657,414 | 54,759,720 | -14.0% |
| CTAs | 512 | 384 | -25.0% |
| Registers/thread | 139 | 168 | +20.9% |
| Dynamic shared memory/CTA | 172.800 KiB | 221.952 KiB | +28.4% |
| Eligible warps/scheduler/cycle | 0.688 | 0.742 | +8.0% |
| DRAM throughput | 6.97% | 8.12% | +1.15 points |
| Local/shared spills | 0 / 0 | 0 / 0 | unchanged |

The SASS change matches the intended algebraic reuse. `PRMT` and
`WARPGROUP` execute exactly 25% fewer times; `LDS`, `SHF`, and `LOP3` fall
23–25%; and `IMAD` falls 15.8%. Tensor work increases only for the sixteen
padded rows: `HGMMA` rises 3.1%. The kernel remains instruction/scheduler
limited rather than HBM limited.

NCU report:
`artifacts/qvq_hopper_large_m/profiles/v4_reuse11_w3_m512_ae90283c_ncu.ncu-rep`.

## W2.5 coalesced recovery store

The earlier reuse-8 kernel excluded W2.5 from the shared-memory transpose used
to coalesce FP32 accumulator stores. Re-testing under the reuse-11 geometry
shows the balance has changed: enabling the same exact transpose for W2.5
reduces its M512 full MLP from 335.828 to **324.326 microseconds**, a further
**1.035x**, with bit-exact eager and CUDA Graph output. Against merged main the
cumulative W2.5 gain is **1.049x**. The benchmark remains behind Marlin and
Machete at 0.507x and 0.363x respectively.

Artifact:
`artifacts/qvq_hopper_large_m/v4_w25_reuse11_coalesced_candidate.json`.

The exact `c5052486` NCU capture reports 124.768 microseconds, 54,884,628
executed warp instructions, 166 registers/thread, 217.856 KiB dynamic shared
memory, 0.743 eligible warps/scheduler/cycle, 7.22% DRAM throughput, and zero
local/shared spills. Its dominant SASS remains decoder/address math:
12.61M `IMAD`, 8.76M `R2UR`, 6.75M `LDS`, 4.76M `PRMT`, 4.39M `SHF`, and
4.37M `LOP3`, versus 4.33M `HGMMA`. This confirms that the coalesced store did
not trade the measured gain for spills or HBM pressure; the next large win
still requires sharing or eliminating decode/address work.

NCU report:
`artifacts/qvq_hopper_large_m/profiles/v4_w25_reuse11_coalesced_c5052486_ncu.ncu-rep`.

## M4096 reuse-11 result

| Weight | Gate/up M×K×N | Down M×K×N | Merged main | Reuse-11 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| W2 | 4096×2048×8192 | 4096×8192×2048 | 2447.298 µs | 2344.182 µs | 1.044× | 0.592× | 0.390× | Yes |
| W2.5 | 4096×2048×8192 | 4096×8192×2048 | 2484.994 µs | 2390.654 µs | 1.039× | 0.581× | 0.382× | Yes |
| W3 | 4096×2048×8192 | 4096×8192×2048 | 2503.269 µs | 2409.439 µs | 1.039× | 0.576× | 0.379× | Yes |
| W3.5 | 4096×2048×8192 | 4096×8192×2048 | 2468.720 µs | 2397.978 µs | 1.030× | 0.579× | 0.381× | Yes |

All four rows passed warmed CUDA Graph replay. The maximum absolute error is
`1.0729e-6`, and the runtime test requires the sliced logical M4096 output to
be bit-exact to ordinary child execution. No persistent or temporary decoded
weight cache is introduced.

Artifact:
`artifacts/qvq_hopper_large_m/v4_reuse11_m4096_candidate.json`.

The exact `83f29ef9` NCU capture verifies that this is an instruction-reuse
win rather than timing noise:

| Metric | Reuse-8 | Reuse-11 | Change |
| --- | ---: | ---: | ---: |
| Gate/up kernel duration | 1092.640 µs | 960.704 µs | -12.1% |
| Executed warp instructions | 509,056,242 | 437,936,958 | -14.0% |
| CTAs | 4096 | 3072 | -25.0% |
| Registers/thread | 139 | 168 | +20.9% |
| Dynamic shared memory/CTA | 172.800 KiB | 221.952 KiB | +28.4% |
| Eligible warps/scheduler/cycle | 0.700 | 0.753 | +7.4% |
| DRAM throughput | 11.21% | 13.15% | +1.94 points |
| Local/shared spills | 0 / 0 | 0 / 0 | unchanged |

Source-correlated SASS agrees with the row-wave algebra: `PRMT` and
`WARPGROUP` fall exactly 25%; `LDS` falls 24.8%; `SHF` and `LOP3` fall
23.2–23.4%; and `IMAD` falls 15.8%. The zero-row padding raises `HGMMA` by
exactly 3.125%. The kernel remains instruction/scheduler limited, with only
13.15% DRAM activity.

NCU report:
`artifacts/qvq_hopper_large_m/profiles/v4_reuse11_w3_m4096_83f29ef9_ncu.ncu-rep`.
