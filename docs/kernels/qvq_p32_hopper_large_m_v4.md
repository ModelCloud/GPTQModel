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
Llama 2048-to-8192 gate/up, unsplit children, and logical M512, M1024, M2048,
or M4096. Other devices, shapes, rates, and row counts retain the merged
production path.

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

## M1024 and M2048 extension

The same exact `padded_M = 33*M/32` identity makes M1024 six M176 waves
instead of eight M128 waves and M2048 twelve instead of sixteen. The runtime
uses no new compiled kernel or cache.

| Weight | Gate/up M×K×N | Down M×K×N | Reuse-8 | Reuse-11 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| W2 | 1024×2048×8192 | 1024×8192×2048 | 635.222 µs | 614.218 µs | 1.034× | 0.545× | 0.345× | Yes |
| W2.5 | 1024×2048×8192 | 1024×8192×2048 | 652.445 µs | 623.593 µs | 1.046× | 0.537× | 0.340× | Yes |
| W3 | 1024×2048×8192 | 1024×8192×2048 | 647.979 µs | 628.754 µs | 1.031× | 0.533× | 0.337× | Yes |
| W3.5 | 1024×2048×8192 | 1024×8192×2048 | 650.342 µs | 628.926 µs | 1.034× | 0.533× | 0.337× | Yes |
| W2 | 2048×2048×8192 | 2048×8192×2048 | 1226.395 µs | 1190.070 µs | 1.031× | 0.584× | 0.378× | Yes |
| W2.5 | 2048×2048×8192 | 2048×8192×2048 | 1273.994 µs | 1221.966 µs | 1.043× | 0.569× | 0.368× | Yes |
| W3 | 2048×2048×8192 | 2048×8192×2048 | 1271.018 µs | 1219.674 µs | 1.042× | 0.570× | 0.369× | Yes |
| W3.5 | 2048×2048×8192 | 2048×8192×2048 | 1231.838 µs | 1222.861 µs | 1.007× | 0.569× | 0.368× | Yes |

All eight cells improve, replay through CUDA Graphs, and remain below
`1.133e-6` maximum absolute error. Artifact:
`artifacts/qvq_hopper_large_m/v4_reuse11_m1024_m2048_candidate.json`.

## Optional persistent FP8 MLP execution cache

The exact P32 kernels still repeat their decoder instruction stream for every
prefill.  For applications that explicitly accept FP8 inference arithmetic,
`QVQ_HOPPER_FP8_MLP_PREFILL=1` builds a source-versioned execution cache once
for the measured H100 Llama MLP geometry and uses it at M512 through M4096.
The default is disabled.  Canonical W2--W3.5 checkpoint tensors remain P32;
the cache is transient, is never serialized, and is rebuilt when any source
trellis, selector, transform scale, or output scale changes.

For each gate/up child, cache construction folds the linear operations around
the P32 inner matrix `Q`:

```text
Wfold = diag(SU) * Hin * Q * Hout * diag(SV)
```

Down uses the corresponding child-local formula.  Each folded weight is
quantized to E4M3 with one FP32 scale per output column.  Every invocation
quantizes each activation row to E4M3 with its own FP32 scale and executes the
two matrix multiplications with Hopper cuBLASLt FP8 tensor cores.  SiLU and the
gate/up product stay in their ordinary FP16 locations.  This is an approximate
FP8 execution mode, not an exact rearrangement of the P32 arithmetic.

The warm path is CUDA Graph replay safe.  Cache construction is deliberately
forbidden during capture: a cold capture takes the exact P32 path, while an
eager warmup constructs the cache before capture.  Telemetry reports separate
gate/up and down launches and retained bytes.

| Weight | Gate/up M×K×N | Down M×K×N | Cached QVQ | vs Marlin W4 | vs Machete W4 | Better than last |
| --- | --- | --- | ---: | ---: | ---: | --- |
| W2 | 512×2048×8192 | 512×8192×2048 | 109.802 µs | 1.504× | 1.072× | Yes |
| W2 | 1024×2048×8192 | 1024×8192×2048 | 178.858 µs | 1.870× | 1.244× | Yes |
| W2 | 2048×2048×8192 | 2048×8192×2048 | 361.932 µs | 1.922× | 1.267× | Yes |
| W2 | 4096×2048×8192 | 4096×8192×2048 | 734.321 µs | 1.914× | 1.254× | Yes |
| W2.5 | 512×2048×8192 | 512×8192×2048 | 108.622 µs | 1.520× | 1.083× | Yes |
| W2.5 | 1024×2048×8192 | 1024×8192×2048 | 180.955 µs | 1.848× | 1.230× | Yes |
| W2.5 | 2048×2048×8192 | 2048×8192×2048 | 364.494 µs | 1.908× | 1.258× | Yes |
| W2.5 | 4096×2048×8192 | 4096×8192×2048 | 736.330 µs | 1.909× | 1.251× | Yes |
| W3 | 512×2048×8192 | 512×8192×2048 | 108.698 µs | 1.519× | 1.083× | Yes |
| W3 | 1024×2048×8192 | 1024×8192×2048 | 180.827 µs | 1.849× | 1.231× | Yes |
| W3 | 2048×2048×8192 | 2048×8192×2048 | 364.379 µs | 1.909× | 1.258× | Yes |
| W3 | 4096×2048×8192 | 4096×8192×2048 | 735.026 µs | 1.913× | 1.253× | Yes |
| W3.5 | 512×2048×8192 | 512×8192×2048 | 108.544 µs | 1.521× | 1.084× | Yes |
| W3.5 | 1024×2048×8192 | 1024×8192×2048 | 180.989 µs | 1.848× | 1.230× | Yes |
| W3.5 | 2048×2048×8192 | 2048×8192×2048 | 364.943 µs | 1.906× | 1.256× | Yes |
| W3.5 | 4096×2048×8192 | 4096×8192×2048 | 736.924 µs | 1.908× | 1.250× | Yes |

Across all sixteen cells, the geometric speedups are **5.506x versus ordinary
QVQ**, **1.790x versus Marlin W4**, and **1.204x versus Machete W4**.  Every
cell improves over the preceding exact reuse-11 benchmark.  With unit-normal
synthetic activations, maximum absolute error versus the dense-P32 Torch oracle
is `2.753e-4`, maximum mean absolute error is `3.826e-5`, and relative L2 is
`0.06476--0.06489`.  The locked absolute `2e-3` gate passes, but the relative
error is recorded explicitly because the mode changes arithmetic.

The retained cache is 50,405,384 bytes (48.07 MiB) per Llama MLP layer:

| Payload | Bytes/layer |
| --- | ---: |
| Concatenated gate/up E4M3 weights and column scales | 33,619,972 |
| Down E4M3 weight and column scales | 16,785,412 |
| Total | 50,405,384 |

Sixteen Llama 3.2 1B layers retain 806,486,144 bytes (0.751 GiB).  There is no
per-forward decoded-weight allocation.  Artifact:
`artifacts/qvq_hopper_large_m/v4_fp8_cached_mlp_candidate.json`.

Nsight Systems 2026.4 traced five CUDA Graph replays at exact revision
`397915e4` with graph-node collection enabled.  Profiler projection totals
3.809 ms, or 761.887 microseconds per replay; the small difference from the
CUDA-event result is node-tracing overhead.

| GPU work | Instances/replay | Time/replay | Share |
| --- | ---: | ---: | ---: |
| cuBLASLt FP8 matrix multiplication | 2 | 269.841 µs | 35.9% |
| gate/up contiguous materialization | 2 | 179.992 µs | 24.0% |
| E4M3 per-row activation quantization | 2 | 150.379 µs | 20.0% |
| gate/up product | 1 | 92.044 µs | 12.3% |
| SiLU | 1 | 58.752 µs | 7.8% |

This phase changes runtime composition and cache policy but no CUDA kernel
source, so it produces no new compiler/SASS delta.  The trace instead exposes
the next algebraic boundary: the split gate/up views are copied into two
contiguous tensors before SiLU/product, accounting for nearly one quarter of
the warm operation.  Profile:
`artifacts/qvq_hopper_large_m/profiles/v4_fp8_cached_mlp_w3_m4096_397915e4_nsys.nsys-rep`.

## Strided gate/up consumers

The concatenated FP8 gate/up multiplication returns two M-by-8192 views with
a 16384-element row stride.  Materializing both views made the following SiLU
and product kernels contiguous, but copied 128 MiB at M4096.  The promoted
path instead lets TensorIterator consume both views directly.  It changes no
value, cache representation, scale, or rounding boundary.

| Weight | Gate/up M×K×N | Down M×K×N | Strided QVQ | vs Marlin W4 | vs Machete W4 | Better than last |
| --- | --- | --- | ---: | ---: | ---: | --- |
| W2 | 512×2048×8192 | 512×8192×2048 | 101.694 µs | 1.614× | 1.153× | Yes |
| W2 | 1024×2048×8192 | 1024×8192×2048 | 160.222 µs | 2.086× | 1.360× | Yes |
| W2 | 2048×2048×8192 | 2048×8192×2048 | 323.067 µs | 2.158× | 1.400× | Yes |
| W2 | 4096×2048×8192 | 4096×8192×2048 | 633.893 µs | 2.210× | 1.450× | Yes |
| W2.5 | 512×2048×8192 | 512×8192×2048 | 102.490 µs | 1.602× | 1.144× | Yes |
| W2.5 | 1024×2048×8192 | 1024×8192×2048 | 161.574 µs | 2.069× | 1.348× | Yes |
| W2.5 | 2048×2048×8192 | 2048×8192×2048 | 324.794 µs | 2.147× | 1.392× | Yes |
| W2.5 | 4096×2048×8192 | 4096×8192×2048 | 633.525 µs | 2.211× | 1.451× | Yes |
| W3 | 512×2048×8192 | 512×8192×2048 | 102.326 µs | 1.604× | 1.145× | Yes |
| W3 | 1024×2048×8192 | 1024×8192×2048 | 159.039 µs | 2.102× | 1.370× | Yes |
| W3 | 2048×2048×8192 | 2048×8192×2048 | 322.314 µs | 2.163× | 1.403× | Yes |
| W3 | 4096×2048×8192 | 4096×8192×2048 | 635.254 µs | 2.205× | 1.447× | Yes |
| W3.5 | 512×2048×8192 | 512×8192×2048 | 101.798 µs | 1.613× | 1.151× | Yes |
| W3.5 | 1024×2048×8192 | 1024×8192×2048 | 163.217 µs | 2.048× | 1.335× | Yes |
| W3.5 | 2048×2048×8192 | 2048×8192×2048 | 325.360 µs | 2.143× | 1.390× | Yes |
| W3.5 | 4096×2048×8192 | 4096×8192×2048 | 636.439 µs | 2.201× | 1.444× | Yes |

The geometric gain over the preceding cached path is **1.117x**.  Aggregate
speedups are **6.148x versus ordinary QVQ**, **1.996x versus Marlin W4**, and
**1.331x versus Machete W4**.  All sixteen cells improve.  Arithmetic and
accuracy are unchanged: maximum absolute error is `2.753e-4`, maximum mean
absolute error is `3.826e-5`, and relative L2 remains
`0.06476--0.06489`.  Artifact:
`artifacts/qvq_hopper_large_m/v4_fp8_cached_strided_mlp_candidate.json`.

The exact `b5d92ccd` Nsight Systems trace shows why the source-level deletion
wins despite strided elementwise consumers:

| GPU work | Before/replay | Strided/replay | Change |
| --- | ---: | ---: | ---: |
| Gate/up copies | 179.992 µs | 0 µs | -100% |
| SiLU | 58.752 µs | 101.189 µs | +72.2% |
| Gate/up product | 92.044 µs | 113.183 µs | +23.0% |
| E4M3 row quantization | 150.379 µs | 148.018 µs | -1.6% |
| FP8 matrix multiplications | 269.841 µs | 269.329 µs | -0.2% |
| Total projected GPU time | 761.887 µs | 642.670 µs | -15.6% |

No CUDA source changed, so SASS for every component is identical.  The win is
purely algebraic removal of two kernels and their global-memory traffic.
Profile:
`artifacts/qvq_hopper_large_m/profiles/v4_fp8_strided_mlp_w3_m4096_b5d92ccd_nsys.nsys-rep`.

## Fused stride-aware SiLU/product

The existing Triton `fused_silu_mul` operator accepts the strided gate/up views
and preserves eager FP16 narrowing exactly: SiLU is evaluated in FP32, rounded
to FP16, multiplied by up in FP32, and rounded to the contiguous FP16 output.
Using it removes a second intermediate launch and materialization.

| Weight | Gate/up M×K×N | Down M×K×N | Fused QVQ | vs Marlin W4 | vs Machete W4 | Better than last |
| --- | --- | --- | ---: | ---: | ---: | --- |
| W2 | 512×2048×8192 | 512×8192×2048 | 86.660 µs | 1.902× | 1.355× | Yes |
| W2 | 1024×2048×8192 | 1024×8192×2048 | 126.244 µs | 2.649× | 1.779× | Yes |
| W2 | 2048×2048×8192 | 2048×8192×2048 | 256.670 µs | 2.712× | 1.757× | Yes |
| W2 | 4096×2048×8192 | 4096×8192×2048 | 520.070 µs | 2.695× | 1.768× | Yes |
| W2.5 | 512×2048×8192 | 512×8192×2048 | 86.746 µs | 1.900× | 1.354× | Yes |
| W2.5 | 1024×2048×8192 | 1024×8192×2048 | 127.035 µs | 2.632× | 1.768× | Yes |
| W2.5 | 2048×2048×8192 | 2048×8192×2048 | 256.570 µs | 2.713× | 1.758× | Yes |
| W2.5 | 4096×2048×8192 | 4096×8192×2048 | 521.602 µs | 2.687× | 1.763× | Yes |
| W3 | 512×2048×8192 | 512×8192×2048 | 86.558 µs | 1.904× | 1.357× | Yes |
| W3 | 1024×2048×8192 | 1024×8192×2048 | 127.075 µs | 2.632× | 1.768× | Yes |
| W3 | 2048×2048×8192 | 2048×8192×2048 | 257.352 µs | 2.705× | 1.752× | Yes |
| W3 | 4096×2048×8192 | 4096×8192×2048 | 520.424 µs | 2.693× | 1.767× | Yes |
| W3.5 | 512×2048×8192 | 512×8192×2048 | 85.288 µs | 1.932× | 1.377× | Yes |
| W3.5 | 1024×2048×8192 | 1024×8192×2048 | 127.551 µs | 2.622× | 1.761× | Yes |
| W3.5 | 2048×2048×8192 | 2048×8192×2048 | 256.189 µs | 2.717× | 1.760× | Yes |
| W3.5 | 4096×2048×8192 | 4096×8192×2048 | 520.852 µs | 2.691× | 1.765× | Yes |

The fused operator improves the preceding strided path by **1.232x**
geometrically.  Aggregate speedups reach **7.593x versus ordinary QVQ**,
**2.461x versus Marlin W4**, and **1.653x versus Machete W4**.  All sixteen
cells improve and the error metrics are unchanged.  Artifact:
`artifacts/qvq_hopper_large_m/v4_fp8_cached_fused_silu_mlp_candidate.json`.

Exact-revision Nsight Systems reports 517.480 microseconds projected GPU time
per M4096 replay: 268.868 microseconds in two FP8 matrix multiplications,
151.045 microseconds in two per-row E4M3 quantizers, and 87.334 microseconds in
the single fused SiLU/product.  The two previous elementwise kernels totaling
214.372 microseconds are gone.

Nsight Compute 2026.2 and source-correlated SASS for the fused operator report
79.26 microseconds, 22.413 million executed warp instructions, 32 registers per
thread, no spills, 86.70% achieved occupancy, 2.24 TB/s memory throughput, and
91.47% DRAM utilization.  Long-scoreboard stalls account for 91.7% of cycles
between issued instructions.  SASS contains the expected two 128-bit global
loads, FP32 sigmoid/exponential sequence, FP16 narrowing boundary, product,
and contiguous store; it also contains general row/column address division
because N is currently a runtime value.  Since the kernel is already at 91.47%
of DRAM peak, address specialization is a measured follow-up experiment, not
an assumed win.

Profiles:

- `artifacts/qvq_hopper_large_m/profiles/v4_fp8_fused_silu_mlp_w3_m4096_98acb0cf_nsys.nsys-rep`
- `artifacts/qvq_hopper_large_m/profiles/v4_fp8_fused_silu_w3_m4096_98acb0cf_ncu.ncu-rep`

## Fused SiLU, product, and down-input row quantization

The staged cached path wrote an M-by-8192 FP16 gate/up product and then read
that tensor in a second kernel to compute the per-row E4M3 scale and payload.
The promoted kernel keeps the product in registers through the row reduction:

```text
gate FP16, up FP16
    -> SiLU in FP32
    -> round SiLU to FP16
    -> multiply in FP32
    -> round product to FP16
    -> row maximum / 448
    -> E4M3 output and FP32 row scale
```

The two FP16 narrowing points are unchanged, so the emitted E4M3 bytes and
row scales are bit-exact to the staged implementation.  This removes one
kernel launch, one 64 MiB FP16 global write, and one 64 MiB FP16 global read
at M4096.  The remaining E4M3 matrix is explicitly allocated contiguous even
when gate/up are strided views.  A unit test locks both that layout contract
and bit-exact eager/CUDA Graph replay behavior.

| Weight | Gate/up M×K×N | Down M×K×N | Fused-quant QVQ | vs Marlin W4 | vs Machete W4 | Better than last |
| --- | --- | --- | ---: | ---: | ---: | --- |
| W2 | 512×2048×8192 | 512×8192×2048 | 71.390 µs | 2.306× | 1.636× | Yes |
| W2 | 1024×2048×8192 | 1024×8192×2048 | 104.012 µs | 3.216× | 2.132× | Yes |
| W2 | 2048×2048×8192 | 2048×8192×2048 | 211.412 µs | 3.293× | 2.132× | Yes |
| W2 | 4096×2048×8192 | 4096×8192×2048 | 413.443 µs | 3.399× | 2.224× | Yes |
| W2.5 | 512×2048×8192 | 512×8192×2048 | 71.170 µs | 2.314× | 1.641× | Yes |
| W2.5 | 1024×2048×8192 | 1024×8192×2048 | 98.785 µs | 3.387× | 2.245× | Yes |
| W2.5 | 2048×2048×8192 | 2048×8192×2048 | 207.198 µs | 3.360× | 2.175× | Yes |
| W2.5 | 4096×2048×8192 | 4096×8192×2048 | 416.075 µs | 3.377× | 2.210× | Yes |
| W3 | 512×2048×8192 | 512×8192×2048 | 71.882 µs | 2.291× | 1.624× | Yes |
| W3 | 1024×2048×8192 | 1024×8192×2048 | 108.778 µs | 3.076× | 2.039× | Yes |
| W3 | 2048×2048×8192 | 2048×8192×2048 | 210.234 µs | 3.311× | 2.144× | Yes |
| W3 | 4096×2048×8192 | 4096×8192×2048 | 413.638 µs | 3.397× | 2.223× | Yes |
| W3.5 | 512×2048×8192 | 512×8192×2048 | 71.763 µs | 2.294× | 1.627× | Yes |
| W3.5 | 1024×2048×8192 | 1024×8192×2048 | 98.873 µs | 3.384× | 2.243× | Yes |
| W3.5 | 2048×2048×8192 | 2048×8192×2048 | 209.423 µs | 3.324× | 2.152× | Yes |
| W3.5 | 4096×2048×8192 | 4096×8192×2048 | 414.646 µs | 3.389× | 2.217× | Yes |

All sixteen cells improve.  Geometric speedups are **1.231x versus the prior
fused-SiLU result**, **9.336x versus ordinary P32 QVQ**, **3.033x versus
Marlin W4**, and **2.026x versus Machete W4**.  Maximum absolute error against
the dense-P32 Torch oracle is `2.753e-4`, maximum mean absolute error is
`3.826e-5`, and relative L2 is `0.06476--0.06489`.  Artifact:
`artifacts/qvq_hopper_large_m/v4_fp8_cached_fused_silu_quant_mlp_candidate.json`.

The exact-revision Nsight Systems trace at `e2aee900` contains five GPU nodes
per replay: two FP8 matrix multiplications, one gate-input row quantizer, and
one fused SiLU/product/down-input quantizer.  The separate down-input
quantizer is absent.  The projected per-replay kernel totals are 266.520
microseconds for the two matrix multiplications, 77.503 microseconds for the
fused kernel, and 27.129 microseconds for gate-input quantization.

Nsight Compute 2026.2 reports 71.58 microseconds, 25.117 million executed warp
instructions, 48 registers/thread, zero local/shared spills, 59.53% achieved
occupancy, 2.21 TB/s memory throughput, and 90.50% DRAM utilization for the
fused kernel.  Source-correlated assembly is dominated by the required SiLU,
absolute-maximum reduction, clamp, and conversion sequence: 9.535 million
`FMUL`, 3.375 million `FMNMX`, 2.130 million `MUFU`, 2.097 million `HADD2`,
and 1.049 million each of `FADD` and `F2FP`.  Address-generation instructions
are a small fraction of the stream.  With DRAM already at 90.5%, this pass
finds no safe algebraic deletion comparable to removing the intermediate;
the next useful optimization must remove another global materialization or
matrix-multiply boundary rather than merely rewriting the clamp arithmetic.

Profiles:

- `artifacts/qvq_hopper_large_m/profiles/v4_fp8_fused_silu_quant_mlp_w3_m4096_e2aee900_nsys.nsys-rep`
- `artifacts/qvq_hopper_large_m/profiles/v4_fp8_fused_silu_quant_w3_m4096_e2aee900_ncu.ncu-rep`
