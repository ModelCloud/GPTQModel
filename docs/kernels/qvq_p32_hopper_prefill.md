# QVQ P32 Hopper prefill kernels

## Objective

The decode-oriented P32 kernel reconstructs one compressed weight fragment in
registers and reuses it across at most four M16 activation tiles.  That is a
good latency trade for decode, but at large prefill row counts the same P32
fragment is reconstructed once for every M64 slab.  For `M=8192`, that is 128
reconstructions of the same weight fragment.

The prefill project removes that repeated work while preserving canonical P32
checkpoint storage.  FP16 and FP8 are execution types for a decoded tile; they
are not new checkpoint formats.

The production controls are:

```text
exact row-reuse P32       numerical and storage reference
folded persistent FP8     large-prefill performance ceiling
Marlin W4 / Machete W4    external latency controls
```

Every benchmark must label the persistent folded-FP8 control separately from a
native compressed-P32 result.

## Mathematical contract

For a legal R0 group, the shared input transform and child projections are

```text
T = H(X * SU_shared)
Z_i = T @ Q_i
Y_i = recover_i(Z_i, SV_i, bias_i, output_hadamard_i)
```

`Q_i` is reconstructed solely from the child's canonical P32 trellis, selector
bytes, alternative bank, and PGC16 codebook.  Grouping concatenates the inner
matrices only along N:

```text
Q_G = [Q_0 | Q_1 | ...]
Z_G = T @ Q_G
```

Child output boundaries, recovery transforms, scales, and biases remain
independent.  The FP16 phase accumulates into FP32.  The FP8 phase also
accumulates into FP32, but adds explicit activation and decoded-level rounding;
therefore it is judged against the dense-P32 oracle rather than required to be
bit-exact to the FP16 phase.

## Phase 0: locked controls

Before changing the prefill path, retain measurements for:

- exact grouped P32 row multiplexing;
- the optional persistent folded-FP8 path;
- three independent Marlin W4 projections;
- three independent Machete W4 projections;
- the dense-P32 Torch oracle.

Use physical H100 only, warmed CUDA Graph replay, CUDA events, and
`M={512,1024,2048,4096,8192,16384}`.  Record median, mean, p95, effective
TFLOP/s, allocated and persistent bytes, mean absolute error, relative L2, and
maximum absolute error.

## Phase 1: on-demand folded FP16 materialization

Phase 1 is an intentionally simple control implementation:

```text
canonical grouped P32 payload
        |
        | one parallel decode over K16 x N64 tiles
        v
temporary FP16 Q_G [K, N_total]
        |
        | fold shared input H/SU and child-local output H/SV
        v
temporary FP16 W_effective [K, N_total]
        |
        | one grouped FP16 x FP16 -> FP32 matrix multiply
        v
recovered FP32 Y_G [M, N_total]
```

The decoder maps one 128-thread CTA to one `K16 x N64` tile.  Its four warps
own the four canonical `N16` P32 tiles.  Each warp reconstructs its unchanged
P32 states and stores the resulting FP16 levels into canonical `[K,N]` order.
There is no full-weight cache: the dense inner/effective matrix is a
per-forward temporary (or graph-private allocation during capture).

For Llama 3.2 1B grouped QKV, the temporary is:

```text
2048 * (2048 + 512 + 512) * 2 bytes = 12 MiB
```

The first inner-only prototype used the ordinary activation-side input/output
transforms.  On W3, M8192 it measured approximately 17.2 microseconds for P32
decode, 158.5 microseconds for the grouped FP16-to-FP32 matrix multiply, and
1,954 microseconds for child recovery.  Recovery therefore dominated the
operation.  The retained Phase-1 control algebraically folds the transforms
into the temporary weight before the matrix multiply, just as the persistent
FP8 control does, while keeping FP16 as the execution weight type.

This phase answers two questions before a much more complex kernel is written:

1. Is decoding once per forward enough to close the large-M gap?
2. How much latency is left in input/output transforms and the dense tensor-core
   matrix multiply?

It is not the final memory design.  Promotion initially requires explicit
opt-in, CUDA Graph replay, maximum error at most `2e-3`, and a complete
end-to-end win over exact row multiplexing.

### Physical-H100 Phase-1 result

The benchmark used 20 warmups, 31 CUDA-event samples, and 10 CUDA Graph replays
per sample.  `Better than last` compares with exact compressed-P32 row
multiplexing at the same rate and M.

| Rate | M x K x aggregate N | FP16 us | vs exact P32 | vs Marlin W4 | vs Machete W4 | Better than last | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| W2 | 8192 x 2048 x 3072 | 588.720 | 3.840x | 0.523x | 0.390x | Yes | 4.469e-6 |
| W2.5 | 8192 x 2048 x 3072 | 589.104 | 3.893x | 0.522x | 0.390x | Yes | 4.450e-6 |
| W3 | 8192 x 2048 x 3072 | 587.958 | 3.902x | 0.523x | 0.390x | Yes | 4.520e-6 |
| W3.5 | 8192 x 2048 x 3072 | 590.397 | 3.940x | 0.521x | 0.389x | Yes | 4.035e-6 |
| W2 | 16384 x 2048 x 3072 | 826.240 | 5.459x | 0.768x | 0.547x | Yes | 4.654e-6 |
| W2.5 | 16384 x 2048 x 3072 | 828.202 | 5.557x | 0.766x | 0.546x | Yes | 4.443e-6 |
| W3 | 16384 x 2048 x 3072 | 828.038 | 5.529x | 0.766x | 0.546x | Yes | 4.678e-6 |
| W3.5 | 16384 x 2048 x 3072 | 828.291 | 5.578x | 0.766x | 0.546x | Yes | 4.562e-6 |

The final FP16 effective matrix is 12 MiB per active grouped-QKV call and is
not retained after the call.  The allocator probe measured a 240--272 MiB
incremental peak because Phase 1 still constructs FP32 folding and transpose
intermediates.  Phase 2 must remove those intermediates; the 12 MiB logical
matrix size is not a claim that the prototype's complete allocator footprint
is only 12 MiB.

### Phase-1 Nsight Compute and generated-instruction result

A focused physical-H100 Nsight Compute capture profiled the committed W3
decoder specialization at `M=8192`, `K=2048`, aggregate `N=3072`.  The decoder
launch is independent of M because it constructs the temporary weight once.

| Metric | Measured value |
| --- | ---: |
| Kernel time | 15.680 us |
| Grid | 48 x 8 CTAs |
| Threads per CTA | 128 |
| Executed warp instructions | 2,759,424 |
| Registers per thread | 40 |
| Static shared memory per CTA | 7,296 bytes |
| Local memory | 0 bytes |
| SM throughput | 17.71% |
| DRAM throughput | 6.33% |
| Eligible warps per cycle | 0.263 |
| Average warp latency per issued instruction | 13.28 cycles |
| Long-scoreboard latency per issued instruction | 5.70 cycles |

The executed-SASS attribution was generated from the same Nsight Compute
report rather than inferred from CUDA source.  The largest opcode families
were `LOP3` (371,712), address `LEA`/`LEA.HI.X` (417,024 combined), shared
loads (196,608), permutation (196,608), level-table loads (196,608), FP16
stores (196,608), and wide multiply-add address formation (193,536).

This exposes more address common-subexpression and bit-extraction work that
could eventually be removed.  It is not the next critical path: 15.68 us is
only about 2.7% of the 587.96 us W3/M8192 operation.  Even deleting the decoder
entirely would leave Phase 1 slower than Marlin and Machete.  Phase 2 therefore
targets the FP32 transpose/Hadamard/scale/cast preparation boundaries and their
allocator traffic before another decoder algebra pass.

## Phase 2: native folded-FP16 preparation

Phase 1 proves that removing activation-side recovery is mandatory.  Phase 2
therefore first removes Python/Torch materialization boundaries from effective
weight preparation:

```text
P32 decode -> K-axis Hadamard/SU -> N-axis child Hadamard/SV -> FP16 store
```

The transform is global along K and each child N axis, so it cannot be replaced
by a purely local K16/N64 operation.  Implement it as an explicit native
multi-kernel plan with stable scratch ownership, vectorized transposes, and
CUDA-Graph-safe allocations.  Profile the decode, transpose, two Hadamard axes,
scale, cast, and GEMM separately.  Promotion requires lower full-operation
latency and allocator peak than Phase 1.

The first native implementation uses four ordered CUDA stages:

1. P32 decode writes FP16 directly as `[N,K]`, eliminating Phase 1's separate
   half-to-float cast and initial transpose;
2. one block per N column performs the normalized K-axis FP32 Hadamard and SU
   scale with the same ascending butterfly order as Phase 1;
3. a padded `32x32` shared-memory tile transposes the FP32 result to `[K,N]`;
4. one segmented grid performs each child's optional normalized N-axis FP32
   Hadamard, applies its own SV, and rounds directly to the final FP16 weight.

The three child segments share a launch but retain separate output widths,
scales, and output-H flags.  The implementation creates no concatenated scale
tensor and adds no new transition-bit specialization: the existing four P32
decoder specializations receive a runtime output-layout flag.  For Llama QKV,
the explicit scratch contract is 72 MiB (one decoded FP16 plane, two FP32
planes, and one final FP16 plane), compared with Phase 1's measured 240--272
MiB allocator peak.

### Physical-H100 Phase-2 result

The formal run used 20 warmups, 31 CUDA-event samples, and 10 CUDA Graph
replays per sample.  `Better than last` compares native Phase 2 with Phase 1
at the same rate and M.

| Rate | M x K x aggregate N | Phase-2 us | vs Phase 1 | vs Marlin W4 | vs Machete W4 | Better than last | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| W2 | 8192 x 2048 x 3072 | 504.749 | 1.162x | 0.605x | 0.456x | Yes | 4.469e-6 |
| W2.5 | 8192 x 2048 x 3072 | 506.512 | 1.164x | 0.603x | 0.454x | Yes | 4.450e-6 |
| W3 | 8192 x 2048 x 3072 | 503.002 | 1.171x | 0.608x | 0.457x | Yes | 4.520e-6 |
| W3.5 | 8192 x 2048 x 3072 | 500.902 | 1.172x | 0.610x | 0.459x | Yes | 4.035e-6 |
| W2 | 16384 x 2048 x 3072 | 743.024 | 1.116x | 0.854x | 0.651x | Yes | 4.654e-6 |
| W2.5 | 16384 x 2048 x 3072 | 739.818 | 1.118x | 0.857x | 0.653x | Yes | 4.443e-6 |
| W3 | 16384 x 2048 x 3072 | 743.027 | 1.112x | 0.854x | 0.651x | Yes | 4.678e-6 |
| W3.5 | 16384 x 2048 x 3072 | 743.706 | 1.111x | 0.853x | 0.650x | Yes | 4.562e-6 |

The full-call allocator probe measured a 156--156.5 MiB incremental peak and
zero retained bytes, down from Phase 1's 240--272 MiB peak.  The native folded
weight is bit-exact to Phase 1 for every rate, and the complete operation stays
well inside the dense-P32 error gate.

### Phase-2 Nsight Compute and SASS result

The committed W3/M8192 CUDA Graph was profiled using Nsight Compute with each
new native stage selected by kernel-name filtering.  Times include profiling
overhead consistently within this matched capture and are used for attribution,
not as replacements for the CUDA-event table above.

| Stage | NCU time | Executed warp instructions | DRAM throughput | SM throughput | Eligible warps/cycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| transposed P32 decode | 23.968 us | 2,482,176 | 4.28% | 11.46% | 0.156 |
| K-axis Hadamard + SU | 83.808 us | 69,009,408 | 7.07% | 79.07% | 4.995 |
| FP32 tiled transpose | 15.584 us | 6,881,280 | 76.22% | 43.03% | 1.583 |
| segmented N-axis Hadamard + SV + FP16 store | 139.072 us | 81,788,928 | 16.86% | 56.30% | 2.593 |

The source-correlated SASS shows the transform stages are dominated by generic
butterfly control and synchronization rather than memory bandwidth.  The
K-axis stage executes about 8.45M branches, 4.23M barrier synchronizations,
4.03M shared stores, 4.03M shared loads, and 3.83M FP32 additions.  The N-axis
stage executes about 9.34M branches and 5.11M barrier synchronizations.  The
next direct-FP16 kernel should use compile-time tile geometry and on-chip
producer/consumer staging so this generic full-matrix preparation disappears,
rather than spending another phase tuning these temporary kernels in isolation.

## Phase 3: on-chip FP16 decode and shared-source WGMMA

Phase 3 removes the global FP16 temporary.  A producer warpgroup loads P32
windows/selectors and decodes one `N64 x K_stage` weight tile into swizzled
shared FP16.  Multiple row consumer warpgroups issue shared-source WGMMA from
that decoded tile.

The initial target is:

```text
CTA output tile: M128 x N64
consumer 0:      rows 0..63
consumer 1:      rows 64..127
K stage:         start at 64 or 128, selected by resource measurement
```

Each consumer retains four M16 FP32 accumulator fragments, matching the proven
M64 path.  The decoded weights are shared instead of retaining eight
accumulator fragments in one warpgroup, which is why this differs from the
rejected reuse-8 experiment.

The double-buffer resource budget must remain below the live device's opt-in
shared-memory limit and must allow useful residency.  Phase 2 keeps the exact
FP16 codebook values and requires the same `2e-3` dense-oracle gate; it also
reports drift relative to Phase 1 because the WGMMA reduction schedule may
differ.

Because the direct compressed kernel still produces inner-space output, its
large-M recovery must be fused or independently brought below the dense
preparation cost.  The Phase-1 attribution is a hard gate: an inner-only win is
not promotable while activation-side recovery remains near two milliseconds.

## Phase 4: on-chip FP8 execution

After Phase 3 establishes tile ownership and synchronization, replace the
decoded shared FP16 weight tile with E4M3 and convert each activation tile to
saturated E5M2.  WGMMA accumulates in FP32.

Candidate scaling is evaluated in this order:

1. one exact fixed scale for the immutable PGC16 codebook;
2. per-K-stage activation scale;
3. per-row or bounded block scale only if the first two fail accuracy.

No scale may require a second pass over the complete activation.  FP8 rounding
is an approximate inference tier and must independently pass maximum absolute
error `<=2e-3`, with mean absolute and relative-L2 error reported at every
rate and M.

FP8 halves shared decoded-weight and activation footprint and should permit a
larger row tile or more resident CTAs.  It is promoted only if the complete
operation beats Phase 3, not merely if its WGMMA instruction has higher peak
throughput.

## Phase 5: grouped QKV recovery and launch removal

Once a native inner kernel wins, fuse only boundaries shown material by an
Nsight Systems trace:

- shared input transform or its final store;
- child split/write boundaries;
- Q/K output recovery, with V retaining its folded output-axis policy;
- FP16 final stores.

This phase must keep the group descriptor role-blind: child widths, bank IDs,
split policy, and output-recovery flags come from R0 descriptors rather than
hard-coded Q/K/V semantics.

## Phase 6: shape coverage and production policy

Promote measured policies for:

- Llama 3.2 1B QKV and MLP shapes;
- Qwen3.8-27B full-attention and linear-attention projection shapes;
- logical M from 17 through 4096;
- downward row multiplexing above 4096, including 8192 and 16384.

The runtime selects exact row-reuse, native FP16 prefill, native FP8 prefill, or
optional persistent folded FP8 from immutable shape/device facts and a cached
autotune result.  Decode `M<=16` remains on the current exact kernel.

## Correctness and graph gates

Every retained phase must cover:

- byte-exact P32 payload preservation and source-version invalidation;
- decoded FP16 matrix equality to the Torch P32 reconstruction;
- output shape, dtype, finite values, and child boundaries;
- dense-P32 maximum absolute error `<=2e-3`, plus mean absolute and relative L2;
- all W2, W2.5, W3, and W3.5 branches;
- repeated eager calls and a non-default stream;
- eager warmup, CUDA Graph capture, and repeated replay;
- unsupported GPU/shape fallback without changing CPU, Ampere, H200, or decode;
- Compute Sanitizer memcheck and synchronization checks after shared-memory
  producer/consumer handoffs are introduced.

## Profiling and promotion loop

Any commit changing generated GPU instructions is followed by:

1. focused correctness on the physical H100;
2. warmed CUDA-event end-to-end timing;
3. Nsight Systems launch/span attribution for a credible winner;
4. matched Nsight Compute `SpeedOfLight`, `LaunchStats`, `Occupancy`,
   `SchedulerStats`, `WarpStateStats`, and `InstructionStats` capture;
5. source-correlated SASS aggregation and an algebraic/movement review;
6. benchmark repetition after the profiler run.

The retained result table always includes `M x K x N`, ratios versus Marlin W4
and Machete W4, and `Better than last` (`No` for a regression).
