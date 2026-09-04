# QVQ P32 Hopper large-M execution

## Scope

This work extends the exact grouped P32 Hopper path beyond sixteen logical
rows.  It does not change the canonical P32 checkpoint, quantization math,
codebook, selector stream, or input/output transform order.

The development order is:

```text
M16 existing -> M32 -> M64 -> M128 -> M256 -> M512 -> M1024
    -> M2048 -> M4096 -> autotuned row multiplexing above M4096
```

M32 is the first promotion boundary.  It is exactly two M16 tensor-core row
tiles and covers every logical row count from 17 through 32 without returning
to the planar CUDA fallback.

## Production state

The native grouped kernels accept one through 4096 logical rows.  They pad
M17-M32 to M32 and larger non-bucket values to a multiple of M64, then use one
native row grid.  M32 reuses each decoded fragment for two M16 tiles; M64 and
larger use four tiles per CTA.  Ordinary P32 children use the same one-segment
path, so the MLP down projection no longer returns to planar GEMV.  On the
measured H100, the production runtime accepts larger logical M by autotuning
and replaying those exact native kernels over contiguous row chunks.

For the narrow Llama `8192 -> 2048` down projection, measured H100 ordered
split counts are `8` through M64, `4` through M128, `2` through M256, and `1`
thereafter.  Full gate/up, activation/product, and down execution remains CUDA
Graph replayable at every promoted boundary through M4096.

## Mathematical contract

For one legal group with shared input transform

```text
T = H(X * SU_shared)
```

each child remains

```text
Z_i = T @ Q_i
Y_i = recover_i(Z_i, SV_i, bias_i, output_hadamard_i)
```

where `Q_i` is reconstructed from that child's unchanged P32 trellis and bank
selectors.  Large-M execution partitions only the row axis:

```text
T_r = T[16*r : min(16*(r+1), M), :]
Z_i[rows_r, :] = T_r @ Q_i
```

No K reduction is reordered by row tiling.  A child that uses ordered split-K
keeps the same left-to-right partial reduction within every output element.
Consequently a generalized row grid must be bit-exact to invoking the same
M16 Hopper operation independently for each padded row tile.

The dense accuracy reference remains the FP32 P32 Torch oracle.  Promotion
requires maximum absolute error at most `2e-3`, plus reported mean absolute
error, root-mean-square error, and relative L2 error.

## Native grid design

The first implementation keeps the proven M16 register-sourced WGMMA tile and
adds a runtime row-tile coordinate:

```text
row_tiles = ceil(M / 16)

work item = (segment, N64 tile, split-K tile, row tile)
```

The input TMA descriptor becomes `[padded_M, K]`.  Each consumer CTA selects
one `[16, 256]` input tile at `row_tile * 16`.  P32 weights and bank selectors
remain shared across the row grid.  Output and ordered-partial offsets include
the row tile while preserving segment-major contiguous tensors:

```text
child output:  [padded_M, N_i]
child partial: [split_i, padded_M, N_i]
```

The M32 milestone uses two row tiles in one native launch.  The same runtime
row-grid mechanism can cover M64, M128, M256, and larger logical M without
compiling a separate kernel for every value.

## Second-stage reuse experiment

The generalized row grid removes fallback and launch duplication but still
decodes a P32 weight tile once per M16 row tile.  M64 is the first major
redesign target because a CTA may profitably decode one weight fragment and
feed it to multiple independent M16 WGMMA B fragments.  Candidate row buckets
are:

```text
M <= 16   one M16 consumer tile
M <= 32   two M16 consumer tiles
M <= 64   four M16 consumer tiles
M <= 128  measured M64 or M128 CTA tile
M > 128   fixed internal row tile across one grid launch
```

Reuse is promoted only if it beats the generalized row-grid baseline.  A
larger CTA is not assumed to be faster: register pressure, shared-memory use,
WGMMA dependency depth, and reduced occupancy are measured on the physical
H100.

## H100 N128 by M64 gate/up tile

Profiling the promoted four-row reuse kernel at W3, M512 showed that decode
reuse was still correct but the single consumer warpgroup left too few
independent warps ready to issue.  The N64 by M64 CTA used 160 threads, 93
registers per thread, 94.98 KiB dynamic shared memory, and achieved only 15.05%
occupancy with 0.80 eligible warps per scheduler.  It also staged the same
input tile independently for adjacent N64 output blocks.

The promoted H100 gate/up tile for M at least 128 combines two adjacent N64
blocks while retaining four M16 row tiles:

```text
one producer warp
  + two independent 128-thread WGMMA consumer warpgroups
  + one shared set of four M16 input tiles
  = one N128 by M64 CTA
```

Each consumer keeps its own P32 words, bank selector, decoded register
fragment, and FP32 accumulator.  Only the input TMA stages and pipeline are
shared.  Therefore the transformation does not combine child projections,
change P32 decode, or alter accumulation order.  The grid has half as many
CTAs and exactly the same output ownership as two N64 by M64 CTAs.

Matched Nsight Compute at W3 M512 measured the inner gate/up kernel falling
from 208.06 to 179.74 microseconds.  Achieved occupancy increased from 15.05%
to 26.36%, active warps per scheduler from 2.41 to 4.22, and eligible warps
per scheduler from 0.80 to 1.49.  Executed instructions also fell from 105.87
million to 100.96 million because one producer/pipeline serves two consumers.
The complete MLP improved from 492.06 to 465.40 microseconds in the matched
W3 M512 check.  M64 did not benefit, so the dispatch gate begins at M128.

## Coalesced FP32 accumulator stores

The RS-WGMMA accumulator layout assigns eight non-contiguous output values to
each consumer thread.  A direct scalar store preserves the exact tensor but,
for the N128 by M64 gate/up kernel, Nsight Compute measured 2,162,688 global
sectors where 1,114,112 were ideal.  The 1,048,576 excess sectors came from
the final FP32 accumulator stores rather than P32 weight traffic.

After all WGMMA stages and pipeline releases complete, the promoted kernel
reclaims the dead TMA input buffer.  Each independent N64 consumer scatters
one 16 by 64 accumulator tile into a private 4 KiB FP32 shared-memory region:

```text
register accumulator coordinate (row, permuted column)
    -> shared[row, canonical column]
    -> aligned float4 global store
```

All 256 consumers first meet at one named barrier so neither consumer group
can overwrite pipeline storage while the other still uses it.  The two
consumer groups then use separate 128-thread named barriers around each
shared transpose.  No arithmetic, accumulator ownership, output ordering, or
rounding boundary changes.  The output is bit-exact to the direct-store
kernel and is safe under CUDA Graph capture and replay.

Matched W3 M512 Nsight Compute reduced global sectors to the ideal 1,114,112
and eliminated all 1,048,576 excessive sectors.  Inner-kernel duration fell
from 179.744 to 177.088 microseconds.  The transpose increases executed
instructions from 100,962,957 to 102,828,993 and excessive shared wavefronts
from 8,388,480 to 9,174,912, but eligible warps per scheduler rise from 1.491
to 1.541 and the global-store improvement wins overall.  W2.5 does not
benefit because its depth-three decoder schedule makes the added barriers
more expensive, so W2.5 retains direct stores.  The measured coalesced path
is limited to W2, W3, and W3.5 at the existing H100 N128 by M64 gate/up gate.

## Template and binary-size budget

Transition widths W2, W2.5, W3, and W3.5 already instantiate the decoder.
The row count and row-tile index are runtime values, so the first large-M
implementation adds zero transition-rate specializations and multiplies the
current kernel template matrix by `1`, not by the number of M buckets.

Any later multi-row-tile-per-CTA design may add at most the measured row reuse
factors rather than one specialization per logical M.  Compile time, object
size, register count, shared memory, and spill count must be recorded before
promotion.

## Correctness and safety gates

Every promoted bucket must pass:

- exact equality to the established tiled-M16 Hopper reference;
- dense P32 FP32 oracle maximum absolute error at most `2e-3`;
- repeatability across multiple launches and seeds;
- odd logical rows at 17, 31, 33, 63, 65, 127, 129, and 255;
- no reads or writes outside the logical result;
- CUDA Graph capture and repeated replay without allocation or host decisions;
- unchanged R0 legality and exact plain fallback for unsupported devices;
- H100-only performance promotion based on runtime device properties.

## Benchmark contract

Benchmarks use only the physical NVIDIA H100, reject foreign compute
processes, and require three zero-utilization samples before setup plus another
idle check before timing.  Timing uses warmed CUDA Graph replay and CUDA
events.  The initial matrix contains M16, M32, M64, M128, and M256; later
coverage includes M512, M1024, M2048, M4096, and M8192.

Each result row reports the realistic `(M, K, N)` projection geometry,
latency distribution, effective throughput, dense-oracle errors, W4 Marlin
ratio, W4 Machete ratio, and whether it improves on the last committed
benchmark.  A regression is always recorded as `No`.

## Autotuned row multiplexing above M4096

The validated low-level Hopper kernels retain their M4096 bound.  A grouped
projection or fused MLP with more logical rows is partitioned at the runtime
boundary instead of weakening that kernel contract:

```text
logical rows M > 4096
    -> contiguous row chunks of T
    -> existing exact grouped projection or fused-MLP path per chunk
    -> concatenate each child/output along rows in original order
```

The candidate downward-multiplexing targets are `T = {512, 1024, 2048,
4096}`.  On the first eager call, each candidate is warmed, captured in its
own CUDA Graph, and measured by CUDA events over repeated graph replays.  The
median per-replay time selects the winner.  Grouped-projection and fused-MLP
plans use separate cache scopes.  Plans are cached by CUDA device, dtype,
next-power-of-two logical-M bucket, input width, child output widths, optional
down output width, and P32 transition rate.  This bounds tuning cardinality
while keeping operation, rate, and geometry decisions independent.

CUDA-event creation and synchronization never occur during graph capture.  A
warm model with a cold row-plan cache captures the conservative 4096-row
target without caching it; a later eager invocation can still tune.  Once
tuned, capture takes only the cached branch.  Canonical payload construction
must still be warmed before capture under the existing R0 lifecycle rule.

Chunking changes only independent row scheduling.  It does not change K
accumulation, P32 decoding, Hadamard order, FP16 rounding, or output row
order.  M4097 testing requires exact equality across all four candidate
targets and repeated CUDA Graph replay for both generic QKV and the full MLP.
The physical H100 M8192 dense-oracle benchmark selects 4096 for W2 through
W3.5.  Maximum absolute error is `1.14e-6` for the full MLP and `1.71e-5` for
grouped QKV.
