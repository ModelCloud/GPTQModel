# QVQ P32 Hopper large-M execution

## Scope

This work extends the exact grouped P32 Hopper path beyond sixteen logical
rows.  It does not change the canonical P32 checkpoint, quantization math,
codebook, selector stream, or input/output transform order.

The development order is:

```text
M16 existing -> M32 -> M64 -> M128 -> M256 -> logical M up to 4096
```

M32 is the first promotion boundary.  It is exactly two M16 tensor-core row
tiles and covers every logical row count from 17 through 32 without returning
to the planar CUDA fallback.

## Current limitation

The production grouped runtime accepts only one through sixteen rows.  The
Hopper kernel creates a TMA tensor with shape `[16, K]`, launches no row-grid
dimension, and writes segment-major `[16, N_i]` outputs.  At M greater than
sixteen the runtime falls back to each child projection independently.  That
also discards the shared input transform and grouped output recovery that are
part of A41/R0 execution.

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

Reuse will be promoted only if it beats the generalized row-grid baseline.  A
larger CTA is not assumed to be faster: register pressure, shared-memory use,
WGMMA dependency depth, and reduced occupancy are measured on the physical
H100.

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
coverage adds M512 through M4096.

Each result row reports the realistic `(M, K, N)` projection geometry,
latency distribution, effective throughput, dense-oracle errors, W4 Marlin
ratio, W4 Machete ratio, and whether it improves on the last committed
benchmark.  A regression is always recorded as `No`.
