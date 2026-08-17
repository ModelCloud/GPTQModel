# QVQ B2-P32 YAQA CUDA performance investigation (2026-08-17)

## Full-sweep bottleneck attribution

The completed Llama 3.2 1B all-linear W2.5 V2B2-P32+YAQA-512 arm took
`7403.02 s`. Quantization stage timers account for `7333.97 s` (`99.1%`),
leaving about `69.05 s` (`0.9%`) for final 512-row KL/Top-K evaluation and
reporting. The observed long tail was therefore not an evaluator bottleneck.

```text
+-------------------------------+------------+----------------+
| Stage                         | Seconds    | Share of total |
+-------------------------------+------------+----------------+
| Q/K/V quantization            |     863.88 |          11.7% |
| O quantization                |     561.46 |           7.6% |
| gate/up quantization          |    3719.85 |          50.2% |
| down quantization             |    2188.78 |          29.6% |
| Final eval/report residual    |      69.05 |           0.9% |
+-------------------------------+------------+----------------+
```

Telemetry identifies complete B2 family reselection as the structural cost:
one canonical YAQA artifact plus three independent alternative-family YAQA
artifacts. Across 112 modules this produced 336 family candidates, 175,680
anti-diagonals, and 131,760 segmented-Viterbi chunks. YAQA feedback and
segmented Viterbi dominate the nested phase totals.

## Optimization

The public segmented-V2 CUDA boundary remains fail-closed. YAQA now validates
corrected targets once on-device across the complete recurrence and uses a
private structurally checked native entry point for its provisional and
constrained tail passes. This removes repeated `.item()` synchronization from
every chunk/pass. Invalid/range-overflow state is accumulated on CUDA and
checked once before returning an artifact.

The three independent family-reselection candidates are submitted to three
persistent CUDA streams. Producer and completion dependencies are represented
by CUDA events/stream waits; no temporary thread pool is created. The canonical
artifact remains first in deterministic full-proxy argmin order, preserving
canonical and lower-family tie precedence.

YAQA anti-diagonal coordinates and CUDA index tensors are immutable for a
module geometry. They are now cached per device/geometry instead of allocating
three index tensors for every anti-diagonal in every candidate. Besides
removing allocation overhead, this avoids allocator ordering pressure among
the three family streams.

Native traceback values are explicitly promoted from immutable FP16 codebook
storage to YAQA's FP32 feedback dtype before advanced indexed commits. This
also fixes the previously latent FP16-codebook YAQA dtype failure.

## A/B performance

CUDA 13.0 (`nvcc 13.0.88`), Torch `2.13.0+cu130`, `sm_80`, NVIDIA
PG506-230/232 96 GB. Times are synchronized medians.

```text
+----------------------+-------+-------+----------+----------+---------+------------------------+
| Benchmark            | Rate  | Batch | Before   | After    | Speedup | Accuracy               |
+----------------------+-------+-------+----------+----------+---------+------------------------+
| Tail-biting B2       | W1.5  | 1     | 5.411 ms | 3.233 ms |   1.67x | bit-exact              |
| Tail-biting B2       | W2    | 1     | 5.162 ms | 3.111 ms |   1.66x | bit-exact              |
| Tail-biting B2       | W2.5  | 1     | 5.077 ms | 3.082 ms |   1.65x | bit-exact              |
| Tail-biting B2       | W3    | 1     | 5.114 ms | 3.027 ms |   1.69x | bit-exact              |
| Full B2 reselect     | W2.5  | 32x64 | 142.97ms | 114.80ms |   1.25x | bit-exact cross-commit |
| Full B2 reselect     | W2.5  |128x512|1176.15ms | 804.67ms |   1.46x | bit-exact repeated     |
+----------------------+-------+-------+----------+----------+---------+------------------------+
```

The 128x512 side-stream run increases measured peak allocation from `28.68 MiB`
to `53.06 MiB`. This remains small relative to model Hessians, but must be
included in larger-shape/real-module gates.

The schedule-cache increment alone improves the post-stream 128x512 median
from `1008.18 ms` to `804.67 ms` (`1.25x`) with unchanged peak allocation and
bit-exact repeated artifacts on four GPUs.

Cross-commit checks for W1, W1.5, W2, W2.5, W3, and W3.5 found zero difference
in dense inner weight, trellis states, selectors, or family ID. This is stronger
than the quantization tolerance of `1e-6`. Inference kernels were unchanged;
their accepted numerical drift remains at most `2e-3`.

### Occupancy-aware default anti-diagonal batches

The segmented grid launches two CTAs per B2 sequence. CUDA YAQA's historical
default `trellis_batch_size=16` therefore launched only 32 CTAs at a time on
the 124-SM test GPUs. The default CUDA policy now coalesces up to 64 independent
tiles from one anti-diagonal (32 at W1, where each CTA uses 128 KiB shared
memory). Explicit non-default caller batch sizes remain unchanged.

Full B2 family-reselection A/B measurements used one 512x512 FP32 matrix,
identical SPD input/output Hessians, three alternative families, and four warm
synchronized samples per arm. The only changed input was batch 16 versus 64.

```text
+-------+-----------+-----------+---------+------------------------+
| Rate  | Batch 16  | Batch 64  | Speedup | Quantization parity    |
+-------+-----------+-----------+---------+------------------------+
| W1.5  | 2624.03ms | 1899.03ms |   1.38x | bit-exact              |
| W2    | 2107.50ms | 1586.87ms |   1.33x | bit-exact              |
| W2.5  | 2209.78ms | 1630.29ms |   1.36x | bit-exact              |
| W3    | 2144.49ms | 1647.52ms |   1.30x | bit-exact              |
+-------+-----------+-----------+---------+------------------------+
```

At W2.5 the measured peak allocation changed from `76.06 MiB` to `78.07 MiB`
for this geometry. Focused CUDA validation passed all six W1--W3.5 B2 YAQA
rates plus the multi-tile partial-batch test (`7 passed`). A pure policy test
covers the CUDA default, W1 memory cap, partial anti-diagonal, explicit caller,
non-CUDA, and Apple branches.

The canonical V2+YAQA oracle had the same underfill independently of the
segmented family paths. Its CUDA default now coalesces up to 32 tiles at W1 and
W1.5 (large shared-memory frontier) and up to 128 tiles at W2 and above.
Explicit non-default batches again remain authoritative.

```text
+----------------------+-------+-----------+-----------+---------+---------------------+
| Benchmark            | Rate  | Batch 16  | Tuned     | Speedup | Accuracy            |
+----------------------+-------+-----------+-----------+---------+---------------------+
| Canonical V2+YAQA    | W1    | 778.90 ms | 525.05 ms |   1.48x | bit-exact           |
| Canonical V2+YAQA    | W1.5  | 673.06 ms | 482.04 ms |   1.40x | bit-exact           |
| Canonical V2+YAQA    | W2    | 657.46 ms | 472.80 ms |   1.39x | bit-exact           |
| Canonical V2+YAQA    | W2.5  | 807.64 ms | 581.84 ms |   1.39x | bit-exact           |
| Canonical V2+YAQA    | W3    | 701.70 ms | 502.37 ms |   1.40x | bit-exact           |
| Full B2 reselect     | W2.5  |4002.35 ms |1697.84 ms |   2.36x | bit-exact repeated  |
+----------------------+-------+-----------+-----------+---------+---------------------+
```

The full 512x512 W2.5 comparison is a same-GPU, six-sample median against
commit `b666f398`, which already contains the segmented-family auto-batching.
Peak allocated memory remained `80.08 MiB`. The larger whole-stage gain comes
from removing dozens of small canonical allocations/launches before the three
side streams; repeated module-like calls no longer accumulate allocator and
stream-ordering pressure.

## Nsight result and remaining target

For W2.5 batch 16, Nsight Compute reports only 32 CTAs on a 124-SM GPU
(`0.13` waves/SM), 49.94% achieved occupancy, 20.11% SM throughput, and a
74.19% launch-configuration speedup estimate. The active grid recurrence is
underfilled rather than DRAM-bound. Family streams increase concurrent work,
but feedback GEMMs contend and cap the observed full-YAQA gain.

The requested 4-10x full-stage target is not yet met. The next structural step
is a family-batched YAQA recurrence: carry the three error histories in one
lockstep anti-diagonal scheduler and submit all family/tile work through one
native work queue. That removes repeated Python scheduling and fills the GPU
without changing the complete candidate set or full-proxy winner contract.

### Deferred canonical validation and four-way overlap

Canonical V2 YAQA previously repeated finite, overlap, and FP32-distance-range
reductions at every provisional and constrained native Viterbi call. A private
V2-only trusted operator now retains all structural checks while deferring
dynamic value validation to YAQA's existing device-side recurrence flag. The
public CUDA operator remains fail-closed. The generated provisional overlap is
also trusted after the native recurrence, avoiding a host-synchronizing
transition assertion that cannot fail without a native-kernel defect.

With those synchronization points removed, canonical V2 can execute on a
fourth persistent stream concurrently with the three independent B2 family
candidates. The producer stream is joined by explicit events, output lifetimes
are recorded on the selecting stream, and the final full-proxy argmin retains
canonical-first and lower-family tie precedence.

```text
+-------------------------------+-----------+-----------+---------+-----------+-----------+----------------------+
| Full W2.5 B2 reselect 512x512 | Before    | Trusted   | Overlap | Total gain| Peak VRAM | Quantization parity  |
+-------------------------------+-----------+-----------+---------+-----------+-----------+----------------------+
| PG506-230 A                   | 1683.30ms | 1449.99ms |1373.82ms|     1.23x | 88.24 MiB | bit-exact repeated   |
| PG506-230 B                   | 1683.30ms | 1449.99ms |1347.50ms|     1.25x | 88.24 MiB | bit-exact repeated   |
+-------------------------------+-----------+-----------+---------+-----------+-----------+----------------------+
```

The trusted boundary contributes `1.16x`; canonical/family overlap adds a
further `1.06-1.08x`. Peak allocation rises from `80.11 MiB` to `88.24 MiB`.
Six W1--W3.5 public/trusted unconstrained and constrained comparisons are
bit-exact. The complete B2 YAQA CUDA gate passes all six rates plus a partial
multi-tile batch. This exceeds the quantization requirement (`1e-6`) because
states, losses, reconstructed weights, selectors, and family IDs are exact.
Inference code is unchanged and retains its separate `2e-3` tolerance.
