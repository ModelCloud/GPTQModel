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
| Full B2 reselect     | W2.5  |128x512|1176.15ms |1008.18ms |   1.17x | bit-exact repeated     |
+----------------------+-------+-------+----------+----------+---------+------------------------+
```

The 128x512 side-stream run increases measured peak allocation from `28.68 MiB`
to `53.06 MiB`. This remains small relative to model Hessians, but must be
included in larger-shape/real-module gates.

Cross-commit checks for W1, W1.5, W2, W2.5, W3, and W3.5 found zero difference
in dense inner weight, trellis states, selectors, or family ID. This is stronger
than the quantization tolerance of `1e-6`. Inference kernels were unchanged;
their accepted numerical drift remains at most `2e-3`.

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
