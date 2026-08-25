# Banked transition-16 small-batch CPU dispatch

## Decision

Use the G-only recurrence for the unconstrained, at-most-two-bank transition-16
shape when `batch_size < 32`; retain legacy from batch 32 upward. The cutoff is
fixed and independent of `at::get_num_threads()`.

This is a speed change only. The FP64 results below establish sampled
non-regression; they do not establish or motivate a quality improvement.

## MEASURED: reproduction and onset

The fresh arm comparison used the unmodified `5ee72d93` implementation and a
pre-existing force-enabled measurement binary. An initial support probe
unexpectedly compiled the pristine source because its content-addressed cache
entry was absent; no candidate source edit or candidate build preceded the
measurements. The force-enabled `.so` was then loaded by explicit path so cache
discovery could not compile during either timing series.

Settings: transition width 16, V=2, 65,536 states, two banks, 128 steps,
32 segment steps, 32 Torch/OpenMP threads, three warmups, 15 repeats per arm,
back-to-back arms with alternating order, and explicit singleton OpenMP places.
Affinity was asserted once per series. Minima are the primary estimator because
this multi-tenant host has positive-only interference and the earlier review
identified order-dependent carryover.

The dedicated reproduction gave these results:

| batch | legacy min ms/row | G-only min ms/row | legacy median ms/row | G-only median ms/row | legacy first/second median ms | G-only first/second median ms |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 22.732 | 2.807 | 24.355 | 2.940 | 386.65/416.38 | 45.82/47.45 |
| 32 | 0.721 | 2.599 | 0.767 | 2.605 | 24.75/24.47 | 83.35/83.35 |

Thus legacy's minimum per-row time falls 31.5x between batches 16 and 32,
despite the larger batch. At batch 16 G-only is 8.1x faster on minima; at batch
32 legacy is 3.6x faster. The cliff reproduced, so the stop condition did not
apply. The one-second pre-series cgroup usage delta was 328,256 us; per-batch
idle checks were 303,590 and 303,932 us. Host load average was not used.

A separate candidate-default confirmation (3 warmups, 15 repeats, affinity
asserted once) measured batch 16 at 44.2 ms minimum / 45.9 ms median and batch
32 at 22.0 ms minimum / 23.0 ms median, confirming that the shipped dispatcher
selects the intended fast arm on each side of the cutoff.

The full onset sweep used the same controls. Values below are ms/row; minima
remain primary.

| batch | legacy min | G-only min | legacy median | G-only median | legacy first/second median ms | G-only first/second median ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.995 | 41.836 | 28.470 | 42.010 | 30.74/27.85 | 41.89/42.03 |
| 2 | 21.120 | 41.658 | 25.153 | 41.761 | 50.50/50.31 | 83.36/83.67 |
| 4 | 23.386 | 41.551 | 31.213 | 41.653 | 118.56/124.85 | 166.45/166.67 |
| 8 | 23.388 | 5.525 | 25.759 | 5.912 | 201.70/210.07 | 47.07/47.39 |
| 12 | 23.071 | 3.715 | 26.234 | 4.062 | 309.04/314.80 | 52.57/47.90 |
| 16 | 23.760 | 2.827 | 26.737 | 2.983 | 415.11/427.79 | 47.02/47.75 |
| 20 | 25.190 | 4.162 | 29.333 | 4.183 | 585.46/589.31 | 83.67/83.60 |
| 24 | 26.097 | 3.470 | 29.985 | 3.484 | 712.42/719.63 | 83.50/85.87 |
| 28 | 24.006 | 2.972 | 26.714 | 3.001 | 772.46/709.50 | 83.58/84.85 |
| 32 | 0.718 | 2.600 | 0.727 | 2.604 | 23.14/23.30 | 83.31/83.34 |

The batch-24 series contained one 11.7-second legacy outlier, and batch 28 one
2.1-second outlier. They do not affect the primary minima and illustrate why
minima, raw samples, and position splits were retained.

## INFERRED: hazard resolution

Legacy and G-only can choose different FP32 near-tie paths. A cutoff based on
`at::get_num_threads()` would therefore make selected states and packed output
depend on the runtime thread count. The fixed cutoff prevents this defect class.

The sweep makes 32 the smallest defensible fixed threshold on this host: legacy
remains pathological at batches 24 and 28 and becomes fast at 32. A cutoff of
24 would leave measured cliffs in production. The tradeoff is intentionally
accepted: a machine with a different thread budget can have a different speed
crossover, but it cannot silently change quantized output merely because the
thread setting changed.

The repository retains mutually exclusive `QVQ_TEST_FORCE_BANKED_LEGACY` and
`QVQ_TEST_FORCE_BANKED_G_ONLY` controls so the comparison is reproducible. Tests
assert that the default selects G-only at batch 16 and legacy at batch 32.

## MEASURED: correctness

Baseline `qvq-banked-t16-baseline-5ee72d93.pt` was captured with
`scripts/benchmark_qvq_viterbi_banked_cpu.py --save-baseline` from unmodified
`origin/main` commit `5ee72d93f3079f2c17e7e441e3c85670ea7a29e4`. Its SHA-256 is
`2d0ded27d12fb81a2bef915270fe4c10b80fbc6c6dbd186502df02bb9f1a78f7`.

- Unflipped batches 32 and 64 are byte-identical for selected states, segment
  bank IDs, packed state words, and packed bank selectors. Returned loss delta
  is zero.
- Candidate FP64 adjudication attempted 144 configurations, completed 72,
  explicitly skipped 72 unsupported/invalid-oracle configurations, and found
  18 divergent completed configurations. At flipped batches 8 and 16, G-only
  won all 10 divergent configurations. Across all divergent rows, the two arms
  tied 10-10. This is sampled non-regression and near-tie noise, not evidence of
  an accuracy improvement.
- The banked invariance gate runs identical transition-16 inputs at 16, 24, and
  32 threads on both sides of the cutoff. Selected states, segment bank IDs,
  packed state words, and packed bank selectors are identical.
- Pytest gate 1: `803 passed, 260 skipped`.
- Pytest gate 2: `178 passed, 8 skipped`.

The candidate cold root was initially empty and compiled all seven translation
units; ccache reduced wall time to 16 seconds. `git diff --check` is clean.

## Hardware

Hardware: AMD EPYC 9V33X (Zen 4 Genoa-X) | AVX-512F/BW/VL/DQ/FMA, no AMX |
          32 logical CPUs used, OMP_NUM_THREADS=32 | torch 2.13.0+cpu |
          host zen5-cpu-6

The cgroup CPU list was
`24,27,28,42-45,54-55,65,90,94,96,104,113-114,118,123,135,139,143,150,156,161,164,169,172-173,175-176,179,183`.
The host exposes 192 logical CPUs; timings used only the 32-CPU cgroup. Compiler:
gcc 15.2.0.
