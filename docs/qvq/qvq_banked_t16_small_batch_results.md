# Banked transition-16 small-batch CPU cliff

## Decision

Fix the small-batch cliff where it originates, inside the legacy recurrence,
instead of routing around it in the dispatcher.

`qvq_viterbi_banked_cpu_legacy` chooses between a row-parallel leg and an
outer-serial / inner-parallel tiled leg on `batch_size >= at::get_num_threads()`.
That predicate is the cliff: on a 32-thread machine every batch from 1 to 31
took the tiled leg, which is far slower per row. The fix clamps the predicate to
`batch_size >= std::min<int64_t>(8, at::get_num_threads())`, which is the idiom
the G-only recurrence in the same file already uses for its own tiling choice.

The transition-16 dispatcher is restored to its `origin/main` form: purely
shape-based, with no batch cutoff. **There is therefore no flipped regime and no
production output change of any kind.** Output is byte-identical to
`origin/main` at every batch size and every thread count.

This supersedes the earlier batch-32 dispatch cutoff, which chose the G-only arm
below batch 32. That approach was blocked in cross-vendor review: its constant
was only correct at 32 threads, and its own onset sweep showed G-only is *slower*
than legacy at batches 1/2/4, so it selected the slower arm there.

**Scope: this is a LATENT improvement on a path production cannot currently
reach.** Both CPU call sites reject `shift > 7`
(`gptqmodel/quantization/qvq.py:1552-1553` and `:1836-1837`, "QVQ banked V2 /
fixed-boundary P32 supports only rates W1 through W3.5"), and transition width
16 is rate W8 (`qvq_rates.py`: `QVQ_TRANSITION_BITS = range(2, 17)`, and
`bits = value // 2` for even widths, so 16 -> W8). The only remaining callers of
the transition-16 shape are `scripts/benchmark_qvq_viterbi_banked_cpu.py` and
the test suite. Nothing here changes shipped quantized output.

## MEASURED: is the tiling predicate output-neutral?

The change is only safe if legacy's two tiling legs are bit-identical. The
supporting argument is that `parallel_inner` partitions only output dimensions
(emission per state, suffix-column argmin, elementwise broadcast-add), that each
suffix column reduces over prefixes serially inside one thread, and that the
final `best_final` scan is serial per row in both legs. That argument was tested,
not assumed.

Two independent probes were run, both at commit `2d921abf`.

**Probe A -- legs forced, thread count held constant.** A measurement-only build
(not shipped; the shipped source contains no `QVQ_PROBE_*` symbol) exposes
`QVQ_PROBE_FORCE_LEG` so the same input can be pushed through the row-parallel
leg and the tiled leg *at the same thread count*, removing the thread count as a
confound. The override is demonstrably not a no-op: forcing the tiled leg at
batch 16 is 19.4x slower at transition width 16 and 6.3x slower at width 7, with
an identical selected-state checksum.

72 configurations were used: {16, 7} transition widths x {1, 2} banks x
{exact-tie, random} codebooks x 3 seeds x {4, 16, 32} batches. The exact-tie
configurations quantize the codebook onto a coarse grid so the 65,536 codewords
collapse onto fewer than 1,024 distinct vectors — nearly every argmin in the
recurrence is then an exact tie decided purely by scan order, which is what a leg
that reordered a reduction would corrupt. The random configurations are the
zero-duplicate control. A tie-density assertion fails the run if the fixture ever
stops producing ties, so the gate cannot pass vacuously.

Each configuration compares row-vs-tile at 16, 24 and 32 threads, plus
row-vs-row and tile-vs-tile across those thread counts, over five output tensors
(selected states, squared error, segment bank IDs, packed state words, packed
bank selectors).

**Probe A result: 432 kernel runs, 2,520 tensor comparisons, 449,904 element
comparisons, 0 mismatches.**

| split | tensor comparisons | mismatches |
|---|---:|---:|
| 16 threads | 360 | 0 |
| 24 threads | 1,080 | 0 |
| 32 threads | 1,080 | 0 |
| transition width 16 | 1,260 | 0 |
| transition width 7 | 1,260 | 0 |

**Probe B -- shipped binary, production predicate.** The same 72 configurations
on the shipped build, crossing the real predicate by varying the thread count
only (16, 24, 32, 4, 2, 1) with no overrides of any kind: **432 kernel runs,
1,800 tensor comparisons, 321,360 element comparisons, 0 mismatches**, again
split evenly 900/900 between transition widths 16 and 7.

Why both transition widths matter: at width 16 the suffix frontier has one
element, so `at::parallel_for(0, suffix_count, ...)` over the suffix-column
argmin cannot partition and that region is never exercised in parallel. A
width-16-only gate is therefore **vacuous** for the argmin. At width 7
`suffix_count` is 512 and the column argmin genuinely partitions. The shipped
dispatcher sends width 7 to the G-only recurrence, so reaching legacy there is
exactly what the measurement-only `QVQ_PROBE_FORCE_LEGACY_ANY` build is for.

The gate therefore passed, and Option C is safe on this evidence.

## MEASURED: batch sweep before and after

Paired legacy row-parallel (post-fix behaviour) versus tiled (pre-fix
behaviour) at 32 threads, transition width 16, two banks, 32 steps. Three
warmups, 15 repeats per arm, arms back-to-back with alternating order, explicit
singleton `OMP_PLACES` built from the cgroup cpuset (`OMP_PLACES=cores` is never
used here — it causes a 4.1x phantom slowdown on this host). Minimum is the
primary estimator because interference on this multi-tenant box is positive-only.

On a 32-thread box the old predicate `batch_size >= at::get_num_threads()` chose
the tiled leg for every batch 1..31; the new `>= min(8, at::get_num_threads())`
chooses the row leg from batch 8 up. **Production behaviour therefore changes
only for batches 8..31** — batches 1/2/4 took the tiled leg before and still do,
and batch 32 took the row leg before and still does. Those rows are included as
controls.

| batch | row med ms/row | row min ms/row | tile med ms/row | tile min ms/row | tile/row med | tile/row min | changed by fix? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 6.138 | 5.918 | 6.914 | 6.556 | 1.13x | 1.11x | no (control) |
| 2 | 3.261 | 3.126 | 6.271 | 6.053 | 1.92x | 1.94x | no (control) |
| 4 | 1.674 | 1.638 | 6.865 | 5.923 | 4.10x | 3.62x | no (control) |
| 8 | 0.956 | 0.922 | 6.214 | 5.785 | 6.50x | 6.28x | **yes** |
| 12 | 0.714 | 0.674 | 6.840 | 5.819 | 9.57x | 8.63x | **yes** |
| 16 | 0.615 | 0.568 | 6.548 | 5.986 | 10.64x | 10.54x | **yes** |
| 24 | 0.497 | 0.457 | 6.397 | 5.938 | 12.88x | 12.98x | **yes** |
| 32 | 0.444 | 0.398 | 8.025 | 6.032 | 18.06x | 15.17x | no (control) |

In the range the fix actually changes, batches 8 through 24, the row leg is
**6.50x to 12.88x faster by median (6.28x to 12.98x on minima)**. The
controls behave as predicted: batch 1 shows almost no gap (1.13x), which is why
clamping at 8 rather than 1 is the right cut, and batch 32 is unchanged in
production regardless of the 18.06x leg-to-leg gap.

Position-in-pair medians (first arm / second arm, milliseconds) expose carryover:

| batch | row 1st / 2nd | tile 1st / 2nd |
|---:|---|---|
| 1 | 5.95 / 6.23 | 6.91 / 6.80 |
| 2 | 6.28 / 6.56 | 12.41 / 13.25 |
| 4 | 6.64 / 6.70 | 30.72 / 26.50 |
| 8 | 7.41 / 7.69 | 49.71 / 51.29 |
| 12 | 8.25 / 8.61 | 82.08 / 81.40 |
| 16 | 9.44 / 9.89 | 102.93 / 104.96 |
| 24 | 11.34 / 12.35 | 155.97 / 149.00 |
| 32 | 13.88 / 14.22 | 267.81 / 238.19 |

Carryover is under 5% on every row except tiled batch 4 (15.4%) and tiled
batch 32 (11.5%); minima are unaffected and remain primary. Per-batch
one-second cgroup `usage_usec` idle deltas ranged 129,377 to 322,120
microseconds against a 32,000,000 us/s budget, i.e. the cgroup was below 1%
utilised before every series. Host uptime and load average were not used and are
meaningless on this 192-CPU multi-tenant box.

## MEASURED: byte-identity against origin/main

The strongest correctness statement available here is that Option C changes
nothing at all. The transition-16 dispatcher is restored to `origin/main`'s
shape-only form and the tiling predicate is proven output-neutral above, so the
shipped kernel should be byte-identical to `origin/main` everywhere, not merely
equivalent.

That was checked directly. A reference dump of 128 configurations was captured
from unmodified `origin/main` `5ee72d93` in its own initially-empty build root
({16, 7} transition widths x {1, 2} banks x {exact-tie, random} codebooks x
2 seeds x {1, 4, 16, 32} batches x {16, 32} threads), recording selected
states, squared error, segment bank IDs, packed state words and packed bank
selectors. The Option C build was then dumped in eight chunks of 16 and each
chunk compared against it as it was produced.

**Result: 128/128 configurations, 640 tensor comparisons, 87,360 element
comparisons, 0 mismatches.**

The reference remains valid against the current `origin/main` `35b347ec`: the
only kernel file `5ee72d93..35b347ec` touches is `qvq_viterbi_cuda.cu`, so the
CPU banked kernel is byte-identical across that range (`git diff --stat`).

## Environment override defects fixed

Three defects in the `QVQ_TEST_FORCE_BANKED_*` controls are fixed here. They are
independent of the cliff fix.

1. **Scope (was blocking).** The old predicate was
   `test_force_legacy || (!test_force_g_only && legacy_t16_shape && ...)`. With
   `QVQ_TEST_FORCE_BANKED_LEGACY` set, *every* banked call went to legacy —
   including the t15, `bank_count` 3-4, V=4 and overlap/entry/exit shapes the
   dispatcher deliberately excludes. Legacy implements those, so this did not
   crash; it silently returned a different answer.

   **This was measured, not merely reasoned about.** At `4b227a66`, on a
   `transition_bits=16`, `bank_count=3`, batch-4, seed-2 case, setting the
   override changed one selected state and all four squared errors, reproducibly
   at 4, 16 and 32 threads. Both overrides are now scoped to `legacy_t16_shape`,
   and that exact case is the regression fixture in
   `test_native_banked_viterbi_force_overrides_are_parsed_and_scoped`.

   A consequence worth stating plainly: because the restored dispatcher already
   routes every `legacy_t16_shape` call to legacy, `QVQ_TEST_FORCE_BANKED_LEGACY`
   now has no remaining effect on dispatch. It is retained as the explicit mirror
   of the G-only override and as a regression pin on that default. The A/B
   measurement capability is unaffected: `QVQ_TEST_FORCE_BANKED_G_ONLY` still
   selects the other arm. The capability that is lost is forcing legacy on shapes
   the dispatcher excludes, which was used for the transition-width-7 probe above
   and now requires a source edit.

2. **Value parsing.** `std::getenv(...) != nullptr` meant
   `QVQ_TEST_FORCE_BANKED_G_ONLY=0` and `=""` both *enabled* the override. The
   values are now parsed with the repository's own convention
   (`gptqmodel/utils/qvq_cpu.py:211`: `.lower() not in ("0", "false", "off", "")`).
   This is not in tension with the rejected finding recorded at
   `docs/qvq_harness.md:718-721`: there, a present-but-falsey snapshot-authority
   variable must fail *closed*, and here a present-but-falsey test override must
   default *off*. Both resolve toward the safe state.

3. **Discoverability.** `TORCH_WARN_ONCE` now fires when either override is
   active, so a value inherited from a stale shell profile cannot silently change
   output with no signal.

## MEASURED: de-aliasing the earlier FP64 adjudication

The superseded results doc reported "18 divergent configurations" and "10/10 at
flipped batches 8/16" without saying whether those were distinct data instances.
They were not. Re-reading `t16-batch-candidate-adjudication.json`:

| level | all divergent | flipped batches 8/16 |
|---|---:|---:|
| raw rows as reported | 18 | 10 |
| distinct `(seed, batch, banks, steps)` | 9 | 5 |
| also collapsing bank aliasing | 5 | 3 |

`segment_steps` (8 vs 16) does not change the input tensors at all, so it
double-counted every instance: 18 -> 9 and 10 -> 5. Beyond that, the harness
(`t16_dispatch_measure.py:45-48`) seeds one generator with
`seed + 1_000_003 * states + batch` — not with `banks` — and draws the sequences
before the codebooks, so the `banks=1` codebook is bit-identical to bank 0 of
the `banks=2` codebook and the sequence tensors are the same. Collapsing that
alias too gives 5 distinct instances overall and 3 at the flipped batches. This
is the same collapse PR #47 found (10-vs-8 becoming 3-vs-2).

This no longer bears on the shipped change: Option C flips nothing, so there is
no divergent regime left to adjudicate. It is recorded because the earlier
figures are still in the git history of this branch and were overstated by 2-3.6x.

## Hardware

Hardware: AMD EPYC 9V33X (Zen 4 Genoa-X) | AVX-512F/BW/VL/DQ/FMA, no AMX |
          32 logical CPUs used, OMP_NUM_THREADS=32 | torch 2.13.0+cpu |
          host zen5-cpu-6

The host exposes 192 logical CPUs; timings used only the 32-CPU cgroup, and host
load average is meaningless here. Idleness was checked from
`/sys/fs/cgroup/cpu.stat` `usage_usec` deltas before every batch. Compiler:
gcc 15.2.0. Each arm used its own initially-empty
`GPTQMODEL_QVQ_CPU_BUILD_ROOT` and a confirmed real compile (ccache-warm, 16s,
matching the precedent recorded for PR #47).
