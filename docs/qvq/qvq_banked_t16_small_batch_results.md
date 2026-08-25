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

Forced legacy was run at batch 16 across 16, 24 and 32 threads, one process per
thread count, comparing selected states, squared error, segment bank IDs, packed
state words and packed bank selectors.

A temporary `fprintf` on the branch confirmed the legs were actually crossed
rather than the test passing vacuously:

| threads | batch | leg taken | calls |
|---:|---:|---|---:|
| 16 | 16 | row-parallel (`16 >= 16`) | 38/38 |
| 24 | 16 | tiled (`16 < 24`) | 38/38 |
| 32 | 16 | tiled (`16 < 32`) | 38/38 |

38 configurations were used: 5 seeds x {1, 2} banks x {exact-tie, near-tie,
random} codebooks, plus 8 extra segment-schedule shapes. The exact-tie
configurations quantize the codebook onto a coarse grid so the 65,536 codewords
collapse onto 214 distinct vectors — 65,322 of 65,536 rows are exact duplicates
of another row, so essentially every argmin in the recurrence is an exact tie
decided purely by scan order. The near-tie configurations use a finer grid
(601-1,243 exact duplicate rows plus many distinct-but-adjacent costs). The
random configurations are the zero-duplicate control.

**Result: 380 bitwise comparisons (38 configs x 5 output tensors x 2 thread
pairings), 0 mismatches.**

At transition width 16 the suffix frontier has one element, so
`at::parallel_for(0, suffix_count, ...)` over the suffix-column argmin is
degenerate and that particular region is not exercised as parallel. To close
that hole the identical probe was repeated at transition width 7, where
`suffix_count` is 512 and the column argmin genuinely partitions: **another 380
comparisons, 0 mismatches.**

The gate therefore passed, and Option C is safe on this evidence.

## MEASURED: batch sweep before and after

<!--SWEEP-->

## MEASURED: byte-identity against origin/main

<!--BITEXACT-->

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
