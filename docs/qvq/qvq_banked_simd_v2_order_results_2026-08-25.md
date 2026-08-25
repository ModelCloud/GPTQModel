# Banked SIMD V=2 order and dispatch-scope review

Base under test: merged PR #49 tree `2a36a047` (GitHub merge commit
`348cb603`). The worktree was recreated from the refreshed local
`origin/main`; the first local ref was stale at `35b347ec` and was not used for
the final evidence.

## Item 1: V=2 emission reduction order

Severity: **NOT A DEFECT** on the built object and compiler used here.

Source inspection was not used to make this ruling. The clean baseline object
was `/home/ubuntu/work/qvq-build-pr49-baseline-a.AnlLpg/a6a73c9cf1052d06/qvq_viterbi_banked_cpu.o`,
compiled by the QVQ CPU JIT with GCC 15.2.0, `-O3`, and FMA enabled. I
re-derived the register mapping from this object's prologue:

- `rsi` is the V=2 `c0` base and `r14 = rsi + state_count * 4` is `c1`
  (`0x13f-0x143`).
- `xmm6` is `target[0]` and `xmm4` is `target[1]` (`0x155-0x159`).
- The vector constants are `zmm15 = target[0]` and `zmm13 = target[1]`
  (`0x1d2-0x1da`).

The vector body at `0x210` first emits
`vmulps zmm0,zmm13,[r15+...]` (`c1 * t1`, rounded), then
`vfmadd231ps zmm0,zmm15,[r13+...]` (`c0 * t0 + dot`, fused). The scalar
prefix/tail at `0x298` and `0x463` emits the same ordered pair:
`vmulss xmm2,xmm4,[r14+...]`, then `vfmadd231ss xmm2,xmm6,[rsi+...]`.
The scalar and vector paths therefore have the same reduction order in this
object. No Item-1 source change is made, and no thread-count divergence is
claimed.

## Item 2: V=4 test override scope

Severity: **LATENT, NOT OBSERVED IN PRODUCTION**. The environment override is
only referenced by the C++ test/measurement hook and is never set by the
production Python path. A direct wrapper call can reach V=4/t16, but production
quantization call sites do not request transition width 16.

Before the fix, a direct V=4/t16/bank-count-1 call with
`QVQ_TEST_FORCE_BANKED_G_ONLY=1` changed all four squared-error values while
leaving states and segment bank IDs equal. The measured values were:

```text
unforced: [0.8787578344345093, 0.9491938948631287,
           1.620749592781067, 1.2397410869598389]
forced:   [0.8787583112716675, 0.949196994304657,
           1.6207460165023804, 1.239741325378418]
```

The dispatcher guard now includes `vector_size == 2`. V=4 stays on the normal
G-only recurrence whether or not the test-only override is set. The focused
pre-fix test failed with:

```text
FAILED tests/test_qvq.py::test_native_banked_viterbi_force_overrides_are_parsed_and_scoped
AssertionError: QVQ_TEST_FORCE_BANKED_G_ONLY escaped the V=2 dispatcher scope
```

After the fix, that test passes exactly for states, squared error, and segment
bank IDs.

## Item 3: suffix-column partition coverage

`test_native_banked_viterbi_suffix_partition_is_thread_count_invariant` covers
transition widths 7 and 16 at thread counts 8, 16, 24, and 32. Width 7 has
`suffix_count == 512`, so the G-only `parallel_for` genuinely partitions suffix
columns; width 16 has `suffix_count == 1` and remains the required control.
The test compares selected states, squared error, segment bank IDs, and packed
trellis words using `torch.equal`.

The fixture is intentionally tie-rich: each suffix codeword is repeated over
all prefix rows. Measured duplicate/tie density is **99.21875%** at width 7
(512 distinct rows repeated across 65,536 states) and **99.99847412109375%** at
width 16 (one distinct row). The selected path is nontrivial at width 7 and
has segment bank IDs `[0, 1]`; the packed words are nonzero, so the check is
not a tie-free or all-zero pass.

This test passed on both unmodified PR-49 base and the fix, so it is labelled
a regression net rather than a fail-first gate. The V=4 scope assertion is the
fail-first gate.

## Test evidence

On the unmodified PR-49 base, each commit-specific build root started empty and
the first QVQ import performed a real native compile:

- `806 passed, 260 skipped` for the first requested suite.
- `182 passed, 17 skipped` for the second requested suite.
- The corrected Item-3 regression net passed at both widths; the Item-2 V=4
  scope assertion failed as quoted above.

On the fixed working tree, the focused scope/partition run was `3 passed`.
Its build root was initially empty and reported a non-instant native compile
before the tests ran. No timing or speedup claim is made.
