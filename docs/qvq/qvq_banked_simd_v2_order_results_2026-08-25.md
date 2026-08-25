# Banked SIMD V=2 order and dispatch-scope review

Base under test: merged PR #49 tree `2a36a047` (GitHub merge commit
`348cb603`). The parent and PR trees were tested in separate worktrees with
distinct initially-empty `GPTQMODEL_QVQ_CPU_BUILD_ROOT` directories.

## Item 1: V=2 emission reduction order

Severity: **NOT A DEFECT** on the built object and compiler used here.

This ruling came from disassembly, not source reading. The clean baseline
object was compiled by the QVQ CPU JIT with GCC 15.2.0, `-O3`, and FMA
enabled. Register mapping was re-derived from that object's prologue:

- `rsi` is the V=2 `c0` base and `r14 = rsi + state_count * 4` is `c1`.
- `xmm6`/`xmm4` are `target[0]`/`target[1]`.
- `zmm15`/`zmm13` are the corresponding vector broadcasts.

The vector body at `0x210` and `0x344` emits `vmulps c1*t1` followed by
`vfmadd231ps c0*t0 + dot`. The scalar prefix/tail at `0x298` and `0x463`
emit the same ordered `vmulss c1*t1` followed by `vfmadd231ss c0*t0`. Both
paths therefore use the same ordered arithmetic in this object. The header is
untouched; no Item-1 output divergence is claimed.

## Item 2: V=4 default dispatch and test override scope

Severity: **LATENT, NOT OBSERVED IN PRODUCTION**. The environment override is
test-only and is never set by the production Python path. Transition width 16
is also rejected by both production call sites, so this remains latent.

The original PR changed the default predicate for unconstrained V=4/t16 calls
by adding `vector_size == 2` to `legacy_t16_shape`. I measured that effect
directly with no environment variables set, comparing parent `2a36a047` with
PR tree `b396dfc2`:

- 32 configurations attempted: seeds `2, 11, 202, 3033`; bank counts `1, 2`;
  batch sizes `1, 4`; and step counts `16, 32`.
- 16 completed comparisons (the 32-step cases), 16 explicitly skipped because
  the packed-word helper rejects 16-step streams that are not whole 32-edge
  blocks, and 0 comparison skips or errors.
- Minimum duplicate/tie density among completed configurations was
  **99.8046875%**.
- With no overrides set, selected states, segment bank IDs, packed words, and
  squared errors were all exactly equal in every completed configuration:
  **MEASURED: default V=4 t16 output is unchanged by this scoping, 16 configs,
  0 divergences.**

The measured result supports keeping the `vector_size == 2` guard. The C++
comment records the measured fact and does not assert which recurrence V=4
belongs to.

The direct pre-fix V=4 override reproducer remains real: with a random
V=4/t16/bank-count-1 call, setting `QVQ_TEST_FORCE_BANKED_G_ONLY=1` changed all
four squared-error values while states and segment bank IDs stayed equal. The
test-only override is therefore scoped to the V=2 predicate. The updated V=4
default-control assertion uses a tie-rich fixture and passes on both parent
and PR trees; it is explicitly a regression net, not a fail-first test. The
dedicated no-environment pin test also asserts exact selected states, segment
bank IDs, and packed words for a 32-step V=4/t16/bank-count-1 fixture with
99.804688% tie density; it passes on both trees and is likewise a regression
net. The bank-count-3 override-scope assertion is the fail-first Item-2 gate.

## Item 3: suffix-column partition coverage

`test_native_banked_viterbi_suffix_partition_is_thread_count_invariant` covers
transition bits 7 and 16 at thread counts 8, 16, 24, and 32. Width 7 has
`suffix_count == 512`, so the G-only `parallel_for` genuinely partitions suffix
columns; width 16 has `suffix_count == 1` and is the required control. The
test compares selected states, squared error, segment bank IDs, and packed
trellis words with `torch.equal`.

The fixture is intentionally tie-rich: measured duplicate/tie density is
99.21875% at width 7 and 99.99847412109375% at width 16. The selected path is
nontrivial at width 7, with segment bank IDs `[0, 1]` and nonzero packed words.
Both parameter cases pass on parent and PR trees, so this is labelled a
regression net rather than a fail-first gate.

## Test evidence

The exact gate commands used for the counts below were:

Gate A:

```text
/home/ubuntu/venvs/qvq/bin/python -m pytest -q tests/test_qvq.py tests/test_qvq_v2b2_p32.py tests/test_qvq_viterbi_cpu_opt.py tests/test_calibration_coverage.py tests/test_qvq_yaqa_factor_ensemble.py
```

Gate B:

```text
/home/ubuntu/venvs/qvq/bin/python -m pytest -q tests/test_qvq_diagnostic_metrics.py tests/test_qvq_lifecycle.py tests/test_qvq_v2b4_p64.py tests/test_qvq_yaqa_mps.py tests/test_qvq_cpu_yaqa.py
```

With those exact file lists, fresh current-tree runs exited zero and produced:

- Parent `4fcf4fbc`: Gate A `806 passed, 260 skipped`; Gate B `182 passed,
  17 skipped`.
- PR tree `aab4417a` before the dedicated pin test: Gate A `808 passed, 260
  skipped`; Gate B `182 passed, 17 skipped`.
- PR working tree with the dedicated pin test: Gate A `809 passed, 260
  skipped`; Gate B `182 passed, 17 skipped`. The one-case increase is the
  new default pin test. Each QVQ-loading run used a distinct empty build root
  and a confirmed real cold native compile with compiler caching disabled.

The prior Item-2 direct values were independently recorded as:

```text
unforced: [0.8787578344345093, 0.9491938948631287,
           1.620749592781067, 1.2397410869598389]
forced:   [0.8787583112716675, 0.949196994304657,
           1.6207460165023804, 1.239741325378418]
```

The focused post-fix scope/partition run passed all 3 cases. No accuracy,
timing, or speedup claim is made.
