# ROCm-native YAQA family recurrence (`e643515d`)

## Scope and result

This is a complete 32x32 YAQA family-reselect/full-sampling solve on physical GPU 0,
an MI355X (`gfx950`, 256 CUs). It includes the four-candidate full proxy and dense
feedback, but not model calibration, preprocessing, serialization, or end-to-end
model quantization. All comparisons use exact tensor equality for weights, states,
segment selectors, and selected family.

| Rate | simplified Torch (ms) | native (ms) | current speedup | speedup vs original ~655 ms Torch reference |
|---:|---:|---:|---:|---:|
| W2 | 403.340 | 4.391 | 91.86x | 149.17x |
| W2.5 | 403.940 | 4.923 | 82.05x | 133.05x |
| W3 | 402.911 | 5.138 | 78.42x | 127.48x |
| W3.5 | 390.437 | 5.388 | 72.46x | 121.56x |

The original 100x target is met at every supported rate against the Torch reference
recorded at the start of this work. The table also reports the stricter denominator
after removing a dead preliminary Torch Block-LDLQ family solve from both paths.
The paired post-commit run used five warmed, alternating samples per path and passed
the idle gate. Raw report: `/tmp/qvq-yaqa-e643515d-postcommit-pipeline.json`.

## Exact algebraic reductions

- Full reselect no longer computes a preliminary family winner unless comparison
  diagnostics require it; all three complementary families are already rescored.
- Canonical V2 is represented by two identical bank-0 tables and batched with the
  three complementary families. Stable flattened tie order therefore reproduces
  canonical bank 0 exactly while one recurrence schedule replaces competing streams.
- The provisional tail-biting solve retains only pointer steps 63 through 127 and
  traces only to the required midpoint overlap. It does not materialize 128 states,
  256 values, eight selectors, or a discarded loss.
- The normal no-diagnostics path leaves winner selection on device and omits selector
  and state churn reductions.
- gfx950 graph-replay tuning selects suffix chunks 16/4/2/4 for W2/W2.5/W3/W3.5.

## Post-commit AMD ISA, LLVM SSA, and execution profile

AMD has no NVIDIA SASS. The equivalent audit used gfx950 AMDGCN ISA, optimized LLVM
IR, Triton IR, and a rocprofv3 executed kernel trace. The trace contains exactly 1536
survivor-step dispatches, four midpoint tracebacks, and eight final tracebacks for
the four rates (one open reference, one midpoint-only provisional pass, and one
closed pass per rate). Aggregate traced device time was 9.987 ms for survivor steps,
0.095 ms for midpoint tracebacks, and 0.465 ms for full tracebacks; this is a profiling
run with compilation/direct launches and is not used as a latency claim.

| Kernel | variants | VGPR range | static ISA range | scratch/private bytes | dynamic LDS bytes |
|---|---:|---:|---:|---:|---:|
| `_survivor_step` | 44 | 12-24 | 100-482 | 0 | 0 |
| `_midpoint_traceback` | 4 | 17-74 | 251-615 | 0 | 0 |
| `_traceback` | 8 | 17-87 | 273-730 | 0 | 0 |

The optimized LLVM SSA across all 56 executed specializations has no `alloca`, FP
divide/remainder, or square-root operations. AMDGCN contains no scratch access,
MFMA, or vector/scalar divide opcodes; constant integer divisions in LLVM lower to
shifts/masks. The emission retains the single required FP32 FMA and the separate
quadratic terms needed for bit-exact Torch ordering. No further common arithmetic
was removable without changing that order.

Raw artifacts:

- compiler cache: `/tmp/qvq-yaqa-e643515d-cache-trace`
- rocprofv3 trace: `/tmp/qvq-yaqa-e643515d-trace`
- ISA/IR summary: `/tmp/qvq-yaqa-e643515d-isa-audit.json`

The system `/opt/rocm` profiler and PyTorch's bundled ROCm runtime cannot be mixed:
the former duplicates LLVM SPIR-V registrations. The matching bundled rocprofv3
successfully captured the kernel trace. Hardware-PMC collection with that bundled
runtime failed in `aqlprofile` before dispatch, so no fabricated PMC values are
reported; dispatch resource fields and exact generated ISA/SSA are still captured.

## Verification

- `51 passed` in `tests/test_qvq_yaqa_amd_native.py`
- changed-code Ruff checks passed (three unrelated pre-existing warnings were ignored)
- `git diff --check` passed
