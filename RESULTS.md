# Segmented CPU Viterbi G-only results

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

## Outcome

- **MEASURED:** the exclusive target case improved from 110.0 ms to 6.7 ms median, **16.49x**, for batch 16,
  128 steps, V2, 65,536 states, two banks, transition bits 5, and 16-step segments (10 warmups, 51 samples).
- **MEASURED:** selected states, segment bank IDs, packed trellis words, and packed bank selectors matched the
  pristine `d9d37181` artifacts exactly for the full-window V2 overlap matrix at batches 1, 8, 32, and 128.
- **MEASURED:** states and bank IDs also matched pristine artifacts over 17 short/full configurations spanning V2/V4,
  1/2/4 banks, transition bits 1/5/15/16, segment lengths 1/16, weighted and overlap paths, and batches
  1/8/16/32/64/128. FP32 squared error differed by at most 1.39e-6 in the transition-15/16 sweep; the full target and
  packed overlap matrix had zero loss delta.
- **INFERRED:** the remaining last-bit loss changes come from storing/reusing the fused FP32 suffix minimum at a
  different point in the recurrence, as in the sibling non-banked G-only kernel. No discrete winner changed.

## Implementation

- **MEASURED by inspection:** the three `tile_batch * bank_count * state_count` float frontiers and three per-step
  parallel phases are gone. Two per-bank suffix `G` buffers remain, and emission, prior-G addition, strict prefix
  argmin, and backpointer generation execute in one fused suffix pass.
- **MEASURED by inspection:** segment boundaries reduce prior banks in ascending bank order with strict `<`, then
  retain the selected bank and that bank's lowest prefix in the boundary backpointer.
- **MEASURED by inspection:** AVX-512 keeps the established V2/V4 emission order. Rates with at least four transition
  bits broadcast one predecessor-G scalar per 16 lanes instead of gathering it.
- **INFERRED:** moving batches of eight or more to row-parallel execution is safe because the G-only row footprint is
  small; it eliminates the small-batch inner barrier regime responsible for the target case's dominant overhead.

## Correctness defects front-loaded

1. **MEASURED:** step counts 1 and 2 are explicit in the regression. A single step retains the legacy first-step
   control flow (no final overlap/exit mask); two steps exercise a step that is both non-first and last.
2. **MEASURED:** constraint presence uses booleans. Negative overlap no longer collides with a `-1` no-constraint
   sentinel.
3. **MEASURED:** overlap and exit values are range-checked as int64 before narrowing or forming `required + 1`.
   Coverage includes `INT64_MIN`, `INT64_MAX`, negative, exact boundary, and a `2**32` truncation alias.
4. **MEASURED:** mixed valid/invalid batches deterministically return state zero and infinite error for invalid rows.
   Zero `segment_steps` is rejected before modulo arithmetic.
5. **MEASURED:** a forced two-step bank switch traces `[bank 0, bank 1]`; fixed-entry/exit segment tests pass; and a
   transition-15 four-bank case forces combined arg 131071, proving boundary backpointers select int32 before
   narrowing can overflow.

The added regression was run unchanged against pristine `d9d37181` and **FAILED** there: invalid one-step overlaps
returned states 1 for `-1` and `INT64_MIN` instead of the invalid sentinel. It passes with the new kernel.

## Measurement protocol

- **MEASURED:** the pre-series cgroup sample was 99.314% idle and the accepted final post-series sample was 99.317%
  idle, computed from `/sys/fs/cgroup/cpu.stat` `usage_usec` deltas over two seconds and the 32-CPU quota.
- **MEASURED:** both accepted series used
  `OMP_PLACES={24},{27},{28},{42},{43},{44},{45},{54},{55},{65},{90},{94},{96},{104},{113},{114},{118},{123},{135},{139},{143},{150},{156},{161},{164},{169},{172},{173},{175},{176},{179},{183}`;
  `OMP_PLACES=cores` was never used.
- **MEASURED:** the versioned benchmark verified 32 distinct singleton worker affinities once before each timed
  series. Runs were exclusive, sequential, and blocking.
- **MEASURED:** pristine default-size throughput was self-consistent for batches 32/64/128 (0.454/0.564/0.541
  ms/row). Batch 16 intentionally used the old inner-parallel, three-barrier regime and was much slower.
- **MEASURED:** one preliminary series that reported one Torch thread was rejected before any claim.

## Gates

| Gate | Pristine `d9d37181` | Final |
|---|---:|---:|
| `tests/test_qvq.py` | 656 passed, 248 skipped | 657 passed, 248 skipped |
| `tests/test_qvq_v2b2_p32.py` | 120 passed, 12 skipped | 120 passed, 12 skipped |
| `tests/test_qvq_viterbi_cpu_opt.py` | 18 passed, 1 xfailed | 18 passed, 1 xfailed |
| `tests/test_qvq_lifecycle.py` | 18 passed, 12 failed, 2 skipped | 18 passed, 12 failed, 2 skipped |
| `tests/test_qvq_cuda.py -k 'banked or segment'` | 312 skipped, 812 deselected | 312 skipped, 812 deselected |

**MEASURED:** all 12 lifecycle failures are pre-existing and identical by test name. This CPU-only host skipped the
CUDA subset; no CUDA execution claim is made. Ruff on changed Python files and `git diff --check` pass.
