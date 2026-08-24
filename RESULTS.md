# Segmented CPU Viterbi G-only results

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

## Outcome

- **MEASURED:** in the self-consistent row-parallel regime (128 steps, V2, 65,536 states, two banks,
  transition_bits=5, segment_steps=16), batches 32/64/128 improved by **1.26x / 1.47x / 1.23x**.
  These are the headline results.
- **MEASURED:** batch 16 improved by **15.76x** separately. This repairs the pristine kernel's pathological
  small-batch inner-parallel regime: each step paid three `at::parallel_for` barriers, while the new row-parallel
  fused recurrence does not. This number is not presented as general throughput.
- **MEASURED:** transition_bits 15/16 do not provide a general benefit. At t15, the suffix frontier has two elements
  and the prefix-heavy G-only recurrence is 0.27-0.28x for batches 32-128 (batch 16 still repairs the barrier
  pathology). At t16, the final kernel rate-specializes unconstrained two-bank cases to the pristine recurrence;
  accepted results are 0.98-1.09x with every discrete output and FP32 cost exact.
- **INFERRED:** the practical ceiling in the normal t5 row-parallel regime is memory traffic and per-row recurrence
  work after barrier removal; eliminating full-state frontier materialization helps, but there is no general 4x.

## Required sweep

All cells use 128 steps, V2, 65,536 states, two banks, 32 threads, 3 warmups, and 15 measured repetitions.
Spread is `max - min`, in milliseconds. `base` is pristine `d9d37181`; `post` is this branch.

### Normal operating point: transition_bits=5, segment_steps=16

| batch | base median | base min | base spread | post median | post min | post spread | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 104.8 | 93.4 | 63.1 | 6.7 | 6.6 | 1.2 | 15.76x |
| 32 | 14.3 | 14.2 | 4.3 | 11.4 | 10.5 | 5.0 | 1.26x |
| 64 | 35.2 | 35.0 | 40.4 | 24.0 | 23.6 | 6.3 | 1.47x |
| 128 | 69.5 | 69.2 | 24.9 | 56.6 | 55.6 | 32.4 | 1.23x |

### Int16/int32 boundary sweep

| t_bits | seg | batch | base med | base min | base spread | post med | post min | post spread | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 16 | 16 | 381.2 | 354.1 | 84.3 | 45.9 | 44.4 | 5.9 | 8.31x |
| 15 | 16 | 32 | 23.1 | 23.0 | 3.5 | 83.7 | 83.5 | 2.3 | 0.28x |
| 15 | 16 | 64 | 45.6 | 45.5 | 1.9 | 166.7 | 166.4 | 9.6 | 0.27x |
| 15 | 16 | 128 | 90.4 | 90.0 | 2.6 | 332.6 | 332.3 | 3.9 | 0.27x |
| 15 | 32 | 16 | 363.2 | 353.4 | 35.3 | 48.3 | 44.8 | 7.9 | 7.52x |
| 15 | 32 | 32 | 23.0 | 22.8 | 1.6 | 83.3 | 83.2 | 2.3 | 0.28x |
| 15 | 32 | 64 | 45.5 | 45.4 | 1.4 | 166.1 | 166.0 | 2.9 | 0.27x |
| 15 | 32 | 128 | 90.3 | 90.0 | 2.4 | 331.2 | 331.1 | 2.6 | 0.27x |
| 16 | 16 | 16 | 372.5 | 350.6 | 78.3 | 379.2 | 368.9 | 124.7 | 0.98x |
| 16 | 16 | 32 | 23.1 | 22.9 | 7.6 | 23.0 | 22.8 | 4.4 | 1.00x |
| 16 | 16 | 64 | 45.6 | 45.4 | 3.5 | 45.5 | 45.3 | 4.0 | 1.00x |
| 16 | 16 | 128 | 90.3 | 90.0 | 4.1 | 87.3 | 85.1 | 7.6 | 1.03x |
| 16 | 32 | 16 | 366.4 | 338.6 | 52.8 | 366.4 | 349.2 | 32.8 | 1.00x |
| 16 | 32 | 32 | 25.0 | 23.0 | 7.0 | 23.0 | 22.9 | 4.0 | 1.09x |
| 16 | 32 | 64 | 45.5 | 45.4 | 2.1 | 46.0 | 45.4 | 10.9 | 0.99x |
| 16 | 32 | 128 | 90.3 | 90.0 | 16.1 | 90.0 | 88.1 | 3.9 | 1.00x |

**MEASURED:** speedup holds modestly at t5 for batches 32/64/128 and strongly for the pathological batch-16
barrier regime. It does not hold at t15 for batches 32-128. T16 deliberately retains the pristine recurrence.

## Correctness

- **MEASURED:** V=2 AVX-512 now starts at zero and performs fmadd(v0), fmadd(v1), matching the scalar reference
  order. Non-aligned ATen suffix chunks remain scalar so a 16-lane predecessor broadcast cannot cross a boundary.
- **MEASURED:** the new 16-vs-24-thread regression failed before the fix with 31/32 states, one segment bank ID,
  and seven packed words different. It passes after the fix.
- **MEASURED:** versus the pre-review PR binary, all 12 t5/t15/t16 batch artifacts were discrete-exact after the
  arithmetic fix. Against pristine, the accepted t5 packed matrix and all t16 fallback cells are exact.
- **MEASURED:** t15 remains intentionally different from pristine because pristine narrows the combined bank/prefix
  backpointer to int16 and emits overflowed negative states (observed minima -65007 to -65512 in the sweep; a
  four-bank short case emitted `-26359`). This branch validates/selects in int32 and emits valid states and banks.
- **MEASURED:** step_count 1/2, sentinel collisions, int64 validate-before-narrowing, invalid inputs, forced segment
  bank switches, and int16/int32 boundary traceback remain covered.

## Measurement protocol and gates

- **MEASURED:** pristine and accepted post-series cgroup samples were 99.341% and 99.191% idle, computed from
  `/sys/fs/cgroup/cpu.stat` `usage_usec` deltas over two seconds and the 32-CPU quota.
- **MEASURED:** both series used 32 explicit singleton placements:
  `{24},{27},{28},{42},{43},{44},{45},{54},{55},{65},{90},{94},{96},{104},{113},{114},{118},{123},{135},{139},{143},{150},{156},{161},{164},{169},{172},{173},{175},{176},{179},{183}`.
  `OMP_PLACES=cores` was never used; runs were exclusive and sequential.
- **MEASURED:** `tests/test_qvq.py`: 658 passed/248 skipped; `tests/test_qvq_v2b2_p32.py`: 120 passed/12 skipped
  (119 passed plus the corrected targeted rerun); `tests/test_qvq_viterbi_cpu_opt.py`: 18 passed/1 xfailed;
  CUDA banked/segment subset: 312 skipped/812 deselected. Ruff and `git diff --check` pass.
