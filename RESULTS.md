# CPU G-only Viterbi results

## Hardware and scope

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, `OMP_NUM_THREADS=32` | torch 2.13.0+cpu | host zen5-cpu-6

This change is limited to the non-banked `gptqmodel_ext/qvq/qvq_viterbi_cpu.cpp`. The banked kernel was not modified.
The non-banked recurrence now retains two `[batch, suffix_count]` FP32 `G` buffers and performs emission,
predecessor-`G` addition, strict suffix argmin, and backpointer production in one `at::parallel_for` per step. The
former full-state `costs`, `next_costs`, and `emission_buf` frontiers and three parallel phases per step are gone.

## Correctness and tests

The local baseline was measured at pristine `origin/main` commit `ede2695e` before editing. The after counts match
it exactly:

| Command | Before | After |
|---|---:|---:|
| `pytest tests/test_qvq.py -k "viterbi or tail_biting" -q` | 90 passed, 112 skipped, 653 deselected | 90 passed, 112 skipped, 653 deselected |
| `pytest tests/test_qvq_v2b2_p32.py -q` | 119 passed, 12 skipped, 1 failed | 119 passed, 12 skipped, 1 failed |

The V2B2 failure is the same pre-existing configuration failure before and after:
`test_qvq_v2b2_p32_config_accepts_yaqa_and_weighted_block_ldlq` constructs the default YAQA rounding with a
`hessian_diagonal` objective, which current configuration validation rejects. It does not execute either Viterbi
kernel. The historically reported L18 V4 `1.19e-6` loss edge case did not fail on this host: its selected path test
passed before and after.

`git diff --check` passes. Ruff 0.14.2 (the version pinned by `pyproject.toml`) was run repository-wide and reports
535 pre-existing Python violations; this PR changes no Python file, so there is no changed-Python Ruff finding to
address within scope.

The saved raw-op artifact comparison reported `torch.equal == True` for both selected states and FP32 squared
error. The focused tests cover deterministic ties, weighted and constrained paths, V2/V4, W1 through W8,
tail-biting, eager-oracle parity, and planar packing. The unchanged banked benchmark artifact comparison was exact
for states, squared error, and segment bank IDs at batch sizes 16, 32, 64, and 128; the V2B2 suite covers packed
selector words and bank IDs.

### Tie ordering and FP32 operation order

This implementation takes option **(a): it preserves the original reduction order exactly**.

For a suffix `x`, candidates are visited in increasing prefix `h`, exactly as the old `column_argmin` visited
matrix rows. Both AVX-512 and scalar paths replace the winner only on strict `<`, so equal candidates retain the
lowest prefix. The AVX-512 emission instruction sequence is unchanged for V2 and V4, and the predecessor `G` is
added only after the same emission value has been formed. `G` is precisely the old column minimum, so this addition
is the same FP32 operation on the same operands as the old broadcast-add phase; no sums are reassociated.

The final state is selected from each suffix winner by comparing `(cost, full_state_index)`, with the lower full
state index winning equal costs. This is exactly equivalent to the old strict-`<` scan over full states in ascending
index even though the candidates are partitioned by suffix. Traceback consumes the same lowest-prefix winners.

## Pinned measurements

Every accepted timing used this explicit placement (never `OMP_PLACES=cores`):

```text
OMP_NUM_THREADS=32
OMP_PROC_BIND=close
OMP_PLACES={24},{27},{28},{42},{43},{44},{45},{54},{55},{65},{90},{94},{96},{104},{113},{114},{118},{123},{135},{139},{143},{150},{156},{161},{164},{169},{172},{173},{175},{176},{179},{183}
taskset: 24,27,28,42-45,54-55,65,90,94,96,104,113-114,118,123,135,139,143,150,156,161,164,169,172-173,175-176,179,183
```

Before timing, `/proc/self/task/*/status` was asserted to contain all 32 distinct singleton worker masks listed
above; the process master mask was CPU 24 after OpenMP binding. A mismatch aborted the run. No other Python,
pytest, or benchmark process exceeded 5% CPU, and sampled host idle was 84–87%. Runs were sequential and blocking.

Raw `gptqmodel_qvq.viterbi_cpu`, batch 1, 128 steps, V2, 65,536 states, transition bits 5, fixed generated inputs,
3 warmups and 21 samples:

| Revision | Median | Minimum | Maximum | Exact vs before |
|---|---:|---:|---:|---:|
| pristine `ede2695e` | 5.980552 ms | 5.722426 ms | 7.639101 ms | reference |
| fused G-only | 2.438298 ms | 2.399361 ms | 113.260344 ms | states yes; squared error yes |

Median speedup: **2.45x**. The after maximum is one scheduler outlier; 20/21 after samples were 2.399–8.456 ms,
and the median lies in the steady 2.40–3.07 ms cluster.

The repository's `scripts/benchmark_qvq_viterbi.py` is CUDA-only (it requires `--physical-gpu`, an idle NVIDIA
GPU, and CUDA events), so it cannot measure the requested CPU raw op. The raw op was therefore called directly
without adding a new harness. The existing `scripts/benchmark_qvq_viterbi_banked_cpu.py` was used for the required
banked regression artifact check.

Rejected banked timing evidence: the first saved pre-change medians were 99.7, 14.5, 35.8, and 69.2 ms for batches
16, 32, 64, and 128. After runs had 95.5–800.5 ms sample spreads and apparent 0.30–0.78x ratios even though the
banked translation unit was byte-unchanged. A repeat still ranged 18.1–159.4 ms. These measurements are rejected as
contaminated and are not used to claim a performance delta. Their artifact comparisons were nevertheless bit-exact
for states, squared error, and bank IDs.

## Verdict

Accepted: the non-banked raw op is bit-exact and 2.45x faster by pinned median, within the 1.5–3x target. The banked
kernel remains out of scope, source-unchanged, suite-non-regressed, and artifact-exact.
