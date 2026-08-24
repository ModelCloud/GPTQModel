# CPU G-only Viterbi results

# QVQ direct CPU GEMV small-N scheduling results

## Hardware and method

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, `OMP_NUM_THREADS=32` | torch 2.13.0+cpu | host zen5-cpu-6

The direct kernel now uses one worker per four-output block when there are at most four blocks. Thus the
2048x256 geometry uses four workers instead of committing the full 32-thread team; larger geometries retain the
existing `at::parallel_for`. Decode, FMA, FP32 arithmetic, and per-output accumulation order are unchanged. This
does not change dispatch policy.

Both matrices used 10 warmups and 50 uninterrupted timings per series with
`OMP_PROC_BIND=close`, `QVQ_CPU_GEMV_DENSE_CACHE=0`, and explicit
`OMP_PLACES={24},{27},{28},{42},{43},{44},{45},{54},{55},{65},{90},{94},{96},{104},{113},{114},{118},{123},{135},{139},{143},{150},{156},{161},{164},{169},{172},{173},{175},{176},{179},{183}`.
Immediately before each series, `/proc/self/task/*/status` was read once. Large direct series proved all 32
singleton masks. Small direct series proved the selected map `{24},{27},{28},{42}`. Small dense series could
retain four workers or regrow all 32 during warmup; only those two exact maps were accepted. Raw data are
`/home/ubuntu/work/qvq-findings/gemv_smalln_before.csv` and `gemv_smalln_after.csv`.

## Latency matrix

Times are medians in milliseconds. Ratios are direct/dense within the same run; `speed` is before-direct /
after-direct. The large-shape direct results are effectively unchanged. Every N=256 case improved: 1.063-1.171x,
with geometric-policy mean 1.129x. Dense parity is not claimed: N=256 M1 remains 2.05-2.13x dense.

| KxN | Rate | M | dense before | direct before | ratio before | dense after | direct after | ratio after | speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048x2048 | W2 | 1/8/32 | 0.3721/0.1845/0.3333 | 0.1728/0.2232/0.4293 | 0.464/1.210/1.288 | 0.3802/0.1813/0.3235 | 0.1730/0.2213/0.4124 | 0.455/1.221/1.275 | 0.998/1.009/1.041x |
| 2048x2048 | W3.5 | 1/8/32 | 0.3711/0.2230/0.3232 | 0.1823/0.2306/0.4109 | 0.491/1.034/1.271 | 0.3412/0.1827/0.3236 | 0.1830/0.2311/0.4001 | 0.536/1.265/1.237 | 0.996/0.998/1.027x |
| 2048x2048 | W4 | 1/8/32 | 0.2786/0.1879/0.3250 | 0.1748/0.2274/0.3999 | 0.627/1.210/1.230 | 0.3732/0.1827/0.3236 | 0.1747/0.2283/0.4002 | 0.468/1.250/1.237 | 1.000/0.996/0.999x |
| 2048x8192 | W2 | 1/8/32 | 1.1849/0.6472/1.1663 | 0.6587/0.8299/1.3631 | 0.556/1.282/1.169 | 1.2373/0.6500/1.1600 | 0.6571/0.8264/1.3139 | 0.531/1.271/1.133 | 1.002/1.004/1.037x |
| 2048x8192 | W3.5 | 1/8/32 | 1.2382/0.6462/1.1739 | 0.6950/0.8577/1.3838 | 0.561/1.327/1.179 | 1.4306/0.6455/1.1689 | 0.6943/0.8524/1.3707 | 0.485/1.321/1.173 | 1.001/1.006/1.010x |
| 2048x8192 | W4 | 1/8/32 | 1.2234/0.6451/1.1760 | 0.6685/0.8432/1.3510 | 0.546/1.307/1.149 | 1.4204/0.6438/1.1594 | 0.6665/0.8397/1.3419 | 0.469/1.304/1.157 | 1.003/1.004/1.007x |
| 8192x2048 | W2 | 1/8/32 | 1.2197/0.6932/1.1726 | 0.6490/0.8062/1.3177 | 0.532/1.163/1.124 | 1.6117/0.6553/1.2082 | 0.6497/0.8029/1.3090 | 0.403/1.225/1.083 | 0.999/1.004/1.007x |
| 8192x2048 | W3.5 | 1/8/32 | 1.2087/0.6643/1.1638 | 0.6903/0.8370/1.3642 | 0.571/1.260/1.172 | 1.2895/0.6597/1.2082 | 0.6890/0.8425/1.3576 | 0.534/1.277/1.124 | 1.002/0.993/1.005x |
| 8192x2048 | W4 | 1/8/32 | 1.2600/0.6689/1.1820 | 0.6626/0.8322/1.3775 | 0.526/1.244/1.165 | 1.2793/0.6599/1.2121 | 0.6613/0.8237/1.3491 | 0.517/1.248/1.113 | 1.002/1.010/1.021x |
| 2048x256 | W2 | 1/8/32 | 0.0502/0.0341/0.0544 | 0.1229/0.1493/0.2428 | 2.449/4.376/4.462 | 0.0516/0.0332/0.0536 | 0.1080/0.1286/0.2225 | 2.094/3.869/4.155 | **1.138/1.162/1.091x** |
| 2048x256 | W3.5 | 1/8/32 | 0.0522/0.0346/0.0546 | 0.1305/0.1572/0.2571 | 2.498/4.547/4.706 | 0.0523/0.0334/0.0535 | 0.1114/0.1343/0.2326 | 2.131/4.026/4.347 | **1.171/1.170/1.105x** |
| 2048x256 | W4 | 1/8/32 | 0.0505/0.0338/0.0545 | 0.1216/0.1495/0.2429 | 2.410/4.420/4.455 | 0.0524/0.0336/0.0541 | 0.1075/0.1327/0.2285 | 2.052/3.949/4.223 | **1.131/1.127/1.063x** |

## Accuracy matrix

Entries are max absolute error versus the dense reconstructed FP32 reference for M=1/8/32. Before and after are
bit-for-bit identical in all 36 cells; the worst remains 1.739501953e-3, below the 2e-3 contract.

| KxN | Rate | before M=1/8/32 | after M=1/8/32 |
|---|---:|---:|---:|
| 2048x2048 | W2 | 6.103515625e-5 / 2.899169922e-4 / 3.662109375e-4 | 6.103515625e-5 / 2.899169922e-4 / 3.662109375e-4 |
| 2048x2048 | W3.5 | 4.577636719e-5 / 2.593994141e-4 / 4.577636719e-4 | 4.577636719e-5 / 2.593994141e-4 / 4.577636719e-4 |
| 2048x2048 | W4 | 4.577636719e-5 / 3.356933594e-4 / 3.814697266e-4 | 4.577636719e-5 / 3.356933594e-4 / 3.814697266e-4 |
| 2048x8192 | W2 | 6.103515625e-5 / 3.814697266e-4 / 4.119873047e-4 | 6.103515625e-5 / 3.814697266e-4 / 4.119873047e-4 |
| 2048x8192 | W3.5 | 5.340576172e-5 / 3.204345703e-4 / 3.509521484e-4 | 5.340576172e-5 / 3.204345703e-4 / 3.509521484e-4 |
| 2048x8192 | W4 | 7.629394531e-5 / 3.356933594e-4 / 4.577636719e-4 | 7.629394531e-5 / 3.356933594e-4 / 4.577636719e-4 |
| 8192x2048 | W2 | 1.296997070e-4 / 1.037597656e-3 / 1.739501953e-3 | 1.296997070e-4 / 1.037597656e-3 / 1.739501953e-3 |
| 8192x2048 | W3.5 | 1.373291016e-4 / 1.037597656e-3 / 1.556396484e-3 | 1.373291016e-4 / 1.037597656e-3 / 1.556396484e-3 |
| 8192x2048 | W4 | 1.373291016e-4 / 1.007080078e-3 / 1.373291016e-3 | 1.373291016e-4 / 1.007080078e-3 / 1.373291016e-3 |
| 2048x256 | W2 | 4.005432129e-5 / 2.441406250e-4 / 2.593994141e-4 | 4.005432129e-5 / 2.441406250e-4 / 2.593994141e-4 |
| 2048x256 | W3.5 | 4.577636719e-5 / 2.059936523e-4 / 3.128051758e-4 | 4.577636719e-5 / 2.059936523e-4 / 3.128051758e-4 |
| 2048x256 | W4 | 4.577636719e-5 / 2.136230469e-4 / 2.975463867e-4 | 4.577636719e-5 / 2.136230469e-4 / 2.975463867e-4 |

## Tests, rejected experiment, and verdict

| Command | Before | After |
|---|---:|---:|
| `pytest tests/test_qvq.py -q` | 613 passed, 247 skipped | 613 passed, 247 skipped |
| `pytest tests/test_qvq_v2b2_p32.py -q` | 119 passed, 12 skipped, 1 failed | 119 passed, 12 skipped, 1 failed |

The unchanged failure is
`test_qvq_v2b2_p32_config_accepts_yaqa_and_weighted_block_ldlq`; it is configuration validation and never executes
the kernel. A fully serial N=256 path was rejected: W2/W3.5/W4 M1 measured 0.4075/0.4204/0.4040 ms and M32
0.7851/0.8128/0.7966 ms, roughly 3.2-3.4x worse than baseline. Four workers are needed for the four output blocks.

No unpack-versus-decode instrumentation or secondary decode tuning was performed. The scheduling-only primary
change materially improves every target case without any numerical change, but does not close the dense gap; the
remaining N=256 M1 ratio is 2.05-2.13x.

## Hardware and scope

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, `OMP_NUM_THREADS=32` | torch 2.13.0+cpu | host zen5-cpu-6

This change is limited to the non-banked `gptqmodel_ext/qvq/qvq_viterbi_cpu.cpp`. The banked kernel was not modified.
The non-banked recurrence now retains two `[batch, suffix_count]` FP32 `G` buffers and performs emission,
predecessor-`G` addition, strict suffix argmin, and backpointer production in one `at::parallel_for` per step. The
former full-state `costs`, `next_costs`, and `emission_buf` frontiers and three parallel phases per step are gone.

## Correctness and tests

The local baseline was measured at pristine `origin/main` commit `ede2695e` before editing. Production coverage
remains at baseline; the focused total increases only by four added regression cases:

| Command | Before | After |
|---|---:|---:|
| `pytest tests/test_qvq.py -k "viterbi or tail_biting" -q` | 90 passed, 112 skipped, 653 deselected | 95 passed, 112 skipped, 653 deselected |
| `pytest tests/test_qvq_v2b2_p32.py -q` | 119 passed, 12 skipped, 1 failed | 119 passed, 12 skipped, 1 failed |

The V2B2 failure is the same pre-existing configuration failure before and after:
`test_qvq_v2b2_p32_config_accepts_yaqa_and_weighted_block_ldlq` constructs the default YAQA rounding with a
`hessian_diagonal` objective, which current configuration validation rejects. It does not execute either Viterbi
kernel. The historically reported L18 V4 `1.19e-6` loss edge case did not fail on this host: its selected path test
passed before and after.

`git diff --check` passes. Ruff 0.14.2 passes on the changed test file. A repository-wide run reports 535
pre-existing Python violations unrelated to this change.

### Adversarial-review correction: single-step overlap

PR review found a real blocking bug in the first version. For `state_count=16`, `transition_bits=2`, `overlap=1`,
and one step, the old kernel constrained only the initial high bits and could choose state 6. The first G-only
version also restricted the final suffix and incorrectly chose state 5. The regression test was added first and
observed failing exactly as reported (`[[5]] != [[6]]`) before kernel code changed.

The fix applies the final suffix restriction only when `step_count > 1`, preserving the old step-0 early-continue
semantics. The regression includes the concrete state-6 case and a randomized eager-Torch legacy recurrence sweep
over one and two steps, transition bits 1-4, multiple batches, overlaps, ties, and weighted/unweighted costs. States
and squared errors compare exactly.

The adjacent-step audit found no second defect. At one step, only emission, the initial high-bit constraint, and
global final selection apply; there is no backpointer or final low-bit mask. At two steps, `G0` supplies backpointer
zero, the second fused pass computes `G1`, and the final low-bit constraint applies exactly where the old step-1
branch applied it. No-overlap behavior and all longer paths are unchanged.

### Adversarial-review correction: invalid overlap values

A second review found two real native-boundary defects. First, `-1` was overloaded as both "no initial
constraint" and a real negative overlap, so negative inputs ran an unconstrained first pass. Second, AVX-512
narrowed an int64 overlap before proving it was in range; `2**32 + 1` could therefore become predecessor 1. Tests
were added before the fix. In a mixed batch they observed states 59 and 6 for those invalid values at one step,
and tracebacks `[12, 48, 0]` and `[4, 16, 0]` at three steps, rather than zero paths with infinite error.

The high-level `batched_viterbi_quantize` path rejects negative and out-of-range overlaps (and normalizes them to
zero when overlap bits are zero), so ordinary quantization calls cannot reach the defects. The Python
`qvq_cpu_viterbi` wrapper does not range-check, however, and the public torch op is directly callable. The native
policy is now explicit: a separate boolean distinguishes absence of a constraint, int64 values are checked against
`[0, suffix_count)` before AVX narrowing, and invalid values create an all-infinity first frontier. AVX-512 and
scalar paths therefore return an all-zero path with infinite error for every step count, including mixed batches.

Coverage includes negative and truncating-large overlaps in mixed batches for one, two, and three steps, asserting both
states and squared errors. A deterministic two-step fixture also guarantees the final mask is exercised: the
unconstrained unique final state is 10 (suffix 2), while overlap 1 produces `[6, 9]` with error 8.

### Adversarial-review correction: final-range overflow

A third review found that final selection formed `suffix_begin + 1` before checking whether `suffix_begin` was a
valid suffix. With `INT64_MAX` and more than one step this is signed-overflow undefined behavior. The expanded
ordinary regression passed on the unfixed binary because this compiler's observed wraparound was immediately
overwritten by the following invalid-range branch; no output failure is claimed. A focused UBSan reproduction of
the same expression reported `signed integer overflow: 9223372036854775807 + 1 cannot be represented`.

Final selection now forms `suffix_begin + 1` only inside the valid, constrained branch. The invalid-overlap matrix
now covers step counts 1, 2, and 3; `INT64_MIN`, `INT64_MAX`, `suffix_count` exactly, negative values, and a
`2**32 + valid_overlap` truncation alias in a mixed batch. All assert zero paths and infinite errors.

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

Raw `gptqmodel_qvq.viterbi_cpu`, batch 1, 128 steps, V2, 65,536 states, transition bits 5, fixed generated inputs.
The post-fix completion measurement paired the cached pristine and final fixed shared objects in the same quiet
window, with 10 warmups and 51 samples each:

| Revision | Median | Minimum | Maximum | Exact vs before |
|---|---:|---:|---:|---:|
| pristine `ede2695e` | 5.893175 ms | 5.618564 ms | 28.181893 ms | reference |
| final fixed fused G-only | 2.465951 ms | 2.403298 ms | 10.674923 ms | states yes; squared error yes |

Post-second-review median speedup: **2.39x**. The output tensors have the same combined SHA-256 digest,
`5be22fad56a88fff21c3b510ebcf9c982b7fd298e36247a5fa391ba4f563f86e`. The initial pre-review measurement was
5.980552 -> 2.438298 ms (2.45x); the paired 51-sample result above is authoritative.

After the third fix, the fixed binary measured 2.502110 ms median (2.386086 ms minimum, 7.950348 ms maximum),
consistent with the authoritative 2.465951 ms fixed median. Three same-protocol pristine attempts were rejected as
contaminated: medians 9.160888, 7.490947, and 7.783496 ms with maxima 83.325833, 92.529305, and 114.239829 ms.
Because those baselines were unstable, no new speedup is computed; the prior clean paired **2.39x** remains the
reported before/after result. Every run verified the same 32 singleton affinities and identical output digest.

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

Accepted: after the blocking single-step correction, the non-banked raw op is bit-exact and 2.34x faster by pinned
paired median, within the 1.5–3x target. The banked
kernel remains out of scope, source-unchanged, suite-non-regressed, and artifact-exact.
