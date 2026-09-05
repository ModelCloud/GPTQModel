# Sketch-B / Fisher collection: factor offload and strict FP32 projection tiles

This work starts from `a75e6f73` on `origin/main`, which includes PRs #130 and #135.
The final four-batch H200 benchmarks measure **2.68x at B4** and **2.01x at B16**,
with all 800 factors bitwise equal to main. See the final confirmation section for
workloads, distributions, and limits.
The changes preserve the rank-256 Gaussian estimator and IEEE FP32 statistics.
Phase 1 overlaps final factor offload and removes redundant validation. Phase 2 changes
output tiling of the two long-channel Gaussian projections on Hopper, with exact factor
comparisons against main. The measurements below distinguish the two phases; short-capture
results must not be extrapolated to thousands of sequences.

## Phase 1: scheduling and value reuse

`capture_yaqa_sketch_b` now finalizes each module on its last batch, computes the existing
source-diagonal reduction, validates its accumulators, and queues pinned CPU copies on a
separate CUDA stream. Later modules can backpropagate while these copies run.

- The copy stream waits for the producer stream. `record_stream` protects GPU allocations,
  and the collector waits for all transfers before exposing CPU factors, including on errors.
- Module reuse within a forward remains rejected. Therefore the final batch has exactly one
  final update per target, and no subsequent update can modify an offloaded accumulator.
- Intermediate finiteness scans are redundant on this path: a NaN/Inf cannot become finite
  through subsequent additions. Final source-square reductions inspect every source element;
  both exact-diagonal accumulators are also checked before any factors are returned.
- GPU validation results are reused during CPU descriptor construction only when normalization
  divides by at least one. This cannot overflow, and the existing clamp supplies nonnegativity.
  Small Fisher weights retain CPU checks. Positive-diagonal / zero-source consistency is always checked.
- CUDA collection no longer forces a process-wide cyclic-GC sweep at batch cleanup. Normal
  automatic GC remains enabled. CPU and MPS retain their previous explicit cleanup behavior.
- CPU accumulation, MPS, and exact dense-Gram transfer paths retain their previous implementation.
  This phase introduces no new kernel, lower precision, estimator rank, seed schedule, or quantization format.

For activations `A_b[T,I]`, output gradients `D_b[T,O]`, and the same seeded Gaussian projections,
these operations are unchanged:

```text
input source  += sum_b A_b.T @ (D_b @ R_output,b)
output source += sum_b D_b.T @ (A_b @ R_input,b)
input diagonal  += sum_b,t A_b * ((D_b @ D_b.T) @ A_b)
output diagonal += sum_b,t D_b * ((A_b @ A_b.T) @ D_b)
```

Sequence weighting, masking, summation order, IEEE FP32 matmul mode, normalization, and
congruence materialization are preserved. This phase is value reuse and scheduling, not
an approximate-arithmetic experiment.

## Phase 1 measured results

Hardware: physical GPU 0, H200, PCI `00000000:1C:00.0`, UUID
`GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea`, SM90, 132 SMs, 150,101,688,320 bytes VRAM.
Driver 610.57.04; Torch `2.15.0.dev20260901+cu130`; CUDA 13.0; Transformers 5.5.4;
FLA 0.5.2; 16 CPU threads.

The checkpoint is the local Qwen3.5-27B geometry proxy used in the earlier Qwen3.8 benchmark,
not the unavailable Qwen3.8 checkpoint itself. Both arms use identical BF16 model execution,
eager full attention, FLA chunk forward/backward and fused gated normalization. Convolution
uses the repository's Torch `F.conv1d` compatibility shim, not a fused causal-convolution kernel.
The benchmark records the actual backend callables and package versions.

Both arms collect all 400 targets with sequence length 64, rank 256, CUDA accumulation, and
checkpoint recomputation disabled. Model loading is excluded; model forward, Fisher loss,
backward, accumulation, CPU offload, and factor construction are included. Primary timing uses
separate processes, two warmups per arm, and six repeats without clearing the CUDA allocator
between repeats. Python cleanup before each timed call is excluded for both arms.

```text
+----------------------+-----------+-----------+-------------------+------------+-------------+
| Workload / arm       | Median s  | Mean s    | Min..max s        | Tokens/s   | Peak GiB    |
+----------------------+-----------+-----------+-------------------+------------+-------------+
| B4, 4 rows: main     | 1.757151  | 1.753560  | 1.689725..1.801899 | 145.69     | 62.879      |
| B4, 4 rows: retained | 0.841143  | 0.841485  | 0.838630..0.845910 | 304.35     | 56.955      |
| B16, 16 rows: main   | 2.184188  | 2.452237  | 2.177391..2.995132 | 468.82     | 80.992      |
| B16, 16 rows: kept   | 1.253767  | 1.237352  | 1.185697..1.272590 | 816.74     | 74.436      |
+----------------------+-----------+-----------+-------------------+------------+-------------+
| Median speedup      | B4: 2.089x; B16: 1.742x                                        |
+----------------------+------------------------------------------------------------------------+
```

B4 uses rows 16–19, seed 20260906, 256 valid tokens. B16 uses rows 32–47, seed 20260907,
1,024 valid tokens, one warmup and three repeats; it is a scale check rather than the six-repeat
primary gate. Retained factor size is unchanged at 6.535 GiB. Offload now pins approximately
that amount of host storage. Peak VRAM drops because finished GPU accumulators can be released sooner.

Strict initial preflight passed with 0% utilization, 0 MiB residency, three one-second samples,
and no foreign GPU process. Pre-timing exclusivity also passed. Formal confirmation allows
64 MiB of driver overhead: registering the full pinned factor set raised otherwise-unattributed
driver residency above the original 16 MiB allowance. This is not an allowance for foreign workloads.

A separate, warmed full-model trace observed 7.017 GB of pinned factor transfers and approximately
83% overlap of copy duration with compute. Trace timings are diagnostic, not the speedup measurement.
Alternating reference/candidate experiments showed allocation-sensitive reference timings; the
reported primary result instead uses matched independent processes.

## Phase 1 correctness and instruction audit

All 800 input/output factors are bitwise equal to main on both real-model populations, including
`source`, exact `diagonal`, cached `source_diagonal`, normalizer, and seed. Local floating drift is
zero for these compared factors. This is not a claim that a full 27B quantization/evaluation ran.

The retained regression selection passed **91 tests**, including CUDA quantization/lifecycle replay,
CPU reference behavior, weighted and masked multi-batch captures, non-default streams, asynchronous
failure cleanup, early-batch NaN/Inf propagation, zero-source consistency, and normalization overflow
with small sequence weights. Ruff and `git diff --check` pass. MPS execution was not available.

Source identities:

```text
main qvq_yaqa.py:     851486a6c52063a268007abd1b534a331b9debfdfad972f2d4ab9186bc270645
retained qvq_yaqa.py: a5d456e1bdc26844edff2abd8788a771ef8bd3a9eeb28d384f62eb728e1d5524
```

Nsight Compute 2026.2.1 captured the actual collector on a synthetic linear probe at
B=4, T=64, I=5120, O=17408, rank=256, FP32. This audit includes the probe forward/loss/backward;
it is operator instruction evidence, not model-quality evidence.

```text
+-------------------------------------+----------------+----------------+
| Metric                              | Main           | Retained       |
+-------------------------------------+----------------+----------------+
| CUDA kernels                        | 150            | 115            |
| Executed warp instructions          | 1,988,594,428  | 1,971,487,656  |
| Predicated thread instructions      | 62,895,589,526 | 62,395,581,264 |
| Executed FFMA instructions          | 1,555,574,016  | 1,555,574,016  |
| Maximum registers/thread            | 255            | 255            |
| Local spill traffic, bytes          | 154,496        | 154,496        |
| Shared spill traffic, bytes         | 0              | 0              |
| Excess shared wavefronts            | 77,289,102     | 77,289,062     |
+-------------------------------------+----------------+----------------+
```

These values come from the exported raw metrics (`sass__inst_executed_per_opcode`,
`sass__thread_inst_executed_true_per_opcode`, `launch__registers_per_thread`, spilling metrics,
and `derived__memory_l1_wavefronts_shared_excessive`). The raw local-spill unit is Kbyte;
154.496 Kbyte is reported above as 154,496 bytes. Excess wavefronts are not mislabeled as zero bank conflicts.

All 12 GEMM/RNG kernels have identical SASS and executed opcode counts after normalizing relocated
control-flow addresses. The removed work is the five redundant validation chains, totaling 35 launches.
The source/SASS review found no added math, address/decode work, or register-spill regression in those kernels.
Their achieved occupancy range is 3.125–50.478% before and 3.125–50.492% after; eligible warps/cycle
range is 0.379–3.589 before and 0.377–3.587 after. Correctness and unprofiled timing were rerun afterward.

## Phase 1 reproduction and artifacts

The phase-1 snapshot is `/tmp/qvq-next2x-phase1.py` with the hash above.
Export the baseline and use the same arguments for both processes:

```bash
git show a75e6f73:gptqmodel/quantization/qvq_yaqa.py > /tmp/qvq-next2x-reference.py
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea
export MAX_JOBS=8 NINJAFLAGS=-j8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2
yaqa_args=(--rows 4 --batch-size 4 --row-start 16 --seed 20260906
  --all-targets --arms streaming_256 --accumulator-device cuda
  --no-activation-checkpointing --idle-max-driver-memory-mib 64 --warmup 2 --repeats 6)
python scripts/benchmark_qvq_yaqa_qwen38.py "${yaqa_args[@]}" \
  --collector-source /tmp/qvq-next2x-reference.py --output /tmp/qvq-next2x-reference-clean.json
python scripts/benchmark_qvq_yaqa_qwen38.py "${yaqa_args[@]}" \
  --collector-source /tmp/qvq-next2x-phase1.py --output /tmp/qvq-next2x-candidate-clean.json
# Separate value-parity check; comparisons are outside the timed region.
python scripts/benchmark_qvq_yaqa_qwen38.py "${yaqa_args[@]}" \
  --collector-source /tmp/qvq-next2x-phase1.py \
  --verify-source /tmp/qvq-next2x-reference.py --output /tmp/qvq-next2x-parity-rerun.json
```

Instruction capture (run once per collector, adding `--collector-source` for main):

```bash
ncu --profile-from-start off --section SpeedOfLight --section LaunchStats \
  --section Occupancy --section InstructionStats --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats --section SourceCounters \
  -o /tmp/qvq-next2x-candidate-audit --force-overwrite \
  python scripts/profile_qvq_yaqa_collection.py --collector-source /tmp/qvq-next2x-phase1.py
```

Validation-host artifacts:

- `/tmp/qvq-next2x-reference-clean.json`, `/tmp/qvq-next2x-candidate-clean.json`
- `/tmp/qvq-next2x-candidate-b16.json` (includes real B16 factor parity), `/tmp/qvq-next2x-reference-b16.json`
- `/tmp/qvq-next2x-paired-confirmation.json` (includes B4 parity; allocation-sensitive timings are not the primary result)
- `/tmp/qvq-next2x-reference-audit.ncu-rep`, `/tmp/qvq-next2x-candidate-audit.ncu-rep`
- Matching `*-audit-metrics.csv`, `*-audit-sass.csv`, and `/tmp/qvq-next2x-audit-summary.json`
- `/tmp/qvq-next2x-overlap-trace.json`, `/tmp/qvq-next2x-overlap-trace-metadata.json`
- `/tmp/qvq-next2x-tests-retained.log`

## Rejected contraction approaches

Before phase 2, the large-batch FP32 contractions were the next bottleneck. The B16 gate/up geometry microbenchmark
measures approximately 0.655 ms for `D @ R_output` and 0.531 ms for `D @ D.T` within a 2.45 ms update.
CUDA graph replay did not improve this large geometry. Per-sequence GEMMs and initial split-K/flattening
prototypes change floating-point evaluation and are **not retained**. Synthetic timing/error probes alone
cannot approve a quantization change. Phase 2 instead preserves the contraction accumulation order
and compares real-model factors exactly before instruction and timing checks.

## Phase 2: strict FP32 projection tiles

`qvq_yaqa_cuda.py` replaces only `D_b @ R_output,b` and `A_b @ R_input,b` on
compute capability 9.0, with Triton available, IEEE matmul enabled, FP32 inputs,
B=2..16, T=64, K in {5120,17408}, R=256, and unit innermost strides. Sliced
batch/row strides are supported. Other cases and autograd inputs retain `torch.bmm`.
The batch-one matrix-multiply path, token Gram contractions, exact diagonal math,
source reductions, Gaussian generation, and normalization remain unchanged.

Each output uses increasing-K FP32 fused multiply-adds. There is no split-K sum,
TF32, reduced precision, or changed estimator. The emitted PTX contains
`fma.rn.f32` chains in channel order. Triton 3.8 defaults to an `sm_90a` target on
this H200; the specialization is isolated behind the runtime capability check.
The output tile is 16x16 for B<=4 and 32x64 for B>4, with K tiles of 64 and four
warps. Fixed validated dimensions remove tail masks and generic output addressing.

The initial full-model B16 integration check compared all 800 factors bitwise to
main successfully. The kernel regression matrix tested both channel widths,
every B=2..16, three independent seeds, ten repetitions per seed, padded/sliced
batch and row strides, cancellation, wide dynamic range, zeros, constants, subnormal inputs,
NaN/Inf propagation, and a non-default stream. Reference error and run spread were
zero in all finite cases. Invalid/unvalidated dispatch cases were checked against
the PyTorch fallback. The initial post-audit regression selection passed 123 tests; the expanded matrix
passed **141 tests**, including every supported batch size, after adding row-strided
projection inputs. See `/tmp/qvq-next2x-final-tests-expanded.log`.

### Executed instruction audit

The full collector B4 audit is `/tmp/qvq-next2x-tiled-audit.ncu-rep`. B16 projection
captures are `/tmp/qvq-next2x-{reference,tiled}-b16-audit.ncu-rep`. Matching raw CSV,
source-correlated SASS, and `/tmp/qvq-next2x-projection-audit-summary.json` retain
per-opcode details. The B4 control is the phase-1 audit: its math kernels were
already proven identical to main. B16 captures use main directly.

```text
+------+-------+---------+-------------+-----------+----------+---------+----------------+
| B    | K     | Arm     | Warp inst.  | Reg/thread| Occup. % | Eligible| Shared excess  |
+------+-------+---------+-------------+-----------+----------+---------+----------------+
| 4    | 17408 | main    | 61,082,624  | 80        | 3.125    | 0.379   | 3,342,336      |
| 4    | 17408 | tiled   | 60,598,272  | 44        | 12.207   | 0.363   | 2,244,608      |
| 4    | 5120  | main    | 18,025,472  | 80        | 3.125    | 0.377   | 983,040        |
| 4    | 5120  | tiled   | 17,934,336  | 44        | 12.188   | 0.366   | 671,744        |
| 16   | 17408 | main    | 244,330,496 | 80        | 6.047    | 0.374   | 13,369,344     |
| 16   | 17408 | tiled   | 172,098,048 | 79        | 6.250    | 0.535   | 417,792        |
| 16   | 5120  | main    | 72,101,888  | 80        | 6.045    | 0.373   | 3,932,160      |
| 16   | 5120  | tiled   | 50,692,608  | 79        | 6.249    | 0.529   | 122,880        |
+------+-------+---------+-------------+-----------+----------+---------+----------------+
```

Eligible means eligible warps per active scheduler cycle; shared excess means
excessive shared-memory wavefronts, not a fabricated bank-conflict counter. Both
projection kernels have zero local and shared spills in both arms. At B4,
predicated thread instructions increase slightly despite fewer warp instructions;
the gain comes primarily from more populated output tiles and lower register
usage. At B16, fewer address/control instructions and async loads reduce executed
warp instructions by about 30%, and eligible warps rise about 42%. The remaining
FFMA count difference is the cuBLAS epilogue; no channel terms are omitted.
SASS review confirms reduced address/setup work, contiguous vectorized loads, and
no TF32/tensor-core arithmetic. B4 trades more shared-load/barrier instructions for
higher occupancy; this tradeoff is accepted only with the post-profile exactness
and unprofiled timing gates.

## Final four-batch confirmation

These are independent-process runs of the final implementation and main, each
with two warmups and five timed repeats. Both use all 400 targets, sequence length
64, rank 256, BF16 model execution, IEEE FP32 statistics, CUDA accumulators, and
no activation checkpointing. The runtime includes complete model forward/loss/
backward collection, normalization, and CPU factor delivery. No profiler ran
concurrently with these timings.

```text
+------------------------+-----------+-----------+--------------------+----------+----------+
| Workload / arm         | Median s  | Mean s    | Min..max s         | Tokens/s | Peak GiB |
+------------------------+-----------+-----------+--------------------+----------+----------+
| B4, 16 rows: reference |  6.482025 |  6.493859 | 6.474688..6.542685 |   157.98 |   63.600 |
| B4, 16 rows: final     |  2.419687 |  2.419174 | 2.402871..2.441488 |   423.20 |   63.600 |
| B16, 64 rows: reference |  8.049418 |  8.042787 | 8.005162..8.083965 |   505.63 |   80.987 |
| B16, 64 rows: final     |  4.007776 |  4.008419 | 4.005709..4.012807 |  1015.53 |   80.987 |
+------------------------+-----------+-----------+--------------------+----------+----------+
```

Median speedup: **B4 2.679x; B16 2.008x**.

B4 uses 16 sequences at rows 64–79, seed 20260908. B16 uses 64 sequences at rows
96–159, seed 20260909 (4,070 valid tokens). These populations are disjoint from
the initial single-batch confirmation. Both final runs compared all 800 factors
bitwise to main, including every source, exact diagonal, source diagonal,
normalizer, and seed. Factor storage remains 6.535 GiB. Peak VRAM is unchanged
in these multi-batch runs: earlier batches still hold all GPU accumulators.

The measured 2x target is met for these H200, 64-token, rank-256 workloads. This
is not a universal 2x claim for other sequence lengths, ranks, batch counts,
backends, or thousands of calibration sequences. CPU/MPS and unvalidated GPU
geometries retain their PyTorch math path. Full 27B quantization/evaluation and
MPS runtime validation were not performed.

Final source hashes:

```text
qvq_yaqa.py: b5c2e1093efd1880c31e884a89dc06783f49e57d1b028e3d5acde1ea5119e860
qvq_yaqa_cuda.py: facb531fecbf67724b004ac4d67bb07c81f9251be10a20e440ad728063c0d312
```

Final timing and parity artifacts:

- `/tmp/qvq-next2x-reference-b4multi.json`, `/tmp/qvq-next2x-final-b4multi.json`
- `/tmp/qvq-next2x-reference-b16multi.json`, `/tmp/qvq-next2x-final-b16multi.json`

Reproduce either workload by using the same arguments in separate processes:

```bash
# B4 / 16 rows; use --rows 64 --batch-size 16 --row-start 96 --seed 20260909 for B16.
yaqa_final_args=(--rows 16 --batch-size 4 --row-start 64 --seed 20260908
  --all-targets --arms streaming_256 --accumulator-device cuda
  --no-activation-checkpointing --idle-max-driver-memory-mib 64 --warmup 2 --repeats 5)
python scripts/benchmark_qvq_yaqa_qwen38.py "${yaqa_final_args[@]}" \
  --collector-source /tmp/qvq-next2x-reference.py --output /tmp/yaqa-main.json
# Allow telemetry to settle before the next independent preflight.
sleep 3
python scripts/benchmark_qvq_yaqa_qwen38.py "${yaqa_final_args[@]}" \
  --verify-source /tmp/qvq-next2x-reference.py --output /tmp/yaqa-final.json
```

Coverage instrumentation (`/tmp/qvq-next2x-kernel-coverage.json`) ran 51 kernel
and dispatch tests, including import without Triton. All 23 Python host statements
and all six host branches executed. The raw whole-file figure is 69%: the 11
Triton-body statements at lines 28–38 and its two loop branches execute on the
GPU and are not Python trace events. They are inspected separately in the executed
source/SASS audit. No exclusions or mocked GPU math were added to raise coverage.
The first module-name coverage filter triggered a NumPy repeated-import error;
using the source-directory filter completed normally. This is not a whole-repository
coverage claim.
