# F6 shared-levels dispatch regression (2026-09-08)

Scope: public P32 CUDA ABI, SM80, transition bits F6, M=1..4, K=8192,
N=2048, scalar variant, dynamic-N dispatch with shared lookup levels.
Baseline upstream `e5483c7009958b5a1cb64ecf2dcd82310fe20408` still contains
the faulty template calls.

The kernel template parameter order is `StaticN, StaticSplitCount, StaticK,
UseSharedLevels`. Four launch sites passed `0, 0, true`, which instantiates
`StaticK=1, UseSharedLevels=false`. The intended call is `0, 0, 0, true`.
This is a dispatch correctness repair, not a new arithmetic/precision policy.
The packed payload, level values, FP16 activations and FP32 accumulation/reduction
contract remain unchanged. Four host launch sites change their template binding;
the potential dispatch matrix is unchanged: four transition widths, four row
counts, three thread counts and four pipeline depths (192 instantiations).
The shared-levels runtime condition is only reachable for F6, so only 48 of
those combinations are eligible for that branch; no broader reachability or
build-time optimization is claimed by this narrow repair.
A static assertion rejects positive K values smaller than or
not divisible by the K tile size.

Observed in the actual ZML/PJRT Llama native path, physical GPU 0,
PCI `00000000:DE:00.0`, UUID `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`,
NVIDIA PG506-230. CUDA memcheck traced the first invalid 16-byte global read
to `p32_window_ampere_m1_kernel_body<6,2,128,16,2,0,0,1,false>`;
the address was misaligned by two bytes. Full diagnostic log:
`/tmp/zml_abi_memcheck_compat.log` (7505 reported errors; only eight printed).
The diagnostic process ran alone on the GPU; no performance measurement is
claimed. Earlier concurrent startup attempts are not timing evidence.

M=1 may silently compute the wrong result without a misaligned second-row
load. Earlier end-to-end evaluation results using this branch must be rerun;
they are not reliable evidence of snapshot quality.

Regression test: `tests/test_qvq_p32_f6_native_stride.py`. The source check
verifies all four launch sites and the compile-time guard. Opt-in GPU checks use
the public C ABI directly with three seeds, M=1..4, split counts 1/8, native and
partials reduction, a non-default stream, eager execution and ten graph replays
with changed input contents at stable addresses. The independent reference
reconstructs the same packed weights and uses FP32 matmul with TF32 disabled.
Per-case gates are finite outputs, MAE <= 0.002 and max error <= 0.046875.
Synthetic fixtures establish kernel correctness only, never quantization quality.

Corrected native tests: **49 passed** (one source guard plus 48 GPU cases),
with eleven reference checks per GPU case. Largest per-case MAE was
`1.06757616e-05`; largest absolute error was `0.000116348267`. Log:
`/tmp/qvq_f6_stride_tests.log`. Runtime: PyTorch `2.13.0+cu130`, driver
`610.43.02`, 124 SMs, CUDA architecture 8.0. The downstream ZML build consumes
this exact fix alongside its existing integration patches; tested library SHA256:
`38031246b4fec5106469bc5fcb8f24a900ac29c5b1f3fd730fb966b6e24f0da5`.

The full native Llama reproducer now reports **zero memcheck errors**:
`/tmp/zml_abi_memcheck_fixed.log`. Paged/non-paged token parity and three rounds
of continuous-batch isolation/reuse passed on the CUDA-converted rank-8 snapshot:
`/tmp/zml_native_parity.log`. PJRT QVQ callback counts transition from 80 during
warmup/capture to zero on subsequent decode replay. This is not an all-device
graph-safety or model-quality claim.

## Generated-code audit

Code revision: `72af0c1a2f994d0589356978bc8a1c571dde4af0`. On the same isolated
GPU, Nsight Compute 2026.2.1 captured the M2/K8192/N2048/split8 native kernel:

```sh
ncu --section SpeedOfLight --section InstructionStats --section LaunchStats \
  --section Occupancy --kernel-name 'regex:p32_window_ampere_m1_kernel.*' \
  --launch-skip 2 --launch-count 1 --export /tmp/qvq_f6_stride_72af0c1 \
  python -m pytest \
  'tests/test_qvq_p32_f6_native_stride.py::test_f6_shared_levels_native_stride_eager_and_graph[7-1-8-2]' -q -s
```

Environment: `QVQ_P32_TEST_LIBRARY` names the tested downstream library above;
`CUDA_VISIBLE_DEVICES` is the recorded GPU UUID; the ZML CUDA sandbox's `lib/compat`
and `lib` precede other library directories. `PYTHONPATH=.` selects this worktree.
The selected correctness test also passed under the profiler.

Raw reported fields (`/tmp/qvq_f6_stride_ncu_details.txt`):

| Metric | Value |
|---|---:|
| Executed Instructions | 7,961,344 |
| Registers Per Thread | 72 |
| Local Memory Spilling Requests | 0 bytes |
| Shared Memory Spilling Requests | 0 bytes |
| Duration (profiled single kernel) | 99.71 us |
| Compute (SM) Throughput | 13.24% |
| Memory Throughput | 14.85% |
| Achieved Occupancy | 6.25% |

The exported executed SASS (`/tmp/qvq_f6_stride_sass.txt`) identifies
`StaticK=0, UseSharedLevels=true`. Its row address uses `IMAD.WIDE.U32` with the
runtime K argument before `LDGSTS.E.BYPASS.128`, rather than the erroneous
one-element row stride. The existing full-K FP32 accumulation is retained;
this repair introduces no reassociation, casts, or approximate math.

No before/after speedup is valid: the baseline M2 launch performs invalid
memory reads. The single profiled duration is not an end-to-end timing or a
speedup claim. Broader shape coverage, a matched performance campaign, all
instruction categories and model-quality benchmarks remain outside this narrow
correctness result.
