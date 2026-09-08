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

The sanitizer rerun, PJRT model parity and external-runtime graph replay remain pending.
No kernel speedup, complete coverage, SASS/profile acceptance, or model-quality
pass is claimed.
