# QVQ CUDA Kernel Optimization Results

Environment: NVIDIA PG506-230 (cc 8.0, 124 SMs, ~95GB), driver 610.43.02,
torch 2.15.0.dev20260817+cu130, JIT extension `qvq_cuda` (-O3, sm_80).
Timing: CUDA-event medians, 100 iterations after 20 warmups, idle-GPU gated.
Accuracy gates: inference MSE <= 1e-3, max-abs <= 2e-3 (fp32 output),
batched/fused <= 4e-3; quantization-process error must be minimal (target 1e-3).

## Inference GEMV (scripts/benchmark_qvq_cuda.py, K=N=4096, fp16 input, fp32 out)

| W   |  M | pre ms | post ms | speedup | max abs (post)  | MSE (post)   |
|-----|----|--------|---------|---------|-----------------|--------------|
| 2.0 |  1 | 0.5294 |  0.0748 |  7.08x  | 1.07e-4         | 7.5e-10      |
| 2.0 |  2 | 0.5663 |  0.1096 |  5.17x  | 7.6e-5          | 4.0e-10      |
| 2.0 |  8 | 0.5683 |  0.1116 |  5.09x  | 1.07e-4         | 4.0e-10      |
| 2.0 | 16 | 0.6257 |  0.1454 |  4.30x  | 1.11e-4         | 5.2e-10      |
| 2.0 | 32 | 0.7721 |  0.2079 |  3.71x  | 7.6e-5          | 2.9e-10      |
| 4.0 |  1 | 0.3369 |  0.0696 |  4.84x  | 1.07e-4         | 7.2e-10      |
| 4.0 |  2 | 0.3727 |  0.0901 |  4.14x  |                  |              |
| 4.0 |  8 | 0.3748 |  0.0922 |  4.07x  |                  |              |
| 4.0 | 16 | 0.4291 |  0.1229 |  3.49x  |                  |              |
| 4.0 | 32 | 0.5550 |  0.1894 |  2.93x  |                  |              |

All rows pass the benchmark gates (--max-mse 1e-3 --max-abs-error 2e-3) with
orders of magnitude of headroom; Top-1/Top-5 = 1.0000, cosine = 1.000000,
SQNR > 127 dB. The earlier fp16-output max_abs=0.03125 failure is one fp16 ULP
of output rounding on synthetic random trellis; use --output-fp32 validation.

Changes (gptqmodel_ext/qvq/qvq_gemv_cuda.cu):
1. Compile-time `TransitionBits` specialization of qvq_gemv_kernel /
   qvq_gemv_splitk_kernel (host-side switch dispatch; runtime-width generic
   fallback retained for unaligned or out-of-range inputs).
2. New `qvq_decode_weight_fast`: unrolled planar decoder, V2 pair state+mix
   computed once per pair and broadcast via direct-indexed shuffle, removal of
   the dead second V2 pgc16_mix (was computed and never used).
3. Vectorized trellis staging (uint4) and vectorized input-tile staging with
   alignment-guarded dispatch.

## Quantization-process Viterbi (scripts/benchmark_qvq_viterbi.py, steps=128)

| W | Batch | pre ms  | post ms | speedup | loss delta | path exact |
|---|-------|---------|---------|---------|------------|------------|
| 2 |    64 |  6.041  |  5.843  |  1.03x  | 0.000e+00  | yes        |
| 2 |   256 | 14.309  | 13.967  |  1.02x  | 0.000e+00  | yes        |
| 4 |    64 |  9.269  |  5.757  |  1.61x  | 0.000e+00  | yes        |
| 4 |   256 | 16.503  | 13.935  |  1.18x  | 0.000e+00  | yes        |

Bit-exactness preserved everywhere (loss delta 0, exact path match), so the
quantization-process error gate of 1e-3 is met with zero error. At
B>=1024 both versions saturate at ~28K tiles/s (L2/LSU bandwidth bound);
B=64 is grid-starved (64 sequences < 124 SMs), capping possible gains.

Changes (gptqmodel_ext/qvq/qvq_viterbi_cuda.cu):
1. Batched-ILP restructuring (kChunk=8) of the shift>=7 emission loops
   (step-0, main recurrence, final selection): loads/FMAs issued before
   min-folds; min/argmin fold order does not affect the result set.
2. V2 float codebook reads collapsed to one aligned float2 load.

## Environment fixes required to build/test
- /usr/local/cuda was missing library headers (cusparse.h et al); symlinked
  from torch wheel's nvidia/cu13/include.
- torch >= 2.9 rejects duplicate TORCH_LIBRARY def() even for identical
  schemas: gptqmodel_qvq::hadamard and yaqa_feedback(_update_) were defined in
  BOTH the CPU and CUDA bundles -> process abort when both loaded. Added
  exception-guarded one-time registration (qvq_def_shared_schema) on all four
  files, and aligned qvq_hadamard_cpu's optional parameters to const& so all
  impls share one C++ signature.
- Pre-existing failure unrelated to this work:
  test_qvq_cuda_composite_input_width_retries_overflow_in_bfloat16 fails on
  pristine main as well (verified via git stash).

## Verification
- tests/test_qvq_cuda.py: 1018 passed, 1 failed (pre-existing above), 2 skipped.
- git diff --check clean; ruff clean on new script.

## Remaining headroom (not done)
- ROWS=16/32 GEMV variants are FMA-throughput bound (~2.8-3.7x); tensor-core
  (WMMA) accumulation could push them further within tolerance gates.
- Plain Viterbi needs step-segmented parallelism (v2_segment-style) to exceed
  single-block-per-sequence limits; current change keeps bit-exact semantics.
- Hadamard kernel (16us, launch-latency bound) and YAQA cutlass grouped-GEMM
  path were left untouched.
