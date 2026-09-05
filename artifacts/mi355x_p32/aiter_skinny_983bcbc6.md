# AITER small-M and packed-dot experiments

Production dispatch remains unchanged. Experimental implementation 983bcbc6;
native-wrapper isolation 5ebc4646; graph-check scope fix cd40f302.
Overall target stays 1.5x in every requested case against c89459e3.

## Results

- AITER wvSpltK: all 84 M1/2/4 cases passed canonical max-absolute <=2e-3.
  Useful M2/M4 gains; little M1 benefit. Max error 0.0019464493.
- AITER LLMM1: rejected, 24/28 cases failed, max error 3.910244. Four
  unchanged down cases passed. Do not use this route for the tested K.
- Native binding with cached immutable weight metadata: 36/36 targeted
  cases passed; wrapper savings alone were small.
- Post-profile native-binding full sweep: 364/364 cases passed, 57/364
  at least 1.5x versus c89459e3, geometric mean 1.2777304x. This includes
  21 additional small-M cases above target relative to retained production's
  36/364. Remaining 307 cases fail the speed target. Not production promotion.
- All 124 applicable stream and graph checks passed. The unchanged gate/up
  fallback at M>=1024 performs host-side validation and is not graph-safe;
  this limitation is explicitly recorded, not claimed as passing capture.
- Packed-dot Triton M1: post-profile 28/28 passes, max error 0.0019340515.
  End-to-end timing remains close to the preceding full-K candidate despite
  reduced instructions. No blanket replacement is justified.
- Focused tests: 310 passed, 62 warnings in 14.78s after instruction profiling.
  Ruff and whitespace checks passed. Tests use synthetic legal packed
  weights for kernel algebra; they do not establish real-model quality.

The accepted full sweep ran 5ebc4646 plus the exact benchmark-only scope
fix subsequently committed as cd40f302. The earlier partial full report
is invalid: it stopped at case 271 when an overbroad graph assertion
attempted capture of the unchanged fallback. All 364 cases were rerun
in a fresh process; no partial result is used for the accepted count.

## Numerical and dispatch contract

AITER wvSpltK uses physical contiguous N-by-K FP16 weights, FP16 input,
FP32 packed-dot accumulation and same-type output. The experimental
wrapper routes only M<=4, no composite recovery, no residual term,
and no FP32-output request. Attention M<=32 preserves the existing
residual-disabled behavior. All other cases use the retained operator.
Public QVQ mutation/eligibility guards remain active.

The private-binding experiment retains the same AITER kernel. It caches
only converted weight metadata plus an owning operand reference, converts
each new input/output, and sets the current HIP stream exactly as AITER's
develop=True wrapper does. No new dense weight copy or input/result cache
is introduced. It is a research interface, not a portable public API.

Triton dot2 reinterprets adjacent FP16 pairs as packed registers and uses
v_dot2c_f32_f16 with FP32 results, followed by FP32 reduction. It does not
round products to FP16. Residual-enabled cases retain the earlier full-K
FP32 path. Reassociation can change rounding; correctness is measured,
not asserted from FP32 output alone.

LLMM1 source explanation: K5120/6144 launch 640/768 threads, hence 10/12
waves. The second shuffle reduction repeatedly halves num_warps and uses
groups of num_warps contiguous lanes, which is not a correct general
reduction for these non-power-of-two counts. Source:
csrc/kernels/custom_kernels.cu, LLGemm1_kernel/LLGemm1. The failure is
measured; no patched upstream kernel or model-quality claim is included.
LLMM1 was rejected before performance profiling because correctness failed.

## Hardware, build and provenance

Physical GPU0 MI355X gfx950, BDF0000:83:00.0, unique ID
0x333ef6e01ec019b3, 256 CUs. Torch2.13.0+rocm10.0.0,
Triton3.8.0+git4cff872c.rocm10.0.0. Strict idle and pre-timing gates passed
for accepted reports. No foreign process was terminated.

Installed AITER source: /opt/aiter-glm53-tune,
7440ef72503e1c3fadc5be85a5c74eb7c9c34841. The isolated build initially
failed to link -lamdhip64. Retried successfully with process-local
LIBRARY_PATH and LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib. No system
configuration or installed source was modified.

Build: GPU_ARCHS=gfx950, AITER_JIT_DIR=/tmp/qvq-aiter-skinny-jit,
MAX_JOBS=8, NINJAFLAGS=-j8, CMAKE_BUILD_PARALLEL_LEVEL=8, NVCC_THREADS=2.
Exact flags are retained in build/module_custom/build/build.ninja there;
includes -O3, -fgpu-flush-denormals-to-zero and the AITER-specific LLVM
flags. Denormal/model-wide behavior beyond the recorded tests is not certified.

module_custom.so SHA256:
2ad274de7eaaf87c62bc2b6415c0112d4cda715d964e5bf1b37b936cb7747b41.
Extracted gfx950 code-object SHA256:
f3d967a69a4127a2721acf6c99e9637aa6c322d9d31b582d983b0abe67fbb2e8.
The import log identifies this exact module, and disassembly resolves
each profiled mangled .kd descriptor to its paired function symbol.

## Post-commit instruction and algebra audit

AITER full_q_gate K5120,N12288, grid256 CTAs, 1024 threads/CTA,
64 KiB LDS, no scratch. Selected specializations:

| M | Static instructions | Static dot2 | VGPR | Issued VALU | Issued SALU | Issued LDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 247 | 16 | 40 | 1073152 | 563712 | 64000 |
| 2 | 329 | 32 | 48 | 1751040 | 614400 | 128000 |
| 4 | 626 | 64 | 60 | 2886400 | 474624 | 163840 |

Issued MFMA is zero; packed dot2 is counted in VALU. The kernel stages
activation data in LDS, reuses it across output columns and keeps FP32
partial sums until the final wave reduction. This avoids both repeating
the full activation load for every output and reducing every K tile to
shared memory. No occupancy/stall/bank-conflict metric was collected;
resource counts are not evidence of measured occupancy.

Triton dot2, M1,K5120,N17408, block-N2, four waves:

| Metric | Previous full-K scalar products | Packed dot2 |
| --- | ---: | ---: |
| Static instructions | 256 | 213 |
| VGPR | 76 | 50 |
| Issued VALU | 6963200 | 5292032 |
| Issued SALU | 557056 | 800768 |
| Issued LDS | 69632 | 69632 |
| Dynamic LDS bytes | 32 | 32 |
| Scratch bytes | 0 | 0 |

The new static body has 32 dot2 instructions and removes the explicit
FP16-to-FP32 conversion/product sequence. Padding and per-thread partial
reduction remain. Prior counts are the same-shape full-K audit recorded
in full_k_gemv_cdaab451.md. Normal warmed timing, not PMC replay latency,
determines whether the candidate is faster.

Next concrete kernel experiment: retain vector/packed FP32 partial sums
across the K loop and reduce once after it. The retained GEMV currently
reduces each K tile before accumulating scalar partial sums. Delaying
that reduction may combine AITER's reuse with the existing Triton wrapper
without materializing full padded K. This is untested, not a claimed win.

## Reports and commands

- Full native-binding sweep: aiter_direct_5ebc4646_full.json
- Rejected LLMM1: aiter_llmm1_983bcbc6_rejected.json
- Packed-dot post-profile sweep: dot2_983bcbc6_post.json
- Native exact-symbol ISA summary: aiter_wv_983bcbc6_isa.json (three AITER
  functions resolved; twelve unrelated baseline/reference symbols are
  explicitly absent from this AITER code object, not silently audited).
- AITER counters: /tmp/qvq-aiter-wv-profile/raw/ubuntu2404-mi350x/772418_counter_collection.csv
- AITER disassembly: /tmp/qvq-aiter-wv-isa/
- Dot2 counters/compiler artifacts: /tmp/qvq-dot2-profile/raw/ and /tmp/qvq-dot2-profile-cache/
- Test log: /tmp/qvq-aiter-dot2-tests.log

Full sweep: benchmark_qvq_p32_amd_butterfly.py --butterfly none
--aiter-skinny wv --aiter-direct --baseline-forward-commit c89459e3
--baseline-amd-commit c89459e3 --full-sweep --warmup 10 --iterations 30.
Output
/tmp/qvq-aiter-direct-full-retry/report.json. Dot2 uses --gemv-dot2
--gemv-block-n 2, M1, baseline d299b88a, same sample counts.

Profiler: sudo rocprofv3 with process-local LD_LIBRARY_PATH including
/opt/rocm/core-10.0/lib/rocprofiler-sdk. AITER capture uses
--mangled-kernels --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS
SQ_INSTS_MFMA, regex .*wvSplitK.*|_qvq_p32_folded_gemv.*|Cijk.*,
full_q_gate M1/2/4, warmup1/iterations2. Dot2 capture uses VALU/SALU/LDS,
regex folded_gemv_dot2_kernel|_qvq_p32_folded_gemv.*, M1 gate/up/full-KV.
Both instruction captures follow device experiment commit 983bcbc6;
the later private-wrapper and graph-scope edits do not change native or
Triton device instructions.
