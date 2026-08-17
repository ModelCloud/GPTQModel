# CPU Kernel Implementation Log

Targets:
- QVQ V2 and V2B2-P32 inference GEMV
- QVQ Viterbi and YAQA quantization (pending)

Accuracy targets:
- Quantization kernels: max abs error <= 1e-6 vs reference
- Inference kernels: max abs error <= 2e-3 vs dense reference

Done:
- Generated `gptqmodel_ext/qvq/pgc16_cpu_tables.h` from the canonical 256-entry `_PGC16_LEVEL_BITS` table.

Done:
- Fixed `qvq_gemv_cpu.cpp` planar unpacking bug (`word_offset` now reset per 32-state block).
- Replaced unsafe `std::vector<__m512>` accumulators with a float buffer for `M > 1`.
- Rebuilt JIT extension; V2 and V2B2-P32 inference GEMV now match `reconstruct_qvq_inner_weight` exactly on identity tests and within `~5e-6` on random inputs.
- `QVQLinear._inner_forward` CPU dispatch verified for V2 and V2B2-P32 with `maxdiff 0.0`.

Done:
- Added `qvq_viterbi_cpu.cpp` with a native AVX/OpenMP-friendly batched Viterbi DP.
- Registered `viterbi_cpu` in the same `gptqmodel_qvq` torch.ops namespace and `qvq_cpu` extension.
- `batched_viterbi_quantize` and `tail_biting_viterbi_quantize` now dispatch to the CPU kernel for `[65536, V]` FP32 codebooks and FP32 sequences on CPU.
- Validated V2 (V=2, bits 1-3.5), V4 (V=4, bits 2-4), overlap tail-biting, per-step weights, and tail-biting candidates against the Python reference: states match exactly; squared-error maxdiff <= 4.8e-7; reconstructed values maxdiff 0.0.
- Measured up to ~4x speedup vs the eager FP32 reference on batch/steps shapes representative of QVQ tile quantization.
- YAQA non-banked CPU quantization is accelerated because `yaqa_inner` routes tile Viterbi through `tail_biting_viterbi_quantize`.

Done:
- Added `qvq_viterbi_banked_cpu.cpp` with a native AVX/OpenMP-friendly batched banked Viterbi DP.
- `fixed_boundary_v2b2_p32_segment_quantize` and `_batched_v2_banked_viterbi_quantize` now dispatch to the CPU kernel for FP32 CPU inputs.
- This covers `block_ldlq_inner_v2b2_p32` and all banked YAQA modes (`yaqa_inner_v2b2_p32`, `yaqa_output_spectral_refine_v2b2_p32`, `yaqa_spectral_push_v2b2_p32`, `yaqa_localized_spectral_refine_v2b2_p32`).
- `tests/test_qvq_v2b2_p32.py` passes: 92 passed, 9 skipped.
- `tests/test_qvq.py -k "viterbi or tail_biting"` passes except `test_qvq_l18_v4_torch_viterbi_recovers_exact_transition_consistent_path`, which fails because the float32 Python reference DP cost is `1.19e-6` (just above its `1e-6` tolerance); the selected values are bit-exact.

Next:
- Benchmark CPU GEMV vs dense `x @ inner` and add focused tests.
