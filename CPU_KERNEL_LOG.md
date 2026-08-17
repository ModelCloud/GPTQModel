# CPU Kernel Implementation Log

Targets:
- QVQ V2 and V2B2-P32 inference GEMV
- QVQ Viterbi and YAQA quantization

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
- YAQA non-banked CPU quantization is accelerated because `yaqa_inner` routes tile Viterbi through `tail_biting_viterbi_quantize`.

Done:
- Added `qvq_viterbi_banked_cpu.cpp` with a native AVX/OpenMP-friendly batched banked Viterbi DP.
- `fixed_boundary_v2b2_p32_segment_quantize` and `_batched_v2_banked_viterbi_quantize` now dispatch to the CPU kernel for FP32 CPU inputs.
- This covers `block_ldlq_inner_v2b2_p32` and all banked YAQA modes (`yaqa_inner_v2b2_p32`, `yaqa_output_spectral_refine_v2b2_p32`, `yaqa_spectral_push_v2b2_p32`, `yaqa_localized_spectral_refine_v2b2_p32`).
- `tests/test_qvq_v2b2_p32.py` passes: 92 passed, 9 skipped.
- `tests/test_qvq.py -k "viterbi or tail_biting"` passes except `test_qvq_l18_v4_torch_viterbi_recovers_exact_transition_consistent_path`, which fails because the float32 Python reference DP cost is `1.19e-6` (just above its `1e-6` tolerance); the selected values are bit-exact.

Done (AVX-512 SIMD pass):
- Added shared `gptqmodel_ext/qvq/qvq_viterbi_simd.h` with AVX-512F/BW/VL/DQ/FMA intrinsics for:
  - `emit_distance` (squared Euclidean distance, V=2 and V=4 specializations, clamped to `[0, inf)`)
  - `column_argmin` (row-min and argmin over prefix states)
  - `broadcast_add` (broadcast transition cost over emission states)
- Rewrote `qvq_viterbi_cpu.cpp` and `qvq_viterbi_banked_cpu.cpp` to:
  - Transpose codebook(s) to `[..., vector_size, state_count]` for contiguous 16-state loads.
  - Precompute codebook norms once.
  - Parallelize over `state_count` and `suffix_count` with `at::parallel_for` instead of only over batch.
  - Use `__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))` SIMD helpers with scalar fallbacks and runtime `__builtin_cpu_supports` dispatch.
- Accuracy:
  - `tests/test_qvq_v2b2_p32.py`: 92 passed, 9 skipped.
  - `tests/test_qvq.py -k "viterbi or tail_biting"`: 89 passed, 112 skipped, 1 failed (same pre-existing `squared_error` 1.19e-6 tolerance edge case; CPU result matches Python fallback exactly).
  - Raw op micro-benchmark: CPU and Python fallback produce identical `squared_error` (diff = 0).
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu):
  - Raw `viterbi_cpu`: 18.408 ms -> 2.183 ms (~8.4x)
  - Raw `viterbi_banked_cpu`: 78.203 ms -> 5.423 ms (~14.4x)
  - End-to-end `scripts/benchmark_qvq_banked_yaqa.py --device cpu --bits 3.5`:
    - V2: 669.944 ms -> 43.829 ms (~15.3x)
    - B2 fixed: 1966.070 ms -> 102.556 ms (~19.2x)
    - B2 reselect: 4631.295 ms -> 234.168 ms (~19.8x)
    - B4: 5755.283 ms -> 262.696 ms (~21.9x)
    - All arms report `exact: yes`.
- `ruff check` and `git diff --check` pass.

Done (GEMV shift-register decode pass):
- Replaced the gather-based `decode_tile` in `gptqmodel_ext/qvq/qvq_gemv_cpu.cpp` with a 16-bit shift register over the 128 E-bit codes.
- This reduces per-tile decode work from `128 * 16` bit-gather operations to 128 table lookups plus a small tail preload, while remaining bit-exact for `E = 2..16`.
- Removed the `StepWindow` helper array and `build_step_windows`; `decode_tile` is now `inline` and takes `E` directly.
- Accuracy:
  - `tests/test_qvq_v2b2_p32.py`: 119 passed, 12 skipped.
  - `tests/test_qvq.py -k "viterbi or tail_biting"`: 89 passed, 112 skipped, 1 failed (same pre-existing `squared_error` 1.19e-6 tolerance edge case; CPU result matches Python fallback exactly).
  - CPU GEMV max abs diff vs dense reference for `bits = [2, 3, 3.5, 4, 5, 6, 7, 8]` and V2B2-P32 `bits=3.5` is `<= 2.3e-4`, within the 2e-3 inference tolerance.
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu):
  - Raw `qvq_cpu_gemv` for `x=[1,2048] weight=[2048,2048] bits=3.5`: 104.8 ms (Python fallback) -> 1.09 ms (~96.6x)
  - Dense `torch.matmul` for the same shape: ~0.04 ms.
  - V2B2-P32 small-shape check also passes within tolerance.
- `ruff check` and `git diff --check` pass.

Done (native CPU Hadamard kernel):
- Added `gptqmodel_ext/qvq/qvq_hadamard_cpu.cpp` implementing a fused fast Walsh-Hadamard transform with AVX-512 butterfly add/sub, optional pre/post scale, bias, and scale modes 0-4.
- Registered the `hadamard` op for `gptqmodel_qvq` CPU dispatch and added it to `gptqmodel/utils/qvq_cpu.py` as `qvq_cpu_hadamard`.
- Updated `QVQLinear._qvq_hadamard_fused` to route CPU float32, power-of-two, contiguous inputs through the native kernel instead of the Python `matmul_hadU` fallback.
- Non-power-of-two widths and non-contiguous/non-float32 inputs still fall back to the existing Python path, preserving the previous behavior.
- Accuracy:
  - Bit-exact vs `matmul_hadU` / `matmul_hadU_stable` for unsigned/random-sign inputs (`max abs diff = 0.0`).
  - With arbitrary float32 `pre_scale`/`post_scale`/`bias`, max abs diff is `<= 9.5e-7`, well inside the 2e-3 inference tolerance (and below 1e-6 for quantization if the kernel were used there).
  - `tests/test_qvq_v2b2_p32.py`: 119 passed, 12 skipped.
  - `tests/test_qvq.py -k "viterbi or tail_biting"`: 89 passed, 112 skipped, 1 failed (same pre-existing `squared_error` 1.19e-6 tolerance edge case).
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu):
  - Python `matmul_hadU` -> native CPU Hadamard for `n = [1024, 2048, 4096]`:
    - m=1: ~0.24-0.32 ms -> ~0.008-0.016 ms (~19-30x)
    - m=4: ~0.29-0.42 ms -> ~0.010-0.022 ms (~20-28x)
    - m=16: ~0.36-0.73 ms -> ~0.016-0.031 ms (~21-24x)
  - This removes the Python Hadamard overhead from the QVQ CPU inference path, where it previously sat next to the ~1 ms GEMV kernel.
- `ruff check` and `git diff --check` pass.

Next:
- Continue reducing the gap to dense matmul by fusing decode + FMA over output-channel tiles and vectorizing `unpack_tile_codes`, or explore dense-weight precompute fallback for larger batch regimes.
