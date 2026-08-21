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

Done (AVX-512 vectorized QVQ GEMV decode):
- Added `decode_tile_avx512` in `gptqmodel_ext/qvq/qvq_gemv_cpu.cpp` that computes the 16-bit PGC state vector for 16 consecutive trellis steps at once using AVX-512 integer shifts/ORs, then gathers the two float values per state from `g_state_values` with two `_mm512_i32gather_ps` calls and stores them interleaved into `tile_weights`.
- This breaks the serial state-update dependency chain in the old scalar `decode_tile` and reduces per-tile table lookups from 128 serial loads to 8 vector gathers (2 per 16-step block).
- `accumulate_tile_m1_avx512` and `accumulate_tile_mn_avx512` now call `decode_tile_avx512`; the scalar `decode_tile` remains for the non-AVX-512 fallback path.
- Accuracy:
  - `tests/test_qvq_v2b2_p32.py`: 119 passed, 12 skipped.
  - `tests/test_qvq.py -k "viterbi or tail_biting"`: 89 passed, 112 skipped, 1 failed (same pre-existing `squared_error` 1.19e-6 tolerance edge case).
  - CPU GEMV max abs diff vs dense reference for `bits = [2, 3, 3.5, 4]` is `<= 2.44e-4`, inside the 2e-3 inference tolerance.
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu):
  - Raw `qvq_cpu_gemv` for `x=[1,2048] weight=[2048,2048]`:
    - bits=3.5 (E=7): 1.04 ms -> 0.90 ms (~1.15x)
    - bits=2.0 (E=4): ~0.53 ms
    - bits=3.0 (E=6): ~0.71 ms
    - bits=4.0 (E=8): ~0.52 ms
  - Python fallback remains ~100 ms; speedup vs fallback is now 114-190x depending on E.
- `ruff check` and `git diff --check` pass.

Done (AVX-512 vectorized QVQ GEMV planar unpack):
- Reverted the inline PGC16-mix experiment in `decode_tile_avx512` because it added integer operations to the per-state critical path and regressed `qvq_cpu_gemv` from ~0.90 ms back to ~0.97-1.15 ms. The pre-mixed `g_state_values` gather path is restored.
- Added `unpack_tile_codes_avx512` in `gptqmodel_ext/qvq/qvq_gemv_cpu.cpp` that extracts 32 16-bit transition codes per 32-state block with AVX-512 `permutexvar_epi32`, `srlv_epi32`, `cvtepi32_epi16`, and `slli_epi16`/`or` across binary planes. This removes the scalar per-code bit extraction loops that were the remaining large bottleneck for non-power-of-two E (3.5/6/7).
- `accumulate_tile_m1_avx512` and `accumulate_tile_mn_avx512` now call `unpack_tile_codes_avx512`; the scalar `unpack_tile_codes` remains for the non-AVX-512 fallback path.
- Accuracy:
  - `tests/test_qvq_v2b2_p32.py`: 119 passed, 12 skipped.
  - `tests/test_qvq.py -k "viterbi or tail_biting"`: 89 passed, 112 skipped, 1 failed (same pre-existing `squared_error` 1.19e-6 tolerance edge case).
  - CPU GEMV max abs diff vs dense reference for `bits = [2, 3, 3.5, 4]` is `<= 2.44e-4`, inside the 2e-3 inference tolerance.
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu):
  - Raw `qvq_cpu_gemv` for `x=[1,2048] weight=[2048,2048]`:
    - bits=2.0 (E=4): 0.53 ms -> 0.39 ms (~1.37x)
    - bits=3.0 (E=6): 0.71 ms -> 0.40 ms (~1.78x)
    - bits=3.5 (E=7): 0.90 ms -> 0.43 ms (~2.09x)
    - bits=4.0 (E=8): 0.52 ms -> 0.37 ms (~1.41x)
  - Python fallback remains ~100 ms; speedup vs fallback is now 223-263x depending on E.
  - Dense `torch.matmul` for the same shape is ~0.04 ms; the remaining ~10x gap is now in the per-tile decode+FMA loop itself.
- `git diff --check` passes. No Python files were changed in this pass; the pre-existing `ruff check` findings are unrelated.

Done (dense-weight dequantize cache for CPU GEMV):
- Added `qvq_inner_weight_cpu` native op in `gptqmodel_ext/qvq/qvq_gemv_cpu.cpp` that decodes the packed QVQ trellis into a dense `[in_features, out_features]` float32 weight matrix using the existing `unpack_tile_codes_avx512` / `decode_tile_avx512` AVX-512 helpers and `at::parallel_for`.
- Registered the op in `gptqmodel_qvq` and exposed it through `gptqmodel/utils/qvq_cpu.py` as `_qvq_cpu_inner_weight_op`.
- `qvq_cpu_gemv` now optionally builds/reuses a dense dequantized weight copy and calls `torch.matmul(x, inner)` for the GEMV, removing the per-token planar decode cost from the critical inference path.
- Cache is keyed by `id(trellis)` plus decode options; a `weakref` finalizer removes the dense copy when the source trellis tensor is garbage collected.  Set `QVQ_CPU_GEMV_DENSE_CACHE=0` (or `false`/`off`) to keep the previous on-the-fly AVX-512 GEMV path.
- Accuracy:
  - `tests/test_qvq_v2b2_p32.py`: 119 passed, 12 skipped.
  - `tests/test_qvq.py -k "viterbi or tail_biting"`: 89 passed, 112 skipped, 1 failed (same pre-existing `squared_error` 1.19e-6 tolerance edge case; CPU result matches Python fallback exactly).
  - With the cache enabled, CPU GEMV output is bit-exact against the Python `reconstruct_qvq_inner_weight` dense reference for `x=[1,2048] weight=[2048,2048] bits=3.5` (`max abs diff = 0.0`).
  - With the cache disabled, the on-the-fly AVX-512 GEMV stays within `2.29e-4` max abs diff vs the dense reference, inside the 2e-3 inference tolerance.
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu):
  - For `x=[1,2048] weight=[2048,2048] bits=3.5`:
    - Python fallback: ~295 ms
    - On-the-fly `qvq_cpu_gemv` (cache disabled): ~0.49 ms (~600x vs fallback)
    - Dense `torch.matmul` reference: ~0.04 ms
    - Cached `qvq_cpu_gemv` (cache enabled): ~0.06 ms (~9x faster than on-the-fly, ~0.7x of dense BLAS, ~4800x vs fallback)
  - The cached path effectively closes the remaining gap to dense BLAS by moving the planar decode cost out of the per-token inference loop.
- `ruff check` on `gptqmodel/utils/qvq_cpu.py` passes; `git diff --check` passes.

Next:
- Investigate whether transposing/storing the cached dense weight in a BLAS-friendlier layout can shrink the remaining ~0.02 ms overhead over native dense `torch.matmul` for M=1, or whether this is within `torch.matmul` dispatch overhead.
- Continue CPU optimization for quantization kernels (Viterbi/YAQA) or other inference paths if targets remain.

Done (CPU GEMV wrapper / `QVQLinear` dense-weight cache pass):
- Added `qvq_cpu_inner_weight` to `gptqmodel/utils/qvq_cpu.py` to expose the native dense dequantize op directly, with a fast `int(bits * 2)` transition-bits path and the same validation/contiguous handling as `qvq_cpu_gemv`.
- Cached `qvq_cpu_supported()` with `functools.lru_cache(maxsize=1)` so the hot-path CPU check is evaluated once per process.
- Cached `qvq_transition_bits()` in `gptqmodel/quantization/qvq_rates.py` with `functools.lru_cache(maxsize=128)` to avoid repeated `Fraction` construction from the same rate string/float.
- Added `QVQLinear._qvq_cpu_dense_inner()` which lazily builds and caches the dense `[in_features, out_features]` FP32 weight directly on the module, keyed by `(trellis id+version, bank_ids id+version, bank_alt_id, v2b4_p64, v2b2_p32)`. The `_inner_forward` CPU branch now calls `torch.matmul(x.contiguous(), self._qvq_cpu_dense_inner(...))` instead of the `qvq_cpu_gemv` wrapper, removing the per-token global-cache lookup and `qvq_cpu_gemv` call overhead.
- The transient `_qvq_cpu_dense_inner_cache` is excluded from `__getstate__` and cleared in `__setstate__` to avoid serializing the dense copy and to rebuild it after unpickling.
- Accuracy:
  - `qvq_cpu_inner_weight` output is bit-exact against `reconstruct_qvq_inner_weight` for V2 and V2B2-P32 test cases.
  - `QVQLinear` CPU forward for V2 and V2B2-P32 remains deterministic and within the 2e-3 inference tolerance.
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu):
  - `qvq_cpu_gemv` for `x=[1,2048] weight=[2048,2048] bits=3.5`: 0.1268 ms -> 0.1108 ms (~13% wrapper overhead reduction; now ~0.90x of dense `torch.matmul` at 0.1005 ms).
  - `QVQLinear._inner_forward` CPU branch after the first call is dominated by `torch.matmul` (~0.10 ms for the same shape), with the remaining ~0.01 ms being the in-module cache check, `x.contiguous()`, and `torch.matmul` Python dispatch.
- `ruff check` passes for `gptqmodel/utils/qvq_cpu.py`, `gptqmodel/quantization/qvq_rates.py`, and `gptqmodel/nn_modules/qlinear/qvq.py`.
- `tests/test_qvq.py -k "viterbi or tail_biting"` results unchanged: 89 passed, 112 skipped, 1 failed (pre-existing `squared_error` 1.19e-6 tolerance edge case in `test_qvq_l18_v4_torch_viterbi_recovers_exact_transition_consistent_path`).
- `tests/test_qvq_v2b2_p32.py` shows 1 new unrelated failure (`test_qvq_v2b2_p32_config_accepts_yaqa_and_weighted_block_ldlq`) caused by a recent `QVQConfig` validation rule (YAQA requires `viterbi_objective='euclidean'`) conflicting with that test; it is not touched by this CPU kernel pass.

Done (mimic CUDA VAQA/YAQA factored feedback on CPU):
- Added `gptqmodel_ext/qvq/qvq_yaqa_cpu.cpp` implementing a native CPU `yaqa_feedback` op that mirrors the CUDA factored kernel.
  - Computes the corrected 16x16 tile stack for one anti-diagonal:
    `corrected = source + bias + left(16,K) @ output_feedback(K,16) + left(16,16) + right(16,16)`.
  - Reuses a single `cross_tile` buffer and calls `at::mm_out` per tile, then fuses the source/bias/left/right/cross epilogue in a scalar C++ loop.
- Registered the op as `gptqmodel_qvq.yaqa_feedback` for CPU dispatch and exposed it through `gptqmodel/utils/qvq_cpu.py` as `qvq_cpu_yaqa_feedback` (with validation/contiguity helpers) and `_qvq_cpu_yaqa_feedback_op`.
- Updated `gptqmodel/quantization/qvq.py`:
  - Added `_incremental_cpu_factored_feedback` to `yaqa_inner` and the `yaqa_inner_v2b2_p32`/`encode_at_scale` call sites.
  - CPU factored feedback is now enabled for `device.type in ("cpu", "mps")` when `qvq_cpu_supported()` is true, replacing the per-tile Python `torch.matmul` fallback.
  - The precomputed `left_transformed_error`/`right_transformed_error` caches and the bottom-of-anti-diagonal update loop are reused directly for CPU (previously CUDA-only).
- Added `scripts/benchmark_qvq_yaqa_feedback_cpu.py` to compare the native op against the equivalent Python reference for one or more anti-diagonals.
- Accuracy:
  - `tests/test_qvq.py -k yaqa`: 91 passed, 1 skipped, 1 failed (`test_yaqa_config_supports_exact_rate_regularization_overrides`, pre-existing config regression not touched by this pass).
  - `tests/test_qvq_v2b2_p32.py -k yaqa`: 35 passed, 1 failed (pre-existing `test_qvq_v2b2_p32_config_accepts_yaqa_and_weighted_block_ldlq` config issue).
  - `tests/test_qvq.py::test_qvq_dual_v2_quantize_pack_and_torch_linear_are_synchronized[yaqa]` and `test_qvq_l18_v4_yaqa_quantize_pack_and_torch_linear_are_synchronized` pass with exact reconstruction.
  - Native `qvq_cpu_yaqa_feedback` is bit-exact vs the Python reference (`max abs diff = 0.0`).
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu) from `scripts/benchmark_qvq_yaqa_feedback_cpu.py`:
  - 256x256: Python 4.18 ms -> native 1.47 ms (~2.84x)
  - 512x512: Python 14.72 ms -> native 3.34 ms (~4.41x)
  - 1024x1024: Python 60.02 ms -> native 44.79 ms (~1.34x)
  - 2048x2048: Python 237.74 ms -> native 63.28 ms (~3.76x)
  - 4096x4096: Python 1038.54 ms -> native 327.71 ms (~3.17x)
- `ruff check` passes for `gptqmodel/quantization/qvq.py`, `gptqmodel/utils/qvq_cpu.py`, and `scripts/benchmark_qvq_yaqa_feedback_cpu.py`.
- `git diff --check` passes.

Done (mirror CUDA commit 053be9c1 "fuse YAQA cache updates" on CPU):
- Added a native fused in-place factored-feedback cache update `gptqmodel_qvq.yaqa_feedback_update_` in
  `gptqmodel_ext/qvq/qvq_yaqa_cpu.cpp`, mirroring the batched cuBLAS update in `qvq_yaqa_cuda.cu`. For each 16x16
  reconstructed tile of one anti-diagonal it applies
  `left[:, o0:o0+16] -= input_feedback[i0:i0+16, :].T @ Q` and `right[i0:i0+16, :] -= Q @ output_feedback[o0:o0+16, :]`
  directly into the strided cache tiles, removing the two batched `bmm` temporaries plus indexed scatters the Python
  path used.
  - Destinations are disjoint across an anti-diagonal, so no atomics are needed; work is split with `at::parallel_for`
    over left-cache rows and right-cache column chunks.
  - Runtime dispatch: AVX-512F/VL/DQ, then AVX2+FMA, then a scalar fallback (also used on non-x86).
  - Each rank-16 update is accumulated in registers starting from zero and folded into the cache once, so rounding
    stays at the scale of the (small) update rather than the (large) running cache value. This made the kernel
    bit-exact against the BLAS reference instead of ~1e-6 off.
- `gptqmodel/utils/qvq_cpu.py`: added `qvq_cpu_yaqa_feedback_update` plus `_qvq_cpu_yaqa_feedback_update_op`, and
  listed `yaqa_feedback_update_` in the extension's `required_ops`.
- `gptqmodel/quantization/qvq.py`: the cache-update branch now resolves the CPU op for
  `_incremental_cpu_factored_feedback` instead of unconditionally calling the CUDA op.
- Added `tests/test_qvq_cpu_yaqa.py` (dense FP32 reference for 32x64 / 64x32 / 48x48, plus geometry rejection) and
  `scripts/benchmark_qvq_yaqa_feedback_update_cpu.py`.
- Accuracy (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu):
  - Native vs Python/BLAS reference: `max abs diff = 0.0` (bit-exact) for 256/512/1024/2048 square cases.
  - Native and Python have identical FP64-reference relative error (2.0e-7 .. 5.4e-7), well inside the 1e-6 gate.
  - End-to-end `yaqa_inner(..., _incremental_cpu_factored_feedback=True)` is bit-exact vs the non-incremental
    reference for (32, 48) and (48, 32).
- Performance from `scripts/benchmark_qvq_yaqa_feedback_update_cpu.py` (full anti-diagonal sweep, median of 10):
  - 256x256: Python 1.66 ms -> native 0.38 ms (~4.35x)
  - 512x512: Python 4.73 ms -> native 0.75 ms (~6.35x)
  - 1024x1024: Python 20.43 ms -> native 3.25 ms (~6.28x)
  - 2048x2048: Python 114.80 ms -> native 21.28 ms (~5.40x)
- Tests: `tests/test_qvq_cpu_yaqa.py` 4 passed; `tests/test_qvq.py -k "yaqa or feedback"` 91 passed, 1 skipped,
  1 failed; `tests/test_qvq_v2b2_p32.py -k "yaqa or feedback"` 35 passed, 1 failed. Both failures
  (`test_yaqa_config_supports_exact_rate_regularization_overrides`,
  `test_qvq_v2b2_p32_config_accepts_yaqa_and_weighted_block_ldlq`) reproduce on a clean tree and are pre-existing
  config regressions.
- `ruff check` passes for the changed Python files; `git diff --check` passes.
