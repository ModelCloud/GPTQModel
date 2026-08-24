# CPU Kernel Implementation Log

Targets:
- QVQ V2 and V2B2-P32 inference GEMV
- QVQ Viterbi and YAQA quantization

Accuracy targets:
- Quantization kernels: max abs error <= 1e-6 vs reference
- Inference kernels: max abs error <= 2e-3 vs dense reference

## Hardware provenance (READ BEFORE COMPARING ANY NUMBERS IN THIS LOG)

Benchmark numbers in this log are **not comparable across entries unless they name
the same host**. Kernel timings, speedup ratios and thread-scaling conclusions are
all properties of a specific CPU, core count and thread budget -- not of the kernel
alone.

**Every entry that reports timings MUST begin with a `Hardware:` block.** Entries
without one predate this convention and are annotated below.

### Host A -- Intel Xeon Platinum 8559C (all entries before 2026-08-24)

Every performance number recorded in this log prior to 2026-08-24 was measured on:

```text
CPU:      Intel Xeon Platinum 8559C (Emerald Rapids)
ISA:      AVX-512 (full-width 512-bit datapath), AMX available
Threads:  8 logical cores (benchmarks used OMP_NUM_THREADS=8)
torch:    2.13.0+cpu
```

Carry-over caveats when comparing Host A numbers to any other machine:
- **8 logical cores.** Any speedup that came from `at::parallel_for` scaling will
  land differently on a host with a different core count. Absolute times are not
  transferable at all.
- **Full-width AVX-512.** Intel executes 512-bit ops on a 512-bit datapath. AMD
  Zen 4 double-pumps them through 256 bits, so AVX-512 register-width wins are
  smaller there even though the ISA is nominally identical.
- **AMX present.** Emerald Rapids has AMX, which also silently changes behaviour
  inside libraries: oneDNN and torch select different code paths when
  `dnnl::get_effective_cpu_isa() >= avx512_core_amx`. A kernel that appeared fast
  on Host A may have been benefiting from an AMX-gated library fast path that does
  not exist on non-AMX hardware.

### Host B -- AMD EPYC 9V33X (entries from 2026-08-24)

```text
CPU:        AMD EPYC 9V33X (Zen 4 "Genoa-X", 3D V-Cache)
CPUID:      family 25, model 17, stepping 1
-march:     znver4
ISA:        avx512f, avx512_vnni, avx512_bf16.  NO AMX
            (CPUID.(EAX=7,ECX=0).EDX = 0x10000010; amx_tile/amx_bf16/amx_int8 absent)
Topology:   96 physical cores / 192 threads present, but cgroup-capped to
            32 logical CPUs (nproc=32). Threads are scattered across CCDs.
Cache:      32 KiB L1D and 1 MiB L2 per core; 96 MiB L3 per CCD (1.1 GiB total)
Clock:      3716 MHz max
Kernel:     7.0.12-xing-7.0.12-epyc9v33x-native
Toolchain:  gcc 15.2.0, torch 2.13.0+cpu
```

Notes specific to Host B:
- **The hostname is misleading.** These machines are named `zen5-cpu-N` but report
  Zen 4 (`family 25, model 17`, `gcc -march=native` -> `znver4`). Trust CPUID, not
  the hostname.
- **Instances are not interchangeable.** Two hosts of this same model
  (`zen5-cpu-1` and `zen5-cpu-6`) produced torch SDPA medians differing by
  11-34% for identical work, due to differing cpuset allocation, CCD placement and
  tenancy. Re-baseline on the machine you are actually using; do not reuse a
  number from a sibling instance.
- **The large CCD-local L3 favours larger packed panels** than a typical x86
  target, so tile sizes tuned on Host A are not automatically optimal here.

### Required block for new entries

```text
Hardware: <CPU model> | <ISA flags that matter> | <N logical cores used,
          OMP_NUM_THREADS=N> | torch <version> | host <hostname>
```

### Measurement hygiene (learned the hard way on Host B)

- Do **not** use `OMP_PLACES=cores` under a cgroup cpuset. libgomp enumerates the
  full 192-CPU topology and pins the master thread to the lowest allowed CPU,
  ignoring `numactl --physcpubind`. This produced a **4.1x phantom slowdown**.
  Emit an explicit place list instead: `OMP_PLACES='{c1},{c2},...'`.
- Assert real affinity before timing. Read `/proc/self/status` `Cpus_allowed_list`
  and per-thread masks from `/proc/self/task/*/status`, and abort if the process
  does not actually hold the expected CPUs. A prior run measured a nominally
  32-thread baseline while confined to a single CPU, inflating it **2.26x**.
- Run benchmarks **exclusively**. Two concurrent timing jobs on the same 32-CPU
  cpuset inflated results **10-68%**.
- Sanity-check a baseline by **self-consistency of implied throughput across
  problem sizes**, not by agreement with a previously recorded number from a
  different host.

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

Not ported yet (CUDA commit 6b66f16f "batch YAQA family candidates on CUDA"):
- The new `_yaqa_inner_v2b2_family_batch_cuda` schedule runs all B2 family candidates through one candidate-batched
  pass, which required giving the CUDA `yaqa_feedback`/`yaqa_feedback_update_` ops an optional leading `families`
  dimension. The CPU side has no equivalent of the family-batched segmented Viterbi op
  (`viterbi_v2_segment_family_grid_trusted`), so adding the family dimension to the CPU feedback kernels alone would
  leave dead code. Revisit once (or if) a CPU family-grid Viterbi kernel exists.

Rejected (CUDA commit 6b66f16f "batch YAQA family candidates on CUDA", family-grid Viterbi on CPU):
- Built the missing prerequisite as an experiment: a family-aware refactor of `gptqmodel_ext/qvq/qvq_viterbi_banked_cpu.cpp`
  exporting `viterbi_banked_family_cpu` with `[families, batch, steps, V]` sequences and `[families, banks, 65536, V]`
  codebooks, flattening families into the row dimension exactly like
  `qvq_viterbi_v2_segment_family_grid_trusted_cuda` does, plus a CPU family-batched branch for the
  `yaqa_v2b2_sampled_family_selection` two-pass schedule.
- Exactness held: states, squared errors, segment bank ids, quantized values, selected `block_alt_id`, and the final
  quadratic loss were all bit-identical (`torch.equal`) to looping `qvq_cpu_viterbi_banked` once per family.
- Performance was a regression, so the experiment was not retained (Intel Xeon Platinum 8559C, AVX-512, 8 logical
  cores, torch 2.13.0+cpu; 3 families, 2 banks/family, 128 steps, V=2, 65536 states, transition_bits=5, segment 16):
  - 32 tiles: looped native 172.6 ms -> family-batched 225.4 ms (0.77x)
  - 64 tiles: looped native 408.5 ms -> family-batched 493.0 ms (0.83x)
  - 128 tiles: looped native 830.9 ms -> family-batched 1216.0 ms (0.68x)
- Why the CUDA lesson does not transfer: the CUDA win comes from filling an under-occupied grid and removing launch
  latency. The CPU kernel has neither problem, and its per-row cost/emission working set is
  `batch * banks * 65536 * 4 B` (~67 MB at 128 tiles), so tripling the row count triples an already
  out-of-cache footprint and makes the recurrence more memory-bound. Batching cannot amortize anything on CPU here:
  three native calls cost three Python dispatches, which are noise next to a ~150 ms kernel.
- Follow-up worth trying instead (not attempted here, and independent of this CUDA commit): block the existing CPU
  banked Viterbi over the batch dimension so the cost/emission buffers stay cache-resident. That would help the
  looped path too, and would be the precondition for family batching ever being neutral on CPU.

## 2026-08-21 sync: CUDA commit 05f5152d "batch B2 Block-LDLQ families on CUDA"

Not CPU-portable (same conclusion and evidence as the YAQA family batching in 6b66f16f):

- The commit adds `_block_ldlq_v2b2_family_batch_cuda`, which runs the three alternative B2 Block-LDLQ error
  histories in one candidate-batched pass through the CUDA family-grid Viterbi op
  (`viterbi_v2_segment_family_grid_trusted`). The win comes from filling an under-occupied CUDA grid (six CTAs per
  logical tile) and amortizing launch latency/segment barriers with a 128-tile window.
- The CPU equivalent was already built and measured for 6b66f16f (family-aware `viterbi_banked_family_cpu`
  experiment recorded above): bit-exact but 0.68x-0.83x versus looping the existing native banked Viterbi once per
  family, because the per-row cost/emission working set (~67 MB at 128 tiles) is already out of cache and tripling
  the row count only makes the recurrence more memory-bound. That measured regression applies unchanged to the
  Block-LDLQ family loop, which uses the same banked Viterbi kernel per candidate.
- The new dispatch is explicitly gated on `inner_weight.device.type == "cuda"`; the CPU path keeps the serial
  per-family `_block_ldlq_inner_v2_banked` loop (now expressed as a generator) with identical semantics.
- Verified the refactor left CPU behavior intact (Intel CPU, torch 2.13.0+cpu, native qvq_cpu extension built via
  the JIT path): `tests/test_qvq_v2b2_p32.py` 119 passed / 12 skipped / 1 pre-existing config failure
  (`test_qvq_v2b2_p32_config_accepts_yaqa_and_weighted_block_ldlq`, reproduces on a clean tree);
  `tests/test_qvq.py -k "v2b2 or block_ldlq or family"` 25 passed; `tests/test_qvq_cpu_yaqa.py` 4 passed;
  `ruff check gptqmodel/quantization/qvq.py` and `git diff --check` clean.
- The batch-dimension cache-blocking follow-up noted above remains the precondition for family batching ever being
  neutral on CPU; still not attempted.

## 2026-08-21 CPU follow-up: batch-dimension cache blocking in the banked Viterbi kernel

This is the follow-up flagged twice above (the precondition for CPU family batching): instead of widening the row
dimension, keep each row's recurrence independent but cache-resident.

- File changed: `gptqmodel_ext/qvq/qvq_viterbi_banked_cpu.cpp` (new benchmark: `scripts/benchmark_qvq_viterbi_banked_cpu.py`).
- Technique: the per-step loop was hoisted into a single shared `run_tile(tile_start, tile_end, parallel_inner)`
  body, so scratch (`costs`, `next_costs`, `emission`, `column_argmin` results) is allocated per tile rather than
  for the whole batch. Two regimes:
  - `batch >= at::get_num_threads()`: one row per thread (`parallel_inner=false`), so each thread's working set is
    `3 * banks * states * 4 B` (~1.5 MB at 2 banks / 65536 states) and the state loops run without inner
    `at::parallel_for` overhead.
  - smaller batches: tiles sized to a 4 MB budget, with the existing state/suffix `at::parallel_for` splits kept
    inside the tile so small batches still use all cores.
  The large `final_costs`-style staging buffer is gone; only compact per-row `best_final_costs` /
  `best_final_banks` / `final_end_states` survive the recurrence for traceback. Final-state selection keeps the
  original bank-major, state-minor scan with strict `<`, so tie-breaking is unchanged.
- Everything else is untouched: operator signature and registration, validation, codebook transpose + norm
  precompute, weighted/overlap/entry-exit constraint paths, int16 vs int32 backpointer choice and layouts, and the
  output tensors.
- Exactness: `states`, `squared_error`, and `segment_bank_ids` are bit-identical (`torch.equal`) to the previous
  implementation for batches 8/16/32/64/128, plain and weighted+overlap, plus edge sweeps over V=2/4, 1-4 banks,
  entry/exit constraints, transition-bit edges (int32 backpointer case) and both regimes.
- Performance (Intel Xeon Platinum 8559C, AVX-512, 8 logical cores, torch 2.13.0+cpu; 128 steps, V=2, 65536 states,
  2 banks, transition_bits=5, segment 16; median of 5 timed iterations, 2 warmups, 3 interleaved A/B rebuilds):
  - batch 16: 28.8 ms -> 22.4 ms (1.29x)
  - batch 32: 58.9 ms -> 44.2 ms (1.33x)
  - batch 64: 142.8 ms -> 90.3 ms (1.58x)
  - batch 128: 301.3 ms -> 180.6 ms (1.67x)
  Speedup grows with batch size, as expected for a footprint fix. Single-shot runs of this kernel are noisy
  (+-15%); only interleaved rebuild-and-measure passes were trusted.
- Tests: `tests/test_qvq.py -k "banked or viterbi"` 78 passed / 87 skipped / 1 pre-existing L18 tolerance failure
  (1.1920928955078125e-06 vs atol 1e-06, reproduces on a clean tree); `tests/test_qvq_v2b2_p32.py` 119 passed /
  12 skipped / 1 pre-existing YAQA config failure (`YAQA requires viterbi_objective='euclidean'`, also reproduces
  on a clean tree); `tests/test_qvq_cpu_yaqa.py` 4 passed; `ruff check` and `git diff --check` clean. No CUDA test
  ran (CPU-only host).
- Family batching on CPU is still not worth revisiting: the regression measured for 6b66f16f came from tripling an
  out-of-cache row footprint, and this change lowers the footprint per row rather than making wide rows cheaper.

## 2026-08-21 sync check: no new CUDA/MLX/MPS CPU-portable changes

Branch tip `b471bdbf` (CPU banked-Viterbi cache blocking) is in sync with `origin/agent/qvq-dual-v4`; working tree
clean. The newest GPU-side commits (`05f5152d` B2 Block-LDLQ family batching, `6b66f16f` YAQA family-candidate
batching) are already reviewed above and recorded as not CPU-portable, and no MLX/MPS kernel or
`gptqmodel/quantization/qvq.py` / `gptqmodel/nn_modules/qlinear/qvq.py` change has landed since. Nothing to port
this run.

## 2026-08-21 (07:39 UTC) sync check: no new CUDA/MLX/MPS CPU-portable changes

Branch tip `9388a66f` matches `origin/agent/qvq-dual-v4`; working tree clean. Newest GPU-side commits are still
`05f5152d` (B2 Block-LDLQ family batching) and `6b66f16f` (YAQA family-candidate batching), both already reviewed
and recorded as not CPU-portable, and no MLX/MPS kernel or `gptqmodel/quantization/qvq.py` /
`gptqmodel/nn_modules/qlinear/qvq.py` change has landed since. Nothing to port this run.

## 2026-08-21 (07:52 UTC) sync check: no new CUDA/MLX/MPS CPU-portable changes

Branch tip `c35340e4` matches `origin/agent/qvq-dual-v4`; working tree clean. Newest GPU-side commits remain
`05f5152d` (B2 Block-LDLQ family batching) and `6b66f16f` (YAQA family-candidate batching), both already reviewed
and recorded as not CPU-portable; no MLX/MPS kernel or `gptqmodel/quantization/qvq.py` /
`gptqmodel/nn_modules/qlinear/qvq.py` change has landed since. Nothing to port this run.

## 2026-08-21 (07:58 UTC) sync check: no new CUDA/MLX/MPS CPU-portable changes

Branch tip `ee8457a0` matches `origin/agent/qvq-dual-v4` (`git pull --rebase` reported "Already up to date");
working tree clean. Newest GPU-side commits are unchanged: `05f5152d` (B2 Block-LDLQ family batching) and
`6b66f16f` (YAQA family-candidate batching on CUDA), both already reviewed above and recorded as not CPU-portable.
No MLX/MPS kernel exists under `gptqmodel_ext/qvq/`, and no `gptqmodel/quantization/qvq.py` /
`gptqmodel/nn_modules/qlinear/qvq.py` change has landed since. Nothing to port this run.

## 2026-08-21 (08:05 UTC) sync check: no new CUDA/MLX/MPS CPU-portable changes

Branch tip `b84835b0` matches `origin/agent/qvq-dual-v4` (`git pull --rebase` reported "Already up to date");
working tree clean. Newest GPU-side commits are unchanged: `05f5152d` (B2 Block-LDLQ family batching) and
`6b66f16f` (YAQA family-candidate batching on CUDA), both already reviewed above and recorded as not CPU-portable.
No MLX/MPS kernel exists under `gptqmodel_ext/qvq/`, and no `gptqmodel/quantization/qvq.py` /
`gptqmodel/nn_modules/qlinear/qvq.py` change has landed since. Nothing to port this run.

## 2026-08-21 (07:59 UTC) sync check: no new CUDA/MLX/MPS CPU-portable changes

Tip `f3877fe5` matches `origin/agent/qvq-dual-v4` (pull --rebase: already up to date); tree clean. Newest GPU-side
commits are still `05f5152d` and `6b66f16f`, both already recorded as not CPU-portable; no MLX/MPS kernel or
`qvq.py` quant/qlinear change since. Nothing to port.

## 2026-08-21 (08:01 UTC) sync check: no new CUDA/MLX/MPS CPU-portable changes

Tip `c416a58e` matches `origin/agent/qvq-dual-v4` (pull --rebase: already up to date); tree clean. Newest GPU-side
commits are still `05f5152d` and `6b66f16f`, both already recorded as not CPU-portable; no MLX/MPS kernel exists
under `gptqmodel_ext/qvq/` and no `qvq.py` quant/qlinear change since. Nothing to port.

## 2026-08-21 (08:12 UTC) sync check: no new CUDA/MLX/MPS CPU-portable changes

Tip `b4e61800` matches `origin/agent/qvq-dual-v4` (pull --rebase: already up to date); tree clean. Newest GPU-side
commits are still `05f5152d` and `6b66f16f`, both already recorded as not CPU-portable; no MLX/MPS kernel exists
under `gptqmodel_ext/qvq/` and no `qvq.py` quant/qlinear change since. Nothing to port.

## 2026-08-21 (08:14 UTC) sync check: no new CUDA/MLX/MPS CPU-portable changes

Tip `60aa95d2` matches `origin/agent/qvq-dual-v4` (pull --rebase: already up to date); tree clean. Newest GPU-side
commits are still `05f5152d` and `6b66f16f`, both already recorded as not CPU-portable; no MLX/MPS kernel exists
under `gptqmodel_ext/qvq/` and no `qvq.py` quant/qlinear change since. Nothing to port.

## 2026-08-23 CPU sync: compile-time decode specialization and batched GEMV panels (superseded)

Ported CPU-relevant mechanics from CUDA commits `299795ba` (compile-time TransitionBits specialization and per-tile
decode reuse), `682ab5e9` (large-batch GEMV), `049e0a94` (double-buffered staging), and `a4f74f63` (large-row staging
gate).

> The benchmark and M>1 panel mechanics in this original entry were invalid: the panel loop surrounded the tile
> decode and multiplied decode work by `ceil(M/4)`. The corrected decode-once implementation and measurements are
> recorded in the follow-up section below.

- File changed: `gptqmodel_ext/qvq/qvq_gemv_cpu.cpp`. No Python wrapper change was needed.
- AVX-512 unpack, decode, and accumulation helpers are specialized for `E=2..8` and selected by a host-side
  switch. The existing runtime-generic AVX-512 path remains the default for `E>=9`, and the scalar path remains
  reachable on CPUs without AVX-512.
- M>1 now uses a four-row register-resident panel, four-output-tile blocking, and 1/2/3-tile tails. Each decoded
  tile is reused across active panel rows while `__m512` accumulators remain live across the input-tile loop.
  The M=1 single-tile tail is dispatched through the same E-specialized helpers; no specialized accumulation helper
  is left unused.
- A software-prefetch experiment for the next tile's trellis words and bank byte was rejected. Using the same
  generated inputs, `OMP_NUM_THREADS=8`, warmup 1, and 5 timed iterations, prefetch was slower in 23/30 cases
  (mean prefetch/no-prefetch ratio 1.242x); timings were noisy and there was no stable win. The prefetch code was
  removed.

Before/after benchmark matrix (same generated inputs, `OMP_NUM_THREADS=8`, warmup 3, 10 timed iterations; native
cache disabled with `QVQ_CPU_GEMV_DENSE_CACHE=0`):

```text
shape  bits/E  M  before_ms after_ms speedup fallback_ms dense_ms max_abs mean_abs rel_l2
2048  2.0/4   1     0.3789   0.2925   1.295    107.8967   0.0547 1.983643e-04 2.815373e-05 8.374410e-07
2048  2.0/4   4     0.4381   0.5144   0.852    113.6270   0.0969 2.288818e-04 2.638850e-05 8.258501e-07
2048  2.0/4   8     0.5142   1.2332   0.417    125.6121   0.1200 3.662109e-04 2.706542e-05 8.485807e-07
2048  2.0/4  16     0.6437   2.1473   0.300    127.8544   0.2127 2.899170e-04 2.715579e-05 8.498584e-07
2048  2.0/4  32     0.8858   4.3331   0.204    123.6179   0.3467 2.822876e-04 2.692133e-05 8.483215e-07
2048  3.5/7   1     0.6238   0.5083   1.227    122.6366   0.0554 2.746582e-04 2.677591e-05 8.364130e-07
2048  3.5/7   4     0.7150   0.5448   1.312    120.9331   0.1015 2.593994e-04 2.730524e-05 8.340151e-07
2048  3.5/7   8     0.5715   1.0684   0.535    128.2008   0.1415 2.975464e-04 2.715777e-05 8.382312e-07
2048  3.5/7  16     1.2641   2.1455   0.589    133.4214   0.2183 3.204346e-04 2.677210e-05 8.384485e-07
2048  3.5/7  32     1.0826   4.2471   0.255    126.8544   0.3362 4.119873e-04 2.723680e-05 8.458420e-07
2048  5.0/10   1     0.3489   0.6752   0.517    123.5511   0.0602 2.975464e-04 2.688648e-05 8.493782e-07
2048  5.0/10   4     0.6288   0.8111   0.775    106.4912   0.1045 2.746582e-04 2.708281e-05 8.522954e-07
2048  5.0/10   8     0.7791   1.2065   0.646    101.5774   0.1024 2.746582e-04 2.702863e-05 8.455462e-07
2048  5.0/10  16     0.5904   1.8974   0.311    110.3753   0.2456 3.051758e-04 2.712607e-05 8.531118e-07
2048  5.0/10  32     0.9438   3.8276   0.247    123.6177   0.2727 3.585815e-04 2.716054e-05 8.481605e-07
4096  2.0/4   1     2.7157   1.3730   1.978    495.2257   0.3562 4.730225e-04 5.271901e-05 1.198250e-06
4096  2.0/4   4     2.0724   1.3820   1.500    506.9257   0.5823 5.950928e-04 5.297381e-05 1.192989e-06
4096  2.0/4   8     3.1731   5.3063   0.598    488.4990   0.7963 6.713867e-04 5.323759e-05 1.181818e-06
4096  2.0/4  16     3.7583   5.5431   0.678    502.1492   1.0433 5.798340e-04 5.342880e-05 1.185504e-06
4096  2.0/4  32     7.2714  12.7508   0.570    501.1231   1.3320 8.697510e-04 5.436152e-05 1.192706e-06
4096  3.5/7   1     2.0064   2.1162   0.948    552.7051   0.3518 5.035400e-04 5.303171e-05 1.191142e-06
4096  3.5/7   4     2.3179   1.3073   1.773    532.0345   0.6957 5.645752e-04 5.409462e-05 1.188045e-06
4096  3.5/7   8     3.2185   2.6214   1.228    544.6705   0.6818 5.798340e-04 5.337541e-05 1.190042e-06
4096  3.5/7  16     4.3085   5.2554   0.820    527.2081   0.8638 7.324219e-04 5.415780e-05 1.190389e-06
4096  3.5/7  32     7.7199  11.8492   0.652    531.4359   1.2461 7.019043e-04 5.357194e-05 1.186637e-06
4096  5.0/10   1     2.4136   1.6578   1.456    520.1604   0.3724 4.730225e-04 5.285863e-05 1.149504e-06
4096  5.0/10   4     2.8771   2.8562   1.007    504.6780   0.6054 5.493164e-04 5.505025e-05 1.200765e-06
4096  5.0/10   8     2.4948   4.0198   0.621    573.7360   0.6639 5.798340e-04 5.340867e-05 1.184227e-06
4096  5.0/10  16     2.3740   7.8420   0.303    506.7728   0.9771 6.713867e-04 5.345992e-05 1.178163e-06
4096  5.0/10  32     6.8339  22.4651   0.304    512.3156   1.3577 7.171631e-04 5.359608e-05 1.188666e-06
```

The native-vs-dense accuracy gate passed for every row (`max_abs <= 8.697510e-04`, `mean_abs <= 5.436152e-05`,
`relative-L2 <= 1.200765e-06`); Python fallback outputs were exact against the same dense references. The
pre-change inner-weight artifact comparison remained bit-identical (`torch.equal`) for all 39 unbanked and banked
cases, and the wider GEMV sweep remained at `max_abs=7.62939453e-06` before the final tail-dispatch edit.

Verification:

- `pytest -q tests/test_qvq.py tests/test_qvq_v2b2_p32.py tests/test_qvq_v2b4_p64.py tests/test_qvq_output_alignment.py`:
  764 passed, 263 skipped, 9 failed. Eight configuration failures reproduced on a clean
  `origin/agent/qvq-dual-v4` worktree; the remaining bitshift case passed when isolated and is order-sensitive.
- `ruff check`: no changed Python files.
- `git diff --check`: clean.
- JIT build time increased from approximately 16 seconds before specialization to approximately 22 seconds after it
  (individual rebuilds varied up to 26 seconds under test load).
- No CUDA, MLX, or MPS tests ran on this CPU-only host.

## 2026-08-23 CPU sync follow-up: decode-once M>1 GEMV blocking

This follow-up fixes the M>1 regression documented in the superseded entry above. The source CUDA lessons were
`299795ba` (compile-time TransitionBits specialization and per-tile decode reuse), `682ab5e9` (large-batch reuse),
`049e0a94` (double-buffered staging), and `a4f74f63` (large-row staging gate).

- File changed: `gptqmodel_ext/qvq/qvq_gemv_cpu.cpp`; no Python wrapper change was needed.
- AVX-512 unpack, decode, and M=1 accumulation helpers are specialized for `E=2..8`; host switches select those
  helpers, with runtime-generic AVX-512 for `E>=9` and the scalar fallback retained.
- The corrected M>1 path keeps the four-output-tile block outermost. For each input tile and output tile in the
  block it decodes once, then walks all M rows in four-row panels. A 64-byte-aligned `float` buffer sized
  `M*B*16` is allocated once per `at::parallel_for` task and accumulators are loaded/stored once per row/tile/input
  tile, with the 16 FMAs chained in registers. M tails and 1/2/3-tile output tails remain supported.
- The earlier panel-outside-decode structure was rejected because every four-row panel re-decoded all input tiles,
  multiplying decode work by `ceil(M/4)` and causing the measured M>1 regression. The corrected dataflow fixes the
  cause rather than masking the symptom.
- Software prefetch of the next trellis words and bank byte was also rejected: it was slower in 23/30 cases
  (mean prefetch/no-prefetch ratio 1.242x) with no stable win.

Benchmark protocol: same generated inputs, `OMP_NUM_THREADS=8`, `QVQ_CPU_GEMV_DENSE_CACHE=0`, five warmups, and
30 native timings. Python fallback and dense columns use five warmups and five timings. `speed` is before median /
after median; `speed_min` is before minimum / after minimum. Before is the original pre-specialization kernel.

```text
shape bits/E M before_med before_min after_med after_min speed speed_min python dense max_abs mean_abs rel_l2
2048 2.0/4  1 0.4364 0.3224 0.2430 0.2380 1.796 1.355 99.1 0.059 1.983643e-04 2.815373e-05 8.374410e-07
2048 2.0/4  4 0.4393 0.4270 0.3322 0.2986 1.323 1.430 98.4 0.089 2.288818e-04 2.638850e-05 8.258501e-07
2048 2.0/4  8 0.5180 0.4892 0.3074 0.3033 1.685 1.613 96.7 0.100 3.662109e-04 2.706542e-05 8.485807e-07
2048 2.0/4 16 0.5313 0.5197 0.3739 0.3672 1.421 1.416 99.0 0.149 2.899170e-04 2.715579e-05 8.498584e-07
2048 2.0/4 32 1.3265 1.3165 0.4968 0.4886 2.670 2.694 100.4 0.253 2.822876e-04 2.692133e-05 8.483215e-07
2048 3.5/7  1 0.4559 0.3911 0.2657 0.2605 1.716 1.501 97.7 0.048 2.746582e-04 2.677591e-05 8.364130e-07
2048 3.5/7  4 0.5133 0.4795 0.3611 0.3155 1.421 1.520 98.0 0.094 2.593994e-04 2.730524e-05 8.340151e-07
2048 3.5/7  8 0.5790 0.5368 0.3235 0.3178 1.790 1.689 100.4 0.100 2.975464e-04 2.715777e-05 8.382312e-07
2048 3.5/7 16 0.6291 0.6020 0.3867 0.3825 1.627 1.574 95.2 0.153 3.204346e-04 2.677210e-05 8.384485e-07
2048 3.5/7 32 1.1933 1.1653 0.5068 0.5008 2.354 2.327 102.1 0.292 4.119873e-04 2.723680e-05 8.458420e-07
2048 5.0/10 1 0.3520 0.3447 0.5694 0.5529 0.618 0.623 120.1 0.054 2.975464e-04 2.688648e-05 8.493782e-07
2048 5.0/10 4 0.4618 0.4561 0.7825 0.6840 0.590 0.667 108.6 0.103 2.746582e-04 2.708281e-05 8.522954e-07
2048 5.0/10 8 0.5348 0.5089 0.7809 0.5528 0.685 0.921 97.5 0.100 2.746582e-04 2.702863e-05 8.455462e-07
2048 5.0/10 16 0.5907 0.5625 0.5407 0.5339 1.093 1.053 94.6 0.146 3.051758e-04 2.712607e-05 8.531118e-07
2048 5.0/10 32 1.3710 1.3265 1.1542 1.1485 1.188 1.155 100.3 0.260 3.585815e-04 2.716054e-05 8.481605e-07
4096 2.0/4  1 1.5822 1.3553 1.0751 1.0589 1.472 1.280 436.1 0.359 4.730225e-04 5.271901e-05 1.198250e-06
4096 2.0/4  4 1.8556 1.8084 1.2726 1.2624 1.458 1.433 497.1 0.486 5.950928e-04 5.297381e-05 1.192989e-06
4096 2.0/4  8 2.1814 2.1185 1.3279 1.3114 1.643 1.615 445.4 0.524 6.713867e-04 5.323759e-05 1.181818e-06
4096 2.0/4 16 2.6279 2.2381 1.6212 1.5761 1.621 1.420 442.3 0.663 5.798340e-04 5.342880e-05 1.185504e-06
4096 2.0/4 32 5.3761 5.2780 2.1317 2.1254 2.522 2.483 455.0 0.950 8.697510e-04 5.436152e-05 1.192706e-06
4096 3.5/7  1 1.7320 1.5288 1.0269 1.0217 1.687 1.496 477.0 0.355 5.035400e-04 5.303171e-05 1.191142e-06
4096 3.5/7  4 1.9632 1.9012 1.1450 1.1317 1.714 1.680 473.2 0.486 5.645752e-04 5.409462e-05 1.188045e-06
4096 3.5/7  8 2.2019 2.1452 1.2694 1.2543 1.735 1.710 481.3 0.535 5.798340e-04 5.337541e-05 1.190042e-06
4096 3.5/7 16 2.5056 2.3924 1.5233 1.5123 1.645 1.582 483.3 0.672 7.324219e-04 5.415780e-05 1.190389e-06
4096 3.5/7 32 5.5433 5.4956 2.0510 2.0295 2.703 2.708 475.2 0.948 7.019043e-04 5.357194e-05 1.186637e-06
4096 5.0/10 1 1.4091 1.3945 1.3866 1.3766 1.016 1.013 479.2 0.375 4.730225e-04 5.285863e-05 1.149504e-06
4096 5.0/10 4 1.8918 1.8538 1.7079 1.6905 1.108 1.097 536.2 0.618 5.493164e-04 5.505025e-05 1.200765e-06
4096 5.0/10 8 2.1775 2.1170 3.0985 3.0280 0.703 0.699 510.2 0.779 5.798340e-04 5.340867e-05 1.184227e-06
4096 5.0/10 16 2.4314 2.3204 2.3288 2.2341 1.044 1.039 457.7 0.670 6.713631e-04 5.345992e-05 1.178163e-06
4096 5.0/10 32 5.4548 5.4158 5.5539 4.9851 0.982 1.086 472.5 0.949 7.171631e-04 5.359608e-05 1.188666e-06
```

For E=2/4 and E=3.5/7, every required M>1 case beats the pristine baseline by median. E=5.0/10 remains on
the generic fallback; its near-neutral and negative deltas are not attributed to the specialized panel. The
maximum native-vs-dense errors are `max_abs=8.697510e-04`, `mean_abs=5.436152e-05`, and `relative-L2=1.200765e-06`.
The sub-10% and generic-path cases were repeated with the same five-warmup/30-timing protocol; results varied
substantially between runs (for example, 4096/E10/M1 was 1.408 -> 2.347 ms and 2048/E10/M1 was 0.349 -> 0.351
ms), confirming that those small deltas are measurement noise rather than a panel conclusion.

The requested test command reported `764 passed, 263 skipped, 9 failed`; eight failures reproduce on clean origin.
The bitshift case was run isolated and in the full selected set on clean origin and passed in both modes, so it is
not order-sensitive on origin. No Python files changed (Ruff not applicable), and `git diff --check` was clean.
The JIT build increased from approximately 16 seconds before specialization to approximately 22 seconds after it.

## 2026-08-24 non-banked G-only Viterbi recurrence

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

- Replaced the non-banked full-state `costs`, `next_costs`, and `emission_buf` frontiers with two per-suffix FP32
  `G` buffers. Emission, predecessor-`G` addition, strict suffix argmin, and compressed backpointer production now
  run in one suffix-parallel pass per DP step instead of three `at::parallel_for` phases.
- Preserved the V2/V4 AVX-512 emission instruction order and the ascending-prefix strict-`<` reduction order. Final
  candidates compare their reconstructed full state indices on equal costs, preserving the original lowest-state
  tie rule. Saved pre-change states and squared error compare bit-exactly; eager-oracle, ties, weighted, overlap,
  tail-biting, W1-W8, and planar packing coverage passed.
- Initial raw `viterbi_cpu` measurement (batch 1, 128 steps, V2, 65,536 states, transition bits 5; 3 warmups, 21
  samples): median 5.980552 ms -> 2.438298 ms, 2.45x. The authoritative post-review paired rerun used the cached
  pristine and final fixed shared objects in the same quiet window with 10 warmups and 51 samples: 5.896012 ms ->
  2.514544 ms, **2.34x** (minima 5.721410 -> 2.484028 ms).
- Local before/after tests: focused Viterbi 90 -> 91 passed / 112 skipped / 653 deselected (one new regression);
  V2B2-P32 119
  passed / 12 skipped / 1 unrelated configuration failure both times. The historical L18 V4 loss edge case passed
  on this host before and after. `git diff --check` and Ruff on the changed test passed. Repository-wide Ruff 0.14.2
  reported 535 pre-existing findings.
- Blocking defect found in adversarial review: with `state_count=16`, `transition_bits=2`, `overlap=1`, and one
  step, the first G-only version applied both the initial high-bit and final low-bit constraints and selected state
  5, while the old step-0 early-continue semantics applied only the initial constraint and selected state 6. The
  regression was written and observed failing before the kernel fix. Final selection now constrains the suffix only
  when `step_count > 1`.
- The new test covers the concrete counterexample plus randomized one- and two-step comparisons against an eager
  Torch implementation of the legacy recurrence, across transition bits 1-4, overlap values, multiple batches,
  ties, and weighted/unweighted costs. States and squared errors are exact. The adjacent-step audit found no other
  first/last-step control-flow mismatch.
- A second adversarial review found two native-boundary defects: `-1` conflated "no initial constraint" with a
  negative overlap, and AVX-512 narrowed a large int64 overlap before checking its range. Tests were written first;
  a mixed batch returned invalid one-step states 59 and 6, and invalid three-step tracebacks `[12,48,0]` and
  `[4,16,0]`, instead of zero paths/infinite errors. The high-level quantizer already validates the range, but the
  Python CPU wrapper and public torch op remain directly reachable. A separate constraint boolean plus an int64
  `[0,suffix_count)` check before narrowing now implements the invalid-overlap policy identically in AVX-512 and
  scalar code for one and multiple steps.
- New coverage includes negative, truncating-large, and mixed valid/invalid batches at one and three steps, checking
  states and squared error. A deterministic adjacent-boundary fixture proves the two-step final mask is active:
  unconstrained state 10 has suffix 2, while overlap 1 produces path `[6,9]` with squared error 8.
- Final focused tests are 95 passed / 112 skipped / 653 deselected (baseline 90 passed; five added cases). V2B2-P32
  remains 119 passed / 12 skipped / 1 identical unrelated configuration failure. The historical L18 V4 case still
  passes, and `git diff --check` plus Ruff pass.
- Post-second-review paired raw timing, using the same explicit placement, verified 32 singleton affinities, 10
  warmups, and 51 samples: pristine 5.893175 ms median (5.618564 min) -> fixed 2.465951 ms (2.403298 min), **2.39x**.
  States/error had identical combined SHA-256
  `5be22fad56a88fff21c3b510ebcf9c982b7fd298e36247a5fa391ba4f563f86e`.
- A third adversarial review found `suffix_begin + 1` was evaluated before the final overlap range check, causing
  signed-overflow UB for `INT64_MAX`. The expanded ordinary test passed before the fix because observed wraparound
  was overwritten; a focused UBSan run reported the overflow explicitly. The addition is now formed only in the
  valid constrained branch. Invalid mixed-batch coverage includes step counts 1/2/3, `INT64_MIN`, `INT64_MAX`,
  `suffix_count`, negative values, and a `2**32 + valid_overlap` alias, checking zero paths and infinite errors.
- Post-third-fix fixed median was 2.502110 ms (minimum 2.386086 ms), consistent with the authoritative 2.465951 ms.
  Three pristine reruns were rejected for 83-114 ms maxima and unstable 9.160888/7.490947/7.783496 ms medians;
  no inflated speedup is claimed, and the prior clean paired **2.39x** remains authoritative. Affinity verification
  and the exact output digest passed in every run.
- Banked scope check: `qvq_viterbi_banked_cpu.cpp` was not modified. Saved artifacts for batches 16/32/64/128 were
  exact for states, squared error, and segment bank IDs.
- Rejected banked timings: pre-change medians 99.7/14.5/35.8/69.2 ms; contaminated after runs showed 95.5-800.5 ms
  spreads and false 0.30-0.78x ratios, with a repeat spanning 18.1-159.4 ms. Because the banked source was unchanged
  and the host noise was obvious, these numbers are recorded but not used for a performance conclusion.
- Measurement placement was explicitly
  `{24},{27},{28},{42},{43},{44},{45},{54},{55},{65},{90},{94},{96},{104},{113},{114},{118},{123},{135},{139},{143},{150},{156},{161},{164},{169},{172},{173},{175},{176},{179},{183}`;
  `/proc/self/task/*/status` proved all 32 singleton worker affinities before timing. Runs were exclusive, sequential,
  and blocking. `scripts/benchmark_qvq_viterbi_banked_cpu.py` drove the banked artifact check; the repository's
  `scripts/benchmark_qvq_viterbi.py` is CUDA-only, so raw CPU timing called the registered op directly.

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

## 2026-08-24 CORRECTION: scope of the non-banked G-only bit-exactness claim

**This entry corrects the entry "2026-08-24 non-banked G-only Viterbi recurrence" above.**

That entry stated that saved pre-change states **and squared error** compared
bit-exactly, alongside a list of coverage (eager-oracle, ties, weighted, overlap,
tail-biting, W1-W8, planar packing). **The squared-error half of that claim was
overstated.** It was verified for the specific saved/hashed benchmark inputs used by
that PR, but the wording implied general preservation, and that broader claim is
false.

### Measured

Driving the production `gptqmodel_qvq.viterbi_cpu` op with identical deterministic
inputs on `ede2695e` (pre-change) and `2f4bea8a` (post-change), across the 17 valid
configurations exercised by `tests/test_qvq_viterbi_cpu_opt.py`:

| Property | Result |
|---|---|
| Complete selected-state paths | **identical in all 17 configurations** |
| Tie winners, traceback paths, bank IDs | **identical** |
| FP32 squared-error bit pattern | **changed in 6 of 17 configurations (12 of 44 batch outputs)** |

Deltas are last-bit FP32 differences, roughly `1e-7` to `2e-6` absolute.

### Cause

Removing the full-state frontier changes *when* a rounded FP32 minimum is stored and
reused, even though the emission instruction order was deliberately retained. These
are ordinary non-associative FP32 recurrence-order differences, not a different
discrete minimum -- every selected path matched. The independently written
`qvq_viterbi_cpu_opt` kernel reproduces the pre-change baseline bit-for-bit in all 17
configurations, which corroborates that the change in encoding came from the G-only
schedule.

Notably, no single FP32 encoding is universally "correct" here: across the 12 changed
outputs the eager Torch oracle matched the pre-change baseline 3 times, the
post-change baseline 4 times, and neither 5 times.

### Consequence, and what remains guaranteed

- **Load-bearing and still exact:** selected states, tie winners, bank IDs and packed
  words. Ordinary reconstruction and packing consume `states`/`values`, not the cost
  (`qvq.py:2723-2731`, `:3333-3349`, `:4305-4320`, `:7588-7601`).
- **Conditionally load-bearing:** `squared_error` ranks candidates when
  `tail_biting_candidates > 1` (`qvq.py:2217-2235` flattened/argmin,
  `:2237-2258` serial `candidate.squared_error < best.squared_error`). A last-bit
  perturbation can only change packed output if it crosses or flips a candidate
  comparison, i.e. at an exact or near tie. **This is a small but real risk and is not
  currently covered by a test.**
- Everywhere else -- including the zero-overlap/full-shift path (`qvq.py:2121-2131`)
  and the default single-candidate flow -- the terminal cost is reporting metadata
  after states are selected.

### Guidance for future entries

**Do not claim bit-exactness more broadly than you measured.** State the exact
configuration set that was compared. Bitwise equality of a derived FP32 cost between
two independently scheduled implementations is a stronger contract than this
algorithm requires, and it is not durable across legitimate re-orderings; assert exact
discrete outputs plus a numerical tolerance on cost instead.

## 2026-08-24 — QVQ direct GEMV small-output team sizing

- **Change:** when the AVX-512 direct GEMV has at most four four-output blocks, run the unchanged block kernels in
  an explicit team with one worker per block. N=256 therefore uses four workers instead of a 32-worker
  `at::parallel_for`. Larger shapes keep the existing scheduling. Dispatch, FP32 arithmetic, decode, FMA, and
  per-output accumulation order are unchanged.
- **Accepted result:** across W2/W3.5/W4 and M=1/8/32, N=256 improved 1.063-1.171x (mean 1.129x). M1 changed from
  0.1229/0.1305/0.1216 ms to 0.1080/0.1114/0.1075 ms. The remaining direct/dense ratio is 2.05-2.13x, so this is
  not dense parity. Large-shape direct medians remained effectively unchanged.
- **Accuracy:** all 36 before/after max-absolute-error cells were identical. Worst remained 1.739501953e-3 at
  8192x2048 W2 M32, within the 2e-3 contract.
- **Rejected:** a fully serial N=256 path measured 0.4075/0.4204/0.4040 ms for W2/W3.5/W4 M1 and
  0.7851/0.8128/0.7966 ms at M32, roughly 3.2-3.4x slower than baseline. The four output blocks need four workers.
- **Protocol:** 10 warmups, 50 timings, exclusive runs, explicit singleton `OMP_PLACES` over CPUs
  24,27,28,42-45,54-55,65,90,94,96,104,113-114,118,123,135,139,143,150,156,161,164,169,172-173,175-176,179,183.
  Affinity was asserted once immediately before each timed series; N=256 direct proved `{24},{27},{28},{42}`.
  Raw matrices are `/home/ubuntu/work/qvq-findings/gemv_smalln_before.csv` and `gemv_smalln_after.csv`; full tables
  and test baselines are in `RESULTS.md`.
