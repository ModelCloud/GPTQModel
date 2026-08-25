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
  and test baselines are in `docs/kernels/qvq_segmented_viterbi_cpu_results.md`.
## 2026-08-24 direct packed GEMV production dispatch and dense-cache deletion

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4, no AMX) | 32 logical cores, OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

- Deleted `QVQLinear._qvq_cpu_dense_inner` and its persistent `_qvq_cpu_dense_inner_cache`, including the dead
  pickle handling. Optimized CPU inference now explicitly passes `use_dense_cache=False` to `qvq_cpu_gemv`, so the
  native packed-trellis op is selected even when the standalone wrapper's dense-cache environment policy is enabled.
- `_reference_inner_forward` is unchanged. Training and unsupported configurations still reconstruct locally and
  remain covered by the full suites. A hostile-environment regression sets `QVQ_CPU_GEMV_DENSE_CACHE=1`, replaces
  dense materialization with an assertion failure, runs production forward successfully, and proves the module has
  no dense-cache attribute.
- Timing used the repository `scripts/benchmark_qvq_gemv_cpu.py`, 10 warmups and 50 measured iterations for each
  uninterrupted direct and dense series. `ratio` is direct/dense, so values over 1 are direct regressions. The
  script asserted the explicit singleton placements once immediately after warmup and before every timed series.

```text
KxN        rate M  dense_ms direct_ms ratio
2048x2048  W2    1   0.2801    0.1732 0.618
                   8   0.1810    0.2215 1.224
                  32   0.3129    0.4144 1.324
           W3.5  1   0.3800    0.1824 0.480
                   8   0.1801    0.2335 1.297
                  32   0.3167    0.4132 1.305
           W4    1   0.2807    0.1766 0.629
                   8   0.1785    0.2269 1.271
                  32   0.3048    0.3992 1.310
2048x8192  W2    1   1.1937    0.6626 0.555
                   8   0.6525    0.8272 1.268
                  32   1.1524    1.3672 1.186
           W3.5  1   1.2419    0.6973 0.561
                   8   0.6632    0.8485 1.279
                  32   1.1707    1.3978 1.194
           W4    1   1.1994    0.6731 0.561
                   8   0.6633    0.8368 1.262
                  32   1.1732    1.3745 1.172
8192x2048  W2    1   1.2133    0.6509 0.536
                   8   0.6621    0.8073 1.219
                  32   1.1670    1.4240 1.220
           W3.5  1   1.2539    0.6919 0.552
                   8   0.6457    0.8365 1.295
                  32   1.1709    1.4160 1.209
           W4    1   1.5449    0.6647 0.430
                   8   0.6593    0.8215 1.246
                  32   1.1528    1.4496 1.258
2048x256   W2    1   0.0509    0.1238 2.432
                   8   0.0339    0.1502 4.431
                  32   0.0551    0.2508 4.552
           W3.5  1   0.0500    0.1310 2.620
                   8   0.0351    0.1555 4.430
                  32   0.0535    0.2499 4.671
           W4    1   0.0512    0.1228 2.398
                   8   0.0333    0.1479 4.441
                  32   0.0538    0.2541 4.723
```

- The regression is explicit: at M=8/32, direct is 1.17-1.32x slower for the large shapes and 4.43-4.72x slower
  for N=256. At M=1, N=256 is 2.40-2.62x slower. These measurements preceded PR #18. While this work was in
  progress, PR #18 merged into `main`; the branch inherited it only when rebased onto the new PR target afterward.
- The current post-rebase N=256 medians below include inherited PR #18. Its four-worker direct team and the active
  dense team were each asserted immediately before their series. The real regression remains 2.09-2.17x at M=1,
  3.73-3.99x at M=8, and 4.01-4.25x at M=32.

```text
rate M  dense_ms direct_ms ratio
W2    1   0.0502    0.1088 2.167
      8   0.0350    0.1305 3.729
     32   0.0557    0.2233 4.009
W3.5  1   0.0519    0.1115 2.149
      8   0.0345    0.1378 3.994
     32   0.0547    0.2304 4.212
W4    1   0.0512    0.1072 2.094
      8   0.0345    0.1329 3.852
     32   0.0547    0.2323 4.247
```
- All 36 accuracy cases passed at `rtol=0`, `max_abs <= 2e-3`. The measured worst case was
  `1.739502e-3` at 8192x2048 W2 M32. No tolerance was loosened.
- Seven-layer fresh-process RSS (two 2048x2048, two 2048x8192, one 8192x2048, two 2048x256): W2 direct added
  0.65 MiB and dense retention added 205.0 MiB more; W3.5 direct added 0.81 MiB and dense added 203.9 MiB more;
  W4 direct added 0.65 MiB and dense added 223.0 MiB more. RSS came from `/proc/self/smaps_rollup` after packed
  construction, after one direct call per layer, and after one retained-dense call per layer.
- A two-iteration affinity probe was rejected before measurement because torch initialized its team size from the
  master thread's singleton binding. The benchmark now restores the explicit requested team with
  `torch.set_num_threads(32)` and verifies the complete placement set. The rejected probe is not used for any
  performance conclusion.
- Local before baseline: `tests/test_qvq.py` 613 passed / 247 skipped;
  `tests/test_qvq_v2b2_p32.py` 119 passed / 12 skipped / one pre-existing configuration failure. After:
  655 passed / 248 skipped and 119 passed / 12 skipped / the same one failure, respectively. The primary suite's
  final count includes the oracle tests merged upstream while this branch was in progress.
- Measurement placement was
  `{24},{27},{28},{42},{43},{44},{45},{54},{55},{65},{90},{94},{96},{104},{113},{114},{118},{123},{135},{139},{143},{150},{156},{161},{164},{169},{172},{173},{175},{176},{179},{183}`.
  Runs were exclusive, sequential, and blocking. `OMP_PLACES=cores` was never used.

## 2026-08-24 — opt-in Viterbi discrete-divergence defect record (not a performance entry)

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4, no AMX) | 1 logical core used,
          OMP_NUM_THREADS=1 | torch 2.13.0+cpu | host zen5-cpu-6

- **MEASURED** at `de25f203b7d405f0e9bc3f910785d102f9225aef`: production `qvq_cpu_viterbi` matched the eager
  Torch oracle exactly for complete states, planar packed words, and squared error in all five divergent
  configurations. The opt-in/benchmark-only `qvq_cpu_viterbi_opt` differed from both on states and packed words.
  **INFERRED directly from those measurements:** this is an opt-kernel defect, not a production correctness bug.
- **MEASURED affected configurations** `(V, transition_bits, steps, batch)`:
  `(2, 8, 32, 128)`, `(2, 7, 128, 32)`, `(2, 7, 128, 128)`, `(2, 8, 128, 32)`, and
  `(2, 8, 128, 128)`.
- **MEASURED** across the standardized 96-configuration matrix: 17 cases exceeded the former `atol=2e-6` cost
  gate; the maximum absolute production/opt cost delta was `1.1205673217773438e-5`. The distribution was 73 exact
  zeros, 4 in `(0, 1e-6]`, 2 in `(1e-6, 2e-6]`, 6 in `(2e-6, 5e-6]`, 10 in `(5e-6, 1e-5]`, and 1 above `1e-5`.
- **MEASURED:** the previously documented exact-discrete-equivalence contract is false and has been withdrawn.
  Cost checks now use a matrix-calibrated, non-universal `atol=1.25e-5, rtol=0`; discrete checks remain exact.
- **MEASURED by inspection:** this defect record contains no timing or speed claim.
- **RESOLVED 2026-08-25 -- follow-up, record above retained as history.** The V=2 half of this
  defect was root-caused and fixed; see the `qvq_viterbi_cpu_opt`: V=2 emission accumulation order
  entry at the end of this log. **MEASURED:** the cause was the V=2 arm of `emission_avx512`
  opening its 16-lane body with `_mm512_mul_ps(v1, t1v)`, which pre-rounds one product instead of
  accumulating from zero in coordinate order; both lanes now accumulate from zero, verified by
  objdump on cold builds. On a 24-configuration matrix on this host, discrete state mismatches
  against `qvq_cpu_viterbi` went 1 -> 0 and non-bit-identical squared error went 14 -> 2, with the
  2 residual cases both at V=4 and numerically unchanged. `test_qvq_viterbi_opt_known_v2_rate8_large_batch_divergence`
  XPASSed after the fix and its `xfail` marker was removed. **STILL OPEN from this record:** the V=4
  squared-error deltas, and the fact that discrete agreement is measured on one host and one
  compiler rather than guaranteed -- the opt kernel is still two-sweep where production is fused.
  The matrix-calibrated `atol=1.25e-5, rtol=0` cost tolerance was deliberately NOT tightened.

## 2026-08-24 segmented banked G-only Viterbi recurrence

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

- **MEASURED by inspection:** replaced three banked full-state float frontiers with two per-bank suffix-G buffers;
  fused emission, predecessor-G addition, strict prefix argmin, and backpointer generation into one suffix pass.
  Segment-boundary reduction remains bank-major with strict `<` and carries the winning bank's prefix.
- **MEASURED:** pristine `d9d37181` artifacts matched exactly for states and segment bank IDs across 17 V2/V4
  configurations spanning batches 1/8/16/32/64/128, transition bits 1/5/15/16, short/full windows, 1/2/4 banks,
  weights, overlap, and segment switches. Full-window V2 overlap artifacts also matched packed trellis words and
  packed bank selectors exactly. Maximum observed FP32 loss delta was 1.39e-6; the target case delta was zero.
- **MEASURED:** an adversarial regression covers step counts 1/2, `-1` sentinel collision, `INT64_MIN/MAX`, exact
  range boundaries, a `2**32` alias, mixed valid/invalid rows, zero segment length, a forced bank switch, fixed-exit
  traceback, and transition-15 combined bank/prefix overflow. The unchanged test failed on pristine and passes now.
- **MEASURED:** with the cgroup 99.314%/99.317% idle before accepted series, 32 explicit singleton affinities,
  10 warmups, 51 samples, and exclusive sequential runs, batch 16 / 128 steps / V2 / 65,536 states / two banks /
  transition bits 5 / segment 16 improved from 110.0 ms to 6.7 ms median: **16.49x**.
- **MEASURED:** final gates were 657 passed/248 skipped (`test_qvq.py`), 120/12 (`test_qvq_v2b2_p32.py`), and
  18 passed/1 xfailed (`test_qvq_viterbi_cpu_opt.py`). Lifecycle retained the identical pre-existing 12 failures
  (18 passed/2 skipped); the CPU-only CUDA banked/segment subset skipped all 312 selected cases.
- **MEASURED:** the explicit placement was
  `{24},{27},{28},{42},{43},{44},{45},{54},{55},{65},{90},{94},{96},{104},{113},{114},{118},{123},{135},{139},{143},{150},{156},{161},{164},{169},{172},{173},{175},{176},{179},{183}`.
  `OMP_PLACES=cores` was never used; the benchmark asserts the placement once before timing.

## 2026-08-24 segmented banked Viterbi review correction

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4) | 32 cores, OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

- **MEASURED:** corrected the V=2 AVX-512 reduction to scalar-reference order and kept non-aligned suffix chunks
  scalar so predecessor broadcasts cannot cross a boundary. The 16-vs-24-thread regression failed at `612b15e6`
  with 31 states, one segment bank ID, and seven packed words changed; it passes afterward.
- **CLARIFICATION (added by a later independent re-verification; see the wording-correction entry at the end of
  this log):** the bullet above describes a regression that was introduced by *this branch's own* fused rewrite at
  `612b15e6` and repaired by its follow-up commit. It is **not** evidence that pristine `main` diverged by thread
  count. Independently re-measured with a fresh per-commit `GPTQMODEL_QVQ_CPU_BUILD_ROOT`, the same test **passes**
  on pristine `d9d37181`, **fails** at `612b15e6`, and **passes** at `1c567c00`.
- **MEASURED:** replaced the former 16.49x headline. In the self-consistent t5 row-parallel regime, accepted
  batch-32/64/128 speedups are 1.26x/1.47x/1.23x. Batch 16 is reported separately at 15.76x as repair of the
  pristine three-barrier small-batch pathology.
- **MEASURED:** a 16-cell t15/t16, segment-16/32 sweep showed t15 G-only is 0.27-0.28x for batches 32-128; it remains
  enabled because pristine t15 overflows combined bank/prefix int16 backpointers and emits negative states. T16
  unconstrained two-bank cases retain the pristine recurrence and measure 0.98-1.09x with exact outputs and costs.
- **MEASURED:** every timing cell used 3 warmups and 15 samples and reports median, minimum, and max-minus-min spread
  in `docs/kernels/qvq_segmented_viterbi_cpu_results.md`. Pristine/post cgroup idle samples were 99.341%/99.191%. Both series used the explicit 32 singleton
  placements recorded above; runs were exclusive and `OMP_PLACES=cores` was never used.
## 2026-08-24 — METHODOLOGY CORRECTION: `uptime` load average is not a valid idle check on this host

Hardware: AMD EPYC 9V33X (Zen 4 Genoa-X, no AMX) | host `zen5-cpu-6` | 32-CPU cgroup carved from a
          192-CPU physical host | torch 2.13.0+cpu

This entry corrects a measurement practice used by several earlier entries in this log. It contains no
timing or speed claim of its own.

### The defect

**MEASURED.** Several 2026-08-24 entries state that the machine was verified quiet before timing, using
`uptime` / `/proc/loadavg`. That signal is **host-wide**, not cgroup-scoped. Sampled simultaneously:

```
/proc/loadavg                     ->  55.34 51.94 35.55   67/7476 tasks
getconf _NPROCESSORS_CONF         ->  192          (physical host)
nproc                             ->  32           (our cgroup)
/sys/fs/cgroup/cpu.stat delta     ->  0.23 CPU-seconds used of 96 possible over 3s  = 0.24%
sum of our processes' %CPU        ->  9.2%
```

A load average of 55 was observed while this container was **99.8% idle**. The number is dominated by
roughly 7,400 tasks belonging to other tenants of the same physical host.

### Correct idle check

Use a cgroup-scoped delta, not a host-global average:

```bash
A=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat); sleep 3
B=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
# busy fraction = (B-A) / (3e6 * nproc)
```

### Consequences for numbers already recorded in this log

- **INFERRED (high):** the practical effect was agents *waiting* on a signal that was never theirs. It did
  not create false confidence in a busy cgroup, so no recorded number is invalidated by this alone.
- **INFERRED (high):** this host is **multi-tenant**. Other tenants share its L3 and memory bandwidth with
  our pinned CPUs. Absolute millisecond figures in this log should therefore be read as *"on a shared
  host,"* and are not reproducible to better than tens of percent.
- **INFERRED (high):** this is the most likely explanation for the previously recorded 11–34% discrepancy
  between hosts `zen5-cpu-1` and `zen5-cpu-6` on identical work, which was originally attributed to
  per-instance CCD allocation. Neighbour load is the simpler explanation.
- **Ratios remain sound.** Every speedup recorded here compares two arms measured back-to-back within
  seconds on the same box. Neighbour noise affects both arms alike, so A/B ratios are far more robust than
  the absolute timings. The 2.39x Viterbi and 1.93x GEMV ratios are not called into question by this entry.

### Standing rule this reinforces

Validate a baseline by **self-consistency of implied throughput across problem sizes**, not by agreement
with a previously recorded absolute number. On a multi-tenant host that is the only sound test. A baseline
whose implied GFLOP/s is flat across a wide range of N is trustworthy; one that scatters is contended,
regardless of what any load average reported at the time.


## 2026-08-24 — METHODOLOGY CORRECTION 2: a stale JIT extension cache can silently answer for the wrong commit

Hardware: AMD EPYC 9V33X (Zen 4 Genoa-X, no AMX) | host `zen5-cpu-6` | 32-CPU cgroup | torch 2.13.0+cpu

This entry contains no timing or speed claim. It records a second way a measurement on this repository can be
wrong while looking correct, and it corrects one specific claim already recorded above.

### The trap

**MEASURED.** QVQ builds its CPU extensions lazily through `TorchOpsJitExtension`. The build root is
`~/.cache/gptqmodel/torch_extensions/<subdir>/<source-fingerprint>/` (`gptqmodel/utils/cpp.py:371-377`,
`:924-927`), and it is **not** controlled by `TORCH_EXTENSIONS_DIR` — the per-extension override for the CPU
Viterbi/GEMV ops is `GPTQMODEL_QVQ_CPU_BUILD_ROOT` (`gptqmodel/utils/qvq_cpu.py:69`).

Consequently, checking out an older commit and running pytest does **not** guarantee that commit's kernel was
executed. If a fingerprint collides with, or is reused from, another worktree's build, the test silently loads a
**different commit's `.so`** and reports a result about code that is not on disk.

### Required practice for any before/after or fail-first claim

- Export a **distinct, empty** `GPTQMODEL_QVQ_CPU_BUILD_ROOT` for every commit under test.
- Confirm a real compile occurred: a first-run wall time on the order of ~30 s. An instant load means a cached
  binary answered, and the result is void.
- Never compare a "before" and an "after" that shared a build root.

### The claim this corrects

**MEASURED.** The segmented-Viterbi review-correction entry above originally read as though the 16-vs-24-thread
divergence (31 states, one segment bank ID, seven packed words) demonstrated a defect in merged production code.
Re-verified here with clean per-commit build roots, running one identical test file at three commits:

| kernel commit | `..._v2_is_thread_count_invariant` |
|---|---|
| `d9d37181` (pristine `main`) | **PASSES** |
| `612b15e6` (this branch's fused rewrite, pre-fix) | **FAILS** — 31 states differ |
| `1c567c00` (after its follow-up fix) | **PASSES** |

The divergence was therefore **introduced and repaired inside one branch**, not inherited from `main`. That entry
has been annotated in place. The separately reported V=2 reduction-order finding in the *non-banked*
`qvq_viterbi_cpu.cpp` is a different file and remains open on its own evidence; its severity should be recorded as
**latent unless and until a thread sweep on that file demonstrates a differing selected state**.

### What was genuinely pre-existing

**MEASURED.** In the same three-commit re-verification, the branch's new adjacent-step/invalid-sentinel test does
not merely fail on pristine `d9d37181` — it terminates the interpreter with `Fatal Python error: Floating-point
exception` (SIGFPE), reachable from the public banked Viterbi op. The same test passes at `1c567c00`. That crash,
not the speedup, is the strongest justification for that change.

## 2026-08-24 QVQ Viterbi V=2 FP32 reduction-order defect record (not a performance entry)

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4, no AMX) | 32 logical cores,
          OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

Credit: the disagreement was identified by an independent cross-vendor review of `qvq_viterbi_cpu.cpp`.

- **MEASURED by disassembly** (gcc 15.2.0, `-O3`, default `-ffp-contract=fast`), `origin/main` kernel
  `41309aa602d31edec38af37a855ce369`, object `qvq_viterbi_cpu.o`, function `fused_g_argmin_avx512`:
  the V=2 emission dot product was computed two different ways in one file. Vector body at `0x6a2`:
  `vmulps (%r15),%zmm13` then `vfmadd231ps (%r13),%zmm12`, i.e. `RN(c0*t0 + RN(c1*t1))` -- the **c1**
  product is pre-rounded. Scalar helper `fused_candidate_scalar`, entered for V<=2 at `0x471`:
  `vxorps %xmm2` (zero seed), then `vfmadd231ss` on c0 at `0x37a` and on c1 at `0x39a`, i.e.
  `RN(c1*t1 + RN(c0*t0))` -- the **c0** product is pre-rounded. The scalar helper serves both the
  trailing `chunk mod 16` columns of every `at::parallel_for` chunk and the entire non-AVX fallback.
- **MEASURED by disassembly** after the fix (object `e223b567dff12df2`): the vector body becomes
  `vmovaps %zmm11,%zmm0` (zero seed), `vfmadd132ps (%r13),%zmm14` (c0), `vfmadd231ps (%r9),%zmm12`
  (c1) at `0x6c3`/`0x6d2`. The two memory operands swap order so c0 is consumed first, matching the
  scalar chain. V=4 already used this order and is byte-for-byte unchanged.
- **MEASURED:** which path a suffix column takes depends on `at::get_num_threads()`. `at::parallel_for`
  uses `#pragma omp parallel` with no `num_threads` clause; `omp_get_dynamic()` is 0 on this host and
  the team size equals the request exactly, including under heavy load from a second tenant. For
  `suffix_count = 512` the chunk is `divup(512, min(threads, 32))`: 16 -> 32 and 32 -> 16 leave no
  scalar remainder, 24 -> 22 sends the trailing 6 columns of every chunk through the scalar helper.
- **MEASURED severity, `origin/main` kernel.** One configuration diverges: seed 149, one row sliced
  from a 128-row draw, V=2, transition bits 7, 128 steps, batch 1, tail-biting overlap. 16 threads vs
  24 threads differ in **54 of 128 selected states** and **13 of 28 packed trellis words**, with a
  squared-error delta of 7.15256e-07; 32 threads match 16 exactly. Confirmed with one thread count per
  fresh process and bit-identical across 3 repeats at each thread count, so this is not a race.
- **MEASURED severity, independent sweep, same kernel: 0 divergences in 572 completed configurations**
  (508 direct-construction: 64 seeds x transition bits 5/6/7/8 x overlap on/off x 32 steps; 64
  sliced-construction: 64 seeds x transition bits 7 x 128 steps). 0 unstable, 0 skipped. Detection was
  proven live throughout by a positive-control canary re-run every 20-50 configurations, which
  reproduced the seed-149 divergence 16 of 16 times. The 572 swept configurations are disjoint from
  seed 149. **The defect is therefore real and OBSERVED, not latent -- but rare: one known divergent
  configuration and zero further instances in 572 swept.**
- **MEASURED after the fix:** the same 64 sliced configurations and seed 149 itself all show 0
  divergences across 16/24/32 threads.
- **MEASURED:** `step_count == 0` segfaulted the interpreter on `origin/main` (exit 139); the
  traceback indexes one element before the start of the output buffer. A `TORCH_CHECK` at the op
  boundary now rejects it with `RuntimeError`. The thread-invariance regression FAILS and the empty
  step regression SEGFAULTS on `origin/main`; both pass here.
- **MEASURED, audit of sibling kernels (reported, not edited).**
  `qvq_viterbi_cpu_opt.cpp`: at **V=2 the two paths agree**. Its vector body pre-rounds c1
  (`_mm512_mul_ps(v1, t1v)`) and its scalar head/tail `target[0]*c0[s] + target[1]*c1[s]` also
  pre-rounds c1, because gcc contracts the *first* product into the FMA and leaves the *second* as a
  `vmulss` (verified with a standalone probe and in the shipped object at `0x6b0` and `0x740`). At
  **V=4 they disagree**: the vector body is zero-seeded c0->c3 (operand order `rsi,rbx,r12,r10` at
  `0x2ca`) while the scalar expression pre-rounds c1 (`mul rbx` then `rsi,r12,r10` at `0x220`).
  `qvq_gemv_cpu.cpp`: no analogous defect -- every accumulator is `_mm512_setzero_ps()` followed by
  `_mm512_fmadd_ps` chains over fixed 16-lane tiles with no head/tail split, and `accumulate_tile_scalar`
  is selected only by `cpu_has_avx512()` for the whole run, never mixed with the vector path.
  `qvq_hadamard_cpu.cpp`: no analogous defect -- the transform is `_mm512_add_ps`/`_mm512_sub_ps`
  butterflies with no dot product and no FMA; its scalar (stride < 16) and vector (stride >= 16) paths
  cover disjoint stride ranges and each output is a single add or sub of the same two inputs.
- **INFERRED** from the two probes above, not measured across toolchains: the V=2 agreement in
  `qvq_viterbi_cpu_opt.cpp` is incidental to gcc's contraction choice rather than structural. A
  different compiler, or `-ffp-contract=off`, would break it.
- **MEASURED, performance, no regression.** 6 interleaved A/B rounds (before/after alternating within
  one series to cancel drift), 5 warmups and 21 samples per round, `OMP_NUM_THREADS=32`, the explicit
  32 singleton placement recorded below, cgroup 98.99%-99.23% idle before each series. Min-of-mins
  moved +1.3% / -0.1% / +0.7% for V2/t7/128 steps/batch 32, batch 8, and V2/t5/batch 32 respectively;
  median-of-medians moved -4.3% / +1.9% / +5.8%. The medians move in both directions across rounds and
  single-run spreads reached 60 ms against 25 ms medians, so the deltas are run-to-run noise on this
  shared host, not a regression.
- **MEASURED, gates.** Before (`origin/main` kernel): `test_qvq.py` 658 passed / 248 skipped / 2
  deselected (the two new regressions cannot run: one fails, one segfaults the session);
  `test_qvq_v2b2_p32.py` 120 passed / 12 skipped; `test_qvq_viterbi_cpu_opt.py` 18 passed / 1 xfailed.
  After: 660 passed / 248 skipped; 120 passed / 12 skipped; 18 passed / 1 xfailed.
- **MEASURED:** correcting the production reduction order shifted the production-vs-opt cost delta in
  `test_qvq_viterbi_opt_large_magnitudes` to 0.046875 absolute on a cost of about 1.54e5, i.e.
  3.04e-07 relative or roughly two float32 ulp, which exceeded that assertion's absolute-only
  `atol=1.25e-5, rtol=0`. The discrete `torch.equal` assertion in the same test still passes exactly.
  The absolute bound is retained and `rtol=2e-5` added -- the same rtol the oracle assertion a few
  lines above already uses -- so the bound stays valid as input magnitude changes.
- **MEASURED, methodology corrections for this repository.** (1) The JIT build line is
  `cxx = ccache c++`, so "confirm a real ~30 s compile" is **not** a reliable freshness signal: ccache
  returned all six objects in 533 ms for a byte-identical source tree in a freshly emptied
  `GPTQMODEL_QVQ_CPU_BUILD_ROOT`. That is correct behaviour, but wall-clock cannot distinguish it from
  a stale load. The reliable checks are the source-hash fingerprint subdirectory and reading the
  emitted instructions out of the `.o`. (2) An aborted sweep is not a clean sweep: an early
  `pack_trellis_states` failure on open (non-overlap) paths killed a 1024-configuration run at its
  first configuration and left a log with zero divergences. Sweeps here now report ATTEMPTED /
  COMPLETED / UNSTABLE / SKIPPED / DIVERGED separately and carry a positive-control canary, so a
  zero is falsifiable rather than merely empty.
- **MEASURED by inspection:** the only timing claim in this entry is the no-regression A/B above.
- Measurement placement was
  `{24},{27},{28},{42},{43},{44},{45},{54},{55},{65},{90},{94},{96},{104},{113},{114},{118},{123},{135},{139},{143},{150},{156},{161},{164},{169},{172},{173},{175},{176},{179},{183}`.
  `OMP_PLACES=cores` was never used. Idle was checked with a `/sys/fs/cgroup/cpu.stat` `usage_usec`
  delta, never `uptime` or `/proc/loadavg`.

## 2026-08-24 V=2 reduction-order severity: controlled thread-invariance sweep with an unreachable control group

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4 `znver4`, no AMX) | 32 logical
          cores available, thread counts set per call to 16/24/32 | torch 2.13.0+cpu | host zen5-cpu-6

Independent confirmation and refinement of the entry above. All numbers below were measured on the
**pristine `d2222255` kernel checked out into its own worktree**, so the shared working tree's source
state could not affect them, with `CCACHE_DISABLE=1` and a distinct empty `GPTQMODEL_QVQ_CPU_BUILD_ROOT`
(real 28-30 s compile confirmed; pristine fingerprint `60c10c9fa1dcde19`, fixed `e223b567dff12df2`).

- **MEASURED, reachability model.** `at::parallel_for` runs `#pragma omp parallel` with no `num_threads`
  clause, so `chunk = divup(suffix_count, min(threads, divup(suffix_count, grain)))` and the vector body
  covers only whole 16-column groups. For 65,536 states the trailing-scalar column count per chunk is:

  | transition bits | suffix_count | 16 threads | 24 threads | 32 threads | exposed? |
  |---|---|---|---|---|---|
  | 1 | 32768 | 0 | 6 | 0 | yes |
  | 5 | 2048 | 0 | 6 | 0 | yes |
  | 7 (**production W3.5**) | 512 | 0 | 6 | 0 | **yes** |
  | 8 (W4) | 256 | 0 | 0 | 0 | no |
  | 9 | 128 | 0 | 0 | 0 | no |
  | 15 | 2 | all | all | all | no (all-scalar at every count) |

  A config is only exposed when the vector/scalar split **differs** between thread counts. Transition
  bits 8 and 9 are 16-aligned at all three counts; transition bits 15 leaves `suffix_count < grain`, so
  every column is scalar everywhere. **Only transition bits <= 7 are exposed, which includes the
  production W3.5 rate but not W4.**

- **MEASURED, sweep with per-config `try`/`except`, four counts:**
  **ATTEMPTED 1632 / COMPLETED 1632 / SKIPPED 0 / DIVERGED 0.**
  Grid: 6 seeds x {V=2 2^16, V=4 2^16, V=2 2^12} x transition bits {1,5,7,8,9,15} x steps {32,128} x
  batch {1,4} x overlap {on,off} x step weights {on,off}, each run at 16, 24 and 32 threads and compared
  against the 16-thread result. DIVERGED counts **discrete state** divergence. Zero configs were skipped,
  so the zero rests on the full 1632, not on an aborted run.

- **MEASURED, and this is the substantive refinement: the returned squared error IS thread-dependent,
  broadly.** Counting configs whose `squared_error` differs at all between thread counts:
  **218 of the 672 remainder-reachable configs (32.4%), and 0 of the 960 unreachable configs.**
  Maximum delta 1.52587890625e-05. The separation is total: every affected config is one the reachability
  model predicts, and no unaffected config is. That is a controlled experiment with its own negative
  control, not a coincidence.

- **MEASURED, production-shaped probe** (V=2, 65,536 states, transition bits 7, 128 steps, batch 32,
  tail-biting overlap, 12 seeds): ATTEMPTED 12 / COMPLETED 12 / SKIPPED 0 / DIVERGED 0 discrete.
  But **12 of 12 seeds returned a different squared error at 24 threads** (1.907e-06 to 4.053e-06) and
  **12 of 12 returned exactly 0.0 delta at 32 threads** — precisely the pattern the table above predicts.

- **MEASURED severity, stated honestly.** The defect is **OBSERVED, not latent**. Two distinct
  observations: (a) the discrete selected path changes in the seed-149 witness — 54 of 128 states and 13
  of 28 packed words between 16 and 24 threads — reproduced here independently; (b) the returned cost is
  thread-dependent in about a third of exposed configurations and in 12 of 12 production-shaped ones.
  **Discrete path flips are rare** (0 further instances in 1632 completed configs); **numeric
  thread-dependence is not rare**. Both vanish after the fix.

- **MEASURED:** `pack_trellis_states` was exercised on 96 configurations during the sweep with **0
  failures** (`pack_checked 96, pack_skipped 0`). The `ValueError: QVQ states must form one
  transition-consistent tail-biting path` seen in an earlier aborted sweep was a bad generated config in
  that sweep driver, **not** a kernel defect.

- **MEASURED, `vmulps` census of `fused_g_argmin_avx512`.** Pristine has 6 `vmulps`: three are the
  `weight != 1.0f` scaling clones, one is the auto-vectorised scalar dot loop used at V>=4, and **two are
  the V=2 emission reduction** (`0x1f9` and `0x6a2`). The fixed object has 4 — the same four non-reduction
  sites, at `0x1a6`, `0x352`, `0x5db`, `0x6f0`. **The fix removes exactly the two V=2 reduction sites and
  changes nothing else in the function.**

- **MEASURED, audit correction.** The entry above is right that `qvq_viterbi_cpu_opt.cpp` disagrees with
  itself at V=4; confirmed here independently. Scalar head/tail at `0x220`:
  `vmulss (%rbx),%xmm6` (pre-rounds `c1*t1`) then `vfmadd231ss` on `%rsi`/`%r12`/`%r10`. Vector body at
  `0x2ca`: `vfmadd132ps (%rsi),%zmm2` from the zero seed, then `%rbx`/`%r12`/`%r10`. Scalar order is
  `c1,c0,c2,c3` with `c1` pre-rounded; vector order is `c0,c1,c2,c3` from zero. Register identity from
  the prologue: `%rsi`=`c0`, `%rbx`=`c1`, `%xmm8`=`t0`, `%xmm6`=`t1`. Reported, not fixed — that kernel
  is opt-in/benchmark-only and already carries a discrete-divergence defect record above.

- **MEASURED, gates at the branch tip.** Reported in the PR; pristine baseline for `test_qvq.py` with the
  two new tests excluded was 658 passed / 248 skipped / 2 deselected, so the fix adds two tests and
  changes no pre-existing outcome.

- **MEASURED by inspection:** this entry contains no timing or speed claim.

## 2026-08-25 legacy banked transition-15 backpointer-width correction

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4 `znver4`, no AMX) | 32-CPU
          cgroup out of 192 host CPUs, OMP_NUM_THREADS=32 | torch 2.13.0+cpu | host zen5-cpu-6

- **INFERRED by whole-repository caller audit:** severity is **LATENT, NOT OBSERVED**. The legacy function is not
  separately registered as a Torch op and has no Python, test, or benchmark caller. Its sole caller is the public
  dispatcher, which selects it only for unconstrained transition-16 configurations; that path already allocates
  int32 backpointers. No reachable public configuration exposed the defective transition-15 narrowing.
- **MEASURED with a temporary t15 dispatch hook:** the pre-fix legacy path returned state `-1` instead of `65535`
  when its winning two-bank boundary predecessor flattened to `65535`. A fresh 16-second cached-compiler build
  failed the focused test; after changing the predicate to cover `bank_count * prefix_count - 1`, a distinct fresh
  build passed. The temporary hook and test are not shipped because unchanged production dispatch cannot reach
  legacy at t15; the same test would pass before and after through G-only and would therefore be a false regression.
- **INFERRED by source audit:** non-boundary prefix backpointers store only `0..prefix_count-1`. The shared width
  predicate is conservative for that array and exact for the flattened boundary array, so separate dtypes are not
  required for correctness.
- **MEASURED eager-oracle gate:** G-only and fixed legacy both matched 8 targeted configurations covering one/two
  banks, V=2/V=4, steps 1/2/4/32, segment boundaries, weights, exact ties, states, bank IDs, and packed words/selectors.
  The larger benchmark then exposed discrete differences. On the exact batch-64 seed, legacy differed on rows
  4/7/30/43 and failed the eager state oracle while G-only matched all four. Legacy also failed the batch-128 eager
  adjudication. Therefore transition-15 dispatch remains G-only.
- **MEASURED but rejected for dispatch:** with cgroup preflight deltas of 1.56 and 1.26 CPU-seconds over 3 seconds,
  explicit singleton placement, 3 warmups, and 15 samples, legacy/G-only speedups at batch 32/64/128 were
  3.87x/3.99x/3.92x. Legacy medians were 23.9/46.2/94.4 ms (mins 22.9/45.6/90.7; spreads 2.8/44.2/25.1 ms), versus
  G-only medians 92.6/184.4/369.6 ms (mins 92.5/184.3/368.4; spreads 3.7/2.9/6.2 ms). The speedup is not enabled
  because selected states fail the eager oracle. Both binaries came from distinct empty build roots with ccache
  disabled and real 25-second compiles (28.7/29.5 seconds wall time). `OMP_PLACES=cores` was never used.
- **MEASURED gates, before and after:** `test_qvq.py` stayed at 661 passed/248 skipped;
  `test_qvq_v2b2_p32.py` plus `test_qvq_viterbi_cpu_opt.py` stayed at 138 passed/12 skipped/1 xfailed; lifecycle stayed
  at the same two pre-existing `damp_percent` failures with 29 passed/2 skipped; `git diff --check` passed.
- **MEASURED after merging current `origin/main`:** `test_qvq.py` remained 661 passed/248 skipped; the combined
  V2B2/opt gate became 141 passed/12 skipped because upstream added opt tests and removed the xfail; lifecycle still
  had the identical two pre-existing failures with 29 passed/2 skipped; `git diff --check` passed.
## 2026-08-24 -- `qvq_viterbi_cpu_opt`: V=2 emission accumulation order + zero-step guard

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/CD/IFMA/VBMI (Zen 4,
          no AMX; hostname says `zen5-cpu-6` but the silicon is `znver4`) |
          32 logical cores visible to the cgroup, torch default thread pool |
          torch 2.13.0+cpu | gcc 15.2.0 | host zen5-cpu-6

Base commit: `ecf7081ee08a2beed93597b9508d98ab5b000025`.
This entry contains no timing or speedup claim; the only durations quoted are
compile wall-times recorded to prove the builds were cold.

### What changed

`gptqmodel_ext/qvq/qvq_viterbi_cpu_opt.cpp` only:

1. `emission_avx512`, V=2 arm: the 16-lane body and the scalar prologue/tail both
   now accumulate from zero in coordinate order (`c0` then `c1`). The body
   previously opened with `_mm512_mul_ps(v1, t1v)`, which pre-rounds one product
   and leaves the corresponding choice in the scalar lanes to the compiler's
   contraction of `target[0]*c0[s] + target[1]*c1[s]`. The V=4 arm is untouched.
2. `TORCH_CHECK(step_count > 0, "qvq_viterbi_cpu_opt: step_count must be positive")`
   at the op boundary, matching the guard merged into `qvq_viterbi_cpu.cpp:238`.

### Disassembly, cold builds, `emission_avx512` V=2 arm

Distinct initially-empty `GPTQMODEL_QVQ_CPU_BUILD_ROOT` per source tree,
`CCACHE_DISABLE=1`; the compiler reported 25 s in every case, so no cached `.so`
answered. Before: fingerprint `e5e4b2c3cc01eb89` (29.557 s wall / 104.970 cgroup
CPU-seconds for build plus a one-call smoke script). After: fingerprint
`26360bb60ffda61f` (40.647 s wall / 289.220 cgroup CPU-seconds for build plus the
full gate-2 suite). Two further cold roots were used along the way: an
instrumented probe build, and `dc333810a37215fa` for the pre-comment revision of
this same fix, whose V=2 schedule is identical to the final one.

Register mapping, established from the function prologue and stable across the
whole body: `rsi = codebook_t = c0`, `rbx = codebook_t + state_count = c1`,
`xmm8/zmm5 = target[0]`, `xmm6/zmm4 = target[1]`.

```text
BEFORE -- V=2 scalar prologue/tail            BEFORE -- V=2 16-lane body
740: vmulss      xmm2,xmm6,[rbx+rax*4]        804: vmulps      zmm0,zmm4,[rbx+rax*4]
     # round(t1*c1)                                # round(t1*c1)
74a: vfmadd231ss xmm2,xmm8,[rsi+rax*4]        812: vfmadd231ps zmm0,zmm5,[rsi+rax*4]
     # += t0*c0, fused                             # += t0*c0, fused

AFTER -- V=2 scalar prologue/tail             AFTER -- V=2 16-lane body
73e: vxorps      xmm9,xmm9,xmm9               7ba: vxorps      xmm2,xmm2,xmm2
74c: vfmadd132ss xmm2,xmm9,[rsi+rax*4]        7e2: vfmadd132ps zmm0,zmm2,[rsi+rax*4]
     # t0*c0 + 0                                   # t0*c0 + 0
757: vfmadd231ss xmm2,xmm6,[rbx+rax*4]        7f0: vfmadd231ps zmm0,zmm4,[rbx+rax*4]
     # += t1*c1, fused                             # += t1*c1, fused
```

Both lanes now emit the identical ordered accumulation from zero, and the
schedule matches `fused_g_argmin_avx512` / `fused_candidate_scalar` in
`qvq_viterbi_cpu.cpp`. The V=4 arm was verified untouched by extracting the 125
floating-point instructions in `[0x5b, 0x608)` from both objects and diffing them:
identical. Outside that range only branch displacements move, because the V=2 arm
got 16 bytes shorter.

### Correction to the incoming cross-review (MEASURED)

The review that prompted this work read offsets `0x740`/`0x804` as evidence that
the scalar lane pre-rounded coordinate 0 while the vector lane pre-rounded
coordinate 1. That attribution is wrong: `rbx` is a single register holding `c1`
throughout the function, so **both** pre-fix V=2 lanes pre-rounded the *same*
product, `t1*c1`. The vector mapping is confirmed independently by source
correspondence -- `_mm512_mul_ps(v1, t1v)` is exactly `vmulps zmm0,zmm4,[rbx]`.

Applying the same corrected mapping to the V=4 arm inverts the review's other
conclusion: pre-fix V=4 *does* pre-round opposite products (vector rounds
`t0*c0` into zero at `0x2ca`; scalar rounds `t1*c1` at `0x220`). V=4 was declared
out of scope for this change and is left untouched -- see the reachability note
below for why it is not urgent, and treat it as an open item for the owner.

### Reachability of the scalar prologue/tail (MEASURED, instrumented build)

A throwaway instrumented build (atomic counters on all four scalar loops in
`emission_avx512`, reported from a static destructor) recorded:

```text
PROBE v2_prologue=0 v2_tail=0 v4_prologue=0 v4_tail=0
```

after (a) the entire `tests/test_qvq_viterbi_cpu_opt.py` suite and (b) the
16/24/32-thread `state_count=65536, transition_bits=7` config, V=2 and V=4.

Root cause: `_opt` partitions sweep B as `chunks_per_batch = (state_count+4095)/4096`
with `chunk_states = ceil(state_count / chunks_per_batch)`. That is a function of
`state_count` alone -- not of the thread count -- and the op requires `state_count`
to be a power of two >= 16, so every `[begin, end)` is 16-aligned with a
16-multiple length. The scalar prologue/tail is therefore dead code for every
input the op accepts. This is why the thread-invariance test below cannot fail
pre-fix, and why the residual V=4 asymmetry is not currently reachable either.

### Discrete impact (MEASURED, 24 configs)

`qvq_cpu_viterbi_opt` vs `qvq_cpu_viterbi`, exact state comparison, V in {2,4} x
`state_count` in {1024, 4096, 65536} x `transition_bits` in {5,7,8,12,16} x
(batch,steps) in {(8,16), (128,32)}, 16 threads:

```text
before: 24 configs, 1 with a discrete mismatch
        (V=2, state_count=65536, transition_bits=16, batch=128, steps=32
         -> 2 of 4096 state entries differ)
after:  24 configs, 0 with a discrete mismatch
```

Squared error over the same 24 configurations, exact `torch.equal`:

```text
before: 14 of 24 configs not bit-identical; worst relative delta 1.450e-3
        (V=2, state_count=65536, transition_bits=16, batch=128, steps=32).
        All 12 V=2 configs were non-bit-identical.
after:   2 of 24 configs not bit-identical; worst relative delta 7.373e-6.
        Both are V=4, transition_bits=16, and their values are numerically
        UNCHANGED by the fix (2.822e-6 and 7.373e-6 before and after).
        All 12 V=2 configs are now bit-identical.
```

So this is **not** a latent-only change: it removes a measured discrete
divergence and a measured squared-error divergence across every V=2
configuration tested. All 24 V=4 configs are identical before and after on
states, and the two V=4 squared-error deltas are bit-for-bit the same before and
after -- a second, numeric confirmation that the V=4 arm was not disturbed. The
residual V=4 deltas are attributed (INFERRED) to the structural two-sweep versus
fused difference, not to the V=4 rounding asymmetry, which the probe showed is
unreachable dead code.

The pre-existing `xfail` on
`test_qvq_viterbi_opt_known_v2_rate8_large_batch_divergence` covers exactly the
one config that diverged. After the fix it XPASSes, so **the marker was removed**
-- deliberately and recorded here, not silently -- and the test now asserts
agreement. Caveat: this is one host and one compiler. `_opt` still differs from
the production kernel structurally (two-sweep emission-then-add vs fused), so the
owner should confirm the agreement on their CI before relying on it.

**Follow-up completed in this same PR (2026-08-25):** the `qvq_cpu_viterbi_opt`
docstring in `gptqmodel/utils/qvq_cpu.py` had continued to advertise the
now-fixed divergence and has been rewritten to state what is now true, keeping
its MEASURED/INFERRED labels and scoping every number to this host and this
24-configuration matrix. A dated `RESOLVED` follow-up was appended to the
2026-08-24 defect record earlier in this log; that record itself was retained.
Verified by grep that no other in-repo text still advertises the divergence: the
removed `xfail` reason string is gone, and `qvq_viterbi_cpu_opt.cpp` carries no
such comment. Two historical entries still describe it in the past tense and
were deliberately left alone as history -- in particular the V=4 audit-correction
entry, which is under third-vendor adjudication.

### Zero-step guard (MEASURED)

Pre-fix, the documented reproducer reached
`states_ptr[b * step_count + step_count - 1] = end_state;` with `step_count == 0`
and wrote at index -1:

```text
$ PYTHONPATH=. python -c "...qvq_cpu_viterbi_opt(torch.empty((1,0,2)), torch.zeros((16,2)), 2)"
Segmentation fault; exit status 139
```

The new `test_qvq_viterbi_opt_rejects_empty_steps`, run pre-fix in a subprocess,
takes the interpreter down with it -- `pytest` itself exits **139**. Post-fix the
op raises `RuntimeError: qvq_viterbi_cpu_opt: step_count must be positive` and the
test passes. The banked/segmented operator already guards `step_count > 0` at
`qvq_viterbi_banked_cpu.cpp:260` and `:697-699`; it was confirmed and not touched.

### Gates (MEASURED, before and after, same host)

```text
                                          before                    after
tests/test_qvq.py                         661 passed, 248 skipped   661 passed, 248 skipped
tests/test_qvq_v2b2_p32.py +              138 passed, 12 skipped,   141 passed, 12 skipped
  tests/test_qvq_viterbi_cpu_opt.py         1 xfailed
tests/test_qvq_lifecycle.py               2 failed, 29 passed,      2 failed, 29 passed,
                                            2 skipped                 2 skipped
```

Two baseline deviations from the brief this work was handed, both measured:

- `test_qvq.py` is 661 passed on `ecf7081e`, not the 660 the brief quotes; `main`
  moved (PR #35) between the brief and this branch.
- `test_qvq_lifecycle.py` has **2 pre-existing failures on `ecf7081e`**, not
  `30 passed, 2 skipped`:
  `test_qvq_yaqa_lifecycle_collects_full_model_factors_and_wires_them_to_quantizer[1]`
  and `[1.5]`, both `assert 0.1 == 0.0001` on `damp_percent`. Identical before and
  after this change, unrelated to the Viterbi kernels, and presumably owned by the
  in-flight `fix/qvq-explicit-rounding-rate-damping` workstream.

Gate 2 gains 3: the two new tests, plus the de-xfailed divergence test.

## 2026-08-25 CORRECTION -- the "eager oracle" is not an oracle for near-tie path selection

Hardware: AMD EPYC 9V33X 96-Core Processor | AVX-512F/BW/VL/DQ/FMA (Zen 4 `znver4`, no AMX) | 32-CPU
          cgroup out of 192 host CPUs | torch 2.13.0+cpu | host zen5-cpu-6

### What this corrects

The 2026-08-25 legacy transition-15 entry above states that legacy "differed on rows 4/7/30/43 and
**failed the eager state oracle**", and that transition-15 dispatch stays on G-only "because selected
states fail the eager oracle". Similar oracle-conformance language appears elsewhere in this log and
in `gptqmodel/utils/qvq_cpu.py`.

**That framing is wrong, and this entry corrects it.** Disagreeing with the eager reference is not
the same as being incorrect. The eager reference accumulates in FP32 like the kernels do, so on
near-ties it is *also* capable of selecting the worse path -- and measurably does.

### MEASURED: FP64 adjudication of the disagreements

Method: for each disagreeing row, both complete candidate state paths were extracted and their
**total squared error recomputed in float64** from the original sequences and codebooks, independent
of any FP32 kernel accumulation. Both paths were additionally checked for validity; **every legacy,
eager, and G-only path packed successfully**, proving transition consistency and tail-biting closure.
No candidate was illegal. These are genuine alternative optima-adjacent paths, not corruption.

Transition width 16 (the reachable production dispatch predicate: `bank_count <= 2`, unconstrained):

| seed; batch; banks; steps; segment | row | legacy FP64 | eager FP64 | genuine winner | rel. diff |
|---|---:|---:|---:|---|---:|
| 20260821; 16; 1; 32; 8   | 13 | 0.03372126385533342   | 0.03372192710214201   | **legacy** | 1.97e-05 |
| 20260821; 16; 1; 128; 8  |  3 | 0.06033789687043987   | 0.06033856011724845   | **legacy** | 1.10e-05 |
| 20260822; 128; 2; 32; 8  |  9 | 0.003980354006866752  | 0.003980362759275328  | **legacy** | 2.20e-06 |
| 20260821; 16; 2; 32; 8   | 10 | 0.0025584008785275196 | 0.0025579453464234400 | eager      | 1.78e-04 |
| 20260821; 64; 2; 128; 32 | 43 | 0.036088594007618865  | 0.036087340711329670  | eager      | 3.47e-05 |
| 20260822; 32; 1; 128; 8  | 12 | 0.013112702866913933  | 0.013112697300806359  | eager      | 4.25e-07 |
| 20260822; 128; 2; 32; 8  | 60 | 0.0025891787639937235 | 0.0025882629459672533 | eager      | 3.54e-04 |
| 20260821; 128; 2; 128; 8 | 24 | 0.028722643170709935  | 0.028722286093767698  | eager      | 1.24e-05 |

**Legacy is genuinely better in 3 of 8; the eager/G-only path is genuinely better in 5 of 8.**
Row 60 also covers differing segment selectors: legacy `[1,0,0,0]`, eager `[1,0,0,1]`.

Transition width 15, re-adjudicating the rows cited in the entry above:

| batch | row | legacy FP64 | eager FP64 | genuine winner |
|---:|---:|---:|---:|---|
| 64  |  4 | 0.018935450916467410 | 0.018935582084779842 | **legacy** |
| 64  |  7 | 0.028982814541042044 | 0.028982827673776943 | **legacy** |
| 64  | 30 | 0.024162003140010840 | 0.024161946689342035 | eager |
| 64  | 43 | 0.042650861021430950 | 0.042650754240254830 | eager |
| 128 | 55 | 0.024012466221767410 | 0.024012505978935634 | **legacy** |
| 128 | 92 | 0.019251765456184587 | 0.019251679434209266 | eager |

**The earlier blanket claim does not stand.** On rows 4, 7 and 55 the eager/G-only path was the
suboptimal one. Rows 30, 43 and 92 are genuine legacy losses, so the *decision* to keep transition-15
dispatch on G-only remains justified -- but the stated *reason* ("legacy fails the oracle") was not
accurate, and the 3.87x-3.99x legacy speedup recorded above was rejected on a partly incorrect basis.

### INFERRED: what is actually true

- **Neither FP32 implementation is a universal oracle.** Both legacy and the eager/G-only recurrence
  select suboptimal near-tie paths, in different configurations, purely as a consequence of FP32
  accumulation order.
- **Severity of the transition-16 finding is MODERATE production quality, not corruption.** All
  produced paths are valid and packable. The defect is that the production path is measurably the
  worse choice more often than not (5 of 8 sampled disagreements) at `transition_bits == 16`,
  `bank_count <= 2`, unconstrained. Relative cost differences ranged 4.25e-07 to 3.54e-04.
- Choosing between these paths on quality grounds is a **product decision**, and it trades against
  speed: G-only is slower at transition width 16, which is why legacy is dispatched there at all.
  Eliminating the class -- rather than picking a winner -- would require higher-precision
  accumulation in the recurrence, which is a design change, not a kernel tweak.

### METHODOLOGY RULE (adopt this)

**"Matches the eager reference" is not a correctness proof, and "differs from the eager reference"
is not a defect report.** For discrete Viterbi outputs, disagreement between two FP32
implementations at a near tie must be adjudicated by:

1. recomputing **both** candidate paths' total cost in **float64** from the original inputs, and
2. verifying **both** candidates are valid (transition-consistent, tail-biting-closed).

Only then does "wrong" mean anything. An exact FP64 tie falls back to the documented lowest-index
rule. This log previously recorded two separate defect claims that this procedure partially
overturns; both were derived from oracle-conformance alone.

Corollary: an eager reference is a useful *cross-check*, and its agreement with a kernel is
meaningful evidence about that kernel's schedule. It is not a ground truth for which of two
near-tied discrete paths is correct.
