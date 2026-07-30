---
name: gptqmodel-cpu-packing
description: Optimize CPU-side tensor packing, dtype conversion, and thread-parallel kernels in GPT-QModel. Use when profiling or modifying pack_block_cpu, pack_awq_cpu, pack_qqq_cpu, or any host-side tensor transform that feeds a quantized linear backend.
---

# CPU tensor/packing optimization for GPT-QModel

The CPU `pack` path converts dense float weight/scales/zeros into bit-packed `qweight`/`qzeros` tensors used by GPTQ, AWQ, and QQQ backends. This document captures the lessons learned from optimizing `pack_block_cpu`, `pack_awq_cpu`, and `pack_qqq_cpu` while keeping 100% bit-for-bit parity with the native Python reference.

## 1. The biggest cost is usually dtype conversion, not the bit-packing loop

`pack_block_cpu` must convert the incoming weight, scales, zeros, and `g_idx` to the internal dtype it uses for math (`float32` and `int32`). On PyTorch CPU these conversions are multi-threaded by default inside ATen.

Key observations:

- For large weights, multi-threaded `aten::to` is dramatically faster than single-threaded conversion. Forcing `at::set_num_threads(1)` around the conversions can cost more than the entire packing kernel.
- Conversions should happen once, on contiguous tensors: `weight.contiguous().to(at::kFloat)`.
- Avoid guarding the whole pack region with a global thread-limit that includes ATen dispatch; it also serializes those conversions.
- The actual bit-packing loops are small, highly cache-sensitive, and benefit most from SIMD (AVX-512/AVX2) and good memory layout, not from saturating cores.

## 2. Prefer `at::parallel_for` over raw `std::thread` for CPU packing

We experimented with a custom `std::thread` pool (`parallel_for_n`) to bypass ATen's thread management. The result was slower on the target shapes and had the following downsides:

- Per-call thread spawn/teardown overhead dominates for the small workloads common in model layers.
- Raw `std::thread` loses ATen's exception propagation and can terminate the process on unhandled errors.
- `at::parallel_for` is backed by the existing OpenMP/TBB thread pool, so it reuses workers and composes correctly with `torch.get_num_threads()`.
- `at::parallel_for` accepts a `grain` parameter that controls task granularity; use it to limit the number of tasks without serializing the dispatch.

Example pattern used in `pack_block_cpu`:

```cpp
const int64_t threads_eff = clamped_threads(threads, in_features, out_features);
const int64_t grain = std::max<int64_t>(1, total_work / threads_eff);
at::parallel_for(0, total_work, grain, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        pack_one_row_or_block(i);
    }
});
```

## 3. Fuzzy/auto thread selection by workload size

The `pack_threads` config defaults to `None`/`<=0` (auto). The C++ helper `clamped_threads` selects a thread count that scales with matrix size instead of always using `torch.get_num_threads()`:

- Target roughly one thread per 64K output elements.
- Cap the result by `at::get_num_threads()` and a hard limit (32) to avoid massive task counts on tiny layers.
- Keep the user override path: if `pack_threads` is positive, respect it (clamped to available threads and the hard limit).

```cpp
inline int64_t clamped_threads(int64_t requested, int64_t rows, int64_t cols) {
    const int64_t hard_limit = 32;
    const int64_t available = at::get_num_threads();
    if (requested > 0) {
        return std::max<int64_t>(1, std::min<int64_t>(requested, std::min<int64_t>(available, hard_limit)));
    }
    const int64_t n = rows * cols;
    const int64_t per_thread = 65536;
    int64_t t = (n + per_thread - 1) / per_thread;
    if (t < 1) {
        t = 1;
    }
    return std::min<int64_t>(t, std::min<int64_t>(available, hard_limit));
}
```

Why 64K? Empirical scaling tables on an 8-core Xeon (AVX-512) showed that tiny matrices (< 256K elements) do not benefit from all 8 threads, while large matrices continue to scale up to 8+ threads. One thread per ~64K elements keeps OpenMP overhead low and avoids regressions on small layers.

## 4. Interaction with `threadpoolctl` and `at::get_num_threads`

- `torch.get_num_threads()` returns the number of OpenMP threads PyTorch is allowed to use. In CI/headless environments this is typically 8 unless `threadpoolctl`/`OMP_NUM_THREADS` limits it.
- `at::get_num_threads()` is the C++ equivalent and should be used inside kernels.
- `threadpoolctl` can limit BLAS/OpenMP threads at runtime. If you set it to 1, `at::get_num_threads()` may still report 8 because PyTorch's own thread pool is separate from BLAS. Do not assume a `threadpoolctl` limit will shrink `at::get_num_threads()`.
- If you need a reproducible single-thread benchmark, use `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`, not just `threadpoolctl`.

## 5. Allocator and `PYTORCH_ALLOC_CONF` effects

PyTorch's CPU caching allocator behavior affects wall-clock time for small, repeated `pack` calls:

- `PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5` was observed to stabilize peak memory in the test harness, but it is not a correctness requirement.
- The CPU cache is process-wide; first calls are slower because the allocator must request pages. Warm-up runs are essential before reporting latency.
- Reusing a persistent `TorchLinear`/`QLinear` object avoids repeated Python-level overhead and gives more stable timing than creating a new module per call.

## 6. SIMD paths must have scalar fallbacks and explicit gating

`pack_block_cpu` uses AVX-512 and AVX2 when safe, but the dispatch must:

- Gate AVX-512/AVX2 with `cpu_supports_avx512()` / `cpu_supports_avx2()` at runtime, not compile-time `#ifdef` only.
- Keep scalar fallbacks for non-x86 builds, odd shapes, or unsupported bit widths.
- Avoid `__builtin_cpu_supports` directly in JIT-loaded shared libraries until `__builtin_cpu_init()` has run; call it from a constructor function.

```cpp
#if PACK_BLOCK_CPU_X86
__attribute__((constructor))
static void pack_block_cpu_init_cpu_features() {
    __builtin_cpu_init();
}
#endif
```

## 7. How to benchmark CPU packing and interpret scaling tables

Use focused scripts, not the full test suite, and always warm up:

```python
import time
import torch
from gptqmodel.nn_modules.qlinear import TorchLinear

for _ in range(warmup):
    qlinear.pack_block(linear, scales, zeros, g_idx=g_idx, workers=workers)

t0 = time.perf_counter()
for _ in range(repeats):
    qlinear.pack_block(linear, scales, zeros, g_idx=g_idx, workers=workers)
    torch.cpu.synchronize()  # important for accurate CPU timing
ms_per_call = (time.perf_counter() - t0) / repeats * 1e3
```

When sweeping thread counts, record:

- `bits`, `group_size`, `in_features`, `out_features`, `desc_act`
- `workers`/`pack_threads` value (1, 2, 4, 8, ...)
- `ms/call` and, if possible, a comparison against `pack_original` (Python reference)

A healthy result:

- Extension `pack` is 8-16x faster than the Python reference on representative shapes.
- Speedup vs 1 thread increases monotonically up to the matrix-size sweet spot, then plateaus.
- Small matrices (< 256K elements) may show no gain or a slight slowdown beyond 2-4 threads; this is expected and is why fuzzy auto selection limits thread count by workload.

## 8. Preserving 100% bit-for-bit parity

Performance work must not change the packed output:

- Compare `qweight` and `qzeros` against the Python reference for every combination of `bits` in `{2,3,4,8}`, multiple `group_size` values, `desc_act=True/False`, and `sym=True/False`.
- Use `torch.testing.assert_close(..., atol=0, rtol=0)` on the integer tensors; any non-zero diff is a regression.
- Focused tests: `tests/test_pack.py`, `tests/test_packing_matrix.py`, `tests/test_awq_cpu_packing.py`, `tests/test_packing_speed.py`.
- Pre-existing CUDA-only failures (e.g., `AwqTrilinLinear` 3-bit, `AwqGEMMLinear` without CUDA) must remain unchanged.

## 9. Common pitfalls

- **Forcing single-threaded `aten::to`**: wrapping conversions in `NumThreadsGuard(1)` is a large regression for large weights.
- **Too many tiny OpenMP tasks**: `at::parallel_for(0, N, 0, ...)` with a very small `N` and `threads=8` creates more tasks than needed. Compute `grain = max(1, N / threads)`.
- **Ignoring `at::get_num_threads()` return value**: always clamp user-provided `threads` against `at::get_num_threads()`; do not spawn more tasks than the runtime can actually run.
- **Hard-coding thread count for all shapes**: one-size-fits-all thread counts hurt tiny layers. Use the fuzzy `clamped_threads` helper.
- **Changing the C++ thread implementation without re-running `test_pack.py`**: bit-level output must be verified after every AVX or loop-order change.

## 10. Build flags for CPU packing

The JIT extension uses `-O3` and, on Linux, `-fopenmp` so `at::parallel_for` can use multiple cores. No custom `std::thread` linkage is required when using `at::parallel_for`.

```python
_pack_block_extra_cflags = ["-O3", f"-std={_jit_cxx_standard()}"]
_pack_block_extra_ldflags = []
if platform.system() == "Linux":
    _pack_block_extra_cflags.append("-fopenmp")
    _pack_block_extra_ldflags.append("-fopenmp")
```

## 11. References

- `gptqmodel_ext/pack_block_cpu.cpp`: C++ JIT extension source.
- `gptqmodel/nn_modules/qlinear/pack_block_ext.py`: Python wrapper that loads the JIT extension and calls `torch.ops.gptqmodel.pack_block_cpu`.
- `gptqmodel/quantization/config.py`: `BaseQuantizeConfig.pack_threads` field definition.
- `gptqmodel/utils/model.py`: where `pack_threads` from the quant config is forwarded to `module.pack()` / `module.pack_block()`.
- `tests/test_packing_speed.py`, `tests/test_pack.py`, `tests/test_packing_matrix.py`, `tests/test_awq_cpu_packing.py`: correctness and speed tests.
