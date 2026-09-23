---
name: gptqmodel-contiguous-memory
description: Diagnose and prevent silent performance or correctness regressions caused by non-contiguous tensor layouts in GPT-QModel quantization and kernel paths. Use when GPU/CPU kernels fallback to slow eager code, produce wrong strided reads, or when moving tensors between host, disk, and device.
---

# Tensor contiguity in GPT-QModel

Non-contiguous tensors are a common cause of slowdowns and correctness bugs when a kernel's declared stride contract does not match the actual view. But non-contiguous is not synonymous with uncoalesced, and contiguous is not synonymous with fast. A strided view can have a contiguous hot dimension, while materializing a large `.contiguous()` copy can dominate the operation. Many GPU Triton/CUDA kernels and some CPU fast paths do intentionally require a dense layout; when that contract is violated, the kernel may either:

- silently take a slow reference fallback, or
- read the wrong memory locations and return incorrect scale/zero/weight values.

Always verify contiguity on the boundary between a high-level quantizer and a low-level kernel, not inside the kernel after launch.

## Quick checks

1. Log or assert `x.is_contiguous()` before passing a tensor to a Triton/CUDA kernel or a C++/BLAS routine.
2. If the kernel requires contiguous storage, either reject the view clearly or materialize it at the narrowest stable boundary. Measure the materialization as part of the complete operator; do not assume its cost is small. If the kernel accepts strides, preserve the view when the hot lane mapping remains coalesced.
3. Preserve the original values; `contiguous()` is a layout-only copy and does not change the numeric content.
4. When adding telemetry, record shape, strides, storage offset, pointer alignment, whether a materialization occurred, bytes copied, and which fast/fallback path ran. For CUDA, inspect lane address compactness/sectors rather than using `is_contiguous()` as a coalescing proxy.

## Known regressions and fixes

### GPTQ activation scale search (`Quantizer.find_params_batched`)

- Location: `gptqmodel/quantization/quantizer.py`, `gptqmodel/quantization/_scale_search_triton.py`
- Symptom: `mlp.experts.*.down_proj` quantization with `group_size=32` was spending ~1s per module in scale search instead of ~0.01s.
- Root cause: `gptq.py` passes reshaped column slices that are non-contiguous. The Triton scale-search kernels load `[rows, num_groups, group_size]` with pointer+stride arithmetic and silently skipped because `x.is_contiguous()` was false.
- Fix: copy `x` to contiguous in `Quantizer.find_params_batched` before the Triton path. The non-batched `find_params` has no Triton fast path and does not need the copy.
- Verification: `tests/test_quantizer_scale_search.py` compares Triton and eager outputs; scale/zero max diff must be `0.0`.

### LazyTurtle grouped checkpoint loading

- Location: `gptqmodel/utils/structure.py` (`LazyTurtle.materialize_submodule`)
- Symptom: loading a group of MoE expert tensors one-by-one was dominated by per-tensor progress-bar overhead (LogBar) before the parallel grouped loader was added.
- Related contiguity note: after `safe_open`/`get_tensor`, call `tensor.detach().contiguous()` before `copy_` when the destination expects a dense layout. This also helps batched host-to-device copies stay aligned.

### Module loading telemetry (`shell_module_materialize`)

- Location: `gptqmodel/models/base.py`
- Performance note: building the `module_load` telemetry label must not trigger an `O(total_modules)` `named_modules()` scan on every per-module materialize. Pass `module_path` from the caller (`named_module.full_name`) instead of scanning.

## Code comments to preserve

When a kernel requires contiguous memory, add a comment at both the caller and the kernel wrapper:

```python
# The Triton activation/hessian scale-search kernels load x with pointer+stride
# arithmetic that assumes a contiguous [rows, num_groups, group_size] layout.
# gptq.py passes reshaped column slices that are not contiguous, so copy the
# local batch to unlock the fast GPU path without changing quantized scale/zero values.
if isinstance(x, torch.Tensor) and not x.is_contiguous():
    x = x.contiguous()
```

In the Triton module docstring:

```
Note: these kernels assume the [rows, num_groups, group_size] input is contiguous.
Callers must ensure x.is_contiguous() before launch; otherwise pointer+stride
arithmetic reads strided columns incorrectly.
```

## Tests and benchmarks

- Add a unit test that passes a deliberately non-contiguous tensor to the kernel/wrapper and asserts the output matches the contiguous reference bit-exactly.
- Add a timed benchmark comparing contiguous vs non-contiguous inputs for the target shape/dtype/group size. Report fallback vs fast-path latency, not just end-to-end time.

## See also

- `$gptqmodel-cuda-kernels` for Triton/CUDA/C++ kernel implementation and debugging.
- `$gptqmodel-quantization` for GPTQ/AWQ/QQQ algorithm details and scale-search behavior.
- `$gptqmodel-gpu-profiling` for identifying launch gaps and fallback hotspots.


## CUDA A100+ layout audit

Before adding a `.contiguous()` to a hot inference path:

1. Write the actual lane address expression for the kernel's dominant load/store.
2. Check natural alignment for any `half2`, `uint4`, vectorized Triton block,
   TMA, or library operand. A contiguous tensor with a nonzero storage offset can
   still violate a vector alignment assumption.
3. Determine whether the stride is on a cold outer dimension or the lane-varying
   inner dimension. Only the latter necessarily damages coalescing.
4. Measure requested bytes versus transferred sectors and L1/L2/DRAM traffic.
5. Compare three complete-operator candidates when practical: stride-aware
   consumption, one-time prepared repack/cache, and per-forward materialization.
6. For immutable quantized weights/metadata, prefer a versioned prepared layout
   over repeated `.contiguous()` in every forward when it is storage-efficient.
7. For fused QKV/gate-up outputs, consider stride-aware downstream elementwise
   kernels before copying slices merely to make them contiguous.
