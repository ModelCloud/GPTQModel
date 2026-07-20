# Kernel integration workflow

## Repository path

| Concern | Location |
| --- | --- |
| Central JIT extension registry | `gptqmodel/extension.py` |
| Fingerprinting, caching, compiler flags | `gptqmodel/utils/cpp.py` |
| Python wrappers | `gptqmodel/utils/` and `gptqmodel/nn_modules/qlinear/` |
| CUDA/C++ sources | `gptqmodel_ext/` |
| Focused kernel tests | `tests/kernels/` and `tests/test_*_jit.py` |
| Timed experiments | `scripts/benchmark_*.py` |

Inspect `gptqmodel/utils/marlin.py`, `gptqmodel/utils/grasshopper.py`, and their matching extension directories for representative patterns. Reuse their loading and error-reporting conventions; do not import private compiled namespaces directly from unrelated modules.

## Correctness sequence

1. Define a deterministic reference computation and seed.
2. Generate legal packed inputs through the real packer when possible; random integers alone may miss layout errors.
3. Test accumulation-sensitive cases, including long reduction dimensions and adversarial signs/magnitudes.
4. Exercise the last partial tile independently in M, N, and K.
5. Check inputs that are non-contiguous only if the public contract accepts them; otherwise assert a clear rejection.
6. Verify device/stream behavior and that temporary workspace lifetime covers asynchronous execution.
7. Run the kernel through the QuantLinear/backend path, not only as a raw operator.

## Performance sequence

1. Finish JIT compilation and warmup before collecting samples.
2. Use CUDA events for device time or synchronize immediately around host timing.
3. Record median and at least one tail statistic over enough iterations for stable results.
4. Separate decode-like small-M and prefill-like larger-M cases.
5. Compare matched outputs and configs; never compare different precision, layout, or effective work without labeling it.
6. Inspect occupancy, memory traffic, and instruction mix only after end-to-end timing identifies the bottleneck.
