---
name: pangolin-cpu-kernel
description: Re-validate the CPU Pangolin planar GEMV kernel for gptq_p 3/5/6/7-bit on x86-64. Covers forced JIT rebuild, fallback-disable flags, pytest correctness, and sanity/throughput benchmarks.
---

# Testing the CPU Pangolin planar GEMV kernel

Use this skill when you need to verify the `gptqmodel_ext/planar/planar_gemv_cpu.cpp` JIT kernel or its Python wrappers.

## Environment

- Target Python: `/home/ubuntu/.pyenv/versions/3.12.8/bin/python` (the repo has also been validated with torch `2.13.0+cpu`).
- Install the repo in editable mode and ensure `ninja` is available:
  ```bash
  /home/ubuntu/.pyenv/versions/3.12.8/bin/python -m pip install -e .[test,quality]
  ```
- The JIT C++ extension needs `g++` with `-fopenmp` support and `python3-dev` headers.

## Key environment variables

| Variable | Effect |
| --- | --- |
| `GPTQMODEL_PANGOLIN_CPU_FORCE_REBUILD=1` | Delete the cached `pangolin_cpu` JIT build and recompile `planar_gemv_cpu.cpp`. |
| `GPTQMODEL_PANGOLIN_CPU_DISABLE_AVX512=1` | Force runtime dispatch away from AVX-512 (falls back to AVX2 or scalar). |
| `GPTQMODEL_PANGOLIN_CPU_DISABLE_AVX2=1` | Force runtime dispatch away from AVX2 (falls back to scalar). |
| `GPTQMODEL_EXT_VERBOSE=1` | Show `torch.utils.cpp_extension` compile progress and build directory. |

## Pre-test compile check

Before running the full suite, force a rebuild to ensure the current source is what is being tested:

```bash
cd /home/ubuntu/repos/gpt-qmodel-ultra
GPTQMODEL_PANGOLIN_CPU_FORCE_REBUILD=1 \
GPTQMODEL_EXT_VERBOSE=1 \
/home/ubuntu/.pyenv/versions/3.12.8/bin/python - <<'EOF'
from gptqmodel.utils.pangolin import ensure_pangolin_cpu_runtime_available
assert ensure_pangolin_cpu_runtime_available(), "Pangolin CPU JIT extension failed to load"
print("pangolin_cpu ready")
EOF
```

Verify in the build log that `-fopenmp` appears in both compile and link flags and that the build completes with `torch.ops JIT extension ready`.

## Test commands

### Lint
Use the repo's own `format/ruff.toml` config and include the files touched by CPU kernel work (the `_g_idx_block_uniform` cache is shared with the Triton planar paths, so lint those too):

```bash
/home/ubuntu/.pyenv/versions/3.12.8/bin/ruff check \
  gptqmodel/utils/pangolin.py \
  gptqmodel/nn_modules/triton_utils/planar.py \
  gptqmodel/nn_modules/qlinear/tritonv2.py \
  scripts/benchmark_pangolin_cpu.py \
  tests/test_pangolin_cpu_kernel.py \
  --config format/ruff.toml
```

### Correctness
```bash
/home/ubuntu/.pyenv/versions/3.12.8/bin/pytest -q tests/test_pangolin_cpu_kernel.py --no-header
```
Expected: `160 passed` (4 bits × 10 supported M values × 4 fallback-disable modes).

### Sanity benchmark
```bash
/home/ubuntu/.pyenv/versions/3.12.8/bin/python \
  scripts/benchmark_pangolin_cpu.py --sanity --threads 8
```
Expected: speedup > 1.00x for bits 3/5/6/7 and thread scaling with 2/4 threads not regressing vs 1 thread.

### Real-shape sweep across M=1/2/4/8 (optional but useful)
```bash
/home/ubuntu/.pyenv/versions/3.12.8/bin/python \
  scripts/benchmark_pangolin_cpu.py \
  --shapes laguna --bits 3 5 6 7 --batches 1 2 4 8 --threads 8 \
  --output pangolin_cpu_laguna_bench.md
```
Expected: every row reports `speedup` > 1.00x.  The `--output` argument always writes Markdown-formatted tables regardless of the file extension you choose (e.g. `.json` will still contain Markdown text), so either read it as Markdown or post-process it to JSON if you need JSON.

## Common pitfalls

- `pytest` may use a stale cached `.so` if you do not set `GPTQMODEL_PANGOLIN_CPU_FORCE_REBUILD=1` after a C++ source change.
- The CPU extension is only loaded when `x.is_cuda` is `False` in `pangolin_gemv()`; make sure test tensors are on CPU.
- The kernel validates `M` is in `PANGOLIN_SUPPORTED_M` = `(1,2,3,4,5,6,7,8,16,32)` and `K % 32 == 0` / `N % 32 == 0`.
- If the test host does not report AVX-512, the `avx512` fallback tests will simply exercise AVX2/scalar; the `disable` env flags still work.

## Devin Secrets Needed

None.
