---
name: limit-python-test-threads
description: Run or profile CPU-heavy Python tests with native BLAS and OpenMP thread pools hard-capped at 16 threads. Use whenever Codex runs pytest, unittest, tox, nox, CPU profiling, or direct Python test/benchmark code that exercises NumPy, SciPy, scikit-learn, PyTorch CPU operations, compiled OpenMP extensions, or other CPU-parallel native libraries, especially on large-core hosts or after observing high load or thread oversubscription.
---

# Limit Python Test Threads

Cap native CPU thread pools before running Python tests or CPU profiling. The hard maximum is 16 threads. Preserve an explicit lower limit; clamp every request above 16 to 16.

## Run tests

1. Identify the project interpreter. Prefer the active environment, then the project's `.venv/bin/python`, then `python3`.
2. Check that both `pytest` and `threadpoolctl` import in that interpreter.
3. If `threadpoolctl` is missing, add it to the appropriate test/development dependency file when modifying the project is in scope. Otherwise ask before installing it into an existing environment.
4. Run pytest through `scripts/run_cpu_tests.py`, passing the interpreter and all pytest arguments after `--`:

```bash
python /absolute/path/to/skill/scripts/run_cpu_tests.py --python .venv/bin/python -- tests/test_cpu.py -q
```

5. Report the effective thread cap with the test result. Do not invoke bare `pytest` for CPU-heavy tests.

`--limit` accepts lower limits for constrained hosts or shared-core policies. Values above 16 are intentionally clamped:

```bash
python /absolute/path/to/skill/scripts/run_cpu_tests.py --limit 64 --print-env
```

For unittest, tox, nox, benchmarks, CPU profilers, or direct Python commands, export the environment variables printed by `scripts/run_cpu_tests.py --print-env` before launching the command. Cap any tool-specific `--threads`, worker, or Torch intra-op setting at the same effective limit. Where the tested code loads supported BLAS/OpenMP libraries in the current Python process, also wrap the relevant execution scope with `threadpoolctl.threadpool_limits(limits=effective_limit)`.

## Guarantees and limits

The runner sets common BLAS/OpenMP environment variables before Python imports and holds a `threadpoolctl` runtime limiter around the entire pytest session. The repository pytest hook independently applies the same maximum before importing NumPy and Torch, so accidental bare pytest invocations are also capped. This covers libraries loaded before the runtime limiter and gives later-loaded libraries their startup limit.

This does not cap Python-created threads, multiprocessing workers, pytest-xdist workers, application-specific pools, or every nested combination of distinct OpenMP runtimes. Limit those separately when present. Avoid combining 16 threads per process with many xdist/process workers; keep the total concurrency appropriate for the host.

Do not silently add `taskset`, CPU affinity, or process-count restrictions. Those are separate controls.
