---
name: gptq-coverage-review
description: Review or extend GPTQ/GPTQModel quantization tests with measured line and branch coverage, realistic and adversarial numerical cases, dense-reference accuracy checks, and nondeterminism-aware CUDA/Triton validation. Use for quantization math, adaptive damping or clipping, scale search, Hessian/error correction, packing kernels, GPU accuracy regressions, coverage gates, or claims that GPTQ tests are complete.
---

# GPTQ coverage review

Establish both structural coverage and numerical-behavior coverage. Treat post-quantization accuracy as the primary
gate and performance/VRAM as later gates.

## Establish the contract

1. Read the repository `AGENTS.md`, test configuration, changed-code diff against the target branch, and relevant
   math documentation.
2. Inventory every changed or exposed math region, conditional branch, fallback, backend dispatch, dtype conversion,
   and stateful ordering dependency.
3. Define the dense reference, supported configuration space, accuracy metrics, tolerances, repeat count, and failure
   policy before interpreting results.
4. Read [test-matrix.md](references/test-matrix.md) when designing or auditing the test matrix and final report.

Do not equate one of these with another:

- 100% line coverage means every measured line ran.
- 100% branch coverage means every measured decision outcome ran.
- Data-spectrum coverage means representative, boundary, invalid, and adversarial values ran.
- Numerical validation means results agreed with a trustworthy dense or exact reference.

Never claim “full coverage” unless all four are evidenced for the stated scope. Report whole-repository coverage
separately from changed/in-scope algorithm coverage.

## Run review tools

Use the active repository environment. Verify tool resolution before running checks; do not silently use tools from
an unrelated Python environment.

```bash
command -v python ty ruff pytest coverage pip-audit bandit pre-commit check-manifest
ty check
ruff check gptqmodel tests scripts
bash format/format.sh
python -m pytest <targeted-tests>
pip-audit
bandit -q -r gptqmodel
python -m build
twine check dist/*
check-manifest
```

Run `pre-commit run --all-files` only when a repository hook configuration exists. Treat security and packaging
findings according to scope; do not hide unrelated existing findings.

## Measure line and branch coverage

Prefer a normal GIL-enabled Python supported by CI for coverage instrumentation. Free-threaded Python 3.14 can stall
while importing this repository under coverage; a timeout is an instrumentation failure, not evidence of coverage.

Use separate CPU and GPU coverage data, then combine it:

```bash
coverage erase
coverage run --branch --parallel-mode --source=gptqmodel -m pytest <cpu-tests>
CUDA_VISIBLE_DEVICES=<assigned-device> \
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.50 \
coverage run --branch --parallel-mode --source=gptqmodel -m pytest <gpu-tests>
coverage combine
coverage report --show-missing
coverage html
coverage json -o coverage.json
```

Apply `--fail-under=100` only to a scope actually expected to reach 100%. Require 100% line and branch coverage for
changed accuracy-critical math unless the user explicitly accepts a documented exception. Do not add
`pragma: no cover`, omit files, mock away math, or broaden exclusions merely to raise the percentage.

Use coverage contexts or separate reports when CPU mocks would otherwise make a GPU path look covered. Verify that
real CUDA/Triton lines executed on real hardware.

## Validate numerical behavior

For every changed quantization region:

1. Compare dequantized weights and raw outputs against a full dense FP32 reference whenever feasible; add BF16/FP16
   references when those are production dtypes.
2. Validate local math and end-to-end propagation. For GPTQ, include the sequential inverse-Hessian correction and
   prior-group update ordering.
3. Measure at least MAE, RMSE, relative L2, maximum absolute error, finite-value status, and relevant logical outputs.
   Add KLD and top-1 agreement for output distributions.
4. Test invariants such as zero preservation, monotonic bounds, scale positivity, legal code ranges, symmetry, shape
   preservation, and fail-closed fallbacks.
5. Use independently seeded held-out activations. Never evaluate only on calibration samples.
6. Compare proposed behavior with an unchanged control using identical weights, activations, seeds, dtype, device,
   and quantization settings.

Do not use decoded text drift as the only accuracy signal. Do not accept a lower mean error when worst-case error,
KLD, task accuracy, or finite-value behavior regresses beyond the declared gate.

## Treat GPU kernels as potentially nondeterministic

Never assume CUDA or Triton kernels are deterministic. Atomics, reduction order, launch scheduling, autotuning,
compiler changes, and concurrent streams can alter floating-point results.

For a changed GPU path:

1. Warm up separately, synchronize before reading results, and record GPU model, compute capability, driver, CUDA,
   PyTorch, Triton, kernel configuration, dtype, and shapes.
2. Run at least 10 same-seed repetitions and multiple independent seeds for an accuracy-critical kernel.
3. Compare every repetition independently with the dense/exact reference. Report worst-case and distributional
   error, not only the mean across repetitions.
4. Track run-to-run spread separately from reference error. A stable wrong answer must fail; a variable answer within
   tolerance must be reported as nondeterministic.
5. Exercise fresh processes, reordered test execution, concurrent streams or workers when the production path
   permits concurrency, and allocator pressure where relevant.
6. Test deterministic-algorithm mode as an additional diagnostic when supported, not as a substitute for the
   production configuration.
7. Require bitwise equality only for contracts that truly require it, such as exact logical candidate selection or
   packed integer codes. For floating outputs, derive dtype- and operation-aware `atol`/`rtol` before running the
   candidate.

Never choose tolerances by observing the regression and widening them until it passes. Fail closed on non-finite
results, missing calibration geometry, skipped reference comparisons, unavailable metrics, or insufficient repeats.

## Cover realistic and adversarial data

Build a parameterized matrix rather than isolated happy-path tests. Include:

- ordinary model-like Gaussian and structured projection inputs;
- exact boundaries, partial groups, non-contiguous tensors, and supported dtypes/bits/group sizes;
- zeros, constants, ties, rounding midpoints, and dynamic-range extremes;
- weight and activation outliers, correlated inputs, singular/rank-deficient Hessians, and ill-conditioned Hessians;
- invalid shapes/configurations and missing/invalid calibration geometry;
- static and adaptive damping, adaptive clipping objectives, scale-search modes, feature combinations, and explicit
  disabled controls;
- stateful prior-group updates, shared-Hessian ownership, cleanup after exceptions, and concurrent subset execution.

Use pairwise or generated coverage only after all known semantic boundaries are pinned explicitly. Seed randomized
or property tests and print the seed on failure.

## Gate and report

Run tests in layers: tiny reference unit tests, parameterized math tests, GPU kernel micro-tests, integration tests,
model-held-out telemetry, then performance/VRAM checks. Accuracy failures stop later optimization claims.

Present full ASCII tables with explicit columns for algorithm, adaptive damping, adaptive clipping, scale search,
reference dtype, test dtype, device, repeats, coverage, accuracy metrics, nondeterminism spread, runtime, and VRAM.
Never label distinct features with a generic “adaptive” column.

End with one of these conclusions for each stated scope:

- `PASS`: all declared structural, spectrum, reference, and nondeterminism gates ran and passed.
- `PARTIAL`: identify exact untested lines, branches, configurations, devices, or data classes.
- `FAIL`: identify the first violated accuracy or validity gate.
- `BLOCKED`: identify unavailable hardware, reference, dependency, or instrumentation; never convert a block into a
  pass.
