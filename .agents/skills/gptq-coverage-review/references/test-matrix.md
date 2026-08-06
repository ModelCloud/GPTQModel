# GPTQ coverage and accuracy matrix

Use this reference to design tests and render the review report. Adapt unsupported combinations explicitly; do not
silently drop them.

## Contents

- Coverage inventory
- Data spectrum
- Quantization configurations
- Algorithm-specific cases
- GPU nondeterminism matrix
- Final results template

## Coverage inventory

Map each in-scope region before adding tests:

```text
+----------------------+----------------+----------------+----------------------+----------------------+
| Region               | Lines/branches | Dense reference| Data classes         | Backend/device       |
+----------------------+----------------+----------------+----------------------+----------------------+
| Hessian accumulation |                | X.T @ X FP32   | normal/outlier/rank  | CPU/CUDA             |
| Damping selection    |                | direct formula | isotropic/singular   | CPU/CUDA             |
| GPTQ correction      |                | scalar loop    | groups/order/ties    | CPU/CUDA             |
| Range/scale search   |                | exhaustive loop| boundary/outlier     | eager/Triton         |
| Packing/dequant      |                | logical codes  | bits/sym/group tails | kernel/fallback      |
| Output telemetry     |                | dense forward  | stable/extreme logits| CPU/CUDA             |
+----------------------+----------------+----------------+----------------------+----------------------+
```

## Data spectrum

### General tensors

- All zero, constant positive, constant negative, alternating sign, identity, diagonal, and orthogonal cases.
- Gaussian, uniform, skewed, heavy-tailed, sparse, and model-like Q/K/V/O/gate/up/down projections.
- Exact quantization values, half-step rounding boundaries, saturation boundaries, tied candidate losses, and
  one-ULP perturbations.
- Single weight outlier, per-row outliers, activation-only outliers, dead activation columns, and extreme-but-finite
  dynamic ranges.
- Contiguous, transposed/non-contiguous, sliced, odd strides, minimum legal shapes, and large realistic shapes.
- Columns smaller than a group, exactly one group, one past a group, multiple groups, and a final partial group.
- Invalid NaN/Inf inputs where the API promises sanitization or rejection; verify the documented fail-closed result.

### Quantization configurations

Cover every configuration supported by the changed code, including:

- bit widths 2/3/4/8 where supported;
- symmetric and asymmetric quantization;
- grouped and ungrouped quantization; group sizes 32/64/128 plus boundary sizes relevant to the change;
- activation ordering on/off, static groups on/off, and group-aware ordering on/off;
- FP32 reference with BF16/FP16 production inputs;
- eager/fallback and each changed CUDA/Triton kernel path;
- legacy configuration aliases and serialized round trips.

Do not create a Cartesian explosion blindly. Pin every semantic boundary, then use pairwise/property sampling for the
remaining interactions.

## Algorithm-specific cases

### Adaptive damping

- Isotropic Hessian: spectral ratio is one and damping reduces to its documented baseline before clamping.
- Diagonal Hessian with known spectrum: verify the exact expected ratio and clamp boundaries.
- Correlated, rank-deficient, singular, nearly singular, zero-diagonal, and ill-conditioned Hessians.
- Power-iteration, Lanczos, and diagonal estimators where supported; repeat probes and test deterministic seeding
  explicitly.
- Fixed damping control versus adaptive damping only; keep scale search identical.
- Module prior, group-size prior, group-error feedback, online feedback, raw residual, and Hessian-weighted feedback
  independently on/off.
- Verify adaptive selection still feeds the canonical GPTQ inverse-Hessian correction rather than a weight-only
  surrogate.
- Cholesky recovery at initial success, one increment, multiple increments, invalid inverse, and exhausted recovery.

### Adaptive clipping

- Fixed damping plus adaptive clipping versus fixed damping plus activation scale search.
- Adaptive damping plus clipping versus adaptive damping plus activation scale search.
- `gptq_error`, `hessian_diag`, and `mse` objectives tested as distinct methods.
- Candidate `1.0`, winning clipped candidate, tied candidates, duplicate candidates, unordered candidates if
  permitted, and boundary factors.
- Exact sequential GPTQ-error reference including all updates from prior groups before evaluating the next group.
- Missing/invalid inverse-Cholesky geometry must choose the full unclipped range; never fall back silently to
  weight-only MSE.
- Unsupported `static_groups`, GPTAQ, and FOEM combinations must fail closed when required.

### Scale search

- Disabled min/max control and explicit MSE, activation-diagonal, full-Hessian, hybrid, and supported Marlin
  objectives.
- Exhaustive reference winner and bitwise logical scale/zero or candidate-index checks where exact selection is the
  contract.
- Adversarial BF16 values for which an approximate shortlist chooses the wrong candidate.
- Zero/invalid activation importance fallback, skewed importance, full versus diagonal Hessian, and partial groups.
- Batched versus scalar reference after all prior-group weight updates.

### Shared Hessian and lifecycle

- Shared accumulation matches independently accumulated Hessians.
- Borrowed Hessians are never mutated by damping, permutation, or cleanup.
- No unused full-Hessian clones remain live.
- Concurrent subsets cannot observe another task's shared-state flags.
- Normal completion, early exception, OOM/retry, and fallback paths release temporary state without deleting
  caller-owned dictionary entries during iteration.

### Output-error telemetry

- Identical dense/quantized outputs produce zero MAE/RMSE/relative-L2/KLD and full top-1 agreement.
- Known perturbations match a separately calculated reference.
- Zero reference norm, one-class/one-column outputs, ties, extreme logits, and non-finite inputs follow documented
  behavior.
- KLD uses numerically stable log-softmax/softmax computation and declares its reduction direction.
- Calibration capture is bounded; held-out evaluation uses an independent RNG stream.

## GPU nondeterminism matrix

For each changed kernel, record:

```text
+---------+------+-------+---------+------------+------------+-------------+--------------+----------------+
| Kernel  | GPU  | Dtype | Shape   | Repetitions| Seed count | vs-ref max  | run spread   | Exact contract |
+---------+------+-------+---------+------------+------------+-------------+--------------+----------------+
|         |      |       |         | >=10       | >=3        |             |              | yes/no         |
+---------+------+-------+---------+------------+------------+-------------+--------------+----------------+
```

Run same-process repetitions, fresh-process repetitions, warm and cold allocator states, reordered tests, and
concurrent execution when supported. Synchronize before measurement. Report maximum reference error and maximum
pairwise/run-to-run spread.

## Final results template

```text
+----------------------+--------+--------+---------+----------+---------+----------+----------+---------+
| Scope                | Lines  | Branch | Spectrum| Reference| Repeats | Accuracy | Nondet   | Status  |
+----------------------+--------+--------+---------+----------+---------+----------+----------+---------+
| Changed GPTQ math    |        |        |         |          |         |          |          |         |
| GPU/Triton paths     |        |        |         |          |         |          |          |         |
| Whole repository     |        |        |         |          |         |          |          |         |
+----------------------+--------+--------+---------+----------+---------+----------+----------+---------+
```

List every partial or blocked cell below the table with its exact file/region, missing condition, and next
executable test.
