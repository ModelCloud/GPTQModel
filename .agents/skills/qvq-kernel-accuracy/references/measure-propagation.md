# Measuring propagation when arithmetic changes

Use this protocol for a numerical-strategy experiment or to localize a kernel regression. It supplements the
max-absolute inference gate; it cannot override it. Do not require a whole-model sensitivity survey for a small,
already-validated exact decode/address-reuse change.

## Define the measured system

Pin checkpoint revision, actual modeling implementation, tokenizer/token IDs, evaluation positions, dtypes,
attention/cache implementation, tensor parallelism, device/compiler identity, and the kernel dispatch. Inventory actual
modules and fused partitions; do not hard-code another model revision's layer count or assume equal sensitivity.
For a linear-only experiment, keep embeddings, LM head, nonlinear/state kernels, and other non-target code fixed.

Use held-out real text/code at relevant prefill and decode shapes/context lengths. Teacher-force identical token
sequences. Each run owns fresh KV, convolution, and recurrent caches; never share mutated state between arms.
Repeat the unchanged baseline to measure nondeterminism/noise. Compute differences in sufficient precision and retain
actual output casting. Do not change downstream precision to obtain a smoother result.

## Isolate a module's measured error

For one selected module i, compute eager output Y_i and candidate output Yhat_i on the same input X_i. Continue the
model with Yhat_i; every other module uses the reference implementation. Repeat for the module's invocations in the
declared sequence. Record the actual local error, not its configured acceptance limit:

```text
epsilon_i = ||Yhat_i - Y_i||_F / ||Y_i||_F
E_i       = ||z_i - z_0||_F / ||z_0||_F
g_i       = E_i / epsilon_i
```

z_0 and z_i are raw final logits on identical evaluation positions. For block sensitivity, replace logits with the
chosen block boundary and name that boundary. Define token/mask aggregation explicitly. When denominators vanish or
the perturbation is below measurement resolution, report undefined/below-resolution; a silent denominator clamp can
produce a misleading g. The repository's max-absolute 2e-3 gate is NOT epsilon_i in these formulas.

Prefer the candidate's actual error direction. Injecting Y_i + s * (Yhat_i - Y_i) can probe nearby amplitudes; remeasure
epsilon_i after casting. Synthetic noise on real activations is diagnostic only and does not certify the actual kernel
or model quality. Different algorithms at the same L2 error can have different bias, channel concentration, and g.
Probe neighboring amplitudes to check local linearity; do not extrapolate a small-error slope across an untested
1e-3-to-2e-2 sweep and label it observed.

## Check combined behavior

Enable all selected candidate modules and measure the actual final difference. The first-order, uncorrelated-error
prediction is:

```text
E_pred = sqrt(sum_i((g_i * epsilon_i)^2))
```

Compare it with the all-enabled measurement. Correlation, nonlinear interactions, and changed live inputs can invalidate
the prediction. g is conditional on the module, checkpoint, inputs, error direction, precision, and observation point;
values such as 0.1, 0.3, or 1 are assumptions until measured.

A cheaper descriptive summary avoids isolated full forwards for every module: run the complete candidate with paired
eager computations inside each selected linear on that run's current inputs. Measure each local epsilon_i and return
the candidate output. Then:

```text
g_effective = E_all / sqrt(sum_i(epsilon_i^2))
```

This ratio absorbs interactions; it is not automatically the RMS isolated sensitivity or a prediction for another
kernel. If all epsilons are equal and propagated errors are uncorrelated, E_pred reduces to
sqrt(N) * g_RMS * epsilon. Neither N * epsilon nor sqrt(N) * epsilon alone predicts model accuracy.

## Preserve decision-relevant evidence

Report distributions across prompts, token positions, module roles, contexts, and dispatch shapes, including tails.
Record block outputs, raw or explicitly centered logit errors, KL(P_reference || P_candidate), top-token
agreement/margins, and paired task outcomes. Do not mix raw-logit and centered-logit sensitivity definitions.
An unchanged head still receives perturbed hidden states; unchanged nonlinearities can attenuate or amplify errors.
Tensor drift, KL, and task-accuracy percentage points are different quantities.

Use a small disjoint real-model screen to locate problems, then expand only where the decision needs more evidence.
For quality investigation and uncertainty-aware escalation, use
[the quantization-regressions skill](../../gptqmodel-quantization-regressions/SKILL.md).
Maintain the existing kernel contract throughout a quality investigation.
