# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Native-MLX execution for differentiable RCO allocation search.

Torch remains the checkpoint and allocation-schema boundary. This module owns
the expensive model replay: interpolation, teacher KL, projected Gumbel
straight-through gradients, and Adam run in MLX. The small budget DP and retraction solve on the CPU.
"""

import math
from pathlib import Path

import numpy as np


SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen3"})


def rco_mlx_available() -> bool:
    """Return whether Apple silicon has the MLX packages needed by RCO."""
    import platform
    import sys

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import mlx_lm.utils  # noqa: F401
    except ImportError:
        return False
    return True


def rco_mlx_supports_model(model_type) -> bool:
    """Return whether the dense decoder layout is supported by the MLX path."""
    return model_type in SUPPORTED_MODEL_TYPES


def _child(model, name):
    for part in name.split("."):
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model


def _replace_child(model, name, value):
    head, _, leaf = name.rpartition(".")
    parent = _child(model, head)
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _softmax(values):
    import mlx.core as mx

    return mx.softmax(values.astype(mx.float32), axis=-1, precise=True)


def _expected_cost(alpha, costs, fraction):
    import mlx.core as mx

    probabilities = _softmax(alpha)
    return mx.sum((probabilities * costs).sum(axis=-1) * fraction)


def _budget_normal(alpha, costs, fraction):
    probabilities = _softmax(alpha)
    expectation = (probabilities * costs).sum(axis=-1, keepdims=True)
    return probabilities * (costs - expectation) * fraction[:, None]


def _tangent_gradient(alpha, gradient, costs, fraction):
    """Project a gradient off the budget normal without leaving MLX."""
    import mlx.core as mx

    normal = _budget_normal(alpha, costs, fraction)
    flat_gradient = gradient.reshape(-1)
    flat_normal = normal.reshape(-1)
    coefficient = mx.sum(flat_gradient * flat_normal) / (
        mx.sum(flat_normal * flat_normal) + 1e-12
    )
    return gradient - coefficient * normal


def _adam_update(alpha, gradient, momentum, velocity, step_number, learning_rate):
    """Apply the same bias-corrected Adam recurrence in both RCO backends."""
    import mlx.core as mx

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    momentum = beta1 * momentum + (1 - beta1) * gradient
    velocity = beta2 * velocity + (1 - beta2) * (gradient * gradient)
    corrected_momentum = momentum / (1 - beta1 ** step_number)
    corrected_velocity = velocity / (1 - beta2 ** step_number)
    alpha = alpha - learning_rate * corrected_momentum / (
        mx.sqrt(corrected_velocity) + epsilon
    )
    return alpha, momentum, velocity


def _retraction(alpha, costs, target, fraction, *, max_iter=60, tol=1e-6):
    """Shift logits along the cost vector until the exact budget is restored.

    The rank-one retraction converges through a scalar bisection. Keep that
    scalar recurrence on the host in NumPy, synchronizing the small control
    tensors once instead of synchronizing every bisection probe on Metal. The
    updated alpha is still submitted to MLX for the model replay.
    """
    alpha_values = np.asarray(alpha, dtype=np.float64)
    cost_values = np.asarray(costs, dtype=np.float64)
    fractions = np.asarray(fraction, dtype=np.float64)

    def cost_at(shift):
        shifted = alpha_values + float(shift) * cost_values
        shifted -= shifted.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return float(np.sum(probabilities * cost_values, axis=-1) @ fractions)

    current = cost_at(0.0)
    if abs(current - target) < tol:
        return 0.0

    low, high = -1.0, 1.0
    for _ in range(64):
        low_cost = cost_at(low)
        high_cost = cost_at(high)
        if low_cost <= target <= high_cost:
            break
        if high_cost < target:
            high *= 2.0
        if low_cost > target:
            low *= 2.0
    else:
        raise RuntimeError("RCO retraction could not bracket the budget target")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if abs(cost_at(mid) - target) < tol:
            low = high = mid
            break
        if cost_at(mid) > target:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def budget_constrained_argmax_mlx(scores, weights, costs, target, resolution=500):
    """Host DP returning an MLX assignment with the Torch solver's tie behavior."""
    import mlx.core as mx

    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("RCO scores must be finite floating [groups, options] logits")
    n_layers, n_options = scores.shape
    if not n_layers or not n_options:
        raise ValueError("RCO assignment requires nonempty groups and options")
    if weights.shape != (n_layers,) or costs.shape != (n_options,):
        raise ValueError("RCO assignment requires per-group weights and per-option costs")
    budget = round(target * resolution)
    totals = float(weights.sum())
    if budget < 0 or not math.isfinite(totals) or totals <= 0:
        raise ValueError("RCO rounding requires a positive target and parameter total")
    unit_costs = np.rint(
        costs[None, :] * weights[:, None] / totals * resolution
    ).astype(np.int64)
    table = np.full(budget + 1, -np.inf, dtype=np.float64)
    table[0] = 0.0
    choice = np.zeros((n_layers, budget + 1), dtype=np.int64)
    for layer in range(n_layers):
        updated = np.full(budget + 1, -np.inf, dtype=np.float64)
        for option in range(n_options):
            step = int(unit_costs[layer, option])
            if step > budget:
                continue
            best = np.full(budget + 1, -np.inf, dtype=np.float64)
            if step == 0:
                best = table + scores[layer, option]
            else:
                best[step:] = table[: budget + 1 - step] + scores[layer, option]
            improved = (best > updated) & np.isfinite(best)
            updated[improved] = best[improved]
            choice[layer, improved] = option
        table = updated
    if not np.isfinite(table).any():
        raise RuntimeError("RCO rounding found no feasible assignment under the budget")
    position = int(np.argmax(table))
    assignment = np.zeros(n_layers, dtype=np.int64)
    for layer in range(n_layers - 1, -1, -1):
        option = int(choice[layer, position])
        assignment[layer] = option
        position -= int(unit_costs[layer, option])
    return mx.array(assignment)


class MLXInterpolatedLlamaLayers:
    """Differentiable interpolation of dense decoder projection candidates."""

    def __init__(self, model, layer_weights, weights, costs, *, groups=None):
        import mlx.core as mx

        self.model = model
        names = list(layer_weights)
        if groups is None:
            groups = [[name] for name in names]
        else:
            groups = [list(group) for group in groups]
            flattened = [name for group in groups for name in group]
            if sorted(flattened) != sorted(names):
                raise ValueError("RCO groups must partition the interpolated modules exactly")
        self.names = names
        self.groups = groups
        self._group_of = {
            name: index for index, group in enumerate(groups) for name in group
        }
        self.n_groups = len(groups)
        self.n_layers = self.n_groups
        self.n_options = len(costs)
        self.costs = mx.array(np.asarray(costs, dtype=np.float32))
        weights = mx.array(np.asarray(weights, dtype=np.float32))
        self.weights = weights
        self.fraction = weights / weights.sum()
        self.alpha = mx.zeros((self.n_groups, self.n_options), dtype=mx.float32)
        self._probs_override = None
        self._original_layers = {}
        self._deltas = {}

        for name in names:
            module = _child(model, name)
            if not hasattr(module, "weight") or module.weight.ndim != 2:
                raise ValueError(
                    f"MLX RCO requires an unquantized dense Linear at {name}"
                )
            dtype = module.weight.dtype
            options = [
                mx.array(np.asarray(option.detach().float().cpu(), dtype=np.float32)).astype(dtype)
                for option in layer_weights[name]
            ]
            if any(option.shape != module.weight.shape for option in options):
                raise ValueError(f"MLX RCO candidate shape differs from {name}")
            reference = options[-1]
            deltas = [(option - reference) for option in options[:-1]]
            self._original_layers[name] = module
            self._deltas[name] = deltas
            replacement = _MLXWeightInterpolation(
                owner=self,
                name=name,
                reference=reference,
                bias=getattr(module, "bias", None),
                output_dtype=dtype,
                deltas=deltas,
            ).module
            _replace_child(model, name, replacement)
        mx.eval(self.alpha, self._deltas)

    def probabilities(self, index):
        if self._probs_override is not None:
            return self._probs_override[index]
        return _softmax(self.alpha[index:index + 1])[0]

    def index_of(self, name):
        return self._group_of[name]

    def reference_index(self):
        return self.n_options - 1

    def set_probs(self, probabilities):
        """Set straight-through probabilities used by every projection."""
        self._probs_override = probabilities

    def clear_probs(self):
        self._probs_override = None

    def current_bits(self):
        return float(_expected_cost(self.alpha, self.costs, self.fraction).item())

    def retract(self, target, **kwargs):
        shift = _retraction(
            self.alpha, self.costs, target, self.fraction, **kwargs
        )
        self.alpha = self.alpha + shift * self.costs
        return self.current_bits()

    def cleanup(self):
        import mlx.core as mx

        for name, module in self._original_layers.items():
            _replace_child(self.model, name, module)
        self._original_layers.clear()
        self._deltas.clear()
        self._probs_override = None
        mx.eval(self.model.parameters())

    def torch_outputs(self):
        """Return detached Torch probabilities and logits for the common schema."""
        import mlx.core as mx
        import torch

        probabilities = _softmax(self.alpha)
        mx.eval(probabilities, self.alpha)
        return (
            torch.from_dlpack(probabilities).detach().cpu(),
            torch.from_dlpack(self.alpha).detach().cpu(),
        )


class _MLXWeightInterpolation:
    """Replacement linear layer with the same projection output contract."""

    def __init__(self, *, owner, name, reference, bias, output_dtype, deltas):
        import mlx.nn as nn

        class Interpolation(nn.Module):
            def __init__(self):
                super().__init__()
                self.owner = owner
                self.name = name
                self.reference = reference
                self.bias = bias
                self.output_dtype = output_dtype
                self.deltas = list(deltas)

            def __call__(self, values):
                probabilities = self.owner.probabilities(self.owner.index_of(self.name))
                mixed = self.reference
                for offset, delta in enumerate(self.deltas):
                    # Keep the interpolation in the model dtype. The Torch
                    # backend multiplies a zero-dimensional float32
                    # probability by a model-dtype delta, and Torch treats a
                    # zero-dimensional operand as a scalar for type
                    # promotion, so that product stays in the delta's dtype.
                    # MLX promotes a float32 array against a low-precision
                    # array to float32 instead, which would widen every
                    # projection to float32 and diverge from the reference
                    # search. Cast the scalar explicitly so both backends run
                    # the same arithmetic in the same precision.
                    mixed = mixed + probabilities[offset].astype(mixed.dtype) * delta
                output = values @ mixed.T
                if self.bias is not None:
                    output = output + self.bias
                # The candidate search must not silently change the activation
                # dtype observed by the surrounding decoder block.
                return output.astype(self.output_dtype)

        self.module = Interpolation()

    def __getattr__(self, name):
        return getattr(self.module, name)


class MLXTeacherKL:
    """Mean-token dense-teacher KL with resident or on-disk hidden states."""

    def __init__(self, model, batches, directory, *, chunk_tokens=128, resident_cache=False):
        import mlx.core as mx

        if not batches:
            raise ValueError("RCO teacher KL requires nonempty batches")
        if (
            isinstance(chunk_tokens, bool)
            or not isinstance(chunk_tokens, int)
            or chunk_tokens < 1
        ):
            raise ValueError("RCO KL chunk_tokens must be a positive integer")
        self.model = model
        self.batches = [mx.array(np.asarray(batch, dtype=np.int64)) for batch in batches]
        self.chunk_tokens = chunk_tokens
        self.resident_cache = resident_cache
        self._resident = []
        self._entropy = []
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.paths = []
        for index, ids in enumerate(self.batches):
            hidden = mx.stop_gradient(model.model(ids))
            mx.eval(hidden)  # Finish each batch before capturing the next one.
            self._entropy.append(self._teacher_entropy(hidden))
            if resident_cache:
                self._resident.append(hidden)
                self.paths.append(None)
            else:
                path = directory / f"{index:05d}.safetensors"
                mx.save_safetensors(str(path), {"hidden": hidden})
                self.paths.append(path)
        mx.eval(self._resident, self._entropy)

    def cleanup(self):
        """Release dense teacher hidden-state scratch after optimization."""
        self._resident.clear()
        self._entropy.clear()
        self.paths.clear()

    def _logits(self, hidden):
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(hidden)
        return self.model.lm_head(hidden)

    def _teacher_entropy(self, hidden):
        """Return ``sum(t * log t)`` for one batch of teacher hidden states.

        This is the constant term of ``KL(t || s)``: it depends only on the
        frozen teacher, so it is folded into the objective once at construction
        instead of being recomputed on every optimizer step.
        """
        import mlx.core as mx
        import mlx.nn as nn

        flat = hidden.reshape(-1, hidden.shape[-1])
        total = mx.zeros((), dtype=mx.float32)
        for start in range(0, flat.shape[0], self.chunk_tokens):
            stop = start + self.chunk_tokens
            teacher_logp = nn.log_softmax(
                self._logits(flat[start:stop]).astype(mx.float32), axis=-1
            )
            total = total + mx.sum(mx.exp(teacher_logp) * teacher_logp)
        return mx.stop_gradient(total)

    def __call__(self, index):
        import mlx.core as mx
        import mlx.nn as nn

        student = self.model.model(self.batches[index])
        if self.resident_cache:
            teacher = self._resident[index]
        else:
            teacher = mx.load(str(self.paths[index]))["hidden"]
        if teacher.shape != student.shape:
            raise ValueError("RCO teacher/student hidden shapes differ")
        student = student.reshape(-1, student.shape[-1])
        teacher = teacher.reshape(-1, teacher.shape[-1])

        def chunk_loss(student_hidden, teacher_hidden):
            student_logits = self._logits(student_hidden).astype(mx.float32)
            teacher_logp = mx.stop_gradient(nn.log_softmax(
                self._logits(teacher_hidden).astype(mx.float32), axis=-1
            ))
            # KL(t || s) = sum(t log t) - sum(t * s_logits) + logsumexp(s_logits).
            # The teacher entropy is constant and folded in below, so a step runs
            # two student reductions over the vocabulary instead of materializing
            # the full-vocabulary student log-probabilities.
            return -mx.sum(mx.exp(teacher_logp) * student_logits) + mx.sum(
                mx.logsumexp(student_logits, axis=-1)
            )

        # Rematerialize vocabulary intermediates in backward instead of retaining
        # all chunks' logits for the lifetime of the decoder graph.
        checkpointed_loss = mx.checkpoint(chunk_loss)
        loss = mx.zeros((), dtype=mx.float32)
        for start in range(0, student.shape[0], self.chunk_tokens):
            stop = start + self.chunk_tokens
            loss = loss + checkpointed_loss(student[start:stop], teacher[start:stop])
        return (loss + self._entropy[index]) / student.shape[0]


class MLXCausalLMLoss:
    """Causal next-token cross entropy for the legacy RCO objective."""

    def __init__(self, model, batches):
        import mlx.core as mx

        if not batches:
            raise ValueError("RCO cross entropy requires nonempty batches")
        self.model = model
        self.batches = [mx.array(np.asarray(batch, dtype=np.int64)) for batch in batches]

    def __call__(self, index):
        import mlx.core as mx
        import mlx.nn as nn

        ids = self.batches[index]
        if ids.ndim != 2 or ids.shape[1] < 2:
            raise ValueError("RCO cross entropy needs [batch, tokens] with at least two tokens")
        logits = self.model(ids)[:, :-1, :].astype(mx.float32)
        labels = ids[:, 1:]
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean"
        )

    def cleanup(self):
        """Release references owned by the objective before replacing the model."""
        self.batches.clear()


def projected_gumbel_search_mlx(
    interpolated,
    batches,
    target,
    *,
    steps=200,
    lr=0.1,
    tau_init=1.0,
    tau_min=0.01,
    gumbel_samples=1,
    objective=None,
    log_every=25,
    batch_weights=None,
    noise_sampler=None,
):
    """Run paper-aligned projected Gumbel search entirely through MLX autodiff."""
    import mlx.core as mx

    if objective is None:
        raise ValueError("RCO search requires an objective callable")
    if not batches or steps < 1 or gumbel_samples < 1:
        raise ValueError("RCO search requires batches, positive steps and Gumbel samples")
    batch_weights = [1.0] * len(batches) if batch_weights is None else list(batch_weights)
    if (
        len(batch_weights) != len(batches)
        or any(not math.isfinite(weight) or weight <= 0 for weight in batch_weights)
    ):
        raise ValueError("RCO batch weights must be finite, positive and match the batches")
    total_weight = sum(batch_weights)
    batch_fractions = [weight / total_weight for weight in batch_weights]
    interpolated.retract(target)

    momentum = mx.zeros_like(interpolated.alpha)
    velocity = mx.zeros_like(interpolated.alpha)
    history = []
    for step in range(steps):
        progress = step / max(steps - 1, 1)
        temperature = max(
            tau_min, tau_init * (tau_min / tau_init) ** progress
        )
        draws = []
        for _ in range(gumbel_samples):
            if noise_sampler is None:
                uniform = mx.clip(
                    mx.random.uniform(
                        shape=interpolated.alpha.shape, dtype=mx.float32
                    ),
                    mx.array(1e-20),
                    mx.array(1 - 1e-20),
                )
                noise = -mx.log(-mx.log(uniform) + 1e-20)
            else:
                noise = noise_sampler(interpolated.alpha.shape)
            noisy = (interpolated.alpha + noise) / mx.array(temperature)
            hard = budget_constrained_argmax_mlx(
                noisy,
                interpolated.weights,
                interpolated.costs,
                target,
            )
            onehot = mx.eye(interpolated.n_options, dtype=mx.float32)[hard]
            draws.append((noise, onehot))

        loss = mx.zeros((), dtype=mx.float32)
        raw_gradient = mx.zeros_like(interpolated.alpha)
        for noise, onehot in draws:
            for batch, fraction in zip(batches, batch_fractions):
                def training_loss(alpha):
                    soft = _softmax((alpha + noise) / mx.array(temperature))
                    interpolated.set_probs(onehot - mx.stop_gradient(soft) + soft)
                    try:
                        return objective(batch) * (fraction / gumbel_samples)
                    finally:
                        interpolated.clear_probs()

                batch_loss, batch_gradient = mx.value_and_grad(training_loss)(interpolated.alpha)
                loss = loss + batch_loss
                raw_gradient = raw_gradient + batch_gradient
                # A lazy sum would retain every batch/draw's decoder activations.
                mx.eval(loss, raw_gradient)
                del batch_loss, batch_gradient
        gradient = _tangent_gradient(
            interpolated.alpha,
            raw_gradient,
            interpolated.costs,
            interpolated.fraction,
        )

        step_number = step + 1
        interpolated.alpha, momentum, velocity = _adam_update(
            interpolated.alpha,
            gradient,
            momentum,
            velocity,
            step_number,
            lr,
        )
        achieved = interpolated.retract(target)

        normal = _budget_normal(
            interpolated.alpha, interpolated.costs, interpolated.fraction
        )
        tangent_norm = normal.reshape(-1)
        coefficient = mx.sum(momentum.reshape(-1) * tangent_norm) / (
            mx.sum(tangent_norm * tangent_norm) + 1e-12
        )
        momentum = momentum - coefficient * normal
        mx.eval(loss, interpolated.alpha, momentum, velocity)

        record = {
            "step": step,
            "tau": temperature,
            "loss": float(loss.item()),
            "budget": achieved,
            "drift": abs(achieved - target),
        }
        history.append(record)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(
                f"  rco step {step+1:4d}/{steps} tau={temperature:.4f} "
                f"loss={record['loss']:.6f} bits={achieved:.6f} "
                f"drift={record['drift']:.2e}",
                flush=True,
            )

    probabilities = _softmax(interpolated.alpha)
    mx.eval(probabilities)
    return probabilities, history


def write_mlx_assignment(path, *, probabilities, logits, history, weights, costs, names,
                          groups, target_bits):
    """Serialize MLX results under the exact Torch allocator schema."""
    import torch

    payload = {
        "probs": torch.from_dlpack(probabilities).detach().cpu(),
        "logits": torch.from_dlpack(logits).detach().cpu(),
        "history": history,
        "weights": torch.as_tensor(np.asarray(weights), dtype=torch.float32),
        "costs": torch.as_tensor(np.asarray(costs), dtype=torch.float32),
        "names": names,
        "groups": groups,
        "target_bits": target_bits,
    }
    torch.save(payload, path)
    return payload
