# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Riemannian budget-constrained optimization (RCO) for per-layer bit allocation.

GSQ produces ordinary scalar layers, so a model may mix widths across blocks.
Choosing that assignment is a multiple-choice knapsack problem: pick one option
per layer to minimize a non-decomposable objective (model loss after
compression) under an exact equality budget. The GSQ paper obtains its 2.37 and
2.62 bits-per-parameter Llama configurations this way (arXiv 2604.18556, Sec.
4.2 and Table 1, citing Helcig & Alistarh, arXiv 2605.00649).

This module is a dependency-free port of that method. The constraint surface is

    M = { alpha : C(alpha) = B },   C(alpha) = sum_i w_i * sum_k p_ik * c_k,

with ``p_i = softmax(alpha_i)`` over the width options of layer ``i``, ``w_i``
the layer's parameter count and ``c_k`` the cost (bits per parameter) of option
``k``. Rather than a penalty or a Lagrange multiplier, every update is kept on
``M`` exactly: the gradient is projected onto the tangent plane, a retraction
restores the budget by bisection, and Adam's first moment is transported onto
the rotated tangent plane.

References
----------
Helcig & Alistarh, "Model Compression with Exact Budget Constraints via
Riemannian Manifolds", arXiv 2605.00649, Secs. 2-3 (Prop. 1-3) and App. A.5.
"""

import math

import numpy as np
import torch


def budget_normal(alpha, costs, weights=None):
    """Normal vector ``dC/dalpha`` of the budget level surface.

    ``dC/dalpha_ik = w_i * p_ik * (c_k - E_i[c])``. The tangent space at
    ``alpha`` is the orthogonal complement of this direction.
    """
    probs = torch.softmax(alpha, dim=-1)
    expectation = (probs * costs).sum(dim=-1, keepdim=True)
    normal = probs * (costs - expectation)
    if weights is not None:
        normal = normal * weights.unsqueeze(-1)
    return normal


def expected_cost(alpha, costs, weights=None):
    """``C(alpha)``: parameter-weighted expected bits per parameter."""
    return float((torch.softmax(alpha, dim=-1) * costs).sum(dim=-1).mul(weights).sum())


def project_gradient(alpha, costs, weights=None):
    """Remove the budget-normal component of ``alpha.grad`` in place.

    Makes each optimizer step budget-preserving by construction. Returns
    ``(coefficient, raw_norm, projected_norm)`` for diagnostics.
    """
    with torch.no_grad():
        normal = budget_normal(alpha, costs, weights).flatten()
        gradient = alpha.grad.flatten()
        raw_norm = gradient.norm().item()
        coefficient = (gradient @ normal) / (normal @ normal + 1e-12)
        alpha.grad.sub_(coefficient * normal.view_as(alpha.grad))
        projected_norm = alpha.grad.norm().item()
    return coefficient.item(), raw_norm, projected_norm


def retraction(alpha, costs, target, weights=None, max_iter=60, tol=1e-6):
    """Shift ``alpha`` along ``costs`` until the budget equals ``target``.

    ``t -> C(alpha + t*costs)`` is monotone (Prop. 3), so bisection converges in
    ``O(log)`` steps. The bracket doubles outward until it contains the root.
    Modifies ``alpha`` in place and returns the achieved budget.
    """
    def cost_at(shift):
        with torch.no_grad():
            probs = torch.softmax(alpha + shift * costs, dim=-1)
            row = (probs * costs).sum(dim=-1)
            return float((row * weights).sum()) if weights is not None else float(row.sum())

    current = cost_at(0.0)
    if abs(current - target) < tol:
        return current

    low, high = -1.0, 1.0
    for _ in range(64):
        if cost_at(low) <= target <= cost_at(high):
            break
        if cost_at(high) < target:
            high *= 2.0
        if cost_at(low) > target:
            low *= 2.0
    else:
        raise RuntimeError('RCO retraction could not bracket the budget target')

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if abs(cost_at(mid) - target) < tol:
            low = high = mid
            break
        if cost_at(mid) > target:
            high = mid
        else:
            low = mid
    with torch.no_grad():
        alpha.add_((0.5 * (low + high)) * costs)
    return cost_at(0.0)


def vector_transport(optimizer, alpha, costs, weights=None):
    """Transport Adam's first moment onto the rotated tangent plane.

    After retraction the tangent plane turns; the stored momentum keeps a small
    budget-normal component that would leak adaptive capacity into the
    constraint direction. Re-projecting removes it (a first-order approximation
    to parallel transport, App. A.5). The second moment is directionless and
    left untouched.
    """
    with torch.no_grad():
        normal = budget_normal(alpha, costs, weights).flatten()
        norm_sq = (normal @ normal).item() + 1e-12
        for group in optimizer.param_groups:
            for parameter in group['params']:
                state = optimizer.state.get(parameter)
                if not state or 'exp_avg' not in state:
                    continue
                momentum = state['exp_avg']
                coefficient = (momentum.flatten() @ normal) / norm_sq
                momentum.sub_(coefficient * normal.view_as(momentum))


def round_with_budget(probs, weights, costs, target, resolution=500):
    """Exact discrete assignment under the budget (multiple-choice knapsack).

    Maximizes the total log-probability of the soft solution subject to the
    discretized budget, so ``sum_i w_i * cost(choice_i) / sum_i w_i`` is as
    close to ``target`` as the width grid allows. Returns option indices.
    """
    if probs.ndim != 2:
        raise ValueError('RCO rounding requires [layers, options] probabilities')
    n_layers, n_options = probs.shape
    if weights.shape != (n_layers,) or costs.shape != (n_options,):
        raise ValueError('RCO rounding requires per-layer weights and per-option costs')

    budget = round(target * resolution)
    totals = float(weights.sum())
    if budget < 0 or not math.isfinite(totals) or totals <= 0:
        raise ValueError('RCO rounding requires a positive target and parameter total')

    # Compute the rounded unit costs on the caller's device, then read the
    # table on the host: the search calls this once per Gumbel draw, so a Python
    # loop over positions here would dominate the step time.
    log_probs = torch.log(probs.clamp_min(1e-30))
    unit_costs = torch.round(costs * weights[:, None] / totals * resolution).to(torch.long)
    gained = log_probs.detach().float().cpu().numpy()
    steps = unit_costs.detach().cpu().numpy()

    table = np.full(budget + 1, -np.inf, dtype=np.float64)
    table[0] = 0.0
    choice = np.zeros((n_layers, budget + 1), dtype=np.int64)
    for layer in range(n_layers):
        updated = np.full(budget + 1, -np.inf, dtype=np.float64)
        for option in range(n_options):
            step = int(steps[layer, option])
            if step > budget:
                continue
            # ``best[j]`` is the score of arriving at budget ``j`` via this
            # option; a single vectorized shift replaces the inner loop.
            best = np.full(budget + 1, -np.inf, dtype=np.float64)
            if step == 0:
                best = table + gained[layer, option]
            else:
                best[step:] = table[:budget + 1 - step] + gained[layer, option]
            improved = best > updated
            improved &= np.isfinite(best)
            updated[improved] = best[improved]
            choice[layer, improved] = option
        table = updated

    # Select the best objective over every feasible budget, not the largest
    # feasible budget: the table entry is the total log-probability, and a
    # lower-cost assignment can score better. Matches the reference solver.
    if not np.isfinite(table).any():
        raise RuntimeError('RCO rounding found no feasible assignment under the budget')
    position = int(np.argmax(table))

    assignment = torch.zeros(n_layers, dtype=torch.long)
    for layer in range(n_layers - 1, -1, -1):
        option = int(choice[layer, position])
        assignment[layer] = option
        position -= int(steps[layer, option])
    # Return on the caller's device: the search indexes this alongside the
    # probability tensors it keeps on the accelerator.
    return assignment.to(probs.device)


def achieved_bits(assignment, weights, costs):
    """Parameter-weighted bits per parameter of a discrete assignment."""
    # The assignment comes back on the search device while the caller may hold
    # the cost vector on the host, so gather on the assignment's device.
    costs = costs.to(assignment.device)
    selected = costs[assignment].float()
    return float((selected * weights.to(assignment.device)).sum() / weights.sum())




class InterpolatedLlamaLayers:
    """Differentiable weight interpolation across per-layer width options.

    RCO needs the model's own loss as a function of the bit assignment, so each
    candidate width for a decoder block is materialized once and the live
    weights are expressed as

        W = W_ref + sum_k p_ik * (W_k - W_ref),

    with ``p_i = softmax(alpha_i)``. ``W_ref`` is the widest option, whose delta
    is exactly zero, so it reproduces the reference implementation's baseline.
    """

    def __init__(self, model, layer_weights, weights, costs, *, control_device=None,
                 groups=None):
        """
        ``groups`` partitions the module names into independent decisions, each
        with one ``alpha`` row. The default gives every module its own group.
        Staged GSQ trains and exports one width per decoder block, so a caller
        that can only act on whole blocks passes the seven projections of each
        block as one group, and the budget is then enforced over exactly the
        units the checkpoint can express.
        """
        self.model = model
        self.layer_weights = layer_weights
        self.control_device = torch.device(control_device or 'cuda:0')
        # Every control tensor must live on the same device as ``alpha``: the
        # budget helpers combine them elementwise, and the database is built on
        # the host while the search runs on an accelerator.
        self.costs = costs.to(self.control_device)
        self.weights = weights.to(self.control_device)
        self.fraction = self.weights / self.weights.sum()
        self.names = list(layer_weights)
        if groups is None:
            groups = [[name] for name in self.names]
        else:
            groups = [list(group) for group in groups]
            flattened = [name for group in groups for name in group]
            if sorted(flattened) != sorted(self.names):
                raise ValueError('RCO groups must partition the interpolated modules exactly')
        self.groups = groups
        self._group_of = {name: index for index, group in enumerate(groups) for name in group}
        self.n_groups = len(groups)
        # Backwards-compatible alias: one group per module is one row per module.
        self.n_layers = self.n_groups
        self.n_options = len(costs)
        self.alpha = torch.zeros(self.n_groups, self.n_options, dtype=torch.float32,
                                 device=self.control_device, requires_grad=True)
        self._originals = {}
        self._deltas = {}
        self._probs_override = None

    def probabilities(self, index):
        """Probabilities for one group, honoring a straight-through override."""
        if self._probs_override is not None:
            return self._probs_override[index].to(self.control_device)
        return torch.softmax(self.alpha[index].to(self.control_device), dim=-1)

    def setup(self):
        """Replace live weights in place with the interpolation graph."""
        from torch import nn

        reference_index = self.n_options - 1
        for name in self.names:
            module = self.model.get_submodule(name)
            # Place options on the module's device: the database may be stored on
            # the host, while the live block is already on an accelerator.
            device = module.weight.device
            dtype = module.weight.dtype
            options = [option.to(dtype=dtype, device=device) for option in self.layer_weights[name]]
            self.layer_weights[name] = options
            reference = options[reference_index]
            self._originals[name] = reference
            # Deltas must keep the parametrized weight's dtype: registering a
            # parametrization may not change it. The reference implementation
            # likewise stores bfloat16 deltas against a bfloat16 base.
            self._deltas[name] = [(option - reference).to(dtype) for option in options]
            module.weight = nn.Parameter(reference.clone(), requires_grad=False)
            nn.utils.parametrize.register_parametrization(
                module, 'weight', _WeightInterpolation(self, name))
        self.model.requires_grad_(False)
        return self

    def deltas(self, name):
        return self._deltas[name]

    def index_of(self, name):
        return self._group_of[name]

    def reference_index(self):
        return self.n_options - 1

    def set_probs(self, probs):
        """Freeze interpolation to explicit probabilities (straight-through)."""
        self._probs_override = probs

    def clear_probs(self):
        self._probs_override = None

    def current_bits(self):
        with torch.no_grad():
            probs = torch.softmax(self.alpha, dim=-1)
            return float((probs * self.costs).sum(dim=-1).mul(self.fraction).sum())

    def retract(self, target, **kwargs):
        return retraction(self.alpha, self.costs, target, self.fraction, **kwargs)

    def cleanup(self):
        """Detach the parametrizations and restore the reference weights."""
        from torch import nn

        for name in self.names:
            module = self.model.get_submodule(name)
            nn.utils.parametrize.remove_parametrizations(module, 'weight', leave_parametrized=False)
            with torch.no_grad():
                module.weight.copy_(self._originals[name].to(module.weight.dtype))
        self._deltas.clear()
        self._probs_override = None


class _WeightInterpolation(torch.nn.Module):
    """Parametrization applying ``W_ref + sum_k p_k * (W_k - W_ref)``."""

    def __init__(self, owner, name):
        super().__init__()
        self.owner = owner
        self.name = name

    def forward(self, weight):
        deltas = self.owner.deltas(self.name)
        probs = self.owner.probabilities(self.owner.index_of(self.name)).to(weight.device)
        mixed = weight
        for offset, delta in enumerate(deltas):
            if offset == self.owner.reference_index():
                continue  # reference delta is exactly zero
            mixed = mixed + probs[offset] * delta
        return mixed


def projected_gumbel_search(interpolated, batches, target, *, steps=200, lr=0.1,
                            tau_init=1.0, tau_min=0.01, gumbel_samples=1,
                            objective=None, log_every=25):
    """Run the projected Gumbel-STE search and return per-layer probabilities.

    Each step draws Gumbel noise at an exponentially annealed temperature, solves
    the budget-constrained knapsack for the hard assignment, and passes a
    straight-through gradient through the softmax of the same perturbed logits.
    The gradient is projected off the budget normal, Adam takes the step, the
    retraction restores the budget exactly and momentum is transported.
    """
    import torch.nn.functional as F

    if objective is None:
        raise ValueError('RCO search requires an objective callable')
    optimizer = torch.optim.Adam([interpolated.alpha], lr=lr)
    history = []
    for step in range(steps):
        progress = step / max(steps - 1, 1)
        tau = max(tau_min, tau_init * (tau_min / tau_init) ** progress)
        optimizer.zero_grad()
        total, count = 0.0, 0
        # Average the loss over every batch and Gumbel sample in the step, as
        # the reference implementation does. Summing instead would scale the
        # gradient by the batch count, making the configured learning rate
        # depend on how the corpus happens to be split.
        scale = 1.0 / (len(batches) * gumbel_samples)
        for _sample in range(gumbel_samples):
            uniform = torch.rand_like(interpolated.alpha).clamp(1e-20, 1 - 1e-20)
            noise = -torch.log(-torch.log(uniform) + 1e-20)
            # The hard assignment is identical across batches for a given draw,
            # because it depends only on the logits and the noise, so solve the
            # knapsack once. It is computed under ``no_grad``: the graph below
            # must be rebuilt per batch, since a second backward through a
            # retained logit graph would free the saved intermediates.
            with torch.no_grad():
                detached = F.softmax((interpolated.alpha + noise) / tau, dim=-1)
                hard = round_with_budget(detached, interpolated.weights,
                                         interpolated.costs, target)
                onehot = F.one_hot(hard, interpolated.n_options).float()
            for batch in batches:
                noisy = (interpolated.alpha + noise) / tau
                soft = F.softmax(noisy, dim=-1)
                interpolated.set_probs(onehot - soft.detach() + soft)
                with torch.enable_grad():
                    loss = objective(batch)
                (loss * scale).backward()
                total += float(loss.detach())
                count += 1
        interpolated.clear_probs()
        project_gradient(interpolated.alpha, interpolated.costs, interpolated.fraction)
        optimizer.step()
        achieved = interpolated.retract(target)
        vector_transport(optimizer, interpolated.alpha, interpolated.costs, interpolated.fraction)
        history.append({'step': step, 'tau': tau,
                        'loss': total / max(count, 1),
                        'budget': achieved, 'drift': abs(achieved - target)})
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f'  rco step {step+1:4d}/{steps} tau={tau:.4f} '
                  f'loss={history[-1]["loss"]:.6f} bits={achieved:.6f} '
                  f'drift={history[-1]["drift"]:.2e}', flush=True)
    return torch.softmax(interpolated.alpha, dim=-1).detach(), history
