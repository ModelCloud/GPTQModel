# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU-aware calibration helpers.

The helpers in this module deliberately operate on dense tensors.  They are
used before a QVQ projection is replaced, so the exact reparameterization can
be folded into the dense weights without changing the model's floating-point
function.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def silu_derivative(values: torch.Tensor) -> torch.Tensor:
    """Return the exact derivative of SiLU without autograd graph creation."""

    sigmoid = torch.sigmoid(values)
    return sigmoid * (1.0 + values * (1.0 - sigmoid))


def apply_swiglu_reparameterization(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an exact channel-wise SwiGLU up/down reparameterization.

    Linear weights use the usual ``[out_features, in_features]`` layout.  If
    ``s`` is applied to an up row, the matching down column is divided by the
    same value.  The gate is returned unchanged because SiLU is not
    homogeneous and therefore cannot safely participate in this rescaling.
    """

    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("SwiGLU projection weights must be rank-2 tensors.")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("SwiGLU gate and up weights must have identical shapes.")
    if down_weight.shape[1] != up_weight.shape[0]:
        raise ValueError("SwiGLU down columns must match gate/up output channels.")
    if scales.ndim != 1 or scales.numel() != up_weight.shape[0]:
        raise ValueError("SwiGLU scales must have one value per intermediate channel.")
    if gate_weight.device != scales.device or not scales.is_floating_point():
        raise ValueError(
            "SwiGLU weights and scales must share a floating-point device."
        )
    if not torch.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError("SwiGLU scales must be finite and strictly positive.")

    scale = scales.to(dtype=up_weight.dtype)
    return (
        gate_weight.clone(),
        up_weight * scale[:, None],
        down_weight / scale[None, :].to(dtype=down_weight.dtype),
    )


def _validate_calibration_geometry(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    if inputs.ndim != 2:
        raise ValueError("SwiGLU calibration inputs must be rank-2 [tokens, hidden].")
    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("SwiGLU projection weights must be rank-2 tensors.")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("SwiGLU gate and up weights must have identical shapes.")
    if (
        inputs.shape[1] != gate_weight.shape[1]
        or down_weight.shape[1] != gate_weight.shape[0]
    ):
        raise ValueError(
            "SwiGLU calibration tensors have incompatible feature dimensions."
        )
    if not all(
        value.is_floating_point()
        for value in (inputs, gate_weight, up_weight, down_weight)
    ):
        raise TypeError("SwiGLU calibration tensors must be floating point.")
    if (
        len({inputs.device, gate_weight.device, up_weight.device, down_weight.device})
        != 1
    ):
        raise ValueError("SwiGLU calibration tensors must share one device.")
    return inputs.to(torch.float32)


def choose_swiglu_scales(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    group_size: int = 16,
    candidate_exponents: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    fake_quant_objective: Callable[[torch.Tensor], torch.Tensor | float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Choose Smooth-SwiGLU scales with a channel-grouped error proxy.

    The proxy models the two amplified terms that matter for weight-only
    quantization: up error is weighted by ``SiLU(g)^2`` and down error is
    weighted by the post-Hadamard activation ``h^2``.  A caller may provide a
    ``fake_quant_objective`` to replace the proxy with an actual quantized MLP
    output loss; the callback receives each candidate scale vector.
    """

    inputs = _validate_calibration_geometry(inputs, gate_weight, up_weight, down_weight)
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 1
    ):
        raise ValueError("SwiGLU `group_size` must be a positive integer.")
    if not candidate_exponents or any(
        not math.isfinite(float(value)) for value in candidate_exponents
    ):
        raise ValueError(
            "SwiGLU candidate exponents must be a nonempty finite sequence."
        )
    if any(value <= 0 for value in (scale_min, scale_max)) or scale_min > scale_max:
        raise ValueError("SwiGLU scale bounds must be positive and ordered.")

    with torch.no_grad():
        gate = inputs @ gate_weight.to(torch.float32).transpose(0, 1)
        up = inputs @ up_weight.to(torch.float32).transpose(0, 1)
        silu_gate = F.silu(gate)
        hidden = silu_gate * up
        down_energy = down_weight.to(torch.float32).square().sum(dim=0)
        # A and B are the leading-order up/down error weights.  The fourth
        # root of their group-sum ratio minimizes the shared-scale proxy.
        up_error_weight = silu_gate.square().mean(dim=0) * up_weight.to(
            torch.float32
        ).square().mean(dim=1)
        down_error_weight = hidden.square().mean(dim=0) * down_energy
        epsilon = torch.finfo(torch.float32).eps
        scales = torch.ones_like(up_error_weight)
        proxy_objective = torch.zeros((), device=scales.device, dtype=torch.float32)
        for start in range(0, up_error_weight.numel(), group_size):
            stop = min(start + group_size, up_error_weight.numel())
            # For one scale shared by a group, the proxy is
            #
            #   L_G(s) = s^2 * sum(A_i) + s^-2 * sum(B_i).
            #
            # Its unconstrained minimizer is the fourth root of the ratio of
            # the group sums.  Averaging per-channel optima (especially in
            # log space) is not equivalent when a few channels dominate.
            group_a = up_error_weight[start:stop].sum()
            group_b = down_error_weight[start:stop].sum()
            group_base = (group_b.clamp_min(epsilon) / group_a.clamp_min(epsilon)).pow(0.25)
            group_base = group_base.clamp(min=scale_min, max=scale_max)
            candidates = torch.as_tensor(
                [
                    group_base.item() * (2.0 ** float(exponent))
                    for exponent in candidate_exponents
                ],
                device=scales.device,
                dtype=torch.float32,
            ).clamp(min=scale_min, max=scale_max)
            if fake_quant_objective is None:
                scores = (
                    up_error_weight[start:stop, None] * candidates[None, :].square()
                    + down_error_weight[start:stop, None] / candidates[None, :].square()
                ).sum(dim=0)
                selected = int(scores.argmin().item())
                proxy_objective += scores[selected]
            else:
                score_values = []
                for candidate in candidates:
                    candidate_scales = torch.ones_like(scales)
                    candidate_scales[start:stop] = candidate
                    score = fake_quant_objective(candidate_scales)
                    score_values.append(
                        torch.as_tensor(score, device=scales.device, dtype=torch.float32)
                    )
                scores = torch.stack(score_values)
                selected = int(scores.argmin().item())
                proxy_objective += scores[selected]
            scales[start:stop] = candidates[selected]

        salience = swiglu_jacobian_salience(gate, up, down_weight)
        stats = {
            "tokens": int(inputs.shape[0]),
            "intermediate_channels": int(scales.numel()),
            "group_size": group_size,
            "groups": math.ceil(scales.numel() / group_size),
            "scale_min": float(scales.min().item()),
            "scale_max": float(scales.max().item()),
            "scale_geomean": float(scales.clamp_min(epsilon).log().mean().exp().item()),
            "proxy_objective": float(proxy_objective.item()),
            "gate_salience_sum": float(salience["gate"].sum().item()),
            "up_salience_sum": float(salience["up"].sum().item()),
            "down_salience_sum": float(salience["down"].sum().item()),
        }
    return scales, stats


def swiglu_jacobian_salience(
    gate: torch.Tensor,
    up: torch.Tensor,
    down_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return gate/up/down channel salience from the SwiGLU Jacobian."""

    if gate.shape != up.shape or gate.ndim < 2:
        raise ValueError("SwiGLU gate and up activations must have matching rank >= 2.")
    if down_weight.ndim != 2 or down_weight.shape[1] != gate.shape[-1]:
        raise ValueError("SwiGLU down weight must be [output, intermediate].")
    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    down_energy = down_weight.to(torch.float32).square().sum(dim=0)
    hidden = F.silu(gate) * up
    gate_sensitivity = (
        (up * silu_derivative(gate)).square().mean(dim=tuple(range(gate.ndim - 1)))
    )
    up_sensitivity = F.silu(gate).square().mean(dim=tuple(range(gate.ndim - 1)))
    down_sensitivity = hidden.square().mean(dim=tuple(range(hidden.ndim - 1)))
    return {
        "gate": gate_sensitivity * down_energy,
        "up": up_sensitivity * down_energy,
        "down": down_sensitivity,
    }


def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
    flattened = values.detach().to(torch.float32).flatten()
    if flattened.numel() == 0:
        return {
            name: 0.0
            for name in (
                "mean",
                "mean_abs",
                "min",
                "p01",
                "p05",
                "p50",
                "p95",
                "p99",
                "max",
                "negative_fraction",
                "positive_fraction",
            )
        }
    quantiles = torch.quantile(
        flattened,
        torch.tensor((0.01, 0.05, 0.5, 0.95, 0.99), device=flattened.device),
    )
    return {
        "mean": float(flattened.mean().item()),
        "mean_abs": float(flattened.abs().mean().item()),
        "min": float(flattened.min().item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "max": float(flattened.max().item()),
        "negative_fraction": float((flattened < 0).to(torch.float32).mean().item()),
        "positive_fraction": float((flattened > 0).to(torch.float32).mean().item()),
    }


def swiglu_error_diagnostics(
    dense_gate: torch.Tensor,
    dense_up: torch.Tensor,
    quantized_gate: torch.Tensor,
    quantized_up: torch.Tensor,
    dense_down: torch.Tensor,
    quantized_down: torch.Tensor,
) -> dict[str, Any]:
    """Measure gate/up, post-SwiGLU, and down-projection amplification."""

    if dense_gate.shape != dense_up.shape or dense_gate.shape != quantized_gate.shape:
        raise ValueError("Dense and quantized gate/up activation shapes must match.")
    if quantized_up.shape != dense_up.shape:
        raise ValueError("Dense and quantized up activation shapes must match.")
    if dense_down.ndim != 2 or quantized_down.shape != dense_down.shape:
        raise ValueError(
            "Dense and quantized down weights must have matching rank-2 shapes."
        )
    dense_h = F.silu(dense_gate) * dense_up
    quantized_h = F.silu(quantized_gate) * quantized_up
    dense_y = dense_h @ dense_down.to(torch.float32).transpose(0, 1)
    quantized_y = quantized_h @ quantized_down.to(torch.float32).transpose(0, 1)
    gate_error = (
        (quantized_gate.to(torch.float32) - dense_gate.to(torch.float32))
        .square()
        .sum(dim=-1)
    )
    up_error = (
        (quantized_up.to(torch.float32) - dense_up.to(torch.float32))
        .square()
        .sum(dim=-1)
    )
    h_error = (quantized_h - dense_h).square().sum(dim=-1)
    y_error = (quantized_y - dense_y).square().sum(dim=-1)
    eps = torch.finfo(torch.float32).eps
    result = {
        "gate_relative_error": _distribution_summary(
            gate_error
            / dense_gate.to(torch.float32).square().sum(dim=-1).clamp_min(eps)
        ),
        "up_relative_error": _distribution_summary(
            up_error / dense_up.to(torch.float32).square().sum(dim=-1).clamp_min(eps)
        ),
        "post_swiglu_relative_error": _distribution_summary(
            h_error / dense_h.square().sum(dim=-1).clamp_min(eps)
        ),
        "down_output_relative_error": _distribution_summary(
            y_error / dense_y.square().sum(dim=-1).clamp_min(eps)
        ),
        "A_swiglu": _distribution_summary(
            h_error / (gate_error + up_error).clamp_min(eps)
        ),
        "A_down": _distribution_summary(y_error / h_error.clamp_min(eps)),
    }
    result["cross_error"] = _distribution_summary(
        2.0
        * (
            dense_up.to(torch.float32)
            * silu_derivative(dense_gate.to(torch.float32))
            * (quantized_gate.to(torch.float32) - dense_gate.to(torch.float32))
            * F.silu(dense_gate.to(torch.float32))
            * (quantized_up.to(torch.float32) - dense_up.to(torch.float32))
        ).sum(dim=-1)
    )
    return result


def select_swiglu_candidate_triplet(
    inputs: torch.Tensor,
    dense_gate: torch.Tensor,
    dense_up: torch.Tensor,
    dense_down: torch.Tensor,
    gate_candidates: tuple[torch.Tensor, ...] | list[torch.Tensor],
    up_candidates: tuple[torch.Tensor, ...] | list[torch.Tensor],
    down_candidates: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    beam_size: int = 4,
) -> dict[str, Any]:
    """Select QVQ gate/up/down candidates as one nonlinear MLP.

    Candidate tensors are reconstructed dense weights, not packed payloads.
    The QVQ caller can therefore generate a small bank of exact candidates,
    call this selector, and commit the corresponding serialized payloads.
    Every gate/up/down combination is evaluated before pruning.  This keeps a
    down candidate in the search even when its best partner is not the best
    pair under the dense down projection, which is necessary to preserve
    nonlinear error cancellation opportunities.
    """

    if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
        raise ValueError("SwiGLU candidate `beam_size` must be a positive integer.")
    inputs = _validate_calibration_geometry(inputs, dense_gate, dense_up, dense_down)
    if not gate_candidates or not up_candidates or not down_candidates:
        raise ValueError("SwiGLU candidate banks must be nonempty.")
    expected_gate = dense_gate.shape
    expected_down = dense_down.shape
    for name, candidates, expected in (
        ("gate", gate_candidates, expected_gate),
        ("up", up_candidates, expected_gate),
        ("down", down_candidates, expected_down),
    ):
        if any(candidate.shape != expected for candidate in candidates):
            raise ValueError(
                f"SwiGLU {name} candidates do not match the dense weight geometry."
            )
        if any(candidate.device != dense_gate.device for candidate in candidates):
            raise ValueError("SwiGLU candidate weights must share one device.")

    with torch.no_grad():
        target = (
            F.silu(inputs @ dense_gate.float().transpose(0, 1))
            * (inputs @ dense_up.float().transpose(0, 1))
        ) @ dense_down.float().transpose(0, 1)
        triplets = []
        gate_outputs = [
            inputs @ gate_weight.float().transpose(0, 1)
            for gate_weight in gate_candidates
        ]
        up_outputs = [
            inputs @ up_weight.float().transpose(0, 1)
            for up_weight in up_candidates
        ]
        for gate_index, gate_output in enumerate(gate_outputs):
            for up_index, up_output in enumerate(up_outputs):
                hidden = F.silu(gate_output) * up_output
                for down_index, down_weight in enumerate(down_candidates):
                    output = hidden @ down_weight.float().transpose(0, 1)
                    loss = (output - target).square().mean()
                    triplets.append((loss, gate_index, up_index, down_index))
        triplets.sort(key=lambda item: float(item[0].item()))
        best_loss, gate_index, up_index, down_index = triplets[0]
        return {
            "gate_index": int(gate_index),
            "up_index": int(up_index),
            "down_index": int(down_index),
            "loss": float(best_loss.item()),
            "evaluated_triplets": len(triplets),
            "beam": [
                {
                    "gate_index": int(item[1]),
                    "up_index": int(item[2]),
                    "down_index": int(item[3]),
                    "loss": float(item[0].item()),
                }
                for item in triplets[: min(beam_size, len(triplets))]
            ],
        }
