# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Post-quantization normalization tuning (norm-only knowledge distillation).

Only the parameters of the selected normalization modules are trained so the
quantized model's next-token distribution matches an unquantized teacher.
Every packed quantized tensor is left untouched, so the result is a drop-in
checkpoint for any kernel that can run the source checkpoint.
Related method: Norm Tweaking, https://arxiv.org/abs/2309.02784.

The tuner is parameterization-driven: a handler probes each selected module to
learn how its parameters produce the effective normalization, so families such
as Gemma-style ``1 + weight`` gains and LayerNorm weight+bias pairs train with
the same loop instead of class-name special cases.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.logger import setup_logger
from .config import FORMAT, NormConfig, NormType, normalize_norm_types


log = setup_logger()

# Probe with small, known gains so this bound only absorbs serving-dtype
# rounding. It must not grow with the checkpoint's gain or bias magnitude.
_PROBE_TOLERANCE = 0.05


@dataclass(frozen=True)
class NormParam:
    """One trainable normalization parameter resolved from a live module."""

    module: str
    kind: str
    param: nn.Parameter
    # Effective normalization value is ``param + offset`` (0.0 for plain gains).
    offset: float
    # Regularization scale: the mean absolute effective value of the parameter.
    scale: float

    @property
    def label(self) -> str:
        return f"{self.module}.{self.kind}"


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def _logits(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits


def norm_type_of(module: nn.Module) -> Optional[NormType]:
    """Normalization family of ``module``, or ``None`` when it is not a norm.

    Definitions tag norm modules with ``:norm``; this classifies the live
    module so ``NormConfig.types`` can filter what actually gets trained.
    """
    if isinstance(module, nn.RMSNorm):
        return NormType.RMSNORM
    if isinstance(module, nn.LayerNorm):
        return NormType.LAYERNORM
    name = type(module).__name__
    if name.endswith("RMSNorm"):
        return NormType.RMSNORM
    if name.endswith("LayerNorm"):
        return NormType.LAYERNORM
    return None


def _probe_gain_offset(module: nn.Module, name: str) -> float:
    """Return the gain offset applied by ``module``: ``0.0`` for ``weight``.

    Evaluate both hypotheses on a private copy at two small stored gains.
    This avoids dividing by zero-valued probe coordinates, cancellation from
    large biases, and ambiguous rounding of large checkpoint gains. Recheck
    every instance: the same class can configure plain or offset gains.
    """
    weight = getattr(module, "weight", None)
    if not isinstance(weight, nn.Parameter) or not weight.is_floating_point():
        raise ValueError(f"Norm tuning: `{name}` ({type(module).__name__}) has no floating point `weight`.")

    if weight.ndim == 0 or weight.numel() == 0:
        raise ValueError(f"Norm tuning: `{name}` has an empty or scalar gain shape.")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"Norm tuning: `{name}.weight` contains non-finite values.")

    cls = type(module)
    # Alternating signs also work for odd widths. Normalize over the complete
    # affine shape, as nn.LayerNorm/nn.RMSNorm do, and account for the live eps.
    probe = ((torch.arange(weight.numel(), device=weight.device) % 2) * 2 - 1).to(weight.dtype)
    probe = probe.reshape(weight.shape)
    probe = torch.stack((probe, -probe))
    axes = tuple(range(-weight.ndim, 0))
    unit = probe.float()
    if norm_type_of(module) == NormType.LAYERNORM:
        unit = unit - unit.mean(dim=axes, keepdim=True)
    eps = getattr(module, "eps", getattr(module, "variance_epsilon", 0.0))
    if eps is None:
        eps = torch.finfo(weight.dtype).eps
    unit = unit * torch.rsqrt(unit.square().mean(dim=axes, keepdim=True) + eps)
    errors = {0.0: 0.0, 1.0: 0.0}
    with torch.no_grad():
        try:
            candidate = copy.deepcopy(module).eval()
            if getattr(candidate, "bias", None) is not None:
                candidate.bias.zero_()
        except Exception as exc:  # noqa: BLE001 - report the module, not the raw error
            raise ValueError(
                f"Norm tuning: cannot probe `{name}` ({cls.__name__}) to resolve its gain parameterization: {exc}"
            ) from exc
        for gain in (-0.5, 0.5):
            candidate.weight.fill_(gain)
            try:
                output = candidate(probe)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Norm tuning: cannot probe `{name}` ({cls.__name__}) to resolve its gain parameterization: {exc}"
                ) from exc
            if not isinstance(output, torch.Tensor):
                raise ValueError(f"Norm tuning: `{name}` ({cls.__name__}) did not return a tensor for the gain probe.")
            if output.shape != probe.shape:
                raise ValueError(
                    f"Norm tuning: `{name}` ({cls.__name__}) returned shape {tuple(output.shape)} for a "
                    f"{tuple(probe.shape)} gain probe."
                )
            if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(unit).all()):
                raise ValueError(f"Norm tuning: `{name}` ({cls.__name__}) produced non-finite gain probe values.")
            for offset in errors:
                errors[offset] = max(errors[offset], (output.float() - (gain + offset) * unit).abs().max().item())

    matches = [offset for offset, error in errors.items() if error <= _PROBE_TOLERANCE]
    if len(matches) != 1:
        raise ValueError(
            f"Norm tuning: `{name}` ({cls.__name__}) has an unrecognized gain parameterization or an ambiguous "
            "probe; cannot distinguish `weight` from `1 + weight`. Exclude it with `NormConfig.skip`."
        )

    return matches[0]


def resolve_norm_params(
    model: nn.Module,
    module_names: Sequence[str],
    skip: Sequence[str] = (),
    types: Any = None,
) -> List[NormParam]:
    """Resolve the trainable parameters of every selected normalization module."""
    requested = normalize_norm_types(types)
    resolved: List[NormParam] = []
    resolved_names = set()
    for name in module_names:
        if any(name.endswith(suffix) for suffix in skip):
            continue
        try:
            module = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError(f"Norm tuning: `{name}` is not a module of this model.") from exc
        family = norm_type_of(module)
        if family is None:
            raise ValueError(
                f"Norm tuning: `{name}` ({type(module).__name__}) is not a supported normalization module."
            )
        if family not in requested:
            continue
        # A repeated name would train the same parameter twice and double-count
        # its regularization term.
        if name in resolved_names:
            continue

        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter) or not weight.is_floating_point():
            raise ValueError(f"Norm tuning: `{name}.weight` is not a floating point trainable gain.")
        bias = getattr(module, "bias", None)
        if isinstance(bias, nn.Parameter):
            if not bias.is_floating_point():
                raise ValueError(f"Norm tuning: `{name}.bias` is not a floating point trainable parameter.")
            if bias.shape != weight.shape:
                raise ValueError(
                    f"Norm tuning: `{name}.bias` shape {tuple(bias.shape)} does not match "
                    f"`{name}.weight` shape {tuple(weight.shape)}."
                )
            if not bool(torch.isfinite(bias).all()):
                raise ValueError(f"Norm tuning: `{name}.bias` contains non-finite values.")
        offset = _probe_gain_offset(module, name)
        # The regularization scale is the mean absolute effective gain, so
        # `1 + weight` norms are not pinned by a near-zero stored gain.
        gain_scale = (weight.detach().float() + offset).abs().mean().clamp_min(1e-6)
        resolved.append(NormParam(module=name, kind="weight", param=weight, offset=offset, scale=float(gain_scale)))

        if isinstance(bias, nn.Parameter):
            # A zero-initialized bias has no scale of its own; share the gain's
            # scale so the pull toward the checkpoint value stays meaningful.
            resolved.append(
                NormParam(module=name, kind="bias", param=bias, offset=0.0, scale=float(gain_scale))
            )
        resolved_names.add(name)
    if not resolved:
        raise ValueError(
            "Norm tuning: no normalization modules selected; tag them with `:norm` in the model definition "
            "or widen `NormConfig.types`."
        )
    return resolved


def validate_tuning_model(model: nn.Module) -> None:
    """Reject packing/offload state before changing gains, modes or caches.

    Packing produces checkpoint-format zero points, while eval-mode TorchLinear
    consumes v2 zero points. The checkpoint loader owns that conversion. Fresh
    quantization can also leave meta or inference tensors behind; do not mutate
    their storage/ownership here merely to make backward possible.
    """
    from ..nn_modules.qlinear.torch import TorchLinear

    recovery = "Save the quantized model, then reload it with `backend=BACKEND.TORCH` before norm tuning."
    for name, tensor in model.named_parameters():
        if tensor.is_meta or torch.is_inference(tensor):
            raise ValueError(f"Norm tuning: `{name}` is an offloaded/meta or inference tensor. {recovery}")
    for name, tensor in model.named_buffers():
        if tensor.is_meta:
            raise ValueError(f"Norm tuning: `{name}` is an offloaded/meta tensor. {recovery}")
    for name, module in model.named_modules():
        # Native v2/planar modules need no conversion and may retain the default
        # qzero_format marker (1); it records conversion, not the export format.
        if isinstance(module, TorchLinear) and module.format == FORMAT.GPTQ and module.qzero_format() != 2:
            raise ValueError(f"Norm tuning: `{name}` has checkpoint-format GPTQ-v1 zero points. {recovery}")


def _prepare_runtime_for_tuning(model: nn.Module) -> list[tuple[nn.Module, bool, int]]:
    """Drop inference-only quantized linear caches before enabling autograd."""
    from ..nn_modules.qlinear.torch import TorchLinear

    states = []
    for module in model.modules():
        if not isinstance(module, TorchLinear):
            continue
        states.append(
            (
                module,
                module._streaming_enabled,
                module._stream_tile_cols,
            )
        )
        module.clear_weight_cache()
        module.enable_streaming(False)
    return states


def _restore_runtime_after_tuning(states) -> None:
    for module, streaming_enabled, tile_cols in states:
        module.enable_streaming(streaming_enabled, tile_cols=tile_cols)


def tune_norm_params(
    model: nn.Module,
    teacher: nn.Module,
    batches: List[Dict[str, torch.Tensor]],
    module_names: List[str],
    config: NormConfig,
) -> Dict[str, Any]:
    """Distill ``teacher`` logits into ``model`` by training only the selected norm parameters.

    ``model`` must run with a kernel that supports autograd through its
    quantized linears (e.g. ``BACKEND.TORCH``). Parameters are optimized as FP32
    master copies and written back in their native dtype after every step.
    """
    if not batches:
        raise ValueError("Norm tuning requires calibration data.")

    validate_tuning_model(model)
    runtime_states = _prepare_runtime_for_tuning(model)
    trainable = {param: param.requires_grad for param in model.parameters()}
    model_training, teacher_training = model.training, teacher.training
    # Kernels such as TorchQuantLinear switch packed layouts in train mode;
    # stay in eval mode, autograd still flows through the dequantized matmuls.
    params: List[nn.Parameter] = []
    try:
        with torch.inference_mode(False), torch.enable_grad():
            entries = resolve_norm_params(model, module_names, config.skip, config.types)
            names = []
            for entry in entries:
                if entry.module not in names:
                    names.append(entry.module)
            # Keep all module paths for persistence, but optimize tied tensors
            # only once. Otherwise the second alias consumes a cleared grad.
            unique = {}
            for entry in entries:
                previous = unique.get(entry.param)
                if previous is not None and (previous.offset, previous.scale) != (entry.offset, entry.scale):
                    raise ValueError(f"Norm tuning: `{entry.label}` shares a parameter with incompatible gain scales.")
                unique.setdefault(entry.param, entry)
            train_entries = list(unique.values())
            params = [entry.param for entry in train_entries]
            for param in params:
                param.grad = None
            base = [param.detach().float().clone() for param in params]
            scales = [entry.scale for entry in train_entries]
            master = [b.clone().requires_grad_(True) for b in base]
            optimizer = torch.optim.Adam(master, lr=config.lr)

            model.eval()
            teacher.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            for param in params:
                param.requires_grad_(True)

            # Calibration tensors may have been created under an outer inference scope.
            safe_batches = [
                {
                    key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                for batch in batches
            ]
            student_device = _module_device(model.get_input_embeddings())
            teacher_device = _module_device(teacher.get_input_embeddings())
            rng = random.Random(config.seed)
            torch.manual_seed(config.seed)
            log_every = max(1, config.steps // 10)
            history = []
            skipped = 0
            start = time.time()
            for step in range(1, config.steps + 1):
                batch = safe_batches[rng.randrange(len(safe_batches))]
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    target = F.log_softmax(
                        _logits(teacher, input_ids.to(teacher_device), attention_mask.to(teacher_device)).float(), dim=-1
                    ).to(student_device)
                mask = attention_mask.to(student_device).float()
                student = F.log_softmax(
                    _logits(model, input_ids.to(student_device), attention_mask.to(student_device)).float(), dim=-1
                )
                token_kl = (target.exp() * (target - student)).sum(-1)
                kl = (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
                kl.backward()
                del target, student, token_kl

                penalty = sum(
                    (((m - b) / s).square().mean()).to(kl.device)
                    for m, b, s in zip(master, base, scales)
                )
                (config.reg * penalty).backward()

                if step == 1:
                    missing = [entry.label for entry, param in zip(train_entries, params) if param.grad is None]
                    if missing:
                        raise RuntimeError(
                            f"Norm tuning: no gradient reached {missing[:3]}. Load the quantized model with a "
                            "kernel that supports autograd, e.g. `backend=BACKEND.TORCH`."
                        )

                finite = True
                for m, param in zip(master, params):
                    if param.grad is None or m.grad is None:
                        finite = False
                        continue
                    m.grad.add_(param.grad.detach().float().to(m.device))
                    finite = finite and bool(torch.isfinite(m.grad).all())
                    param.grad = None
                if finite:
                    optimizer.step()
                    with torch.no_grad():
                        for m, param in zip(master, params):
                            param.copy_(m.to(param.dtype))
                else:
                    skipped += 1
                optimizer.zero_grad(set_to_none=False)

                if step == 1 or step % log_every == 0 or step == config.steps:
                    entry = {"step": step, "kl": float(kl.detach()), "penalty": float(penalty.detach())}
                    history.append(entry)
                    log.info(
                        f"Norm tuning: step {step}/{config.steps} kl={entry['kl']:.5f} "
                        f"penalty={entry['penalty']:.5f} ({time.time() - start:.1f}s)"
                    )
    finally:
        for param in params:
            param.grad = None
        for param, flag in trainable.items():
            param.requires_grad_(flag)
        model.train(model_training)
        teacher.train(teacher_training)
        _restore_runtime_after_tuning(runtime_states)

    if skipped:
        log.warn(f"Norm tuning: skipped {skipped} step(s) with non-finite gradients.")
    return {
        "modules": names,
        "types": [norm_type.value for norm_type in config.types],
        "params": [entry.label for entry in entries],
        "steps": config.steps,
        "skipped_steps": skipped,
        "history": history,
    }


__all__ = [
    "NormParam",
    "norm_type_of",
    "resolve_norm_params",
    "tune_norm_params",
    "validate_tuning_model",
]
