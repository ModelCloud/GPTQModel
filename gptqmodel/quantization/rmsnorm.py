# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Post-quantization RMSNorm gain tuning (norm-only knowledge distillation).

Only the gains of the given normalization modules are trained so the
quantized model's next-token distribution matches an unquantized teacher.
Every packed quantized tensor is left untouched, so the result is a drop-in
checkpoint for any kernel that can run the source checkpoint.
Related method: Norm Tweaking, https://arxiv.org/abs/2309.02784.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.logger import setup_logger
from .config import FORMAT, RMSNormConfig


log = setup_logger()


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def _logits(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits


def _resolve_gains(model: nn.Module, module_names: List[str], skip) -> Dict[str, nn.Parameter]:
    gains = {}
    for name in module_names:
        if any(name.endswith(suffix) for suffix in skip):
            continue
        weight = getattr(model.get_submodule(name), "weight", None)
        if not isinstance(weight, nn.Parameter) or weight.dim() != 1 or not weight.is_floating_point():
            raise ValueError(f"RMSNorm tuning: `{name}` has no 1-D floating point `weight` gain.")
        gains[name] = weight
    if not gains:
        raise ValueError("RMSNorm tuning: no RMSNorm modules selected; tag them with `:rmsnorm` in the model definition.")
    return gains


def validate_tuning_model(model: nn.Module) -> None:
    """Reject packing/offload state before changing gains, modes or caches.

    Packing produces checkpoint-format zero points, while eval-mode TorchLinear
    consumes v2 zero points. The checkpoint loader owns that conversion. Fresh
    quantization can also leave meta or inference tensors behind; do not mutate
    their storage/ownership here merely to make backward possible.
    """
    from ..nn_modules.qlinear.torch import TorchLinear

    recovery = "Save the quantized model, then reload it with `backend=BACKEND.TORCH` before RMSNorm tuning."
    for name, tensor in model.named_parameters():
        if tensor.is_meta or torch.is_inference(tensor):
            raise ValueError(f"RMSNorm tuning: `{name}` is an offloaded/meta or inference tensor. {recovery}")
    for name, tensor in model.named_buffers():
        if tensor.is_meta:
            raise ValueError(f"RMSNorm tuning: `{name}` is an offloaded/meta tensor. {recovery}")
    for name, module in model.named_modules():
        # Native v2/planar modules need no conversion and may retain the default
        # qzero_format marker (1); it records conversion, not the export format.
        if isinstance(module, TorchLinear) and module.format == FORMAT.GPTQ and module.qzero_format() != 2:
            raise ValueError(f"RMSNorm tuning: `{name}` has checkpoint-format GPTQ-v1 zero points. {recovery}")


def _prepare_runtime_for_tuning(model: nn.Module) -> list[tuple[nn.Module, bool, bool, bool, int, Any]]:
    """Drop inference-only quantized linear caches before enabling autograd."""
    from ..nn_modules.qlinear.torch import TorchLinear

    states = []
    for module in model.modules():
        if not isinstance(module, TorchLinear):
            continue
        states.append(
            (
                module,
                module._cache_enabled,
                module._lookahead_enabled,
                module._streaming_enabled,
                module._stream_tile_cols,
                module._lookahead_next,
            )
        )
        module.clear_weight_cache()
        module._stream_reset_cache()
        module._reset_prefetch_state()
        module.enable_weight_cache(False)
        module.enable_lookahead(False)
        module.enable_streaming(False)
    return states


def _restore_runtime_after_tuning(states) -> None:
    for module, cache_enabled, lookahead_enabled, streaming_enabled, tile_cols, lookahead_next in states:
        module.enable_weight_cache(cache_enabled)
        module.enable_streaming(streaming_enabled, tile_cols=tile_cols)
        module.enable_lookahead(lookahead_enabled)
        module.set_lookahead_next(lookahead_next)


def tune_rmsnorm_gains(
    model: nn.Module,
    teacher: nn.Module,
    batches: List[Dict[str, torch.Tensor]],
    module_names: List[str],
    config: RMSNormConfig,
) -> Dict[str, Any]:
    """Distill ``teacher`` logits into ``model`` by training only the named norm gains.

    ``model`` must run with a kernel that supports autograd through its
    quantized linears (e.g. ``BACKEND.TORCH``). Gains are optimized as FP32
    master copies and written back in their native dtype after every step.
    """
    if not batches:
        raise ValueError("RMSNorm tuning requires calibration data.")

    validate_tuning_model(model)
    runtime_states = _prepare_runtime_for_tuning(model)
    trainable = {param: param.requires_grad for param in model.parameters()}
    model_training, teacher_training = model.training, teacher.training
    # Kernels such as TorchQuantLinear switch packed layouts in train mode;
    # stay in eval mode, autograd still flows through the dequantized matmuls.
    params = []
    try:
        with torch.inference_mode(False), torch.enable_grad():
            gains = _resolve_gains(model, module_names, config.skip)
            names = list(gains)
            params = [gains[name] for name in names]
            for param in params:
                param.grad = None
            base = [param.detach().float().clone() for param in params]
            scale = [b.abs().mean().clamp_min(1e-6) for b in base]
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
                    for m, b, s in zip(master, base, scale)
                )
                (config.reg * penalty).backward()

                if step == 1:
                    missing = [name for name, param in zip(names, params) if param.grad is None]
                    if missing:
                        raise RuntimeError(
                            f"RMSNorm tuning: no gradient reached {missing[:3]}. Load the quantized model with a "
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
                        f"RMSNorm tuning: step {step}/{config.steps} kl={entry['kl']:.5f} "
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
        log.warn(f"RMSNorm tuning: skipped {skipped} step(s) with non-finite gradients.")
    return {"modules": names, "steps": config.steps, "skipped_steps": skipped, "history": history}


__all__ = ["tune_rmsnorm_gains"]
