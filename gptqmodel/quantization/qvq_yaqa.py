# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full-model real-Fisher collection for QVQ's YAQA-v3 quantizer."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

YAQA_PAPER_REGULARIZATION = 1e-4
YAQA_PAPER_MINIMUM_SEQUENCES = 2_000
YAQA_PAPER_RECOMMENDED_SEQUENCES = 65_536

_YAQA_CHECKPOINT_RECOMPUTING: ContextVar[bool] = ContextVar(
    "yaqa_checkpoint_recomputing",
    default=False,
)
_MISSING_FORWARD = object()


@contextmanager
def _yaqa_recomputation_context() -> Iterator[None]:
    token = _YAQA_CHECKPOINT_RECOMPUTING.set(True)
    try:
        yield
    finally:
        _YAQA_CHECKPOINT_RECOMPUTING.reset(token)


def _yaqa_checkpoint_contexts():
    return nullcontext(), _yaqa_recomputation_context()


@contextmanager
def _checkpoint_module_forwards(modules: Sequence[nn.Module]) -> Iterator[None]:
    """Checkpoint module bodies without replacing model-tree objects.

    YAQA needs the original target-module identities for its gradient hooks.
    Temporarily wrapping each decoder ``forward`` keeps those identities stable,
    and the exact prior instance state is restored even when forward/backward
    raises. Non-reentrant checkpointing matches the official YAQA collector.
    """

    checkpoint_modules = tuple(modules)
    if len({id(module) for module in checkpoint_modules}) != len(checkpoint_modules):
        raise ValueError("YAQA activation-checkpoint modules must be unique")
    original_forwards = []
    try:
        for module in checkpoint_modules:
            if not isinstance(module, nn.Module):
                raise TypeError("YAQA activation-checkpoint targets must be modules")
            raw_forward = module.__dict__.get("forward", _MISSING_FORWARD)
            forward = module.forward
            if not callable(forward):
                raise TypeError("YAQA activation-checkpoint target forward must be callable")
            original_forwards.append((module, raw_forward))

            @wraps(forward)
            def checkpointed_forward(*args, _forward=forward, **kwargs):
                return checkpoint(
                    _forward,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    context_fn=_yaqa_checkpoint_contexts,
                    **kwargs,
                )

            module.forward = checkpointed_forward
        yield
    finally:
        for module, raw_forward in reversed(original_forwards):
            if raw_forward is _MISSING_FORWARD:
                module.__dict__.pop("forward", None)
            else:
                module.__dict__["forward"] = raw_forward


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    raise TypeError(f"Cannot extract a tensor from {type(value)!r}")


def _select_valid_token_rows(tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError(f"expected a [batch, sequence, ...] tensor, got shape {tuple(tensor.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(f"expected a rank-2 attention mask, got shape {tuple(attention_mask.shape)}")
    if tuple(tensor.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError(
            f"attention mask shape {tuple(attention_mask.shape)} does not match tensor token geometry "
            f"{tuple(tensor.shape[:2])}"
        )
    keep = attention_mask.to(device=tensor.device, dtype=torch.bool)
    if not bool(keep.any()):
        raise ValueError("attention mask contains no valid tokens")
    return tensor[keep]


def yaqa_real_fisher_loss(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Sample the model distribution and return the sum of sequence score losses.

    The YAQA Sketch-B sample is one complete sequence, not one token.  Its
    score is therefore the sum of the valid-token log probabilities.  Summing
    those sequence scores in a batched backward preserves the same distinct
    per-sequence gradients as separate backwards and is invariant to how the
    sequences are grouped into batches.  On the paper's fixed-length contexts
    this differs from the reference implementation's global token mean only
    by one common scalar, which does not change the YAQA feedback factors.
    """

    valid_logits = _select_valid_token_rows(logits, attention_mask)
    if not torch.isfinite(valid_logits).all():
        raise ValueError("YAQA full-model logits must contain only finite values")
    probabilities = F.softmax(valid_logits.detach().float(), dim=-1)
    sampled_tokens = torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)
    loss = F.cross_entropy(valid_logits.float(), sampled_tokens, reduction="sum")
    if not torch.isfinite(loss):
        raise ValueError("YAQA full-model score loss overflowed")
    return loss, sampled_tokens.numel()


def _default_first_decoder_layer(model: nn.Module) -> nn.Module:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("YAQA Sketch B requires at least one decoder layer")
    return layers[0]


def capture_yaqa_sketch_b(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    modules: dict[str, nn.Linear],
    *,
    device: torch.device,
    seed: int = 0,
    minimum_sequences: int = 1,
    first_decoder_layer: nn.Module | None = None,
    checkpoint_modules: Sequence[nn.Module] = (),
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    accumulator_device: torch.device | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Collect exact per-sequence YAQA Sketch-B factors from full-model score gradients."""

    if model.training:
        raise ValueError("YAQA Sketch B requires the full model to be in eval mode")
    if not modules:
        raise ValueError("YAQA Sketch B requires at least one target linear module")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("YAQA Sketch B seed must be an integer")
    if isinstance(minimum_sequences, bool) or not isinstance(minimum_sequences, int) or minimum_sequences < 1:
        raise ValueError("YAQA Sketch B minimum sequence count must be a positive integer")
    if any(not isinstance(module, nn.Linear) for module in modules.values()):
        raise TypeError("YAQA Sketch B targets must all be linear modules")
    if len({id(module) for module in modules.values()}) != len(modules):
        raise ValueError("YAQA Sketch B target modules must be unique")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("YAQA Sketch B progress callback must be callable")
    if accumulator_device is not None:
        accumulator_device = torch.device(accumulator_device)
        if accumulator_device.type not in {"cpu", "cuda"}:
            raise ValueError("YAQA Sketch-B accumulators support CPU or CUDA storage")
        if accumulator_device.type == "cuda" and device.type != "cuda":
            raise ValueError("YAQA CUDA accumulators require CUDA collection")
    if first_decoder_layer is None:
        first_decoder_layer = _default_first_decoder_layer(model)
    if not isinstance(first_decoder_layer, nn.Module):
        raise TypeError("YAQA Sketch B first decoder layer must be a module")

    available_sequences = 0
    for batch in batches:
        attention_mask = batch.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            # Missing attention_mask is a malformed calibration batch, matching the lifecycle's validation contract.
            raise ValueError("every YAQA calibration batch must include attention_mask as a tensor")  # noqa: TRY004
        if attention_mask.ndim != 2:
            raise ValueError("YAQA attention masks must be rank-2")
        valid_by_sequence = attention_mask.ne(0).sum(dim=1)
        if not bool(valid_by_sequence.gt(0).all()):
            raise ValueError("every YAQA calibration sequence must contain at least one valid token")
        available_sequences += attention_mask.shape[0]
    if available_sequences < minimum_sequences:
        if available_sequences == 0:
            raise ValueError(
                "YAQA Sketch B observed no independent calibration sequences "
                f"(configured minimum: {minimum_sequences})"
            )
        raise ValueError(
            "YAQA Sketch B calibration has "
            f"{available_sequences} independent sequences, below the configured minimum of {minimum_sequences}; "
            "add independent calibration sequences or explicitly reduce the minimum for a diagnostic-only run"
        )

    factor_bytes = sum(
        (module.in_features * module.in_features + module.out_features * module.out_features) * 4
        for module in modules.values()
    )
    if accumulator_device is None:
        accumulator_device = torch.device("cpu")
        if device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            reserve_bytes = max(16 * 1024**3, total_bytes // 4)
            if factor_bytes <= max(0, free_bytes - reserve_bytes):
                accumulator_device = device
    input_accumulators: dict[str, torch.Tensor] = {}
    output_accumulators: dict[str, torch.Tensor] = {}
    sequence_counts = dict.fromkeys(modules, 0)
    total_sequences = 0
    total_valid_tokens = 0
    active_mask: torch.Tensor | None = None
    active_calls: set[str] = set()
    tensor_hook_handles = []
    module_hook_handles = []
    parameters = tuple(model.parameters())
    parameter_requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    generator = torch.Generator(device=device).manual_seed(seed)
    nonfinite_update = (
        torch.zeros((), dtype=torch.bool, device=device) if accumulator_device.type == "cuda" else None
    )
    capture_started = time.perf_counter()
    capture_start_event = capture_end_event = None
    if device.type == "cuda":
        capture_start_event = torch.cuda.Event(enable_timing=True)
        capture_end_event = torch.cuda.Event(enable_timing=True)
        capture_start_event.record()

    def seed_decoder_gradient(_module, args, kwargs):
        if args:
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("YAQA decoder's first positional input must be a tensor")
            return (hidden.detach().requires_grad_(True), *args[1:]), kwargs
        if "hidden_states" not in kwargs or not isinstance(kwargs["hidden_states"], torch.Tensor):
            raise ValueError("YAQA could not locate the first decoder layer's hidden states")
        updated_kwargs = dict(kwargs)
        updated_kwargs["hidden_states"] = kwargs["hidden_states"].detach().requires_grad_(True)
        return args, updated_kwargs

    def accumulate_gradient(module_name: str, activation: torch.Tensor, gradient: torch.Tensor) -> None:
        if gradient.ndim != 3:
            raise ValueError(
                f"YAQA module {module_name} output gradient must have [batch, sequence, channels] geometry"
            )
        if tuple(gradient.shape[:2]) != tuple(activation.shape[:2]):
            raise ValueError(f"YAQA module {module_name} input/output token geometry does not match")
        if active_mask is None or tuple(active_mask.shape) != tuple(activation.shape[:2]):
            raise ValueError(f"YAQA module {module_name} attention-mask geometry does not match")

        keep = active_mask.to(device=gradient.device, dtype=torch.bool).unsqueeze(-1)
        gradient = torch.where(keep, gradient.detach().float(), 0)
        activation = torch.where(keep, activation.float(), 0)
        per_sequence_gradient = torch.bmm(gradient.transpose(1, 2), activation)
        if not torch.isfinite(per_sequence_gradient).all():
            raise ValueError(f"YAQA module {module_name} produced a non-finite full-model weight gradient")
        input_update = torch.bmm(per_sequence_gradient.transpose(1, 2), per_sequence_gradient).sum(dim=0)
        output_update = torch.bmm(per_sequence_gradient, per_sequence_gradient.transpose(1, 2)).sum(dim=0)
        if nonfinite_update is None:
            if not torch.isfinite(input_update).all() or not torch.isfinite(output_update).all():
                raise ValueError(f"YAQA module {module_name} produced an overflowing Sketch-B Gram update")
        else:
            nonfinite_update.logical_or_(~torch.isfinite(input_update).all())
            nonfinite_update.logical_or_(~torch.isfinite(output_update).all())
        input_update = input_update.detach().to(device=accumulator_device)
        output_update = output_update.detach().to(device=accumulator_device)
        if module_name in input_accumulators:
            input_accumulators[module_name].add_(input_update)
            output_accumulators[module_name].add_(output_update)
        else:
            input_accumulators[module_name] = input_update.contiguous()
            output_accumulators[module_name] = output_update.contiguous()
        sequence_counts[module_name] += per_sequence_gradient.shape[0]

    for name, module in modules.items():

        def module_hook(_module, args, output, module_name=name):
            if _YAQA_CHECKPOINT_RECOMPUTING.get():
                return
            if active_mask is None:
                raise RuntimeError("YAQA target module ran without an active attention mask")
            if module_name in active_calls:
                raise ValueError(
                    f"YAQA target module {module_name} was reused within one forward pass; "
                    "shared-module gradients must be combined before forming Sketch B"
                )
            active_calls.add(module_name)
            activation = _first_tensor(args).detach()
            output_tensor = _first_tensor(output)
            if activation.ndim != 3:
                raise ValueError(f"YAQA module {module_name} input must have [batch, sequence, channels] geometry")
            if not output_tensor.requires_grad:
                raise RuntimeError(f"YAQA module {module_name} output is not connected to the full-model loss")
            handle = output_tensor.register_hook(
                lambda gradient, module_name=module_name, activation=activation: accumulate_gradient(
                    module_name, activation, gradient
                )
            )
            tensor_hook_handles.append(handle)

        module_hook_handles.append(module.register_forward_hook(module_hook))

    decoder_seed_handle = first_decoder_layer.register_forward_pre_hook(seed_decoder_gradient, with_kwargs=True)
    previous_fp32_precision = None
    previous_allow_tf32 = None
    cuda_matmul = torch.backends.cuda.matmul
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        if device.type == "cuda":
            if hasattr(cuda_matmul, "fp32_precision"):
                previous_fp32_precision = cuda_matmul.fp32_precision
                cuda_matmul.fp32_precision = "ieee"
            else:  # pragma: no cover - older PyTorch
                previous_allow_tf32 = cuda_matmul.allow_tf32
                cuda_matmul.allow_tf32 = False

        with _checkpoint_module_forwards(checkpoint_modules):
            for batch_index, batch in enumerate(batches, start=1):
                encoded = {name: value.to(device) for name, value in batch.items() if name != "labels"}
                active_mask = encoded["attention_mask"]
                if active_mask.ndim != 2:
                    raise ValueError("YAQA attention masks must be rank-2")
                valid_by_sequence = active_mask.ne(0).sum(dim=1)
                if not bool(valid_by_sequence.gt(0).all()):
                    raise ValueError("every YAQA calibration sequence must contain at least one valid token")
                active_calls.clear()
                outputs = model(**encoded, use_cache=False)
                logits = getattr(outputs, "logits", None)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("YAQA full-model forward must return tensor logits")
                loss, valid_tokens = yaqa_real_fisher_loss(logits, active_mask, generator=generator)
                loss.backward()
                missing = set(modules) - active_calls
                if missing:
                    raise ValueError(f"YAQA full-model forward did not execute target modules {sorted(missing)}")
                batch_sequences = active_mask.shape[0]
                total_sequences += batch_sequences
                total_valid_tokens += valid_tokens
                if progress_callback is not None:
                    progress_callback(
                        {
                            "completed_batches": batch_index,
                            "total_batches": len(batches),
                            "completed_sequences": total_sequences,
                            "valid_tokens": total_valid_tokens,
                        }
                    )
                for handle in tensor_hook_handles:
                    handle.remove()
                tensor_hook_handles.clear()
                active_mask = None
                # Explicitly drop the autograd graph before the next sequence.
                # MPS command buffers are asynchronous and otherwise keep the
                # logits/loss graph live until allocator pressure forces a flush.
                del outputs, logits, loss
                gc.collect()
                if device.type == "mps":
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
    finally:
        active_mask = None
        decoder_seed_handle.remove()
        for handle in tensor_hook_handles:
            handle.remove()
        for handle in module_hook_handles:
            handle.remove()
        for parameter, requires_grad in zip(parameters, parameter_requires_grad):
            parameter.requires_grad_(requires_grad)
        if previous_fp32_precision is not None:
            cuda_matmul.fp32_precision = previous_fp32_precision
        if previous_allow_tf32 is not None:
            cuda_matmul.allow_tf32 = previous_allow_tf32

    if total_sequences < 1:
        raise ValueError("YAQA Sketch B observed no independent calibration sequences")
    for name in modules:
        if sequence_counts[name] != total_sequences or name not in input_accumulators:
            raise ValueError(f"YAQA module {name} did not produce one gradient for every sequence")

    if capture_end_event is not None:
        capture_end_event.record()
    transfer_started = time.perf_counter()
    if nonfinite_update is not None and bool(nonfinite_update):
        raise ValueError("YAQA produced a non-finite Sketch-B Gram update")
    for name in modules:
        input_accumulators[name] = input_accumulators[name].to(device="cpu")
        output_accumulators[name] = output_accumulators[name].to(device="cpu")
        if not torch.isfinite(input_accumulators[name]).all() or not torch.isfinite(
            output_accumulators[name]
        ).all():
            raise ValueError(f"YAQA module {name} produced an overflowing Sketch-B accumulator")
    transfer_seconds = time.perf_counter() - transfer_started
    capture_cuda_ms = (
        None if capture_start_event is None else capture_start_event.elapsed_time(capture_end_event)
    )

    input_hessians: dict[str, torch.Tensor] = {}
    output_hessians: dict[str, torch.Tensor] = {}
    for name, module in modules.items():
        if sequence_counts[name] != total_sequences or name not in input_accumulators:
            raise ValueError(f"YAQA module {name} did not produce one gradient for every sequence")
        out_features, in_features = module.weight.shape
        input_hessians[name] = input_accumulators[name].div(total_sequences * out_features).contiguous()
        output_hessians[name] = output_accumulators[name].div(total_sequences * in_features).contiguous()

    input_factor_elements = sum(factor.numel() for factor in input_hessians.values())
    output_factor_elements = sum(factor.numel() for factor in output_hessians.values())
    return (
        input_hessians,
        output_hessians,
        {
            "method": "YAQA-v3 Sketch B real Fisher",
            "full_model_backward": True,
            "independent_sequences": total_sequences,
            "valid_output_samples": total_valid_tokens,
            "monte_carlo_samples_per_output": 1,
            "sequence_loss_reduction": "per_sequence_token_sum",
            "activation_checkpointing": bool(checkpoint_modules),
            "checkpointed_modules": len(checkpoint_modules),
            "accumulator_device": accumulator_device.type,
            "accumulator_bytes": factor_bytes,
            "capture_wall_seconds": time.perf_counter() - capture_started,
            "capture_cuda_ms": capture_cuda_ms,
            "final_host_transfer_seconds": transfer_seconds,
            "minimum_sequences": minimum_sequences,
            "factor_dtype": "float32",
            "input_factor_elements": input_factor_elements,
            "output_factor_elements": output_factor_elements,
            "factor_storage_bytes": (input_factor_elements + output_factor_elements) * 4,
            "tf32": False,
            "seed": seed,
        },
    )


__all__ = [
    "YAQA_PAPER_MINIMUM_SEQUENCES",
    "YAQA_PAPER_RECOMMENDED_SEQUENCES",
    "YAQA_PAPER_REGULARIZATION",
    "capture_yaqa_sketch_b",
    "yaqa_real_fisher_loss",
]
