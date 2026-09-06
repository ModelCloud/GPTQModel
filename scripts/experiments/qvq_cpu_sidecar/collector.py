# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full-model real-Fisher collection for QVQ's YAQA-v3 quantizer."""

from __future__ import annotations

import gc
import math
import time
import zlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from gptqmodel.quantization.qvq_activation import (
    fake_quantize_qvq_fp8_activation,
    normalize_qvq_fp8_activation_format,
    normalize_qvq_fp8_activation_scale_method,
)
from gptqmodel.quantization.qvq_yaqa_cuda import project as _project_cuda

YAQA_PAPER_REGULARIZATION = 1e-4
YAQA_DEFAULT_REGULARIZATION = 0.05
YAQA_DEFAULT_RATE_REGULARIZATION = (
    (1.0, 0.1),
    (1.5, 0.1),
    (2.0, 0.1),
    (2.5, 0.1),
    (3.0, 0.1),
    (3.5, 0.1),
    (4.0, 0.1),
)
YAQA_PAPER_MINIMUM_SEQUENCES = 2_000
YAQA_PAPER_RECOMMENDED_SEQUENCES = 65_536

_YAQA_CHECKPOINT_RECOMPUTING: ContextVar[bool] = ContextVar(
    "yaqa_checkpoint_recomputing",
    default=False,
)
_MISSING_FORWARD = object()


from gptqmodel.quantization.qvq_yaqa import YaqaGramSketch


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
    token_weights: torch.Tensor | None = None,
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
    token_loss = F.cross_entropy(valid_logits.float(), sampled_tokens, reduction="none")
    if token_weights is not None:
        if token_weights.shape != attention_mask.shape:
            raise ValueError("YAQA token weights must have the same shape as attention_mask")
        valid_weights = token_weights[attention_mask.to(device=token_weights.device, dtype=torch.bool)].to(
            device=token_loss.device, dtype=token_loss.dtype
        )
        if not torch.isfinite(valid_weights).all() or bool(valid_weights.le(0).any()):
            raise ValueError("YAQA token weights must be finite and positive on valid tokens")
        loss = (token_loss * valid_weights).sum()
    else:
        loss = token_loss.sum()
    if not torch.isfinite(loss):
        raise ValueError("YAQA full-model score loss overflowed")
    return loss, sampled_tokens.numel()


def _default_first_decoder_layer(model: nn.Module) -> nn.Module:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("YAQA Sketch B requires at least one decoder layer")
    return layers[0]


def _exact_symmetric_gram(matrix: torch.Tensor) -> torch.Tensor:
    """Project a numerically accumulated Gram matrix onto exact symmetry."""

    return (matrix + matrix.T) * 0.5


def _sketch_b_gram_updates(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    *,
    strategy: str,
    projections: tuple[torch.Tensor, torch.Tensor] | None = None,
    sequence_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two exact Sketch-B Gram sums without changing their objective.

    For each sequence ``b``, Sketch-B forms the full-model weight score
    ``G_b = D_b.T @ A_b``.  All strategies below compute
    ``sum_b G_b.T @ G_b`` and ``sum_b G_b @ G_b.T``; they differ only in
    contraction grouping and temporary storage.
    """

    if strategy not in {"batched", "flattened", "projected", "token_space"}:
        raise ValueError("YAQA Gram strategy must be `batched`, `flattened`, `projected`, or `token_space`")
    if activation.ndim != 3 or gradient.ndim != 3:
        raise ValueError("YAQA Gram inputs must have [batch, sequence, channels] geometry")
    if tuple(activation.shape[:2]) != tuple(gradient.shape[:2]):
        raise ValueError("YAQA Gram activation and gradient token geometry must match")
    if activation.dtype != torch.float32 or gradient.dtype != torch.float32:
        raise TypeError("YAQA Gram inputs must be FP32")

    batch_sequences, _, out_features = gradient.shape
    in_features = activation.shape[-1]
    if sequence_weights is not None:
        if sequence_weights.ndim != 1 or sequence_weights.shape[0] != batch_sequences:
            raise ValueError("YAQA sequence weights must have one scalar per independent sequence")
        if sequence_weights.device != gradient.device:
            raise ValueError("YAQA sequence weights must share the gradient device")
        if sequence_weights.device.type == "cpu" and (
            not bool(torch.isfinite(sequence_weights).all()) or not bool(sequence_weights.gt(0).all())
        ):
            raise ValueError("YAQA sequence weights must be finite and positive")
        # Sketch-B accumulates G_b.T@G_b and G_b@G_b.T. Scaling the
        # per-token score gradient by sqrt(alpha_b) therefore contributes
        # exactly alpha_b times each sequence Gram without duplicating rows.
        gradient = gradient * sequence_weights.to(dtype=gradient.dtype).sqrt().view(-1, 1, 1)
    if strategy == "projected":
        if projections is None or len(projections) != 2:
            raise ValueError("Projected YAQA Gram collection requires output and input projections")
        output_projection, input_projection = projections
        if tuple(output_projection.shape[:1]) != (out_features,) or tuple(input_projection.shape[:1]) != (
            in_features,
        ):
            raise ValueError("Projected YAQA Gram geometry does not match the module channels")
        if output_projection.shape[1] != input_projection.shape[1]:
            raise ValueError("Projected YAQA input and output ranks must match")
        if output_projection.device != gradient.device or input_projection.device != gradient.device:
            raise ValueError("Projected YAQA Gram projections must share the activation device")
        # For R with iid +/-1/sqrt(k), E[R R.T] = I. Therefore
        # E[(G.T R)(G.T R).T] = G.T G and likewise for G R. This
        # preserves a PSD Kronecker factor while replacing the opposite
        # channel dimension by the tunable projection rank k.
        projected_gradient = torch.bmm(
            activation.transpose(1, 2),
            gradient @ output_projection,
        )
        projected_activation = torch.bmm(
            gradient.transpose(1, 2),
            activation @ input_projection,
        )
        input_source = projected_gradient.permute(1, 0, 2).reshape(in_features, -1)
        output_source = projected_activation.permute(1, 0, 2).reshape(out_features, -1)
        input_update = input_source @ input_source.T
        output_update = output_source @ output_source.T
        return _exact_symmetric_gram(input_update), _exact_symmetric_gram(output_update)
    if strategy == "token_space":
        # Associativity gives A.T @ (D @ D.T) @ A and
        # D.T @ (A @ A.T) @ D.  This avoids materializing G when the token
        # dimension is smaller than the module channel dimensions.
        gradient_token_gram = torch.bmm(gradient, gradient.transpose(1, 2))
        activation_token_gram = torch.bmm(activation, activation.transpose(1, 2))
        input_update = torch.bmm(
            activation.transpose(1, 2),
            torch.bmm(gradient_token_gram, activation),
        ).sum(dim=0)
        output_update = torch.bmm(
            gradient.transpose(1, 2),
            torch.bmm(activation_token_gram, gradient),
        ).sum(dim=0)
        return _exact_symmetric_gram(input_update), _exact_symmetric_gram(output_update)

    per_sequence_gradient = torch.bmm(gradient.transpose(1, 2), activation)
    if strategy == "flattened":
        # Concatenating G_b vertically/horizontally turns each sum of Grams
        # into one GEMM and avoids B full Gram outputs.
        input_source = per_sequence_gradient.reshape(batch_sequences * out_features, in_features)
        output_source = per_sequence_gradient.permute(1, 0, 2).reshape(out_features, batch_sequences * in_features)
        input_update = input_source.T @ input_source
        output_update = output_source @ output_source.T
        return _exact_symmetric_gram(input_update), _exact_symmetric_gram(output_update)

    input_update = torch.bmm(per_sequence_gradient.transpose(1, 2), per_sequence_gradient).sum(dim=0)
    output_update = torch.bmm(per_sequence_gradient, per_sequence_gradient.transpose(1, 2)).sum(dim=0)
    return _exact_symmetric_gram(input_update), _exact_symmetric_gram(output_update)


def _streaming_projected_updates(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    *,
    rank: int,
    seed: int,
    sequence_weights: torch.Tensor | None = None,
    defer_diagonals: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the concatenated per-sequence scores into two compact factors.

    If ``X`` vertically concatenates every per-sequence weight score, the
    input update is ``X.T @ R`` for an iid Gaussian ``R``. The output side
    applies the same construction to the horizontal score concatenation.
    Accumulating these sources across batches is therefore one streaming
    random projection of the complete calibration set, without ever forming
    a dense per-batch Gram matrix. Token-space contractions additionally
    retain the exact input and output diagonal at O(T^2(I+O)) cost.
    """

    if activation.ndim != 3 or gradient.ndim != 3:
        raise ValueError("YAQA streaming projection inputs must have [batch, sequence, channels] geometry")
    if tuple(activation.shape[:2]) != tuple(gradient.shape[:2]):
        raise ValueError("YAQA streaming projection activation and gradient token geometry must match")
    if activation.dtype != torch.float32 or gradient.dtype != torch.float32:
        raise TypeError("YAQA streaming projection inputs must be FP32")
    batch_sequences, _, out_features = gradient.shape
    in_features = activation.shape[-1]
    if sequence_weights is not None:
        gradient = gradient * sequence_weights.to(dtype=gradient.dtype).sqrt().view(-1, 1, 1)

    projection_generator = torch.Generator(device=gradient.device).manual_seed(seed)
    projection = torch.empty(
        (batch_sequences, out_features + in_features, rank),
        dtype=torch.float32,
        device=gradient.device,
    ).normal_(generator=projection_generator)
    output_projection = projection[:, :out_features]
    input_projection = projection[:, out_features:]
    activation_transpose = activation.transpose(1, 2)
    gradient_transpose = gradient.transpose(1, 2)

    if batch_sequences == 1:
        input_source = activation[0].T @ (gradient[0] @ output_projection[0])
        output_source = gradient[0].T @ (activation[0] @ input_projection[0])
    else:
        input_source = torch.bmm(
            activation_transpose,
            _project_cuda(gradient, output_projection),
        ).sum(dim=0)
        output_source = torch.bmm(
            gradient_transpose,
            _project_cuda(activation, input_projection),
        ).sum(dim=0)
    if defer_diagonals:
        return input_source, output_source, None, None
    gradient_token_gram = torch.bmm(gradient, gradient_transpose)
    activation_token_gram = torch.bmm(activation, activation_transpose)
    input_diagonal = (activation * torch.bmm(gradient_token_gram, activation)).sum(dim=(0, 1))
    output_diagonal = (gradient * torch.bmm(activation_token_gram, gradient)).sum(dim=(0, 1))
    return input_source, output_source, input_diagonal, output_diagonal


class _YaqaFactorTransfer:
    """Overlap final compact-factor copies with the remaining model backward."""

    def __init__(self, device: torch.device) -> None:
        self.stream = torch.cuda.Stream(device=device)

    def submit(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        self.stream.wait_stream(torch.cuda.current_stream(self.stream.device))
        copied = []
        with torch.cuda.stream(self.stream):
            for tensor in tensors:
                host = torch.empty_like(tensor, device="cpu", pin_memory=True)
                host.copy_(tensor, non_blocking=True)
                tensor.record_stream(self.stream)
                copied.append(host)
        return copied

    def finish(self) -> None:
        self.stream.synchronize()


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
    mps_cleanup_interval: int = 8,
    mps_pack_symmetric_grams: bool = True,
    gram_strategy: str = "batched",
    gram_projection_rank: int | None = None,
    chat_template_config=None,
    activation=None,
    activation_modules: dict[str, nn.Linear] | None = None,
) -> tuple[
    dict[str, torch.Tensor | YaqaGramSketch],
    dict[str, torch.Tensor | YaqaGramSketch],
    dict[str, Any],
]:
    """Collect exact or streaming-projected YAQA Sketch-B factors."""

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
    if activation is None:
        activation_format = None
        activation_scale_method = None
        if activation_modules is not None:
            raise ValueError("YAQA activation modules require an activation-quantization config")
    else:
        if isinstance(activation, dict):
            activation_bits = activation.get("bits", 8)
            activation_format = activation.get("format")
            activation_scale_method = activation.get("scale_method")
            activation_target = activation.get("target", "p32_operand")
        else:
            activation_bits = getattr(activation, "bits", 8)
            activation_format = getattr(activation, "format", None)
            activation_scale_method = getattr(activation, "scale_method", None)
            activation_target = getattr(activation, "target", "p32_operand")
        if activation_bits != 8:
            raise ValueError("YAQA QVQ activation quantization requires 8 bits")
        if activation_target != "linear_input":
            raise ValueError(
                "YAQA activation-aware Sketch-B currently supports only "
                "activation.target=`linear_input`; `p32_operand` requires a "
                "post-SU/Hadamard collector"
            )
        activation_format = normalize_qvq_fp8_activation_format(activation_format)
        activation_scale_method = normalize_qvq_fp8_activation_scale_method(activation_scale_method)
        activation_modules = modules if activation_modules is None else activation_modules
        if not activation_modules or any(not isinstance(module, nn.Linear) for module in activation_modules.values()):
            raise TypeError("YAQA activation-quantization targets must be a nonempty linear-module dictionary")
        if len({id(module) for module in activation_modules.values()}) != len(activation_modules):
            raise ValueError("YAQA activation-quantization target modules must be unique")
        if not set(modules).issubset(activation_modules):
            raise ValueError("YAQA Sketch-B targets must be included in the activation-quantization module set")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("YAQA Sketch B progress callback must be callable")
    if isinstance(mps_cleanup_interval, bool) or not isinstance(mps_cleanup_interval, int) or mps_cleanup_interval < 1:
        raise ValueError("YAQA MPS cleanup interval must be a positive integer")
    if not isinstance(mps_pack_symmetric_grams, bool):
        raise TypeError("YAQA MPS symmetric-Gram packing flag must be boolean")
    if gram_strategy not in {"batched", "flattened", "projected", "streaming_projected", "token_space"}:
        raise ValueError(
            "YAQA Gram strategy must be `batched`, `flattened`, `projected`, `streaming_projected`, or `token_space`"
        )
    if gram_strategy in {"projected", "streaming_projected"}:
        if (
            isinstance(gram_projection_rank, bool)
            or not isinstance(gram_projection_rank, int)
            or gram_projection_rank < 1
        ):
            raise ValueError("Projected YAQA Gram collection requires a positive projection rank")
    elif gram_projection_rank is not None:
        raise ValueError("YAQA Gram projection rank is valid only with the projected strategy")
    if accumulator_device is not None:
        accumulator_device = torch.device(accumulator_device)
        if accumulator_device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("YAQA Sketch-B accumulators support CPU, CUDA, or MPS storage")
        if accumulator_device.type == "cuda" and device.type != "cuda":
            raise ValueError("YAQA CUDA accumulators require CUDA collection")
        if accumulator_device.type == "mps" and device.type != "mps":
            raise ValueError("YAQA MPS accumulators require MPS collection")
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
        sequence_weights = batch.get("fisher_sequence_weight")
        if sequence_weights is not None:
            sequence_weights = torch.as_tensor(sequence_weights, dtype=torch.float64).reshape(-1)
            if sequence_weights.shape[0] != attention_mask.shape[0]:
                raise ValueError("YAQA Fisher weights must provide one scalar per independent sequence")
            if not bool(torch.isfinite(sequence_weights).all()) or not bool(sequence_weights.gt(0).all()):
                raise ValueError("YAQA Fisher weights must be finite and positive")
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

    if gram_strategy == "streaming_projected":
        assert gram_projection_rank is not None
        factor_bytes = sum(
            (module.in_features + module.out_features) * (gram_projection_rank + 2) * 4 for module in modules.values()
        )
    else:
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
    if accumulator_device.type == "cuda" and accumulator_device.index is None:
        accumulator_device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    input_accumulators: dict[str, torch.Tensor] = {}
    output_accumulators: dict[str, torch.Tensor] = {}
    input_diagonal_accumulators: dict[str, torch.Tensor] = {}
    output_diagonal_accumulators: dict[str, torch.Tensor] = {}
    input_source_diagonals: dict[str, torch.Tensor] = {}
    output_source_diagonals: dict[str, torch.Tensor] = {}
    factor_transfer = (
        _YaqaFactorTransfer(device)
        if gram_strategy == "streaming_projected" and device.type == "cuda" and accumulator_device == device
        else None
    )
    packed_symmetric_accumulators = (
        gram_strategy != "streaming_projected"
        and mps_pack_symmetric_grams
        and device.type == "mps"
        and accumulator_device.type == "cpu"
    )
    gram_projections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if gram_strategy == "projected":
        assert gram_projection_rank is not None
        projection_generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x59415141)
        projection_scale = gram_projection_rank**-0.5
        for name, module in modules.items():
            output_projection = (
                torch.randint(
                    0,
                    2,
                    (module.out_features, gram_projection_rank),
                    generator=projection_generator,
                    dtype=torch.int8,
                )
                .to(dtype=torch.float32)
                .mul_(2)
                .sub_(1)
                .mul_(projection_scale)
                .to(device)
            )
            input_projection = (
                torch.randint(
                    0,
                    2,
                    (module.in_features, gram_projection_rank),
                    generator=projection_generator,
                    dtype=torch.int8,
                )
                .to(dtype=torch.float32)
                .mul_(2)
                .sub_(1)
                .mul_(projection_scale)
                .to(device)
            )
            gram_projections[name] = (output_projection, input_projection)
    sequence_counts = dict.fromkeys(modules, 0)
    total_sequences = 0
    total_valid_tokens = 0
    total_effective_sequence_weight = 0.0
    total_effective_weighted_tokens = 0.0
    active_mask: torch.Tensor | None = None
    active_sequence_weights: torch.Tensor | None = None
    active_batch_index = 0
    active_calls: set[str] = set()
    tensor_hook_handles = []
    module_hook_handles = []
    activation_hook_handles = []
    activation_errors: dict[str, dict[str, Any]] = {}
    parameters = tuple(model.parameters())
    parameter_requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    generator = torch.Generator(device=device).manual_seed(seed)
    # A Python bool() on a CUDA/MPS reduction synchronizes the accelerator.
    # Accumulate the identical predicate on-device and inspect it once after
    # collection instead of forcing three command-buffer drains per module.
    nonfinite_update = torch.zeros((), dtype=torch.bool, device=device) if device.type in {"cuda", "mps"} else None
    capture_started = time.perf_counter()
    phase_seconds = {
        "forward": 0.0,
        "loss": 0.0,
        "backward_and_sketch": 0.0,
        "python_gc": 0.0,
        "mps_synchronize": 0.0,
        "mps_empty_cache": 0.0,
    }
    mps_cleanup_count = 0
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

    def complete_cpu_diagonal(name, a, d, last, updates):
        groups = (input_diagonal_accumulators, output_diagonal_accumulators)
        for group, update in zip(groups, updates):
            if name in group:
                group[name].add_(update)
            else:
                group[name] = update
        if last:
            for group in groups:
                if not bool(torch.isfinite(group[name]).all()):
                    nonfinite_update.fill_(True)

    width = 8
    max_batch = max(batch["input_ids"].shape[0] for batch in batches)
    fixed_batch = all(batch["input_ids"].shape[0] == max_batch for batch in batches)
    cpu_finalize = _os.getenv("QVQ_CPU_FINALIZE", "1") == "1"
    sidecar = (
        _factor_sidecar_module.FactorSidecar(
            modules, sum(batch["input_ids"].shape[0] for batch in batches), gram_projection_rank, seed, YaqaGramSketch
        )
        if cpu_finalize
        else _NullSidecar()
    )
    for graph in _group_graph_cache.values():
        graph.reset()
    pending_gradients = []
    batch_group_names = None

    def accumulate_gradient_now(module_name: str, activation: torch.Tensor, gradient: torch.Tensor) -> None:
        if gradient.ndim != 3:
            raise ValueError(
                f"YAQA module {module_name} output gradient must have [batch, sequence, channels] geometry"
            )
        if tuple(gradient.shape[:2]) != tuple(activation.shape[:2]):
            raise ValueError(f"YAQA module {module_name} input/output token geometry does not match")
        if active_mask is None or tuple(active_mask.shape) != tuple(activation.shape[:2]):
            raise ValueError(f"YAQA module {module_name} attention-mask geometry does not match")

        hybrid = False
        keep = active_mask.to(device=gradient.device, dtype=torch.bool).unsqueeze(-1)
        gradient = torch.where(keep, gradient.detach().float(), 0)
        activation = torch.where(keep, activation.float(), 0)
        batch_sequences = gradient.shape[0]
        if nonfinite_update is None:
            if not torch.isfinite(gradient).all():
                raise ValueError(f"YAQA module {module_name} produced a non-finite full-model weight gradient")
        elif factor_transfer is None:
            nonfinite_update.logical_or_(~torch.isfinite(gradient).all())
        if hybrid:
            sidecar.submit(module_name, activation, gradient, active_batch_index == len(batches))
        if gram_strategy == "streaming_projected":
            assert gram_projection_rank is not None
            module_seed = zlib.crc32(module_name.encode("utf-8"), seed & 0xFFFFFFFF)
            projection_seed = (module_seed + active_batch_index * 0x9E3779B1) & 0x7FFFFFFFFFFFFFFF
            input_update, output_update, input_diagonal_update, output_diagonal_update = _streaming_projected_updates(
                activation,
                gradient,
                rank=gram_projection_rank,
                seed=projection_seed,
                sequence_weights=active_sequence_weights,
                defer_diagonals=hybrid,
            )
        else:
            input_update, output_update = _sketch_b_gram_updates(
                activation,
                gradient,
                strategy=gram_strategy,
                projections=gram_projections.get(module_name),
                sequence_weights=active_sequence_weights,
            )
        if nonfinite_update is None:
            updates_finite = torch.isfinite(input_update).all() and torch.isfinite(output_update).all()
            if gram_strategy == "streaming_projected":
                updates_finite = (
                    updates_finite
                    and torch.isfinite(input_diagonal_update).all()
                    and torch.isfinite(output_diagonal_update).all()
                )
            if not updates_finite:
                raise ValueError(f"YAQA module {module_name} produced an overflowing Sketch-B Gram update")
        elif factor_transfer is None:
            # GPU streaming accumulators are validated when finalized. An
            # infinity/NaN in an update cannot become finite through subsequent
            # additions, and the source-square reduction checks every element.
            nonfinite_update.logical_or_(~torch.isfinite(input_update).all())
            nonfinite_update.logical_or_(~torch.isfinite(output_update).all())
            if gram_strategy == "streaming_projected":
                nonfinite_update.logical_or_(~torch.isfinite(input_diagonal_update).all())
                nonfinite_update.logical_or_(~torch.isfinite(output_diagonal_update).all())
        if packed_symmetric_accumulators:
            from gptqmodel.utils.qvq_mlx import (
                qvq_mlx_pack_symmetric_gram_from_torch_mps,
            )

            input_update = qvq_mlx_pack_symmetric_gram_from_torch_mps(input_update.detach().contiguous()).to(
                device="cpu"
            )
            output_update = qvq_mlx_pack_symmetric_gram_from_torch_mps(output_update.detach().contiguous()).to(
                device="cpu"
            )
        else:
            input_update = input_update.detach().to(device=accumulator_device)
            output_update = output_update.detach().to(device=accumulator_device)
            if gram_strategy == "streaming_projected" and not hybrid:
                input_diagonal_update = input_diagonal_update.detach().to(device=accumulator_device)
                output_diagonal_update = output_diagonal_update.detach().to(device=accumulator_device)
        if module_name in input_accumulators:
            input_accumulators[module_name].add_(input_update)
            output_accumulators[module_name].add_(output_update)
            if gram_strategy == "streaming_projected" and not hybrid:
                input_diagonal_accumulators[module_name].add_(input_diagonal_update)
                output_diagonal_accumulators[module_name].add_(output_diagonal_update)
        else:
            input_accumulators[module_name] = input_update.contiguous()
            output_accumulators[module_name] = output_update.contiguous()
            if gram_strategy == "streaming_projected" and not hybrid:
                input_diagonal_accumulators[module_name] = input_diagonal_update.contiguous()
                output_diagonal_accumulators[module_name] = output_diagonal_update.contiguous()
        if factor_transfer is not None and active_batch_index == len(batches):
            # This module cannot receive another update in the final batch.
            # Preserve the exact reduction and validation, then let the copy
            # engine drain its factors while earlier layers backpropagate.
            input_source_diagonals[module_name] = input_accumulators[module_name].square().sum(dim=1)
            output_source_diagonals[module_name] = output_accumulators[module_name].square().sum(dim=1)
            assert nonfinite_update is not None
            validation_groups = (input_source_diagonals, output_source_diagonals)
            if not hybrid:
                validation_groups += (input_diagonal_accumulators, output_diagonal_accumulators)
            for values in validation_groups:
                nonfinite_update.logical_or_(~torch.isfinite(values[module_name]).all())
            groups = (
                input_accumulators,
                output_accumulators,
                input_source_diagonals,
                output_source_diagonals,
            )
            if not hybrid:
                groups += (input_diagonal_accumulators, output_diagonal_accumulators)
            copied = factor_transfer.submit([group[module_name] for group in groups])
            for group, host in zip(groups, copied):
                group[module_name] = host
        sequence_counts[module_name] += batch_sequences
        sidecar.drain(False)

    def accumulate_gradient(module_name, activation, gradient):
        eligible = (
            fixed_batch
            and gram_strategy == "streaming_projected"
            and gram_projection_rank == 256
            and factor_transfer is not None
            and active_sequence_weights is None
            and activation.dtype == torch.bfloat16
            and gradient.dtype == torch.bfloat16
            and activation.shape[1] == 64
            and activation.shape[0] == 2
            and activation.shape[2] % 64 == 0
            and gradient.shape[2] % 64 == 0
            and activation.stride(2) == 1
            and gradient.stride(2) == 1
            and (activation.shape[-1], gradient.shape[-1]) in _TESTED_GEOMETRIES
        )
        if eligible:
            pending_gradients.append((module_name, activation.detach().clone(), gradient.detach().clone()))
            geometry = (tuple(activation.shape), tuple(gradient.shape))
            ready = [
                entry for entry in pending_gradients if (tuple(entry[1].shape), tuple(entry[2].shape)) == geometry
            ]
            if len(ready) == width:
                pending_gradients[:] = [
                    entry for entry in pending_gradients if (tuple(entry[1].shape), tuple(entry[2].shape)) != geometry
                ]
                process_pending_gradients(ready)
        else:
            accumulate_gradient_now(module_name, activation, gradient)

    def process_pending_gradients(ready=None):
        nonlocal batch_group_names
        groups = {}
        for name, act, gradient in pending_gradients if ready is None else ready:
            groups.setdefault((tuple(act.shape), tuple(gradient.shape)), []).append((name, act, gradient))
        if ready is None:
            pending_gradients.clear()
        for entries in groups.values():
            entries.sort(key=lambda entry: entry[0])
        plan_key = (device, max_batch, gram_projection_rank)
        keep = active_mask.to(dtype=torch.bool)
        last = active_batch_index == len(batches)
        for geometry in sorted(groups):
            entries = groups[geometry]
            for begin in range(0, len(entries), width):
                chunk = entries[begin : begin + width]
                if len(chunk) != width:
                    for name, act, gradient in chunk:
                        accumulate_gradient_now(name, act, gradient)
                    continue
                names = tuple(entry[0] for entry in chunk)
                raw_a = torch.stack([entry[1] for entry in chunk])
                raw_d = torch.stack([entry[2] for entry in chunk])
                key = (plan_key, names)
                if key not in _group_graph_cache:
                    if len(_group_graph_cache) >= 128:
                        raise RuntimeError("Experimental graph cache limit reached")
                    pool = _group_pools.setdefault(plan_key, torch.cuda.graph_pool_handle())
                    _group_graph_cache[key] = _group_statistics_module.GroupStatistics(
                        raw_a,
                        raw_d,
                        keep,
                        gram_projection_rank,
                        _group_project_module,
                        pool,
                        _group_streams.setdefault(plan_key, torch.cuda.Stream(device=device)),
                    )
                seeds = [
                    (zlib.crc32(name.encode("utf-8"), seed & 0xFFFFFFFF) + active_batch_index * 0x9E3779B1)
                    & 0x7FFFFFFFFFFFFFFF
                    for name in names
                ]
                cpu_a, cpu_d, source_a, source_d, norm_a, norm_d, bad = _group_graph_cache[key](
                    raw_a, raw_d, keep, seeds, last=last
                )

                if last:
                    nonfinite_update.logical_or_(bad | (~torch.isfinite(cpu_a).all()) | (~torch.isfinite(cpu_d).all()))
                    source_a, source_d, norm_a, norm_d, cpu_a, cpu_d = factor_transfer.submit(
                        [source_a, source_d, norm_a, norm_d, cpu_a, cpu_d]
                    )
                    if cpu_finalize:
                        factor_event = torch.cuda.Event(blocking=True)
                        factor_event.record(factor_transfer.stream)
                        sidecar.submit(names, [source_a, source_d, norm_a, norm_d, cpu_a, cpu_d], factor_event)
                for index, name in enumerate(names):
                    input_diagonal_accumulators[name] = cpu_a[index]
                    output_diagonal_accumulators[name] = cpu_d[index]
                    input_accumulators[name] = source_a[index]
                    output_accumulators[name] = source_d[index]
                    if last:
                        input_source_diagonals[name] = norm_a[index]
                        output_source_diagonals[name] = norm_d[index]
                    sequence_counts[name] += max_batch
            entries.clear()

    if activation_format is not None:
        assert activation_modules is not None

        for activation_name, activation_module in activation_modules.items():

            def activation_pre_hook(_module, args, module_name=activation_name):
                if not args:
                    raise ValueError(f"YAQA module {module_name} received no positional activation input")
                source = _first_tensor(args)
                if not source.is_floating_point():
                    raise TypeError(f"YAQA module {module_name} activation input must be floating point")
                _, scale, dequantized = fake_quantize_qvq_fp8_activation(
                    source,
                    format=activation_format,
                    scale_method=activation_scale_method,
                    straight_through=True,
                    validate=False,
                )
                if not _YAQA_CHECKPOINT_RECOMPUTING.get():
                    source_fp32 = source.detach().to(torch.float32)
                    error = dequantized.detach().to(torch.float32) - source_fp32
                    update = {
                        "elements": source.numel(),
                        "source_square_sum": source_fp32.square().sum(),
                        "error_square_sum": error.square().sum(),
                        "maximum_absolute_error": error.abs().amax(),
                        "minimum_scale": scale.amin(),
                        "maximum_scale": scale.amax(),
                        "finite": torch.isfinite(source_fp32).all() & torch.isfinite(dequantized).all(),
                    }
                    accumulated = activation_errors.get(module_name)
                    if accumulated is None:
                        activation_errors[module_name] = update
                    else:
                        accumulated["elements"] += update["elements"]
                        accumulated["source_square_sum"].add_(update["source_square_sum"])
                        accumulated["error_square_sum"].add_(update["error_square_sum"])
                        accumulated["maximum_absolute_error"] = torch.maximum(
                            accumulated["maximum_absolute_error"], update["maximum_absolute_error"]
                        )
                        accumulated["minimum_scale"] = torch.minimum(
                            accumulated["minimum_scale"], update["minimum_scale"]
                        )
                        accumulated["maximum_scale"] = torch.maximum(
                            accumulated["maximum_scale"], update["maximum_scale"]
                        )
                        accumulated["finite"].logical_and_(update["finite"])
                return (dequantized, *args[1:])

            activation_hook_handles.append(activation_module.register_forward_pre_hook(activation_pre_hook))

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
                active_batch_index = batch_index
                template_mask = batch.get("chat_template_mask")
                encoded = {
                    name: value.to(device)
                    for name, value in batch.items()
                    if name in {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
                }
                active_mask = encoded["attention_mask"]
                if active_mask.ndim != 2:
                    raise ValueError("YAQA attention masks must be rank-2")
                valid_by_sequence = active_mask.ne(0).sum(dim=1)
                if not bool(valid_by_sequence.gt(0).all()):
                    raise ValueError("every YAQA calibration sequence must contain at least one valid token")
                batch_sequence_weights = batch.get("fisher_sequence_weight")
                if batch_sequence_weights is None:
                    active_sequence_weights = None
                    effective_weights = torch.ones(
                        active_mask.shape[0], dtype=torch.float64, device=active_mask.device
                    )
                else:
                    raw_sequence_weights = torch.as_tensor(
                        batch_sequence_weights, dtype=torch.float64, device="cpu"
                    ).reshape(-1)
                    if raw_sequence_weights.shape[0] != active_mask.shape[0]:
                        raise ValueError("YAQA Fisher weights must provide one scalar per independent sequence")
                    if not bool(torch.isfinite(raw_sequence_weights).all()) or not bool(
                        raw_sequence_weights.gt(0).all()
                    ):
                        raise ValueError("YAQA Fisher weights must be finite and positive")
                    active_sequence_weights = raw_sequence_weights.to(device=device, dtype=torch.float32)
                    effective_weights = raw_sequence_weights
                active_calls.clear()
                phase_started = time.perf_counter()
                outputs = model(**encoded, use_cache=False)
                phase_seconds["forward"] += time.perf_counter() - phase_started
                logits = getattr(outputs, "logits", None)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("YAQA full-model forward must return tensor logits")
                phase_started = time.perf_counter()
                token_weights = None
                if template_mask is not None:
                    template_mask = template_mask.to(device=device, dtype=torch.bool)
                    valid_template = template_mask & active_mask.bool()
                    valid_content = active_mask.bool() & ~template_mask
                    content_count = valid_content.sum(dim=1).to(dtype=logits.dtype)
                    template_count = valid_template.sum(dim=1).to(dtype=logits.dtype)
                    content_mass = float(getattr(chat_template_config, "content_weight", 0.97))
                    template_mass = 1.0 - content_mass
                    gamma = (template_mass * content_count) / (content_mass * template_count.clamp_min(1.0))
                    token_weights = torch.where(
                        valid_template, gamma.unsqueeze(1), torch.ones_like(gamma).unsqueeze(1)
                    )
                    token_weights = torch.where(active_mask.bool(), token_weights, torch.ones_like(token_weights))
                    # Normalize each sequence so weighting changes token composition,
                    # not the relative scale of independent sequence gradients.
                    valid_count = active_mask.sum(dim=1).to(dtype=logits.dtype)
                    weight_sum = (token_weights * active_mask).sum(dim=1).clamp_min(1.0)
                    token_weights = token_weights * (valid_count / weight_sum).unsqueeze(1)
                loss, valid_tokens = yaqa_real_fisher_loss(
                    logits,
                    active_mask,
                    generator=generator,
                    token_weights=token_weights,
                )
                phase_seconds["loss"] += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                loss.backward()
                process_pending_gradients()
                sidecar.drain(True)
                phase_seconds["backward_and_sketch"] += time.perf_counter() - phase_started
                missing = set(modules) - active_calls
                if missing:
                    raise ValueError(f"YAQA full-model forward did not execute target modules {sorted(missing)}")
                batch_sequences = active_mask.shape[0]
                total_sequences += batch_sequences
                total_valid_tokens += valid_tokens
                total_effective_sequence_weight += float(effective_weights.sum().item())
                total_effective_weighted_tokens += float(
                    (
                        effective_weights
                        * valid_by_sequence.to(
                            device=effective_weights.device,
                            dtype=torch.float64,
                        )
                    )
                    .sum()
                    .item()
                )
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
                active_sequence_weights = None
                # Explicitly drop the autograd graph before the next sequence.
                # MPS command buffers are asynchronous and otherwise keep the
                # logits/loss graph live until allocator pressure forces a flush.
                # CUDA releases this graph through reference counting; retain
                # normal cyclic GC instead of scanning the entire model here.
                del outputs, logits, loss
                cleanup_due = batch_index == len(batches) or batch_index % mps_cleanup_interval == 0
                if device.type != "cuda" and cleanup_due:
                    phase_started = time.perf_counter()
                    gc.collect()
                    phase_seconds["python_gc"] += time.perf_counter() - phase_started
                if device.type == "mps" and cleanup_due:
                    phase_started = time.perf_counter()
                    torch.mps.synchronize()
                    phase_seconds["mps_synchronize"] += time.perf_counter() - phase_started
                    phase_started = time.perf_counter()
                    torch.mps.empty_cache()
                    phase_seconds["mps_empty_cache"] += time.perf_counter() - phase_started
                    mps_cleanup_count += 1
    finally:
        active_mask = None
        active_sequence_weights = None
        decoder_seed_handle.remove()
        for handle in tensor_hook_handles:
            handle.remove()
        for handle in module_hook_handles:
            handle.remove()
        for handle in activation_hook_handles:
            handle.remove()
        for parameter, requires_grad in zip(parameters, parameter_requires_grad):
            parameter.requires_grad_(requires_grad)
        if previous_fp32_precision is not None:
            cuda_matmul.fp32_precision = previous_fp32_precision
        if previous_allow_tf32 is not None:
            cuda_matmul.allow_tf32 = previous_allow_tf32

        try:
            sidecar.finish()
        finally:
            if factor_transfer is not None:
                factor_transfer.finish()

    if total_sequences < 1:
        raise ValueError("YAQA Sketch B observed no independent calibration sequences")
    if not math.isfinite(total_effective_sequence_weight) or total_effective_sequence_weight <= 0:
        raise ValueError("YAQA Sketch B observed no positive effective sequence weight")
    for name in modules:
        if sequence_counts[name] != total_sequences or name not in input_accumulators:
            raise ValueError(f"YAQA module {name} did not produce one gradient for every sequence")

    if gram_strategy == "streaming_projected" and factor_transfer is None:
        for name in modules:
            input_source_diagonals[name] = input_accumulators[name].square().sum(dim=1)
            output_source_diagonals[name] = output_accumulators[name].square().sum(dim=1)
            if nonfinite_update is not None and accumulator_device.type == device.type:
                nonfinite_update.logical_or_(~torch.isfinite(input_source_diagonals[name]).all())
                nonfinite_update.logical_or_(~torch.isfinite(output_source_diagonals[name]).all())
                nonfinite_update.logical_or_(~torch.isfinite(input_diagonal_accumulators[name]).all())
                nonfinite_update.logical_or_(~torch.isfinite(output_diagonal_accumulators[name]).all())
    if capture_end_event is not None:
        capture_end_event.record()
    transfer_started = time.perf_counter()
    if nonfinite_update is not None and bool(nonfinite_update):
        raise ValueError("YAQA produced a non-finite Sketch-B Gram update")
    for name, module in modules.items():
        if packed_symmetric_accumulators:
            from gptqmodel.utils.qvq_mlx import (
                qvq_mlx_unpack_symmetric_gram_to_torch_cpu,
            )

            input_accumulators[name] = qvq_mlx_unpack_symmetric_gram_to_torch_cpu(
                input_accumulators[name],
                width=module.in_features,
            )
            output_accumulators[name] = qvq_mlx_unpack_symmetric_gram_to_torch_cpu(
                output_accumulators[name],
                width=module.out_features,
            )
        elif factor_transfer is None:
            input_accumulators[name] = input_accumulators[name].to(device="cpu")
            output_accumulators[name] = output_accumulators[name].to(device="cpu")
            if gram_strategy == "streaming_projected":
                input_diagonal_accumulators[name] = input_diagonal_accumulators[name].to(device="cpu")
                output_diagonal_accumulators[name] = output_diagonal_accumulators[name].to(device="cpu")
                input_source_diagonals[name] = input_source_diagonals[name].to(device="cpu")
                output_source_diagonals[name] = output_source_diagonals[name].to(device="cpu")
        accelerator_streaming_validated = (
            gram_strategy == "streaming_projected"
            and nonfinite_update is not None
            and accumulator_device.type == device.type
        )
        if not accelerator_streaming_validated and (
            not torch.isfinite(input_accumulators[name]).all() or not torch.isfinite(output_accumulators[name]).all()
        ):
            raise ValueError(f"YAQA module {name} produced an overflowing Sketch-B accumulator")
    transfer_seconds = time.perf_counter() - transfer_started
    capture_cuda_ms = None if capture_start_event is None else capture_start_event.elapsed_time(capture_end_event)

    input_hessians: dict[str, torch.Tensor | YaqaGramSketch] = {}
    output_hessians: dict[str, torch.Tensor | YaqaGramSketch] = {}
    for name, module in modules.items():
        if sequence_counts[name] != total_sequences or name not in input_accumulators:
            raise ValueError(f"YAQA module {name} did not produce one gradient for every sequence")
        out_features, in_features = module.weight.shape
        if name in sidecar.factors:
            input_hessians[name], output_hessians[name] = sidecar.factors[name]
            continue
        if gram_strategy == "streaming_projected":
            input_hessians[name] = YaqaGramSketch(
                source=input_accumulators[name].contiguous(),
                diagonal=input_diagonal_accumulators[name]
                .div(total_effective_sequence_weight * out_features)
                .clamp_min_(0)
                .contiguous(),
                normalizer=total_effective_sequence_weight * out_features * gram_projection_rank,
                seed=seed,
                source_diagonal=input_source_diagonals[name].contiguous(),
                _source_diagonal_validated=True,
                # GPU finalization checked both unscaled vectors. Dividing by
                # >= 1 cannot overflow; clamp_min makes the diagonal nonnegative.
                # Keep CPU validation for small Fisher weights and CPU storage.
                _finite_nonnegative_validated=(
                    factor_transfer is not None and total_effective_sequence_weight * out_features >= 1
                ),
            )
            output_hessians[name] = YaqaGramSketch(
                source=output_accumulators[name].contiguous(),
                diagonal=output_diagonal_accumulators[name]
                .div(total_effective_sequence_weight * in_features)
                .clamp_min_(0)
                .contiguous(),
                normalizer=total_effective_sequence_weight * in_features * gram_projection_rank,
                seed=seed,
                source_diagonal=output_source_diagonals[name].contiguous(),
                _source_diagonal_validated=True,
                _finite_nonnegative_validated=(
                    factor_transfer is not None and total_effective_sequence_weight * in_features >= 1
                ),
            )
        else:
            input_hessians[name] = (
                input_accumulators[name].div(total_effective_sequence_weight * out_features).contiguous()
            )
            output_hessians[name] = (
                output_accumulators[name].div(total_effective_sequence_weight * in_features).contiguous()
            )

    if gram_strategy == "streaming_projected":
        input_factor_elements = sum(
            factor.source.numel() + factor.diagonal.numel() + factor.source_diagonal.numel()
            for factor in input_hessians.values()
        )
        output_factor_elements = sum(
            factor.source.numel() + factor.diagonal.numel() + factor.source_diagonal.numel()
            for factor in output_hessians.values()
        )
        dense_factor_storage_bytes = sum(
            (module.in_features**2 + module.out_features**2) * 4 for module in modules.values()
        )
    else:
        input_factor_elements = sum(factor.numel() for factor in input_hessians.values())
        output_factor_elements = sum(factor.numel() for factor in output_hessians.values())
        dense_factor_storage_bytes = (input_factor_elements + output_factor_elements) * 4
    activation_error_summary = None
    if activation_format is not None:
        missing_activation_modules = set(activation_modules or {}) - set(activation_errors)
        if missing_activation_modules:
            raise ValueError(
                f"YAQA full-model forward did not execute activation-quantization targets "
                f"{sorted(missing_activation_modules)}"
            )
        module_summaries = {}
        total_elements = 0
        total_source_square = 0.0
        total_error_square = 0.0
        maximum_absolute_error = 0.0
        minimum_scale = math.inf
        maximum_scale = 0.0
        for name, accumulated in activation_errors.items():
            if not bool(accumulated["finite"].item()):
                raise ValueError(f"YAQA module {name} produced non-finite A8 calibration values")
            elements = int(accumulated["elements"])
            source_square = float(accumulated["source_square_sum"].item())
            error_square = float(accumulated["error_square_sum"].item())
            module_maximum_error = float(accumulated["maximum_absolute_error"].item())
            module_minimum_scale = float(accumulated["minimum_scale"].item())
            module_maximum_scale = float(accumulated["maximum_scale"].item())
            module_summaries[name] = {
                "elements": elements,
                "rmse": math.sqrt(error_square / max(1, elements)),
                "relative_rmse": math.sqrt(error_square / max(source_square, torch.finfo(torch.float32).tiny)),
                "maximum_absolute_error": module_maximum_error,
                "minimum_scale": module_minimum_scale,
                "maximum_scale": module_maximum_scale,
            }
            total_elements += elements
            total_source_square += source_square
            total_error_square += error_square
            maximum_absolute_error = max(maximum_absolute_error, module_maximum_error)
            minimum_scale = min(minimum_scale, module_minimum_scale)
            maximum_scale = max(maximum_scale, module_maximum_scale)
        activation_error_summary = {
            "bits": 8,
            "format": activation_format,
            "scale_method": activation_scale_method,
            "elements": total_elements,
            "rmse": math.sqrt(total_error_square / max(1, total_elements)),
            "relative_rmse": math.sqrt(total_error_square / max(total_source_square, torch.finfo(torch.float32).tiny)),
            "maximum_absolute_error": maximum_absolute_error,
            "minimum_scale": minimum_scale,
            "maximum_scale": maximum_scale,
            "modules": module_summaries,
        }
    return (
        input_hessians,
        output_hessians,
        {
            "method": (
                "YAQA-v3 Sketch B streaming randomized Fisher"
                if gram_strategy == "streaming_projected"
                else "YAQA-v3 Sketch B real Fisher"
            ),
            "full_model_backward": True,
            "independent_sequences": total_sequences,
            "unique_sequences": total_sequences,
            "valid_output_samples": total_valid_tokens,
            "raw_valid_tokens": total_valid_tokens,
            "effective_weighted_sequences": total_effective_sequence_weight,
            "effective_weighted_tokens": total_effective_weighted_tokens,
            "monte_carlo_samples_per_output": 1,
            "sequence_loss_reduction": "per_sequence_token_sum",
            "activation_checkpointing": bool(checkpoint_modules),
            "checkpointed_modules": len(checkpoint_modules),
            "accumulator_device": accumulator_device.type,
            "factor_transfer_overlap": factor_transfer is not None,
            "phase_wall_seconds": phase_seconds,
            "mps_cleanup_interval": mps_cleanup_interval,
            "mps_cleanup_count": mps_cleanup_count,
            "accumulator_bytes": sum(
                (module.in_features * (module.in_features + 1) + module.out_features * (module.out_features + 1)) * 2
                for module in modules.values()
            )
            if packed_symmetric_accumulators
            else factor_bytes,
            "packed_symmetric_accumulators": packed_symmetric_accumulators,
            "gram_strategy": gram_strategy,
            "gram_projection_rank": gram_projection_rank,
            "gram_projection_distribution": ("gaussian" if gram_strategy == "streaming_projected" else None),
            "gram_exact_diagonal": gram_strategy == "streaming_projected",
            "factor_approximate": gram_strategy in {"projected", "streaming_projected"},
            "capture_wall_seconds": time.perf_counter() - capture_started,
            "cpu_sidecar_module_jobs": sidecar.module_jobs,
            "capture_cuda_ms": capture_cuda_ms,
            "final_host_transfer_seconds": transfer_seconds,
            "minimum_sequences": minimum_sequences,
            "factor_dtype": "float32",
            "input_factor_elements": input_factor_elements,
            "output_factor_elements": output_factor_elements,
            "factor_storage_bytes": (input_factor_elements + output_factor_elements) * 4,
            "dense_factor_storage_bytes": dense_factor_storage_bytes,
            "factor_compression_ratio": dense_factor_storage_bytes
            / max(1, (input_factor_elements + output_factor_elements) * 4),
            "tf32": False,
            "seed": seed,
            "activation_quantization_error": activation_error_summary,
        },
    )


__all__ = [
    "YAQA_DEFAULT_RATE_REGULARIZATION",
    "YAQA_DEFAULT_REGULARIZATION",
    "YAQA_PAPER_MINIMUM_SEQUENCES",
    "YAQA_PAPER_RECOMMENDED_SEQUENCES",
    "YAQA_PAPER_REGULARIZATION",
    "YaqaGramSketch",
    "capture_yaqa_sketch_b",
    "yaqa_real_fisher_loss",
]


import hashlib
import os as _os
import sys
from pathlib import Path

from gptqmodel.quantization import qvq_yaqa as _baseline
from gptqmodel.quantization import qvq_yaqa_cuda as _group_project_module
from scripts.experiments.qvq_cpu_sidecar import sidecar as _factor_sidecar_module
from scripts.experiments.qvq_cpu_sidecar import statistics as _group_statistics_module

_BASELINE_SHA256 = "b5c2e1093efd1880c31e884a89dc06783f49e57d1b028e3d5acde1ea5119e860"
_BASELINE_COMPATIBLE = hashlib.sha256(Path(_baseline.__file__).read_bytes()).hexdigest() == _BASELINE_SHA256
_TESTED_GEOMETRIES = {
    (17408, 5120),
    (5120, 17408),
    (6144, 5120),
    (5120, 1024),
    (5120, 12288),
    (5120, 6144),
    (5120, 10240),
}
_group_graph_cache = {}
_group_pools = {}
_group_streams = {}
_capture_plan = None


class _NullSidecar:
    def __init__(self):
        self.factors = {}
        self.module_jobs = 0

    def drain(self, all):
        pass

    def finish(self):
        pass


_capture_hybrid_impl = capture_yaqa_sketch_b


@wraps(_capture_hybrid_impl)
def capture_yaqa_sketch_b(model, batches, modules, *, device, **kwargs):
    """Opt-in H200/B2 experiment; retain the accepted collector outside its scope."""
    device = torch.device(device)
    supported = (
        _BASELINE_COMPATIBLE
        and sys.platform == "linux"
        and device.type == "cuda"
        and str(torch.__version__) == "2.15.0.dev20260901+cu130"
        and kwargs.get("gram_strategy") == "streaming_projected"
        and kwargs.get("gram_projection_rank") == 256
        and not kwargs.get("checkpoint_modules")
        and kwargs.get("activation") is None
        and batches
        and all(tuple(batch["input_ids"].shape) == (2, 64) for batch in batches)
        and len(modules) == 400
        and all(module.weight.dtype == torch.bfloat16 for module in modules.values())
        and hasattr(sys, "_is_gil_enabled")
        and not sys._is_gil_enabled()
    )
    if supported:
        selected_index = device.index if device.index is not None else torch.cuda.current_device()
        accumulator = torch.device(kwargs.get("accumulator_device") or device)
        accumulator_index = accumulator.index if accumulator.index is not None else torch.cuda.current_device()
        supported = accumulator.type == "cuda" and accumulator_index == selected_index
    if supported:
        properties = torch.cuda.get_device_properties(device)
        supported = (
            (properties.major, properties.minor) == (9, 0)
            and properties.multi_processor_count == 132
            and "H200" in properties.name
        )
    if supported:
        inventory = _factor_sidecar_module.cpu_runtime_inventory()
        supported = all(inventory["numa_allowed_cpus"].get(f"node{node}") for node in range(2))
    if not supported or not _factor_sidecar_module.capture_lock.acquire(blocking=False):
        return _baseline.capture_yaqa_sketch_b(model, batches, modules, device=device, **kwargs)
    global _capture_plan
    try:
        plan = (
            device,
            tuple(sorted(_os.sched_getaffinity(0))),
            tuple(sorted((name, module.in_features, module.out_features) for name, module in modules.items())),
        )
        if plan != _capture_plan:
            _group_graph_cache.clear()
            _group_pools.clear()
            _group_streams.clear()
            _capture_plan = plan
        result = _capture_hybrid_impl(model, batches, modules, device=device, **kwargs)
        result[2]["cpu_sidecar"] = (
            "factor_finalize" if _os.getenv("QVQ_CPU_FINALIZE", "1") == "1" else "disabled_grouped_gpu"
        )
        return result
    finally:
        _factor_sidecar_module.capture_lock.release()


def clear_cache():
    """Release this experiment's graph plan and CPU workers after in-flight work."""
    global _capture_plan
    with _factor_sidecar_module.capture_lock:
        _group_graph_cache.clear()
        _group_pools.clear()
        _group_streams.clear()
        _factor_sidecar_module.close_workers()
        _capture_plan = None
