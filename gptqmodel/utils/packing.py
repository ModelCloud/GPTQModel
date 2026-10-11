# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Stable (``hf_``-prefixed) packing and repacking APIs for external integrations.

Downstream projects (transformers/optimum, auto-round, ...) use these entry
points to inspect GPT-QModel kernel support and to pack or repack a single
quantized layer.  Every function in this module is considered public API:
keep the signatures backward compatible and add a new ``hf_`` function instead
of changing an existing one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from ..nn_modules.qlinear import AWQuantLinear, BaseQuantLinear, GPTQQuantLinear
from ..quantization import FORMAT, METHOD, QuantizeConfig
from .backend import BACKEND, normalize_backend
from .importer import (
    AUTO_BACKEND_KERNEL_MAPPING,
    BACKEND_TO_METHOD_FORMAT_MAPPING,
    _as_selector_device,
    select_quant_linear,
    validate_quant_linear,
)
from .logger import setup_logger


log = setup_logger()

__all__ = [
    "hf_check_best_packing_format",
    "hf_check_packing_feasibility",
    "hf_pack_layer",
    "hf_post_init",
    "hf_repack_layer",
]


# Core GPTQ/AWQ packed tensors copied when repacking between kernels that share
# the same checkpoint layout.
_PACKED_STATE_KEYS = ("qweight", "qzeros", "scales", "g_idx")

# Kernels whose `post_init()` converts the packed buffers into a private
# runtime ABI (Marlin tile layout, Swordfish prepack, ...).  They are valid
# repack *targets*, but their buffers cannot be used as a copy source.
_ABI_TRANSFORMING_QUANT_TYPES = frozenset({
    "marlin",
    "awq_marlin",
    "machete",
    "awq_machete",
    "swordfish",
    "awq_swordfish",
    "gptq_bitblas",
    "awq_bitblas",
})


def _normalize_enum(value, enum_cls, field: str):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported {field}: `{value}`") from exc
    raise ValueError(f"{field} must be a string or `{enum_cls.__name__}`, got `{type(value)}`")


def _normalize_method(quant_method: Optional[Union[str, METHOD]]) -> METHOD:
    if quant_method is None:
        return METHOD.GPTQ
    return _normalize_enum(quant_method, METHOD, "quant_method")


def _normalize_format(format_value: Optional[Union[str, FORMAT]]) -> FORMAT:
    if format_value is None:
        return FORMAT.GPTQ
    return _normalize_enum(format_value, FORMAT, "format")


def _resolve_pack_dtype(
    pack_dtype: Optional[torch.dtype],
    method: METHOD,
    format_value: FORMAT,
) -> torch.dtype:
    """Return the effective storage word dtype, mirroring `hf_select_quant_linear_v2`."""
    if pack_dtype is not None:
        return pack_dtype
    return torch.int16 if method == METHOD.AWQ and format_value == FORMAT.GEMV_FAST else torch.int32


def _normalize_pack_impl(pack_impl: Optional[str], module: BaseQuantLinear) -> str:
    """Map a user supplied pack implementation to one of gpu/block/original."""
    requested = (pack_impl or "original").strip().lower()

    if requested in {"cpu", "block", "pack_block"}:
        if not hasattr(module, "pack_block"):
            log.warning(
                "hf_pack_layer: block packing requested but `%s` lacks pack_block; "
                "falling back to original pack.",
                type(module).__name__,
            )
            return "original"
        return "block"
    if requested in {"original", "pack_original"}:
        return "original"
    if requested == "gpu":
        if not torch.cuda.is_available():
            log.warning(
                "hf_pack_layer: GPU packing requested but CUDA is unavailable; "
                "falling back to original pack."
            )
            return "original"
        if not hasattr(module, "pack_gpu"):
            log.warning(
                "hf_pack_layer: GPU packing requested but `%s` lacks pack_gpu; "
                "falling back to original pack.",
                type(module).__name__,
            )
            return "original"
        return "gpu"

    log.warning(
        "hf_pack_layer: unknown pack_impl `%s`; defaulting to original pack.",
        pack_impl,
    )
    return "original"


def _resolve_zeros(
    module: BaseQuantLinear,
    scales: Optional[torch.Tensor],
    zeros: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return explicit zero points, synthesizing symmetric ones when omitted.

    ``scales`` follows the internal packing contract with shape
    ``[out_features, num_groups]``; symmetric GPTQ/AWQ zero points are
    ``2 ** (bits - 1)`` for every group.
    """
    if zeros is not None:
        return zeros

    if not getattr(module, "sym", True):
        raise ValueError(
            "hf_pack_layer: asymmetric packing (`sym=False`) requires explicit `zeros`."
        )
    if scales is None:
        raise ValueError(
            "hf_pack_layer: `scales` is required when `zeros` is omitted for symmetric packing."
        )
    if scales.dim() != 2:
        raise ValueError(
            "hf_pack_layer: `scales` must be 2-D `[out_features, num_groups]`, "
            f"got shape {tuple(scales.shape)}."
        )

    bits = int(getattr(module, "bits", 0))
    if bits <= 0:
        raise ValueError("hf_pack_layer: `module.bits` must be a positive integer.")

    out_features, num_groups = scales.shape
    return torch.full(
        (out_features, num_groups),
        float(2 ** (bits - 1)),
        dtype=torch.int32,
        device=scales.device,
    )


def _module_quant_method(module: BaseQuantLinear) -> METHOD:
    methods = getattr(type(module), "SUPPORTS_METHODS", None) or []
    if len(methods) == 1:
        method = methods[0]
        return method if isinstance(method, METHOD) else METHOD(str(method).lower())
    if isinstance(module, AWQuantLinear):
        return METHOD.AWQ
    if isinstance(module, GPTQQuantLinear):
        return METHOD.GPTQ
    raise NotImplementedError(
        f"hf_ packing API: cannot infer the quantization method of `{type(module).__name__}`."
    )


def _is_repack_target(kernel_cls: Type[BaseQuantLinear], fmt: FORMAT, method: METHOD) -> bool:
    """Whether `kernel_cls` can receive already-packed tensors in `fmt`."""
    if method == METHOD.GPTQ and hasattr(kernel_cls, "repack_from_gptq"):
        return True
    if method == METHOD.AWQ and hasattr(kernel_cls, "repack_from_awq"):
        return True
    return fmt in (getattr(kernel_cls, "SUPPORTS_FORMATS", None) or {})


def _module_checkpoint_format(module: BaseQuantLinear, method: METHOD) -> FORMAT:
    checkpoint_format = getattr(module, "format", None)
    if checkpoint_format is not None:
        return _normalize_format(checkpoint_format)

    # Planar (split-plane `gptq_p`) modules do not always carry `format`, but
    # their packed words only round-trip through the planar layout.
    if getattr(module, "planar", False):
        return FORMAT.GPTQ_P

    # Fall back to the source kernel's own best-supported format so the
    # compatibility checks below compare like with like.
    supported = getattr(type(module), "SUPPORTS_FORMATS", None) or {}
    if supported:
        best_format, _ = max(
            supported.items(), key=lambda item: (int(item[1]), str(item[0].value))
        )
        return _normalize_format(best_format)

    return FORMAT.GEMM if method == METHOD.AWQ else FORMAT.GPTQ


def _select_kernel_class(
    *,
    bits: int,
    group_size: int,
    desc_act: bool,
    sym: bool,
    format: FORMAT,
    quant_method: METHOD,
    backend: Optional[Union[str, BACKEND]],
    device,
    dtype: Optional[torch.dtype],
    pack_dtype: Optional[torch.dtype],
    dynamic: Optional[Dict[str, Dict[str, Any]]],
    allow_marlin: bool,
    pack: bool,
    multi_select: bool = False,
    is_sharded: bool = False,
):
    return select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        device=device,
        backend=backend,
        format=format,
        quant_method=quant_method,
        pack=pack,
        allow_marlin=allow_marlin,
        dynamic=dynamic,
        pack_dtype=pack_dtype,
        dtype=dtype,
        multi_select=multi_select,
        is_sharded=is_sharded,
    )


def hf_check_packing_feasibility(
    bits: int,
    group_size: int = -1,
    desc_act: bool = False,
    sym: bool = True,
    *,
    format: Union[str, FORMAT] = FORMAT.GPTQ,
    quant_method: Union[str, METHOD] = METHOD.GPTQ,
    backend: Optional[Union[str, BACKEND]] = BACKEND.AUTO,
    device=None,
    dtype: Optional[torch.dtype] = None,
    pack_dtype: Optional[torch.dtype] = None,
    dynamic: Optional[Dict[str, Dict[str, Any]]] = None,
    in_features: Optional[int] = None,
    out_features: Optional[int] = None,
    allow_marlin: bool = True,
    is_sharded: bool = False,
    repack: bool = False,
) -> bool:
    """Return whether a kernel can pack the given quantization contract.

    Stable API for external integrators (auto-round, optimum, ...).  This is a
    non-raising probe around :func:`gptqmodel.utils.importer.select_quant_linear`
    with ``pack=True``: it answers whether a kernel exists that both accepts the
    contract and supports packing on the target device.

    Args:
        bits: Quantization bit width.
        group_size: Quantization group size; ``-1`` for per-channel.
        desc_act: Whether activation ordering (``g_idx``) is enabled.
        sym: Whether zero points are symmetric.
        format: Checkpoint/packing format (``"gptq"``, ``"gptq_v2"``, ...).
        quant_method: Quantization method (``"gptq"``, ``"awq"``, ...).
        backend: Kernel backend name or ``BACKEND`` value; ``"auto"`` probes
            the full auto-selection order.
        device: Target device (``"cpu"``, ``"cuda:0"``, ``DEVICE``, ...).
        dtype: Activation/output dtype the kernel must support.
        pack_dtype: Storage word dtype for packed tensors.
        dynamic: Per-module overrides forwarded to kernel selection.
        in_features: Optional layer input width for shape validation.
        out_features: Optional layer output width for shape validation.
        allow_marlin: Allow Marlin kernels during selection.
        is_sharded: Require support for sharded checkpoints.
        repack: Probe repack targets instead of packing implementations.
            ``pack()`` kernels (Torch, ATen, Triton, BitBLAS, ...) pack raw
            weights; GPU-only kernels such as Marlin and ExllamaV2 instead
            receive GPTQ-packed tensors and convert them in ``post_init()``.
            Set ``repack=True`` to ask "can this backend be a repack target".

    Returns:
        ``True`` when at least one kernel can pack the contract, ``False``
        otherwise.  Invalid arguments (unknown format/method/backend) raise
        ``ValueError``.
    """
    fmt = _normalize_format(format)
    method = _normalize_method(quant_method)
    resolved_pack_dtype = _resolve_pack_dtype(pack_dtype, method, fmt)
    resolved_backend = normalize_backend(backend, quant_method=method)
    wants_shape_validation = in_features is not None or out_features is not None

    try:
        selected = _select_kernel_class(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            format=fmt,
            quant_method=method,
            backend=resolved_backend,
            device=device,
            dtype=dtype,
            pack_dtype=resolved_pack_dtype,
            dynamic=dynamic,
            allow_marlin=allow_marlin,
            pack=not repack,
            multi_select=wants_shape_validation or repack,
            is_sharded=is_sharded,
        )
    except Exception as exc:
        log.debug(
            "hf_check_packing_feasibility: no kernel for method=%s format=%s backend=%s device=%s: %s",
            method.value,
            fmt.value,
            backend,
            device,
            exc,
        )
        return False

    candidates = selected if isinstance(selected, list) else [selected]

    if repack:
        candidates = [kernel_cls for kernel_cls in candidates if _is_repack_target(kernel_cls, fmt, method)]
        if not candidates:
            log.debug(
                "hf_check_packing_feasibility: no repack target for method=%s format=%s backend=%s.",
                method.value,
                fmt.value,
                backend,
            )
            return False

    if not wants_shape_validation:
        return bool(candidates)

    for kernel_cls in candidates:
        ok, err = validate_quant_linear(
            kernel_cls,
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=resolved_pack_dtype,
            dtype=dtype,
            dynamic=None,
            device=device,
        )
        if ok:
            return True
        log.debug(
            "hf_check_packing_feasibility: `%s` rejected in_features=%s out_features=%s: %s",
            kernel_cls.__name__,
            in_features,
            out_features,
            err,
        )
    return False


def hf_check_best_packing_format(
    bits: int,
    group_size: int = -1,
    desc_act: bool = False,
    sym: bool = True,
    *,
    quant_method: Union[str, METHOD] = METHOD.GPTQ,
    device=None,
    dtype: Optional[torch.dtype] = None,
    pack_dtype: Optional[torch.dtype] = None,
    dynamic: Optional[Dict[str, Dict[str, Any]]] = None,
    in_features: Optional[int] = None,
    out_features: Optional[int] = None,
    formats: Optional[Sequence[Union[str, FORMAT]]] = None,
    allow_marlin: bool = True,
) -> FORMAT:
    """Return the best performing feasible packing format for a contract.

    Stable API for external integrators (auto-round, optimum, ...).  Formats
    are ranked by the priority declared by their fastest kernel
    (``SUPPORTS_FORMATS``); ties are broken deterministically by format name.
    The winning format is validated with
    :func:`hf_check_packing_feasibility` on the requested device.

    Args:
        bits: Quantization bit width.
        group_size: Quantization group size; ``-1`` for per-channel.
        desc_act: Whether activation ordering (``g_idx``) is enabled.
        sym: Whether zero points are symmetric.
        quant_method: Quantization method (``"gptq"``, ``"awq"``, ...).
        device: Target device (``"cpu"``, ``"cuda:0"``, ``DEVICE``, ...).
        dtype: Activation/output dtype the kernel must support.
        pack_dtype: Storage word dtype for packed tensors.
        dynamic: Per-module overrides forwarded to kernel selection.
        in_features: Optional layer input width for shape validation.
        out_features: Optional layer output width for shape validation.
        formats: Restrict the search to these formats.
        allow_marlin: Allow Marlin kernels during selection.

    Returns:
        The preferred feasible :class:`~gptqmodel.quantization.FORMAT`.

    Raises:
        ValueError: When no candidate format is feasible for the contract.
    """
    method = _normalize_method(quant_method)

    if formats is None:
        candidates = list(BACKEND_TO_METHOD_FORMAT_MAPPING.get(method, {}).keys())
    else:
        candidates = [_normalize_format(candidate) for candidate in formats]
    if not candidates:
        raise ValueError(f"No packing formats are registered for quantization method `{method.value}`.")

    ranked: List[tuple] = []
    for fmt in candidates:
        kernels = AUTO_BACKEND_KERNEL_MAPPING.get(method, {}).get(fmt)
        if not kernels:
            continue
        best_priority = max(
            int(getattr(kernel_cls, "SUPPORTS_FORMATS", {}).get(fmt, 0))
            for kernel_cls in kernels.values()
        )
        ranked.append((-best_priority, fmt.value, fmt))
    ranked.sort()

    for _, _, fmt in ranked:
        resolved_pack_dtype = _resolve_pack_dtype(pack_dtype, method, fmt)
        if hf_check_packing_feasibility(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            format=fmt,
            quant_method=method,
            backend=BACKEND.AUTO,
            device=device,
            dtype=dtype,
            pack_dtype=resolved_pack_dtype,
            dynamic=dynamic,
            in_features=in_features,
            out_features=out_features,
            allow_marlin=allow_marlin,
        ):
            return fmt

    raise ValueError(
        "No feasible packing format for "
        f"method=`{method.value}`, bits=`{bits}`, group_size=`{group_size}`, "
        f"desc_act=`{desc_act}`, sym=`{sym}`, device=`{device}`."
    )


def hf_pack_layer(
    module: BaseQuantLinear,
    linear: nn.Module,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
    g_idx: Optional[torch.Tensor] = None,
    *,
    pack_impl: Optional[str] = None,
    device=None,
    block_in: int = 8192,
    workers: int = 1,
    scales_extra: Optional[torch.Tensor] = None,
    post_init: bool = False,
    post_init_kwargs: Optional[Dict[str, Any]] = None,
    checkpoint_format: Optional[Union[str, FORMAT]] = None,
) -> BaseQuantLinear:
    """Pack one layer into an already-created quantized kernel module.

    Stable API for external integrators (auto-round, optimum, ...).  This is the
    supported layer-wise entry point behind the internal model packing path: it
    dispatches to the same ``pack``/``pack_block``/``pack_gpu``/``pack_original``
    implementations and accepts the quantizer-native tensor layouts.

    Args:
        module: Quantized kernel instance (created via
            :func:`gptqmodel.utils.importer.hf_select_quant_linear` and
            instantiated by the caller) that receives the packed tensors.
        linear: Original ``nn.Linear``/``nn.Embedding``/``Conv1D`` layer holding
            the weights to pack.
        scales: Quantization scales with shape ``[out_features, num_groups]``.
        zeros: Quantization zero points with shape
            ``[out_features, num_groups]``.  When omitted and the module is
            symmetric (``sym=True``), ``2 ** (bits - 1)`` is used.
        g_idx: Activation-order group index of length ``in_features``.
        pack_impl: Packing implementation: ``"original"`` (default),
            ``"block"``/``"cpu"`` or ``"gpu"``.  GPU/block requests fall back to
            the original packer when unavailable.  Ignored by QQQ and AWQ
            kernels, which expose a single ``pack()`` implementation.
        device: Target CUDA device for ``pack_impl="gpu"``.
        block_in: Input channels per block for ``pack_impl="block"``.
        workers: Worker threads for ``pack_impl="block"``.
        scales_extra: Extra scales used by QQQ kernels.
        post_init: Run ``module.post_init()`` after packing.
        post_init_kwargs: Keyword arguments forwarded to ``module.post_init()``;
            kernels such as ExllamaV2 require a scratch space here.
        checkpoint_format: GPTQ checkpoint format to leave in the packed
            module. Defaults to the module's declared format, or ``"gptq"``.

    Returns:
        The same ``module`` instance, packed in place.  Packed buffers are
        registered on CPU (GPU packing only accelerates the packing math),
        matching the internal model packing path.

    Raises:
        TypeError: When ``module`` is not a :class:`BaseQuantLinear`.
        ValueError: When asymmetric packing is requested without ``zeros``, or
            when the module has no supported packing path.
    """
    if not isinstance(module, BaseQuantLinear):
        raise TypeError(
            f"hf_pack_layer: `module` must be a BaseQuantLinear, got `{type(module).__name__}`."
        )

    quant_type = getattr(type(module), "QUANT_TYPE", None)

    if quant_type == "qqq":
        module.pack(linear=linear, scales=scales, s_extra=scales_extra)
    elif quant_type is not None and (quant_type.startswith("awq_") or quant_type == "llm-awq"):
        module.pack(
            linear=linear,
            scales=scales,
            zeros=_resolve_zeros(module=module, scales=scales, zeros=zeros),
            g_idx=g_idx,
        )
    else:
        if not hasattr(module, "pack_original"):
            raise ValueError(
                f"hf_pack_layer: `{type(module).__name__}` does not implement a supported packing path."
            )

        resolved_zeros = _resolve_zeros(module=module, scales=scales, zeros=zeros)
        effective_impl = _normalize_pack_impl(pack_impl=pack_impl, module=module)

        if effective_impl == "gpu":
            try:
                module.pack_gpu(
                    linear=linear,
                    scales=scales,
                    zeros=resolved_zeros,
                    g_idx=g_idx,
                    device=device,
                )
            except (ValueError, NotImplementedError):
                log.warning(
                    "hf_pack_layer: GPU packing failed for `%s`; falling back to original pack.",
                    type(module).__name__,
                )
                module.pack_original(
                    linear=linear, scales=scales, zeros=resolved_zeros, g_idx=g_idx
                )
        elif effective_impl == "block":
            try:
                module.pack_block(
                    linear=linear,
                    scales=scales,
                    zeros=resolved_zeros,
                    g_idx=g_idx,
                    block_in=block_in,
                    workers=workers,
                )
            except (ValueError, NotImplementedError):
                log.warning(
                    "hf_pack_layer: block packing failed for `%s`; falling back to original pack.",
                    type(module).__name__,
                )
                module.pack_original(
                    linear=linear, scales=scales, zeros=resolved_zeros, g_idx=g_idx
                )
        else:
            module.pack_original(
                linear=linear, scales=scales, zeros=resolved_zeros, g_idx=g_idx
            )

    if isinstance(module, GPTQQuantLinear):
        # Every raw GPTQ packer writes the checkpoint/v1 qzero layout.  A
        # caller may reuse a module that was previously converted to v2, so
        # reset the metadata before deciding whether another conversion is
        # needed.  Without this, a second pack can skip v1->v2 or apply
        # v2->v1 to freshly-written v1 words.
        module.qzero_format(format=1)
        requested_format = checkpoint_format
        if requested_format is None:
            requested_format = getattr(module, "format", None) or FORMAT.GPTQ
        requested_format = _normalize_format(requested_format)
        if requested_format in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
            wanted_qzero_format = 1 if requested_format == FORMAT.GPTQ else 2
            if module.qzero_format() != wanted_qzero_format:
                from .model import (
                    convert_gptq_v1_to_v2_format_module,
                    convert_gptq_v2_to_v1_format_module,
                )
                if wanted_qzero_format == 1:
                    convert_gptq_v2_to_v1_format_module(
                        module=module,
                        quantize_config=QuantizeConfig(bits=module.bits),
                    )
                else:
                    convert_gptq_v1_to_v2_format_module(
                        module=module,
                        bits=module.bits,
                        pack_dtype=getattr(module, "pack_dtype", torch.int32),
                    )

    if post_init:
        module.post_init(**(post_init_kwargs or {}))

    return module


def hf_post_init(module: nn.Module, **post_init_kwargs) -> nn.Module:
    """Run the kernel post-load initialization for quantized layers.

    Stable API for external integrators (auto-round, optimum, ...).  Calls
    ``post_init()`` on every :class:`BaseQuantLinear` found in ``module``:
    passing a single quantized layer initializes that layer, while passing a
    container initializes each quantized child in depth-first order.  This is
    the layer-scoped counterpart of
    :func:`gptqmodel.utils.model.hf_gptqmodel_post_init`, which also performs
    model-level work such as activation-order and scratch-buffer setup.

    Args:
        module: A quantized layer or a container holding quantized layers.
        **post_init_kwargs: Keyword arguments forwarded to each layer's
            ``post_init()``; kernels such as ExllamaV2 require a scratch space.

    Returns:
        The same ``module``, initialized in place.

    Raises:
        TypeError: When ``module`` is not an ``nn.Module``.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"hf_post_init: `module` must be an nn.Module, got `{type(module).__name__}`.")

    if isinstance(module, BaseQuantLinear):
        module.post_init(**post_init_kwargs)
        return module

    for submodule in module.modules():
        if isinstance(submodule, BaseQuantLinear):
            submodule.post_init(**post_init_kwargs)
    return module


def _instantiate_kernel(
    kernel_cls: Type[BaseQuantLinear],
    module: BaseQuantLinear,
    *,
    backend: BACKEND,
    dtype: Optional[torch.dtype],
) -> BaseQuantLinear:
    group_size = getattr(module, "requested_group_size", getattr(module, "group_size", -1))
    init_kwargs: Dict[str, Any] = {
        "bits": module.bits,
        "group_size": group_size,
        "desc_act": getattr(module, "desc_act", False),
        "sym": getattr(module, "sym", True),
        "in_features": module.in_features,
        "out_features": module.out_features,
        "pack_dtype": getattr(module, "pack_dtype", torch.int32),
        "bias": getattr(module, "bias", None) is not None,
        "name": getattr(module, "name", None),
        "backend": backend,
        "register_buffers": True,
        "adapter": getattr(module, "adapter", None),
    }
    # Keep each kernel's default dtype when the caller did not request one;
    # some constructors (BitBLAS) reject an explicit ``None``.
    if dtype is not None:
        init_kwargs["dtype"] = dtype
    # GPTQ kernels need the checkpoint format to pick between the continuous
    # (gptq/gptq_v2) and planar (gptq_p) packed layouts.
    if issubclass(kernel_cls, GPTQQuantLinear):
        init_kwargs["format"] = _module_checkpoint_format(module, _module_quant_method(module))

    return kernel_cls(**init_kwargs)


def _copy_packed_state(source: BaseQuantLinear, target: BaseQuantLinear) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()

    if not any(key in source_state for key in ("qweight", "qzeros")):
        raise ValueError(
            f"hf_repack_layer: `{type(source).__name__}` has no packed tensors; "
            "pack the layer before repacking."
        )

    missing = [
        key
        for key in _PACKED_STATE_KEYS
        if key in source_state and key not in target_state
    ]
    if missing:
        raise NotImplementedError(
            f"hf_repack_layer: `{type(target).__name__}` does not expose the packed buffers "
            f"{missing} required to copy a `{type(source).__name__}` layout."
        )

    shared = {key: value for key, value in source_state.items() if key in target_state}
    target.load_state_dict(shared, strict=False)

    source_qzero_format = getattr(source, "qzero_format", None)
    target_qzero_format = getattr(target, "qzero_format", None)
    if callable(source_qzero_format) and callable(target_qzero_format):
        target_qzero_format(format=source_qzero_format())


def _resolve_single_torch_device(device) -> Optional[torch.device]:
    """Resolve one public device value, or ``None`` for multi-device selectors."""
    from ..models._const import DEVICE

    if device is None:
        return None
    if isinstance(device, DEVICE):
        return None if device == DEVICE.ALL else device.to_torch_device()
    if not isinstance(device, (str, int, torch.device)):
        return None

    # Reuse the selector normalization so string/int/ROCm handling matches
    # kernel selection exactly.
    resolved = _as_selector_device(device)
    if isinstance(resolved, DEVICE):
        return None if resolved == DEVICE.ALL else resolved.to_torch_device()
    return resolved


def _align_qzero_format(source: BaseQuantLinear, target: BaseQuantLinear) -> bool:
    """Match the target kernel's GPTQ v1/v2 qzero flavor after a copy.

    ``REQUIRES_FORMAT_V2`` kernels keep qzeros in the corrected v2 domain while
    other kernels keep the checkpoint v1 domain.  A cross-flavor copy must be
    corrected with the same helpers used by the loader/saver, otherwise the
    copied zero points are silently offset.
    """
    source_qzero_format = getattr(source, "qzero_format", None)
    target_qzero_format = getattr(target, "qzero_format", None)
    if not callable(source_qzero_format) or not callable(target_qzero_format):
        raise NotImplementedError(
            f"hf_repack_layer: GPTQ qzero format metadata is unavailable for "
            f"`{type(source).__name__}` or `{type(target).__name__}`."
        )
    source_format = source_qzero_format()
    target_format = 2 if getattr(type(target), "REQUIRES_FORMAT_V2", False) else 1
    if source_format == target_format:
        target_qzero_format(format=target_format)
        return False

    if not isinstance(target, GPTQQuantLinear):
        raise NotImplementedError(
            f"hf_repack_layer: `{type(target).__name__}` requires GPTQ v2 zero points but "
            "does not expose the qzero conversion API."
        )

    # Imported lazily: `gptqmodel.utils.model` pulls in the full loader stack.
    from .model import (
        convert_gptq_v1_to_v2_format_module,
        convert_gptq_v2_to_v1_format_module,
    )

    if source_format == 2 and target_format == 1:
        convert_gptq_v2_to_v1_format_module(
            module=target,
            quantize_config=QuantizeConfig(bits=target.bits),
        )
    elif source_format == 1 and target_format == 2:
        convert_gptq_v1_to_v2_format_module(
            module=target,
            bits=target.bits,
            pack_dtype=getattr(target, "pack_dtype", torch.int32),
        )
    else:
        raise ValueError(f"hf_repack_layer: unsupported qzero format `{source_format}`.")
    return True


def hf_repack_layer(
    module: BaseQuantLinear,
    backend: Optional[Union[str, BACKEND]] = None,
    *,
    device=None,
    dtype: Optional[torch.dtype] = None,
    post_init: bool = True,
    post_init_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseQuantLinear:
    """Repack an already-quantized layer for another kernel backend.

    Stable API for external integrators (auto-round, optimum, ...).  The layer
    must already carry its packed tensors (``qweight``/``qzeros``/``scales``/
    ``g_idx``).  Two repack paths are supported:

    * kernels exposing a native ``repack_from_gptq``/``repack_from_awq`` hook
      (for example BitBLAS), and
    * kernels that consume the same checkpoint format as the source module, in
      which case the packed tensors are copied and ``post_init()`` performs any
      kernel-specific layout conversion.

    Args:
        module: Source quantized kernel instance with loaded packed tensors.
        backend: Target kernel backend; defaults to the source module backend.
        device: Target device for the returned module.
        dtype: Activation/output dtype for the target kernel.
        post_init: Run ``target.post_init()`` after repacking.
        post_init_kwargs: Keyword arguments forwarded to ``target.post_init()``;
            kernels such as ExllamaV2 require a scratch space here.

    Returns:
        A new quantized module using ``backend``.  When the source already uses
        the requested backend the source module is returned unchanged.

    Raises:
        TypeError: When ``module`` is not a :class:`BaseQuantLinear`.
        NotImplementedError: When the target kernel cannot repack the source
            layout.
        ValueError: When no target kernel exists for the requested backend.
    """
    if not isinstance(module, BaseQuantLinear):
        raise TypeError(
            f"hf_repack_layer: `module` must be a BaseQuantLinear, got `{type(module).__name__}`."
        )

    quant_method = _module_quant_method(module)
    target_backend = (
        normalize_backend(backend, quant_method=quant_method)
        if backend is not None
        else module.backend
    )
    if target_backend is None:
        raise ValueError("hf_repack_layer: unable to resolve a target backend.")
    if target_backend == module.backend:
        return module

    checkpoint_format = _module_checkpoint_format(module, quant_method)
    source_pack_dtype = getattr(module, "pack_dtype", None)
    target_cls = _select_kernel_class(
        bits=module.bits,
        group_size=getattr(module, "requested_group_size", getattr(module, "group_size", -1)),
        desc_act=getattr(module, "desc_act", False),
        sym=getattr(module, "sym", True),
        format=checkpoint_format,
        quant_method=quant_method,
        backend=target_backend,
        device=device,
        dtype=dtype,
        pack_dtype=_resolve_pack_dtype(source_pack_dtype, quant_method, checkpoint_format),
        dynamic=None,
        allow_marlin=True,
        pack=False,
    )

    if target_cls is type(module):
        return module

    source_quant_type = getattr(type(module), "QUANT_TYPE", None)
    if source_quant_type in _ABI_TRANSFORMING_QUANT_TYPES:
        raise NotImplementedError(
            f"hf_repack_layer: `{type(module).__name__}` stores its packed tensors in a private "
            "runtime ABI after `post_init()`, so it cannot be used as a repack source. "
            "Repack from the kernel that holds the checkpoint layout instead."
        )

    target = _instantiate_kernel(target_cls, module, backend=target_backend, dtype=dtype)

    repack_hook = None
    if quant_method == METHOD.GPTQ and hasattr(target, "repack_from_gptq"):
        repack_hook = target.repack_from_gptq
    elif quant_method == METHOD.AWQ and hasattr(target, "repack_from_awq"):
        repack_hook = target.repack_from_awq

    if repack_hook is not None:
        repack_hook(module)
    else:
        if checkpoint_format not in target_cls.SUPPORTS_FORMATS:
            raise NotImplementedError(
                f"hf_repack_layer: `{target_cls.__name__}` cannot repack a "
                f"`{quant_method.value}` layer stored in format `{checkpoint_format.value}`; "
                f"supported formats: {[fmt.value for fmt in target_cls.SUPPORTS_FORMATS]}."
            )
        _copy_packed_state(module, target)
        if quant_method == METHOD.GPTQ:
            _align_qzero_format(module, target)

    target_device = _resolve_single_torch_device(device)
    if target_device is not None:
        target.to(target_device)

    if post_init:
        target.post_init(**(post_init_kwargs or {}))

    return target
