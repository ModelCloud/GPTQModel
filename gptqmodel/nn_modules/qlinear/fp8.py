# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.config import (
    _normalize_fp8_fmt,
    _normalize_fp8_weight_block_size,
    _normalize_fp8_weight_scale_method,
)
from ...quantization.dtype import (
    available_float8_dtype_names,
    dequantize_fp8,
    device_supports_native_fp8,
)
from ...utils.backend import BACKEND
from ...utils.fp8_block_triton import (
    TRITON_FP8_FORMATS,
    fp8_block_decode,
    fp8_block_decode_supported,
    fp8_block_gemv,
    fp8_block_gemv_payload_supported,
    triton_fp8_available,
)
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM
from ...utils.torch import TORCH_HAS_COMPILE, dynamo_is_compiling, torch_compile
from . import WeightOnlyQuantLinear
from ._compile_support import COMPILE_LEVEL
from ._numerics import autocast_active, matmul_fp32_reduce


log = setup_logger()

_FP8_ACTIVATION_QUANT_METHODS = ("none", "per_token")

#: Activation row counts served by the fused block-scale product instead of
#: the single-pass decode plus a GEMM. The decode-in-the-loop tiling is
#: bandwidth-bound, so it wins at decode sizes; a general GEMM overtakes it from
#: roughly M=128 on narrow projections and M=256 on wide ones (measured on
#: sm_89 across K/N from 4096/1024 to 14336/4096).
_FUSED_DECODE_MAX_ROWS = 64
_FUSED_DECODE_MAX_ROWS_WIDE = 192
_FUSED_DECODE_WIDE_OUT_FEATURES = 2048


#: ``(hip?, device index, output dtype) -> bool`` answers for the row-wise
#: ``torch._scaled_mm`` contract, measured once per process.
_SCALED_MM_ROW_SCALE_SUPPORT: dict = {}

#: ``GPTQMODEL_FP8_SCALED_MM``: ``auto`` (default) measures the contract,
#: ``on``/``off`` skip the measurement and force the answer. ``off`` keeps a
#: device with a broken row-wise kernel (for example #5174) on the correct
#: dequantize + matmul path without paying for a probe; ``on`` is for a
#: deployment that already knows the kernel is correct.
_FP8_SCALED_MM_ENV = "GPTQMODEL_FP8_SCALED_MM"


def _parse_scaled_mm_override(raw: Optional[str]) -> Optional[bool]:
    """Parse the ``GPTQMODEL_FP8_SCALED_MM`` override into ``True``/``False``/``None``."""

    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"", "auto"}:
        return None
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no"}:
        return False
    raise ValueError(
        f"`{_FP8_SCALED_MM_ENV}` must be one of auto/on/off, got `{raw}`."
    )


#: Resolved once at import so a traced ``forward`` reads a Python constant
#: instead of calling ``os.environ`` (an opaque call for dynamo).
_SCALED_MM_OVERRIDE: Optional[bool] = _parse_scaled_mm_override(
    os.environ.get(_FP8_SCALED_MM_ENV)
)


def _is_compiling() -> bool:
    """Whether this call is being traced by ``torch.compile``/``torch.export``."""

    return dynamo_is_compiling()


def _probe_must_wait() -> bool:
    """Whether the current context cannot host a device-to-host capability probe.

    A measurement ends with a device-to-host read of its verdict, and
    CUDA-graph capture forbids that copy (the pre-#5195 probe also created a
    ``torch.Generator``, which dynamo cannot trace). Inside either context the
    answer is therefore "unknown", not "unsupported": callers take the correct
    fallback and do not cache it, so the fast path is still available to the
    next eager call.
    """

    if _is_compiling():
        return True
    # ``is_initialized`` avoids creating a CUDA context for a CPU-only load;
    # an uninitialised context cannot be capturing.
    if not torch.cuda.is_initialized():
        return False
    return torch.cuda.is_current_stream_capturing()


def _probe_fp8_codes(rows: int, cols: int, device: torch.device, offset: int) -> torch.Tensor:
    """Deterministic FP8 codes for the contract probe.

    Every entry is ``n / 16`` for an integer ``n`` in ``1..16``, which is exact
    in ``float8_e4m3fn``. Unlike ``torch.randn``, this needs no
    ``torch.Generator``, so the probe is traceable (#5195).
    """

    index = torch.arange(rows * cols, device=device, dtype=torch.int64)
    codes = ((index * 7 + offset) % 16 + 1).to(torch.float32) * 0.0625
    return codes.reshape(rows, cols).to(torch.float8_e4m3fn)


def _probe_scaled_mm_row_scale(device: torch.device, out_dtype: torch.dtype) -> torch.Tensor:
    """Device-side verdict for the row-wise ``_scaled_mm`` contract.

    Returns a 0-dim bool tensor, so measuring performs no host synchronisation:
    no ``bool()``, no ``torch.allclose`` and no ``torch.Generator``. Only
    ``_scaled_mm_row_scale_verdict`` reads it, and only outside capture and
    tracing.
    """

    rows, inner, cols = 16, 32, 16
    x_q = _probe_fp8_codes(rows, inner, device, offset=1)
    w_q = _probe_fp8_codes(cols, inner, device, offset=5)
    scale_a = torch.linspace(0.5, 1.5, rows, device=device, dtype=torch.float32).reshape(rows, 1)
    scale_b = torch.linspace(0.5, 1.5, cols, device=device, dtype=torch.float32).reshape(1, cols)
    reference = (x_q.float() @ w_q.t().float()) * scale_a * scale_b
    out = torch._scaled_mm(
        x_q, w_q.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype
    )
    return torch.isclose(out.float(), reference, rtol=2e-2, atol=1e-2).all()


def _scaled_mm_row_scale_verdict(
    device: torch.device, out_dtype: torch.dtype
) -> Optional[bool]:
    """Whether this device computes the row-wise ``_scaled_mm`` contract.

    ``_scaled_mm`` takes ``scale_a=(tokens, 1)`` and ``scale_b=(1, out_features)``
    and is documented to apply both. On sm_121 with torch 2.11.0+cu130 the
    ``float16`` output silently drops the row-wise scaling: it returns a result
    ~1e3x away from ``(A @ B) * scale_a * scale_b`` and raises nothing (#5174),
    while the same call with ``bfloat16`` output is correct. A caller therefore
    cannot detect the defect from an exception, and the answer depends on the
    requested output dtype. The contract is measured once per process, device
    and output dtype and memoised on the device index, so the per-forward guard
    costs one dict lookup.

    ``None`` means "this context could not measure it" -- a CUDA-graph capture,
    a ``torch.compile`` trace, or an allocation failure. It is deliberately not
    cached: the fallback runs for this call and the next eager call asks again,
    so a probe that could not run never disables the fast path for the process
    (#5195).
    """

    if _SCALED_MM_OVERRIDE is not None:
        return _SCALED_MM_OVERRIDE

    index = device.index
    if index is None:  # pragma: no cover - tensors always carry an index
        index = torch.cuda.current_device()
    key = (IS_ROCM, index, out_dtype)
    cached = _SCALED_MM_ROW_SCALE_SUPPORT.get(key)
    if cached is not None:
        return cached

    if _probe_must_wait():
        return None

    try:
        supported = bool(_probe_scaled_mm_row_scale(device, out_dtype))
    except torch.cuda.OutOfMemoryError:
        # An allocation failure says nothing about the kernel contract.
        return None
    except Exception:
        # A build/kernel rejection is a real answer for this combination.
        supported = False

    if _probe_must_wait():
        # Defensive: the context turned into capture/tracing mid-measurement.
        return supported

    _SCALED_MM_ROW_SCALE_SUPPORT[key] = supported
    if not supported:
        log.warn.once(
            "FP8: `torch._scaled_mm` does not compute the row-wise scale contract on "
            f"this device for `{str(out_dtype).replace('torch.', '')}` output; per-token "
            "(W8A8) layers use the dequantize + matmul path."
        )
    return supported


def _scaled_mm_row_scale_supported(device: torch.device, out_dtype: torch.dtype) -> bool:
    """``True`` only when the contract is known and correct for this device.

    An unmeasured combination (``None`` above) reports ``False`` for this call
    but stays uncached, so a capture-time or traced call cannot poison the
    process-wide answer (#5195).
    """

    return _scaled_mm_row_scale_verdict(device, out_dtype) is True


def _normalize_fp8_activation_quant_method(value: Optional[str]) -> str:
    """Normalize the activation-quantization policy of a Torch FP8 layer.

    ``none`` keeps the layer weight-only (W8A16): the packed FP8 weight is
    dequantized for the current matmul and activations stay in FP16/BF16.
    ``per_token`` explicitly opts into W8A8 with one activation scale per
    token. There is deliberately no per-tensor activation mode: a per-tensor
    amax makes a row's output depend on its batch neighbours.
    """

    normalized = "none" if value is None else str(value).strip().lower()
    if normalized not in _FP8_ACTIVATION_QUANT_METHODS:
        raise ValueError(
            "FP8 activation quantization must be one of "
            f"{{{', '.join(repr(method) for method in _FP8_ACTIVATION_QUANT_METHODS)}}}, "
            f"got `{value}`."
        )
    return normalized


def _normalize_linear_scale_semantics(value: Optional[str]) -> str:
    """Normalize a loaded checkpoint's ``weight_scale_inv`` convention.

    A loaded HF/DeepSeek checkpoint can store a multiplier under the
    historical ``weight_scale_inv`` name, while legacy GPT-QModel checkpoints
    store an inverse. This wrapper decodes either.
    """

    normalized = "inverse" if value is None else str(value).strip().lower()
    if normalized not in {"inverse", "multiplier"}:
        raise ValueError(
            "FP8 weight scale semantics must be 'inverse' or 'multiplier', "
            f"got `{value}`."
        )
    return normalized
def _fp8_dtype_from_name(fmt: str) -> torch.dtype:
    return getattr(torch, _normalize_fp8_fmt(fmt))


def _weight_to_matrix(linear: nn.Module) -> torch.Tensor:
    weight = linear.weight.detach()
    if isinstance(linear, _ConvNd):
        weight = weight.flatten(1)
    if isinstance(linear, transformers.pytorch_utils.Conv1D):
        weight = weight.T
    return weight


def _compute_scale_inv(abs_max: torch.Tensor, fp8_max: float) -> torch.Tensor:
    abs_max = abs_max.to(torch.float32)
    eps = torch.finfo(torch.float32).tiny
    return torch.where(
        abs_max > 0,
        # tiny bounds the denominator but does not bound the quotient: even
        # fp8_max / tiny overflows FP32. Keep zero*scale finite too.
        (torch.full_like(abs_max, float(fp8_max)) / abs_max.clamp_min(eps)).clamp_max(torch.finfo(torch.float32).max),
        torch.ones_like(abs_max),
    )


class TorchFP8Linear(WeightOnlyQuantLinear):
    # FP8 stores the scaled weight and its inverse scale; either one missing
    # decodes to zeros or a wrong scale (the issue measured a 20.2 logit diff).
    CHECKPOINT_REQUIRED_BUFFERS = ("weight", "weight_scale_inv")

    # Measured on GB10 (sm_121), torch 2.11.0+cu130: eager, dynamo, aot_eager,
    # inductor and reduce-overhead pass, and an explicit torch.cuda.graph
    # capture succeeds with output matching eager.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.CUDAGRAPH
    # CPU, torch 2.16.0.dev20261021+cpu: the CPU forward is plain tensor
    # math and reaches ``INDUCTOR`` (dynamo/aot_eager/inductor pass); only
    # the CUDA path is capture-safe, so the base stays ``CUDAGRAPH``.
    SUPPORTS_COMPILE_LEVEL_BY_DEVICE = {"cpu": COMPILE_LEVEL.INDUCTOR}
    SUPPORTS_GRAPH_CAPTURE = True

    SUPPORTS_BACKENDS = [BACKEND.FP8_TORCH]
    SUPPORTS_METHODS = [METHOD.FP8]
    SUPPORTS_FORMATS = {FORMAT.FP8: 15}
    SUPPORTS_BITS = [8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    # Trace-clean: ``forward`` branches only on host-visible constants (device
    # type, dtypes, shapes, config strings) and every decision that needs a
    # device-to-host read -- the rowwise ``_scaled_mm`` contract and the
    # decoded-weight range bound -- is resolved eagerly by
    # ``_warm_runtime_decisions()`` at load/opt-in, never from the traced or
    # captured forward (#5195). Measured on
    # sm_121 (tiny Llama, 4 prompts x 32 greedy tokens): static-cache decode
    # 0.126 s eager -> 0.055 s compiled (2.3x, 1 graph, 0 breaks); the reporter
    # of #4669 measured 0.77 s -> 0.175 s on a 4090. FP8 is the format that
    # gains the most from the static-cache auto-compile, so opting out was a
    # 4x regression (#4739).
    SUPPORTS_COMPILE = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    # torch-npu 2.9 exposes FP8 dtypes on CPU, but CANN rejects float8 tensors
    # on Ascend devices, so FP8 is not an NPU-capable torch backend yet.
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    QUANT_TYPE = "fp8"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        format: str = "float8_e4m3fn",
        weight_scale_method: str = "row",
        weight_block_size: Optional[Tuple[int, int]] = None,
        weight_scale_semantics: str = "inverse",
        activation_quant_method: str = "none",
        **kwargs,
    ):
        self.fp8_format = _normalize_fp8_fmt(format)
        self.fp8_dtype = _fp8_dtype_from_name(self.fp8_format)
        block_size = _normalize_fp8_weight_block_size(weight_block_size)
        self.weight_scale_method = _normalize_fp8_weight_scale_method(
            weight_scale_method,
            weight_block_size=block_size,
        )
        self.weight_block_size = block_size
        self.weight_scale_semantics = _normalize_linear_scale_semantics(weight_scale_semantics)
        # Weight-only by default. ``per_token`` is the only supported W8A8
        # opt-in; see ``set_activation_quant_method``.
        self.activation_quant_method = _normalize_fp8_activation_quant_method(
            activation_quant_method
        )
        self._scaled_mm_hard_disabled = False
        # Set when a HIP build cannot cast this format natively, so later forwards
        # go straight to the validated CPU path instead of retrying the GPU cast.
        self._native_dequant_disabled = False
        # The block-scale decode plan is payload-derived; ``_invalidate_compiled``
        # raises the generation so a reload/move cannot reuse a stale answer.
        self._fp8_block_plan_generation = 0
        self._fp8_block_plan_key = None
        self._fp8_block_plan_value = (False, False)
        # Range-bound cache for the in-dtype weight narrowing. ``_apply`` and
        # ``_load_from_state_dict`` reset it, and ``post_init`` warms it, so a
        # captured/traced forward only reads it (#5195).
        self._decoded_weight_bound_ready = False
        self._decoded_weight_bound = None
        self._scale_inv_min = None

        if self.weight_scale_method == "block" and self.weight_block_size is not None:
            block_rows, block_cols = self.weight_block_size
            if out_features % block_rows != 0 or in_features % block_cols != 0:
                raise ValueError(
                    f"TorchFP8Linear block scaling requires out_features/in_features "
                    f"to be divisible by `weight_block_size={self.weight_block_size}`."
                )

        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.FP8_TORCH),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            **kwargs,
        )

        self._compiled_forward = None

        if register_buffers:
            self._allocate_buffers(bias=bias)

    @classmethod
    def validate_once(cls):
        if not available_float8_dtype_names():
            return False, RuntimeError("TorchFP8Linear requires a PyTorch build with FP8 dtypes.")
        return True, None

    def post_init(self):
        """Resolve the host-side runtime decisions after the weight is loaded.

        ``gptqmodel_post_init`` runs this on the final device, which is the last
        safe moment to measure a capability: a later ``torch.cuda.graph``
        capture or ``torch.compile`` trace must not synchronise the device
        (#5195).
        """

        super().post_init()
        self._warm_runtime_decisions()

    def _warm_runtime_decisions(self) -> None:
        """Measure the two device-to-host decisions ``forward`` must not make.

        The rowwise ``_scaled_mm`` contract probe and the decoded-weight FP16
        range bound both read device memory on the host. Doing that from
        ``forward`` broke a first-call CUDA-graph capture and a
        ``torch.compile(fullgraph=True)`` trace (#5195), so both are resolved
        here, in plain eager code. A capture/trace context is left untouched:
        it falls back for that call and asks again eagerly.
        """

        if _probe_must_wait():
            return
        weight = getattr(self, "weight", None)
        scale = getattr(self, "weight_scale_inv", None)
        if weight is None or scale is None or weight.device.type == "meta":
            return

        # ``_invalidate_compiled`` clears this cache on every device/dtype move.
        self._decoded_weight_absmax_bound()

        if self.activation_quant_method != "per_token" or weight.device.type != "cuda":
            return
        for out_dtype in self._scaled_mm_out_dtypes():
            _scaled_mm_row_scale_supported(weight.device, out_dtype)

    def smooth_block_size(self) -> int:
        if self.weight_scale_method == "block" and self.weight_block_size is not None:
            return self.weight_block_size[1]
        return -1

    def _scale_shape(self) -> tuple[int, ...]:
        if self.weight_scale_method == "tensor":
            return ()
        if self.weight_scale_method == "row":
            return (self.out_features,)
        if self.weight_block_size is None:
            raise ValueError("TorchFP8Linear block scaling requires `weight_block_size`.")
        block_rows, block_cols = self.weight_block_size
        return (
            self.out_features // block_rows,
            self.in_features // block_cols,
        )

    def _allocate_buffers(self, *, bias: bool) -> None:
        weight = torch.zeros((self.out_features, self.in_features), dtype=self.fp8_dtype)
        scale = torch.ones(self._scale_shape(), dtype=torch.float32)

        if "weight" in self._buffers:
            self.weight = weight
        else:
            self.register_buffer("weight", weight)

        if "weight_scale_inv" in self._buffers:
            self.weight_scale_inv = scale
        else:
            self.register_buffer("weight_scale_inv", scale)

        if bias:
            # Preserve the full checkpoint bias range. Forward narrows a
            # working copy to the activation dtype after reading the buffer.
            bias_tensor = torch.zeros(self.out_features, dtype=torch.float32)
            if "bias" in self._buffers:
                self.bias = bias_tensor
            else:
                self.register_buffer("bias", bias_tensor)
        else:
            self.bias = None

    def list_buffers(self):
        buffers = []
        if hasattr(self, "weight") and self.weight is not None:
            buffers.append(self.weight)
        if hasattr(self, "weight_scale_inv") and self.weight_scale_inv is not None:
            buffers.append(self.weight_scale_inv)
        if hasattr(self, "bias") and self.bias is not None:
            buffers.append(self.bias)
        return buffers

    @property
    def qweight(self) -> torch.Tensor:
        """The packed FP8 weight, under the name PEFT's GPTQ dispatcher reads.

        ``TorchFP8Linear`` stores its codes in ``weight`` (the name the FP8
        checkpoint and ``WeightOnlyQuantLinear`` use), while every other
        trainable GPT-QModel kernel keeps them in ``qweight``. PEFT routes each
        ``BaseQuantLinear`` through ``peft.tuners.lora.gptq.dispatch_gptq``,
        which does ``target.qweight = target_base_layer.qweight`` and therefore
        raised ``AttributeError`` before any LoRA training could start (#4948).
        The setter exists because the dispatcher then writes that same tensor
        back: ``target`` is this module when it is not already a
        ``BaseTunerLayer``, so ``target.qweight = ...`` lands here. It rebinds
        ``weight`` and keeps the slot's kind, so a rebind cannot silently turn a
        parameter into a plain tensor.
        """

        return self.weight

    @qweight.setter
    def qweight(self, value: torch.Tensor) -> None:
        if "weight" in self._parameters:
            self.weight = torch.nn.Parameter(value, requires_grad=value.requires_grad)
        else:
            self.weight = value

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, format={self.fp8_format}, "
            f"weight_scale_method={self.weight_scale_method}, "
            f"activation_quant_method={self.activation_quant_method}"
        )

    def set_activation_quant_method(self, method: str) -> "TorchFP8Linear":
        """Select the activation arithmetic explicitly.

        ``"none"`` (the default) runs W8A16: the FP8 weight tile is dequantized
        for the current matmul and activations keep their input dtype.
        ``"per_token"`` opts into W8A8, quantizing each token to FP8 with its
        own scale before ``torch._scaled_mm``.
        """

        self.activation_quant_method = _normalize_fp8_activation_quant_method(method)
        self._scaled_mm_hard_disabled = False
        # The documented W8A8 opt-in: measure the contract now, while the caller
        # is in plain eager code, so a later capture/trace keeps the fast path
        # instead of falling back to dequantize + matmul (#5195).
        self._warm_runtime_decisions()
        return self

    def _weight_to_matrix(self, linear: nn.Module) -> torch.Tensor:
        return _weight_to_matrix(linear)

    def _refuse_fp8_encode(self, *_args, **_kwargs):
        raise ValueError(
            "FP8 is load-only: GPT-QModel no longer quantizes or exports FP8 "
            "weights. Load a production FP8 checkpoint (transformers "
            "`FineGrainedFP8`, DeepSeek/Qwen block scales, NVIDIA ModelOpt, "
            "MXFP8, or NVFP8) instead."
        )

    pack = _refuse_fp8_encode
    pack_block = _refuse_fp8_encode
    pack_gpu = _refuse_fp8_encode
    pack_original = _refuse_fp8_encode

    def _resolve_target(self, device=None, dtype=None) -> tuple[torch.device, torch.dtype]:
        target_device = self.weight.device if device is None else torch.device(device)
        target_dtype = torch.float32 if dtype is None else dtype
        return target_device, target_dtype

    def _expanded_scale_inv(self, *, target_device: torch.device, target_dtype: torch.dtype) -> torch.Tensor:
        scale_inv = self.weight_scale_inv
        if scale_inv.device != target_device or scale_inv.dtype != target_dtype:
            scale_inv = scale_inv.to(device=target_device, dtype=target_dtype)

        if self.weight_scale_method == "tensor":
            return scale_inv
        if self.weight_scale_method == "row":
            return scale_inv.view(-1, 1)
        if self.weight_block_size is None:
            raise ValueError("TorchFP8Linear block scaling requires `weight_block_size`.")

        block_rows, block_cols = self.weight_block_size
        expanded = scale_inv.repeat_interleave(block_rows, dim=0)
        expanded = expanded.repeat_interleave(block_cols, dim=1)
        rows, cols = self.weight.shape
        if expanded.shape[0] < rows:
            expanded = torch.cat([expanded, expanded[-1:].expand(rows - expanded.shape[0], -1)], dim=0)
        if expanded.shape[1] < cols:
            expanded = torch.cat([expanded, expanded[:, -1:].expand(-1, cols - expanded.shape[1])], dim=1)
        return expanded[:rows, :cols]

    def dequantize_weight(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device, target_dtype = self._resolve_target(device=device, dtype=dtype)

        # Older GPUs can store the module on CUDA but still miss native FP8 math.
        # In that case, dequantize on CPU directly to fp16/bf16 and only then move.
        prefer_cpu_dequant = target_device.type == "cpu"
        # PyTorch's CUDA cast path for E8M0 currently produces NaNs when used as a
        # dense weight matrix, so force the validated CPU LUT path for that format.
        if self.fp8_format == "float8_e8m0fnu":
            prefer_cpu_dequant = True
        if target_device.type == "cuda" and not IS_ROCM and not device_supports_native_fp8(target_device):
            prefer_cpu_dequant = True

        if not prefer_cpu_dequant and not self._native_dequant_disabled:
            try:
                return self._dequantize_native(target_device=target_device, target_dtype=target_dtype)
            except Exception as exc:
                # This is a *ROCm* probe only.  HIP builds advertise FP8 support
                # differently from CUDA and ``get_device_capability`` cannot be
                # trusted there, so the native cast is attempted once instead.
                # On gfx1201 both ``float8_e4m3fn`` and ``float8_e4m3fnuz`` casts
                # work (measured), so this only fires on builds that lack one.
                # On NVIDIA a failure is a real error (including OOM) and must
                # propagate instead of silently turning into a CPU copy.
                if not IS_ROCM or target_device.type != "cuda":
                    raise
                self._native_dequant_disabled = True
                log.warn.once(
                    f"FP8 native `{self.fp8_format}` dequantization on {target_device} failed "
                    f"({type(exc).__name__}: {exc}); falling back to CPU dequantization "
                    "for the rest of this module's lifetime."
                )

        return self._dequantize_cpu(target_device=target_device, target_dtype=target_dtype)

    def _dequantize_native(self, *, target_device: torch.device, target_dtype: torch.dtype) -> torch.Tensor:
        weight = self.weight if self.weight.device == target_device else self.weight.to(device=target_device)
        # FP8 -> FP32 is exact, so apply the fp32 scale in place instead of
        # materialising a full N x K fp32 scale with `repeat_interleave`.
        # ``weight_scale_semantics`` decides whether the stored tensor is an
        # inverse scale (divide) or a DeepSeek/HF multiplier (multiply).
        deq = weight.to(torch.float32)
        scale_inv = self.weight_scale_inv
        if scale_inv.device != target_device or scale_inv.dtype != torch.float32:
            scale_inv = scale_inv.to(device=target_device, dtype=torch.float32)

        multiply = self.weight_scale_semantics == "multiplier"
        if self.weight_scale_method == "tensor":
            if multiply:
                deq.mul_(scale_inv)
            else:
                deq.div_(scale_inv)
        elif self.weight_scale_method == "row":
            if multiply:
                deq.mul_(scale_inv.view(-1, 1))
            else:
                deq.div_(scale_inv.view(-1, 1))
        else:
            if self.weight_block_size is None:
                raise ValueError("TorchFP8Linear block scaling requires `weight_block_size`.")
            block_rows, block_cols = self.weight_block_size
            rows, cols = deq.shape
            if rows % block_rows == 0 and cols % block_cols == 0:
                view = deq.view(rows // block_rows, block_rows, cols // block_cols, block_cols)
                if multiply:
                    view.mul_(scale_inv[:, None, :, None])
                else:
                    view.div_(scale_inv[:, None, :, None])
            else:
                # Non-divisible block shapes keep the padded/cropped expansion
                # as a fallback; the common path above never allocates it.
                expanded = self._expanded_scale_inv(target_device=target_device, target_dtype=torch.float32)
                if multiply:
                    deq.mul_(expanded)
                else:
                    deq.div_(expanded)
        return deq.to(target_dtype).transpose(0, 1).contiguous()

    def _dequantize_cpu(self, *, target_device: torch.device, target_dtype: torch.dtype) -> torch.Tensor:
        # ``dequantize_fp8``'s ``scale`` argument always multiplies. Convert the
        # stored inverse-scale tensor into the arithmetic this module declares
        # so the CPU fallback is exactly ``_dequantize_native``.
        stored_scale = self.weight_scale_inv.to(device="cpu", dtype=torch.float32)
        if self.weight_scale_semantics == "inverse":
            stored_scale = torch.reciprocal(stored_scale)
        weight = dequantize_fp8(
            self.weight.to(device="cpu"),
            scale=stored_scale,
            axis=None if self.weight_scale_method == "block" else (0 if self.weight_scale_method == "row" else 0),
            target_dtype=target_dtype,
        )
        if target_device.type != "cpu":
            weight = weight.to(device=target_device)
        return weight.transpose(0, 1).contiguous()

    def _scaled_mm_weight_scale(self, *, device: torch.device) -> torch.Tensor:
        scale_inv = self.weight_scale_inv
        if scale_inv.device != device:
            scale_inv = scale_inv.to(device=device)
        scale = scale_inv.to(torch.float32)
        if self.weight_scale_semantics == "inverse":
            scale = torch.reciprocal(scale)
        if self.weight_scale_method == "tensor":
            # Rowwise ``_scaled_mm`` scaling requires a per-column scale_b even
            # when the weight scale itself is per-tensor.
            return scale.reshape(1, 1).expand(1, self.out_features).contiguous()
        if self.weight_scale_method == "row":
            return scale.view(1, -1).contiguous()
        raise NotImplementedError("scaled_mm is only used for tensorwise or rowwise FP8 scales.")

    def _quantize_input_for_scaled_mm(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp8_max = torch.finfo(self.fp8_dtype).max
        x_work = x_flat.to(torch.float32)
        # W8A8 always uses one activation scale per token. A per-tensor amax
        # would couple every row's output to the largest activation in the
        # batch, so it is never used here.
        scale_inv = _compute_scale_inv(x_work.abs().amax(dim=1, keepdim=True), fp8_max)
        x_q = torch.clamp(x_work * scale_inv, min=-fp8_max, max=fp8_max).to(self.fp8_dtype)
        return x_q, torch.reciprocal(scale_inv).to(torch.float32)

    def _scaled_mm_out_dtype(self, x_flat: torch.Tensor) -> torch.dtype:
        """Output dtype ``_forward_scaled_mm`` requests, and the probe's key.

        Rowwise scaling is the W8A8 contract: ``scale_a`` is ``(tokens, 1)`` and
        ``scale_b`` is ``(1, out_features)``. HIP only emits BF16 from that
        kernel; CUDA can keep FP32 so a bias is added before the final rounding.
        """

        return torch.bfloat16 if IS_ROCM else (
            torch.float32 if self.bias is not None else x_flat.dtype
        )

    def _scaled_mm_out_dtypes(self) -> tuple[torch.dtype, ...]:
        """Every output dtype ``_scaled_mm_out_dtype`` can return for this layer.

        Mirrors the per-input answer without an input tensor, so the contract
        can be measured at opt-in/load time. Bias-free CUDA layers request the
        activation dtype, which is FP16 or BF16; a bias keeps FP32 so the
        epilogue can add it before the final rounding.
        """

        if IS_ROCM:
            return (torch.bfloat16,)
        if self.bias is not None:
            return (torch.float32,)
        return (torch.float16, torch.bfloat16)

    def _can_use_scaled_mm(self, x_flat: torch.Tensor) -> bool:
        # ROCm support is decided by the guarded call in ``forward``, not by
        # ``get_device_capability`` (meaningless under HIP).  Measured on gfx1201
        # (R9700, torch 2.16/ROCm 7.16): ``_scaled_mm`` accepts the OCP
        # ``float8_e4m3fn``/``float8_e5m2`` encodings and raises
        # ``HIPBLAS_STATUS_NOT_SUPPORTED`` for the AMD-native
        # ``float8_e4m3fnuz``/``float8_e5m2fnuz`` ones, i.e. the opposite of
        # what the gfx9x CDNA path historically did.  The first failure is
        # logged once and the module then stays on dequantize + matmul.
        return (
            not self._scaled_mm_hard_disabled
            and hasattr(torch, "_scaled_mm")
            and x_flat.device.type == "cuda"
            # Tensorwise amax on an empty input would disable this path permanently.
            and x_flat.numel() > 0
            # ``torch._scaled_mm`` quantizes activations. It is only correct
            # for this weight-only layer when W8A8 was explicitly requested.
            and self.activation_quant_method == "per_token"
            and self.weight_scale_method in {"tensor", "row"}
            # The W8A8 contract is the row-wise scale layout. Some devices
            # accept it and return a wrong result without raising (#5174), so
            # the contract is measured once per process for the exact output
            # dtype this forward would request. Inside capture/tracing the
            # answer is unknown, so this call takes the correct fallback
            # without caching a "no" (#5195).
            and _scaled_mm_row_scale_supported(
                x_flat.device, self._scaled_mm_out_dtype(x_flat)
            )
            and self.fp8_format != "float8_e8m0fnu"
            and x_flat.dtype in {torch.float16, torch.bfloat16}
            and x_flat.shape[-1] == self.in_features
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
            # hipBLASLt's rowwise kernel emits BF16 only, so the FP32 bias
            # epilogue used on CUDA is unavailable. Biased HIP modules stay on
            # the W8A16 path instead of rounding the reduction early.
            and not (IS_ROCM and self.bias is not None)
            and not (
                torch.is_grad_enabled()
                and (x_flat.requires_grad or self.weight.requires_grad or self.weight_scale_inv.requires_grad)
            )
        )

    def _decoded_weight_absmax_bound(self) -> Optional[float]:
        """Upper bound on ``|decoded weight|`` without materialising it.

        Stored codes are bounded by ``finfo(fp8_dtype).max`` and the scale is
        either multiplied in or divided in, so only the (small) scale tensor is
        needed. ``None`` means no bound could be established, and callers keep
        the wide path.
        """
        if self._decoded_weight_bound_ready:
            return self._decoded_weight_bound
        bound: Optional[float] = None
        smallest_scale: Optional[float] = None
        try:
            code_max = float(torch.finfo(self.fp8_dtype).max)
        except (TypeError, ValueError):
            code_max = None
        if code_max is not None and self.weight_scale_inv.numel() > 0:
            scale = self.weight_scale_inv.detach().to(device="cpu", dtype=torch.float32).abs()
            smallest_scale = float(scale.min())
            if self.weight_scale_semantics == "multiplier":
                bound = code_max * float(scale.max())
            else:
                bound = float("inf") if smallest_scale == 0.0 else code_max / smallest_scale
        self._decoded_weight_bound = bound
        self._scale_inv_min = smallest_scale
        self._decoded_weight_bound_ready = True
        return bound

    def _weights_fit_activation_dtype(self, dtype: torch.dtype) -> bool:
        """Whether every decoded weight stays finite in *dtype* when narrowed."""
        if dtype == torch.bfloat16:
            # BF16 keeps the FP32 exponent range, so an FP32-decoded weight
            # cannot overflow it.
            return True
        if dtype != torch.float16:
            return False
        if not self._decoded_weight_bound_ready and _probe_must_wait():
            # Establishing the bound copies the scale to the host. While
            # capturing/tracing, keep the exact wide path instead (#5195).
            return False
        bound = self._decoded_weight_absmax_bound()
        return bound is not None and bound <= float(torch.finfo(torch.float16).max)

    def _native_gemm_eligible(
        self, x_flat: torch.Tensor, device_type: str | None = None
    ) -> bool:
        """Whether the decoded weight can stay in the activation dtype.

        A bias stored wider than the activation dtype is deliberately not part
        of this decision: ``matmul_fp32_reduce`` keeps such a bias in the FP32
        epilogue, so a biased layer gets the same native GEMM as a bias-free
        one (#4681). ``device_type`` is overridable for tests; callers pass the
        real tensor device.
        """

        device_type = x_flat.device.type if device_type is None else device_type
        return (
            x_flat.dtype in (torch.float16, torch.bfloat16)
            and device_type == "cuda"
            and not autocast_active(device_type)
            and self._weights_fit_activation_dtype(x_flat.dtype)
        )

    def _inverse_scales_are_nonzero(self) -> bool:
        """Whether every inverse scale is nonzero, so the epilogue can divide.

        A zero inverse scale decodes a stored code to ``inf``/``NaN`` on the
        full-weight path. Dividing the product by it would instead turn a zero
        product into ``NaN`` and a nonzero one into ``inf``, which differs for
        exactly the rows whose dot product cancels to zero. Those payloads keep
        the exact reference path.
        """

        if not self._decoded_weight_bound_ready:
            if _probe_must_wait():
                # Unknown while capturing/tracing; keep the exact reference
                # path instead of synchronising the device (#5195).
                return False
            self._decoded_weight_absmax_bound()
        return self._scale_inv_min is not None and self._scale_inv_min > 0.0

    def _block_scale_matches_weight(self) -> bool:
        """Whether the block scale has the exact per-block shape of the weight.

        Padded or cropped checkpoint scales are handled by
        ``_expanded_scale_inv`` on the reference path; the in-dtype block
        decode only accepts the exact divisible layout.
        """

        if self.weight_block_size is None:
            return False
        block_rows, block_cols = self.weight_block_size
        rows, cols = self.weight.shape
        if rows % block_rows or cols % block_cols:
            return False
        return self.weight_scale_inv.shape == (rows // block_rows, cols // block_cols)

    def _lean_epilogue_eligible(self, x_flat: torch.Tensor) -> bool:
        """Whether the scale can move from the weight into an output epilogue.

        This is the range-safe CUDA W8A16 case only. The FP8 codes are cast
        straight to the activation dtype, the GEMM runs on the transposed view
        with an FP32 output, and a row/tensor scale is applied to that output.
        Keeping the unscaled product in FP32 is what makes the epilogue safe:
        the codes carry the inverse scale, so the unscaled product can be
        larger than the activation dtype's finite range even when the decoded
        weight is in range.

        Everything this gate rejects keeps the existing dequantize-then-matmul
        path bit-for-bit: ROCm (native FP8 casts are not qualified there), the
        E8M0 format (its CUDA cast produces NaNs), a disabled native cast, a
        non-range-safe payload, zero inverse scales, padded block scales, and
        an autograd graph that needs the wide-GEMM ``Function``.
        """

        if IS_ROCM or self._native_dequant_disabled:
            return False
        if self.fp8_format == "float8_e8m0fnu":
            return False
        if not self._native_gemm_eligible(x_flat):
            return False
        if self.weight_scale_method not in {"row", "tensor", "block"}:
            return False
        if torch.is_grad_enabled() and (
            x_flat.requires_grad
            or self.weight.requires_grad
            or self.weight_scale_inv.requires_grad
        ):
            return False
        if (
            self.weight_scale_semantics == "inverse"
            and not self._inverse_scales_are_nonzero()
        ):
            return False
        if self.weight_scale_method == "block" and not self._block_scale_matches_weight():
            return False
        return True

    def _block_decode_plan(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[bool, bool]:
        """Payload facts for the block-scale decode, computed once per payload.

        The checks below inspect shapes and read the smallest scale tensor on
        the host, which is far too slow for a per-forward gate. They depend
        only on the stored payload and its placement, so the answer is cached
        and invalidated by ``_invalidate_compiled`` and by a weight/scale
        rebind (the cache key carries both storage addresses). Returns
        ``(decode, fused)``.
        """

        weight = self.weight
        scale = self.weight_scale_inv
        key = (
            weight.data_ptr(),
            scale.data_ptr(),
            device,
            dtype,
            scale.dtype,
            self._native_dequant_disabled,
            self._fp8_block_plan_generation,
        )
        if self._fp8_block_plan_key == key:
            return self._fp8_block_plan_value

        if _probe_must_wait():
            # Deciding needs the smallest scale on the host. Inside a graph
            # capture or a trace that copy is forbidden, so this call takes
            # the portable decode and does not cache the answer.
            return (False, False)
        value = self._compute_block_decode_plan(device=device, dtype=dtype)
        self._fp8_block_plan_key = key
        self._fp8_block_plan_value = value
        return value

    def _compute_block_decode_plan(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[bool, bool]:
        if self.weight_scale_method != "block" or self.weight_block_size is None:
            return (False, False)
        if self.fp8_format not in TRITON_FP8_FORMATS:
            return (False, False)
        if self._native_dequant_disabled:
            return (False, False)
        weight = self.weight
        scale = self.weight_scale_inv
        if weight.device != device or scale.device != device:
            return (False, False)
        if not self._block_scale_matches_weight():
            return (False, False)
        if self.weight_scale_semantics == "inverse" and not self._inverse_scales_are_nonzero():
            # A zero inverse scale decodes a code to inf and can turn a zero
            # product into NaN. Keep the exact reference path for it.
            return (False, False)
        if not triton_fp8_available(device):
            return (False, False)
        block_rows, block_cols = self.weight_block_size
        block = (int(block_rows), int(block_cols))
        # A checkpoint may store the tiny per-block scale narrower than FP32;
        # the forward converts one small tensor per call, so the plan only needs
        # the float kind, not the exact storage width.
        decode = (
            scale.is_floating_point()
            # The single-pass decode writes the decoded weight in the
            # activation dtype before the FP32-reduced GEMM. A decoded weight
            # above that dtype's finite range (for example 65536 under FP16)
            # is stored as ``inf`` and turns a finite dot product into NaN
            # (#5405). Those payloads keep the exact FP32 fallback.
            and self._decode_dtype_is_range_safe(dtype)
            and fp8_block_decode_supported(
                weight,
                scale.to(dtype=torch.float32) if scale.dtype != torch.float32 else scale,
                fp8_format=self.fp8_format,
                block_size=block,
                dtype=dtype,
            )
        )
        fused = (
            decode
            and fp8_block_gemv_payload_supported(
                weight,
                scale,
                fp8_format=self.fp8_format,
                block_size=block,
                dtype=dtype,
            )
            # The fused product keeps the codes in the activation dtype through
            # the whole reduction, so it needs the same range safety
            # ``_native_gemm_eligible`` asks for on the row/tensor epilogue.
            # ``decode`` already rejects an out-of-range payload; keeping this
            # term makes the fused gate independent of the decode gate.
            and self._native_gemm_eligible_dtype(dtype)
        )
        return (decode, fused)

    def _decode_dtype_is_range_safe(self, dtype: torch.dtype) -> bool:
        """Whether a decoded block weight can be narrowed into *dtype*.

        ``_forward_block_decode`` decodes into the activation dtype before the
        FP32-reduced GEMM, so it inherits the finite-range requirement of the
        native GEMM. FP32 and BF16 keep the FP32 exponent range and cannot
        overflow; only FP16 needs the decoded-weight bound that
        ``_weights_fit_activation_dtype`` establishes. A payload above it keeps
        the exact reference path, which decodes and multiplies in FP32 and
        narrows only the result.
        """

        if dtype in (torch.float32, torch.bfloat16):
            return True
        if dtype == torch.float16:
            return self._weights_fit_activation_dtype(dtype)
        return False

    def _native_gemm_eligible_dtype(self, dtype: torch.dtype) -> bool:
        """``_native_gemm_eligible`` for a dtype instead of a live tensor.

        ``autocast_active`` stays a per-forward check; every other input is a
        property of the payload, so the cached block-decode plan can hold it.
        """

        return dtype in (torch.float16, torch.bfloat16) and self._weights_fit_activation_dtype(
            dtype
        )

    def _block_decode_eligible(
        self, x_flat: torch.Tensor, *, fused: bool
    ) -> bool:
        """Whether the Triton block decode may serve this call.

        The decode keeps the reference arithmetic (FP32 scale application, one
        rounding into the activation dtype), so it is selected on the same
        payloads the portable block decode accepts, minus any payload whose
        decoded weight would overflow the activation dtype on that rounding.
        What it cannot do is run inside a traced/exported graph: the launch is
        an opaque Python call, so ``torch.compile``/``export`` keep the Torch
        decode and its exact graph behaviour. CUDA-graph capture is fine; the
        launch is graph-safe.

        ``fused`` selects the W8A16 product that folds the decode into the
        GEMM. It needs the exact divisible 128-wide block layout, a range-safe
        payload, and no autograd graph on the activation, weight, scale or
        bias; every other range-safe payload uses the single-pass decode
        followed by ``matmul_fp32_reduce``.
        """

        if self.weight_scale_method != "block" or _is_compiling():
            return False
        if not x_flat.is_contiguous() or x_flat.device != self.weight.device:
            return False
        if x_flat.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        decode, fused_ok = self._block_decode_plan(x_flat.device, x_flat.dtype)
        if not decode:
            return False
        if not fused:
            return True
        if not fused_ok or autocast_active(x_flat.device.type):
            return False
        limit = (
            _FUSED_DECODE_MAX_ROWS_WIDE
            if self.out_features >= _FUSED_DECODE_WIDE_OUT_FEATURES
            else _FUSED_DECODE_MAX_ROWS
        )
        if x_flat.shape[0] > limit:
            # A measured crossover: past these row counts a general GEMM beats
            # this kernel's decode-in-the-loop tiling, so the single-pass
            # decode is the faster of the two.
            return False
        return not (
            torch.is_grad_enabled()
            and (
                x_flat.requires_grad
                or self.weight.requires_grad
                or self.weight_scale_inv.requires_grad
                # The bias is added inside the opaque Triton epilogue, so a
                # trainable bias needs the differentiable decode path too
                # (#5402). A frozen bias is a constant and keeps the fused
                # product.
                or (self.bias is not None and self.bias.requires_grad)
            )
        )

    def _forward_block_decode(
        self, x_flat: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Single-pass block decode, then the shared FP32-reduced GEMM.

        The decoded weight is a temporary of this call only: it is released
        when the forward returns, so no decoded or expanded weight outlives its
        consuming operation.
        """

        device = x_flat.device
        weight = self.weight
        scale = self.weight_scale_inv
        if weight.device != device:
            weight = weight.to(device=device)
        if scale.device != device or scale.dtype != torch.float32:
            # The per-block scale is tiny next to the payload, and the kernel
            # divides in FP32 like the reference decode.
            scale = scale.to(device=device, dtype=torch.float32)
        block_rows, block_cols = self.weight_block_size
        codes = fp8_block_decode(
            weight,
            scale,
            block_size=(int(block_rows), int(block_cols)),
            inverse=self.weight_scale_semantics != "multiplier",
            dtype=x_flat.dtype,
        )
        return matmul_fp32_reduce(x_flat, codes.t(), bias=bias)

    def _forward_block_gemv(
        self, x_flat: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Block-scaled W8A16 product with no decoded-weight temporary."""

        device = x_flat.device
        weight = self.weight
        scale = self.weight_scale_inv
        if weight.device != device:
            weight = weight.to(device=device)
        if scale.device != device or scale.dtype != torch.float32:
            scale = scale.to(device=device, dtype=torch.float32)
        block_rows, block_cols = self.weight_block_size
        return fp8_block_gemv(
            x_flat,
            weight,
            scale,
            block_size=(int(block_rows), int(block_cols)),
            inverse=self.weight_scale_semantics != "multiplier",
            bias=bias,
        )

    def _forward_lean_scaled_matmul(
        self, x_flat: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decode FP8 codes straight to the activation dtype and scale the output.

        The cast, and the in-place block scaling, exist only for this call. No
        decoded weight is stored on the module, so a repeated forward re-reads
        the packed payload (and observes any mutation of it).
        """

        device = x_flat.device
        weight = self.weight
        if weight.device != device:
            weight = weight.to(device=device)
        # FP8 -> FP16/BF16 is exact (both hold more mantissa bits than E4M3 or
        # E5M2), so this pass does not lose weight precision. The scale is not
        # applied yet.
        codes = weight.to(dtype=x_flat.dtype)

        if self.weight_scale_method == "block":
            block_rows, block_cols = self.weight_block_size
            rows, cols = codes.shape
            scale = self.weight_scale_inv
            if self._block_decode_eligible(x_flat, fused=False):
                return self._forward_block_decode(x_flat, bias=bias)
            blocks = codes.view(
                rows // block_rows, block_rows, cols // block_cols, block_cols
            )
            # Keep the scale in FP32: the mixed-dtype in-place op promotes to
            # FP32, divides there and rounds once into the activation dtype.
            # The stored code is exact in FP16/BF16, so this is the same
            # arithmetic as the reference ``weight.float() / scale`` path, but
            # without the full FP32 weight temporary. ``codes`` was just
            # allocated for this call, so scaling it in place is safe.
            if self.weight_scale_semantics == "multiplier":
                blocks.mul_(scale[:, None, :, None])
            else:
                blocks.div_(scale[:, None, :, None])
            return matmul_fp32_reduce(x_flat, codes.t(), bias=bias)

        output = torch.mm(
            x_flat.reshape(-1, x_flat.shape[-1]),
            codes.t(),
            out_dtype=torch.float32,
        )
        scale = self.weight_scale_inv
        if scale.device != device or scale.dtype != torch.float32:
            scale = scale.to(device=device, dtype=torch.float32)
        if self.weight_scale_method == "row":
            scale = scale.view(-1)
        if self.weight_scale_semantics == "multiplier":
            output = output * scale
        else:
            output = output / scale
        if bias is not None:
            if bias.device != device:
                bias = bias.to(device=device)
            output = output + bias.to(dtype=torch.float32)
        return output.to(x_flat.dtype)

    def _forward_dequant_matmul(
        self, x_flat: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Block scales cannot move into an output epilogue, so a block-scaled
        # payload takes a fused product (no decoded temporary at all) or the
        # single-pass Triton decode here. Both run before the range-safe gate
        # because that gate owns the row/tensor epilogue and the FP32 fallback,
        # not this block layout.
        if self._block_decode_eligible(x_flat, fused=True):
            return self._forward_block_gemv(x_flat, bias=bias)
        if self._block_decode_eligible(x_flat, fused=False):
            return self._forward_block_decode(x_flat, bias=bias)
        if self._lean_epilogue_eligible(x_flat):
            return self._forward_lean_scaled_matmul(x_flat, bias=bias)
        if not self._native_gemm_eligible(x_flat):
            # Decode into temporary FP32 storage only when narrowing is not
            # range-safe: a decoded FP8 weight above the activation dtype's
            # finite range (for example 65536 under FP16) becomes inf and turns
            # a finite dot product into NaN. In-range weights take the native
            # GEMM with an FP32 reduction below, which also keeps a wide bias
            # wide.
            with torch.autocast(device_type=x_flat.device.type, enabled=False):
                weight = self.dequantize_weight(device=x_flat.device, dtype=torch.float32)
                output = torch.matmul(x_flat.float(), weight)
                if bias is not None:
                    output = output + bias.to(device=x_flat.device, dtype=torch.float32)
                return output.to(x_flat.dtype)
        weight = self.dequantize_weight(device=x_flat.device, dtype=x_flat.dtype)
        return matmul_fp32_reduce(x_flat, weight, bias=bias)

    def optimize(
        self,
        backend: str = "inductor",
        mode: Optional[str] = None,
        fullgraph: bool = False,
    ) -> "TorchFP8Linear":
        """Compile the CPU dequant+matmul path after the caller explicitly opts in."""

        if self.optimized:
            return self
        if not TORCH_HAS_COMPILE or self.weight.device.type != "cpu":
            return self

        compiled = torch_compile(
            self._forward_dequant_matmul,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
        )
        if compiled is self._forward_dequant_matmul:
            return self

        self._compiled_forward = compiled
        self.optimized = True
        log.info.once("Optimize: `TorchFP8Linear` CPU compilation triggered.")
        return self

    def _invalidate_compiled(self) -> None:
        self._compiled_forward = None
        self.optimized = False
        # A new payload, device or dtype invalidates the cached decode plan.
        self._fp8_block_plan_generation = getattr(self, "_fp8_block_plan_generation", 0) + 1
        self._fp8_block_plan_key = None
        self._fp8_block_plan_value = (False, False)
        # The decoded-weight range bound is derived from the loaded payload.
        self._decoded_weight_bound_ready = False
        self._decoded_weight_bound = None
        self._scale_inv_min = None

    def _apply(self, fn, recurse=True):
        # Module.to()/half()/bfloat16() normally cast every floating buffer.
        # FP8 codes, inverse scales, and the checkpoint bias are storage
        # metadata; move their bytes without changing their native dtypes.
        native_buffers = {
            id(value): (value.dtype, value.shape)
            for name in ("weight", "weight_scale_inv", "bias")
            if isinstance(value := self._buffers.get(name), torch.Tensor)
        }

        def preserve_storage(tensor):
            native = native_buffers.get(id(tensor))
            if native is None:
                return fn(tensor)
            dtype, shape = native
            raw = fn(tensor.reshape(-1).view(torch.uint8))
            if raw.dtype != torch.uint8 or raw.numel() != tensor.numel() * tensor.element_size():
                raise RuntimeError("FP8 buffer move changed its packed byte representation")
            return raw.view(dtype).reshape(shape)

        result = super()._apply(preserve_storage, recurse=recurse)
        self._invalidate_compiled()
        # A device/dtype move invalidates the range bound and can change the
        # ``_scaled_mm`` answer (device index, output dtype). Re-resolve both
        # while still in eager code so the next capture/trace sees a warm
        # cache (#5195).
        self._warm_runtime_decisions()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._invalidate_compiled()

    def _forward_scaled_mm(self, x_flat: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.weight.device == x_flat.device else self.weight.to(device=x_flat.device)
        x_q, scale_a = self._quantize_input_for_scaled_mm(x_flat)
        scale_b = self._scaled_mm_weight_scale(device=x_flat.device)
        out_dtype = self._scaled_mm_out_dtype(x_flat)
        output = torch._scaled_mm(
            x_q,
            weight.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
        )
        if self.bias is None:
            output = output.to(x_flat.dtype)
        return output

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        bias = self.bias
        if bias is not None and bias.device != x_flat.device:
            bias = bias.to(device=x_flat.device)

        if self._can_use_scaled_mm(x_flat):
            try:
                output = self._forward_scaled_mm(x_flat)
            except Exception as exc:
                self._scaled_mm_hard_disabled = True
                log.warn.once(
                    f"FP8 `torch._scaled_mm` is unavailable for `{self.fp8_format}` on "
                    f"{x_flat.device} ({type(exc).__name__}: {exc}); "
                    "falling back to dequantize + matmul for the rest of the process."
                )
                output = self._forward_dequant_matmul(x_flat, bias=bias)
                bias = None
        elif self._compiled_forward is not None:
            try:
                output = self._compiled_forward(x_flat, bias=bias)
                bias = None
            except Exception as exc:
                log.warn.once(
                    f"FP8 compiled CPU forward failed; falling back to eager: {exc}"
                )
                self._invalidate_compiled()
                output = self._forward_dequant_matmul(x_flat, bias=bias)
                bias = None
        else:
            output = self._forward_dequant_matmul(x_flat, bias=bias)
            bias = None

        if bias is not None:
            output = (output + bias.to(dtype=output.dtype)).to(x_flat.dtype)

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)


__all__ = ["TorchFP8Linear"]
