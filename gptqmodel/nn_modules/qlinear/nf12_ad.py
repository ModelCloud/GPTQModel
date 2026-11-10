# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the nf12 reference implementation
# (https://github.com/dphnAI/nf12) by AlpinDale and the nf12 contributors.
"""NF12-AD (NyoomFloat12) inference linear layer.

NF12 is Together AI's lossless 12-bit BF16 weight format, implemented by
AlpinDale at https://github.com/dphnAI/nf12. This module stores the packed
NF12-AD weight as ``nf12_packed`` plus ``nf12_escapes`` and
``nf12_exp_base``, matching the upstream nf12 serialization names. On CUDA the vendored
fused decode-GEMM kernels run
directly from that storage; elsewhere a bounded decode feeds an FP32
accumulated matmul. No dense or dequantized weight is retained between
forwards.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.nf12_ad import NF12ADQuantResult, quantize_nf12_ad_weight, weight_to_matrix
from ...utils import nf12_ad as nf12_ad_runtime
from ...utils import nf12_ad_hip as nf12_ad_hip_runtime
from ...utils import nf12_codec
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import WeightOnlyQuantLinear
from ._numerics import autocast_active, linear_fp32_reduce


log = setup_logger()


def _weight_dtype_is_lossless(activation_dtype: torch.dtype) -> bool:
    """Whether every BF16 NF12-AD weight survives a cast to the activation dtype.

    ``float64`` is included even though :attr:`SUPPORTS_DTYPES` does not list it:
    a caller that already holds float64 activations must not have the decoded
    weight narrowed to float32 on an accelerator. CPU keeps its FP32 decode by
    design (see `forward` and `test_linear_fallback_keeps_float64_activation_precision`).
    """

    return activation_dtype in (torch.bfloat16, torch.float32, torch.float64)


def _autocast_enabled(device_type: str) -> bool:
    """Whether autocast is active, tolerating backends without autocast support.

    ``torch.is_autocast_enabled`` raises for a device type that the installed
    build does not know (for example ``npu`` on a stock Torch). Such a backend
    cannot hold an ambient autocast context, so report ``False`` rather than
    failing the fallback for a module that declares every device family.
    """

    try:
        return autocast_active(device_type)
    except (RuntimeError, TypeError):
        return False


def _autocast_narrows_weight_range(
    activation_dtype: torch.dtype, device_type: str
) -> bool:
    """Whether an active autocast context narrows BF16 NF12-AD weights.

    Autocast rewrites GEMM operands to its own dtype, so an FP16 context
    narrows a BF16 activation to FP16 before the product even though the
    activation dtype itself can hold every BF16 weight. The correctness of the
    narrow branch must follow the context's dtype, not the activation dtype.
    ``float64`` is exempt because autocast never narrows it.
    """

    if activation_dtype is torch.float64 or not _autocast_enabled(device_type):
        return False
    try:
        autocast_dtype = torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError, RuntimeError):
        # Autocast is active but its dtype cannot be queried: assume it narrows,
        # which only widens the GEMM this predicate protects.
        return True
    return not _weight_dtype_is_lossless(autocast_dtype)


def _hip_path(device: torch.device | str) -> bool:
    """True when the device is an RDNA4 GPU that must use the HIP port.

    RDNA4 lacks ``mma.sync``/``wgmma``, so the CUDA kernels cannot be loaded
    there at all. Selecting by GPU family (not by ``IS_ROCM`` alone) keeps the
    fused CUDA MMA path on NVIDIA while routing AMD to the HIP decode port.
    """

    return nf12_ad_hip_runtime.nf12_ad_hip_supported(device)


class NF12ADLinear(WeightOnlyQuantLinear):
    """Weight-only NF12-AD linear layer.

    The packed representation is the checkpoint format. ``decode`` is exact,
    so this path is numerically identical to running the original BF16
    weights apart from the kernel's accumulation order.
    """

    SUPPORTS_BACKENDS = [BACKEND.NF12_AD]
    SUPPORTS_METHODS = [METHOD.NF12_AD]
    SUPPORTS_FORMATS = {FORMAT.NF12_AD: 100}
    SUPPORTS_BITS = [12]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [16]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "nf12_ad"

    def __init__(
        self,
        bits: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        group_size: int = -1,
        desc_act: bool = False,
        sym: bool = True,
        exp_base: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        del group_size, desc_act, sym
        super().__init__(
            bits=int(bits),
            in_features=int(in_features),
            out_features=int(out_features),
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.NF12_AD),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            **kwargs,
        )
        self._compiled_forward = None
        self.requested_exp_base = exp_base
        self._exp_base_value: int | None = None
        self._exp_base_tensor: torch.Tensor | None = None
        self._exp_base_version: int | None = None
        if not bias:
            self.bias = None
        if register_buffers:
            self._allocate_buffers(bias=bias, bias_dtype=dtype or torch.bfloat16)

    @classmethod
    def _validate(cls, bits=12, group_size=-1, desc_act=False, sym=True, pack_dtype=None,
                  dtype=None, dynamic=None, in_features=None, out_features=None,
                  device=None, trainable=None, adapter=None):
        del group_size, desc_act, sym, dynamic
        ok, err = cls._validate_shared(
            pack_dtype=pack_dtype,
            dtype=dtype,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )
        if not ok:
            return ok, err
        if bits not in cls.SUPPORTS_BITS:
            return False, NotImplementedError(
                f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual bits = `{bits}`"
            )
        return cls._validate_shape_constraints(
            in_features=in_features,
            out_features=out_features,
        )

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------
    def _allocate_buffers(self, *, bias: bool, bias_dtype: torch.dtype) -> None:
        packed = torch.zeros(
            (self.out_features, self.in_features * 3 // 2),
            dtype=torch.uint8,
        )
        escapes = torch.zeros((0, 8), dtype=torch.int16)
        exp_base = torch.zeros((1,), dtype=torch.uint8)
        self.register_buffer("nf12_packed", packed)
        self.register_buffer("nf12_escapes", escapes)
        self.register_buffer("nf12_exp_base", exp_base)
        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=bias_dtype))
        else:
            self.bias = None

    def list_buffers(self):
        buffers = []
        for name in ("nf12_packed", "nf12_escapes", "nf12_exp_base", "bias"):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                buffers.append(tensor)
        return buffers

    def runtime_device(self):
        packed = getattr(self, "nf12_packed", None)
        return packed.device if isinstance(packed, torch.Tensor) else None

    def _invalidate_compiled(self) -> None:
        self._compiled_forward = None

    def clear_weight_cache(self) -> None:
        """Forget the exponent cached from a previous checkpoint or buffer value."""
        self._exp_base_value = None
        self._exp_base_tensor = None
        self._exp_base_version = None
        self._invalidate_compiled()

    def _weight_to_matrix(self, linear: nn.Module) -> torch.Tensor:
        return weight_to_matrix(linear)

    def _exp_base(self) -> int:
        tensor = self.nf12_exp_base
        if self._exp_base_tensor is not tensor:
            self._exp_base_value = None
            self._exp_base_tensor = tensor
            try:
                self._exp_base_version = tensor._version
            except RuntimeError:
                # Inference tensors have no version counter. Buffer replacement
                # is still detected by identity; in-place updates need an
                # explicit clear_weight_cache() call.
                self._exp_base_version = None
        elif self._exp_base_version is not None and tensor._version != self._exp_base_version:
            self._exp_base_value = None
            self._exp_base_version = tensor._version
        if self._exp_base_value is None:
            self._exp_base_value = int(tensor.item())
        return self._exp_base_value

    def pack(self, linear, scales=None, zeros=None, g_idx=None, **kwargs):
        self.pack_original(linear, scales, zeros, g_idx, **kwargs)

    def pack_block(self, linear, scales=None, zeros=None, g_idx=None, block_in=8192, workers=1, **kwargs):
        del block_in, workers
        self.pack_original(linear, scales, zeros, g_idx, **kwargs)

    def pack_gpu(self, linear, scales=None, zeros=None, g_idx=None, block_in=8192, device=None, **kwargs):
        del block_in, device
        self.pack_original(linear, scales, zeros, g_idx, **kwargs)

    @torch.inference_mode()
    def pack_original(self, linear, scales=None, zeros=None, g_idx=None, *, smooth=None):
        """Encode ``linear.weight`` into the packed NF12-AD buffers."""

        del scales, zeros, g_idx
        if smooth is not None:
            raise ValueError(
                "NF12-AD is a lossless format and does not support smoothing; "
                "disable `smooth` for this method."
            )
        weight = weight_to_matrix(linear)
        result = quantize_nf12_ad_weight(weight, exp_base=self.requested_exp_base)
        self._install_result(result, linear=linear)

    @torch.inference_mode()
    def load_quant_result(self, result: NF12ADQuantResult, *, bias: torch.Tensor | None = None):
        """Install a quantizer result without recomputing the encoding."""

        if result.packed.shape != (self.out_features, self.in_features * 3 // 2):
            raise ValueError(
                f"NF12ADLinear `{self.name}`: packed shape {tuple(result.packed.shape)} "
                f"does not match {(self.out_features, self.in_features * 3 // 2)}"
            )
        self._install_result(result, bias=bias)

    def _install_result(self, result: NF12ADQuantResult, *, linear=None, bias=None):
        if hasattr(self, "nf12_packed"):
            target_device = self.nf12_packed.device
        elif isinstance(getattr(self, "bias", None), torch.Tensor):
            target_device = self.bias.device
        else:
            target_device = torch.device("cpu")
        packed = result.packed.to(device=target_device, dtype=torch.uint8).contiguous()
        escapes = result.escapes.to(device=target_device, dtype=torch.int16).contiguous()
        if "nf12_packed" in self._buffers:
            self.nf12_packed = packed
        else:
            self.register_buffer("nf12_packed", packed)
        if "nf12_escapes" in self._buffers:
            self.nf12_escapes = escapes
        else:
            self.register_buffer("nf12_escapes", escapes)
        base = torch.tensor([result.exp_base], dtype=torch.uint8, device=packed.device)
        if "nf12_exp_base" in self._buffers:
            self.nf12_exp_base = base
        else:
            self.register_buffer("nf12_exp_base", base)
        source_bias = bias
        if source_bias is None and linear is not None:
            source_bias = getattr(linear, "bias", None)
        if source_bias is not None:
            value = source_bias.detach().to(device=packed.device).contiguous()
            if "bias" in self._buffers:
                self.bias = value
            else:
                self.register_buffer("bias", value)
        else:
            self.bias = None
        self.clear_weight_cache()

    # ------------------------------------------------------------------
    # export / inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def dequantize_weight(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Return the exact decoded BF16 matrix.

        This is an explicit export API: callers receive a dense tensor, but it
        is never installed as an inference cache.
        """

        target = self.nf12_packed.device if device is None else torch.device(device)
        packed = self.nf12_packed.to(device=target)
        escapes = self.nf12_escapes.to(device=target)
        if target.type == "cuda" and _hip_path(target):
            try:
                weight = nf12_ad_hip_runtime.decode(packed, escapes, self._exp_base())
            except Exception as exc:  # noqa: BLE001 - fall back to the codec
                log.warn.once(
                    f"NF12-AD HIP decode failed ({type(exc).__name__}: {exc}); "
                    "using the Torch codec."
                )
                weight = nf12_codec.decode(
                    packed, escapes, self._exp_base(), self.out_features, self.in_features
                )
        elif target.type == "cuda" and nf12_ad_runtime.available():
            try:
                weight = nf12_ad_runtime.decode(packed, escapes, self._exp_base())
            except Exception as exc:  # noqa: BLE001 - fall back to the codec
                log.warn.once(
                    f"NF12-AD native decode failed ({type(exc).__name__}: {exc}); "
                    "using the Torch codec."
                )
                weight = nf12_codec.decode(
                    packed, escapes, self._exp_base(), self.out_features, self.in_features
                )
        else:
            weight = nf12_codec.decode(
                packed, escapes, self._exp_base(), self.out_features, self.in_features
            )
        return weight.to(dtype=dtype).contiguous()

    def dequantize(self, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return self.dequantize_weight(dtype=dtype)

    def _native_forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor | None:
        if not x.is_cuda or x.dtype != torch.bfloat16:
            return None
        # Native kernels consume BF16 bias. A wider or different bias must stay
        # wide through the fallback epilogue to preserve cancellation and range.
        if bias is not None and bias.dtype != x.dtype:
            return None
        if self.nf12_packed.device != x.device:
            return None
        # RDNA4 has no mma.sync/wgmma, so the HIP port decodes to exact BF16 and
        # uses rocBLAS (SPEC.md 6.5). CUDA hosts keep the fused MMA kernel.
        if _hip_path(x.device):
            if not nf12_ad_hip_runtime.available(x.device):
                return None
            try:
                return nf12_ad_hip_runtime.linear(
                    x.contiguous(),
                    self.nf12_packed,
                    self.nf12_escapes,
                    self._exp_base(),
                    bias,
                )
            except Exception as exc:  # noqa: BLE001 - optional accelerator
                log.warn.once(
                    f"NF12-AD HIP linear failed ({type(exc).__name__}: {exc}); "
                    "using decode + matmul."
                )
                return None
        if not nf12_ad_runtime.available():
            return None
        try:
            return nf12_ad_runtime.linear(
                x.contiguous(),
                self.nf12_packed,
                self.nf12_escapes,
                self._exp_base(),
                bias,
            )
        except Exception as exc:  # noqa: BLE001 - native kernels are optional
            log.warn.once(
                f"NF12-AD native linear failed ({type(exc).__name__}: {exc}); "
                "using decode + matmul."
            )
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = getattr(self, "bias", None)
        if x.device.type == "meta":
            # Metadata-only modules (lazy load, offload shells) have packed
            # buffers with no values. There is nothing to decode or multiply,
            # and materializing `exp_base` or a decoded matrix would raise on
            # a meta tensor. Answer before the online rotation as well: it
            # preserves shape/dtype/device but its native Hadamard op has no
            # meta kernel, so a rotation-enabled shell would still raise.
            # `empty_linear_output` already applies the adapter, exactly once.
            return self.empty_linear_output(x)

        if self.online_full_had or self.online_partial_had:
            x = self._apply_rotation_to_input(x)

        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)

        out = self._native_forward(x, bias)
        if out is None:
            # Bounded scratch: the decoded matrix lives only for this forward.
            # NF12-AD checkpoints store BF16, whose range is far wider than
            # FP16. Narrowing the decoded weight to an FP16 activation dtype
            # before the product turns a finite BF16 value (for example 65536)
            # into `inf`, and flushes values below the FP16 subnormal range to
            # zero. Multiply wide whenever the activation dtype cannot hold
            # every BF16 weight exactly, and round only the final output.
            #
            # The wide scratch dtype follows the activation: float64 (outside
            # SUPPORTS_DTYPES but accepted) must not be narrowed to float32,
            # while every other wide case uses FP32. CPU keeps its established
            # FP32 decode for every activation dtype (`test_linear_fallback_
            # keeps_float64_activation_precision` pins that contract), so only
            # an accelerator promotes float64. A mismatched bias dtype already
            # forces the wide branch.
            wide_dtype = (
                torch.float64
                if x.dtype is torch.float64 and x.device.type != "cpu"
                else torch.float32
            )
            wide = (
                x.device.type == "cpu"
                or not _weight_dtype_is_lossless(x.dtype)
                or _autocast_narrows_weight_range(x.dtype, x.device.type)
                or (bias is not None and bias.dtype != x.dtype)
            )
            compute_dtype = wide_dtype if wide else x.dtype
            weight = self.dequantize_weight(device=x.device, dtype=compute_dtype)
            if wide:
                out_shape = x.shape[:-1] + (self.out_features,)
                # Autocast would narrow this wide GEMM back to the autocast
                # dtype and lose the weight range this branch exists to
                # preserve. CPU autocast narrows an FP32 matmul just like the
                # CUDA path, so the guard covers every device, not only
                # accelerators.
                autocast = (
                    torch.autocast(device_type=x.device.type, enabled=False)
                    if _autocast_enabled(x.device.type)
                    else nullcontext()
                )
                with autocast:
                    out = torch.mm(
                        x.reshape(-1, x.shape[-1]).to(compute_dtype),
                        weight.t(),
                    )
                    if bias is not None:
                        out = out + bias.to(compute_dtype)
                out = out.to(x.dtype).reshape(out_shape)
            else:
                out = linear_fp32_reduce(x, weight, bias)
            del weight

        if self.adapter is not None:
            out = self.adapter.apply(x=x, out=out)
        return out

    def post_init(self):
        super().post_init()
        # Validate the loaded checkpoint's packing geometry once.
        if self.nf12_packed.shape != (self.out_features, self.in_features * 3 // 2):
            raise ValueError(
                f"NF12ADLinear `{self.name}`: unexpected `nf12_packed` shape "
                f"{tuple(self.nf12_packed.shape)}"
            )

    def extra_repr(self) -> str:
        exponent = getattr(self, "nf12_exp_base", None)
        # Construction and disk offload can leave checkpoint buffers absent or
        # on meta; repr must not materialize a scalar from either state.
        available = (isinstance(exponent, torch.Tensor)
                     and exponent.device.type != "meta" and exponent.numel() == 1)
        exp_base = self._exp_base() if available else "unavailable"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={getattr(self, 'bias', None) is not None}, "
            f"exp_base={exp_base}"
        )


__all__ = ["NF12ADLinear"]
