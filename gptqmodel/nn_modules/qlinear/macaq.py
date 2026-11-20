# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-FileCopyrightText: 2026 the MACAQ authors
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# MACaQ MXFP4 inference kernels.
#
# Portions of this file are adapted from Huawei CSL's MACaQ reference
# implementation (https://github.com/huawei-csl/MACaQ, Apache License 2.0):
#   `quantize/macaq_mxfp4.py` (OCP MXFP4 packing / dequantization)
#   `vllm/macaq_weight_only.py` (W4A16 forward)
# Algorithm: "MACaQ: Fast, Model Agnostic Calibrated Quantization for
# Low-Precision Weights and Activations", Muller et al., arXiv:2611.10649.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization import macaq as macaq_core
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.macaq_triton import (
    TRITON_AVAILABLE,
    mxfp4_w4a4_linear,
    mxfp4_w4a16_linear,
    triton_mxfp4_available,
)
from ...utils.torch import dynamo_is_compiling
from . import WeightOnlyQuantLinear
from ._compile_support import COMPILE_LEVEL
from ._numerics import linear_fp32_reduce
from ._runtime_cache import tensor_cache_key


log = setup_logger()


# Extreme decoded MXFP4 magnitudes: every decoded weight is one of
# `E2M1_MAGNITUDES` times a power-of-two E8M0 scale.  Use the codebook's Python
# constants so this module also imports under `torch.device("meta")` (#5776).
_E2M1_MIN_NONZERO = macaq_core.E2M1_MIN_NONZERO_MAGNITUDE
_E2M1_MAX = macaq_core.E2M1_MAX_MAGNITUDE

# The activation dtypes the Torch fallback decodes a weight for. The answers
# are computed eagerly so a compiled graph reads them as Python constants
# (#5631), the same contract as the AWQ decode-range cache.
_DECODE_RANGE_DTYPES = (torch.float16, torch.bfloat16)


def _triton_failure_is_transient(exc: BaseException) -> bool:
    """Whether a failed Triton launch may succeed on a later forward.

    A transient allocation failure must not poison the module for the rest of
    the process: the next forward can have enough free memory. Everything
    else (a compilation/resource limit or a pointer/device mismatch) is a
    property of this kernel on this device and stays latched.
    """

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "outofmemory" in message


class MacaqLinear(WeightOnlyQuantLinear):
    """Weight-only-safe MACaQ Linear with a portable Torch fallback.

    The packed weight stays in its MXFP4 representation between calls; the
    dense rotated-frame weight is reconstructed only for the current forward
    and released immediately.
    """

    # Packed E2M1 weights and E8M0 scales are always stored; the activation
    # transforms (`BH`, fitted `pre_scale`) are registered only for the
    # configured rotation/pre-scale mode, so an absent one is skipped.
    CHECKPOINT_REQUIRED_BUFFERS = ("W_packed", "W_scales", "BH", "pre_scale")

    # Measured on GB10 (sm_121), torch 2.11.0+cu130: every compile level passes,
    # but explicit capture fails because `dequantize_mxfp4` copies the CPU
    # `E2M1_MAGNITUDES` table to the device on each call.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.CUDAGRAPH
    SUPPORTS_GRAPH_CAPTURE = False
    # The policy flag is about dynamo, not the eager capture: a static-cache
    # `generate()` traces this forward to one graph with zero breaks and the
    # compiled decode is up to 9x faster than the eager torch path (0.73 s vs
    # 6.67 s, #4984), so the auto-compile must stay on for `MACAQ_TORCH`.
    SUPPORTS_COMPILE = True

    SUPPORTS_BACKENDS = [BACKEND.MACAQ_TORCH]
    SUPPORTS_METHODS = [METHOD.MACAQ]
    SUPPORTS_FORMATS = {FORMAT.MXFP4_MACAQ: 100, FORMAT.MXFP4: 100}
    SUPPORTS_BITS = [4]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.uint8]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "macaq"
    # The Hadamard block and the smoothing pre-scale are built in
    # `compute_dtype` and multiply the activations at the activation dtype.
    # The bias is not: it is added after the FP32 reduction, so a checkpoint
    # FP32 bias must survive a BF16 load instead of being rounded (#5632).
    MODEL_DTYPE_BUFFERS = ("BH", "pre_scale")

    # Decode-range answers derived from the packed scales. The class defaults
    # keep a module usable before the first refresh; `clear_weight_cache()`
    # recomputes them on every load, move and state-dict copy (#5631).
    _packed_range_key = None
    _packed_range_fits = True
    _packed_range_fits_by_dtype = None

    def __init__(
        self,
        bits: int,
        group_size: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        block_size: int = macaq_core.MXFP4_BLOCK,
        had_block: int = macaq_core.MXFP4_BLOCK,
        activation_precision: str = "w4a4",
        pre_mode: str = "balance",
        pre_alpha: float = 0.5,
        rotation: str = "hadamard",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            bits=int(bits),
            in_features=int(in_features),
            out_features=int(out_features),
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.MACAQ_TORCH),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            dtype=dtype,
            **kwargs,
        )

        self.block_size = int(block_size)
        self.had_block = int(had_block)
        self.activation_precision = str(activation_precision).strip().lower()
        self.pre_mode = str(pre_mode).strip().lower()
        self.pre_alpha = float(pre_alpha)
        self.rotation = str(rotation).strip().lower()
        if self.block_size != macaq_core.MXFP4_BLOCK:
            raise ValueError("MacaqLinear: OCP MXFP4 requires `block_size=32`.")
        if self.had_block < 32 or self.had_block & (self.had_block - 1):
            raise ValueError("MacaqLinear: `had_block` must be a power of two of at least 32.")
        if self.in_features % self.block_size or self.in_features % self.had_block:
            raise ValueError(
                f"MacaqLinear: in_features {self.in_features} must be divisible by "
                f"block_size={self.block_size} and had_block={self.had_block}."
            )
        if self.activation_precision not in ("w4a4", "w4a16"):
            raise ValueError("MacaqLinear: `activation_precision` must be `w4a4` or `w4a16`.")
        if self.rotation not in ("hadamard", "none"):
            raise ValueError("MacaqLinear: `rotation` must be `hadamard` or `none`.")
        if self.pre_mode not in ("balance", "none"):
            raise ValueError("MacaqLinear: `pre_mode` must be `balance` or `none`.")

        self.compute_dtype = (
            dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
        )
        self._triton_disabled = False
        # Identity BH for the fused W4A4 Triton kernel when `rotation="none"`;
        # built on first use, never saved.
        self._kernel_identity_bh = None

        # The quant config decides which activation transforms exist:
        # `rotation="hadamard"` stores `BH` and `pre_mode="balance"` stores
        # `pre_scale`. With neither, the layer is plain OCP MXFP4 (`W_packed`,
        # `W_scales`, optional `bias`) and the checkpoint format is `mxfp4`.
        # When present they are small and materialized eagerly so the module
        # has a real device to allocate packed storage on later (the looper
        # moves the module to the quantization device before calling `pack`).
        device = next(iter(self._parameters.values())).device if self._parameters else torch.device("cpu")
        if self.rotation == "hadamard":
            self.register_buffer(
                "BH", macaq_core.make_hadamard(self.had_block, device=device, dtype=self.compute_dtype)
            )
        else:
            self.BH = None
        if self.pre_mode == "balance":
            self.register_buffer(
                "pre_scale", torch.ones(self.in_features, dtype=self.compute_dtype, device=device)
            )
        else:
            self.pre_scale = None
        if bias:
            self.register_buffer(
                "bias", torch.zeros(self.out_features, dtype=self.compute_dtype, device=device)
            )
        else:
            self.bias = None

        if register_buffers:
            self._allocate_buffers(bias=bias)

    # ------------------------------------------------------------------
    # buffers
    # ------------------------------------------------------------------
    def _storage_device(self) -> torch.device:
        for buffer in self._buffers.values():
            if buffer is not None:
                return buffer.device
        return torch.device("cpu")

    def _allocate_buffers(self, *, bias: bool, device: Optional[torch.device] = None) -> None:
        """Allocate packed storage outside inference mode."""

        if device is None:
            device = self._storage_device()
        groups = self.in_features // self.block_size
        with torch.inference_mode(False):
            if "W_packed" not in self._buffers:
                self.register_buffer(
                    "W_packed",
                    torch.zeros((self.out_features, self.in_features // 2), dtype=torch.uint8, device=device),
                )
            if "W_scales" not in self._buffers:
                # Store E8M0 as raw bytes so every safetensors writer can
                # serialize the checkpoint; kernels view them as E8M0.
                self.register_buffer(
                    "W_scales",
                    torch.zeros((self.out_features, groups), dtype=torch.uint8, device=device),
                )
            if bias and self.bias is None:
                self.register_buffer(
                    "bias", torch.zeros(self.out_features, dtype=self.compute_dtype, device=device)
                )

    def _validate_loaded_state(self) -> None:
        groups = self.in_features // self.block_size
        if tuple(self.W_packed.shape) != (self.out_features, self.in_features // 2):
            raise ValueError(
                f"MacaqLinear `{self.name}`: unexpected W_packed shape {tuple(self.W_packed.shape)}; "
                f"expected {(self.out_features, self.in_features // 2)}."
            )
        if tuple(self.W_scales.shape) != (self.out_features, groups):
            raise ValueError(
                f"MacaqLinear `{self.name}`: unexpected W_scales shape {tuple(self.W_scales.shape)}; "
                f"expected {(self.out_features, groups)}."
            )
        if self.pre_scale is not None and tuple(self.pre_scale.shape) != (self.in_features,):
            raise ValueError(f"MacaqLinear `{self.name}`: unexpected pre_scale shape.")
        if self.BH is not None and tuple(self.BH.shape) != (self.had_block, self.had_block):
            raise ValueError(f"MacaqLinear `{self.name}`: unexpected BH shape.")

    def post_init(self):
        self._validate_loaded_state()
        self._refresh_packed_range_fits()
        super().post_init()

    @property
    def plain_mxfp4(self) -> bool:
        """Whether the layer is plain OCP MXFP4 with no activation transform.

        True for ``rotation="none"`` with ``pre_mode="none"``, which trades
        accuracy for faster inference: the W4A16 forward is a plain MXFP4
        dequantize-matmul.
        """

        return self.BH is None and self.pre_scale is None

    # ------------------------------------------------------------------
    # packing
    # ------------------------------------------------------------------
    def pack(self, linear: nn.Module = None, *, macaq_result=None, **kwargs):
        """Fill the packed buffers from a :class:`MacaqQuantResult`."""

        del kwargs
        if macaq_result is None:
            raise ValueError(
                "MacaqLinear.pack requires a `macaq_result` produced by the MACaQ quantizer."
            )
        bias = None if linear is None else getattr(linear, "bias", None)
        if "W_packed" not in self._buffers:
            device = None if self._buffers else macaq_result.W_packed.device
            self._allocate_buffers(bias=bias is not None, device=device)
        self.load_quant_result(macaq_result, bias=bias)

    @torch.no_grad()
    def load_quant_result(self, result, bias: Optional[torch.Tensor] = None) -> None:
        device = self.W_packed.device
        if self.BH is None:
            eye = torch.eye(result.BH.shape[0], dtype=result.BH.dtype, device=result.BH.device)
            if not torch.equal(result.BH, eye):
                raise ValueError(
                    f"MacaqLinear `{self.name}`: `rotation=\"none\"` layer received a rotated result."
                )
        if self.pre_scale is None and result.pre_scale is not None:
            raise ValueError(
                f"MacaqLinear `{self.name}`: `pre_mode=\"none\"` layer received a D1 pre-scale."
            )
        self.W_packed.data.copy_(result.W_packed.to(device=device, dtype=torch.uint8))
        self.W_scales.data.copy_(result.W_scales.to(device=device).view(torch.uint8))
        if self.pre_scale is not None:
            if result.pre_scale is not None:
                self.pre_scale.data.copy_(result.pre_scale.to(device=device, dtype=self.compute_dtype))
            else:
                # Zero-sample fallback modules skip D1 inside a D1 layout.
                self.pre_scale.data.fill_(1.0)
        if self.BH is not None:
            self.BH.data.copy_(result.BH.to(device=device, dtype=self.compute_dtype))
        if bias is not None and self.bias is not None:
            # Keep the source dtype: an FP32 bias such as -257 is -256 in
            # BF16, and the rounded value no longer cancels the product
            # (#5632). `linear_fp32_reduce` adds a wider bias in FP32 and
            # rounds only the final output. A dtype change replaces the
            # buffer outside inference mode, like `_allocate_buffers`.
            stored = bias.detach().to(device=device)
            if stored.dtype == self.bias.dtype:
                self.bias.data.copy_(stored)
            else:
                with torch.inference_mode(False):
                    self.bias = stored.clone()
        # The packed storage is final for this checkpoint; decide once whether
        # a half activation dtype can hold it (#5631).
        self._refresh_packed_range_fits()

    # ------------------------------------------------------------------
    # packed decode range
    # ------------------------------------------------------------------
    def _compute_packed_range_fits(self, dtype: torch.dtype) -> bool:
        """Whether every decoded MXFP4 weight is finite and nonzero in ``dtype``.

        The checkpoint stores one E8M0 scale per 32-value block, so a decoded
        weight is at most ``6 * scale`` and its smallest nonzero magnitude is
        ``0.5 * scale``. Above the dtype's largest finite value the decode
        yields ``inf``; below its smallest subnormal it yields ``0``. Either
        corruption happens inside the decode, before the FP32 reduction can
        help, so the caller must decode wide instead (#5631). The bound is
        conservative: it reads the scales, not the packed codes, so a block
        whose codes stay small may pick the wide decode for nothing.
        """

        if dtype not in _DECODE_RANGE_DTYPES:
            return True
        scales = getattr(self, "W_scales", None)
        if scales is None or scales.numel() == 0 or scales.device.type == "meta":
            return True
        scale = macaq_core.decode_mxfp4_scales(
            scales, self.out_features, self.in_features // self.block_size
        ).abs()
        converted = scale.to(dtype)
        representable = (
            converted.isfinite()
            & (converted * _E2M1_MAX).isfinite()
            & ~((scale > 0) & ((converted * _E2M1_MIN_NONZERO) == 0))
        )
        # An E8M0 0xFF block decodes to NaN in every dtype; widening cannot
        # recover it, so it must not select the wide path.
        return bool((representable | torch.isnan(scale)).all())

    def _refresh_packed_range_fits(self) -> None:
        """Precompute the per-dtype answers a compiled forward reads.

        A host reduction is neither traceable nor legal during CUDA-graph
        capture, so the compiled path reads a plain Python constant (dynamo
        guards on it, and a change recompiles). This runs wherever the packed
        storage can change: ``post_init()``, ``load_quant_result()``, and a
        device/dtype move or state-dict load (both via ``_apply`` ->
        ``clear_weight_cache()``).
        """

        self._packed_range_key = None
        self._packed_range_fits_by_dtype = {
            dtype: self._compute_packed_range_fits(dtype) for dtype in _DECODE_RANGE_DTYPES
        }

    def clear_weight_cache(self) -> None:
        """Drop the derived decode-range answers after a load, move or mutation."""

        self._packed_range_key = None
        self._packed_range_fits = True
        if hasattr(self, "W_scales"):
            self._refresh_packed_range_fits()

    def _packed_range_fits_dtype(self, dtype: torch.dtype) -> bool:
        """Whether the packed weight decodes into ``dtype`` without range loss."""

        if dtype not in _DECODE_RANGE_DTYPES:
            return True
        scales = getattr(self, "W_scales", None)
        if scales is None or scales.device.type == "meta":
            return True
        if dynamo_is_compiling() or (
            scales.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        ):
            precomputed = self._packed_range_fits_by_dtype
            if precomputed is None:
                # A caller that skipped `post_init`: the wide decode is always
                # correct, so do not guess from a host read here.
                return False
            return precomputed[dtype]
        key = (tensor_cache_key(scales), dtype)
        if key == self._packed_range_key:
            return self._packed_range_fits
        fits = self._compute_packed_range_fits(dtype)
        self._packed_range_key = key
        self._packed_range_fits = fits
        # Keep the compiled-region answer in step with an eager recompute (a
        # caller that mutates the scales in place and runs an eager forward
        # before compiling).
        if self._packed_range_fits_by_dtype is not None:
            self._packed_range_fits_by_dtype[dtype] = fits
        return fits

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Decode the packed MXFP4 weight in the rotated frame (current call only)."""

        return macaq_core.dequantize_mxfp4(
            self.W_packed,
            self.W_scales,
            self.in_features,
            dtype=self.compute_dtype if dtype is None else dtype,
        )

    def _forward_dtype(self, x: torch.Tensor) -> torch.dtype:
        """Follow the activation dtype so a later ``.to()`` cannot go stale.

        ``compute_dtype`` is fixed when the module is built, while ``.to()``
        moves ``BH``/``pre_scale``/``bias`` to a new dtype.  Deriving the
        forward dtype from the input keeps both the Torch fallback and the
        Triton kernels' operands consistent and returns the activation dtype,
        even when the module was moved after ``__init__``.
        """

        if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return x.dtype
        return self.compute_dtype

    def _forward_torch(self, x_flat: torch.Tensor) -> torch.Tensor:
        dtype = self._forward_dtype(x_flat)
        # A finite MXFP4 weight that the activation dtype cannot hold decodes
        # to `inf`, and its zero lanes then turn the dot into `0 * inf = NaN`
        # (#5631). Keep the decode and the reduction in FP32 there and round
        # only the final output (`forward` rounds to `x.dtype`); the narrow
        # path stays untouched when the packed range fits.
        wide = not self._packed_range_fits_dtype(dtype)
        weight = self.dequantize(dtype=torch.float32 if wide else dtype)
        pre_scale, bh = self.pre_scale, self.BH
        if self.activation_precision == "w4a4":
            xb = macaq_core.macaq_quantize_activations(
                x_flat, pre_scale, bh, self.block_size
            ).to(dtype)
        else:
            xb = x_flat.to(dtype)
            if pre_scale is not None:
                xb = xb * pre_scale.to(dtype)
            if bh is not None:
                xb = macaq_core.rotate_blocks(xb, bh)
        if wide:
            # The activation transform above stays at the activation dtype;
            # only the dot's operands widen.
            xb = xb.to(torch.float32)
        # Pass the stored bias unchanged; `linear_fp32_reduce` keeps a wider
        # bias out of the low-precision operands and adds it in FP32 (#5632).
        return linear_fp32_reduce(xb, weight, self.bias)

    def _forward_triton(self, x_flat: torch.Tensor) -> torch.Tensor:
        # Like `_forward_torch`, the Triton path must not trust the build-time
        # `compute_dtype`: `.to(dtype)` moves the buffers but not the attribute.
        dtype = self._forward_dtype(x_flat)
        if self.activation_precision == "w4a4":
            # The fused rotate+quantize kernel needs a BH: its unfused
            # `bh=None` fallback costs more than rotating by an identity.
            bh = self.BH
            if bh is None:
                bh = self._kernel_identity_bh
                if bh is None or bh.device != x_flat.device or bh.dtype != dtype:
                    bh = torch.eye(self.had_block, dtype=dtype, device=x_flat.device)
                    self._kernel_identity_bh = bh
            return mxfp4_w4a4_linear(
                x_flat,
                self.W_packed,
                self.W_scales,
                pre_scale=self.pre_scale,
                bh=bh,
                bias=self.bias,
            )
        return mxfp4_w4a16_linear(
            x_flat,
            self.W_packed,
            self.W_scales,
            pre_scale=self.pre_scale,
            bh=self.BH,
            bias=self.bias,
            compute_dtype=dtype,
        )

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        return (
            not self._triton_disabled
            and self.backend == BACKEND.MACAQ_TRITON
            and TRITON_AVAILABLE
            and x.is_cuda
            and triton_mxfp4_available(x.device)
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        if x_flat.shape[0] == 0:
            output = self.empty_linear_output(x_flat)
        elif self._can_use_triton(x_flat):
            try:
                output = self._forward_triton(x_flat)
            except Exception as exc:
                if not _triton_failure_is_transient(exc):
                    self._triton_disabled = True
                log.warn.once(
                    f"MACaQ Triton kernel unavailable on {x_flat.device} "
                    f"({type(exc).__name__}: {exc}); using the portable Torch path."
                )
                output = self._forward_torch(x_flat)
        else:
            output = self._forward_torch(x_flat)

        output = output.to(x.dtype)
        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)
        return output.reshape(original_shape)


class MacaqTritonLinear(MacaqLinear):
    """NVIDIA CUDA and RDNA4 gfx1201 Triton MXFP4 Linear."""

    # Measured on GB10 (sm_121), torch 2.11.0+cu130: eager runs and an explicit
    # torch.cuda.graph capture succeeds, but dynamo breaks on
    # `torch.cuda.device_count()` (non-Tensor return): EAGER only.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = True
    # The torch path's #4984 opt-in covers `MacaqLinear`'s straight-line
    # MXFP4 dequant; this Triton decode breaks the dynamo graph on
    # `torch.cuda.device_count()`, so it must not inherit the opt-in.
    SUPPORTS_COMPILE = False

    SUPPORTS_BACKENDS = [BACKEND.MACAQ_TRITON]
    SUPPORTS_METHODS = [METHOD.MACAQ]
    SUPPORTS_FORMATS = {FORMAT.MXFP4_MACAQ: 100, FORMAT.MXFP4: 100}
    SUPPORTS_BITS = [4]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.uint8]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def validate_once(cls):
        if not TRITON_AVAILABLE:
            return False, RuntimeError("MacaqTritonLinear requires Triton.")
        if not torch.cuda.is_available():
            return False, RuntimeError("MacaqTritonLinear requires an accelerator device.")
        return True, None

    @classmethod
    def validate_device(cls, device):
        super().validate_device(device)
        if not triton_mxfp4_available(device):
            raise NotImplementedError(
                "MacaqTritonLinear requires NVIDIA CUDA or ROCm RDNA4 gfx1201 "
                "with Triton; use BACKEND.MACAQ_TORCH otherwise."
            )

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.MACAQ_TRITON)
        super().__init__(*args, **kwargs)
        self.backend = BACKEND.MACAQ_TRITON


__all__ = ["MacaqLinear", "MacaqTritonLinear"]
