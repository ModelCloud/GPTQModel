# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


import math
import os
import warnings
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, HAS_CUDA, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.arch import gemm_fused_dtypes as _arch_gemm_fused_dtypes
from ...utils.arch import is_rocm_fused_arch, rocm_arch
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM
from ...utils.torch import torch_compile
from ._runtime_cache import tensor_cache_key


try:
    from ..triton_utils.dequant import dequant as triton_dequant

    _TRITON_DEQUANT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton_dequant = None
    _TRITON_DEQUANT_AVAILABLE = False


try:
    from ..triton_utils.gptq_gemm import MAX_IN_FEATURES as _ROCM_GEMM_MAX_IN_FEATURES
    from ..triton_utils.gptq_gemm import SMALL_M_MAX as _ROCM_GEMM_SMALL_M_MAX
    from ..triton_utils.gptq_gemm import gate_limits as _ROCM_GEMM_GATE_LIMITS
except Exception:  # pragma: no cover - optional dependency
    _ROCM_GEMM_SMALL_M_MAX = 0
    _ROCM_GEMM_MAX_IN_FEATURES = 0
    _ROCM_GEMM_GATE_LIMITS = None


log = setup_logger()


# ``gptqmodel.utils.arch`` owns the cached ``gcnArchName`` lookup. The alias is
# kept so existing imports of ``_rocm_arch_name`` inside this module (and by
# downstream users) keep working. It returns ``""`` rather than ``None`` because
# the historical signature always returned a string.
def _rocm_arch_name(device: torch.device) -> str:
    """Normalized ROCm architecture (``gfx950``) or ``""`` when unavailable."""
    return rocm_arch(device) or ""


# One-shot diagnostics for fallbacks: the forward path is hot, so each reason is
# logged at debug level at most once per process instead of every forward.
_LOGGED_FALLBACKS: set[str] = set()


def _log_fallback_once(reason: str, message: str) -> None:
    if reason in _LOGGED_FALLBACKS:
        return
    _LOGGED_FALLBACKS.add(reason)
    log.debug(message)


def _right_shift_unpack(values: torch.Tensor, shifts: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if values.device.type != "npu":
        return torch.bitwise_right_shift(values, shifts).to(dtype)

    # CANN does not implement bitwise_right_shift. Use int64 arithmetic shift
    # equivalents on NPU so GPTQ unpacking avoids torch-npu's CPU fallback path.
    values_i64 = values.to(torch.int64)
    shifts_i64 = shifts.to(device=values.device, dtype=torch.int64)
    divisors = torch.pow(torch.full_like(shifts_i64, 2), shifts_i64)
    return torch.floor_divide(values_i64, divisors).to(dtype)


class _LinearWeightMetadata:
    """Tensor-like metadata shim for integrations that only inspect `weight` attrs."""

    def __init__(self, module: "TorchLinear", transposed: bool = False):
        self._module = module
        self._transposed = transposed

    def _shape(self) -> torch.Size:
        shape = (self._module.out_features, self._module.in_features)
        if self._transposed:
            shape = (shape[1], shape[0])
        return torch.Size(shape)

    def _first_tensor(self) -> torch.Tensor | None:
        for name in ("qweight", "scales", "bias", "qzeros", "g_idx"):
            tensor = getattr(self._module, name, None)
            if tensor is not None:
                return tensor
        return None

    @property
    def device(self) -> torch.device:
        tensor = self._first_tensor()
        return tensor.device if tensor is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for name in ("bias", "scales", "qweight"):
            tensor = getattr(self._module, name, None)
            if tensor is not None:
                return tensor.dtype
        return torch.float16

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> torch.Size:
        return self._shape()

    @property
    def requires_grad(self) -> bool:
        return False

    @property
    def T(self) -> "_LinearWeightMetadata":
        return _LinearWeightMetadata(self._module, transposed=not self._transposed)

    def size(self, dim: int | None = None):
        shape = self._shape()
        return shape if dim is None else shape[dim]

    def __repr__(self) -> str:
        return (
            f"_LinearWeightMetadata(device={self.device}, dtype={self.dtype}, "
            f"shape={tuple(self.shape)})"
        )


class TorchLinear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 20, FORMAT.GPTQ_V2: 20, FORMAT.GPTQ_P: 20}
    SUPPORTS_BITS = [2, 3, 4, 5, 6, 7, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

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
        format: Optional[FORMAT] = None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            enable_wf_unsqueeze=kwargs.pop("enable_wf_unsqueeze", True),
            format=format,
            **kwargs)

        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8
        self._streaming_enabled = bool(int(os.environ.get("GPTQ_TORCH_STREAMING", "0")))
        self._stream_tile_cols = int(os.environ.get("GPTQ_TORCH_STREAM_TILE", "512"))
        self._stream_double_buffers = 2
        self._g_idx_long_cache = None
        self._g_idx_long_cache_state = None
        self._zeros_cache = None
        self._zeros_cache_state = None
        self._stream_dequant_streams = {}
        self._cache_enabled = False
        triton_flag = os.environ.get("GPTQ_TORCH_TRITON_DEQUANT")
        if triton_flag is None:
            self._triton_dequant_enabled = _TRITON_DEQUANT_AVAILABLE
        else:
            self._triton_dequant_enabled = (
                triton_flag not in {"0", "false", "False"}
            ) and _TRITON_DEQUANT_AVAILABLE
        self._lookahead_enabled = False
        self._rocm_gemv_enabled = bool(int(os.environ.get("GPTQ_TORCH_ROCM_GEMV", "1")))
        self._rocm_gemm_enabled = bool(int(os.environ.get("GPTQ_TORCH_ROCM_GEMM", "1")))
        self._lookahead_next = None
        self._weight_metadata = _LinearWeightMetadata(self)
        if bool(int(os.environ.get("GPTQ_TORCH_CACHE_WEIGHTS", "0"))):
            self.enable_weight_cache(True)
        if bool(int(os.environ.get("GPTQ_TORCH_LOOKAHEAD", "0"))):
            self.enable_lookahead(True)

        # if self.group_size != self.in_features:
        #     self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        # else:
        #     self.padded_infeatures = self.in_features

    def post_init(self):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        super().post_init()

        # torch benefits the most from torch.compile, enable it by default
        self.optimize()
        self.clear_weight_cache()

    @property
    def weight(self):
        return self._weight_metadata

    def dequantize_weight(self, num_itr: int = 1):
        # Triton dequant currently handles the common single-iteration layout.
        # Multi-iteration requests (num_itr > 1) are routed to the torch path below.
        if (
            num_itr == 1
            and self._triton_dequant_enabled
            and self._can_use_triton_dequant()
        ):
            return self._dequantize_weight_triton()

        # Eval-time fast path for 2/4/8-bit torch dequant.
        # This is also the fallback when Triton is enabled but not eligible.
        if not self.training and self.bits in (2, 4, 8):
            return self._dequantize_weight_cached_248(num_itr=num_itr)

        return super().dequantize_weight(num_itr=num_itr)

    def optimize(self, backend: str = None, mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return

        device_type = self.list_buffers()[0].device.type
        if device_type == "npu":
            # torch.compile/inductor is not reliable on torch-npu in supported
            # deployments; the eager Torch path is the portable NPU backend.
            self.optimized = True
            log.info.once("Optimize: `TorchLinear` torch.compile skipped on NPU.")
            return

        if backend is None:
            # MPS doesn't support inductor.
            backend = "inductor" if device_type != "mps" else "aot_eager"

        # compile dequantize
        self.dequantize_weight = torch_compile(self.dequantize_weight, backend=backend, mode=mode, fullgraph=fullgraph)

        if self.adapter:
            self.adapter.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

        super().optimize()

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self

        from ...utils.model import convert_gptq_v1_to_v2_format_module

        # IPEX kernel will use Torch for training only and switches back to IPEX for eval/inference
        # If the kernel inherits Torch kernel only for training and can do its own inference in v1 (IPEX, Marlin) then
        # we can support training for all these v1 kernels by enabling this flag. We need to switch qzero states
        # by overriding module train() and swapping qzero back between v1 and v2 (Torch kernel requires v2)
        if self.SUPPORTS_TRAINING_USE_TORCH_KERNEL:
            # training starts
            if mode:
                # one time clone v1 qzeros and save both v1 and v2 qzeros in memory
                if self.qzero_format() == 1:
                    if not hasattr(self, "qzeros_data_v1"):
                        self.qzeros_data_v1 = self.qzeros.data.clone()
                        convert_gptq_v1_to_v2_format_module(self, bits=self.bits, pack_dtype=self.pack_dtype)
                        self.qzeros_data_v2 = self.qzeros.data
                        self._stream_reset_cache()
                    else:
                        self.qzeros.data = self.qzeros_data_v2
                        self.qzero_format(format=2)
                        self._stream_reset_cache()

            # training switching to inference/eval
            else:
                if hasattr(self, "qzeros_data_v1"):
                    # switch qzero back to v1 for inference/eval
                    self.qzeros.data = self.qzeros_data_v1
                    self.qzero_format(format=1)
                    self._stream_reset_cache()

        self.clear_weight_cache()
        return super().train(mode=mode)

    def forward(self, x: torch.Tensor):
        # if x.size(-1) != self.padded_infeatures:
        #     x = F.pad(x, (0, self.padded_infeatures - self.in_features))

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        out = self._forward(x, out_shape)
        return out

    def _forward(self, x, out_shape):
        x = self._apply_rotation_to_input(x)
        if self._should_use_streaming(x):
            out = self._forward_streaming(x, out_shape)
        else:
            out = self._forward_eager(x, out_shape)

        return out

    def _forward_eager(self, x: torch.Tensor, out_shape):
        if self._can_use_rocm_gemv(x):
            from ..triton_utils.gptq_gemv import gptq_gemv

            out = gptq_gemv(
                x, self.qweight, self.scales, self.qzeros, self.g_idx, bits=self.bits
            ).reshape(out_shape)
            if self.bias is not None:
                out.add_(self.bias)
            if self.adapter:
                out = self.adapter.apply(x=x, out=out)
            return out

        if self._can_use_rocm_gemm(x):
            from ..triton_utils.gptq_gemm import gptq_gemm

            out = gptq_gemm(
                x, self.qweight, self.scales, self.qzeros, self.g_idx, bits=self.bits
            ).reshape(out_shape)
            if self.bias is not None:
                out.add_(self.bias)
            if self.adapter:
                out = self.adapter.apply(x=x, out=out)
            return out

        num_itr = self.g_idx.shape[0] // x.shape[-1]
        weights = self.dequantize_weight(num_itr=num_itr)
        if weights.device != x.device or weights.dtype != x.dtype:
            # Quantized modules can be staged on a different accelerator than the
            # caller tensor during multi-device kernel validation; matmul still
            # needs both operands on the same device and dtype.
            weights = weights.to(device=x.device, dtype=x.dtype)
        out = torch.matmul(x, weights).reshape(out_shape)
        del weights
        if self.bias is not None:
            bias = self.bias
            if bias.device != out.device or bias.dtype != out.dtype:
                bias = bias.to(device=out.device, dtype=out.dtype)
            out.add_(bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def _can_use_rocm_gemv(self, x: torch.Tensor) -> bool:
        return (
            self._rocm_gemv_enabled
            and x.shape == (1, self.in_features)
            and self._can_use_rocm_fused(x)
        )

    def _rocm_gemm_gate(self, x: torch.Tensor) -> tuple[int, int]:
        """Return ``(small_m_max, max_in_features)`` for this device's arch.

        Unregistered architectures yield ``(0, 0)`` so the fused GEMM stays off,
        matching the pre-RDNA4 behaviour.
        """
        if _ROCM_GEMM_GATE_LIMITS is None:
            return _ROCM_GEMM_SMALL_M_MAX, _ROCM_GEMM_MAX_IN_FEATURES
        return _ROCM_GEMM_GATE_LIMITS(_rocm_arch_name(x.device))

    def _rocm_gemm_dtype_supported(self, x: torch.Tensor) -> bool:
        """Return whether the fused GEMM is a measured win for this activation dtype.

        RDNA4's fused small-M GEMM is a uniform win in FP16 (1.11x-2.53x on
        gfx1201) but not in BF16 (0.43x-1.42x, median 0.85x), so BF16 stays on
        the split path there.  CDNA keeps both dtypes (its historical values are
        unchanged).  Unregistered architectures have an empty set.
        """
        return str(x.dtype).removeprefix("torch.") in _arch_gemm_fused_dtypes(_rocm_arch_name(x.device))

    def _can_use_rocm_gemm(self, x: torch.Tensor) -> bool:
        small_m_max, max_in_features = self._rocm_gemm_gate(x)
        return (
            self._rocm_gemm_enabled
            and 2 <= x.shape[0] <= small_m_max
            and x.shape[1] == self.in_features
            and self.in_features <= max_in_features
            and self._rocm_gemm_dtype_supported(x)
            and self._can_use_rocm_fused(x)
        )

    def _can_use_rocm_fused(self, x: torch.Tensor) -> bool:
        """Preconditions shared by the fused ROCm GPTQ GEMV / small-M GEMM kernels."""
        return (
            IS_ROCM
            and self._triton_dequant_enabled
            and self._can_use_triton_dequant()
            and self.bits in (2, 4, 8)
            and self.pack_dtype == torch.int32
            and self.qzero_format() == 2
            and self.in_features >= 32
            and self.in_features % 32 == 0
            and self.out_features >= 32
            and self.out_features % 32 == 0
            and self.g_idx.numel() == self.in_features
            and x.is_contiguous()
            and not x.requires_grad
            and self.scales.dtype == x.dtype
            and self.qweight.device == x.device
            and self._rocm_arch_is_fused(x)
        )

    def _rocm_arch_is_fused(self, x: torch.Tensor) -> bool:
        """Architecture gate for the fused kernels, with a one-shot debug reason.

        Only architecture rejection is logged: the other preconditions (bits,
        packing, shape, training mode) are ordinary fallbacks that fire on every
        model and would be noise. Unregistered architectures keep the previous
        behaviour and silently use dequantize + matmul.
        """
        if not IS_ROCM:
            _log_fallback_once("rocm-not-rocm", "ROCm fused GPTQ kernel skipped: not a ROCm host.")
            return False
        arch = _rocm_arch_name(x.device)
        if not arch:
            _log_fallback_once(
                "rocm-arch-unknown",
                "ROCm fused GPTQ kernel skipped: device architecture is unknown.",
            )
            return False
        if not is_rocm_fused_arch(arch):
            _log_fallback_once(
                f"rocm-arch-{arch}",
                f"ROCm fused GPTQ kernel skipped: {arch} is not a registered fused architecture.",
            )
            return False
        return True

    def _forward_streaming(self, x: torch.Tensor, out_shape):
        tile = max(64, min(self._stream_tile_cols, self.out_features))
        total_tiles = math.ceil(self.out_features / tile)
        device = x.device

        out = torch.empty((x.shape[0], self.out_features), dtype=x.dtype, device=device)
        buffers = self._stream_get_workspace(device=device, dtype=x.dtype, tile=tile)
        widths = [0 for _ in range(self._stream_double_buffers)]

        compute_stream = torch.cuda.current_stream(device=device)
        stream_dequant = self._stream_get_dequant_stream(device)
        zeros = self._stream_decode_qzeros()
        g_idx = self._stream_g_idx_long()
        stream_dequant.wait_stream(compute_stream)
        for tensor in (*buffers, zeros, g_idx, self.qweight, self.scales, self.wf_unsqueeze_neg_one):
            tensor.record_stream(stream_dequant)
        consumed = [None for _ in buffers]

        def schedule(tile_idx: int, buffer_idx: int):
            start = tile_idx * tile
            end = min(start + tile, self.out_features)
            with torch.cuda.stream(stream_dequant):
                if consumed[buffer_idx] is not None:
                    stream_dequant.wait_event(consumed[buffer_idx])
                widths[buffer_idx] = self._stream_dequantize_tile(
                    buffer=buffers[buffer_idx],
                    zeros=zeros,
                    g_idx=g_idx,
                    start=start,
                    end=end,
                    dtype=x.dtype,
                )

        schedule(0, 0)

        for tile_idx in range(total_tiles):
            buffer_idx = tile_idx % self._stream_double_buffers
            compute_stream.wait_stream(stream_dequant)
            width = widths[buffer_idx]
            start = tile_idx * tile

            out_slice = out.narrow(1, start, width)
            out_slice.zero_()
            torch.addmm(
                out_slice,
                x,
                buffers[buffer_idx].narrow(1, 0, width),
                beta=0.0,
                alpha=1.0,
                out=out_slice,
            )
            consumed[buffer_idx] = torch.cuda.Event()
            consumed[buffer_idx].record(compute_stream)

            next_tile = tile_idx + 1
            if next_tile < total_tiles:
                next_buffer_idx = next_tile % self._stream_double_buffers
                schedule(next_tile, next_buffer_idx)

        buffers.clear()
        out = out.reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def _stream_dequantize_tile(
        self,
        buffer: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> int:
        width = end - start
        qweight_tile = self.qweight.narrow(1, start, width)
        weight = _right_shift_unpack(
            qweight_tile.unsqueeze(1).expand(-1, self.pack_factor, -1),
            self.wf_unsqueeze_neg_one,
            self.dequant_dtype,
        )
        weight = torch.bitwise_and(weight, self.maxq)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        weight = weight[:self.in_features]

        zeros_tile = zeros.narrow(1, start, width)
        scales_tile = self.scales.narrow(1, start, width)

        dequant = scales_tile[g_idx] * (weight - zeros_tile[g_idx])
        buffer.narrow(1, 0, width).copy_(dequant.to(dtype))
        return width

    def _stream_decode_qzeros(self):
        cache_state = (tensor_cache_key(self.qzeros), self.scales.shape)
        if self._zeros_cache is not None and self._zeros_cache_state == cache_state:
            return self._zeros_cache

        zeros = _right_shift_unpack(
            self.qzeros.unsqueeze(2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
            self.dequant_dtype,
        )
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.qzeros.shape[0], -1)
        zeros = zeros[:, :self.scales.shape[1]]
        self._zeros_cache = zeros
        self._zeros_cache_state = cache_state
        return zeros

    def _stream_g_idx_long(self, target_device: torch.device = None):
        if target_device is None:
            if self.qweight is not None:
                target_device = self.qweight.device
            else:
                target_device = self.g_idx.device

        cache_state = (tensor_cache_key(self.g_idx), target_device)
        if self._g_idx_long_cache is not None and self._g_idx_long_cache_state == cache_state:
            return self._g_idx_long_cache

        if self.g_idx.device == target_device:
            self._g_idx_long_cache = self.g_idx.long()
        else:
            non_blocking = self.g_idx.device.type == "cpu" and target_device.type in {"cuda", "xpu", "npu"}
            self._g_idx_long_cache = self.g_idx.to(
                device=target_device,
                dtype=torch.long,
                non_blocking=non_blocking,
            )

        self._g_idx_long_cache_state = cache_state
        return self._g_idx_long_cache

    def _stream_reset_cache(self):
        self._zeros_cache = None
        self._zeros_cache_state = None
        self._g_idx_long_cache = None
        self._g_idx_long_cache_state = None

    def _maybe_offload_g_idx_to_cpu(self):
        if self.training or self.g_idx is None:
            return
        if self.g_idx.device.type not in {"cuda", "xpu", "npu"}:
            return
        # Keep original device g_idx when Triton dequant is active/usable.
        if self._triton_dequant_enabled and self._can_use_triton_dequant():
            return
        self.g_idx = self.g_idx.to(device="cpu")
        if self._g_idx_long_cache is not None:
            self._g_idx_long_cache_state = (tensor_cache_key(self.g_idx), self._g_idx_long_cache.device)

    def _device_cache_key(self, device: torch.device) -> int:
        if device.index is not None:
            return device.index
        return torch.cuda.current_device()

    def _stream_get_dequant_stream(self, device: torch.device) -> torch.cuda.Stream:
        key = self._device_cache_key(device)
        stream = self._stream_dequant_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._stream_dequant_streams[key] = stream
        return stream

    def _stream_get_workspace(
        self,
        device: torch.device,
        dtype: torch.dtype,
        tile: int,
    ) -> list[torch.Tensor]:
        return [
            torch.empty((self.in_features, tile), dtype=dtype, device=device)
            for _ in range(self._stream_double_buffers)
        ]

    def _should_use_streaming(self, x: torch.Tensor) -> bool:
        if not self._streaming_enabled:
            return False
        # Streaming decode assumes the 2/4/8-bit wf shift-buffer layout; 3-bit
        # and planar 5/6/7-bit widths use the eager dequant path instead.
        if self.bits not in (2, 4, 8):
            return False
        if x.device.type != "cuda":
            return False
        if not HAS_CUDA:
            return False
        if any(tensor.device != x.device for tensor in (self.qweight, self.qzeros, self.scales)):
            return False
        # Torch kernels with num_itr > 1 already tiled differently.
        if self.g_idx.shape[0] // x.shape[-1] != 1:
            return False
        return True

    def enable_streaming(self, enabled: bool = True, tile_cols: int = None):
        self._streaming_enabled = enabled
        if tile_cols is not None:
            self._stream_tile_cols = tile_cols
        self._stream_reset_cache()
        return self

    def enable_weight_cache(self, enabled: bool = True):
        """Compatibility shim: dense inference weight caching is no longer supported."""
        self._cache_enabled = False
        self.clear_weight_cache()
        if enabled:
            warnings.warn(
                "Dense weight caching is no longer supported; weights remain quantized between forwards.",
                FutureWarning,
                stacklevel=2,
            )
        return self

    def clear_weight_cache(self):
        """Invalidate permitted zero-point and group-index metadata caches."""
        self._stream_reset_cache()

    def enable_lookahead(self, enabled: bool = True):
        """Compatibility shim: speculative dense-weight prefetching is disabled."""
        self._lookahead_enabled = False
        if enabled:
            warnings.warn(
                "Dense weight lookahead is no longer supported; dequantization is local to each forward.",
                FutureWarning,
                stacklevel=2,
            )
        return self

    def set_lookahead_next(self, module: "TorchLinear"):
        """Accept legacy lookahead configuration without retaining target modules."""
        if module is None or isinstance(module, TorchLinear):
            return self

        if isinstance(module, Iterable):
            targets = tuple(m for m in module if m is not None)
            for target in targets:
                if not isinstance(target, TorchLinear):
                    raise TypeError("lookahead targets must be TorchLinear modules or None")
            return self

        raise TypeError("lookahead target must be TorchLinear, iterable of TorchLinear, or None")

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

    def _can_use_triton_dequant(self) -> bool:
        if not _TRITON_DEQUANT_AVAILABLE:
            return False
        if self.training:
            return False
        if self.qweight is None or self.qzeros is None or self.scales is None or self.g_idx is None:
            return False
        if self.qweight.device.type != "cuda":
            return False
        if self.bits not in (2, 3, 4, 8):
            return False
        # Planar 3-bit words do not match the continuous Triton decode layout.
        if self.planar:
            return False
        # Triton assumes complete zero-point words when computing group strides.
        if self.bits in (2, 4, 8) and self.out_features % self.pack_factor:
            return False
        if not (self.qweight.is_contiguous() and self.qzeros.is_contiguous() and self.scales.is_contiguous()):
            return False
        # g_idx is stored as int32 tensor; ensure it resides on the same device.
        if self.g_idx.device != self.qweight.device:
            return False
        return True

    def _dequantize_weight_triton(self) -> torch.Tensor:
        # Use the Triton helper to decode weights directly on device.
        dtype = self.scales.dtype
        weights = triton_dequant(
            dtype,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.pack_dtype_bits,
            self.maxq,
        )
        return weights

    def _dequantize_weight_cached_248(self, num_itr: int = 1) -> torch.Tensor:
        self._init_wf_unsqueeze_buffers()
        zeros = self._stream_decode_qzeros()
        g_idx_long = self._stream_g_idx_long(target_device=self.qweight.device)
        self._maybe_offload_g_idx_to_cpu()

        weight = torch.bitwise_and(
            _right_shift_unpack(
                self.qweight.unsqueeze(1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
                self.dequant_dtype,
            ),
            self.maxq,
        )
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if num_itr == 1:
            return self.scales[g_idx_long] * (weight[:self.in_features] - zeros[g_idx_long])

        num_dim = self.g_idx.shape[0] // num_itr
        out_dim = weight.shape[1] // num_itr
        weights = []
        for i in range(num_itr):
            row_start = i * num_dim
            row_end = (i + 1) * num_dim
            col_start = i * out_dim
            col_end = (i + 1) * out_dim if i < (num_itr - 1) else weight.shape[1]

            scale_i = self.scales[:, col_start:col_end]
            weight_i = weight[row_start:row_end, col_start:col_end]
            zeros_i = zeros[:, col_start:col_end]
            g_idx_i = g_idx_long[row_start:row_end]
            weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))

        return torch.cat(weights, dim=1)

def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchLinear):
            raise ValueError(
                "Only models loaded using TorchLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.GPTQ_TORCH."
            )

        if isinstance(module, TorchLinear):
            # Create a new Linear layer with dequantized weights
            new_module = nn.Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
            )
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = (
                nn.Parameter(module.bias.detach().to("cpu", torch.float16))
                if module.bias is not None else None
            )

            # Replace the module in the model
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


class TorchQuantEmbeddings(TorchLinear):
    """Quantized embedding layer backed by the Torch GPTQ dequant path."""

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 20, FORMAT.GPTQ_V2: 20, FORMAT.GPTQ_P: 20}
    # Embeddings are selected by module role in create_quant_module(), and must
    # not be discovered as a general backend: their forward contract accepts
    # integer token IDs, not hidden states.
    SUPPORTS_BACKEND_SELECTION = False
    SUPPORTS_BITS = [2, 3, 4, 5, 6, 7, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = []

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

    def forward(self, input_ids: torch.Tensor):
        return F.embedding(input_ids, self.dequantize_weight())


__all__ = ["TorchLinear", "TorchQuantEmbeddings", "dequantize_model"]
