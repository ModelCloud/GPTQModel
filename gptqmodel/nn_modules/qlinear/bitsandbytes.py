# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from packaging import version
from torch.nn.modules.conv import _ConvNd

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.config import (
    _normalize_bitsandbytes_block_size,
    _normalize_bitsandbytes_format,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import WeightOnlyQuantLinear
from .gguf import _apply_optional_smoother


log = setup_logger()

MINIMUM_BITSANDBYTES_VERSION = "0.49.0"
BITSANDBYTES_INSTALL_HINT = (
    "bitsandbytes is not installed or is too old. "
    "Install a recent 0.49.x build, for example `pip install bitsandbytes>=0.49.3`."
)

_BITSANDBYTES_4BIT_STATE_BUFFER_NAMES = (
    "weight_absmax",
    "weight_quant_map",
    "weight_nested_absmax",
    "weight_nested_quant_map",
    "weight_quant_state",
)
_BITSANDBYTES_8BIT_STATE_BUFFER_NAMES = ("weight_scb",)


def _is_bitsandbytes_available() -> bool:
    try:
        import bitsandbytes as bnb
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug("bitsandbytes import failed: %s", exc)
        return False

    return version.parse(bnb.__version__) >= version.parse(MINIMUM_BITSANDBYTES_VERSION)


def import_bitsandbytes():
    import bitsandbytes as bnb

    if version.parse(bnb.__version__) < version.parse(MINIMUM_BITSANDBYTES_VERSION):
        raise ImportError(BITSANDBYTES_INSTALL_HINT)
    return bnb


BITSANDBYTES_AVAILABLE = _is_bitsandbytes_available()


def _weight_to_matrix(linear: nn.Module) -> torch.Tensor:
    weight = linear.weight.detach()
    if isinstance(linear, _ConvNd):
        weight = weight.flatten(1)
    if isinstance(linear, transformers.pytorch_utils.Conv1D):
        weight = weight.T
    return weight


def _packed_state_key_to_buffer_name(key: str) -> str:
    if key.startswith("quant_state."):
        return "weight_quant_state"
    return f"weight_{key}"


@lru_cache(maxsize=256)
def _buffer_spec_4bit(
    *,
    in_features: int,
    out_features: int,
    quant_type: str,
    block_size: int,
    compress_statistics: bool,
) -> Tuple[Tuple[str, Tuple[int, ...], torch.dtype], ...]:
    bnb = import_bitsandbytes()

    # Quantize a template once so the registered buffers match the exact
    # packed layout bitsandbytes expects during checkpoint load.
    template = torch.zeros((out_features, in_features), dtype=torch.float16)
    qweight, quant_state = bnb.functional.quantize_4bit(
        template,
        blocksize=block_size,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        quant_storage=torch.uint8,
    )

    spec = [("weight", tuple(qweight.shape), qweight.dtype)]
    for key, tensor in quant_state.as_dict(packed=True).items():
        spec.append((_packed_state_key_to_buffer_name(key), tuple(tensor.shape), tensor.dtype))
    return tuple(spec)


class BitsAndBytesLinear(WeightOnlyQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.BITSANDBYTES]
    SUPPORTS_METHODS = [METHOD.BITSANDBYTES]
    SUPPORTS_FORMATS = {FORMAT.BITSANDBYTES: 40}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    QUANT_TYPE = "bitsandbytes"

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
        dtype: torch.dtype = torch.float16,
        register_buffers: bool = True,
        format: Optional[str] = None,
        block_size: Optional[int] = None,
        compress_statistics: Optional[bool] = None,
        bnb_quant_type: Optional[str] = None,
        bnb_block_size: Optional[int] = None,
        bnb_compress_statistics: Optional[bool] = None,
        **kwargs,
    ):
        raw_format = format if format is not None else bnb_quant_type
        raw_block_size = block_size if block_size is not None else bnb_block_size
        raw_compress_statistics = (
            compress_statistics if compress_statistics is not None else bnb_compress_statistics
        )

        self.bnb_format = _normalize_bitsandbytes_format(raw_format, bits=bits)
        self.bnb_block_size = _normalize_bitsandbytes_block_size(raw_block_size)
        self.bnb_compress_statistics = True if raw_compress_statistics is None else bool(raw_compress_statistics)
        self.compute_dtype = dtype
        self.quant_state = None
        self._quant_state_signature = None

        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.BITSANDBYTES),
            adapter=kwargs.pop("adapter", None),
            register_buffers=False,
            dtype=dtype,
            pack_dtype=pack_dtype,
            **kwargs,
        )

        if register_buffers:
            self._allocate_buffers(bias=bias)

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not BITSANDBYTES_AVAILABLE:
            return False, ImportError(BITSANDBYTES_INSTALL_HINT)

        try:
            import_bitsandbytes()
        except Exception as exc:
            return False, exc
        return True, None

    @property
    def is_4bit(self) -> bool:
        return self.bits == 4

    def smooth_block_size(self) -> int:
        return self.bnb_block_size

    def _allocate_buffers(self, *, bias: bool) -> None:
        if self.is_4bit:
            buffer_spec = _buffer_spec_4bit(
                in_features=self.in_features,
                out_features=self.out_features,
                quant_type=self.bnb_format,
                block_size=self.bnb_block_size,
                compress_statistics=self.bnb_compress_statistics,
            )
        else:
            buffer_spec = (
                ("weight", (self.out_features, self.in_features), torch.int8),
                ("weight_scb", (self.out_features,), torch.float32),
            )

        for buffer_name, shape, dtype in buffer_spec:
            value = torch.zeros(shape, dtype=dtype)
            if buffer_name in self._buffers:
                self._buffers[buffer_name] = value
            else:
                self.register_buffer(buffer_name, value)

        if bias:
            bias_tensor = torch.zeros(self.out_features, dtype=self.compute_dtype)
            if "bias" in self._buffers:
                self._buffers["bias"] = bias_tensor
            else:
                self.register_buffer("bias", bias_tensor)
        else:
            self.bias = None

    def list_buffers(self):
        buffers = []
        if hasattr(self, "weight") and self.weight is not None:
            buffers.append(self.weight)

        state_buffer_names = (
            _BITSANDBYTES_4BIT_STATE_BUFFER_NAMES if self.is_4bit else _BITSANDBYTES_8BIT_STATE_BUFFER_NAMES
        )
        for buffer_name in state_buffer_names:
            tensor = getattr(self, buffer_name, None)
            if tensor is not None:
                buffers.append(tensor)

        if hasattr(self, "bias") and self.bias is not None:
            buffers.append(self.bias)
        return buffers

    def extra_repr(self) -> str:
        extra = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}"
        )
        if self.is_4bit:
            return (
                f"{extra}, format={self.bnb_format}, "
                f"block_size={self.bnb_block_size}, "
                f"compress_statistics={self.bnb_compress_statistics}"
            )
        return f"{extra}, format={self.bnb_format}"

    def _quant_state_payload(self) -> Dict[str, torch.Tensor]:
        payload: Dict[str, torch.Tensor] = {
            "absmax": self.weight_absmax,
            "quant_map": self.weight_quant_map,
            f"quant_state.bitsandbytes__{self.bnb_format}": self.weight_quant_state,
        }
        if hasattr(self, "weight_nested_absmax"):
            payload["nested_absmax"] = self.weight_nested_absmax
        if hasattr(self, "weight_nested_quant_map"):
            payload["nested_quant_map"] = self.weight_nested_quant_map
        return payload

    def _refresh_quant_state(self, *, force: bool = False):
        if not self.is_4bit:
            self.quant_state = None
            self._quant_state_signature = None
            return None

        bnb = import_bitsandbytes()
        signature = tuple(
            (
                name,
                tuple(getattr(self, name).shape),
                str(getattr(self, name).device),
                getattr(self, name).dtype,
            )
            for name in ("weight", "weight_absmax", "weight_quant_map", "weight_quant_state")
        )
        if not force and self.quant_state is not None and signature == self._quant_state_signature:
            return self.quant_state

        self.quant_state = bnb.functional.QuantState.from_dict(
            self._quant_state_payload(),
            device=self.weight.device,
        )
        self._quant_state_signature = signature
        return self.quant_state

    def post_init(self):
        super().post_init()
        if self.is_4bit:
            self._refresh_quant_state(force=True)

    def dequantize_weight(self) -> torch.Tensor:
        bnb = import_bitsandbytes()
        if self.is_4bit:
            quant_state = self._refresh_quant_state()
            return bnb.functional.dequantize_4bit(self.weight, quant_state=quant_state).contiguous()
        return bnb.functional.int8_vectorwise_dequant(self.weight, self.weight_scb).contiguous()

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor = None):
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    def pack_block(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        block_in: int = 8192,
        workers: int = 1,
    ):
        del block_in, workers
        self.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    @torch.inference_mode()
    def pack_original(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        smooth=None,
    ):
        del scales, zeros, g_idx

        bnb = import_bitsandbytes()
        weight = _apply_optional_smoother(
            _weight_to_matrix(linear).to(device="cpu"),
            smooth=smooth,
            group_size=self.smooth_block_size(),
        ).contiguous()

        if self.is_4bit:
            weight = weight.to(torch.float32 if self.compute_dtype == torch.float32 else self.compute_dtype)
            qweight, quant_state = bnb.functional.quantize_4bit(
                weight,
                blocksize=self.bnb_block_size,
                compress_statistics=self.bnb_compress_statistics,
                quant_type=self.bnb_format,
                quant_storage=torch.uint8,
            )
            self._buffers["weight"] = qweight.contiguous()
            for key, tensor in quant_state.as_dict(packed=True).items():
                self._buffers[_packed_state_key_to_buffer_name(key)] = tensor.contiguous()
            self._refresh_quant_state(force=True)
        else:
            qweight, scales, outlier_cols = bnb.functional.int8_vectorwise_quant(
                weight.to(torch.float16),
                threshold=0.0,
            )
            if outlier_cols is not None and outlier_cols.numel() > 0:
                raise NotImplementedError(
                    "BitsAndBytesLinear only supports the direct int8 vectorwise path without outlier routing."
                )
            self._buffers["weight"] = qweight.contiguous()
            self._buffers["weight_scb"] = scales.contiguous()

        if linear.bias is not None:
            bias = linear.bias.detach().to(device="cpu", dtype=self.compute_dtype).contiguous()
            if "bias" in self._buffers:
                self._buffers["bias"] = bias
            else:
                self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor):
        bnb = import_bitsandbytes()

        input_dtype = x.dtype
        compute_dtype = self.compute_dtype if self.compute_dtype is not None else input_dtype
        bias = None if self.bias is None else self.bias.to(compute_dtype)

        if self.is_4bit:
            quant_state = self._refresh_quant_state()
            out = bnb.matmul_4bit(
                x.to(compute_dtype),
                self.weight.t(),
                quant_state=quant_state,
                bias=bias,
            )
            return out.to(input_dtype)

        x_int8, x_stats, outlier_cols = bnb.functional.int8_vectorwise_quant(
            x.to(torch.float16),
            threshold=0.0,
        )
        if outlier_cols is not None and outlier_cols.numel() > 0:
            raise NotImplementedError(
                "BitsAndBytesLinear only supports the direct int8 vectorwise path without outlier routing."
            )

        mm_out = bnb.functional.int8_linear_matmul(x_int8, self.weight)
        out = bnb.functional.int8_mm_dequant(mm_out, x_stats, self.weight_scb, bias=bias)
        return out.to(input_dtype)


BitsAndBytes4bitLinear = BitsAndBytesLinear

__all__ = [
    "BITSANDBYTES_AVAILABLE",
    "BITSANDBYTES_INSTALL_HINT",
    "BitsAndBytes4bitLinear",
    "BitsAndBytesLinear",
    "import_bitsandbytes",
]
