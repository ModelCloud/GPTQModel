# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import sys
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
from ...utils.cpu_features import has_avx512_bf16
from ...utils.env import env_flag
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
_MLX_4BIT_MIN_ELEMENTS = 4 * 1024 * 1024
_MLX_INT8_MIN_ELEMENTS = 8 * 1024 * 1024
_CPU_PACKED_ENV = "GPTQMODEL_BITSANDBYTES_CPU_PACKED"
_CPU_PACKED_SMALL_WEIGHT_ELEMENTS = 512 * 512
_CPU_PACKED_SMALL_WEIGHT_MAX_M = 512
_CPU_PACKED_SMALL_WEIGHT_MULTITHREAD_MAX_M = 64
_CPU_PACKED_WIDE_WEIGHT_MAX_M = 48
_CPU_PACKED_SINGLE_THREAD_MAX_M = 32
_CPU_PACKED_SMALL_WEIGHT_MIN_THREADS = 4
_CPU_PACKED_REL_TOL = 4e-3
_CPU_INT8_DENSE_ENV = "GPTQMODEL_BITSANDBYTES_CPU_INT8_DENSE"
_CPU_INT8_DENSE_MIN_ROWS = 16
_CPU_INT8_DENSE_MAX_SHORT_DIM = 640
_CPU_INT8_DENSE_MAX_WEIGHT_ELEMENTS = 640 * 1024


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


@lru_cache(maxsize=1)
def _cpu_packed_kernels():
    try:
        from bitsandbytes.functional import (
            _convert_weight_packed_for_cpu,
            _convert_weight_packed_for_cpu_inverse,
            has_avx512bf16,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug("bitsandbytes CPU packed kernels unavailable: %s", exc)
        return None

    try:
        if not has_avx512bf16():
            return None
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug("bitsandbytes AVX512-BF16 probe failed: %s", exc)
        return None

    return _convert_weight_packed_for_cpu, _convert_weight_packed_for_cpu_inverse


@lru_cache(maxsize=1)
def _mlx_quantization_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _mlx_array_to_cpu_tensor(value) -> torch.Tensor:
    return torch.from_dlpack(value).cpu().contiguous()


def _quantize_4bit_weight_mlx_to_torch(
    weight: torch.Tensor,
    *,
    bnb,
    quant_type: str,
    block_size: int,
    compress_statistics: bool,
):
    import mlx.core as mx

    from ...quantization.mlx_bitsandbytes import quantize_4bit_weight_mlx

    packed, absmax = quantize_4bit_weight_mlx(
        mx.from_dlpack(weight),
        quant_type=quant_type,
        block_size=block_size,
        compress_statistics=False,
    )
    absmax = _mlx_array_to_cpu_tensor(absmax)
    code = bnb.functional.get_4bit_type(quant_type, device="cpu")
    if compress_statistics:
        offset = absmax.mean()
        nested_absmax, nested_state = bnb.functional.quantize_blockwise(
            absmax - offset,
            blocksize=256,
        )
        quant_state = bnb.functional.QuantState(
            absmax=nested_absmax,
            shape=weight.shape,
            code=code,
            blocksize=block_size,
            quant_type=quant_type,
            dtype=weight.dtype,
            offset=offset,
            state2=nested_state,
        )
    else:
        quant_state = bnb.functional.QuantState(
            absmax=absmax,
            shape=weight.shape,
            code=code,
            blocksize=block_size,
            quant_type=quant_type,
            dtype=weight.dtype,
        )
    return _mlx_array_to_cpu_tensor(packed), quant_state


def _quantize_int8_weight_mlx_to_torch(weight: torch.Tensor):
    import mlx.core as mx

    from ...quantization.mlx_bitsandbytes import quantize_int8_weight_mlx

    packed, scales = quantize_int8_weight_mlx(mx.from_dlpack(weight))
    return _mlx_array_to_cpu_tensor(packed), _mlx_array_to_cpu_tensor(scales)


def _dequantize_int8_weight(
    weight: torch.Tensor,
    weight_scb: torch.Tensor,
) -> torch.Tensor:
    """Dequantize BnB INT8 weights with the same row-scale math as the native op."""

    return weight.to(torch.float32) * (weight_scb.to(torch.float32) / 127.0).unsqueeze(1)


def _cpu_packed_max_rows(
    *,
    in_features: int,
    out_features: int,
    threads: Optional[int] = None,
) -> int:
    """Return the measured packed-kernel crossover for the current CPU shape.

    The packed GEMV path wins decode-sized batches. Its tiled GEMM scales worse
    than canonical dequantize-plus-GEMM as rows grow, while smaller weights
    amortize that overhead over more rows. The crossover also depends on the
    active PyTorch intra-op thread count.
    """

    if threads is None:
        threads = torch.get_num_threads()
    if threads <= 1:
        return _CPU_PACKED_SINGLE_THREAD_MAX_M
    if in_features * out_features <= _CPU_PACKED_SMALL_WEIGHT_ELEMENTS:
        if threads >= _CPU_PACKED_SMALL_WEIGHT_MIN_THREADS:
            return _CPU_PACKED_SMALL_WEIGHT_MAX_M
        return _CPU_PACKED_SMALL_WEIGHT_MULTITHREAD_MAX_M
    return _CPU_PACKED_WIDE_WEIGHT_MAX_M


def _cpu_int8_dense_should_use(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    compute_dtype: torch.dtype,
) -> bool:
    """Return whether recomputing a dense BF16 weight beats BnB's CPU INT8 path."""

    if not env_flag(_CPU_INT8_DENSE_ENV, default=True):
        return False
    if not has_avx512_bf16():
        return False
    if compute_dtype != torch.bfloat16 or rows < _CPU_INT8_DENSE_MIN_ROWS:
        return False
    if min(in_features, out_features) > _CPU_INT8_DENSE_MAX_SHORT_DIM:
        return False
    return in_features * out_features <= _CPU_INT8_DENSE_MAX_WEIGHT_ELEMENTS


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
@torch.inference_mode(False)
@torch.no_grad()
def _buffer_spec_4bit(
    *,
    in_features: int,
    out_features: int,
    compute_dtype: torch.dtype,
    quant_type: str,
    block_size: int,
    compress_statistics: bool,
) -> Tuple[Tuple[str, Tuple[int, ...], torch.dtype], ...]:
    bnb = import_bitsandbytes()

    # Quantize a template once so the registered buffers match the exact
    # packed layout bitsandbytes expects during checkpoint load.
    template_dtype = compute_dtype if compute_dtype is not None else torch.float16
    template = torch.zeros((out_features, in_features), dtype=template_dtype, device="cpu")
    # Metadata construction also creates temporary tensors inside BNB. Do not
    # inherit a caller's meta-device context when deriving the packed schema.
    with torch.device("cpu"):
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
    @staticmethod
    def _weight_to_matrix(linear: nn.Module) -> torch.Tensor:
        """Expose the weight-only processor's reference-weight conversion hook."""
        return _weight_to_matrix(linear)

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
        self._cpu_packing_active = False
        self._cpu_packing_disabled = False
        self._int8_dense_disabled = False

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
                compute_dtype=self.compute_dtype,
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
        self._ensure_unpacked()
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

        if self._cpu_packing_active:
            self._ensure_unpacked()

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

    def _cpu_packing_supported(self, x: torch.Tensor, compute_dtype: torch.dtype) -> bool:
        if self._cpu_packing_disabled or not env_flag(_CPU_PACKED_ENV, default=True):
            return False
        if not self.is_4bit or self.training or self.weight.device.type != "cpu":
            return False
        if compute_dtype != torch.bfloat16:
            return False
        if self.out_features % 32 != 0 or self.in_features % 2 != 0:
            return False
        if self.in_features % self.bnb_block_size != 0:
            return False
        if torch.is_grad_enabled() and (
            x.requires_grad or (self.bias is not None and self.bias.requires_grad)
        ):
            return False

        rows = x.numel() // self.in_features
        if rows <= 0 or rows > _cpu_packed_max_rows(
            in_features=self.in_features,
            out_features=self.out_features,
        ):
            return False
        return _cpu_packed_kernels() is not None

    def _pack_cpu(self) -> bool:
        kernels = _cpu_packed_kernels()
        if kernels is None or self.quant_state is None:
            return False

        convert, _ = kernels
        bnb = import_bitsandbytes()
        try:
            state_copy = bnb.functional.QuantState.from_dict(
                self.quant_state.as_dict(),
                device=self.weight.device,
            )
            packed_weight, packed_state = convert(self.weight, state_copy)
        except Exception as exc:
            log.debug("bitsandbytes CPU packed conversion failed: %s", exc)
            self._cpu_packing_disabled = True
            return False

        self._buffers["weight"] = packed_weight
        self.quant_state = packed_state
        self._cpu_packing_active = True
        return True

    def _ensure_unpacked(self, *, refresh: bool = False) -> None:
        if not self._cpu_packing_active:
            return

        kernels = _cpu_packed_kernels()
        if kernels is None:
            raise RuntimeError("bitsandbytes CPU packed kernel is no longer available.")

        _, inverse = kernels
        qweight, _ = inverse(self.weight, self.quant_state)
        self._buffers["weight"] = qweight
        self._cpu_packing_active = False
        self.quant_state = None
        self._quant_state_signature = None
        if refresh and self.is_4bit:
            self._refresh_quant_state(force=True)

    def _cpu_packing_accuracy_ok(
        self,
        packed_out: torch.Tensor,
        canonical_out: torch.Tensor,
    ) -> bool:
        packed = packed_out.float()
        canonical = canonical_out.float()
        if not torch.isfinite(packed).all() or not torch.isfinite(canonical).all():
            return False

        denominator = canonical.norm()
        if denominator == 0:
            return torch.equal(packed, canonical)
        relative_error = (packed - canonical).norm() / denominator
        return bool(relative_error <= _CPU_PACKED_REL_TOL)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._ensure_unpacked()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._ensure_unpacked()
        quant_state_key = f"{prefix}weight_quant_state"
        loaded_quant_state = state_dict.get(quant_state_key)
        if loaded_quant_state is not None and self.is_4bit:
            current = self._buffers.get("weight_quant_state")
            if current is None or current.shape != loaded_quant_state.shape:
                self._buffers["weight_quant_state"] = torch.empty_like(loaded_quant_state)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._cpu_packing_disabled = False

    def _apply(self, fn, *args, **kwargs):
        self._ensure_unpacked()
        return super()._apply(fn, *args, **kwargs)

    def post_init(self):
        self._ensure_unpacked()
        super().post_init()
        if self.is_4bit:
            self._refresh_quant_state(force=True)

    def dequantize_weight(self) -> torch.Tensor:
        self._ensure_unpacked()
        bnb = import_bitsandbytes()
        if self.is_4bit:
            quant_state = self._refresh_quant_state()
            return bnb.functional.dequantize_4bit(self.weight, quant_state=quant_state).contiguous()
        return _dequantize_int8_weight(self.weight, self.weight_scb).contiguous()

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

    @torch.inference_mode(False)
    @torch.no_grad()
    def pack_original(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        smooth=None,
    ):
        del scales, zeros, g_idx

        self._ensure_unpacked()
        self._cpu_packing_disabled = False
        bnb = import_bitsandbytes()
        weight = _apply_optional_smoother(
            _weight_to_matrix(linear).to(device="cpu").clone(),
            smooth=smooth,
            group_size=self.smooth_block_size(),
        ).contiguous()

        if self.is_4bit:
            weight = weight.to(torch.float32 if self.compute_dtype == torch.float32 else self.compute_dtype)
            if (
                _mlx_quantization_available()
                and weight.numel() >= _MLX_4BIT_MIN_ELEMENTS
            ):
                qweight, quant_state = _quantize_4bit_weight_mlx_to_torch(
                    weight,
                    bnb=bnb,
                    quant_type=self.bnb_format,
                    block_size=self.bnb_block_size,
                    compress_statistics=self.bnb_compress_statistics,
                )
            else:
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
            weight = weight.to(torch.float16).contiguous()
            if _mlx_quantization_available() and weight.numel() >= _MLX_INT8_MIN_ELEMENTS:
                qweight, scales = _quantize_int8_weight_mlx_to_torch(weight)
                outlier_cols = None
            else:
                qweight, scales, outlier_cols = bnb.functional.int8_vectorwise_quant(
                    weight,
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
            x_compute = x.to(compute_dtype)
            if self._cpu_packing_active:
                if not self._cpu_packing_supported(x, compute_dtype):
                    self._ensure_unpacked(refresh=True)
                    quant_state = self._refresh_quant_state()
                    out = bnb.matmul_4bit(
                        x_compute,
                        self.weight.t(),
                        quant_state=quant_state,
                        bias=bias,
                    )
                else:
                    try:
                        out = bnb.matmul_4bit(
                            x_compute,
                            self.weight,
                            quant_state=self.quant_state,
                            bias=bias,
                        )
                    except Exception as exc:
                        log.debug("bitsandbytes CPU packed forward failed: %s", exc)
                        self._cpu_packing_disabled = True
                        self._ensure_unpacked(refresh=True)
                        quant_state = self._refresh_quant_state()
                        out = bnb.matmul_4bit(
                            x_compute,
                            self.weight.t(),
                            quant_state=quant_state,
                            bias=bias,
                        )
            elif self._cpu_packing_supported(x, compute_dtype):
                quant_state = self._refresh_quant_state()
                canonical_out = bnb.matmul_4bit(
                    x_compute,
                    self.weight.t(),
                    quant_state=quant_state,
                    bias=bias,
                )
                if self._pack_cpu():
                    try:
                        packed_out = bnb.matmul_4bit(
                            x_compute,
                            self.weight,
                            quant_state=self.quant_state,
                            bias=bias,
                        )
                    except Exception as exc:
                        log.debug("bitsandbytes CPU packed forward failed: %s", exc)
                        self._cpu_packing_disabled = True
                        self._ensure_unpacked(refresh=True)
                        out = canonical_out
                    else:
                        if self._cpu_packing_accuracy_ok(packed_out, canonical_out):
                            out = packed_out
                        else:
                            self._cpu_packing_disabled = True
                            self._ensure_unpacked(refresh=True)
                            out = canonical_out
                else:
                    out = canonical_out
            else:
                quant_state = self._refresh_quant_state()
                out = bnb.matmul_4bit(
                    x_compute,
                    self.weight.t(),
                    quant_state=quant_state,
                    bias=bias,
                )
            return out.to(input_dtype)

        rows = x.numel() // self.in_features
        if (
            not self._int8_dense_disabled
            and self.weight.device.type == "cpu"
            and not (torch.is_grad_enabled() and x.requires_grad)
            and _cpu_int8_dense_should_use(
                rows=rows,
                in_features=self.in_features,
                out_features=self.out_features,
                compute_dtype=compute_dtype,
            )
        ):
            weight = _dequantize_int8_weight(self.weight, self.weight_scb).to(compute_dtype)
            out = torch.matmul(x.to(compute_dtype), weight.t())
            if bias is not None:
                out = out + bias
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
