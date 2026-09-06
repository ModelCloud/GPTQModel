# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QVQ planar QuantLinear and its readable numerical reference."""

from __future__ import annotations

import threading
from typing import ClassVar

import torch

from ...adapter.adapter import Adapter
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.config import QVQActivationConfig, _normalize_qvq_activation_config
from ...quantization.dtype import device_supports_native_fp8
from ...quantization.qvq import (
    QVQ_BITS,
    pack_qvq_bank_ids,
    pack_qvq_binary_bank_ids,
    reconstruct_qvq_inner_weight,
    repack_p32_planar_to_window,
    repack_p32_window_to_planar,
    unpack_qvq_bank_ids,
    unpack_qvq_binary_bank_ids,
)
from ...quantization.qvq_activation import (
    dequantize_qvq_fp8_activation,
    fake_quantize_qvq_fp8_activation,
    quantize_qvq_fp8_activation,
)
from ...quantization.qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
from ...quantization.qvq_rates import qvq_transition_bits, qvq_words_per_tile
from ...quantization.rotation.hadamard_utils import matmul_hadU, matmul_hadU_stable
from ...utils.backend import BACKEND
from ...utils.qvq_cuda import (
    qvq_cuda_available,
    qvq_cuda_device_supported,
    qvq_cuda_hadamard,
)
from . import BaseQuantLinear, FormatSupport

_QVQ_BUFFER_NAMES = (
    "trellis", "SU", "SV", "bias", "bank_ids", "bank_alt_id",
    "window_words",
    "rank8_A", "rank8_B", "rank8_metadata",
)
# The real Llama/Qwen transforms that exposed delayed-normalization overflow
# start at this width. Preserve the original, slightly more accurate FP16
# operation ordering for narrow transforms; the finite-output retry below
# remains the range guard for adversarial narrow inputs.
_FP16_STABLE_HADAMARD_MIN_WIDTH = 2048
_QVQ_HADAMARD_MAX_WIDTH = 16384
_QVQ_FP16_SAFE_MAGNITUDE = torch.finfo(torch.float16).max / 2


def _qvq_buffer_version(tensor: torch.Tensor) -> int:
    """Return a stable cache version for normal and inference tensors."""
    try:
        return tensor._version
    except RuntimeError:
        # Tensors created under inference_mode intentionally omit version
        # counters. Their object identity still changes whenever ownership is
        # replaced, which is sufficient for the window cache key.
        return -1


def _qvq_fp16_emulated_hadamard_fallback(
    x: torch.Tensor,
    *,
    post_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale_mode: int,
) -> torch.Tensor:
    """Preserve FP16 epilogue math for widths not handled by the fused kernel."""

    narrowed = x.to(torch.float16)
    transform = matmul_hadU_stable if scale_mode == 3 else matmul_hadU
    historical = transform(narrowed)
    if post_scale is not None:
        historical = historical * post_scale.to(torch.float16)
    if bias is not None:
        historical = historical + bias.to(torch.float16)

    historical_finite = torch.isfinite(historical).all()
    if not torch.cuda.is_current_stream_capturing() and bool(historical_finite):
        return historical.to(x.dtype)

    rescue = transform(x)
    if post_scale is not None:
        rescue = rescue * post_scale.to(x.dtype)
    if bias is not None:
        rescue = rescue + bias.to(x.dtype)
    if torch.cuda.is_current_stream_capturing():
        return torch.where(historical_finite, historical.to(x.dtype), rescue)
    return rescue


def _qvq_hadamard_fused(
    x: torch.Tensor,
    *,
    input_scale: torch.Tensor | None = None,
    input_rounding_mode: int = 0,
    pre_scale: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale_mode: int = 0,
    pad_to_16: bool = False,
    output_fp16: bool = False,
    output_bf16: bool = False,
) -> torch.Tensor:
    """One fused Hadamard launch (CUDA or CPU AVX-512); Python butterfly fallback otherwise.

    ``scale_mode`` mirrors the original dispatch: 0 (stable, width >= 2048) or
    1 (end-scale, narrower widths). The fused kernel is bitwise-identical to the
    corresponding Python reference and also absorbs the SU pre-scale, SV
    post-scale, and bias elementwise steps with the same rounding sequence.
    """

    n = x.shape[-1]
    if input_scale is not None:
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None or x.dtype != fp8_dtype:
            raise TypeError("scaled QVQ A8 Hadamard input must use torch.float8_e4m3fn")
        if (
            input_scale.dtype != torch.float32
            or input_scale.device != x.device
            or input_scale.shape != (*x.shape[:-1], 1)
            or not input_scale.is_contiguous()
        ):
            raise ValueError("QVQ A8 Hadamard scale must be contiguous FP32 with one value per input row")
        if input_rounding_mode not in (0, 1):
            raise ValueError("QVQ A8 Hadamard input rounding mode must be 0 (FP16) or 1 (BF16)")
    elif input_rounding_mode != 0:
        raise ValueError("QVQ A8 Hadamard input rounding mode requires input_scale")
    if x.device.type == "mps" and pre_scale is not None:
        # Keep x*SU in FP32 until normalization; Metal GEMV narrows only the
        # already range-reduced transform result to FP16.
        scaled = x.to(torch.float32) * pre_scale.to(torch.float32)
        return (
            matmul_hadU_stable(scaled)
            if n >= _FP16_STABLE_HADAMARD_MIN_WIDTH
            else matmul_hadU(scaled)
        )
    if (
        x.device.type == "cpu"
        and x.dtype == torch.float32
        and x.is_contiguous()
        and n >= 2
        and n & (n - 1) == 0
        and n <= _QVQ_HADAMARD_MAX_WIDTH
    ):
        from ...utils.qvq_cpu import qvq_cpu_hadamard, qvq_cpu_supported

        if qvq_cpu_supported():
            return qvq_cpu_hadamard(
                x,
                pre_scale=pre_scale,
                post_scale=post_scale,
                bias=bias,
                scale_mode=scale_mode,
            )
    if (
        x.device.type == "cuda"
        and (
            x.dtype in (torch.float16, torch.float32)
            or input_scale is not None
        )
        and x.is_contiguous()
        and n >= 2
        and n & (n - 1) == 0
        and n <= _QVQ_HADAMARD_MAX_WIDTH
        and qvq_cuda_device_supported(x.device)
        and qvq_cuda_available()
    ):
        mode = (
            scale_mode
            if scale_mode >= 2 or x.dtype == torch.float32
            else (0 if n >= _FP16_STABLE_HADAMARD_MIN_WIDTH else 1)
        )
        return qvq_cuda_hadamard(
            x,
            input_scale=input_scale,
            input_rounding_mode=input_rounding_mode,
            pre_scale=pre_scale,
            post_scale=post_scale,
            bias=bias,
            scale_mode=mode,
            pad_to_16=pad_to_16,
            output_fp16=output_fp16,
            output_bf16=output_bf16,
        )
    if input_scale is not None:
        source_dtype = torch.bfloat16 if input_rounding_mode == 1 else torch.float16
        x = dequantize_qvq_fp8_activation(x, input_scale, dtype=source_dtype)
        if pre_scale is not None:
            x = x.to(pre_scale.dtype)
    if pad_to_16 or output_fp16 or output_bf16:
        raise RuntimeError(
            "requested QVQ Hadamard output specialization requires the native CUDA path"
        )
    if x.device.type == "cuda" and x.dtype == torch.float32 and scale_mode in (3, 4):
        return _qvq_fp16_emulated_hadamard_fallback(
            x,
            post_scale=post_scale,
            bias=bias,
            scale_mode=scale_mode,
        )
    # Python fallback preserves the elementwise sequence: pre-scale,
    # transform, post-scale, bias. MPS has no fused range-safe transform, so
    # normalize before every FP16 butterfly there; otherwise a narrow but
    # high-amplitude activation can overflow before the mathematically
    # equivalent final normalization.
    if pre_scale is not None:
        x = x * pre_scale
    use_stable = x.dtype == torch.float16 and (
        x.device.type == "mps" or n >= _FP16_STABLE_HADAMARD_MIN_WIDTH
    )
    transformed = matmul_hadU_stable(x) if use_stable else matmul_hadU(x)
    if post_scale is not None:
        transformed = transformed * post_scale
    if bias is not None:
        transformed = transformed + bias
    return transformed


def _qvq_compute_dtype(input_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Choose the preferred activation dtype used by the transforms and inner kernel."""

    if device_type == "cpu":
        return torch.float32
    if device_type == "mps" and input_dtype != torch.float16:
        return torch.float32
    if device_type == "cuda" and input_dtype == torch.float32:
        return torch.float32
    return torch.float16


def _qvq_mps_narrow_with_row_scale(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Narrow an FP32 transformed activation without losing finite rows to FP16 overflow.

    The normalized Hadamard can amplify a coherent row by ``sqrt(K)`` even
    when the unfactored dense linear result remains finite.  Metal QVQ GEMV
    consumes FP16 activations, so choose an exact power-of-two scale per row,
    narrow the bounded value, then restore the scale on its FP32 GEMV output.
    Scale one is selected for ordinary in-range rows, preserving their prior
    activation values exactly.
    """

    row_peak = x.abs().amax(dim=-1, keepdim=True)
    required = (row_peak / _QVQ_FP16_SAFE_MAGNITUDE).clamp_min(1.0)
    row_scale = torch.exp2(torch.ceil(torch.log2(required)))
    narrowed = (x / row_scale).to(torch.float16)
    return narrowed, row_scale


class QVQLinear(BaseQuantLinear):
    """Production QVQ linear with Torch, MPS, and CUDA dispatch."""

    SUPPORTS_BACKENDS: ClassVar[list[BACKEND]] = [BACKEND.QVQ]
    SUPPORTS_METHODS: ClassVar[list[METHOD]] = [METHOD.QVQ]
    SUPPORTS_FORMAT_BIT_MAP: ClassVar[dict[FORMAT, FormatSupport]] = {
        FORMAT.QVQ: FormatSupport(priority=100, bits=QVQ_BITS),
        FORMAT.QVQ_V4: FormatSupport(
            priority=100, bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 4)
        ),
        FORMAT.QVQ_V4_L18: FormatSupport(
            priority=100, bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 2.5)
        ),
        FORMAT.QVQ_DUAL_V2: FormatSupport(priority=100, bits=QVQ_BITS),
        FORMAT.QVQ_V2B4_P64: FormatSupport(
            priority=100, bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 3.5)
        ),
        FORMAT.QVQ_V2B2_P32: FormatSupport(
            priority=100, bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 3.5)
        ),
    }
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY: ClassVar[list[int]] = [16]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY: ClassVar[list[int]] = [16]
    SUPPORTS_PACK_DTYPES: ClassVar[list[torch.dtype]] = [torch.int32]
    SUPPORTS_ADAPTERS: ClassVar[list[type[Adapter]]] = []
    SUPPORTS_DEVICES: ClassVar[list[DEVICE]] = [DEVICE.ALL]
    SUPPORTS_PLATFORM: ClassVar[list[PLATFORM]] = [PLATFORM.ALL]
    SUPPORTS_DTYPES: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    REQUIRES_FORMAT_V2 = False

    SUPPORTS_GROUP_SIZE: ClassVar[list[int]] = [-1]
    SUPPORTS_DESC_ACT: ClassVar[list[bool]] = [False]
    SUPPORTS_SYM: ClassVar[list[bool]] = [True]

    QUANT_TYPE = "qvq"
    AUXILIARY_DTYPE = torch.float32

    def __init__(
        self,
        *,
        bits: float,
        in_features: int,
        out_features: int,
        bias: bool = False,
        backend: BACKEND = BACKEND.QVQ,
        adapter: Adapter | None = None,
        name: str | None = None,
        register_buffers: bool = True,
        dtype: torch.dtype | None = torch.float16,
        group_size: int = -1,
        desc_act: bool = False,
        sym: bool = True,
        pack_dtype: torch.dtype = torch.int32,
        tensors: dict[str, torch.Tensor] | None = None,
        out_dtype: torch.dtype = torch.float16,
        codebook_version: str = PGC16_CODEBOOK_VERSION,
        vector_size: int = 2,
        trellis_window: int = 16,
        bank_count: int = 1,
        dual_v2: bool = False,
        v2b4_p64: bool = False,
        v2b2_p32: bool = False,
        activation: QVQActivationConfig | dict | bool | None = None,
        input_hadamard: bool = True,
        output_hadamard: bool = True,
        window_only: bool = False,
        **kwargs,
    ):
        del kwargs
        tensors = {} if tensors is None else tensors
        has_bias = bool(bias or tensors.get("bias") is not None)
        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            backend=backend,
            adapter=adapter,
            name=name,
            register_buffers=False,
            dtype=dtype,
            validate_kwargs={
                "group_size": group_size,
                "desc_act": desc_act,
                "sym": sym,
                "pack_dtype": pack_dtype,
                "format": (
                    FORMAT.QVQ_V2B2_P32
                    if v2b2_p32
                    else FORMAT.QVQ_V2B4_P64
                    if v2b4_p64
                    else FORMAT.QVQ_DUAL_V2
                    if dual_v2
                    else FORMAT.QVQ_V4_L18
                    if trellis_window == 18
                    else FORMAT.QVQ_V4
                    if vector_size == 4
                    else FORMAT.QVQ
                ),
            },
        )
        if in_features % 16 or out_features % 16:
            raise ValueError(
                "QVQ formats require in_features and out_features divisible by 16"
            )
        self.group_size = group_size
        self.desc_act = desc_act
        self.sym = sym
        self.pack_dtype = pack_dtype
        self.out_dtype = out_dtype
        self.codebook_version = str(codebook_version).strip().lower()
        if vector_size not in (2, 4) or (vector_size == 4 and bits > 4):
            raise ValueError("QVQ vector_size must be 2, or 4 for rates W1 through W4.")
        self.vector_size = vector_size
        if trellis_window not in (16, 18):
            raise ValueError("QVQ trellis_window must be 16 or 18")
        if trellis_window == 18 and vector_size != 4:
            raise ValueError("QVQ L18 requires vector_size=4")
        if trellis_window == 18 and self.bits > 2.5:
            raise ValueError("QVQ L18 supports only rates W1 through W2.5")
        if trellis_window == 18 and bank_count != 1:
            raise ValueError(
                "QVQ L18 uses implicit history-selected banks and requires bank_count=1"
            )
        self.trellis_window = trellis_window
        if not isinstance(dual_v2, bool):
            raise TypeError("QVQ dual_v2 must be a bool")
        if not isinstance(v2b4_p64, bool):
            raise TypeError("QVQ v2b4_p64 must be a bool")
        if not isinstance(v2b2_p32, bool):
            raise TypeError("QVQ v2b2_p32 must be a bool")
        if sum((dual_v2, v2b4_p64, v2b2_p32)) > 1:
            raise ValueError(
                "QVQ Dual-V2, V2B4-P64, and V2B2-P32 are mutually exclusive"
            )
        if dual_v2 and (vector_size != 2 or trellis_window != 16 or bank_count != 1):
            raise ValueError(
                "QVQ Dual-V2 requires vector_size=2, trellis_window=16, and bank_count=1"
            )
        self.dual_v2 = dual_v2
        if v2b4_p64 and (
            vector_size != 2
            or trellis_window != 16
            or bank_count != 4
            or self.bits > 3.5
        ):
            raise ValueError(
                "QVQ V2B4-P64 requires vector_size=2, trellis_window=16, bank_count=4, and W1-W3.5"
            )
        self.v2b4_p64 = v2b4_p64
        if v2b2_p32 and (
            vector_size != 2
            or trellis_window != 16
            or bank_count != 2
            or self.bits > 3.5
        ):
            raise ValueError(
                "QVQ V2B2-P32 requires vector_size=2, trellis_window=16, bank_count=2, and W1-W3.5"
            )
        self.v2b2_p32 = v2b2_p32
        self.activation = _normalize_qvq_activation_config(activation)
        if self.activation is not None and (
            not self.v2b2_p32 or self.bits not in (2, 2.5, 3, 3.5)
        ):
            raise ValueError("QVQ A8 requires V2B2-P32 weights at rates W2 through W3.5")
        if not isinstance(input_hadamard, bool) or not isinstance(
            output_hadamard, bool
        ):
            raise TypeError("QVQ transform-axis flags must be bools")
        self.input_hadamard = input_hadamard
        self.output_hadamard = output_hadamard
        if not isinstance(window_only, bool):
            raise TypeError("QVQ window_only must be a bool")
        self.window_only = window_only
        if (
            isinstance(bank_count, bool)
            or not isinstance(bank_count, int)
            or bank_count not in (1, 2, 4)
        ):
            raise ValueError("QVQ bank_count must be 1, 2, or 4")
        if bank_count == 4 and vector_size != 4 and not v2b4_p64:
            raise ValueError("QVQ bank_count=4 requires V4 or V2B4-P64")
        self.bank_count = bank_count
        if self.bank_count in (2, 4) and tensors and tensors.get("bank_ids") is None:
            raise ValueError("QVQ banked formats require serialized bank_ids selectors")
        if self.bank_count == 2 and tensors and tensors.get("bank_alt_id") is None:
            raise ValueError("QVQ V2B2-P32 requires serialized bank_alt_id metadata")
        if (
            self.bank_count in (2, 4)
            and tensors
            and tensors["bank_ids"].device.type != "meta"
        ):
            # Dense selectors are accepted at the construction API, but the
            # checkpoint/runtime format has one canonical four-per-byte
            # representation. Normalize at the ownership boundary so a
            # module built from dense selectors round-trips into a packed
            # loader shell without a state-dict shape mismatch.
            tile_count = (in_features // 16) * (out_features // 16)
            selector_count = (
                tile_count * 8
                if v2b2_p32
                else tile_count * 4
                if v2b4_p64
                else tile_count
            )
            tensors = dict(tensors)
            tensors["bank_ids"] = (
                pack_qvq_binary_bank_ids(
                    unpack_qvq_binary_bank_ids(tensors["bank_ids"], selector_count)
                )
                if v2b2_p32
                else pack_qvq_bank_ids(
                    unpack_qvq_bank_ids(tensors["bank_ids"], selector_count)
                )
            )
        self._bank_ids_loaded = self.bank_count == 1 or bool(tensors)
        self._dtype_cache: dict[tuple, tuple] = {}
        self._qvq_mps_bank_ids_cache: (
            tuple[torch.Tensor, int, torch.device, torch.Tensor] | None
        ) = None
        # Dense selectors are launch metadata, not a dequantized weight cache.
        self._qvq_cuda_bank_cache: (
            tuple[
                torch.Tensor,
                int,
                torch.device,
                torch.Tensor | None,
                int,
                torch.Tensor,
                int,
            ]
            | None
        ) = None
        self._qvq_cuda_bank_cache_lock = threading.Lock()
        self._qvq_cuda_window_cache: (
            tuple[torch.Tensor, int, torch.device, torch.Tensor] | None
        ) = None
        self._qvq_planar_fallback_cache: (
            tuple[torch.Tensor, int, torch.Tensor] | None
        ) = None
        self._qvq_fp8_levels_cache: tuple[torch.device, torch.Tensor, float] | None = None
        self._qvq_fp8_telemetry_lock = threading.Lock()
        self._qvq_fp8_telemetry = {
            "requested": 0,
            "eligible": 0,
            "executed": 0,
            "fallback": 0,
            "rejected": 0,
            "fallback_reasons": {},
            "rejection_reasons": {},
        }
        self._qvq_amd_folded_hot_cache: tuple | None = None
        self._qvq_p32_amd_warm_key: tuple | None = None
        # Per-device auxiliary resources for the optional rank8 producer. They
        # are created during preparation, never lazily during graph replay.
        # The enclosing graph owner serializes a module's captured execution;
        # one producer stream/event pair is therefore stable across eager and
        # PyTorch's internal CUDA capture stream.
        self._qvq_rank8_concurrent_cache: dict[
            int, tuple[torch.cuda.Stream, torch.cuda.Event, torch.cuda.Event]
        ] = {}
        self._qvq_rank8_concurrent_warm: set[tuple[int, int, int]] = set()
        self._qvq_rank8_factor_cache: dict[str, tuple[torch.Tensor, int, int, torch.device]] = {}
        pgc16_levels_for_version(self.codebook_version)

        required_tensors = {"SU", "SV"}
        if not window_only:
            required_tensors.add("trellis")
        missing = required_tensors - set(tensors) if tensors else set()
        if tensors and missing:
            raise ValueError(
                f"QVQ module `{self.name}` is missing tensors: {sorted(missing)}"
            )
        unexpected = set(tensors) - set(_QVQ_BUFFER_NAMES)
        if unexpected:
            if "tlut" in unexpected:
                raise ValueError(
                    f"QVQ module `{self.name}` rejects serialized codebook tensors: ['tlut']"
                )
            raise ValueError(
                f"QVQ module `{self.name}` received unexpected tensors: {sorted(unexpected)}"
            )

        storage_dtype = dtype or torch.float16
        defaults = {
            "trellis": None if window_only else torch.zeros(
                (
                    (in_features // 16) * (out_features // 16),
                    qvq_words_per_tile(bits, vector_size=vector_size),
                ),
                dtype=torch.int32,
            ),
            # SU/SV are codec auxiliaries, not model activations. The offline
            # encoder authors them in FP32 and their exact values affect the
            # later FP16/BF16 compute conversion. Keep the checkpoint shell at
            # that precision so loading never silently rounds the codec.
            "SU": torch.ones(in_features, dtype=self.AUXILIARY_DTYPE),
            "SV": torch.ones(out_features, dtype=self.AUXILIARY_DTYPE),
            "bias": torch.zeros(out_features, dtype=storage_dtype)
            if has_bias
            else None,
            # A banked loader shell needs a registered placeholder so strict
            # state-dict loading recognizes the serialized selector key.
            "bank_ids": (
                torch.zeros(
                    ((in_features // 16) * (out_features // 16))
                    if v2b4_p64 or v2b2_p32
                    else ((in_features // 16) * (out_features // 16) + 3) // 4,
                    dtype=torch.uint8,
                )
                if bank_count in (2, 4)
                else None
            ),
            "bank_alt_id": torch.ones(1, dtype=torch.uint8) if v2b2_p32 else None,
            "window_words": None,
            "rank8_A": None,
            "rank8_B": None,
            "rank8_metadata": None,
        }
        for buffer_name in _QVQ_BUFFER_NAMES:
            tensor = tensors.get(
                buffer_name, defaults[buffer_name] if register_buffers else None
            )
            if tensor is None and not buffer_name.startswith("rank8_"):
                setattr(self, buffer_name, None)
            else:
                self.register_buffer(buffer_name, tensor)
        if tensors or register_buffers:
            self._validate_tensors()

        # Optional recovery belongs to this operator, never an adapter wrapper.
        # None buffers keep legacy checkpoints and recovery-off storage unchanged.
        self._p32_rank8_enabled = False

    def __getstate__(self):
        """Exclude transient selector state from deepcopy/pickle."""
        state = super().__getstate__()
        state.pop("_qvq_cuda_bank_cache_lock", None)
        state.pop("_qvq_fp8_telemetry_lock", None)
        state.pop("_qvq_grouped_p32_delegate", None)
        state["_qvq_cuda_bank_cache"] = None
        state["_qvq_cuda_window_cache"] = None
        state["_qvq_planar_fallback_cache"] = None
        state["_qvq_fp8_levels_cache"] = None
        state["_qvq_amd_folded_hot_cache"] = None
        state["_qvq_p32_amd_warm_key"] = None
        state["_qvq_rank8_concurrent_cache"] = {}
        state["_qvq_rank8_concurrent_warm"] = set()
        state["_qvq_rank8_factor_cache"] = {}
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._qvq_cuda_bank_cache_lock = threading.Lock()
        self._qvq_fp8_telemetry_lock = threading.Lock()
        self._qvq_cuda_bank_cache = None
        self._qvq_cuda_window_cache = None
        self._qvq_planar_fallback_cache = None
        self._qvq_fp8_levels_cache = None
        if "_qvq_fp8_telemetry" not in self.__dict__:
            self._qvq_fp8_telemetry = {
                "requested": 0,
                "eligible": 0,
                "executed": 0,
                "fallback": 0,
                "rejected": 0,
                "fallback_reasons": {},
                "rejection_reasons": {},
            }
        else:
            self._qvq_fp8_telemetry.setdefault("rejected", 0)
            self._qvq_fp8_telemetry.setdefault("fallback_reasons", {})
            self._qvq_fp8_telemetry.setdefault("rejection_reasons", {})
        self._qvq_amd_folded_hot_cache = None
        self._qvq_p32_amd_warm_key = None
        self._qvq_rank8_concurrent_cache = {}
        self._qvq_rank8_concurrent_warm = set()
        self._qvq_rank8_factor_cache = {}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        delegate = getattr(self, "_qvq_grouped_p32_delegate", None)
        if delegate is None:
            return
        state, consumer_index, _ = delegate
        for name, tensor in state.canonical_child_payload(consumer_index).items():
            destination[f"{prefix}{name}"] = tensor if keep_vars else tensor.detach()

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
        from ...quantization.qvq_rank8 import RANK8_BUFFERS

        for name in RANK8_BUFFERS:
            value = state_dict.get(f"{prefix}{name}")
            if value is not None:
                setattr(self, name, torch.empty_like(value, device=self.runtime_device()))
        # A newly loaded payload must be validated before enabling recovery.
        self._p32_rank8_enabled = False
        self._qvq_p32_amd_warm_key = None
        selector_key = f"{prefix}bank_ids"
        if self.bank_count in (2, 4) and selector_key not in state_dict:
            self._bank_ids_loaded = False
        else:
            self._bank_ids_loaded = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _dtype_cache_clear(self) -> None:
        """Drop cached dtype conversions (call if SU/SV/bias are replaced)."""
        self._dtype_cache = {}
        self._qvq_cuda_aux_cache_signature = None
        self._qvq_rank8_factor_cache = {}
        self._qvq_planar_fallback_cache = None

    def _prepare_planar_fallback(self) -> torch.Tensor:
        """Prepare a legacy planar payload only for an explicit fallback.

        Normal direct-window inference releases planar ownership. Unsupported
        arithmetic or backend policies may still request the reference child
        path; this cache is then prepared eagerly and remains stable for graph
        replay instead of allocating during capture.
        """
        if not self.window_only:
            if self.trellis is None:
                raise RuntimeError("QVQ planar fallback payload is unavailable")
            return self.trellis
        source = self.window_words
        if source is None:
            raise RuntimeError("QVQ window-only module is missing window_words")
        source_version = _qvq_buffer_version(source)
        cached = self._qvq_planar_fallback_cache
        if (
            cached is not None
            and cached[0] is source
            and cached[1] == source_version
        ):
            return cached[2]
        self._require_prepared_outside_capture(source.device, "planar fallback")
        planar = repack_p32_window_to_planar(source, bits=self.bits).contiguous()
        self._qvq_planar_fallback_cache = (source, source_version, planar)
        return planar

    def _cached_rank8_factor(self, name: str) -> torch.Tensor:
        """Return a prepared contiguous FP32 rank8 factor without replay casts."""
        if name not in ("A", "B"):
            raise ValueError("rank8 factor name must be A or B")
        source = getattr(self, "rank8_" + name, None)
        if source is None:
            raise RuntimeError(f"rank8 factor {name} is unavailable")
        key = (id(source), source._version, source.device)
        cached = self._qvq_rank8_factor_cache.get(name)
        if cached is not None and cached[1:] == key:
            return cached[0]
        self._require_prepared_outside_capture(source.device, "rank8 FP32 factors")
        value = source.detach().to(dtype=torch.float32).contiguous()
        self._qvq_rank8_factor_cache[name] = (value, *key)
        return value

    def _prepare_cuda_graph_auxiliary_caches(self) -> None:
        """Materialize constant dtype variants used by graph-safe fallback paths.

        The normal FP16 forward only needs the FP16 transform constants during
        warmup.  A CUDA graph must also be able to take the BF16 overflow-rescue
        path without allocating a new cast buffer while capture is active.
        Keep this preparation explicit and outside capture; the forward path
        remains free of host synchronization and cache mutation.
        """
        if self.runtime_device().type != "cuda":
            return
        self._require_prepared_outside_capture(
            self.runtime_device(), "auxiliary dtype caches"
        )
        signature = tuple(
            (name, id(tensor), tensor._version, tensor.device)
            for name in ("SU", "SV", "bias")
            if (tensor := getattr(self, name)) is not None
        )
        if getattr(self, "_qvq_cuda_aux_cache_signature", None) == signature:
            return
        for compute_dtype in (torch.float16, torch.bfloat16):
            self._cached_cast("SU", compute_dtype)
            for output_dtype in (torch.float16, torch.bfloat16, torch.float32):
                self._cached_cast("SV", compute_dtype, output_dtype)
                self._cached_cast("bias", compute_dtype, output_dtype)
        self._qvq_cuda_aux_cache_signature = signature

    def _rank8_concurrent_resources(self, device: torch.device):
        """Return prepared stream/event resources for concurrent rank8 projection.

        A producer stream is paired with whichever caller stream submits the
        operation. Missing resources during capture are rejected so graph
        replay never creates a stream or event. The resource is keyed by CUDA
        device rather than caller stream because PyTorch uses an internal
        stream for CUDA-graph capture.
        """
        if device.type != "cuda":
            raise RuntimeError("concurrent rank8 projection requires CUDA")
        device_key = int(device.index if device.index is not None else torch.cuda.current_device())
        cached = self._qvq_rank8_concurrent_cache.get(device_key)
        if cached is not None:
            return cached
        self._require_prepared_outside_capture(device, "rank8 concurrent producer")
        resources = (
            torch.cuda.Stream(device=device),
            torch.cuda.Event(enable_timing=False, blocking=False),
            torch.cuda.Event(enable_timing=False, blocking=False),
        )
        self._qvq_rank8_concurrent_cache[device_key] = resources
        return resources

    @staticmethod
    def _require_prepared_outside_capture(device: torch.device, what: str) -> None:
        """Fail closed when a cold cache would allocate or synchronize in capture."""
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"QVQ {what} must be prepared before CUDA Graph capture"
            )

    def _prepare_hopper_p32_window(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        """Build and retain the storage-neutral P32 window payload for direct kernels.

        Serialized checkpoints remain canonical planar P32.  The direct-window
        Hopper and gfx950 kernels use an equivalent bit layout, so convert once
        per module after the weights reach the accelerator and reuse the result.
        """

        source = self.window_words if self.window_only else self.trellis
        if source is None:
            raise RuntimeError("QVQ window-only module is missing window_words")
        source_version = _qvq_buffer_version(source)
        cached = self._qvq_cuda_window_cache
        if (
            cached is not None
            and cached[0] is source
            and cached[1] == source_version
            and cached[2] == device
        ):
            return cached[3]
        self._require_prepared_outside_capture(device, "window payload")
        if self.window_only:
            if source.device != device:
                raise RuntimeError("QVQ window_words must reside on the execution device")
            window = source.contiguous()
        else:
            window = repack_p32_planar_to_window(source.contiguous(), bits=self.bits).to(
                device=device
            )
        current_source = self.window_words if self.window_only else self.trellis
        if current_source is not source or _qvq_buffer_version(source) != source_version:
            raise RuntimeError(
                "QVQ P32 trellis changed while preparing the window payload"
            )
        # Direct window inference owns the lossless window payload after the
        # one-time repack. Release the planar source for evaluated modules so
        # live device storage does not retain two equivalent representations.
        # Training modules keep planar ownership for the differentiable path;
        # explicit preparation is itself the boundary for eval modules.
        if not self.window_only and not self.training:
            self.window_words = window
            self.trellis = None
            self.window_only = True
            source = self.window_words
            source_version = _qvq_buffer_version(source)
        self._qvq_cuda_window_cache = (source, source_version, device, window)
        return window

    def _prepare_hopper_fp8_levels(self, device: torch.device) -> tuple[torch.Tensor, float]:
        """Return the cached E4M3 PGC table and its explicit dequantization scale."""

        cached = self._qvq_fp8_levels_cache
        if cached is not None and cached[0] == device:
            return cached[1], cached[2]
        self._require_prepared_outside_capture(device, "FP8 level table")
        fp8_dtype = torch.float8_e4m3fn
        fp8_max = float(torch.finfo(fp8_dtype).max)
        canonical = pgc16_levels_for_version(self.codebook_version).to(torch.float32)
        level_scale = float(canonical.abs().amax().item() / fp8_max)
        levels = torch.clamp(canonical / level_scale, min=-fp8_max, max=fp8_max).to(
            device=device,
            dtype=fp8_dtype,
        ).contiguous()
        self._qvq_fp8_levels_cache = (device, levels, level_scale)
        return levels, level_scale

    def _record_fp8_kernel(self, event: str, reason: str | None = None) -> None:
        with self._qvq_fp8_telemetry_lock:
            self._qvq_fp8_telemetry[event] += 1
            if reason is not None:
                reason_key = "rejection_reasons" if event == "rejected" else "fallback_reasons"
                reasons = self._qvq_fp8_telemetry[reason_key]
                reasons[reason] = reasons.get(reason, 0) + 1

    def qvq_fp8_kernel_telemetry(self, *, reset: bool = False) -> dict:
        """Return truthful requested/eligible/executed/fallback counters."""

        with self._qvq_fp8_telemetry_lock:
            result = {
                **self._qvq_fp8_telemetry,
                "fallback_reasons": dict(self._qvq_fp8_telemetry["fallback_reasons"]),
                "rejection_reasons": dict(self._qvq_fp8_telemetry["rejection_reasons"]),
                "operand_dtype": "float8_e4m3fn",
                "accumulator_dtype": "float32",
            }
            if reset:
                self._qvq_fp8_telemetry = {
                    "requested": 0,
                    "eligible": 0,
                    "executed": 0,
                    "fallback": 0,
                    "rejected": 0,
                    "fallback_reasons": {},
                    "rejection_reasons": {},
                }
        return result

    def _prepare_amd_p32_metadata(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Snapshot mutable P32 selectors and return the reusable gfx950 payload."""

        with self._qvq_cuda_bank_cache_lock:
            source = self.bank_ids
            alternative = self.bank_alt_id
            if source is None or alternative is None:
                raise RuntimeError("AMD folded P32 requires bank selectors and alternative-bank metadata")
            current_version = source._version
            alternative_version = alternative._version
            cached = self._qvq_cuda_bank_cache
            if (
                cached is not None
                and cached[0] is source
                and cached[1] == current_version
                and cached[2] == device
                and cached[3] is alternative
                and cached[4] == alternative_version
                and self.bank_ids is source
                and source._version == current_version
                and self.bank_alt_id is alternative
                and alternative._version == alternative_version
            ):
                packed = cached[5]
                bank_alt_id = cached[6]
            else:
                self._require_prepared_outside_capture(device, "bank selector payload")
                snapshot = None
                snapshot_version = -1
                for _ in range(3):
                    if self.bank_ids is not source:
                        break
                    before = source._version
                    candidate = source.detach().clone()
                    after = source._version
                    if before == after and self.bank_ids is source:
                        snapshot = candidate
                        snapshot_version = after
                        break
                if snapshot is None:
                    raise RuntimeError("QVQ CUDA bank selector mutated during snapshot")
                tile_count = (self.in_features // 16) * (self.out_features // 16)
                packed = pack_qvq_binary_bank_ids(
                    unpack_qvq_binary_bank_ids(snapshot, tile_count * 8)
                ).to(device=device)
                alternative_version = alternative._version
                bank_alt_id = int(alternative.detach().item())
                if (
                    self.bank_alt_id is not alternative
                    or alternative._version != alternative_version
                    or not 1 <= bank_alt_id <= 3
                ):
                    raise RuntimeError("QVQ CUDA alternative-bank metadata changed during snapshot")
                if self.bank_ids is not source:
                    raise RuntimeError("QVQ CUDA bank selector replaced during snapshot")
                self._qvq_cuda_bank_cache = (
                    source,
                    snapshot_version,
                    device,
                    alternative,
                    alternative_version,
                    packed,
                    bank_alt_id,
                )
            window = self._prepare_hopper_p32_window(device)
        return window, packed, bank_alt_id

    def _qvq_amd_folded_forward(
        self,
        x_2d: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return the gfx950 full-layer folded-cache result when strictly eligible."""

        if (
            self.training
            or x_2d.device.type != "cuda"
            or torch.version.hip is None
            or compute_dtype != torch.float16
            or x_2d.dtype != torch.float16
            or self.trellis_window != 16
            or self.dual_v2
            or not self.v2b2_p32
            or self.vector_size != 2
        ):
            return None
        from ...utils.qvq_amd import (
            _qvq_p32_folded_execute,
            qvq_p32_amd_folded,
            qvq_p32_amd_folded_case_supported,
            qvq_p32_amd_folded_prefers_fp32_output,
            qvq_p32_amd_supported,
        )
        if not qvq_p32_amd_folded_case_supported(
            x_2d.shape[0], self.in_features, self.out_features
        ):
            return None
        cached = self._qvq_amd_folded_hot_cache
        # Resolve registered tensors once. nn.Module buffer lookup is not a plain
        # attribute read, and repeated resolution was material in decode forwards.
        # Window-only deployment releases planar trellis after the CPU repack;
        # the prepared window is the canonical source for cache identity then.
        buffers = self._buffers if type(self) is QVQLinear else {}
        source = self.window_words if self.window_only else (
            buffers["trellis"] if "trellis" in buffers else self.trellis  # noqa: SIM401
        )
        if source is None:
            return None
        bank_ids_source = buffers["bank_ids"] if "bank_ids" in buffers else self.bank_ids  # noqa: SIM401
        bank_alt_source = buffers["bank_alt_id"] if "bank_alt_id" in buffers else self.bank_alt_id  # noqa: SIM401
        su_source = buffers["SU"] if "SU" in buffers else self.SU  # noqa: SIM401
        sv_source = buffers["SV"] if "SV" in buffers else self.SV  # noqa: SIM401
        bias_source = self.bias
        if (
            cached is not None
            and len(cached) == 27
            and cached[0] is source
            and cached[1] == source._version
            and cached[2] is bank_ids_source
            and cached[3] == bank_ids_source._version
            and cached[4] is bank_alt_source
            and cached[5] == bank_alt_source._version
            and cached[6] is su_source
            and cached[7] == su_source._version
            and cached[8] is sv_source
            and cached[9] == sv_source._version
            and cached[10] is bias_source
            and cached[11] == (-1 if bias_source is None else bias_source._version)
            and cached[12] == x_2d.device
            and cached[13] == self.bits
            and cached[14] == self.input_hadamard
            and cached[15] == self.output_hadamard
            and cached[16] == self.codebook_version
        ):
            bias = cached[23]
            output = _qvq_p32_folded_execute(
                x_2d,
                cached[24],
                cached[25],
                cached[26],
                out_features=self.out_features,
                output_fp32=bias is not None
                or qvq_p32_amd_folded_prefers_fp32_output(
                    x_2d.shape[0], self.in_features, self.out_features
                ),
            )
            return output if bias is None else output + bias
        # A cache hit already proved this exact rate eligible when built. Keep
        # normalization on the cold path, including after any rate mutation.
        if qvq_transition_bits(self.bits, vector_size=2) not in (4, 5, 6, 7):
            return None
        from ...utils.qvq_cuda import _pgc16_levels

        if not qvq_p32_amd_supported(x_2d.device):
            return None
        if not self._bank_ids_loaded or self.bank_ids is None or self.bank_ids.device.type == "meta":
            raise RuntimeError("QVQ banked module cannot run before bank_ids selectors are loaded")
        window, bank_ids, bank_alt_id = self._prepare_amd_p32_metadata(x_2d.device)
        bias = self._cached_cast("bias", compute_dtype, torch.float32)
        levels = _pgc16_levels(x_2d.device, self.codebook_version)
        su = self._cached_cast("SU", compute_dtype)
        sv = self._cached_cast("SV", compute_dtype)
        output = qvq_p32_amd_folded(
            x_2d.contiguous(),
            window,
            levels,
            bank_ids,
            su,
            sv,
            self.bits,
            out_features=self.out_features,
            bank_alt_id=bank_alt_id,
            input_hadamard=self.input_hadamard,
            output_hadamard=self.output_hadamard,
            output_fp32=bias is not None
            or qvq_p32_amd_folded_prefers_fp32_output(
                x_2d.shape[0], self.in_features, self.out_features
            ),
        )
        _, _, operand, _, residual_operand, composite_recovery = (
            window._qvq_p32_amd_folded_cache
        )
        self._qvq_amd_folded_hot_cache = (
            source,
            source._version,
            self.bank_ids,
            self.bank_ids._version,
            self.bank_alt_id,
            self.bank_alt_id._version,
            self.SU,
            self.SU._version,
            self.SV,
            self.SV._version,
            self.bias,
            -1 if self.bias is None else self.bias._version,
            x_2d.device,
            self.bits,
            self.input_hadamard,
            self.output_hadamard,
            self.codebook_version,
            window,
            levels,
            bank_ids,
            bank_alt_id,
            su,
            sv,
            bias,
            operand,
            residual_operand,
            composite_recovery,
        )
        return output if bias is None else output + bias

    def _cached_cast(self, name: str, *dtypes: torch.dtype) -> torch.Tensor | None:
        """Convert a constant auxiliary tensor (SU/SV/bias) to the requested
        dtype chain once, caching by (name, dtypes). The identity + _version
        guard invalidates the entry if the source buffer is replaced or
        mutated. The dtype chain (e.g. fp16 then fp32) reproduces the exact
        rounding of the per-forward `tensor.to(a).to(b)` sequence."""
        tensor = getattr(self, name)
        if tensor is None:
            return None
        key = (name, dtypes)
        cached = self._dtype_cache.get(key)
        if cached is None or cached[0] is not tensor or cached[1] != tensor._version:
            self._require_prepared_outside_capture(tensor.device, "auxiliary dtype cache")
            converted = tensor
            for dtype in dtypes:
                converted = converted.to(dtype)
            self._dtype_cache[key] = (tensor, tensor._version, converted)
            return converted
        return cached[2]

    @classmethod
    def validate(cls, **args):
        valid, error = super().validate(**args)
        if not valid:
            return valid, error
        checks = (
            (args.get("group_size", -1) == -1, "group_size=-1"),
            (args.get("desc_act", False) is False, "desc_act=False"),
            (args.get("sym", True) is True, "sym=True"),
            (
                args.get("format", FORMAT.QVQ)
                in (
                    FORMAT.QVQ,
                    FORMAT.QVQ_V4,
                    FORMAT.QVQ_V4_L18,
                    FORMAT.QVQ_DUAL_V2,
                    FORMAT.QVQ_V2B4_P64,
                    FORMAT.QVQ_V2B2_P32,
                ),
                "a supported QVQ format including format=qvq",
            ),
        )
        for accepted, requirement in checks:
            if not accepted:
                return False, NotImplementedError(f"QVQLinear requires {requirement}.")
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if (in_features is not None and in_features % 16) or (
            out_features is not None and out_features % 16
        ):
            return False, NotImplementedError(
                "QVQ formats require in_features and out_features divisible by 16."
            )
        device = args.get("device")
        dtype = args.get("dtype")
        if device == DEVICE.MPS and dtype not in (None, torch.float16):
            return False, NotImplementedError(
                "QVQLinear MPS inference requires float16 activations."
            )
        return True, None

    @classmethod
    def from_tensors(
        cls,
        *,
        bits: float,
        in_features: int,
        out_features: int,
        name: str,
        tensors: dict[str, torch.Tensor],
        codebook_version: str = PGC16_CODEBOOK_VERSION,
        vector_size: int = 2,
        trellis_window: int = 16,
        bank_count: int = 1,
        dual_v2: bool = False,
        v2b4_p64: bool = False,
        v2b2_p32: bool = False,
        activation: QVQActivationConfig | dict | bool | None = None,
        input_hadamard: bool = True,
        output_hadamard: bool = True,
    ) -> QVQLinear:
        return cls(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            name=name,
            tensors=tensors,
            codebook_version=codebook_version,
            vector_size=vector_size,
            trellis_window=trellis_window,
            bank_count=bank_count,
            dual_v2=dual_v2,
            v2b4_p64=v2b4_p64,
            v2b2_p32=v2b2_p32,
            activation=activation,
            input_hadamard=input_hadamard,
            output_hadamard=output_hadamard,
        )

    def _validate_tensors(self) -> None:
        expected_trellis = (
            (self.in_features // 16) * (self.out_features // 16),
            qvq_words_per_tile(self.bits, vector_size=self.vector_size),
        )
        expected = {
            "trellis": (expected_trellis, torch.int32),
            "SU": ((self.in_features,), None),
            "SV": ((self.out_features,), None),
        }
        for name, (shape, dtype) in expected.items():
            tensor = getattr(self, name)
            if tensor is None:
                if name == "trellis" and self.window_only:
                    continue
                raise ValueError(f"QVQ `{name}` is missing from the module")
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"QVQ `{name}` must have shape {shape}, got {tuple(tensor.shape)}"
                )
            if dtype is not None and tensor.dtype != dtype:
                raise TypeError(f"QVQ `{name}` must use {dtype}, got {tensor.dtype}")
            if name != "trellis" and not tensor.is_floating_point():
                raise TypeError(f"QVQ `{name}` must use a floating-point dtype")
        if self.window_only:
            if self.window_words is None:
                raise ValueError("window_only QVQ modules require window_words")
            if tuple(self.window_words.shape) != expected_trellis:
                raise ValueError(
                    f"QVQ `window_words` must have shape {expected_trellis}, got {tuple(self.window_words.shape)}"
                )
            if self.window_words.dtype != torch.int32:
                raise TypeError("QVQ `window_words` must use torch.int32")
        if self.bank_ids is not None:
            if self.bank_count not in (2, 4) or (
                self.vector_size != 4 and not self.v2b4_p64 and not self.v2b2_p32
            ):
                raise ValueError("QVQ bank selectors require a banked format")
            tile_count = (self.in_features // 16) * (self.out_features // 16)
            selector_count = (
                tile_count * 8
                if self.v2b2_p32
                else tile_count * 4
                if self.v2b4_p64
                else tile_count
            )
            packed_count = (selector_count + (7 if self.v2b2_p32 else 3)) // (
                8 if self.v2b2_p32 else 4
            )
            if self.bank_ids.ndim != 1 or self.bank_ids.numel() not in (
                selector_count,
                packed_count,
            ):
                raise ValueError(
                    "QVQ `bank_ids` must be dense or packed for the module tile count"
                )
            if self.bank_ids.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError("QVQ `bank_ids` must use an integer dtype")
            # Accelerate constructs quantized modules on the meta device before
            # streaming checkpoint tensors.  Value validation would force a
            # scalar read from a meta selector; defer it until the real payload
            # is installed, while retaining shape/dtype checks above.
            if self.bank_ids.device.type != "meta":
                if self.v2b2_p32:
                    unpack_qvq_binary_bank_ids(self.bank_ids, selector_count)
                else:
                    unpack_qvq_bank_ids(self.bank_ids, selector_count)
        if self.v2b2_p32:
            if self.bank_alt_id is None or tuple(self.bank_alt_id.shape) != (1,):
                raise ValueError("QVQ V2B2-P32 requires one bank_alt_id value")
            if self.bank_alt_id.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError("QVQ bank_alt_id must use an integer dtype")
            if self.bank_alt_id.device.type != "meta" and not bool(
                ((self.bank_alt_id >= 1) & (self.bank_alt_id <= 3)).all()
            ):
                raise ValueError("QVQ bank_alt_id must be in [1, 3]")
        elif self.bank_alt_id is not None:
            raise ValueError("QVQ bank_alt_id is valid only for a V2B2-P32 format")
        if self.bias is not None:
            if tuple(self.bias.shape) != (self.out_features,):
                raise ValueError(
                    f"QVQ bias must have shape {(self.out_features,)}, got {tuple(self.bias.shape)}"
                )
            if not self.bias.is_floating_point():
                raise TypeError("QVQ `bias` must use a floating-point dtype")
        active_device = (
            self.window_words.device
            if self.window_only and self.window_words is not None
            else self.trellis.device
        )
        devices = {active_device, self.SU.device, self.SV.device}
        if not self.window_only:
            devices.add(self.trellis.device)
        if self.bank_ids is not None:
            devices.add(self.bank_ids.device)
        if self.bank_alt_id is not None:
            devices.add(self.bank_alt_id.device)
        if self.bias is not None:
            devices.add(self.bias.device)
        if len(devices) != 1:
            raise ValueError(
                "QVQ module tensors must share one device (or window_only may keep planar trellis on CPU)"
            )
        floating_tensors = (
            (self.SU, self.SV) if self.bias is None else (self.SU, self.SV, self.bias)
        )
        if any(
            tensor.device.type != "meta" and not torch.isfinite(tensor).all()
            for tensor in floating_tensors
        ):
            raise ValueError(
                "QVQ floating-point tensors must contain only finite values"
            )

    def runtime_device(self) -> torch.device | None:
        if self.window_only and self.window_words is not None:
            return self.window_words.device
        return None if self.trellis is None else self.trellis.device

    def post_init(self) -> None:
        super().post_init()
        # Quantization and checkpoint materialization may construct this
        # module under ``torch.inference_mode()``. Such tensors deliberately
        # have no mutation counter, but the selector/dtype caches below rely
        # on one to reject stale state. Take ownership once at the module
        # boundary as ordinary versioned buffers instead of weakening every
        # cache to identity-only validation.
        for buffer_name in _QVQ_BUFFER_NAMES:
            tensor = getattr(self, buffer_name)
            if tensor is None or tensor.device.type == "meta":
                continue
            try:
                _ = tensor._version
            except RuntimeError:
                with torch.inference_mode(False):
                    setattr(self, buffer_name, tensor.detach().clone())
        self._validate_tensors()
        self._qvq_mps_compander = None
        self._qvq_mps_bank_ids = None
        self._qvq_mps_bank_ids_cache = None
        self._dtype_cache_clear()
        with self._qvq_cuda_bank_cache_lock:
            self._qvq_cuda_bank_cache = None
            self._qvq_cuda_window_cache = None
            self._qvq_amd_folded_hot_cache = None
            self._qvq_p32_amd_warm_key = None
        if self.runtime_device().type == "mps":
            from ...utils.qvq_mps import _prepare_qvq_mps_compander

            self._qvq_mps_compander = _prepare_qvq_mps_compander(
                self.runtime_device(),
                self.codebook_version,
            )
            if self.bank_ids is not None:
                self._qvq_mps_bank_ids = self._prepare_mps_bank_ids(
                    self.runtime_device()
                )

    def _apply(self, fn):
        grouped_runtime = getattr(self, "_gptqmodel_qvq_grouped_runtime", None)
        if grouped_runtime is not None:
            # The grouped Hopper payload is transient state outside the module
            # tree.  Invalidate it before Module._apply replaces any canonical
            # child buffers so a device/dtype move cannot retain VRAM on the
            # previous device or publish a payload under stale source keys.
            grouped_runtime.invalidate()
        self._qvq_mps_compander = None
        self._qvq_mps_bank_ids = None
        self._qvq_mps_bank_ids_cache = None
        with self._qvq_cuda_bank_cache_lock:
            self._qvq_cuda_bank_cache = None
            self._qvq_cuda_window_cache = None
            self._qvq_amd_folded_hot_cache = None
            self._qvq_p32_amd_warm_key = None
        # ModuleLooper performs device handoffs from inference-mode workers.
        # Letting Module._apply inherit that mode would recreate all cache-keyed
        # buffers without mutation counters immediately after post_init made
        # them versioned. Device conversion is ownership transfer, not model
        # inference, so keep the resulting buffers ordinary and guardable.
        with torch.inference_mode(False):
            return super()._apply(fn)

    def _prepare_mps_bank_ids(self, device: torch.device) -> torch.Tensor | None:
        """Return packed selectors from a stable snapshot of mutable bank state.

        Quantization and output-recovery stages may replace or mutate bank IDs
        before inference. Track both object identity and PyTorch's mutation
        version so MPS never reuses selectors from an older assignment. A
        concurrent free-threaded mutation is retried and then rejected rather
        than silently launching Metal with a mixed or stale selector snapshot.
        """

        source = self.bank_ids
        if source is None:
            self._qvq_mps_bank_ids_cache = None
            self._qvq_mps_bank_ids = None
            return None
        cached = self._qvq_mps_bank_ids_cache
        if (
            cached is not None
            and cached[0] is source
            and cached[1] == source._version
            and cached[2] == device
            and self.bank_ids is source
        ):
            return cached[3]

        tile_count = (self.in_features // 16) * (self.out_features // 16)
        selector_count = (
            tile_count * 8
            if self.v2b2_p32
            else tile_count * 4
            if self.v2b4_p64
            else tile_count
        )
        for _ in range(3):
            source = self.bank_ids
            if source is None:
                self._qvq_mps_bank_ids_cache = None
                self._qvq_mps_bank_ids = None
                return None
            source_version = source._version
            snapshot = source.detach().clone()
            packed = (
                pack_qvq_binary_bank_ids(
                    unpack_qvq_binary_bank_ids(snapshot, selector_count)
                )
                if self.v2b2_p32
                else pack_qvq_bank_ids(unpack_qvq_bank_ids(snapshot, selector_count))
            ).to(device=device)
            if source is self.bank_ids and source_version == source._version:
                self._qvq_mps_bank_ids_cache = (source, source_version, device, packed)
                self._qvq_mps_bank_ids = packed
                return packed
        raise RuntimeError(
            "QVQ `bank_ids` changed concurrently while preparing MPS inference selectors"
        )

    def get_inner_weight_tensor(
        self, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Materialize the dense inner weight, in FP32 unless explicitly requested otherwise."""

        trellis = self.trellis
        if self.window_only:
            if self.window_words is None:
                raise RuntimeError("window-only QVQ module is missing window_words")
            # This is an explicit legacy/reference request. Do not create a
            # temporary planar tensor during graph capture; callers that need
            # that path in a graph must load with retain_planar=True.
            self._require_prepared_outside_capture(
                self.window_words.device, "planar reconstruction"
            )
            trellis = repack_p32_window_to_planar(
                self.window_words, bits=self.bits
            )

        return reconstruct_qvq_inner_weight(
            trellis,
            bits=self.bits,
            vector_size=self.vector_size,
            trellis_window=self.trellis_window,
            in_features=self.in_features,
            out_features=self.out_features,
            codebook_version=self.codebook_version,
            bank_ids=self.bank_ids,
            dual_v2=self.dual_v2,
            v2b4_p64=self.v2b4_p64,
            v2b2_p32=self.v2b2_p32,
            bank_alt_id=self.bank_alt_id,
        ).to(dtype=dtype)

    def _reference_inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable training reference; its graph may retain the dense weight.

        This legacy path intentionally follows ``x.dtype`` and is not the accuracy
        oracle. Use :func:`qvq_dense_oracle_forward` for inference comparisons.
        """

        inner = self.get_inner_weight_tensor(dtype=x.dtype)
        return x @ inner

    def _reference_fp8_inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Emulate the exact E4M3-rounded level table deployed by Hopper."""

        fp8_levels, level_scale = self._prepare_hopper_fp8_levels(x.device)
        deployed_levels = fp8_levels.to(torch.float32).mul(level_scale)
        canonical_levels = pgc16_levels_for_version(self.codebook_version).to(
            device=x.device, dtype=torch.float32
        )
        inner = self.get_inner_weight_tensor(dtype=torch.float32)
        level_indices = torch.searchsorted(canonical_levels, inner)
        inner = deployed_levels[level_indices]
        return x.to(torch.float32) @ inner

    def _inner_forward(
        self,
        x: torch.Tensor,
        *,
        return_ordered_partials: bool = False,
        ordered_split_count: int | None = None,
    ) -> torch.Tensor:
        window_config = getattr(self, "_p32_window_config", None)
        if (window_config is not None and (
                window_config.algorithm.startswith("hopper_")
                or window_config.algorithm in ("ampere_window", "amd_gfx950")
            ) and x.dtype in (torch.float16, torch.bfloat16)):
            if return_ordered_partials or ordered_split_count is not None:
                raise ValueError("explicit window policy requires complete inner output")
            from ...quantization.qvq_rank8 import explicit_window_inner

            return explicit_window_inner(self, x, window_config)
        if ordered_split_count is not None and not return_ordered_partials:
            raise ValueError("an explicit ordered split requires partial output")
        if return_ordered_partials:
            supported_ordered_shape = (
                (self.in_features, self.out_features) == (8192, 2048)
                and ordered_split_count is None
            ) or (
                (self.in_features, self.out_features) == (17408, 5120)
                and ordered_split_count in (17, 34)
            )
            if (
                x.device.type != "cuda"
                or self.trellis_window != 16
                or self.dual_v2
                or not self.v2b2_p32
                or self.vector_size != 2
                or x.dtype != torch.float16
                or not 0 < x.shape[0] <= 16
                or not supported_ordered_shape
            ):
                raise RuntimeError(
                    "ordered partial output requires a measured H100 down P32 path"
                )
            properties = torch.cuda.get_device_properties(x.device)
            if properties.name != "NVIDIA H100" or (
                properties.major,
                properties.minor,
            ) != (9, 0):
                raise RuntimeError(
                    "ordered partial output requires the measured physical H100"
                )
        if self.bank_count in (2, 4) and (
            not self._bank_ids_loaded
            or self.bank_ids is None
            or self.bank_ids.device.type == "meta"
        ):
            # Accelerate's direct tensor loader can install buffers without
            # invoking ``_load_from_state_dict`` on the meta shell.  Once a
            # concrete selector arrives, validate it exactly once and mark the
            # module ready; a missing/meta selector still fails closed.
            if self.bank_ids is None or self.bank_ids.device.type == "meta":
                raise RuntimeError(
                    "QVQ banked module cannot run before bank_ids selectors are loaded"
                )
            tile_count = (self.in_features // 16) * (self.out_features // 16)
            selector_count = (
                tile_count * 8
                if self.v2b2_p32
                else tile_count * 4
                if self.v2b4_p64
                else tile_count
            )
            if self.v2b2_p32:
                unpack_qvq_binary_bank_ids(self.bank_ids, selector_count)
            else:
                unpack_qvq_bank_ids(self.bank_ids, selector_count)
            self._bank_ids_loaded = True
        if self.training:
            return self._reference_inner_forward(x)
        if x.device.type == "mps":
            from ...utils.qvq_mps import (
                _prepare_qvq_mps_compander,
                qvq_mps_gemv,
                qvq_mps_supported,
            )

            if (
                self.trellis_window != 16
                or self.dual_v2
                or self.v2b4_p64
                or self.v2b2_p32
                or not qvq_mps_supported()
            ):
                return self._reference_inner_forward(x)

            prepared = getattr(self, "_qvq_mps_compander", None)
            if prepared is None or prepared.levels.device != x.device:
                prepared = _prepare_qvq_mps_compander(
                    x.device,
                    self.codebook_version,
                )
                self._qvq_mps_compander = prepared
            packed_bank_ids = self._prepare_mps_bank_ids(x.device)
            native_x = x.contiguous()
            row_scale = None
            if native_x.dtype == torch.float32:
                native_x, row_scale = _qvq_mps_narrow_with_row_scale(native_x)
            else:
                native_x = native_x.to(torch.float16)
            output = qvq_mps_gemv(
                native_x,
                self.trellis.contiguous(),
                self.bits,
                out_features=self.out_features,
                codebook_version=self.codebook_version,
                vector_size=self.vector_size,
                bank_ids=packed_bank_ids,
                output_fp32=True,
                _prepared_compander=prepared,
            )
            return output if row_scale is None else output * row_scale
        if x.device.type == "cuda":
            from ...utils.qvq_cuda import qvq_cuda_device_supported, qvq_cuda_gemv

            # Native GEMV intentionally supports FP16/BF16 inputs. FP32 has no
            # packed native specialization and remains on the dense reference
            # path; BF16 reaches native GEMV below.
            if self.trellis_window != 16 or self.dual_v2 or x.dtype == torch.float32:
                return self._reference_inner_forward(x)

            cuda_bank_ids = None
            cuda_bank_alt_id = 0
            if self.bank_ids is not None:
                with self._qvq_cuda_bank_cache_lock:
                    source_object = self.bank_ids
                    alt_source_object = self.bank_alt_id if self.v2b2_p32 else None
                    if source_object is None:
                        self._qvq_cuda_bank_cache = None
                        cuda_bank_ids = None
                        source_object = None
                    if source_object is None:
                        pass
                    else:
                        current_version = source_object._version
                    cached = self._qvq_cuda_bank_cache
                    if (
                        source_object is not None
                        and cached is not None
                        and cached[0] is source_object
                        and cached[1] == current_version
                        and cached[2] == x.device
                        and cached[3] is alt_source_object
                        and cached[4]
                        == (
                            -1
                            if alt_source_object is None
                            else alt_source_object._version
                        )
                        and self.bank_ids is source_object
                        and source_object._version == current_version
                    ):
                        cuda_bank_ids = cached[5]
                        cuda_bank_alt_id = cached[6]
                        cached = None
                    if cuda_bank_ids is None:
                        self._require_prepared_outside_capture(
                            x.device, "CUDA bank selector payload"
                        )
                        # Clone between two version reads. If a free-threaded
                        # writer mutates during the clone, retry rather than
                        # publishing a snapshot under the wrong version.
                        source_version = -1
                        source = None
                        for _ in range(3):
                            if self.bank_ids is not source_object:
                                break
                            before = source_object._version
                            candidate = source_object.detach().clone()
                            after = source_object._version
                            if before == after and self.bank_ids is source_object:
                                source_version = after
                                source = candidate
                                break
                        if source is None:
                            raise RuntimeError(
                                "QVQ CUDA bank selector mutated during snapshot"
                            )
                        tile_count = (self.in_features // 16) * (
                            self.out_features // 16
                        )
                        if self.v2b2_p32:
                            selector_count = tile_count * 8
                            cuda_bank_ids = pack_qvq_binary_bank_ids(
                                unpack_qvq_binary_bank_ids(source, selector_count)
                            ).to(device=x.device)
                        elif self.v2b4_p64:
                            selector_count = tile_count * 4
                            cuda_bank_ids = pack_qvq_bank_ids(
                                unpack_qvq_bank_ids(source, selector_count)
                            ).to(device=x.device)
                        else:
                            cuda_bank_ids = unpack_qvq_bank_ids(source, tile_count).to(
                                device=x.device
                            )
                        alt_version = -1
                        if alt_source_object is not None:
                            alt_version = alt_source_object._version
                            cuda_bank_alt_id = int(alt_source_object.detach().item())
                            if (
                                alt_source_object is not self.bank_alt_id
                                or alt_version != alt_source_object._version
                                or not 1 <= cuda_bank_alt_id <= 3
                            ):
                                raise RuntimeError(
                                    "QVQ CUDA alternative-bank metadata changed during snapshot"
                                )
                        if self.bank_ids is not source_object:
                            raise RuntimeError(
                                "QVQ CUDA bank selector replaced during snapshot"
                            )
                        self._qvq_cuda_bank_cache = (
                            source_object,
                            source_version,
                            x.device,
                            alt_source_object,
                            alt_version,
                            cuda_bank_ids,
                            cuda_bank_alt_id,
                        )
            if not qvq_cuda_device_supported(x.device):
                if (
                    torch.version.hip is not None
                    and self.v2b2_p32
                    and self.vector_size == 2
                    and x.dtype == torch.float16
                    and qvq_transition_bits(self.bits, vector_size=2) in (4, 5, 6, 7)
                ):
                    from ...utils.qvq_amd import qvq_p32_amd, qvq_p32_amd_supported
                    from ...utils.qvq_cuda import _pgc16_levels

                    if qvq_p32_amd_supported(x.device):
                        with self._qvq_cuda_bank_cache_lock:
                            window = self._prepare_hopper_p32_window(x.device)
                        return qvq_p32_amd(
                            x.contiguous(),
                            window,
                            _pgc16_levels(x.device, self.codebook_version),
                            cuda_bank_ids,
                            self.bits,
                            out_features=self.out_features,
                            bank_alt_id=cuda_bank_alt_id,
                            output_fp32=True,
                        )
                return self._reference_inner_forward(x)

            # Hopper's RS-WGMMA path consumes the storage-neutral continuous
            # P32 window layout.  Keep checkpoints in canonical planar form,
            # lazily repack once per module. H100 additionally uses the native
            # row-tiled grid through M4096; M32/M64 buckets decode each P32
            # fragment once for two/four independent M16 tensor-core tiles.
            # H200 retains automatic exact M16 tiling until the large-M grid
            # is independently accepted there. Padded rows are always zero.
            if (
                self.v2b2_p32
                and self.vector_size == 2
                and x.dtype == torch.float16
                # The window-owned Hopper large-M path supports the full
                # public M range through 8192.  Keeping the old 4096 guard
                # here silently fell through to the planar CUDA fallback,
                # which is unavailable after a production window-only load.
                and 0 < x.shape[0] <= 8192
                and self.in_features % 256 == 0
                and self.out_features % 256 == 0
                and qvq_transition_bits(self.bits, vector_size=2) in (4, 5, 6, 7)
            ):
                properties = torch.cuda.get_device_properties(x.device)
                if (
                    properties.major == 9
                    and properties.minor == 0
                    and ("H100" in properties.name or "H200" in properties.name)
                ):
                    from ...utils.qvq_cuda import _pgc16_levels
                    from ...utils.qvq_wgmma_cuda import (
                        qvq_h100_large_m_ordered_split_count,
                        qvq_h100_ordered_split_count,
                        qvq_p32_window_wgmma_m16_tma,
                        qvq_p32_window_wgmma_m16_tma_ordered_partials,
                        qvq_p32_window_wgmma_m16_tma_ordered_split,
                        qvq_p32_window_wgmma_single_large_m_packed,
                    )

                    with self._qvq_cuda_bank_cache_lock:
                        window = self._prepare_hopper_p32_window(x.device)
                    wgmma_input = x.contiguous()
                    logical_rows = int(wgmma_input.shape[0])
                    large_m_hopper = logical_rows > 16
                    padded_rows = (
                        16
                        if logical_rows <= 16
                        else 32
                        if logical_rows <= 32
                        else ((logical_rows + 63) // 64) * 64
                    )
                    if wgmma_input.shape[0] != padded_rows:
                        padded = torch.zeros(
                            (padded_rows, self.in_features),
                            dtype=wgmma_input.dtype,
                            device=wgmma_input.device,
                        )
                        padded[: wgmma_input.shape[0]].copy_(wgmma_input)
                        wgmma_input = padded
                    if large_m_hopper:
                        large_m_split = qvq_h100_large_m_ordered_split_count(
                            device_name=properties.name,
                            compute_capability=(properties.major, properties.minor),
                            logical_rows=logical_rows,
                            in_features=self.in_features,
                            out_features=self.out_features,
                        )
                        output = qvq_p32_window_wgmma_single_large_m_packed(
                            wgmma_input,
                            window,
                            _pgc16_levels(x.device, self.codebook_version),
                            cuda_bank_ids,
                            self.bits,
                            out_features=self.out_features,
                            bank_alt_id=cuda_bank_alt_id,
                            # Decode-era split policies are intentionally not
                            # inherited by prefill: their FP32 partial planes
                            # grow as split*M*N and lose once row reuse fills
                            # the grid. A measured large-M policy may override
                            # this in a later phase.
                            split_count=large_m_split,
                        )
                        return output[:logical_rows]
                    transition_bits = qvq_transition_bits(self.bits, vector_size=2)
                    ordered_split = qvq_h100_ordered_split_count(
                        device_name=properties.name,
                        compute_capability=(properties.major, properties.minor),
                        logical_rows=min(int(x.shape[0]), 16),
                        in_features=self.in_features,
                        out_features=self.out_features,
                        transition_bits=transition_bits,
                    )
                    if ordered_split_count is not None:
                        ordered_split = int(ordered_split_count)
                    if return_ordered_partials:
                        if ordered_split <= 1:
                            raise RuntimeError(
                                "ordered partial output requires a measured split policy"
                            )
                        kernel = qvq_p32_window_wgmma_m16_tma_ordered_partials
                    else:
                        kernel = (
                            qvq_p32_window_wgmma_m16_tma_ordered_split
                            if ordered_split
                            else qvq_p32_window_wgmma_m16_tma
                        )
                    kernel_kwargs = (
                        {"split_count": ordered_split} if ordered_split else {}
                    )
                    output = kernel(
                        wgmma_input,
                        window,
                        _pgc16_levels(x.device, self.codebook_version),
                        cuda_bank_ids,
                        self.bits,
                        out_features=self.out_features,
                        bank_alt_id=cuda_bank_alt_id,
                        **kernel_kwargs,
                    )
                    return output if return_ordered_partials else output[: x.shape[0]]

            if self.window_only:
                # Unsupported optional policies may deliberately fall back to
                # the legacy child reference. The planar payload must have been
                # prepared by the eager fallback before graph capture.
                planar = self._prepare_planar_fallback()
                return qvq_cuda_gemv(
                    x.contiguous(),
                    planar,
                    self.bits,
                    out_features=self.out_features,
                    codebook_version=self.codebook_version,
                    output_fp32=x.dtype in (torch.float16, torch.bfloat16),
                    vector_size=self.vector_size,
                    bank_ids=cuda_bank_ids,
                    v2b4_p64=self.v2b4_p64,
                    v2b2_p32=self.v2b2_p32,
                    bank_alt_id=cuda_bank_alt_id,
                    _bank_ids_validated=True,
                )
            return qvq_cuda_gemv(
                x.contiguous(),
                self.trellis.contiguous(),
                self.bits,
                out_features=self.out_features,
                codebook_version=self.codebook_version,
                output_fp32=x.dtype in (torch.float16, torch.bfloat16),
                vector_size=self.vector_size,
                bank_ids=cuda_bank_ids,
                v2b4_p64=self.v2b4_p64,
                v2b2_p32=self.v2b2_p32,
                bank_alt_id=cuda_bank_alt_id,
                _bank_ids_validated=True,
            )
        if x.device.type == "cpu":
            from ...utils.qvq_cpu import qvq_cpu_gemv, qvq_cpu_supported

            if (
                self.trellis_window != 16
                or self.vector_size != 2
                or self.dual_v2
                or self.window_only
                or not qvq_cpu_supported()
            ):
                return self._reference_inner_forward(x)
            bank_alt_id = 0
            if self.bank_ids is not None and self.v2b2_p32:
                bank_alt_id = int(self.bank_alt_id.detach().item())
                if not 1 <= bank_alt_id <= 3:
                    return self._reference_inner_forward(x)
            return qvq_cpu_gemv(
                x,
                self.trellis,
                self.bits,
                out_features=self.out_features,
                vector_size=self.vector_size,
                bank_ids=self.bank_ids,
                v2b4_p64=self.v2b4_p64,
                v2b2_p32=self.v2b2_p32,
                bank_alt_id=bank_alt_id,
                use_dense_cache=False,
            )
        return self._reference_inner_forward(x)

    def _inner_forward_fp8(
        self,
        input: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Consume the exact transformed E4M3 operand through Hopper WGMMA."""

        if self.bank_ids is None or self.bank_alt_id is None:
            raise RuntimeError("QVQ P32 FP8 WGMMA requires loaded bank metadata")
        tile_count = (self.in_features // 16) * (self.out_features // 16)
        selector_count = tile_count * 8
        with self._qvq_cuda_bank_cache_lock:
            source = self.bank_ids
            source_version = source._version
            alt_source = self.bank_alt_id
            alt_version = alt_source._version
            cached = self._qvq_cuda_bank_cache
            if (
                cached is not None
                and cached[0] is source
                and cached[1] == source_version
                and cached[2] == input.device
                and cached[3] is alt_source
                and cached[4] == alt_version
            ):
                cuda_bank_ids = cached[5]
                bank_alt_id = cached[6]
            else:
                # FP8 WGMMA snapshots and packs mutable bank metadata on a
                # cache miss.  This is preparation work and must never occur
                # during CUDA Graph capture, where the clone/device transfer
                # would allocate and the alternative-bank read would sync.
                self._require_prepared_outside_capture(
                    input.device, "FP8 bank selector payload"
                )
                cuda_bank_ids = pack_qvq_binary_bank_ids(
                    unpack_qvq_binary_bank_ids(source.detach().clone(), selector_count)
                ).to(device=input.device)
                bank_alt_id = int(alt_source.detach().item())
                if (
                    self.bank_ids is not source
                    or source._version != source_version
                    or self.bank_alt_id is not alt_source
                    or alt_source._version != alt_version
                    or not 1 <= bank_alt_id <= 3
                ):
                    raise RuntimeError("QVQ P32 FP8 bank metadata changed during snapshot")
                self._qvq_cuda_bank_cache = (
                    source,
                    source_version,
                    input.device,
                    alt_source,
                    alt_version,
                    cuda_bank_ids,
                    bank_alt_id,
                )

        from ...utils.qvq_wgmma_cuda import qvq_p32_window_wgmma_fp8_m16

        window = self._prepare_hopper_p32_window(input.device)
        fp8_levels, level_scale = self._prepare_hopper_fp8_levels(input.device)
        return qvq_p32_window_wgmma_fp8_m16(
            input,
            input_scale,
            window,
            fp8_levels,
            cuda_bank_ids,
            self.bits,
            out_features=self.out_features,
            bank_alt_id=bank_alt_id,
            level_scale=level_scale,
        )

    def _fp8_kernel_ineligible_reason(self, transformed: torch.Tensor) -> str | None:
        if transformed.device.type != "cuda":
            return "non_cuda"
        if not self.v2b2_p32 or self.vector_size != 2 or self.trellis_window != 16:
            return "non_p32"
        if self.bits not in (2, 2.5, 3, 3.5):
            return "unsupported_rate"
        if transformed.ndim != 2 or transformed.shape[0] <= 0:
            return "unsupported_rank_or_rows"
        if self.in_features % 32 or self.out_features % 64:
            return "unsupported_shape"
        properties = torch.cuda.get_device_properties(transformed.device)
        if (properties.major, properties.minor) != (9, 0):
            return "non_sm90"
        if not device_supports_native_fp8(transformed.device):
            return "native_fp8_unavailable"
        return None

    def _prepare_activation_input(
        self,
        x_2d: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        straight_through: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Apply the checkpoint's A8 contract and retain FP8 for the fused CUDA transform."""

        config = self.activation
        if config is None:
            return x_2d.to(compute_dtype), None, 0
        if config.target == "p32_operand":
            # The model-visible activation remains in its native BF16/FP16
            # dtype. Quantization happens after SU/Hadamard, at the exact
            # operand boundary consumed by Hopper WGMMA.
            return x_2d.to(compute_dtype), None, 0
        validate = straight_through or x_2d.device.type == "cpu"
        if straight_through:
            quantized, scale, dequantized = fake_quantize_qvq_fp8_activation(
                x_2d,
                format=config.format,
                scale_method=config.scale_method,
                straight_through=True,
                validate=validate,
            )
        else:
            # The native fused Hadamard consumes FP8 + row scale directly.
            # Do not eagerly allocate and populate a dequantized tensor that
            # this path immediately discards.
            quantized, scale = quantize_qvq_fp8_activation(
                x_2d,
                format=config.format,
                scale_method=config.scale_method,
                # Avoid a device-to-host synchronization in the inference hot
                # path. Runtime non-finites retain ordinary propagation.
                validate=validate,
            )
            dequantized = None
        native_fp8_transform = (
            not straight_through
            and self.input_hadamard
            and x_2d.device.type == "cuda"
            and compute_dtype == torch.float16
            and device_supports_native_fp8(x_2d.device)
        )
        if native_fp8_transform:
            input_rounding_mode = 1 if x_2d.dtype == torch.bfloat16 else 0
            return quantized, scale, input_rounding_mode
        if dequantized is None:
            dequantized = dequantize_qvq_fp8_activation(
                quantized,
                scale,
                dtype=x_2d.dtype,
            )
        return dequantized.to(compute_dtype), None, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"QVQ expected input width {self.in_features}, got {x.shape[-1]}"
            )
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.out_features))
        if getattr(self, "_p32_rank8_enabled", False) and self.training:
            raise RuntimeError("window recovery is inference-only")
        delegate = getattr(self, "_qvq_grouped_p32_delegate", None)
        if delegate is not None:
            state, consumer_index, module_name = delegate
            return state.consume(consumer_index, module_name, self, x)
        input_dtype = x.dtype
        compute_dtype = _qvq_compute_dtype(input_dtype, x.device.type)
        x_2d = x.reshape(-1, self.in_features)

        # Composite-width and BF16 forwards may execute the capture-safe
        # overflow-rescue branch even when ordinary warmup data is finite.
        # Materialize its constant dtype variants during eager warmup so a raw
        # torch.cuda.graph caller receives the same no-allocation guarantee as
        # P32WindowGraphs.capture.
        rescue_possible = input_dtype == torch.bfloat16 or (
            input_dtype == torch.float16
            and (
                self.in_features < _FP16_STABLE_HADAMARD_MIN_WIDTH
                or self.in_features > _QVQ_HADAMARD_MAX_WIDTH
                or self.in_features & (self.in_features - 1)
                or self.out_features > _QVQ_HADAMARD_MAX_WIDTH
                or self.out_features & (self.out_features - 1)
            )
        )
        if (
            rescue_possible
            and x.device.type == "cuda"
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._prepare_cuda_graph_auxiliary_caches()

        # The folded gfx950 path has no low-precision butterfly intermediate,
        # so it cannot trigger the transform-overflow rescue below. Return at
        # this boundary to avoid an otherwise redundant device-wide finite
        # reduction and host synchronization on every Qwen projection.
        amd_folded = (
            None if getattr(self, "_p32_rank8_enabled", False)
            else self._qvq_amd_folded_forward(x_2d, compute_dtype)
        )
        if amd_folded is not None:
            return amd_folded.reshape(*x.shape[:-1], self.out_features).to(input_dtype)

        # The CUDA inner kernel accumulates and returns FP32. Preserve that
        # range through the output Hadamard/SV epilogue, then round only the
        # completed linear result to the model dtype. This removes the
        # factorization-only FP16 overflow without host synchronization or a
        # duplicate BF16 path during CUDA-graph capture.
        output = self._forward_compute_dtype(
            x_2d, compute_dtype, output_dtype=input_dtype
        )
        if (
            input_dtype == torch.bfloat16
            and x.device.type == "cuda"
        ):
            if torch.cuda.is_current_stream_capturing():
                rescued = self._forward_compute_dtype(
                    x.reshape(-1, self.in_features).to(torch.bfloat16),
                    torch.bfloat16,
                )
                output = torch.where(
                    torch.isfinite(output).all(),
                    output,
                    rescued,
                )
            elif not torch.isfinite(output).all():
                output = self._forward_compute_dtype(
                    x.reshape(-1, self.in_features).to(torch.bfloat16),
                    torch.bfloat16,
                )
        elif (
            input_dtype == torch.float16
            and x.device.type == "cuda"
            and (
                self.in_features < _FP16_STABLE_HADAMARD_MIN_WIDTH
                or self.in_features > _QVQ_HADAMARD_MAX_WIDTH
                or self.in_features & (self.in_features - 1)
                or self.out_features > _QVQ_HADAMARD_MAX_WIDTH
                or self.out_features & (self.out_features - 1)
            )
        ):
            # Composite/non-fused transform widths retain a BF16 retry because
            # their Python FP16 butterfly cannot rescue a transient overflow.
            if torch.cuda.is_current_stream_capturing():
                rescued = self._forward_compute_dtype(
                    x.reshape(-1, self.in_features).to(torch.bfloat16),
                    torch.bfloat16,
                )
                output = torch.where(torch.isfinite(output).all(), output, rescued)
            elif not torch.isfinite(output).all():
                output = self._forward_compute_dtype(
                    x.reshape(-1, self.in_features).to(torch.bfloat16),
                    torch.bfloat16,
                )
        return output.reshape(*x.shape[:-1], self.out_features).to(input_dtype)

    def transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this module's complete input-side QVQ transform.

        This is the ownership boundary used by graph planners that share one
        identical input transform across sibling projections.  The returned
        tensor is still an activation; no inner QVQ decode has run.  Callers
        must prove that every consumer has identical ``SU`` and
        ``input_hadamard`` state before reusing it.
        """

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"QVQ expected input width {self.in_features}, got {x.shape[-1]}"
            )
        if self.training:
            raise RuntimeError("shared QVQ input transforms are inference-only")
        if x.numel() == 0:
            return x
        compute_dtype = _qvq_compute_dtype(x.dtype, x.device.type)
        x_2d = x.reshape(-1, self.in_features)
        x_2d, input_scale, input_rounding_mode = self._prepare_activation_input(x_2d, compute_dtype)
        if self.input_hadamard:
            transformed = _qvq_hadamard_fused(
                x_2d,
                input_scale=input_scale,
                input_rounding_mode=input_rounding_mode,
                pre_scale=self._cached_cast("SU", compute_dtype),
                scale_mode=(
                    2
                    if compute_dtype == torch.float16
                    and self.in_features >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                    else 1
                ),
            )
        else:
            transformed = x_2d * self._cached_cast("SU", compute_dtype)
        return transformed.reshape(*x.shape[:-1], self.in_features)

    def forward_pretransformed(
        self,
        transformed: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Run inner decode/output recovery on a proven input transform.

        ``transformed`` must equal :meth:`transform_input` for this module.
        The method intentionally bypasses both ``SU`` and the input Hadamard,
        allowing an architecture-level shared-transform coordinator to avoid
        duplicate launches without changing checkpoint tensor semantics.
        """

        if transformed.shape[-1] != self.in_features:
            raise ValueError(
                f"QVQ expected transformed input width {self.in_features}, "
                f"got {transformed.shape[-1]}"
            )
        if self.training:
            raise RuntimeError(
                "pretransformed QVQ inference is unavailable in training mode"
            )
        if transformed.numel() == 0:
            return transformed.new_empty(
                (*transformed.shape[:-1], self.out_features),
                dtype=output_dtype or transformed.dtype,
            )
        target_dtype = transformed.dtype if output_dtype is None else output_dtype
        compute_dtype = _qvq_compute_dtype(transformed.dtype, transformed.device.type)
        transformed_2d = transformed.reshape(-1, self.in_features).to(compute_dtype)
        output = self._forward_pretransformed_compute_dtype(
            transformed_2d,
            compute_dtype,
            output_dtype=output_dtype,
        )
        return output.reshape(*transformed.shape[:-1], self.out_features).to(
            target_dtype
        )

    def forward_prequantized_fp8(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Consume one shared, already-quantized P32 activation operand.

        Grouped QKV and gate/up projections have a proven-identical SU/Hadamard
        transform.  Their dynamic per-row E4M3 conversion is therefore also
        identical and may be performed once without changing checkpoint or
        arithmetic semantics.
        """

        config = self.activation
        if config is None or config.target != "p32_operand":
            raise RuntimeError(
                "prequantized QVQ input requires target=p32_operand"
            )
        if self.training:
            raise RuntimeError("prequantized QVQ inference is unavailable in training mode")
        if quantized.shape[-1] != self.in_features:
            raise ValueError(
                f"QVQ expected prequantized input width {self.in_features}, "
                f"got {quantized.shape[-1]}"
            )
        compute_dtype = _qvq_compute_dtype(output_dtype, quantized.device.type)
        if output_dtype == torch.bfloat16 and quantized.device.type == "cuda":
            compute_dtype = torch.float32
        quantized_2d = quantized.reshape(-1, self.in_features)
        scale_2d = scale.reshape(-1, 1)
        reason = (
            "kernel_disabled"
            if config.kernel_mode == "disable"
            else self._fp8_kernel_ineligible_reason(quantized_2d)
        )
        if config.kernel_mode != "disable":
            self._record_fp8_kernel("requested")
        if reason is None:
            self._record_fp8_kernel("eligible")
            try:
                output = self._inner_forward_fp8(quantized_2d, scale_2d)
            except RuntimeError:
                if config.kernel_mode == "require":
                    self._record_fp8_kernel("rejected", "launch_error")
                    raise
                reason = "launch_error"
            else:
                self._record_fp8_kernel("executed")
                recovered = self._recover_output_compute_dtype(
                    output, compute_dtype, target_dtype=output_dtype
                )
                return recovered.reshape(
                    *quantized.shape[:-1], self.out_features
                ).to(output_dtype)
        elif config.kernel_mode == "require":
            self._record_fp8_kernel("rejected", reason)
            raise RuntimeError(
                f"required QVQ P32 FP8 WGMMA path is ineligible: {reason}"
            )
        self._record_fp8_kernel("fallback", reason)
        transformed = dequantize_qvq_fp8_activation(
            quantized_2d,
            scale_2d,
            dtype=torch.float32,
        )
        output = self._reference_fp8_inner_forward(transformed)
        recovered = self._recover_output_compute_dtype(
            output, torch.float32, target_dtype=output_dtype
        )
        return recovered.reshape(*quantized.shape[:-1], self.out_features).to(
            output_dtype
        )

    def recover_output(
        self,
        inner_output: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Apply this module's output-side QVQ recovery to decoded inner output.

        Grouped packed decoders use this boundary to decode several sibling
        projections in one kernel while retaining each module's independent
        output Hadamard, ``SV``, and bias.  No input transform or trellis decode
        runs here.
        """

        if inner_output.shape[-1] != self.out_features:
            raise ValueError(
                f"QVQ expected inner output width {self.out_features}, "
                f"got {inner_output.shape[-1]}"
            )
        if self.training:
            raise RuntimeError("separate QVQ output recovery is inference-only")
        target_dtype = inner_output.dtype if output_dtype is None else output_dtype
        if inner_output.numel() == 0:
            return inner_output.to(target_dtype)
        compute_dtype = _qvq_compute_dtype(target_dtype, inner_output.device.type)
        recovered = self._recover_output_compute_dtype(
            inner_output.reshape(-1, self.out_features),
            compute_dtype,
        )
        return recovered.reshape(*inner_output.shape[:-1], self.out_features).to(
            target_dtype
        )

    def _forward_compute_dtype(
        self,
        x_2d: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.training:
            compute_dtype = self._qvq_operand_compute_dtype(x_2d, compute_dtype)
            x_2d, _input_scale, _input_rounding_mode = self._prepare_activation_input(
                x_2d,
                compute_dtype,
                straight_through=True,
            )
            # Keep the differentiable Python butterfly path for the reference
            # forward used in training.
            transformed_input = x_2d * self.SU.to(compute_dtype)
            if self.input_hadamard:
                input_transform = (
                    matmul_hadU_stable
                    if compute_dtype == torch.float16
                    and (
                        transformed_input.device.type == "mps"
                        or transformed_input.shape[-1]
                        >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                    )
                    else matmul_hadU
                )
                transformed = input_transform(transformed_input)
            else:
                transformed = transformed_input
            if (
                self.activation is not None
                and self.activation.target == "p32_operand"
            ):
                _, _, transformed = fake_quantize_qvq_fp8_activation(
                    transformed,
                    format=self.activation.format,
                    scale_method=self.activation.scale_method,
                    straight_through=True,
                    validate=True,
                )
            output = self._inner_forward(transformed)
            if self.output_hadamard:
                output_transform = (
                    matmul_hadU_stable
                    if compute_dtype == torch.float16
                    and (
                        output.device.type == "mps"
                        or output.shape[-1] >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                    )
                    else matmul_hadU
                )
                output = output_transform(output)
            output = output * self.SV.to(compute_dtype)
            if self.bias is not None:
                output = output + self.bias.to(compute_dtype)
        else:
            compute_dtype = self._qvq_operand_compute_dtype(x_2d, compute_dtype)
            hidden = None
            if (getattr(self, "_p32_rank8_enabled", False)
                    and self._p32_window_config.recovery_projection == "input_fused"
                    and compute_dtype == torch.float16):
                from ...quantization.qvq_rank8 import validate_rank8_state
                from ...utils.qvq_rank8_triton import rank8_input_producer

                validate_rank8_state(self)
                transformed, hiddens = rank8_input_producer(
                    x_2d.to(compute_dtype).contiguous(), self._cached_cast("SU", compute_dtype),
                    self.rank8_A, hadamard=self.input_hadamard,
                )
                hidden = hiddens[0]
            else:
                transformed = self._qvq_prepare_inference_input(x_2d, compute_dtype)
            return self._forward_pretransformed_compute_dtype(
                transformed, compute_dtype, output_dtype=output_dtype, rank8_hidden=hidden
            )
        return output

    def _forward_pretransformed_compute_dtype(
        self,
        transformed: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        output_dtype: torch.dtype | None = None,
        rank8_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_p32_rank8_enabled", False):
            from ...quantization.qvq_rank8 import validate_rank8_state

            validate_rank8_state(self)
        config = self.activation
        if config is not None and config.target == "p32_operand":
            quantized, scale = quantize_qvq_fp8_activation(
                transformed,
                format=config.format,
                scale_method=config.scale_method,
                validate=False,
            )
            reason = "kernel_disabled" if config.kernel_mode == "disable" else self._fp8_kernel_ineligible_reason(
                transformed
            )
            if config.kernel_mode != "disable":
                self._record_fp8_kernel("requested")
            if reason is None:
                self._record_fp8_kernel("eligible")
                try:
                    output = self._inner_forward_fp8(quantized, scale)
                except RuntimeError:
                    if config.kernel_mode == "require":
                        self._record_fp8_kernel("rejected", "launch_error")
                        raise
                    reason = "launch_error"
                else:
                    self._record_fp8_kernel("executed")
                    return self._recover_output_compute_dtype(
                        output, compute_dtype, target_dtype=output_dtype
                    )
            elif config.kernel_mode == "require":
                self._record_fp8_kernel("rejected", reason)
                raise RuntimeError(f"required QVQ P32 FP8 WGMMA path is ineligible: {reason}")
            self._record_fp8_kernel("fallback", reason)
            transformed = dequantize_qvq_fp8_activation(
                quantized,
                scale,
                dtype=torch.float32,
            )
            output = self._reference_fp8_inner_forward(transformed)
            return self._recover_output_compute_dtype(
                output, torch.float32, target_dtype=output_dtype
            )
        concurrent_done = None
        concurrent_stream = None
        concurrent = (
            getattr(self, "_p32_rank8_enabled", False)
            and rank8_hidden is None
            and getattr(self._p32_window_config, "recovery_projection", None)
            == "concurrent_reference"
            and compute_dtype == torch.float16
            and transformed.device.type == "cuda"
            and transformed.is_contiguous()
        )
        if concurrent:
            # Launch the numerically reference FP32->FP16 projection on a
            # prepared auxiliary stream while the current stream decodes the
            # P32 window.  Events are captured as dependencies, so replay has
            # no host synchronization or stream allocation.
            current_stream = torch.cuda.current_stream(transformed.device)
            device_key = int(
                transformed.device.index
                if transformed.device.index is not None
                else torch.cuda.current_device()
            )
            m, k = transformed.shape
            warm_key = (device_key, int(m), int(k))
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and warm_key not in self._qvq_rank8_concurrent_warm:
                raise RuntimeError(
                    "QVQ concurrent rank8 projection must be warmed before CUDA Graph capture"
                )
            concurrent_stream, ready, done = self._rank8_concurrent_resources(transformed.device)
            ready.record(current_stream)
            with torch.cuda.stream(concurrent_stream):
                concurrent_stream.wait_event(ready)
                rank8_hidden = (
                    transformed.float() @ self._cached_rank8_factor("A")
                ).half()
                done.record(concurrent_stream)
            concurrent_done = done
            if not capturing:
                self._qvq_rank8_concurrent_warm.add(warm_key)
        output = self._inner_forward(transformed)
        if concurrent_done is not None:
            # Queue the dependency after the window decoder so both branches
            # overlap and the correction cannot observe a stale hidden value.
            torch.cuda.current_stream(transformed.device).wait_event(concurrent_done)
        if getattr(getattr(self, "_p32_window_config", None), "recovery_kernel", None) == "fused_epilogue":
            from ...quantization.qvq_rank8 import fused_rank8_output

            return fused_rank8_output(
                self, transformed, output, compute_dtype, hidden=rank8_hidden, output_dtype=output_dtype
            )
        if getattr(self, "_p32_rank8_enabled", False):
            from ...quantization.qvq_rank8 import add_rank8_correction

            output = add_rank8_correction(self, transformed, output, hidden=rank8_hidden)
        return self._recover_output_compute_dtype(
            output, compute_dtype, target_dtype=output_dtype
        )

    def _recover_output_compute_dtype(
        self,
        output: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        target_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        output_dtype = output.dtype
        if self.output_hadamard:
            target_bf16 = (
                target_dtype == torch.bfloat16
                and output_dtype == torch.float32
            )
            native_bf16_store = (
                target_bf16
                and output.device.type == "cuda"
                and output.is_contiguous()
                and self.out_features >= 2
                and self.out_features & (self.out_features - 1) == 0
                and self.out_features <= _QVQ_HADAMARD_MAX_WIDTH
                and qvq_cuda_device_supported(output.device)
                and qvq_cuda_available()
            )
            recovered = _qvq_hadamard_fused(
                output,
                post_scale=self._cached_cast("SV", compute_dtype, output_dtype),
                bias=self._cached_cast("bias", compute_dtype, output_dtype),
                scale_mode=(
                    3
                    if output_dtype == torch.float32
                    and self.out_features >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                    else 4
                    if output_dtype == torch.float32
                    else 0
                ),
                output_bf16=native_bf16_store,
            )
            return recovered.to(torch.bfloat16) if target_bf16 else recovered
        output = output * self._cached_cast("SV", compute_dtype, output_dtype)
        cached_bias = self._cached_cast("bias", compute_dtype, output_dtype)
        return output if cached_bias is None else output + cached_bias

    def _qvq_operand_compute_dtype(
        self,
        x_2d: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.dtype:
        """Select the shared runtime/replay dtype at the deployed operand boundary."""

        if (
            x_2d.device.type == "cuda"
            and x_2d.dtype == torch.bfloat16
            and self.activation is not None
            and self.activation.target == "p32_operand"
        ):
            # BF16 values can exceed FP16 before SU/Hadamard has reduced their
            # range. Preserve them through that transform; the resulting row
            # is bounded when it is converted to E4M3 below. Replay calls the
            # same helper, so its fitted operand cannot silently narrow first.
            return torch.float32
        return compute_dtype

    def _qvq_prepare_inference_input(
        self,
        x_2d: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        pad_to_16: bool = False,
    ) -> torch.Tensor:
        """Apply the exact inference-side ``SU -> Hadamard`` transform.

        Grouped execution calls this method once on the first child after R0
        has proved that every sibling owns a bit-identical ``SU``.  Keeping the
        operation here prevents the production coordinator from duplicating
        or subtly reordering QVQLinear's numerical contract.
        """

        compute_dtype = self._qvq_operand_compute_dtype(x_2d, compute_dtype)
        x_2d, input_scale, input_rounding_mode = self._prepare_activation_input(
            x_2d, compute_dtype
        )

        if not self.input_hadamard:
            if pad_to_16:
                raise RuntimeError("direct padded input requires an input Hadamard")
            return x_2d * self._cached_cast("SU", compute_dtype)
        return _qvq_hadamard_fused(
            x_2d,
            input_scale=input_scale,
            input_rounding_mode=input_rounding_mode,
            pre_scale=self._cached_cast("SU", compute_dtype),
            scale_mode=(
                2
                if compute_dtype == torch.float16
                and self.in_features >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                else 1
            ),
            pad_to_16=pad_to_16,
        )

    def _qvq_recover_inference_output(
        self,
        output: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        output_fp16: bool = False,
    ) -> torch.Tensor:
        """Apply the exact child-local ``Hadamard -> SV -> bias`` epilogue."""

        output_dtype = output.dtype
        if self.output_hadamard:
            return _qvq_hadamard_fused(
                output,
                post_scale=self._cached_cast("SV", compute_dtype, output_dtype),
                bias=self._cached_cast("bias", compute_dtype, output_dtype),
                scale_mode=(
                    3
                    if output_dtype == torch.float32
                    and self.out_features >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                    else 4
                    if output_dtype == torch.float32
                    else 0
                ),
                output_fp16=output_fp16,
            )
        output = output * self._cached_cast("SV", compute_dtype, output_dtype)
        cached_bias = self._cached_cast("bias", compute_dtype, output_dtype)
        return output if cached_bias is None else output + cached_bias


def qvq_dense_oracle_forward(
    layer: QVQLinear,
    x: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Run the canonical full-layer dense QVQ accuracy oracle in FP32.

    The helper reconstructs and accumulates in FP32 under ``inference_mode`` on
    the explicitly selected device. It never uses production dispatch or
    populates a module cache, and the locally reconstructed dense weight is
    released when the call returns (allocator-reserved memory may remain).

    Kernels claiming the canonical accuracy tier must produce finite FP32 output
    with ``max_abs(actual_fp32 - oracle_fp32) <= 2e-3`` and ``rtol=0``.
    """

    if not isinstance(layer, QVQLinear):
        raise TypeError(f"layer must be a QVQLinear, got {type(layer).__name__}")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.requires_grad:
        raise RuntimeError(
            "qvq_dense_oracle_forward does not accept input that requires gradients"
        )
    if x.shape[-1] != layer.in_features:
        raise ValueError(
            f"QVQ oracle input width must be {layer.in_features}, got {x.shape[-1]}"
        )

    compute_device = torch.device(device)
    inner = None
    try:
        with torch.inference_mode():
            source = layer.window_words if layer.window_only else layer.trellis
            if source is None:
                raise RuntimeError(
                    "QVQ oracle requires a window or planar P32 payload"
                )
            if layer.window_only:
                layer._require_prepared_outside_capture(
                    source.device, "oracle planar reconstruction"
                )
                source = repack_p32_window_to_planar(source, bits=layer.bits)
            inner = reconstruct_qvq_inner_weight(
                source.to(device=compute_device),
                bits=layer.bits,
                vector_size=layer.vector_size,
                trellis_window=layer.trellis_window,
                in_features=layer.in_features,
                out_features=layer.out_features,
                codebook_version=layer.codebook_version,
                bank_ids=None
                if layer.bank_ids is None
                else layer.bank_ids.to(device=compute_device),
                dual_v2=layer.dual_v2,
                v2b4_p64=layer.v2b4_p64,
                v2b2_p32=layer.v2b2_p32,
                bank_alt_id=(
                    None
                    if layer.bank_alt_id is None
                    else layer.bank_alt_id.to(device=compute_device)
                ),
            ).to(dtype=torch.float32)
            x_2d = x.to(device=compute_device).reshape(-1, layer.in_features)
            if (
                layer.activation is not None
                and layer.activation.target == "linear_input"
            ):
                _, _, x_2d = fake_quantize_qvq_fp8_activation(
                    x_2d,
                    format=layer.activation.format,
                    scale_method=layer.activation.scale_method,
                )
            x_2d = x_2d.to(torch.float32)
            transformed = x_2d * layer.SU.to(device=compute_device, dtype=torch.float32)
            if layer.input_hadamard:
                transformed = matmul_hadU(transformed)
            if (
                layer.activation is not None
                and layer.activation.target == "p32_operand"
            ):
                _, _, transformed = fake_quantize_qvq_fp8_activation(
                    transformed,
                    format=layer.activation.format,
                    scale_method=layer.activation.scale_method,
                )
                canonical_levels = pgc16_levels_for_version(
                    layer.codebook_version
                ).to(device=compute_device, dtype=torch.float32)
                fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
                level_scale = float(canonical_levels.abs().amax().item() / fp8_max)
                deployed_levels = torch.clamp(
                    canonical_levels / level_scale,
                    min=-fp8_max,
                    max=fp8_max,
                ).to(torch.float8_e4m3fn).to(torch.float32).mul(level_scale)
                inner = deployed_levels[
                    torch.searchsorted(canonical_levels, inner)
                ]
            output = transformed @ inner
            if layer.output_hadamard:
                output = matmul_hadU(output)
            output = output * layer.SV.to(device=compute_device, dtype=torch.float32)
            if layer.bias is not None:
                output = output + layer.bias.to(
                    device=compute_device, dtype=torch.float32
                )
            return output.reshape(*x.shape[:-1], layer.out_features).detach()
    finally:
        del inner


class QVQReferenceLinear(QVQLinear):
    """Diagnostic/differentiable dense wrapper, not the canonical accuracy oracle.

    Its training graph may retain the reconstructed dense weight. Use
    :func:`qvq_dense_oracle_forward` for memory-clean FP32 accuracy comparisons.
    """

    SUPPORTS_BACKEND_SELECTION = False

    @classmethod
    def verify_supports_params(cls) -> None:
        # This diagnostic subclass inherits the production contract and is
        # deliberately absent from backend selection. BaseQuantLinear's strict
        # declaration audit otherwise requires duplicating every mutable class
        # attribute solely to construct the reference oracle.
        QVQLinear.verify_supports_params()

    def _inner_forward(
        self,
        x: torch.Tensor,
        *,
        return_ordered_partials: bool = False,
        ordered_split_count: int | None = None,
    ) -> torch.Tensor:
        if return_ordered_partials or ordered_split_count is not None:
            raise RuntimeError("reference QVQ execution does not expose split partials")
        return self._reference_inner_forward(x)


__all__ = [
    "QVQLinear",
    "QVQReferenceLinear",
    "qvq_dense_oracle_forward",
]
