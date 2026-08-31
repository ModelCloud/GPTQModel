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
from ...quantization.qvq import (
    QVQ_BITS,
    QVQ_V2B2_P32_LR_TILE_COLS,
    QVQ_V2B2_P32_LR_TILE_ROWS,
    pack_qvq_bank_ids,
    pack_qvq_binary_bank_ids,
    reconstruct_qvq_inner_weight,
    unpack_qvq_bank_ids,
    unpack_qvq_binary_bank_ids,
)
from ...quantization.qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
from ...quantization.qvq_rates import qvq_words_per_tile
from ...quantization.rotation.hadamard_utils import matmul_hadU, matmul_hadU_stable
from ...utils.backend import BACKEND
from ...utils.qvq_cuda import (
    qvq_cuda_available,
    qvq_cuda_device_supported,
    qvq_cuda_hadamard,
)
from . import BaseQuantLinear, FormatSupport

_QVQ_BUFFER_NAMES = ("trellis", "SU", "SV", "bias", "bank_ids", "bank_alt_id")
# The real Llama/Qwen transforms that exposed delayed-normalization overflow
# start at this width. Preserve the original, slightly more accurate FP16
# operation ordering for narrow transforms; the finite-output retry below
# remains the range guard for adversarial narrow inputs.
_FP16_STABLE_HADAMARD_MIN_WIDTH = 2048
_QVQ_HADAMARD_MAX_WIDTH = 16384
_QVQ_FP16_SAFE_MAGNITUDE = torch.finfo(torch.float16).max / 2


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
    pre_scale: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale_mode: int = 0,
) -> torch.Tensor:
    """One fused Hadamard launch (CUDA or CPU AVX-512); Python butterfly fallback otherwise.

    ``scale_mode`` mirrors the original dispatch: 0 (stable, width >= 2048) or
    1 (end-scale, narrower widths). The fused kernel is bitwise-identical to the
    corresponding Python reference and also absorbs the SU pre-scale, SV
    post-scale, and bias elementwise steps with the same rounding sequence.
    """

    n = x.shape[-1]
    if x.device.type == "mps" and pre_scale is not None:
        # Keep x*SU in FP32 until normalization; Metal GEMV narrows only the
        # already range-reduced transform result to FP16.
        scaled = x.to(torch.float32) * pre_scale.to(torch.float32)
        return matmul_hadU_stable(scaled) if n >= _FP16_STABLE_HADAMARD_MIN_WIDTH else matmul_hadU(scaled)
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
        and x.dtype in (torch.float16, torch.float32)
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
        return qvq_cuda_hadamard(x, pre_scale=pre_scale, post_scale=post_scale, bias=bias, scale_mode=mode)
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


def _qvq_mps_narrow_with_row_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        FORMAT.QVQ_V4: FormatSupport(priority=100, bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 4)),
        FORMAT.QVQ_V4_L18: FormatSupport(priority=100, bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 2.5)),
        FORMAT.QVQ_DUAL_V2: FormatSupport(priority=100, bits=QVQ_BITS),
        FORMAT.QVQ_V2B4_P64: FormatSupport(priority=100, bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 3.5)),
        FORMAT.QVQ_V2B2_P32: FormatSupport(priority=100, bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 3.5)),
        FORMAT.QVQ_V2B2_P32_LR: FormatSupport(
            priority=100,
            bits=tuple(bit for bit in QVQ_BITS if float(bit) <= 3.5),
        ),
    }
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY: ClassVar[list[int]] = [16]
    # LR32 uses K32 x N8 logical tiles; the constructor keeps the legacy
    # formats at 16-wide tiles while allowing LR32's N8 shape through the
    # common base validation.
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY: ClassVar[list[int]] = [8]
    SUPPORTS_PACK_DTYPES: ClassVar[list[torch.dtype]] = [torch.int32]
    SUPPORTS_ADAPTERS: ClassVar[list[type[Adapter]]] = []
    SUPPORTS_DEVICES: ClassVar[list[DEVICE]] = [DEVICE.ALL]
    SUPPORTS_PLATFORM: ClassVar[list[PLATFORM]] = [PLATFORM.ALL]
    SUPPORTS_DTYPES: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16, torch.float32]
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
        v2b2_p32_lr: bool = False,
        input_hadamard: bool = True,
        output_hadamard: bool = True,
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
                    FORMAT.QVQ_V2B2_P32_LR
                    if v2b2_p32_lr
                    else
                    FORMAT.QVQ_V2B2_P32
                    if v2b2_p32
                    else FORMAT.QVQ_V2B4_P64
                    if v2b4_p64
                    else FORMAT.QVQ_DUAL_V2
                    if dual_v2
                    else
                    FORMAT.QVQ_V4_L18
                    if trellis_window == 18
                    else FORMAT.QVQ_V4
                    if vector_size == 4
                    else FORMAT.QVQ
                ),
            },
        )
        if v2b2_p32_lr:
            if in_features % QVQ_V2B2_P32_LR_TILE_ROWS or out_features % QVQ_V2B2_P32_LR_TILE_COLS:
                raise ValueError("QVQ V2B2-P32-LR requires in_features divisible by 32 and out_features by 8")
        elif in_features % 16 or out_features % 16:
            raise ValueError("QVQ non-LR formats require in_features and out_features divisible by 16")
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
            raise ValueError("QVQ L18 uses implicit history-selected banks and requires bank_count=1")
        self.trellis_window = trellis_window
        if not isinstance(dual_v2, bool):
            raise TypeError("QVQ dual_v2 must be a bool")
        if not isinstance(v2b4_p64, bool):
            raise TypeError("QVQ v2b4_p64 must be a bool")
        if not isinstance(v2b2_p32, bool):
            raise TypeError("QVQ v2b2_p32 must be a bool")
        if not isinstance(v2b2_p32_lr, bool):
            raise TypeError("QVQ v2b2_p32_lr must be a bool")
        if sum((dual_v2, v2b4_p64, v2b2_p32, v2b2_p32_lr)) > 1:
            raise ValueError("QVQ Dual-V2, V2B4-P64, V2B2-P32, and V2B2-P32-LR are mutually exclusive")
        if dual_v2 and (vector_size != 2 or trellis_window != 16 or bank_count != 1):
            raise ValueError("QVQ Dual-V2 requires vector_size=2, trellis_window=16, and bank_count=1")
        self.dual_v2 = dual_v2
        if v2b4_p64 and (vector_size != 2 or trellis_window != 16 or bank_count != 4 or self.bits > 3.5):
            raise ValueError("QVQ V2B4-P64 requires vector_size=2, trellis_window=16, bank_count=4, and W1-W3.5")
        self.v2b4_p64 = v2b4_p64
        if v2b2_p32 and (vector_size != 2 or trellis_window != 16 or bank_count != 2 or self.bits > 3.5):
            raise ValueError("QVQ V2B2-P32 requires vector_size=2, trellis_window=16, bank_count=2, and W1-W3.5")
        self.v2b2_p32 = v2b2_p32
        if v2b2_p32_lr and (
            vector_size != 2 or trellis_window != 16 or bank_count != 2 or self.bits > 3.5
        ):
            raise ValueError(
                "QVQ V2B2-P32-LR requires vector_size=2, trellis_window=16, bank_count=2, and W1-W3.5"
            )
        self.v2b2_p32_lr = v2b2_p32_lr
        if not isinstance(input_hadamard, bool) or not isinstance(output_hadamard, bool):
            raise TypeError("QVQ transform-axis flags must be bools")
        self.input_hadamard = input_hadamard
        self.output_hadamard = output_hadamard
        if isinstance(bank_count, bool) or not isinstance(bank_count, int) or bank_count not in (1, 2, 4):
            raise ValueError("QVQ bank_count must be 1, 2, or 4")
        if bank_count == 4 and vector_size != 4 and not v2b4_p64:
            raise ValueError("QVQ bank_count=4 requires V4 or V2B4-P64")
        self.bank_count = bank_count
        if self.bank_count in (2, 4) and tensors and tensors.get("bank_ids") is None:
            raise ValueError("QVQ banked formats require serialized bank_ids selectors")
        if self.bank_count == 2 and tensors and tensors.get("bank_alt_id") is None:
            raise ValueError("QVQ V2B2-P32 requires serialized bank_alt_id metadata")
        if self.bank_count in (2, 4) and tensors and tensors["bank_ids"].device.type != "meta":
            # Dense selectors are accepted at the construction API, but the
            # checkpoint/runtime format has one canonical four-per-byte
            # representation. Normalize at the ownership boundary so a
            # module built from dense selectors round-trips into a packed
            # loader shell without a state-dict shape mismatch.
            tile_count = (
                (in_features // QVQ_V2B2_P32_LR_TILE_ROWS) * (out_features // QVQ_V2B2_P32_LR_TILE_COLS)
                if v2b2_p32_lr
                else (in_features // 16) * (out_features // 16)
            )
            selector_count = tile_count * 8 if (v2b2_p32 or v2b2_p32_lr) else tile_count * 4 if v2b4_p64 else tile_count
            tensors = dict(tensors)
            tensors["bank_ids"] = (
                pack_qvq_binary_bank_ids(unpack_qvq_binary_bank_ids(tensors["bank_ids"], selector_count))
                if v2b2_p32 or v2b2_p32_lr
                else pack_qvq_bank_ids(unpack_qvq_bank_ids(tensors["bank_ids"], selector_count))
            )
        self._bank_ids_loaded = self.bank_count == 1 or bool(tensors)
        self._dtype_cache: dict[tuple, tuple] = {}
        self._qvq_mps_bank_ids_cache: tuple[torch.Tensor, int, torch.device, torch.Tensor] | None = None
        # Dense selectors are launch metadata, not a dequantized weight cache.
        self._qvq_cuda_bank_cache: (
            tuple[torch.Tensor, int, torch.device, torch.Tensor | None, int, torch.Tensor, int] | None
        ) = None
        self._qvq_cuda_bank_cache_lock = threading.Lock()
        pgc16_levels_for_version(self.codebook_version)

        missing = {"trellis", "SU", "SV"} - set(tensors) if tensors else set()
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
            raise ValueError(f"QVQ module `{self.name}` received unexpected tensors: {sorted(unexpected)}")

        storage_dtype = dtype or torch.float16
        defaults = {
            "trellis": torch.zeros(
                (
                    (
                        (in_features // QVQ_V2B2_P32_LR_TILE_ROWS)
                        * (out_features // QVQ_V2B2_P32_LR_TILE_COLS)
                        if v2b2_p32_lr
                        else (in_features // 16) * (out_features // 16)
                    ),
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
            "bias": torch.zeros(out_features, dtype=storage_dtype) if has_bias else None,
            # A banked loader shell needs a registered placeholder so strict
            # state-dict loading recognizes the serialized selector key.
            "bank_ids": (
                torch.zeros(
                    (
                        (in_features // QVQ_V2B2_P32_LR_TILE_ROWS)
                        * (out_features // QVQ_V2B2_P32_LR_TILE_COLS)
                        if v2b2_p32_lr
                        else (in_features // 16) * (out_features // 16)
                    )
                    if v2b4_p64 or v2b2_p32 or v2b2_p32_lr
                    else ((in_features // 16) * (out_features // 16) + 3) // 4,
                    dtype=torch.uint8,
                )
                if bank_count in (2, 4)
                else None
            ),
            "bank_alt_id": torch.ones(1, dtype=torch.uint8) if v2b2_p32 or v2b2_p32_lr else None,
        }
        for buffer_name in _QVQ_BUFFER_NAMES:
            tensor = tensors.get(buffer_name, defaults[buffer_name] if register_buffers else None)
            if tensor is None:
                setattr(self, buffer_name, None)
            else:
                self.register_buffer(buffer_name, tensor)
        if tensors or register_buffers:
            self._validate_tensors()

    def __getstate__(self):
        """Exclude transient selector state from deepcopy/pickle."""
        state = super().__getstate__()
        state.pop("_qvq_cuda_bank_cache_lock", None)
        state["_qvq_cuda_bank_cache"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._qvq_cuda_bank_cache_lock = threading.Lock()
        self._qvq_cuda_bank_cache = None

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        selector_key = f"{prefix}bank_ids"
        if self.bank_count in (2, 4) and selector_key not in state_dict:
            self._bank_ids_loaded = False
        else:
            self._bank_ids_loaded = True
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _dtype_cache_clear(self) -> None:
        """Drop cached dtype conversions (call if SU/SV/bias are replaced)."""
        self._dtype_cache = {}

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
                    FORMAT.QVQ_V2B2_P32_LR,
                ),
                "a supported QVQ format including format=qvq",
            ),
        )
        for accepted, requirement in checks:
            if not accepted:
                return False, NotImplementedError(f"QVQLinear requires {requirement}.")
        format_code = args.get("format", FORMAT.QVQ)
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if format_code == FORMAT.QVQ_V2B2_P32_LR:
            if (in_features is not None and in_features % QVQ_V2B2_P32_LR_TILE_ROWS) or (
                out_features is not None and out_features % QVQ_V2B2_P32_LR_TILE_COLS
            ):
                return False, NotImplementedError(
                    "QVQ V2B2-P32-LR requires in_features divisible by 32 and out_features divisible by 8."
                )
        elif (in_features is not None and in_features % 16) or (out_features is not None and out_features % 16):
            return False, NotImplementedError(
                "QVQ non-LR formats require in_features and out_features divisible by 16."
            )
        device = args.get("device")
        dtype = args.get("dtype")
        if device == DEVICE.MPS and dtype not in (None, torch.float16):
            return False, NotImplementedError("QVQLinear MPS inference requires float16 activations.")
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
        v2b2_p32_lr: bool = False,
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
            v2b2_p32_lr=v2b2_p32_lr,
        )

    def _validate_tensors(self) -> None:
        expected_trellis = (
            (
                (self.in_features // QVQ_V2B2_P32_LR_TILE_ROWS)
                * (self.out_features // QVQ_V2B2_P32_LR_TILE_COLS)
                if self.v2b2_p32_lr
                else (self.in_features // 16) * (self.out_features // 16)
            ),
            qvq_words_per_tile(self.bits, vector_size=self.vector_size),
        )
        expected = {
            "trellis": (expected_trellis, torch.int32),
            "SU": ((self.in_features,), None),
            "SV": ((self.out_features,), None),
        }
        for name, (shape, dtype) in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"QVQ `{name}` must have shape {shape}, got {tuple(tensor.shape)}"
                )
            if dtype is not None and tensor.dtype != dtype:
                raise TypeError(f"QVQ `{name}` must use {dtype}, got {tensor.dtype}")
            if name != "trellis" and not tensor.is_floating_point():
                raise TypeError(f"QVQ `{name}` must use a floating-point dtype")
        if self.bank_ids is not None:
            if self.bank_count not in (2, 4) or (
                self.vector_size != 4
                and not self.v2b4_p64
                and not self.v2b2_p32
                and not self.v2b2_p32_lr
            ):
                raise ValueError("QVQ bank selectors require a banked format")
            tile_count = (
                (self.in_features // QVQ_V2B2_P32_LR_TILE_ROWS)
                * (self.out_features // QVQ_V2B2_P32_LR_TILE_COLS)
                if self.v2b2_p32_lr
                else (self.in_features // 16) * (self.out_features // 16)
            )
            selector_count = tile_count * 8 if (self.v2b2_p32 or self.v2b2_p32_lr) else tile_count * 4 if self.v2b4_p64 else tile_count
            packed_count = (selector_count + (7 if self.v2b2_p32 or self.v2b2_p32_lr else 3)) // (8 if self.v2b2_p32 or self.v2b2_p32_lr else 4)
            if self.bank_ids.ndim != 1 or self.bank_ids.numel() not in (
                selector_count,
                packed_count,
            ):
                raise ValueError("QVQ `bank_ids` must be dense or packed for the module tile count")
            if self.bank_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError("QVQ `bank_ids` must use an integer dtype")
            # Accelerate constructs quantized modules on the meta device before
            # streaming checkpoint tensors.  Value validation would force a
            # scalar read from a meta selector; defer it until the real payload
            # is installed, while retaining shape/dtype checks above.
            if self.bank_ids.device.type != "meta":
                if self.v2b2_p32 or self.v2b2_p32_lr:
                    unpack_qvq_binary_bank_ids(self.bank_ids, selector_count)
                else:
                    unpack_qvq_bank_ids(self.bank_ids, selector_count)
        if self.v2b2_p32 or self.v2b2_p32_lr:
            if self.bank_alt_id is None or tuple(self.bank_alt_id.shape) != (1,):
                raise ValueError("QVQ V2B2-P32 requires one bank_alt_id value")
            if self.bank_alt_id.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
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
        devices = {getattr(self, name).device for name in ("trellis", "SU", "SV")}
        if self.bank_ids is not None:
            devices.add(self.bank_ids.device)
        if self.bank_alt_id is not None:
            devices.add(self.bank_alt_id.device)
        if self.bias is not None:
            devices.add(self.bias.device)
        if len(devices) != 1:
            raise ValueError("QVQ module tensors must share one device")
        floating_tensors = (self.SU, self.SV) if self.bias is None else (self.SU, self.SV, self.bias)
        if any(tensor.device.type != "meta" and not torch.isfinite(tensor).all() for tensor in floating_tensors):
            raise ValueError("QVQ floating-point tensors must contain only finite values")

    def runtime_device(self) -> torch.device | None:
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
        with self._qvq_cuda_bank_cache_lock:
            self._qvq_cuda_bank_cache = None
        if self.trellis.device.type == "mps":
            from ...utils.qvq_mps import _prepare_qvq_mps_compander

            self._qvq_mps_compander = _prepare_qvq_mps_compander(
                self.trellis.device,
                self.codebook_version,
            )
            if self.bank_ids is not None:
                self._qvq_mps_bank_ids = self._prepare_mps_bank_ids(self.trellis.device)

    def _apply(self, fn):
        self._qvq_mps_compander = None
        self._qvq_mps_bank_ids = None
        self._qvq_mps_bank_ids_cache = None
        with self._qvq_cuda_bank_cache_lock:
            self._qvq_cuda_bank_cache = None
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

        tile_count = (
            (self.in_features // QVQ_V2B2_P32_LR_TILE_ROWS)
            * (self.out_features // QVQ_V2B2_P32_LR_TILE_COLS)
            if self.v2b2_p32_lr
            else (self.in_features // 16) * (self.out_features // 16)
        )
        selector_count = (
            tile_count * 8
            if self.v2b2_p32 or self.v2b2_p32_lr
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
                pack_qvq_binary_bank_ids(unpack_qvq_binary_bank_ids(snapshot, selector_count))
                if self.v2b2_p32 or self.v2b2_p32_lr
                else pack_qvq_bank_ids(unpack_qvq_bank_ids(snapshot, selector_count))
            ).to(device=device)
            if source is self.bank_ids and source_version == source._version:
                self._qvq_mps_bank_ids_cache = (source, source_version, device, packed)
                self._qvq_mps_bank_ids = packed
                return packed
        raise RuntimeError("QVQ `bank_ids` changed concurrently while preparing MPS inference selectors")

    def get_inner_weight_tensor(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Materialize the dense inner weight, in FP32 unless explicitly requested otherwise."""

        return reconstruct_qvq_inner_weight(
            self.trellis,
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
            v2b2_p32_lr=self.v2b2_p32_lr,
            bank_alt_id=self.bank_alt_id,
        ).to(dtype=dtype)

    def _reference_inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable training reference; its graph may retain the dense weight.

        This legacy path intentionally follows ``x.dtype`` and is not the accuracy
        oracle. Use :func:`qvq_dense_oracle_forward` for inference comparisons.
        """

        inner = self.get_inner_weight_tensor(dtype=x.dtype)
        return x @ inner

    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bank_count in (2, 4) and (
            not self._bank_ids_loaded or self.bank_ids is None or self.bank_ids.device.type == "meta"
        ):
            # Accelerate's direct tensor loader can install buffers without
            # invoking ``_load_from_state_dict`` on the meta shell.  Once a
            # concrete selector arrives, validate it exactly once and mark the
            # module ready; a missing/meta selector still fails closed.
            if self.bank_ids is None or self.bank_ids.device.type == "meta":
                raise RuntimeError("QVQ banked module cannot run before bank_ids selectors are loaded")
            tile_count = (
                (self.in_features // QVQ_V2B2_P32_LR_TILE_ROWS)
                * (self.out_features // QVQ_V2B2_P32_LR_TILE_COLS)
                if self.v2b2_p32_lr
                else (self.in_features // 16) * (self.out_features // 16)
            )
            selector_count = (
                tile_count * 8
                if self.v2b2_p32 or self.v2b2_p32_lr
                else tile_count * 4
                if self.v2b4_p64
                else tile_count
            )
            if self.v2b2_p32 or self.v2b2_p32_lr:
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
                or self.v2b2_p32_lr
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
            # path; BF16 reaches native GEMV below. LR32 uses a dedicated CUDA
            # K32 x N8 decoder selected by qvq_cuda_gemv.
            if (
                self.trellis_window != 16
                or self.dual_v2
                or x.dtype == torch.float32
            ):
                return self._reference_inner_forward(x)

            cuda_bank_ids = None
            cuda_bank_alt_id = 0
            if self.bank_ids is not None:
                with self._qvq_cuda_bank_cache_lock:
                    source_object = self.bank_ids
                    alt_source_object = self.bank_alt_id if self.v2b2_p32 or self.v2b2_p32_lr else None
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
                        and
                        cached is not None
                        and cached[0] is source_object
                        and cached[1] == current_version
                        and cached[2] == x.device
                        and cached[3] is alt_source_object
                        and cached[4] == (-1 if alt_source_object is None else alt_source_object._version)
                        and self.bank_ids is source_object
                        and source_object._version == current_version
                    ):
                        cuda_bank_ids = cached[5]
                        cuda_bank_alt_id = cached[6]
                        cached = None
                    if cuda_bank_ids is None:
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
                            raise RuntimeError("QVQ CUDA bank selector mutated during snapshot")
                        tile_count = (
                            (self.in_features // QVQ_V2B2_P32_LR_TILE_ROWS)
                            * (self.out_features // QVQ_V2B2_P32_LR_TILE_COLS)
                            if self.v2b2_p32_lr
                            else (self.in_features // 16) * (self.out_features // 16)
                        )
                        if self.v2b2_p32 or self.v2b2_p32_lr:
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
                            cuda_bank_ids = unpack_qvq_bank_ids(source, tile_count).to(device=x.device)
                        alt_version = -1
                        if alt_source_object is not None:
                            alt_version = alt_source_object._version
                            cuda_bank_alt_id = int(alt_source_object.detach().item())
                            if (
                                alt_source_object is not self.bank_alt_id
                                or alt_version != alt_source_object._version
                                or not 1 <= cuda_bank_alt_id <= 3
                            ):
                                raise RuntimeError("QVQ CUDA alternative-bank metadata changed during snapshot")
                        if self.bank_ids is not source_object:
                            raise RuntimeError("QVQ CUDA bank selector replaced during snapshot")
                        self._qvq_cuda_bank_cache = (
                            source_object,
                            source_version,
                            x.device,
                            alt_source_object,
                            alt_version,
                            cuda_bank_ids,
                            cuda_bank_alt_id,
                        )
            if (
                not qvq_cuda_device_supported(x.device)
            ):
                return self._reference_inner_forward(x)

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
                v2b2_p32_lr=self.v2b2_p32_lr,
                bank_alt_id=cuda_bank_alt_id,
            )
        if x.device.type == "cpu":
            from ...utils.qvq_cpu import qvq_cpu_gemv, qvq_cpu_supported

            if (
                self.trellis_window != 16
                or self.vector_size != 2
                or self.dual_v2
                or self.v2b2_p32_lr
                or not qvq_cpu_supported()
            ):
                return self._reference_inner_forward(x)
            bank_alt_id = 0
            if self.bank_ids is not None and (self.v2b2_p32 or self.v2b2_p32_lr):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"QVQ expected input width {self.in_features}, got {x.shape[-1]}"
            )
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.out_features))
        input_dtype = x.dtype
        compute_dtype = _qvq_compute_dtype(input_dtype, x.device.type)
        x_2d = x.reshape(-1, self.in_features).to(compute_dtype)

        # The CUDA inner kernel accumulates and returns FP32. Preserve that
        # range through the output Hadamard/SV epilogue, then round only the
        # completed linear result to the model dtype. This removes the
        # factorization-only FP16 overflow without host synchronization or a
        # duplicate BF16 path during CUDA-graph capture.
        output = self._forward_compute_dtype(x_2d, compute_dtype)
        if input_dtype == torch.bfloat16 and x.device.type == "cuda":
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
        elif input_dtype == torch.float16 and x.device.type == "cuda" and (
            self.in_features < _FP16_STABLE_HADAMARD_MIN_WIDTH
            or self.in_features > _QVQ_HADAMARD_MAX_WIDTH
            or self.in_features & (self.in_features - 1)
            or self.out_features > _QVQ_HADAMARD_MAX_WIDTH
            or self.out_features & (self.out_features - 1)
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

    def _forward_compute_dtype(self, x_2d: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        if self.training:
            # Keep the differentiable Python butterfly path for the reference
            # forward used in training.
            transformed_input = x_2d * self.SU.to(compute_dtype)
            if self.input_hadamard:
                input_transform = (
                    matmul_hadU_stable
                    if compute_dtype == torch.float16
                    and (
                        transformed_input.device.type == "mps"
                        or transformed_input.shape[-1] >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                    )
                    else matmul_hadU
                )
                transformed = input_transform(transformed_input)
            else:
                transformed = transformed_input
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
            if self.input_hadamard:
                transformed = _qvq_hadamard_fused(
                    x_2d,
                    pre_scale=self._cached_cast("SU", compute_dtype),
                    scale_mode=(
                        2
                        if compute_dtype == torch.float16 and self.in_features >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                        else 1
                    ),
                )
            else:
                transformed = x_2d * self._cached_cast("SU", compute_dtype)
            output = self._inner_forward(transformed)
            output_dtype = output.dtype
            if self.output_hadamard:
                output = _qvq_hadamard_fused(
                    output,
                    post_scale=self._cached_cast("SV", compute_dtype, output_dtype),
                    bias=self._cached_cast("bias", compute_dtype, output_dtype),
                    scale_mode=(
                        3
                        if output_dtype == torch.float32
                        and self.out_features >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                        else 4 if output_dtype == torch.float32 else 0
                    ),
                )
            else:
                output = output * self._cached_cast("SV", compute_dtype, output_dtype)
                cached_bias = self._cached_cast("bias", compute_dtype, output_dtype)
                if cached_bias is not None:
                    output = output + cached_bias
        return output


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
        raise RuntimeError("qvq_dense_oracle_forward does not accept input that requires gradients")
    if x.shape[-1] != layer.in_features:
        raise ValueError(
            f"QVQ oracle input width must be {layer.in_features}, got {x.shape[-1]}"
        )

    compute_device = torch.device(device)
    inner = None
    try:
        with torch.inference_mode():
            inner = reconstruct_qvq_inner_weight(
                layer.trellis.to(device=compute_device),
                bits=layer.bits,
                vector_size=layer.vector_size,
                trellis_window=layer.trellis_window,
                in_features=layer.in_features,
                out_features=layer.out_features,
                codebook_version=layer.codebook_version,
                bank_ids=None if layer.bank_ids is None else layer.bank_ids.to(device=compute_device),
                dual_v2=layer.dual_v2,
                v2b4_p64=layer.v2b4_p64,
                v2b2_p32=layer.v2b2_p32,
                v2b2_p32_lr=layer.v2b2_p32_lr,
                bank_alt_id=(
                    None if layer.bank_alt_id is None else layer.bank_alt_id.to(device=compute_device)
                ),
            ).to(dtype=torch.float32)
            x_2d = x.to(device=compute_device, dtype=torch.float32).reshape(-1, layer.in_features)
            transformed = x_2d * layer.SU.to(device=compute_device, dtype=torch.float32)
            if layer.input_hadamard:
                transformed = matmul_hadU(transformed)
            output = transformed @ inner
            if layer.output_hadamard:
                output = matmul_hadU(output)
            output = output * layer.SV.to(device=compute_device, dtype=torch.float32)
            if layer.bias is not None:
                output = output + layer.bias.to(device=compute_device, dtype=torch.float32)
            return output.reshape(*x.shape[:-1], layer.out_features).detach()
    finally:
        del inner


def qvq_local_ring_dense_oracle_forward(
    layer: QVQLinear,
    x: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Run the Torch FP32 oracle for a ``qvq_v2b2_p32_lr`` layer."""

    if not isinstance(layer, QVQLinear) or not layer.v2b2_p32_lr:
        raise ValueError("qvq_local_ring_dense_oracle_forward requires a QVQ V2B2-P32-LR layer")
    return qvq_dense_oracle_forward(layer, x, device=device)


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

    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._reference_inner_forward(x)


__all__ = [
    "QVQLinear",
    "QVQReferenceLinear",
    "qvq_dense_oracle_forward",
    "qvq_local_ring_dense_oracle_forward",
]
