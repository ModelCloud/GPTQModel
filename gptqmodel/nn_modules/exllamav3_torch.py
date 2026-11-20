# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Clean-room EXL3 torch reference kernel derived from the public EXL3 tensor
# format and documented runtime layout.

from __future__ import annotations

import math
import struct
from functools import lru_cache
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..exllamav3.q4nx import decode_q4nx_states, unpack_q4nx_states
from ..exllamav3.r4nx import decode_r4nx_states, unpack_r4nx_states
from ..exllamav3.trellis import EXL3_CODEBOOK_MARKERS, trellis_rate
from ..models._const import DEVICE
from ..utils.cpu_features import cpu_bf16x3_scope, has_avx512_bf16, has_cpu_bf16x3_precision
from ..utils.logger import setup_logger
from ..utils.torch import TORCH_HAS_COMPILE, torch_compile
from .compiled_state import compiled_module_getstate
from .exllamav3 import _EXL3_BUFFER_NAMES, _EXL3_REQUIRED_BUFFERS, _torch_dtype


log = setup_logger()

_EXL3_3INST_MULT = 89226354
_EXL3_3INST_ADD = 64248484
_EXL3_MCG_MULT = 0xCBAC1FED
_EXL3_MUL1_MULT = 0x83DCD12D
_EXL3_MUL1_ACC = 0x6400
_EXL3_LOP3_MASK = 0x8FFF8FFF
_EXL3_LOP3_BIAS = 0x3B603B60
_CPU_BF16X3_MIN_WEIGHT_ELEMENTS = 1 << 19
_CPU_BF16X3_MIN_MACS = 1 << 23
# Bound each transient decode to 512 output columns without serializing many small GEMMs.
_EXL3_OUTPUT_TILES_PER_CHUNK = 32


@lru_cache(maxsize=1)
def _cpu_bf16x3_capable() -> bool:
    """Return whether the host itself can run oneDNN BF16x3 FP32 matmul."""

    return has_avx512_bf16() and has_cpu_bf16x3_precision()


def _cpu_bf16x3_support() -> bool:
    """Return whether the active kernel policy enables BF16x3 on this host.

    The policy read stays outside the cache so toggling
    ``KernelConfig.cpu_bf16x3_matmul`` between runs is observed by the next
    forward. Only the hardware capability probe is cached.
    """

    from ..utils.kernel import get_kernel_config

    if not get_kernel_config().cpu_bf16x3_matmul:
        return False
    return _cpu_bf16x3_capable()


# Keep the documented re-arm knob on the capability probe. Callers (and tests)
# that used to clear the old cached predicate now clear only the hardware
# capability, which is exactly what the name promises.
_cpu_bf16x3_support.cache_clear = _cpu_bf16x3_capable.cache_clear


def _cpu_bf16x3_should_use(rows: int, in_features: int, out_features: int) -> bool:
    """Return whether the BF16x3 GEMM is large enough to amortize dispatch."""

    if rows <= 0 or in_features <= 0 or out_features <= 0:
        return False
    weight_elements = in_features * out_features
    return (
        weight_elements >= _CPU_BF16X3_MIN_WEIGHT_ELEMENTS
        or rows * weight_elements >= _CPU_BF16X3_MIN_MACS
    )


def _tensor_core_perm_values() -> list[int]:
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8 + 0] = r0 * 16 + c0
        perm[t * 8 + 1] = r1 * 16 + c0
        perm[t * 8 + 2] = r2 * 16 + c0
        perm[t * 8 + 3] = r3 * 16 + c0
        perm[t * 8 + 4] = r0 * 16 + c1
        perm[t * 8 + 5] = r1 * 16 + c1
        perm[t * 8 + 6] = r2 * 16 + c1
        perm[t * 8 + 7] = r3 * 16 + c1
    return perm


def _inverse_perm_values(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for index, value in enumerate(perm):
        inv[value] = index
    return inv


def _half_scalar_from_bits(bits: int) -> float:
    # Convert a uint16 bit pattern to its IEEE-754 binary16 (float16) value
    # without allocating a torch tensor. The previous implementation used
    # ``torch.tensor([bits], dtype=torch.uint16).view(torch.float16).item()``
    # at module import, which fails under a meta-device preload context with
    # ``RuntimeError: Tensor.item() cannot be called on meta tensors``. The
    # transformers AWQ pathway (``replace_with_awq_linear``) imports
    # ``gptqmodel.quantization`` inside ``with torch.device('meta'):``, which
    # triggered the failure for any GPTQ load on transformers >= 5.6.
    #
    # ``struct`` format ``<e`` is IEEE-754 binary16 little-endian and matches
    # ``torch.float16``'s byte layout on all supported platforms (x86_64 /
    # aarch64). The conversion is bit-equivalent to the tensor-based path.
    packed = struct.pack("<H", int(bits) & 0xFFFF)
    return float(struct.unpack("<e", packed)[0])


# ExLlamaV3's torch fallback decodes packed codebook values with shift
# operations. Keep those shifts behind helpers so Ascend 910B/CANN 9.1 beta can
# use arithmetic equivalents for operators that torch-npu does not expose as
# native device kernels, while CPU/CUDA/XPU keep the standard bitwise ops.
def _torch_shift_factor(shifts: int | torch.Tensor, device: torch.device) -> int | torch.Tensor:
    if torch.is_tensor(shifts):
        # Tensor shifts must be materialized on the target device; otherwise the
        # NPU arithmetic fallback would introduce cross-device operands.
        shifts_i64 = shifts.to(device=device, dtype=torch.int64)
        return torch.pow(torch.full_like(shifts_i64, 2), shifts_i64)
    return 1 << int(shifts)


def _torch_right_shift(values: torch.Tensor, shifts: int | torch.Tensor) -> torch.Tensor:
    if values.device.type != "npu":
        return torch.bitwise_right_shift(values, shifts)

    # CANN 9.1 beta on Ascend 910B may not provide bitwise_right_shift kernels
    # for these tensor paths. floor_divide by powers of two preserves arithmetic
    # right-shift behavior for signed packed int tensors and stays on-device.
    shifted = torch.floor_divide(values.to(torch.int64), _torch_shift_factor(shifts, values.device))
    return shifted.to(values.dtype)


def _torch_left_shift(values: torch.Tensor, shifts: int | torch.Tensor) -> torch.Tensor:
    if values.device.type != "npu":
        return torch.bitwise_left_shift(values, shifts)

    # Mirror left shift as multiplication by powers of two on NPU to avoid
    # missing-kernel or CPU-fallback paths in torch-npu.
    shifted = values.to(torch.int64) * _torch_shift_factor(shifts, values.device)
    return shifted.to(values.dtype)


_EXL3_MUL1_INV = _half_scalar_from_bits(0x1EEE)
_EXL3_MUL1_BIAS = _half_scalar_from_bits(0xC931)


# These factories feed process-global ``lru_cache`` entries. A plain
# ``torch.inference_mode()`` caller (``generate()``, an eval loop, server
# warmup) would otherwise create an *inference* tensor and the cache would hand
# that kind to every later caller. Matmul then raises "Inference tensors cannot
# be saved for backward" for any grad-mode forward of any model in the process.
# Build every constant with inference mode disabled (and no autograd tracking):
# the values never need grad, and an ordinary tensor is valid in every mode.
# A clone inside inference mode is still an inference tensor, so it cannot fix
# this; the construction context itself has to be grad-safe.
@lru_cache(maxsize=None)
@torch.inference_mode(False)
@torch.no_grad()
def _tensor_core_perm(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.tensor(_tensor_core_perm_values(), dtype=torch.long, device=device)


@lru_cache(maxsize=None)
@torch.inference_mode(False)
@torch.no_grad()
def _tensor_core_perm_i(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.tensor(_inverse_perm_values(_tensor_core_perm_values()), dtype=torch.long, device=device)


@lru_cache(maxsize=None)
@torch.inference_mode(False)
@torch.no_grad()
def _hadamard_128(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    had = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    while had.shape[0] < 128:
        had = torch.cat(
            (
                torch.cat((had, had), dim=1),
                torch.cat((had, -had), dim=1),
            ),
            dim=0,
        )
    had *= 1.0 / math.sqrt(128.0)
    return had.contiguous()


@lru_cache(maxsize=None)
@torch.inference_mode(False)
@torch.no_grad()
def _codebook_lut(
    codebook: str,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    values = torch.arange(1 << 16, dtype=torch.int64, device=device)

    if codebook == "3inst":
        raw = (values * _EXL3_3INST_MULT + _EXL3_3INST_ADD) & 0xFFFFFFFF
        raw = _EXL3_LOP3_BIAS ^ (raw & _EXL3_LOP3_MASK)
        halves = torch.stack(
            (
                (raw & 0xFFFF).to(torch.uint16),
                (_torch_right_shift(raw, 16) & 0xFFFF).to(torch.uint16),
            ),
            dim=-1,
        ).contiguous()
        floats = halves.view(torch.float16).to(torch.float32)
        # Native __hadd rounds to half before widening the decoded value.
        return (floats[..., 0] + floats[..., 1]).half().float().contiguous()

    if codebook == "mcg":
        raw = (values * _EXL3_MCG_MULT) & 0xFFFFFFFF
        raw = _EXL3_LOP3_BIAS ^ (raw & _EXL3_LOP3_MASK)
        halves = torch.stack(
            (
                (raw & 0xFFFF).to(torch.uint16),
                (_torch_right_shift(raw, 16) & 0xFFFF).to(torch.uint16),
            ),
            dim=-1,
        ).contiguous()
        floats = halves.view(torch.float16).to(torch.float32)
        # Native __hadd rounds to half before widening the decoded value.
        return (floats[..., 0] + floats[..., 1]).half().float().contiguous()

    if codebook == "mul1":
        raw = (values * _EXL3_MUL1_MULT) & 0xFFFFFFFF
        byte_sum = (
            (raw & 0xFF)
            + (_torch_right_shift(raw, 8) & 0xFF)
            + (_torch_right_shift(raw, 16) & 0xFF)
            + (_torch_right_shift(raw, 24) & 0xFF)
        )
        accum = (byte_sum + _EXL3_MUL1_ACC).to(torch.uint16).contiguous()
        floats = accum.view(torch.float16).to(torch.float32)
        # These half operands are exact in FP32; native __hfma rounds once,
        # after the add, not between the multiply and add.
        return (floats * _EXL3_MUL1_INV + _EXL3_MUL1_BIAS).half().float().contiguous()

    raise ValueError(f"Unsupported EXL3 codebook: {codebook}")


def _hadamard_for(x: torch.Tensor) -> torch.Tensor:
    """Return the cached 128x128 Hadamard matrix in ``x``'s device and dtype.

    ``_hadamard_128`` is cached in FP32, but ``x`` is FP16/BF16 on an
    accelerator (``_forward_eager`` casts to FP16 off CPU). ``torch.matmul``
    requires matching dtypes, so an FP32 matrix against FP16 activations raised
    on ROCm. Cast to the activation dtype (cached per dtype) so the fused
    EQUALITY backend executes instead of failing at the first forward.
    """
    had = _hadamard_128(x.device.type, x.device.index)
    if had.dtype != x.dtype:
        had = _hadamard_128_as(x.device.type, x.device.index, x.dtype)
    return had


@lru_cache(maxsize=None)
@torch.inference_mode(False)
@torch.no_grad()
def _hadamard_128_as(device_type: str, device_index: int | None, dtype: torch.dtype) -> torch.Tensor:
    return _hadamard_128(device_type, device_index).to(dtype=dtype)


def _apply_hadamard_left(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] % 128 != 0:
        raise ValueError(f"EXL3 expects in_features to be divisible by 128, got {x.shape[0]}.")
    had = _hadamard_for(x)
    return (had @ x.view(-1, 128, x.shape[1])).view_as(x)


def _apply_hadamard_right(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] % 128 != 0:
        raise ValueError(f"EXL3 expects out_features to be divisible by 128, got {x.shape[1]}.")
    had = _hadamard_for(x)
    return (x.view(x.shape[0], -1, 128) @ had).view_as(x)


class ExllamaV3TorchLinear(nn.Module):
    QUANT_TYPE = "exl3"
    CHECKPOINT_REQUIRED_BUFFERS = _EXL3_REQUIRED_BUFFERS
    # Every floating EXL3 buffer (the fp16 `suh`/`svh` sign vectors and the
    # bias) loads at its checkpoint dtype; the Hadamard kernels require fp16.
    MODEL_DTYPE_BUFFERS = ()
    SUPPORTS_SHARDS = True
    SUPPORTS_DEVICES = [DEVICE.ALL]

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        name: str,
        tensor_storage: Optional[Dict[str, Any]] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        out_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.out_dtype = out_dtype
        self.tensor_storage = tensor_storage or {}
        self.optimized = False
        self._compiled_forward = None

        self.weight = torch.zeros((1,), dtype=torch.float16, device="meta")

        if tensors is not None:
            for buffer_name in _EXL3_BUFFER_NAMES:
                tensor = tensors.get(buffer_name)
                if tensor is None:
                    setattr(self, buffer_name, None)
                else:
                    self.register_buffer(buffer_name, tensor)
            return

        stored_tensors = (self.tensor_storage or {}).get("stored_tensors", {})
        for buffer_name in _EXL3_BUFFER_NAMES:
            metadata = stored_tensors.get(f"{name}.{buffer_name}")
            if metadata is None:
                setattr(self, buffer_name, None)
                continue

            shape = tuple(metadata["shape"])
            dtype = _torch_dtype(metadata["torch_dtype"])
            self.register_buffer(buffer_name, torch.empty(shape, dtype=dtype, device="meta"))

    @classmethod
    def from_tensors(
        cls,
        *,
        in_features: int,
        out_features: int,
        name: str,
        tensors: Dict[str, torch.Tensor],
    ) -> "ExllamaV3TorchLinear":
        return cls(
            in_features=in_features,
            out_features=out_features,
            name=name,
            tensors=tensors,
        )

    def post_init(self) -> None:
        """EXL3 has no runtime cache to initialize."""

    def _codebook_name(self) -> str:
        for name in EXL3_CODEBOOK_MARKERS:
            if getattr(self, name, None) is not None:
                return name
        return "3inst"

    def _bits_per_weight(self) -> int | float:
        trellis = getattr(self, "trellis", None)
        if trellis is None:
            raise RuntimeError(f"EXL3 module `{self.name}` is missing `trellis`.")
        try:
            return trellis_rate(
                trellis.shape[-1],
                getattr(self, "mcg", None) is not None,
                getattr(self, "mul1", None) is not None,
                q4nx=getattr(self, "q4nx", None) is not None,
                r4nx=getattr(self, "r4nx", None) is not None,
                max_integer_rate=16,
            )
        except ValueError as exc:
            raise ValueError(f"EXL3 module `{self.name}` has invalid trellis geometry: {exc}") from exc

    def _runtime_weight_dtype(self) -> torch.dtype:
        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.device.type == "cpu":
            return torch.float32
        return torch.float16

    def _stored_features(self) -> tuple[int, int]:
        """Validate full Hadamard storage against the logical linear shape."""
        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.ndim != 3 or trellis.dtype != torch.int16:
            raise ValueError(f"EXL3 module `{self.name}` has invalid trellis geometry or dtype")
        self._bits_per_weight()
        inputs, outputs = int(trellis.shape[0]) * 16, int(trellis.shape[1]) * 16
        padded_outputs = ((self.out_features + 127) // 128) * 128
        if (self.in_features <= 0 or self.out_features <= 0
                or inputs != self.in_features or inputs % 128 != 0
                or outputs != padded_outputs):
            raise ValueError(f"EXL3 module `{self.name}` storage differs from logical dimensions or 128-channel padding")
        for suffix, expected in (("suh", inputs), ("svh", outputs)):
            scale = getattr(self, suffix, None)
            if scale is None or tuple(scale.shape) != (expected,):
                raise ValueError(f"EXL3 module `{self.name}` has invalid {suffix} geometry")
        bias = getattr(self, "bias", None)
        if bias is not None and tuple(bias.shape) != (self.out_features,):
            raise ValueError(f"EXL3 module `{self.name}` has invalid logical bias geometry")
        return inputs, outputs

    def _unpack_indices(
        self,
        tile_n_start: int = 0,
        tile_n_end: Optional[int] = None,
    ) -> torch.Tensor:
        trellis = getattr(self, "trellis", None)
        if trellis is None:
            raise RuntimeError(f"EXL3 module `{self.name}` is missing `trellis`.")
        if trellis.device.type == "meta":
            raise RuntimeError(f"EXL3 module `{self.name}` has not been materialized from checkpoint tensors yet.")

        self._stored_features()
        if tile_n_end is None:
            tile_n_end = int(trellis.shape[1])
        if not 0 <= tile_n_start < tile_n_end <= int(trellis.shape[1]):
            raise ValueError(
                f"Invalid EXL3 output tile range [{tile_n_start}, {tile_n_end}) "
                f"for {trellis.shape[1]} tiles."
            )

        packed = trellis[:, tile_n_start:tile_n_end]
        if getattr(self, "r4nx", None) is not None:
            return unpack_r4nx_states(packed)
        if getattr(self, "q4nx", None) is not None:
            return unpack_q4nx_states(packed)

        bits = self._bits_per_weight()
        words = (packed.to(torch.int32) & 0xFFFF).contiguous()
        words = words.view(*words.shape[:-1], -1, 2).flip(-1).reshape(*words.shape)
        if bits != int(bits):
            # Each alternating pair consumes 2*floor(bits)+1 bits. Gather
            # the 16-bit state ending at each position, wrapping at the tile
            # boundary. As for integer tiles, the preceding flip puts each
            # packed uint32 into bitstream-order uint16 words.
            positions = torch.arange(1, 257, device=words.device, dtype=torch.long)
            offsets = (positions * int(bits) + positions // 2 - 16) % (16 * words.shape[-1])
            word_idx = offsets // 16
            joined = _torch_left_shift(words[..., word_idx], 16)
            joined = joined | words[..., (word_idx + 1) % words.shape[-1]]
            return (_torch_right_shift(joined, 16 - offsets % 16) & 0xFFFF).to(torch.long)

        bits = int(bits)
        mask = (1 << bits) - 1
        words = words.view(*words.shape[:-1], 16, bits)

        # Unpack all 16 positions in one pass. Each position selects one of the
        # `bits` words and, for widths that do not divide 16, may straddle two
        # words. The result is laid out tile-major so it is already the
        # [..., 256] symbol stream consumed by the state recurrence below.
        positions = torch.arange(16, device=words.device, dtype=torch.long)
        bit_offsets = positions * bits
        word_idx = bit_offsets // 16
        bit_in_word = bit_offsets % 16
        direct_shift = torch.where(
            bit_in_word + bits <= 16,
            16 - bit_in_word - bits,
            torch.zeros_like(bit_in_word),
        )
        value = (_torch_right_shift(words[..., word_idx], direct_shift) & mask).to(torch.long)
        if 16 % bits:
            bits_first = 16 - bit_in_word
            bits_second = (bits - bits_first).clamp_min(0)
            high = _torch_left_shift(
                words[..., word_idx] & ((1 << bits_first) - 1),
                bits_second,
            )
            low = _torch_right_shift(
                words[..., (word_idx + 1).clamp_max(bits - 1)],
                16 - bits_second,
            )
            value = torch.where(bit_in_word + bits > 16, (high | low) & mask, value)
        symbols = value.reshape(*words.shape[:-2], 256)

        warmup = (16 + bits - 1) // bits - 1
        if warmup == 0:
            return symbols & 0xFFFF

        # The EXL3 stream is tail-biting: encoded[i] is the low 16 bits of the
        # sliding window ending at i, with the previous `warmup` symbols taken
        # from the end of the 256-symbol block. Build all windows at once
        # instead of replaying the recurrence 256 times in Python.
        padded = torch.cat((symbols[..., -warmup:], symbols), dim=-1)
        windows = padded.unfold(-1, warmup + 1, 1)
        weights = torch.tensor(
            [1 << (bits * (warmup - index)) for index in range(warmup + 1)],
            dtype=torch.long,
            device=symbols.device,
        )
        return (windows * weights).sum(dim=-1) & 0xFFFF

    def _decode_inner_weight(
        self,
        tile_n_start: int = 0,
        tile_n_end: Optional[int] = None,
    ) -> torch.Tensor:
        trellis = getattr(self, "trellis", None)
        if trellis is None:
            raise RuntimeError(f"EXL3 module `{self.name}` is missing `trellis`.")
        if trellis.device.type == "meta":
            raise RuntimeError(f"EXL3 module `{self.name}` has not been materialized from checkpoint tensors yet.")

        encoded = self._unpack_indices(tile_n_start, tile_n_end)
        codebook = self._codebook_name()
        if codebook == "r4nx":
            # R4NX decoding keeps the four levels as a trailing dimension;
            # flatten so the 256-entry tile permutation applies per weight.
            decoded = decode_r4nx_states(encoded).flatten(-2)
        elif codebook == "q4nx":
            decoded = decode_q4nx_states(encoded)
        else:
            lut = _codebook_lut(codebook, trellis.device.type, trellis.device.index)
            decoded = lut[encoded]

        perm_i = _tensor_core_perm_i(trellis.device.type, trellis.device.index)
        decoded = decoded[..., perm_i]
        tiles_k, tiles_n = decoded.shape[:2]
        inner = decoded.view(tiles_k, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(
            tiles_k * 16, tiles_n * 16
        )
        return inner.contiguous().to(torch.float32)

    def get_inner_weight_tensor(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        inner = self._decode_inner_weight()
        target_dtype = dtype or self._runtime_weight_dtype()
        if inner.dtype == target_dtype:
            return inner
        return inner.to(dtype=target_dtype)

    def _materialize_weight_fp32(self) -> torch.Tensor:
        inner = self._decode_inner_weight()
        inner = _apply_hadamard_left(inner)
        inner *= getattr(self, "suh").to(dtype=torch.float32).unsqueeze(1)
        inner = _apply_hadamard_right(inner)
        inner *= getattr(self, "svh").to(dtype=torch.float32).unsqueeze(0)
        return inner[:, :self.out_features].contiguous()

    def get_weight_tensor(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        weight = self._materialize_weight_fp32()
        target_dtype = dtype or self._runtime_weight_dtype()
        if weight.dtype == target_dtype:
            return weight
        return weight.to(dtype=target_dtype)

    def get_bias_tensor(self) -> torch.Tensor | None:
        return getattr(self, "bias", None)

    def _forward_chunked(self, x_2d: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        _, stored_outputs = self._stored_features()
        if x_2d.shape[0] == 0:
            return x_2d.new_empty((0, self.out_features))

        suh = getattr(self, "suh").to(dtype=compute_dtype)
        svh = getattr(self, "svh").to(dtype=compute_dtype)
        x_work = _apply_hadamard_right(x_2d * suh)
        y = torch.empty(
            (x_2d.shape[0], self.out_features),
            dtype=compute_dtype,
            device=x_2d.device,
        )

        tiles_n = stored_outputs // 16
        for tile_n_start in range(0, tiles_n, _EXL3_OUTPUT_TILES_PER_CHUNK):
            tile_n_end = min(tile_n_start + _EXL3_OUTPUT_TILES_PER_CHUNK, tiles_n)
            out_start = tile_n_start * 16
            out_end = tile_n_end * 16
            chunk = self._decode_inner_weight(tile_n_start, tile_n_end).to(dtype=compute_dtype)
            y_chunk = _apply_hadamard_right(x_work @ chunk)
            y_chunk *= svh[out_start:out_end]
            logical_end = min(out_end, self.out_features)
            y[:, out_start:logical_end] = y_chunk[:, :logical_end - out_start]
        return y

    def _forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        compute_dtype = torch.float32
        x_2d = x.reshape(-1, self.in_features).to(compute_dtype)
        if (
            x.device.type == "cpu"
            and _cpu_bf16x3_support()
            and _cpu_bf16x3_should_use(
                x_2d.shape[0],
                self.in_features,
                self.out_features,
            )
        ):
            with cpu_bf16x3_scope():
                y = self._forward_chunked(x_2d, compute_dtype)
        else:
            y = self._forward_chunked(x_2d, compute_dtype)
        bias = getattr(self, "bias", None)
        if bias is not None:
            y = y + bias.to(dtype=compute_dtype)
        y = y.view(*x.shape[:-1], self.out_features)
        return y.to(input_dtype)

    def optimize(
        self,
        backend: str = "inductor",
        mode: Optional[str] = None,
        fullgraph: bool = False,
    ) -> "ExllamaV3TorchLinear":
        """Compile the CPU decode path after the caller explicitly opts in."""

        if self.optimized:
            return self
        if not TORCH_HAS_COMPILE:
            return self

        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.device.type != "cpu":
            return self

        compiled = torch_compile(
            self._forward_eager,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
        )
        if compiled is self._forward_eager:
            return self

        self._compiled_forward = compiled
        self.optimized = True
        log.info.once("Optimize: `ExllamaV3TorchLinear` CPU compilation triggered.")
        return self

    def _invalidate_compiled(self) -> None:
        self._compiled_forward = None
        self.optimized = False

    def __getstate__(self):
        """Drop the compiled forward so `torch.save(model)` keeps working."""

        return compiled_module_getstate(self)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._invalidate_compiled()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._invalidate_compiled()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compiled = self._compiled_forward
        if compiled is None:
            return self._forward_eager(x)

        try:
            return compiled(x)
        except Exception as exc:
            log.warn.once(
                f"EXL3 compiled CPU forward failed; falling back to eager: {exc}"
            )
            self._invalidate_compiled()
            return self._forward_eager(x)

    def _multiplier_value(self, name: str) -> Optional[int]:
        tensor = getattr(self, name, None)
        if tensor is None:
            return None
        return int(tensor.view(torch.uint32).item())

    def tensor_storage_entry(self) -> Dict[str, Any]:
        stored_tensors: Dict[str, Dict[str, Any]] = {}
        for buffer_name in _EXL3_BUFFER_NAMES:
            tensor = getattr(self, buffer_name, None)
            if tensor is None:
                continue
            stored_tensors[f"{self.name}.{buffer_name}"] = {
                "shape": list(tensor.shape),
                "torch_dtype": str(tensor.dtype).split(".")[-1],
            }

        entry: Dict[str, Any] = {
            "stored_tensors": stored_tensors,
            "quant_format": "exl3",
            "bits_per_weight": self._bits_per_weight(),
        }

        for name in EXL3_CODEBOOK_MARKERS:
            multiplier = self._multiplier_value(name)
            if multiplier is not None:
                entry[f"{name}_multiplier"] = multiplier

        return entry


__all__ = ["ExllamaV3TorchLinear"]
