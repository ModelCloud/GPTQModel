# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...extension import op as extension_op
from ...extension import register_fake
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from ...utils.logger import setup_logger
from . import detach_cpu_operand
from ._compile_support import COMPILE_LEVEL
from ._cpu_int4pack_layout import int4pack_decode_dtype
from ._fused_train_state import finalize_fused_train_state
from .torch_aten_kernel import TorchAtenLinear


log = setup_logger()


def _fake_gemm_int4_qzero_group_scaled(input, weight, zeros, absmax, blocksize, bias=None):
    """FakeTensor shape for the AVX512 int4 GEMM (see ``extension.register_fake``)."""

    # ``weight`` is ``[out_features, in_features // 2]`` for the GPTQ layout but
    # ``[in_features, out_features // 2]`` for the AWQ layout, so its leading
    # dimension is not the output width. ``absmax``/``scales`` is always
    # ``[groups, out_features]``; the traced reshape needs the real width.
    return input.new_empty((input.shape[0], absmax.shape[-1]))


def pack_hf_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``[out_features, in_features]`` uint4 codes for the HF CPU ABI."""

    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D, got shape={tuple(codes.shape)}")
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype}")

    out_features, in_features = codes.shape
    if out_features % 32 != 0:
        raise ValueError("out_features must be divisible by 32")
    if in_features % 2 != 0:
        raise ValueError("in_features must be even")

    groups = out_features // 32
    grouped = codes.reshape(groups, 32, in_features // 2, 2)
    low_rows = grouped[:, :16]
    high_rows = grouped[:, 16:]
    packed = (low_rows | (high_rows << 4)).permute(0, 2, 1, 3)
    return packed.reshape(out_features, in_features // 2).contiguous()


def unpack_hf_int4(
    packed: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Inverse of :func:`pack_hf_int4`.

    The kernel-facing tensor is shaped ``[out_features, in_features // 2]``,
    but its flat layout is grouped as ``[out_features // 32, in_features // 2,
    32 bytes]``. Reshape through the same grouping before splitting nibbles.
    """

    if packed.ndim != 2:
        raise ValueError(f"packed must be 2D, got shape={tuple(packed.shape)}")
    if out_features % 32 != 0:
        raise ValueError("out_features must be divisible by 32")
    if in_features % 2 != 0:
        raise ValueError("in_features must be even")
    if packed.shape != (out_features, in_features // 2):
        raise ValueError(
            "packed shape does not match the requested layer: "
            f"expected {(out_features, in_features // 2)}, got {tuple(packed.shape)}"
        )

    groups = out_features // 32
    half_in = in_features // 2
    # The flat packed layout is ``[groups, half_in, 16, 2]``: the first 16
    # rows of a group carry the low nibble and rows 16..31 the high nibble,
    # while the last axis pairs two adjacent input channels. Read those two
    # lanes as strided views and transpose only the 16 x half_in block instead
    # of walking the whole 4-D view twice (issue #4884).
    grouped = packed.reshape(groups, half_in, 16, 2)
    low = grouped[..., 0]
    high = grouped[..., 1]
    codes = torch.empty(
        (groups, 32, half_in, 2), dtype=torch.uint8, device=packed.device
    )
    codes[:, :16, :, 0] = torch.bitwise_and(low, 0x0F).transpose(1, 2)
    codes[:, :16, :, 1] = torch.bitwise_and(high, 0x0F).transpose(1, 2)
    codes[:, 16:, :, 0] = torch.bitwise_right_shift(low, 4).transpose(1, 2)
    codes[:, 16:, :, 1] = torch.bitwise_right_shift(high, 4).transpose(1, 2)
    return codes.reshape(out_features, in_features)


def avx512_checkpoint_bytes(
    packed: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Rebuild the checkpoint's packed bytes from the AVX512 runtime words.

    The result is the checkpoint byte layout ``[out_features, in_features //
    2]``: one byte holds two adjacent input-channel nibbles, and four of those
    bytes little-endian form one ``int32`` checkpoint word. Building the bytes
    directly avoids the old ``codes.t().reshape(...)`` plus eight strided
    shift/OR passes over an ``int32`` expansion (issue #4884).
    """

    if packed.ndim != 2:
        raise ValueError(f"packed must be 2D, got shape={tuple(packed.shape)}")
    if out_features % 32 != 0:
        raise ValueError("out_features must be divisible by 32")
    if in_features % 2 != 0:
        raise ValueError("in_features must be even")
    if packed.shape != (out_features, in_features // 2):
        raise ValueError(
            "packed shape does not match the requested layer: "
            f"expected {(out_features, in_features // 2)}, got {tuple(packed.shape)}"
        )

    groups = out_features // 32
    half_in = in_features // 2
    grouped = packed.reshape(groups, half_in, 16, 2)
    low = grouped[..., 0]
    high = grouped[..., 1]
    # Rows 0..15 take the low nibble, rows 16..31 the high nibble, and each
    # byte pairs the two adjacent input channels of one output row.
    low_bytes = torch.bitwise_or(
        torch.bitwise_and(low, 0x0F),
        torch.bitwise_left_shift(torch.bitwise_and(high, 0x0F), 4),
    )
    high_bytes = torch.bitwise_or(
        torch.bitwise_right_shift(low, 4),
        torch.bitwise_and(high, 0xF0),
    )
    checkpoint = torch.empty(
        (groups, 32, half_in), dtype=torch.uint8, device=packed.device
    )
    checkpoint[:, :16] = low_bytes.transpose(1, 2)
    checkpoint[:, 16:] = high_bytes.transpose(1, 2)
    return checkpoint.reshape(out_features, half_in)


class TorchCpuAvx512Linear(TorchAtenLinear):
    # The AVX512 kernel consumes the ragged final group; unlike ATen's
    # int4pack op it does not require in_features % group_size == 0.
    CPU_INT4PACK_REQUIRES_DIVISIBLE_IN_FEATURES = False

    """CPU GPTQ W4A16 kernel using AVX512-BF16."""

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH_CPU_AVX512]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 120, FORMAT.GPTQ_V2: 120}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 128, 256, 512]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    # The fused kernel has no backward path, but training mode uses the same
    # dequantized matmul as TorchAtenLinear. BACKEND.AUTO selects this kernel
    # (e.g. HF/PEFT loading), so it must stay trainable.
    SUPPORTS_TRAINING = True
    # The AVX512 kernel walks full groups and requires in_features % group_size == 0.
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [2]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = TorchAtenLinear.SUPPORTS_ADAPTERS

    # The ported HF kernel is BF16-only: its public entrypoint and SIMD core
    # both consume/produce BF16. FP16 checkpoints use the native ATen FP16
    # int4pack backend instead.
    SUPPORTS_DTYPES = [torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # Measured on an x86-64 AVX512-BF16 host, torch 2.16.0.dev20261021+cpu:
    # eager=True, graph_breaks=0, dynamo/aot_eager/inductor all reach it for the
    # BF16 fused path itself: the AVX512 extension GEMM has a registered
    # FakeTensor impl, and the RLock-guarded one-time transform inherited from
    # TorchAtenLinear is skipped while tracing. CPU-only, so no CUDA-graph
    # capture applies.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.INDUCTOR
    SUPPORTS_COMPILE_EAGER_REASON = (
        "static-cache generate() auto-compile stays off: no CPU win was measured "
        "for this kernel in this change; the INDUCTOR level is opt-in via "
        "optimize() (#4984 policy unchanged)."
    )
    SUPPORTS_GRAPH_CAPTURE = False

    QUANT_TYPE = "torch_cpu_avx512"

    # ``transform_cpu`` widens scales *and* bias to FP32 in place, so both must
    # be restored before a checkpoint load or save.
    _FUSED_TRAIN_STATE_NAMES = ("qweight", "qzeros", "scales", "bias")
    # ``unpack_hf_int4`` is the verified inverse of ``pack_hf_int4``, so the
    # checkpoint ``qweight`` is rebuilt on save instead of kept resident. The
    # remaining snapshot entries are per-column metadata, not the packed weight.
    _FUSED_TRAIN_STATE_REBUILD = "avx512_gptq"

    #: ``dequantize_weight`` already restores the checkpoint input order.
    _runtime_dense_weight_is_checkpoint_order = True

    def _runtime_dense_weight(
        self,
        activation_dtype: torch.dtype | None = None,
        *,
        use_zero_codes: bool = False,
    ):
        """Decode the live AVX512 layout without restoring the checkpoint layout.

        ``transform_cpu`` rewrites ``qzeros`` to the checkpoint's integer zero
        codes, not the ATen affine offsets, so ``use_zero_codes`` already
        describes the installed form and reads the same tensor.

        ``activation_dtype`` selects a decode dtype that holds every stored
        weight for a grad-requiring fallback. ``None`` keeps the export
        behavior and the checkpoint scale dtype.
        """
        if activation_dtype is None:
            return self.dequantize_weight()
        return self.dequantize_weight(
            dtype=int4pack_decode_dtype(self, activation_dtype)
        )

    gemm_int4_qzero_group_scaled_cpu_kernel = None
    pack_gptq_int4_kernel = None
    _fused_op_handles_bias = True

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.GPTQ_TORCH_CPU_AVX512)
        super().__init__(*args, **kwargs)
        self._cpu_transformed = False

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not has_avx512_bf16():
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = None
            err = RuntimeError(avx512_bf16_error())
            log.warning(str(err))
            return False, err

        try:
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = extension_op(
                "int4_qzero_group_scaled_cpu", "gemm_int4_qzero_group_scaled_cpu_forward"
            )
            register_fake(
                "int4_qzero_group_scaled_cpu",
                "gemm_int4_qzero_group_scaled_cpu_forward",
                _fake_gemm_int4_qzero_group_scaled,
            )
            cls.pack_gptq_int4_kernel = extension_op(
                "int4_qzero_group_scaled_cpu", "pack_gptq_int4"
            )
        except Exception as exc:
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = None
            cls.pack_gptq_int4_kernel = None
            return False, exc
        return True, None

    def _cpu_fused_available(self) -> bool:
        if self.gemm_int4_qzero_group_scaled_cpu_kernel is not None:
            return True
        ok, _ = type(self).cached_validate_once()
        return ok

    def _fused_input_dtype(self, x: torch.Tensor) -> bool:
        return x.dtype == torch.bfloat16

    def train(self, mode: bool = True):
        # forward() caches the fused-vs-dequantized dispatch decision. Reset it
        # on mode changes so an eval forward cannot pin later training forwards
        # to the no-grad fused kernel.
        if mode != self.training:
            self.linear_mode = None
        return super().train(mode=mode)

    def transform_cpu(self, dtype: torch.dtype | None = None):
        if self._cpu_transformed:
            return
        if dtype not in (None, torch.bfloat16):
            raise RuntimeError(
                f"{self.__class__.__name__} only supports bfloat16 fused activations, got {dtype}."
            )

        self._dequant_scales_dtype = self.scales.dtype
        # Keep scales in FP32. BF16 scale rounding alone consumes most of the
        # 2e-3 inference error budget on small-K layers; the dot product still
        # uses BF16 inputs with FP32 accumulation.
        #
        # ``to`` returns the same tensor when the dtype already matches, which
        # would leave scales as a view of the checkpoint mapping (#4785).
        scales = self.scales.to(torch.float32)
        if scales.data_ptr() == self.scales.data_ptr():
            scales = detach_cpu_operand(scales)
        self.scales = scales.contiguous()
        if self.bias is not None:
            bias = self.bias.to(torch.float32)
            if bias.data_ptr() == self.bias.data_ptr():
                bias = detach_cpu_operand(bias)
            self.bias = bias.contiguous()

        if self.pack_gptq_int4_kernel is None:
            ok, err = type(self).cached_validate_once()
            if not ok:
                raise RuntimeError(str(err))

        ret_idx = (
            torch.empty(0, dtype=torch.int32)
            if self._ret_idx_is_identity()
            else self._build_ret_idx()
        )
        self.qweight, self.qzeros = self.pack_gptq_int4_kernel(
            self.qweight,
            self.qzeros,
            ret_idx,
            self.out_features,
            self.in_features,
        )
        self._cpu_transform_dtype = torch.bfloat16
        self._cpu_transformed = True
        # ``g_idx`` still aliases the checkpoint mapping and would pin the whole
        # shard even though ``qweight``/``qzeros`` are now private (#4785).
        self.g_idx = detach_cpu_operand(getattr(self, "g_idx", None))
        # ``unpack_hf_int4`` inverts ``pack_hf_int4``, so the checkpoint
        # ``qweight`` can be rebuilt on save. Drop the resident clone once that
        # inverse reproduces the bytes captured before the transform; a layout
        # this build cannot invert keeps it (#4783).
        finalize_fused_train_state(self)

    def dequantize_weight(
        self, num_itr: int = 1, dtype: torch.dtype | None = None
    ):
        if not self._cpu_transformed:
            # The checkpoint layout shares the base decoder, including the
            # range-safe ``dtype`` selection a half-precision fallback needs.
            return super().dequantize_weight(num_itr=num_itr, dtype=dtype)
        if num_itr != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__}.dequantize_weight only supports num_itr=1"
            )

        # ``transform_cpu`` installs the extension's packed nibbles and the
        # checkpoint's integer zero codes, so the live affine identity is
        # ``(code - zero) * scale``. Apply the requested dtype before the
        # arithmetic: a half fallback asks for a dtype that holds every
        # decoded weight, while ``None`` keeps the export behavior (FP32
        # arithmetic, checkpoint scale dtype on return).
        compute_dtype = torch.float32 if dtype is None else dtype
        codes = unpack_hf_int4(
            self.qweight,
            out_features=self.out_features,
            in_features=self.in_features,
        )
        group_size = max(int(self.group_size), 1)
        groups = torch.arange(self.in_features, device=codes.device) // group_size
        zeros = self.qzeros.index_select(0, groups).t()
        scales = self.scales.index_select(0, groups).t()
        weight = (codes.to(compute_dtype) - zeros.to(compute_dtype)) * scales.to(compute_dtype)

        # ``transform_cpu`` sorts input channels by GPTQ group before packing.
        # Restore the checkpoint's original channel order for nn.Linear users.
        if not self._ret_idx_is_identity():
            ret_idx = self._build_ret_idx().to(device=weight.device, dtype=torch.long)
            restored = torch.empty_like(weight)
            restored[:, ret_idx] = weight
            weight = restored
        out_dtype = self._dequant_scales_dtype if dtype is None else dtype
        return weight.t().contiguous().to(out_dtype)

    @torch.no_grad
    def _fused_op_forward(self, x):
        if not self._ret_idx_is_identity():
            x = x[:, self.ret_idx].contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()
        if x.device.type != "cpu":
            raise NotImplementedError

        if not self._fused_input_dtype(x):
            raise RuntimeError(
                f"{self.__class__.__name__} requires bfloat16 activations, got {x.dtype}."
            )
        # Source metadata may be column views into a fused allocation. The
        # native ABI requires compact metadata; keep these copies local.
        zeros = self.qzeros if self.qzeros.is_contiguous() else self.qzeros.contiguous()
        scales = self.scales if self.scales.is_contiguous() else self.scales.contiguous()
        out = self.gemm_int4_qzero_group_scaled_cpu_kernel(
            x,
            self.qweight,
            zeros,
            scales,
            self.group_size,
            self.bias,
        )
        return out


__all__ = [
    "TorchCpuAvx512Linear",
    "avx512_checkpoint_bytes",
    "pack_hf_int4",
    "unpack_hf_int4",
]
