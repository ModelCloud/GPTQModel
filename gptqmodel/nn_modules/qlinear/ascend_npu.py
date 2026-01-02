# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.marlin import replace_parameter


log = setup_logger()

try:
    import torch_npu  # type: ignore

    torch_npu_import_exception: Optional[BaseException] = None
except BaseException as exc:  # pragma: no cover - optional dependency
    torch_npu = None
    torch_npu_import_exception = exc


def unpack_from_int32(
    weight: torch.Tensor,
    num_bits: int,
    packed_dim: int,
) -> torch.Tensor:
    """Unpack unsigned packed int32 weights into signed int8 values.

    Ascend NPU inference kernels expect per-element int8 values (or int4-packed),
    so we unpack GPTQ's packed int32 representation and apply symmetric offset.
    """
    if weight.dtype != torch.int32:
        raise TypeError(f"Expected torch.int32 packed tensor, got {weight.dtype}.")
    if num_bits > 8:
        raise ValueError(f"Expected num_bits <= 8, got {num_bits}.")
    if packed_dim not in (0, 1):
        raise ValueError(f"Expected packed_dim in (0, 1), got {packed_dim}.")

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    shifts = (
        torch.arange(pack_factor, device=weight.device, dtype=weight.dtype) * num_bits
    )

    if packed_dim == 1:
        unpacked = (weight.unsqueeze(-1) >> shifts) & mask
        unpacked = unpacked.reshape(weight.shape[0], weight.shape[1] * pack_factor)
    else:
        unpacked = ((weight.unsqueeze(1) >> shifts.view(1, pack_factor, 1)) & mask).reshape(
            weight.shape[0] * pack_factor, weight.shape[1]
        )

    offset = (1 << num_bits) // 2
    return (unpacked - offset).to(torch.int8)


class AscendNPUQuantLinear(BaseQuantLinear):
    """GPTQ inference kernel backed by Ascend NPU ops (torch_npu)."""

    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.NPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    # for transformers/optimum tests compat
    QUANT_TYPE = "ascend_npu"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        register_buffers: bool = True,
        adapter: Adapter = None,
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
            backend=kwargs.pop("backend", BACKEND.ASCEND_NPU),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )

        # When desc_act=True, we pre-sort g_idx and capture the input permutation.
        self.register_buffer(
            "input_perm",
            torch.empty(0, dtype=torch.int32),
            persistent=False,
        )

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if torch_npu_import_exception is not None:
            return False, ImportError(str(torch_npu_import_exception))
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)

        # Avoid selecting the kernel when Ascend runtime is installed but no NPU is visible.
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise NotImplementedError(
                "Ascend NPU runtime is not available (torch.npu.is_available() is False)."
            )

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok_once, err_once = cls.cached_validate_once()
        if not ok_once:
            return False, err_once

        ok, err = cls._validate(**args)
        if not ok:
            return ok, err

        bits = args.get("bits")
        out_features = args.get("out_features")
        # 4-bit weights are stored packed along output columns (N/8), so require N % 8 == 0.
        if bits == 4 and out_features is not None and (out_features % 8) != 0:
            return False, NotImplementedError(
                f"{cls} requires out_features divisible by 8 for 4-bit weights: out_features={out_features}"
            )

        return True, None

    def post_init(self):
        if torch_npu is None:
            raise ImportError(
                "AscendNPUQuantLinear requires `torch_npu` but it could not be imported."
            )

        device = self.qweight.device

        perm = None
        if self.desc_act:
            # Ascend kernels do not consume g_idx, so we pre-sort inputs/weights to restore
            # contiguous group layout and store the permutation for forward().
            perm = torch.argsort(self.g_idx).to(torch.int32)
            sorted_g_idx = self.g_idx[perm]
            replace_parameter(
                self,
                "g_idx",
                torch.nn.Parameter(sorted_g_idx.to(device=device), requires_grad=False),
            )
            self.input_perm = perm.to(device=device)

        # Prepare anti-quant params. GPTQ v1 stored qzeros with "-1" offset, so we
        # only apply "+1" when qzeros are still in v1 format.
        qzeros = unpack_from_int32(
            self.qzeros.data.contiguous(),
            self.bits,
            packed_dim=1,
        ).to(self.scales.dtype)
        if self.qzero_format() == 1:
            qzeros = qzeros + 1
            self.qzero_format(format=2)

        replace_parameter(
            self,
            "scales",
            torch.nn.Parameter(self.scales.data.contiguous(), requires_grad=False),
        )
        replace_parameter(
            self,
            "qzeros",
            torch.nn.Parameter(qzeros.contiguous(), requires_grad=False),
        )

        qweight = unpack_from_int32(
            self.qweight.data.contiguous(),
            self.bits,
            packed_dim=0,
        )
        if perm is not None:
            qweight = qweight[perm, :]

        # For 4-bit, keep weights packed to save memory (8 x int4 packed into int32).
        if self.bits == 4:
            qweight = torch_npu.npu_convert_weight_to_int4pack(qweight.to(torch.int32))

        replace_parameter(
            self,
            "qweight",
            torch.nn.Parameter(qweight.contiguous(), requires_grad=False),
        )

        if self.bias is not None:
            self.bias = self.bias.to(device=device)

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "input_perm") and self.input_perm is not None:
            buf.append(self.input_perm)
        return buf

    def forward(self, x: torch.Tensor):
        if torch_npu is None:
            raise ImportError(
                "AscendNPUQuantLinear requires `torch_npu` but it could not be imported."
            )

        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        out_shape = x.shape[:-1] + (self.out_features,)
        reshaped_x = x.reshape(-1, x.shape[-1])
        x_for_kernel = reshaped_x

        if self.input_perm.numel() > 0:
            perm = self.input_perm
            if perm.device != reshaped_x.device:
                perm = perm.to(device=reshaped_x.device)
            x_for_kernel = reshaped_x[:, perm]

        bias = self.bias
        if bias is not None and bias.dtype == torch.bfloat16:
            # Some torch_npu kernels expect fp32 bias when activation is bf16.
            bias = bias.float()

        out_2d = torch_npu.npu_weight_quant_batchmatmul(
            x_for_kernel,
            self.qweight,
            antiquant_scale=self.scales,
            antiquant_offset=self.qzeros,
            antiquant_group_size=self.group_size,
            bias=bias,
        )

        if self.adapter:
            out_2d = self.adapter.apply(x=reshaped_x, out=out_2d)

        return out_2d.reshape(out_shape)
