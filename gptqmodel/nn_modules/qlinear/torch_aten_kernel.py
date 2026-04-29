# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from .torch_fused import pack_scales_and_zeros


log = setup_logger()


def _has_local_int4pack_cpu_ops() -> bool:
    return (
        hasattr(torch.ops.aten, "_convert_weight_to_int4pack_for_cpu")
        and hasattr(torch.ops.aten, "_weight_int4pack_mm_for_cpu")
    )


def _cpu_int4pack_zero_offsets(
    zero_codes: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    # aten::_weight_int4pack_mm_for_cpu dequantizes as:
    #   scale * (signed_code - 2^(bits-1)) + zero_offset
    # Convert stored GPTQ zero codes so the fused kernel reproduces:
    #   scale * (code - zero_code)
    zero_center = 1 << (bits - 1)
    return (zero_center - zero_codes.to(dtype=scales.dtype)) * scales


class TorchAtenLinear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH_ATEN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 110, FORMAT.GPTQ_V2: 110}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "torch_aten_kernel"

    gemm_int4_forward_kernel = None

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
            backend=kwargs.pop("backend", BACKEND.GPTQ_TORCH_ATEN),
            adapter=adapter,
            register_buffers=register_buffers,
            enable_wf_unsqueeze=kwargs.pop("enable_wf_unsqueeze", True),
            **kwargs,
        )

        self.linear_mode = None
        self.dequant_dtype = torch.int8

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not _has_local_int4pack_cpu_ops():
            cls.gemm_int4_forward_kernel = None
            err = ImportError(
                "TorchAtenLinear requires aten::_convert_weight_to_int4pack_for_cpu and "
                "aten::_weight_int4pack_mm_for_cpu in this PyTorch build."
            )
            log.warning(str(err))
            return False, err

        cls.gemm_int4_forward_kernel = staticmethod(torch.ops.aten._weight_int4pack_mm_for_cpu)
        return True, None

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return

        super().optimize()

    def _build_ret_idx(self) -> torch.Tensor:
        existing = getattr(self, "ret_idx", None)
        total = self.g_idx.shape[0]
        if isinstance(existing, torch.Tensor) and existing.numel() == total:
            return existing

        device = self.g_idx.device
        ret_idx = torch.zeros(total, dtype=torch.int32, device=device)
        group_size = max(int(self.group_size), 1)
        groups = total // group_size
        remainder = total % group_size
        g_idx = self.g_idx.to(torch.int32)
        g_idx_2 = g_idx * group_size

        if remainder > 0:
            mask = g_idx == groups
            if mask.any():
                g_idx_2[mask] += torch.arange(remainder, device=device, dtype=torch.int32)

        if groups > 0:
            base = torch.arange(group_size, device=device, dtype=torch.int32)
            for i in range(groups):
                mask = g_idx == i
                if not mask.any():
                    continue
                count = int(mask.sum().item())
                g_idx_2[mask] += base[:count]

        ret_idx[g_idx_2] = torch.arange(total, device=device, dtype=torch.int32)
        self.ret_idx = ret_idx
        return ret_idx

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self

        from ...utils.model import convert_gptq_v1_to_v2_format_module

        if self.SUPPORTS_TRAINING_USE_TORCH_KERNEL:
            if mode:
                if self.qzero_format() == 1:
                    if not hasattr(self, "qzeros_data_v1"):
                        self.qzeros_data_v1 = self.qzeros.data.clone()
                        convert_gptq_v1_to_v2_format_module(self, bits=self.bits, pack_dtype=self.pack_dtype)
                        self.qzeros_data_v2 = self.qzeros.data
                    else:
                        self.qzeros.data = self.qzeros_data_v2
                        self.qzero_format(format=2)
            else:
                if hasattr(self, "qzeros_data_v1"):
                    self.qzeros.data = self.qzeros_data_v1
                    self.qzero_format(format=1)

        return super().train(mode=mode)

    def transform_cpu(self):
        self.scales = self.scales.to(torch.bfloat16).contiguous()

        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            ).to(torch.uint8),
            self.maxq,
        )
        ret_idx = self._build_ret_idx()
        weight = (
            weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
            .index_select(0, ret_idx)
            .t()
            .contiguous()
        )
        self.qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(weight.int(), 1).contiguous()

        zero_codes = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
        ).to(torch.uint8)
        zero_codes = torch.bitwise_and(zero_codes, self.maxq).reshape(self.scales.shape)
        self.qzeros = _cpu_int4pack_zero_offsets(zero_codes, self.scales, self.bits).contiguous()
        self.scales_and_zeros = pack_scales_and_zeros(self.scales, self.qzeros)

    def transform(self, device):
        if device == "cpu":
            self.transform_cpu()
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if (
            not self.training
            and not x.requires_grad
            and self.linear_mode is None
            and _has_local_int4pack_cpu_ops()
            and x.device.type == "cpu"
        ):
            self.transform(x.device.type)
            self.linear_mode = "inference"
        elif self.linear_mode is None:
            self.linear_mode = "train"

        if self.linear_mode == "inference":
            out = self._fused_op_forward(x).reshape(out_shape)
        else:
            num_itr = self.g_idx.shape[0] // x.shape[-1]
            weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
            out = torch.matmul(x, weights).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    @torch.no_grad
    def _fused_op_forward(self, x):
        x = x[:, self.ret_idx].contiguous()
        if x.device.type != "cpu":
            raise NotImplementedError

        original_dtype = x.dtype
        if original_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        out = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x,
            self.qweight,
            self.group_size,
            self.scales_and_zeros,
        )
        if original_dtype != torch.bfloat16:
            out = out.to(original_dtype)
        return out

    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None


def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchAtenLinear):
            raise ValueError(
                "Only models loaded using TorchAtenLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.GPTQ_TORCH_ATEN"
            )

        if isinstance(module, TorchAtenLinear):
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = torch.nn.Parameter(module.bias)

            parent = model
            if "." in name:
                parent_name, module_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = ["TorchAtenLinear", "dequantize_model"]
