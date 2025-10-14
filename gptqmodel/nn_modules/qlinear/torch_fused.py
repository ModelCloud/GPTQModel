# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.torch import TORCH_HAS_FUSED_OPS


log = setup_logger()

# TODO: CPU works, not yet working for cuda fused int4 ops
def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    # assert scales.dtype == torch.bfloat16
    # assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .contiguous()
    )

class TorchFusedQuantLinear(PackableQuantLinear):
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.XPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch_fused"

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
            backend=kwargs.pop("backend", BACKEND.TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        self.transformed = False
        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return

        super().optimize()

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self

        from ...utils.model import convert_gptq_v1_to_v2_format_module

        if self.SUPPORTS_TRAINING_USE_TORCH_KERNEL:
            # training starts
            if mode:
                # one time clone v1 qzeros and save both v1 and v2 qzeros in memory
                if self.qzero_format() == 1:
                    if not hasattr(self, "qzeros_data_v1"):
                        self.qzeros_data_v1 = self.qzeros.data.clone()
                        convert_gptq_v1_to_v2_format_module(self, bits=self.bits, pack_dtype=self.pack_dtype)
                        self.qzeros_data_v2 = self.qzeros.data
                    else:
                        self.qzeros.data = self.qzeros_data_v2
                        self.qzero_format(format=2)

            # training switching to inference/eval
            else:
                if hasattr(self, "qzeros_data_v1"):
                    # switch qzero back to v1 for inference/eval
                    self.qzeros.data = self.qzeros_data_v1
                    self.qzero_format(format=1)

        return super().train(mode=mode)

    def transform_xpu(self, dtype):
        self.scales = self.scales.clone().to(dtype).contiguous()
        # Unpack qzeros
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero  # self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(zeros.shape[0], -1)
        # Unpack and reorder qweight
        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq
        )
        self.ret_idx = torch.zeros(self.g_idx.shape[0], dtype=torch.int32).to(self.g_idx.device)
        groups = self.g_idx.shape[0] // self.group_size
        remainder = self.g_idx.shape[0] % self.group_size
        g_idx_2 = self.g_idx * self.group_size
        if remainder > 0:
            g_idx_2[self.g_idx == groups] += torch.arange(remainder).to(self.g_idx_2.device).to(self.g_idx_2.dtype)
        arange_tensor = torch.arange(self.group_size).to(self.g_idx.device).to(self.g_idx.dtype)
        for i in range(groups):
            g_idx_2[self.g_idx == i] += arange_tensor
        self.ret_idx[g_idx_2] = torch.arange(self.g_idx.shape[0]).to(self.ret_idx.device).to(self.ret_idx.dtype)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2]).index_select(0, self.ret_idx).t()
        # Pack qweight
        packed = torch.zeros(weight.shape[0], weight.shape[1] // self.pack_factor, dtype=torch.int32, device=weight.device)
        for col in range(weight.shape[1] // self.pack_factor):
            for i in range(self.pack_factor):
                packed_col = weight[:, col * self.pack_factor + i].to(torch.int32)
                packed[:, col] |= packed_col << (i * self.bits)

        self.qweight = packed.contiguous()
        self.qzeros = zeros.contiguous()

    def transform_cpu(self, dtype):
        self.scales = self.scales.clone().to(dtype).contiguous()
        # Unpack and reorder qweight
        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq
        )
        self.ret_idx = torch.zeros(self.g_idx.shape[0], dtype=torch.int32).to(self.g_idx.device)
        groups = self.g_idx.shape[0] // self.group_size
        remainder = self.g_idx.shape[0] % self.group_size
        g_idx_2 = self.g_idx * self.group_size
        if remainder > 0:
            g_idx_2[self.g_idx == groups] += torch.arange(remainder).to(self.g_idx_2.device).to(self.g_idx_2.dtype)
        arange_tensor = torch.arange(self.group_size).to(self.g_idx.device).to(self.g_idx.dtype)
        for i in range(groups):
            g_idx_2[self.g_idx == i] += arange_tensor
        self.ret_idx[g_idx_2] = torch.arange(self.g_idx.shape[0]).to(self.ret_idx.device).to(self.ret_idx.dtype)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2]).index_select(0, self.ret_idx).t()
        self.qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(weight.int(), 1).contiguous()
        self.qzeros = torch.zeros_like(self.scales).contiguous()
        self.scales_and_zeros = pack_scales_and_zeros(self.scales, self.qzeros)

    def transform(self, dtype, device):
        if device == "xpu":
            self.transform_xpu(dtype)
        elif device == "cpu":
            self.transform_cpu(dtype)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        out = self._forward(x, out_shape)
        return out

    def _forward(self, x, out_shape):
        num_itr = self.g_idx.shape[0] // x.shape[-1]

        if not self.training and not self.transformed and TORCH_HAS_FUSED_OPS:
            # one-time transform per module for xpu aten fused ops
            self.transform(x.dtype, x.device.type)
            self.transformed = True

        if self.transformed:
            x = x[:, self.ret_idx].contiguous()
            # fused ops optimized for xpu using torch.ops
            # note _weight_int4pack_mm_with_scales_and_zeros is added by intel for xpu only
            if x.device.type == "xpu":
                out = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
                    x, self.qweight, self.group_size, self.scales, self.qzeros
                ).reshape(out_shape)
            elif x.device.type == "cpu":
                out = torch.ops.aten._weight_int4pack_mm_for_cpu(
                    x, self.qweight, self.group_size, self.scales_and_zeros
                ).reshape(out_shape)
            else:
                raise NotImplementedError
        else:
            # make sure dequant dtype matches input x
            weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
            out = torch.matmul(x, weights).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchFusedQuantLinear):
            raise ValueError(
                "Only models loaded using TorchFusedQuantLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.TORCH_FUSED"
            )

        if isinstance(module, TorchFusedQuantLinear):
            # Create a new Linear layer with dequantized weights
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = torch.nn.Parameter(module.bias)

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


__all__ = ["TorchFusedQuantLinear", "dequantize_model"]
