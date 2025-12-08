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


log = setup_logger()

gemm_int4_forward_kernel = None
gemm_int4_forward_kernel_exception = None

try:
    from kernels import get_kernel

    gemm_int4_forward_kernel = get_kernel("kernels-community/quantization_gptq").gemm_int4_forward
except Exception as exc:  # pragma: no cover - best effort fallback
    gemm_int4_forward_kernel_exception = str(exc)
    log.warning("Failed to load CPU gemm_4bit kernel: %s. Use fallback path. \
                Please make sure you already `pip install kernels` and the kernels >= 0.11.1", str(exc))


class HFKernelLinear(PackableQuantLinear):
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

    # for transformers/optimum tests compat
    QUANT_TYPE = "hf_kernel"

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
        self.dequant_dtype = torch.int8

    @classmethod
    def validate(cls, **args):
        if gemm_int4_forward_kernel_exception is not None:
            return False, ImportError(gemm_int4_forward_kernel_exception)

        return cls._validate(**args)

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

    def convert_weight_packed_zp(self, block_n: int = 32):
        """
        qweight: int4_weight (*, K, N) uint8 (0-15)
        return: packed_weight uint8 (*, N, K/2) (low 4bit + high 4bit)
        """
        assert self.qweight.dtype == torch.uint8, "qweight must be uint8"
        sizes = list(self.qweight.shape)
        if len(sizes) < 2:
            raise ValueError("qweight_final rank error")
        N, K = sizes[-2], sizes[-1]
        assert N % block_n == 0, "N must be divisible by block_n"
        assert K % 2 == 0, "K must be even"
        BLOCK_N = block_n
        BIT_COUNT = 32  # (=32 low +32 high)
        prefix = sizes[:-2]
        new_shape = prefix + [N // BLOCK_N, BLOCK_N, K // 2, 2]
        out_shape = prefix + [N, K // 2]
        qw = self.qweight.reshape(new_shape)                # (..., N/B, B, K/2, 2)
        qw = qw.transpose(-3, -2).contiguous()               # (..., N/B, K/2, B, 2)
        qw = qw.reshape(-1, BIT_COUNT * 2)                   # [-1, 64]
        high = qw[:, BIT_COUNT:]                             # high 32
        low  = qw[:, :BIT_COUNT]                             # low 32
        packed = ((high << 4) | low).to(torch.uint8)         # combine
        final_qweight = packed.reshape(out_shape)

        self.qweight = final_qweight.contiguous()

    def transform_cpu(self):
        self.scales = self.scales.to(torch.bfloat16).contiguous()
        # Unpack and reorder qweight
        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one  # self.wf.unsqueeze(-1)
            ).to(torch.uint8),
            self.maxq
        )
        ret_idx = self._build_ret_idx()
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2]).index_select(0, ret_idx).t()
        self.qweight = weight.contiguous()
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero  # self.wf.unsqueeze(0),
        ).to(torch.uint8)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(zeros.shape[0], -1)
        self.qzeros = zeros.contiguous()

    def transform(self, device):
        if device == "cpu":
            self.transform_cpu()
            self.convert_weight_packed_zp()
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if not self.training and not self.transformed and gemm_int4_forward_kernel is not None:
            self.transform(x.device.type)
            self.transformed = True

        if self.transformed:
            out = self._fused_op_forward(x).reshape(out_shape)
        else:
            # make sure dequant dtype matches input x
            num_itr = self.g_idx.shape[0] // x.shape[-1]
            weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
            out = torch.matmul(x, weights).reshape(out_shape)

        # Add bias and adapter
        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    @torch.no_grad
    def _fused_op_forward(self, x):
        x = x[:, self.ret_idx].contiguous()
        if x.device.type == "cpu":
            out = gemm_int4_forward_kernel(x, self.qweight, self.qzeros, self.scales, self.group_size)
        else:
            raise NotImplementedError

        return out

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, HFKernelLinear):
            raise ValueError(
                "Only models loaded using HFKernelLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.HF_KERNEL"
            )

        if isinstance(module, HFKernelLinear):
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


__all__ = ["HFKernelLinear", "dequantize_model"]
