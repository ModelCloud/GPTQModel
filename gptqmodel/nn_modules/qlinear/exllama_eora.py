# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2

from typing import Optional, Tuple

import torch
from torch.nn import Parameter

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...utils.logger import setup_logger


exllama_v2v_import_exception = None

try:
    import gptqmodel_exllama_eora
except ImportError as e:
    exllama_v2v_import_exception = e

log = setup_logger()


# TODO remove this?
def _torch_device(idx):
    if idx == -1:
        return "cpu"
    return f"cuda:{idx}"

def gptq_gemm(x, qweight, qzeros, scales, g_idx, bit):
    return gptqmodel_exllama_eora.gptq_gemm(x, qweight, qzeros, scales, g_idx, True, bit)


def gptq_gemm_lora(x, qweight, qzeros, scales, g_idx, bit, A, B):
    return gptqmodel_exllama_eora.gptq_gemm_lora(x, qweight, qzeros, scales, g_idx, True, bit, A, B)

def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    gptqmodel_exllama_eora.gptq_shuffle(q_weight, q_perm, bit)


class ExllamaEoraQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = [4, 8] # fused eora only validated for 4 bits
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True] # TODO: validate False
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False # TODO: validate True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    # for transformers/optimum tests compat
    QUANT_TYPE = "exllama_v2v"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(self,
         bits: int,
         group_size: int,
         desc_act: bool,
         sym: bool,
         in_features: int,
         out_features: int,
         pack_dtype: torch.dtype,
         adapter: Adapter,
         bias: bool, **kwargs,
    ):
        if exllama_v2v_import_exception is not None:
            raise ValueError(
                f"Trying to use the exllama v2 backend, but could not import the C++/CUDA dependencies with the following error: {exllama_v2v_import_exception}"
            )

        # # backup original values
        # self.original_out_features = out_features
        # self.original_in_features = in_features
        #
        # # auto pad
        # group_size = group_size if group_size != -1 else in_features
        # out_features = out_features + (-out_features % 32)
        # in_features = in_features + (-in_features % group_size)
        # self.in_features_padding_size = in_features - self.original_in_features
        # self.in_features_padding_shape = (0, self.in_features_padding_size)

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            register_buffers=True,
            register_buffers_in_features=in_features,  # self.original_in_features
            register_buffers_out_feature=out_features, # self.original_out_features
            **kwargs)


    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if exllama_v2v_import_exception is not None:
            return False, exllama_v2v_import_exception
        return cls._validate(**args)

    def post_init(self):
        # resize due to padding after model weights have been loaded
        # if self.out_features != self.original_out_features or self.in_features != self.original_in_features:
        #     self.qweight.resize_(self.in_features // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.in_features / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_(math.ceil(self.in_features / self.group_size), self.out_features)
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.in_features)], dtype=torch.int32, device=self.g_idx.device)
        #     if self.bias is not None:
        #         self.bias.resize_(self.out_features)

        super().post_init()

        self.qzeros = Parameter(self.qzeros.data, requires_grad=False)
        self.qweight = Parameter(self.qweight.data, requires_grad=False)
        self.g_idx = Parameter(self.g_idx.data, requires_grad=False)
        self.scales = Parameter(self.scales.data, requires_grad=False)

        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if self.desc_act:
            self.g_idx.data = torch.argsort(self.g_idx).to(torch.int32)
        else:
            self.g_idx.data = torch.empty((0,),
                                           dtype=torch.int32,
                                           device=self.g_idx.device)

        gptq_shuffle(self.qweight, self.g_idx, self.bits)

    def forward(self, x):
        x_dtype = x.dtype
        if x_dtype != torch.float16:
            log.warn.once(
                f"Exllama EoRA kernel requires a float16 input activation, while {x.dtype} was passed. Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 intermediate activations in the model."
            )

            x = x.to(dtype=torch.float16)

        # sync with vllm
        # log.info(f"x shape: {x.shape}")
        # log.info(f"qweight shape: {self.qweight.shape}")
        # log.info(f"in_features: {self.in_features}")
        # log.info(f"out_features: {self.out_features}")
        # log.info(f"x.shape[:-1]: {x.shape[:-1]}")
        # log.info(f"self.qweight.shape[-1],: {self.qweight.shape[-1],}")

        out_shape = x.shape[:-1] + (self.out_features,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        # log.info(f"out_shape: {out_shape}")
        # log.info(f"reshaped_x: {reshaped_x.shape}")

        # TODO: need to run checks to make sure there is no performance regression padding with F.pad
        # if in_features is padded, we need to pad the input as well
        # if x.size(-1) != self.in_features:
        #     x = F.pad(x, self.in_features_padding_shape)

        if self.adapter:
            # only 4 bits fused eora kernel has been validated
            if self.bits == 4:
                output = gptq_gemm_lora(x, self.qweight, self.qzeros, self.scales, self.g_idx, self.bits, x @ self.adapter.lora_A, self.adapter.lora_B) # fused
            else:
                output = gptq_gemm(reshaped_x, self.qweight, self.qzeros, self.scales, self.g_idx, self.bits).add_((reshaped_x @ self.adapter.lora_A) @ self.adapter.lora_B) # normal
        else:
            output = gptq_gemm(reshaped_x, self.qweight, self.qzeros, self.scales, self.g_idx, self.bits)


        if self.bias is not None:
            output.add_(self.bias)

        # log.info(f"output: {output.shape}")

        # sync with vllm
        output = output.reshape(out_shape)
        # log.info(f"output reshaped: {output.shape}")

        return output.to(dtype=x_dtype)
