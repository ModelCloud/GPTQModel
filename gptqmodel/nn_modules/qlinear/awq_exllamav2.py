# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization.awq.utils.module import try_import
from ...quantization.awq.utils.packing_utils import unpack_reorder_pack
from ...utils.backend import BACKEND
from ...utils.exllamav2 import ScratchSpace
from ...utils.logger import setup_logger


log = setup_logger()

exlv2_ext, msg = try_import("gptqmodel_exllamav2_kernels")

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


class AwqExllamaV2QuantLinear(AWQuantLinear):
    SUPPORTS_BITS = [4]
    # TODO: intel is reporting v2 has accuracy issues with group-size == 16 for this kernel
    # disable for now until we can validate this issue: ref https://github.com/ModelCloud/GPTQModel/issues/1515
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA] # ROCm has broken accuracies issues
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_exllamav2"

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
            backend=kwargs.pop("backend", BACKEND.EXLLAMA_V2),
            adapter=adapter,
            **kwargs)

    def post_init(self, scratch_space: ScratchSpace):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        if exlv2_ext is None:
            raise ModuleNotFoundError("External ExLlama kernels are not properly installed." + msg)

        # awq only accepts float16
        self.scales = self.scales.to(dtype=torch.float16)

        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        self.qweight, self.qzeros = unpack_reorder_pack(
            self.qweight, self.qzeros, self.bits
        )

        temp_dq_size = self.temp_dq_size()
        temp_dq = scratch_space.get_slice(temp_dq_size)
        self.q_handle = exlv2_ext.make_q_matrix(
            self.qweight,
            none_tensor,
            none_tensor,
            none_tensor,
            none_tensor,
            none_tensor,
            self.qzeros,
            self.scales,
            none_tensor,
            temp_dq,
        )

        super().post_init()

    def forward(self, x: torch.Tensor):
        assert self.q_handle is not None, (
            "module.post_init() must be called before module.forward(). "
            "Use exllamav2_post_init() on the whole model."
        )
        if exlv2_ext is None:
            raise ModuleNotFoundError("External ExLlamaV2 kernels are not properly installed." + msg)

        input_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.out_features,)

        if input_dtype != torch.float16:
            x = x.to(dtype=torch.float16)

        x = x.view(-1, x.shape[-1])

        out = torch.empty(
            (x.shape[0], self.out_features),
            dtype=torch.float16,
            device=x.device,
        )
        exlv2_ext.gemm_half_q_half(x, self.q_handle, out, False)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.view(out_shape)

    def temp_dq_size(self):
        """
        Returns the size of the temporary buffer required for the dq kernel.
        """
        return self.in_features * self.out_features * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        """
        Returns the size of the temporary buffer required for the fwd kernel.
        """
        return self.out_features * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        """
        Returns the size of the fixed scratch space required for the kernel.
        """
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


__all__ = ["AwqExllamaV2QuantLinear"]
