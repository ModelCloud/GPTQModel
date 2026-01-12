# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2

from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.exllamav2 import ScratchSpace
from ...utils.logger import setup_logger


log = setup_logger()


# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
NONE_TENSOR = torch.empty((1, 1), device="meta")


def _torch_device(idx):
    if idx == -1:
        return "cpu"
    return f"cuda:{idx}"

class ExllamaV2QuantLinear(BaseQuantLinear):
    SUPPORTS_BACKEND = BACKEND.EXLLAMA_V2
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 80, FORMAT.GPTQ_V2: 80}
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
    QUANT_TYPE = "exllamav2"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    gptqmodel_exllamav2_kernels = None

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
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs, ):
        # backup original values
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
            backend=kwargs.pop("backend", BACKEND.EXLLAMA_V2),
            adapter=adapter,
            register_buffers=register_buffers,
            register_buffers_in_features=in_features,
            register_buffers_out_feature=out_features,
            **kwargs)

        self.q_handle = None
        self.q_tensors = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            import gptqmodel_exllamav2_kernels
            cls.gptqmodel_exllamav2_kernels = gptqmodel_exllamav2_kernels
            return True, None
        except ImportError as e:
            return False, e

    def post_init(self, scratch_space: ScratchSpace):
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

        # ext_make_q_matrix only accepts float16
        self.scales = self.scales.to(dtype=torch.float16)

        self.q_tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
            "g_idx": self.g_idx,
        }
        temp_dq = scratch_space.get_slice(self.temp_dq_size())
        self.q_handle = self.ext_make_q_matrix(self.q_tensors, temp_dq)

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "q_tensors") and self.q_tensors is not None:
            buf.append(self.q_tensors)
        return buf

    def forward(self, x: torch.Tensor, force_cuda=False):
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        x_dtype = x.dtype
        if x_dtype != torch.float16:
            # log.warn.once(
            #     f"Exllama v2 kernel requires a float16 input activation, while {x.dtype} was passed. Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 intermediate activations in the model."
            # )

            x = x.to(dtype=torch.float16)

        # TODO: need to run checks to make sure there is no performance regression padding with F.pad
        # if in_features is padded, we need to pad the input as well
        # if x.size(-1) != self.in_features:
        #     x = F.pad(x, self.in_features_padding_shape)


        out = self.ext_gemm_half_q_half(x, self.q_handle, self.out_features, force_cuda)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.to(dtype=x_dtype)

    def temp_dq_size(self):
        return self.in_features * self.out_features * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.out_features * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)

    def ext_gemm_half_q_half(self, x, q_handle, q4_width, force_cuda):
        """Matrix multiplication, returns x @ q4"""
        output_shape = x.shape[:-1] + (q4_width,)
        x = x.view(-1, x.shape[-1])
        output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
        self.gptqmodel_exllamav2_kernels.gemm_half_q_half(x, q_handle, output, force_cuda)
        return output.view(output_shape)

    def ext_make_q_matrix(self, w: dict, temp_dq, key: str = None):
        """
        Create Q matrix
        """
        # EXL2
        # won't work as the moment because the tensors are not the same.
        if "q_weight" in w:
            w["q_scale_max"] /= 256
            w["q_perm"] = w["q_perm"].short()
            w["q_invperm"] = w["q_invperm"].short()
            return self.gptqmodel_exllamav2_kernels.make_q_matrix(
                w["q_weight"],
                w["q_perm"],
                w["q_invperm"],
                w["q_scale"],
                w["q_scale_max"],
                w["q_groups"],
                NONE_TENSOR,
                NONE_TENSOR,
                NONE_TENSOR,
                temp_dq,
            )
        # GPTQ
        elif "qweight" in w:
            if w["scales"].dtype == torch.float:
                w["scales"] = w["scales"].half()

            # GPTQ with g_idx (act_order)
            if "g_idx" in w and not (w["g_idx"] == 0).all().item():
                w["q_perm"] = torch.empty(
                    (w["qweight"].shape[0] * 8,),
                    dtype=torch.short,
                    device=w["qweight"].device,
                )
                w["q_invperm"] = torch.empty_like(w["q_perm"])
                # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
                return self.gptqmodel_exllamav2_kernels.make_q_matrix(
                    w["qweight"],
                    w["q_perm"],
                    w["q_invperm"],
                    NONE_TENSOR,
                    NONE_TENSOR,
                    NONE_TENSOR,
                    w["qzeros"],
                    w["scales"],
                    w["g_idx"].cpu(),
                    temp_dq,
                )
            # GPTQ without g_idx
            else:
                return self.gptqmodel_exllamav2_kernels.make_q_matrix(
                    w["qweight"],
                    NONE_TENSOR,
                    NONE_TENSOR,
                    NONE_TENSOR,
                    NONE_TENSOR,
                    NONE_TENSOR,
                    w["qzeros"],
                    w["scales"],
                    NONE_TENSOR,
                    temp_dq,
                )
        else:
            raise ValueError("q_weight not found in exllama v2 quantized weights")
