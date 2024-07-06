# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2

import math
from logging import getLogger

import torch
import torch.nn.functional as F
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel_exllamav2_kernels import gemm_half_q_half, make_q_matrix

logger = getLogger(__name__)



# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
NONE_TENSOR = torch.empty((1, 1), device="meta")


def _torch_device(idx):
    if idx == -1:
        return "cpu"
    return f"cuda:{idx}"


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


def ext_make_q_matrix(w: dict, temp_dq, key: str = None):
    """
    Create Q matrix
    """
    # EXL2
    # won't work as the moment because the tensors are not the same.
    if "q_weight" in w:
        w["q_scale_max"] /= 256
        w["q_perm"] = w["q_perm"].short()
        w["q_invperm"] = w["q_invperm"].short()
        return make_q_matrix(
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
            return make_q_matrix(
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
            return make_q_matrix(
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


class ExllamaV2QuantLinear(BaseQuantLinear):
    SUPPORTED_BITS = [4]

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(self, bits: int, group_size: int, desc_act: bool, sym: bool, infeatures: int, outfeatures: int,
                 bias: bool,  **kwargs,):
        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, **kwargs)

        self.q_handle = None
        self.q_tensors = None

        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures

        # auto pad
        self.outfeatures = outfeatures + (-outfeatures % 32)
        self.infeatures = infeatures + (-infeatures % self.group_size)

        # backup original values
        self.original_outfeatures = outfeatures
        self.original_infeatures = infeatures
        self.maxq = 2**self.bits - 1

        assert self.infeatures % 32 == 0
        assert self.outfeatures % 32 == 0

        # I need to register the tensors, otherwise, we won't be able to load them easily using transformers ...
        self.register_buffer(
            "qweight",
            torch.zeros((self.original_infeatures // 32 * self.bits, self.original_outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(self.original_infeatures / self.group_size),
                    self.original_outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(self.original_infeatures / self.group_size), self.original_outfeatures),
                dtype=torch.float16,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(self.original_infeatures)], dtype=torch.int32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((self.original_outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self, temp_dq):
        self.validate_device(self.qweight.device.type)
        assert self.qweight.device.index is not None

        # resize due to padding after model weights have been loaded
        if self.outfeatures != self.original_outfeatures or self.infeatures != self.original_infeatures:
            self.qweight.resize_(self.infeatures // 32 * self.bits, self.outfeatures)
            self.qzeros.resize_(
                math.ceil(self.infeatures / self.group_size),
                self.outfeatures // 32 * self.bits
            )
            self.scales.resize_(math.ceil(self.infeatures / self.group_size), self.outfeatures)
            self.g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32, device=self.g_idx.device)
            if self.bias is not None:
                self.bias.resize_(self.outfeatures)

        self.q_tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
            "g_idx": self.g_idx,
        }
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq)

    def forward(self, x, force_cuda=False):
        if x.dtype != torch.float16:
            logger.warning_once(
                f"The exllama v2 kernel for GPTQ requires a float16 input activation, while {x.dtype} was passed. Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 intermediate activations in the model."
            )

            x = x.half()

        # TODO: need to run checks to make sure there is no performance regression padding with F.pad
        # if infeatures is padded, we need to pad the input as well
        if x.size(-1) != self.infeatures and self.infeatures > self.original_infeatures:
            x = F.pad(x, (0, self.infeatures - self.original_infeatures))

        output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, force_cuda)

        if self.bias is not None:
            output.add_(self.bias)
        return output

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)


class ExLlamaV2DeviceTensors:
    device_idx: int
    scratch_bytes: int
    scratch_idx: int
    scratch: torch.tensor = None

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty(
            (self.scratch_bytes // 2,),
            dtype=torch.half,
            device=_torch_device(self.device_idx),
        )

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
