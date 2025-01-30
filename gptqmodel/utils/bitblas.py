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

import os

import bitblas
import threadpoolctl as tctl
import torch
from accelerate.utils import find_tied_parameters
from bitblas.quantization import general_compress

from .safetensor import untie_weights
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.bitblas import BitBLASQuantLinear
from ..quantization import FORMAT, QuantizeConfig
from ..utils.logger import setup_logger
from .model import load_checkpoint_in_model_then_tie_weights, recurse_getattr, recurse_setattr
from .progress import ProgressBar
from .torch import torch_empty_cache


logger = setup_logger()

def unpack_qzeros(qzeros, bits):
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )
    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >> (bits * i))

    # Follow the instruction in AutoGPTQ qlinear_cuda_old.py line 303
    # NOTE: It appears that casting after the `unpacked_zeros  + 1` is important.
    return torch.bitwise_and(unpacked_zeros + 1, 2**bits - 1)

# For gptqv2 from gptqmodel
def unpack_qzeros_v2(qzeros, bits):
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )
    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >> (bits * i))

    # Follow the instruction in AutoGPTQ qlinear_cuda_old.py line 303
    # NOTE: It appears that casting after the `unpacked_zeros  + 1` is important.
    return torch.bitwise_and(unpacked_zeros, 2**bits - 1)

def unpack_qweight(qweight, bits):
    qweight = qweight.view(torch.int8)
    elems_per_int8 = 8 // bits
    unpacked_weight = torch.zeros(
        (qweight.shape[0], qweight.shape[1] * elems_per_int8),
        dtype=torch.int8,
        device=qweight.device,
        requires_grad=False,
    )
    for col in range(unpacked_weight.shape[1]):
        i = col % elems_per_int8
        unpacked_weight[:, col] = (qweight[:, col // elems_per_int8] >> (bits * i))

    return torch.bitwise_and(unpacked_weight, 2**bits - 1)


def repack_from_gptq(self, gptq_module, device="cuda"):
    # qweight in gptq old quant linear stored with (out_features, in_features), should be transposed.
    qweight = gptq_module.qweight.T.contiguous().view(self.TORCH_STORAGE_DTYPE)
    intweight = unpack_qweight(qweight, self.bits).contiguous()
    if self.bitblas_matmul.weight_transform is not None:
        qweight = self.bitblas_matmul.weight_transform(intweight.cpu()).to(device)
    self.qweight = qweight
    # scales in gptq old quant linear stored with (in_features // group_size, out_features), should be transposed.
    scales = gptq_module.scales.T.contiguous().view(self.TORCH_DTYPE)
    self.scales = scales
    # qzeros should be dequantized to int zeros.
    intzeros = unpack_qzeros(gptq_module.qzeros, self.bits).T.contiguous()
    if self.bitblas_matmul.config.zeros_mode == "original":
        self.zeros = intzeros.to(torch.float16).contiguous()
    elif self.bitblas_matmul.config.zeros_mode == "rescale":
        self.zeros[:, :] = intzeros.to(torch.float16)[:, :] * self.scales[:, :]
    elif self.bitblas_matmul.config.zeros_mode == "quantized":
        self.zeros = (
            torch.Tensor(general_compress(intzeros.T.contiguous().cpu().numpy(), self.bits)).to(
                self.qweight.device).to(self.zeros.dtype).contiguous())
    else:
        raise ValueError(f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_mode}")
    if self.bias is not None:
        self.bias = gptq_module.bias.data.to(torch.float16).contiguous()


def repack_from_gptq_v2(self, gptq_module):
    # qweight in gptq old quant linear stored with (out_features, in_features), should be transposed.
    qweight = gptq_module.qweight.T.contiguous().view(self.TORCH_STORAGE_DTYPE)
    intweight = unpack_qweight(qweight, self.bits).contiguous()
    if self.bitblas_matmul.weight_transform is not None:
        qweight = self.bitblas_matmul.weight_transform(intweight.cpu()).cuda()
    self.qweight = qweight
    # scales in gptq old quant linear stored with (in_features // group_size, out_features), should be transposed.
    scales = gptq_module.scales.T.contiguous().view(self.TORCH_DTYPE)
    self.scales = scales
    # qzeros should be dequantized to int zeros.
    intzeros = unpack_qzeros_v2(gptq_module.qzeros, self.bits).T.contiguous()
    if self.bitblas_matmul.config.zeros_mode == "original":
        self.zeros = intzeros.to(torch.float16).contiguous()
    elif self.bitblas_matmul.config.zeros_mode == "rescale":
        self.zeros[:, :] = intzeros.to(torch.float16)[:, :] * self.scales[:, :]
    elif self.bitblas_matmul.config.zeros_mode == "quantized":
        self.zeros = (
            torch.Tensor(general_compress(intzeros.T.contiguous().cpu().numpy(), self.bits)).to(
                self.qweight.device).to(self.zeros.dtype).contiguous())
    else:
        raise ValueError(f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_mode}")
    if self.bias is not None:
        self.bias = gptq_module.bias.data.to(torch.float16).contiguous()


def prepare_model_for_bitblas_load(
        model,
        qcfg: QuantizeConfig,
        quant_linear_class,
        torch_dtype,
        model_save_name,
        device_map,
        sym: bool,
        desc_act: bool,
        load_checkpoint_in_model: bool,
):
    # The model (e.g. model.safetensors) is already serialized in the BitBLAS format, load it directly.
    if qcfg.format == FORMAT.BITBLAS:
        # if the checkpoint is already in bitblas format, we can load it directly.
        logger.info(f"Loading a GPTQ model, detected BitBLAS serialized format at {model_save_name}.")
        model = convert_to_bitblas(model, quant_linear_class, qcfg, sym, desc_act, repack=False)
        load_checkpoint_in_model_then_tie_weights(
            model,
            dtype=torch_dtype,
            checkpoint=model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True,
        )
    else:
        # Loading the GPTQ checkpoint to do the conversion.
        # TODO: Avoid loading the model with wrong QuantLinear, and directly use
        # BitBLAS ones. The repacking can be done directly on the safetensors, just
        # as for AWQ checkpoints.
        if not load_checkpoint_in_model:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=torch_dtype,
                checkpoint=model_save_name,
                device_map=device_map,
                offload_state_dict=True,
                offload_buffers=True,
            )
        # Convert model to bitblas, repacking weights into BitBLAS format.
        model = convert_to_bitblas(model, quant_linear_class, qcfg, sym, desc_act, repack=True)

        # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
        untie_weights(model)
    return model


@torch.no_grad()
def convert_to_bitblas(model, model_quantlinear, qcfg: QuantizeConfig, sym: bool, desc_act: bool, repack: bool):
    """
    Converts GPTQ-packed weights to the Bitblas format.

    Arguments:
        repack (`bool`):
            Whether to repack the qweights from `model` into the BitBLAS's QuantLinear layers.
    """
    if repack:
        message = "Repacking weights to be compatible with BitBLAS kernel..."
    else:
        # TODO: load directly BitBLAS QuantLinear.
        message = "Overriding QuantLinear layers to use BitBLAS's QuantLinear..."

    # TODO: need to benchmark to see multiple threads help with bitblas/tvm compilation and runtime
    with tctl.threadpool_limits(limits=1):
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

        # Note that due to tvm compilation of per layer modules shapes, the first layer loop is
        # relatively much slower if caching is not available. estimate time remaining is highly inaccurate
        for name, module in ProgressBar(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
            if not isinstance(module, model_quantlinear):
                continue

            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name[len(parent_name) + 1:]

            # We could use `torch.count_nonzero(module.bias) > 0` here to discard zero bias, but this has issues when loading weights
            # from checkpoints holding zero bias.
            with torch.device("meta"):
                bitblas_module = BitBLASQuantLinear(
                    bits=qcfg.bits,
                    group_size=qcfg.group_size,
                    sym=sym,
                    desc_act=desc_act,
                    infeatures=module.infeatures,
                    outfeatures=module.outfeatures,
                    pack_dtype=qcfg.pack_dtype,
                    bias=module.bias is not None,
                    enable_tuning=True
                )

            # convert to bitblas format
            if repack:
                repack_from_gptq_v2(bitblas_module, gptq_module=module)

            # Save to parent.
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, layer_name, bitblas_module)

            # Free cuda memory.
            del module
            torch_empty_cache()

    # Set quantization config to be BitBLAS.
    qcfg.runtime_format = FORMAT.BITBLAS

    return model
