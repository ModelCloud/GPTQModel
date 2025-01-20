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

import threadpoolctl as tctl
import torch
from accelerate.utils import find_tied_parameters

from ..nn_modules.qlinear.bitblas import BitBLASQuantLinear
from ..quantization import FORMAT, QuantizeConfig
from ..utils.logger import setup_logger
from .model import recurse_getattr, recurse_setattr, load_checkpoint_in_model_then_tie_weights
from .progress import ProgressBar
from .torch import torch_empty_cache

logger = setup_logger()

def prepare_model_for_bitblas_load(
        model,
        quantize_config: QuantizeConfig,
        quant_linear_class,
        torch_dtype,
        model_save_name,
        device_map,
        sym: bool,
        desc_act: bool,
        load_checkpoint_in_model: bool,
):
    # The model (e.g. model.safetensors) is already serialized in the BitBLAS format, load it directly.
    if quantize_config.format == FORMAT.BITBLAS:
        # if the checkpoint is already in bitblas format, we can load it directly.
        logger.info(f"Loading a GPTQ model, detected BitBLAS serialized format at {model_save_name}.")
        model = convert_to_bitblas(model, quant_linear_class, quantize_config, sym, desc_act, repack=False)
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
        model = convert_to_bitblas(model, quant_linear_class, quantize_config, sym, desc_act, repack=True)

        # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
        tied_params = find_tied_parameters(model)

        for weight_group in tied_params:
            for param_name in weight_group:
                if isinstance(recurse_getattr(model, param_name), torch.nn.Parameter):
                    recurse_setattr(
                        model,
                        param_name,
                        torch.nn.Parameter(recurse_getattr(model, param_name).clone()),
                    )
                else:
                    recurse_setattr(
                        model,
                        param_name,
                        recurse_getattr(model, param_name).clone(),
                    )
    return model


@torch.no_grad()
def convert_to_bitblas(model, model_quantlinear, quant_config: QuantizeConfig, sym: bool, desc_act: bool, repack: bool,
                       strict: bool = False):
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
                    bits=quant_config.bits,
                    group_size=quant_config.group_size,
                    sym=sym,
                    desc_act=desc_act,
                    infeatures=module.infeatures,
                    outfeatures=module.outfeatures,
                    bias=module.bias is not None,
                    enable_tuning=True
                )

            # Dequantize the weight.
            if repack:
                bitblas_module.repack_from_gptq(module)

            # Save to parent.
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, layer_name, bitblas_module)

            # Free cuda memory.
            del module
            torch_empty_cache()

    # Set quantization config to be BitBLAS.
    quant_config.runtime_format = FORMAT.BITBLAS

    return model
