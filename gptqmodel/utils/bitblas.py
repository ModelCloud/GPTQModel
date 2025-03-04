# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
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

from ..nn_modules.qlinear.bitblas import BitBLASQuantLinear
from ..quantization import FORMAT, QuantizeConfig
from ..utils.logger import setup_logger
from .model import load_checkpoint_in_model_then_tie_weights
from .torch import torch_empty_cache

log = setup_logger()

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
        log.info(f"Loading a GPTQ model, detected BitBLAS serialized format at {model_save_name}.")
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
        if load_checkpoint_in_model:
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
        for name, module in log.pb(list(model.named_modules())).title(message):
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
                    in_features=module.in_features,
                    out_features=module.out_features,
                    pack_dtype=qcfg.pack_dtype,
                    bias=module.bias is not None,
                    enable_tuning=True,
                    adapter=qcfg.adapter,
                )

            # convert to bitblas format
            if repack:
                bitblas_module.repack_from_gptq(module)

            # Save to parent.
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, layer_name, bitblas_module)

            # Free cuda memory.
            del module
            torch_empty_cache()

    # Set quantization config to be BitBLAS.
    qcfg.runtime_format = FORMAT.BITBLAS

    return model
