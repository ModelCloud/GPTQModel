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

import torch
from accelerate.utils import find_tied_parameters

from ..nn_modules.qlinear.marlin import MarlinQuantLinear, _get_perms, unpack_qzeros
from ..quantization import FORMAT, QuantizeConfig
from ..utils.logger import setup_logger
from .model import recurse_getattr, recurse_setattr, load_checkpoint_in_model_then_tie_weights
from .progress import ProgressBar
from .rocm import IS_ROCM
from .torch import torch_empty_cache

logger = setup_logger()


def prepare_model_for_marlin_load(
    model,
    quantize_config: QuantizeConfig,
    quant_linear_class,
    torch_dtype,
    current_model_save_name,
    device_map,
    sym: bool,
    desc_act: bool,
    load_checkpoint_in_model: bool,
):
    # The model (e.g. model.safetensors) is already serialized in the Marlin format, load it directly.
    if quantize_config.format == FORMAT.MARLIN:
        model_save_name = current_model_save_name
        logger.info(f"Loading a GPTQ model, detected Marlin serialized format at {model_save_name}.")
        model = convert_to_marlin(model, quant_linear_class, quantize_config, sym, desc_act, repack=False)
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
        # Marlin ones. The repacking can be done directly on the safetensors, just
        # as for AWQ checkpoints.
        if not load_checkpoint_in_model:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=torch_dtype,  # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=current_model_save_name,
                device_map=device_map,
                offload_state_dict=True,
                offload_buffers=True,
            )
        # Convert model to marlin, repacking weights into Marlin format.
        model = convert_to_marlin(model, quant_linear_class, quantize_config, sym, desc_act, repack=True)

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


# Validate marlin support
def _validate_marlin_device_support() -> bool:
    """
    Validates if the current device is compatible for Marlin.
    ref: https://github.com/IST-DASLab/marlin?tab=readme-ov-file#requirements

    Returns:
        bool: indicates if CUDA device is compatible for Marlin
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and not IS_ROCM


# Adapted from https://github.com/rib-2/marlin/tree/conversion
def _validate_marlin_compatibility(cfg: QuantizeConfig, throw_error: bool = False):
    validate, err = MarlinQuantLinear.validate(bits=cfg.bits, group_size=cfg.group_size, desc_act=cfg.desc_act, sym=cfg.sym, dynamic=cfg.dynamic)
    if throw_error and err is not None:
        raise ValueError(err)
    return err


@torch.no_grad()
def convert_to_marlin(
    model, model_quantlinear, quantization_config: QuantizeConfig, sym: bool, desc_act: bool, repack: bool, strict: bool = False
):
    """
    Converts GPTQ-packed weights to the Marlin format. This assumes that the model already meets Marlin kernel constraints.

    Arguments:
        repack (`bool`):
            Whether to repack the qweights from `model` into the Marlin's QuantLinear layers.
    """
    if repack:
        message = "Repacking weights to be compatible with Marlin kernel"
    else:
        # TODO: load directly Marlin QuantLinear.
        message = "Overriding QuantLinear layers to use Marlin's QuantLinear"

    for name, module in ProgressBar(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
        if not isinstance(module, model_quantlinear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1 :]

        # We could use `torch.count_nonzero(module.bias) > 0` here to discard zero bias, but this has issues when
        # loading weights from checkpoints holding zero bias.
        with torch.device("meta"):
            new_module = MarlinQuantLinear(
                bits=4,
                group_size=module.group_size,
                sym=sym,
                desc_act=desc_act,
                infeatures=module.original_infeatures,
                outfeatures=module.original_outfeatures,
                bias=module.bias is not None,
            )

        # workspace is never in the state_dict, thus we need to allocate it manually.
        new_module.workspace = torch.zeros(new_module.outfeatures // 128 * 16, dtype=torch.int, device=module.device)

        # Dequantize the weight.
        if repack:
            import gptqmodel_marlin_cuda

            qweight = module.qweight
            if new_module.infeatures != new_module.original_infeatures or new_module.outfeatures != new_module.original_outfeatures:
                padded_qweight = torch.zeros((new_module.infeatures, new_module.outfeatures), dtype=torch.int, device=module.qweight.device)
                padded_qweight[:module.qweight.size(0), :module.qweight.size(1)] = qweight
                qweight = padded_qweight

            marlin_repacked_weight = gptqmodel_marlin_cuda.gptq_repack(qweight)

            if strict:
                dequantized_qzeros = unpack_qzeros(module.qzeros)

                if not torch.all(dequantized_qzeros == 8):
                    raise ValueError(
                        "Marlin kernel is compatible only with checkpoints using symmetric quantization."
                        "Found non-symmetric quantization for the weight {name}."
                    )

            _, _scale_perm, _scale_perm_single = _get_perms()

            s = module.scales.data.clone()

            if new_module.infeatures != new_module.original_infeatures or new_module.outfeatures != new_module.original_outfeatures:
                padded_s = torch.zeros((s.size(0), new_module.outfeatures), dtype=torch.half, device=s.device)
                padded_s[:s.size(0), :s.size(1)] = s
                s = padded_s

            if module.group_size != module.infeatures:
                s = s.reshape((1, -1))
                s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
            else:
                s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
            s = s.reshape((-1, new_module.outfeatures)).contiguous()

            new_module.B = marlin_repacked_weight
            new_module.s = s
            new_module.bias = module.bias

            new_module = new_module.to(module.device)

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)

        # Free cuda memory.
        del module
        if repack:
            del marlin_repacked_weight

        torch_empty_cache()

    # Set quantization config to be Marlin.
    quantization_config.runtime_format = FORMAT.MARLIN

    return model
