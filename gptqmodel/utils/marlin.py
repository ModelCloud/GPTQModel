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

import torch

from ..nn_modules.qlinear.marlin import MarlinQuantLinear, _get_perms
from ..quantization import FORMAT, QuantizeConfig
from ..utils.logger import setup_logger
from .model import load_checkpoint_in_model_then_tie_weights
from .progress import ProgressBar
from .rocm import IS_ROCM
from .torch import torch_empty_cache

logger = setup_logger()

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
    validate, err = MarlinQuantLinear.validate(bits=cfg.bits, group_size=cfg.group_size, desc_act=cfg.desc_act, sym=cfg.sym, pack_dtype=cfg.pack_dtype, dynamic=cfg.dynamic)
    if throw_error and err is not None:
        raise ValueError(err)
    return err


@torch.no_grad()
def convert_to_marlin(
    model, model_quantlinear, qcfg: QuantizeConfig, sym: bool, desc_act: bool, repack: bool
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

    for name, module in ProgressBar(model.named_modules(), info=message, total=len(list(model.named_modules()))):
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
                in_features=module.original_in_features,
                out_features=module.original_out_features,
                pack_dtype=module.pack_dtype,
                bias=module.bias is not None,
            )

        # workspace is never in the state_dict, thus we need to allocate it manually.
        new_module.workspace = torch.zeros(new_module.out_features // 128 * 16, dtype=module.pack_dtype, device=module.device)

        # Dequantize the weight.
        if repack:
            import gptqmodel_marlin_cuda

            qweight = module.qweight
            # if new_module.in_features != new_module.original_in_features or new_module.out_features != new_module.original_out_features:
            #     padded_qweight = torch.zeros((new_module.in_features, new_module.out_features), dtype=torch.int, device=module.qweight.device)
            #     padded_qweight[:module.qweight.size(0), :module.qweight.size(1)] = qweight
            #     qweight = padded_qweight

            marlin_repacked_weight = gptqmodel_marlin_cuda.gptq_repack(qweight)

            # if strict:
            #     dequantized_qzeros = unpack_qzeros(module.qzeros)
            #
            #     if not torch.all(dequantized_qzeros == 8):
            #         raise ValueError(
            #             "Marlin kernel is compatible only with checkpoints using symmetric quantization."
            #             "Found non-symmetric quantization for the weight {name}."
            #         )

            _, _scale_perm, _scale_perm_single = _get_perms()

            s = module.scales.data.clone()

            if new_module.in_features != new_module.original_in_features or new_module.out_features != new_module.original_out_features:
                padded_s = torch.zeros((s.size(0), new_module.out_features), dtype=torch.half, device=s.device)
                padded_s[:s.size(0), :s.size(1)] = s
                s = padded_s

            if module.group_size != module.in_features:
                s = s.reshape((1, -1))
                s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
            else:
                s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
            s = s.reshape((-1, new_module.out_features)).contiguous()

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
    qcfg.runtime_format = FORMAT.MARLIN

    return model
