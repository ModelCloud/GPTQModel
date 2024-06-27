import gc
from logging import getLogger

import accelerate
import torch
from accelerate.utils import find_tied_parameters
from tqdm import tqdm

from ..nn_modules.qlinear.qlinear_bitblas import QuantLinear as BitBLASQuantLinear
from ..quantization import FORMAT, QuantizeConfig
from .model import recurse_getattr, recurse_setattr

logger = getLogger(__name__)


def prepare_model_for_bitblas_load(
        model,
        quantize_config: QuantizeConfig,
        quant_linear_class,
        torch_dtype,
        model_save_name,
        device_map,
        sym: bool,
        desc_act: bool,
        converted_gptq_v1_to_v2: bool,
):
    # The model (e.g. model.safetensors) is already serialized in the BitBLAS format, load it directly.
    if quantize_config.format == FORMAT.BITBLAS:
        # if the checkpoint is already in bitblas format, we can load it directly.
        logger.info(f"Loading a GPTQ model, detected BitBLAS serialized format at {model_save_name}.")
        model = convert_to_bitblas(model, quant_linear_class, quantize_config, sym, desc_act, repack=False)
        accelerate.utils.modeling.load_checkpoint_in_model(
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
        if not converted_gptq_v1_to_v2:
            accelerate.load_checkpoint_in_model(
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
def convert_to_bitblas(model, model_quantlinear, quantization_config: QuantizeConfig, sym: bool, desc_act: bool, repack: bool,
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

    for name, module in tqdm(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
        if not isinstance(module, model_quantlinear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1:]

        # We could use `torch.count_nonzero(module.bias) > 0` here to discard zero bias, but this has issues when loading weights
        # from checkpoints holding zero bias.
        with torch.device("meta"):
            bitblas_module = BitBLASQuantLinear(
                bits=quantization_config.bits,
                group_size=quantization_config.group_size,
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
        gc.collect()

    # Set quantization config to be BitBLAS.
    quantization_config.format = FORMAT.BITBLAS

    return model
