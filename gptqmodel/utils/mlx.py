from ..quantization import QuantizeConfig, FORMAT
from transformers import PreTrainedModel
from ..nn_modules.qlinear.torch import TorchQuantLinear
import torch
import mlx.core as mx
from mlx_lm.utils import _get_classes, load_config, quantize_model, get_model_path

def convert_gptq_to_mlx_weights(model_id_or_path: str, gptq_model: PreTrainedModel, gptq_config: QuantizeConfig):
    if gptq_config["bits"] not in [2, 3, 4, 8]:
        raise ValueError("Model bits is not in [2,3,4,8]")

    if gptq_config["checkpoint_format"] not in [FORMAT.GPTQ, FORMAT.GPTQ_V2]:
        raise ValueError("Model checkpoint format is not gptq or gptq_v2")

    if gptq_config["desc_act"] == True:
        raise ValueError("desc_act=True is not supported")

    if gptq_config["dynamic"]:
        print(gptq_config["dynamic"])
        for _, config in gptq_config["dynamic"].items():
            if config != {}:
                if config["bits"] not in [2, 3, 4, 8]:
                    raise ValueError(f'Model bits {config["bits"]} in dynamic, it not in [2,3,4,8]')

                if config["desc_act"] == True:
                    raise ValueError("desc_act=True in dynamic, it not supported")

    config = load_config(get_model_path(model_id_or_path))

    # Initialize MLX model
    model_class, model_args_class = _get_classes(config=config)
    mlx_model = model_class(model_args_class.from_dict(config))

    # Convert weights
    weights = {}
    for name, module in gptq_model.named_modules():
        if isinstance(module, TorchQuantLinear):
            weights[f"{name}.weight"] = mx.array(
                module.dequantize_weight().T.detach().to("cpu", torch.float16).numpy()
            )

        elif hasattr(module, "weight") and (
                name != "lm_head" if config.get("tie_word_embeddings", False) else True):
            weights[f"{name}.weight"] = mx.array(
                module.weight.detach().to("cpu", torch.float16).numpy()
            )

        if hasattr(module, "bias"):
            if module.bias is not None:
                weights[f"{name}.bias"] = mx.array(
                    module.bias.detach().to("cpu", torch.float16).numpy()
                )

    # Load and quantize weights
    mlx_model.load_weights(list(weights.items()))
    weights, mlx_config = quantize_model(mlx_model, config, q_group_size=gptq_config["group_size"],
                                     q_bits=gptq_config["bits"])

    return weights, mlx_config
