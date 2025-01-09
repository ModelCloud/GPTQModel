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
                
    # mlx does not support group_size = -1, 16, so we need to convert it to 64, 64 is the default group_size for mlx
    if gptq_config["group_size"] in [-1, 16]:
        gptq_config["group_size"] = 64

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

@torch.inference_mode()
def mlx_generate(model, tokenizer, **kwargs,):
    from mlx_lm import generate
    prompt = kwargs.pop("prompt", None)
    if prompt is None:
        raise ValueError("MLX requires prompts to be provided")

    verbose = kwargs.pop("verbose", False)
    formatter = kwargs.pop("formatter", None)

    sampling_params = {}
    sampling_params["max_tokens"] = kwargs.pop("max_tokens", 256)
    sampling_params["sampler"] = kwargs.pop("sampler", None)
    sampling_params["logits_processors"] = kwargs.pop("logits_processors", None)
    sampling_params["max_kv_size"] = kwargs.pop("max_kv_size", None)
    sampling_params["prompt_cache"] = kwargs.pop("prompt_cache", None)
    sampling_params["prefill_step_size"] = kwargs.pop("prefill_step_size", 512)
    sampling_params["kv_bits"] = kwargs.pop("kv_bits", None)
    sampling_params["kv_group_size"] = kwargs.pop("kv_group_size", 64)
    sampling_params["quantized_kv_start"] = kwargs.pop("quantized_kv_start", 0)
    sampling_params["prompt_progress_callback"] = kwargs.pop("prompt_progress_callback", None)
    if kwargs.pop("temp", None) is not None:
        sampling_params["temp"] = kwargs.pop("temp")
    elif kwargs.pop("temperature", None) is not None:
        sampling_params["temp"] = kwargs.pop("temperature")
    sampling_params["repetition_penalty"] = kwargs.pop("repetition_penalty", None)
    sampling_params["repetition_context_size"] = kwargs.pop("repetition_context_size", None)
    sampling_params["top_p"] = kwargs.pop("top_p", None)
    sampling_params["min_p"] = kwargs.pop("min_p", None)
    sampling_params["min_tokens_to_keep"] = kwargs.pop("min_tokens_to_keep", None)

    return generate(model=model, tokenizer=tokenizer, prompt=prompt, formatter=formatter ,verbose=verbose, **sampling_params)