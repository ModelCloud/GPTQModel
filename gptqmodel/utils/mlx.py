# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from copy import deepcopy
from pathlib import Path
from typing import Union

import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel

from ..models import BaseQModel
from ..nn_modules.qlinear.mlx import AwqMlxQuantLinear, MlxQuantLinear
from ..nn_modules.qlinear.torch import TorchLinear
from ..nn_modules.qlinear.torch_awq import AwqTorchLinear
from ..quantization import FORMAT
from ..quantization.config import resolve_quant_format
from .logger import setup_logger
from .torch import torch_empty_cache


try:
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from mlx_lm.utils import _get_classes, load_config, quantize_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

log = setup_logger()


def _packed_mlx_weights(model, config, lm_head_name):
    """Copy supported 4-bit layers without expanding and requantizing weights."""
    quantized = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (TorchLinear, AwqTorchLinear))]
    if not quantized:
        return None

    layer_params = {}
    for name, module in quantized:
        mlx_linear = MlxQuantLinear if isinstance(module, TorchLinear) else AwqMlxQuantLinear
        if not mlx_linear.source_compatible(module):
            return None
        group_size = module.in_features if module.requested_group_size == -1 else module.group_size
        layer_params[name] = {"group_size": group_size, "bits": 4, "mode": "affine"}

    weights = {}
    tied_embeddings = config.get("tie_word_embeddings", False)
    for name, module in model.named_modules():
        if name in layer_params:
            mlx_linear = MlxQuantLinear if isinstance(module, TorchLinear) else AwqMlxQuantLinear
            weight, scale, biases, _ = mlx_linear.pack_source(module)
            weights[f"{name}.weight"] = mx.array(weight)
            weights[f"{name}.scales"] = mx.array(scale)
            weights[f"{name}.biases"] = mx.array(biases)
        elif hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            # Tied embedding weights are supplied by the embedding module.
            if tied_embeddings and name == lm_head_name:
                continue
            weights[f"{name}.weight"] = mx.array(
                module.weight.detach().to("cpu", torch.float16).numpy()
            )
        if getattr(module, "bias", None) is not None:
            weights[f"{name}.bias"] = mx.array(
                module.bias.detach().to("cpu", torch.float16).numpy()
            )

    mlx_config = deepcopy(config)
    default = next(iter(layer_params.values()))
    mlx_config["quantization"] = dict(default)
    for name, params in layer_params.items():
        if params != default:
            mlx_config["quantization"][name] = params
    mlx_config["quantization_config"] = mlx_config["quantization"]

    model_class, model_args_class = _get_classes(config=mlx_config)
    mlx_model = model_class(model_args_class.from_dict(mlx_config))
    found = set()

    def predicate(path, module):
        if path in layer_params and hasattr(module, "to_quantized"):
            found.add(path)
            return layer_params[path]
        return False

    nn.quantize(mlx_model, group_size=default["group_size"], bits=4,
                class_predicate=predicate)
    if found != set(layer_params):
        log.warn("MLX packed layer names do not match the model; using float conversion.")
        return None
    mlx_model.load_weights(list(weights.items()))
    return mlx_model, mlx_config


def convert_gptq_to_mlx_weights(model_id_or_path: str, model: Union[PreTrainedModel, BaseQModel], gptq_config: dict, lm_head_name: str):
    if not MLX_AVAILABLE:
        raise ValueError("MLX not installed. Please install via `pip install gptqmodel[mlx] --no-build-isolation`.")

    # Keep conversion on CPU while restoring MLX's GPU default for inference.
    with mx.stream(mx.cpu):
        return _convert_gptq_to_mlx_weights(model_id_or_path, model, gptq_config, lm_head_name)


def _convert_gptq_to_mlx_weights(model_id_or_path, model, gptq_config, lm_head_name):

    if gptq_config["bits"] not in [2, 3, 4, 8]:
        raise ValueError("Model bits is not in [2,3,4,8]")

    quant_format = resolve_quant_format(gptq_config.get("format"), gptq_config.get("method", gptq_config.get("quant_method")))
    if quant_format not in [FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.GEMM]:
        raise ValueError("MLX conversion requires GPTQ, GPTQ_V2, or AWQ GEMM format")

    if gptq_config.get("dynamic") is not None:
        print(gptq_config["dynamic"])
        for _, config in gptq_config["dynamic"].items():
            if config != {}:
                if config["bits"] not in [2, 3, 4, 8]:
                    raise ValueError(f'Model bits {config["bits"]} in dynamic, it not in [2,3,4,8]')

    model_path = Path(model_id_or_path)
    if not model_path.exists():
        model_path = Path(snapshot_download(
            model_id_or_path, allow_patterns=["config.json", "generation_config.json"]
        ))
    config = load_config(model_path)

    if isinstance(model, BaseQModel):
        model = model.model

    packed = _packed_mlx_weights(model, config, lm_head_name)
    if packed is not None:
        log.info("MLX: transferred packed 4-bit weights without requantization")
        return packed

    # Requantization needs an MLX-supported group size.
    if gptq_config["group_size"] in [-1, 16]:
        gptq_config["group_size"] = 64

    # Convert weights
    weights = {}
    n = 1
    pb = log.pb(list(model.named_modules())).title("Format: Converting to mlx ->").manual()
    for name, module in pb:
        pb.subtitle(f"{name}").draw()
        if isinstance(module, (TorchLinear, AwqTorchLinear)):
            if isinstance(module, AwqTorchLinear):
                from ..quantization.awq.utils.packing_utils import dequantize_gemm
                dequantized = dequantize_gemm(
                    module.qweight, module.qzeros, module.scales,
                    module.bits, module.group_size,
                )
            else:
                dequantized = module.dequantize_weight()
            weights[f"{name}.weight"] = mx.array(
                dequantized.T.detach().to("cpu", torch.float16).numpy()
            )

            if isinstance(module, TorchLinear):
                module._empty_gptq_only_weights()

            if n % 10 == 0:
                # Below saves memory but also make each iter slower: test call every N loop
                torch_empty_cache()

            n += 1
        # Handle normal layers with weight (exclude lm_head if embeddings tied)
        elif hasattr(module, "weight") and not (config["tie_word_embeddings"] and name == lm_head_name):
            weights[f"{name}.weight"] = mx.array(
                module.weight.detach().to("cpu", torch.float16).numpy()
            )

            n += 1

        if hasattr(module, "bias"):
            if module.bias is not None:
                weights[f"{name}.bias"] = mx.array(
                    module.bias.detach().to("cpu", torch.float16).numpy()
                )

    del model.model
    torch_empty_cache()

    # Initialize MLX model
    model_class, model_args_class = _get_classes(config=config)
    mlx_model = model_class(model_args_class.from_dict(config))

    # Load and quantize weights
    log.info("Starting MLX quantization...")
    mlx_model.load_weights(list(weights.items()))
    weights, mlx_config = quantize_model(mlx_model, config, group_size=gptq_config["group_size"],
                                     bits=gptq_config["bits"])
    log.info("MLX quantization completed")

    return weights, mlx_config

@torch.inference_mode()
def mlx_generate(model, tokenizer, **kwargs,):
    if not MLX_AVAILABLE:
        raise ValueError("MLX not installed. Please install via `pip install gptqmodel[mlx] --no-build-isolation`.")

    prompt = kwargs.pop("prompt", None)
    if prompt is None:
        raise ValueError("MLX requires prompts to be provided")

    verbose = kwargs.pop("verbose", False)
    kwargs.pop("formatter", None)

    sampling_params = {}
    sampling_params["max_tokens"] = kwargs.pop("max_tokens", 256)
    sampler = kwargs.pop("sampler", None)
    temperature = kwargs.pop("temp", None)
    if temperature is None:
        temperature = kwargs.pop("temperature", None)
    top_p = kwargs.pop("top_p", None)
    min_p = kwargs.pop("min_p", None)
    min_tokens_to_keep = kwargs.pop("min_tokens_to_keep", None)
    if sampler is None and any(value is not None for value in (temperature, top_p, min_p, min_tokens_to_keep)):
        sampler = make_sampler(
            temp=0.0 if temperature is None else temperature,
            top_p=0.0 if top_p is None else top_p,
            min_p=0.0 if min_p is None else min_p,
            min_tokens_to_keep=1 if min_tokens_to_keep is None else min_tokens_to_keep,
        )
    if sampler is not None:
        sampling_params["sampler"] = sampler

    logits_processors = kwargs.pop("logits_processors", None)
    repetition_penalty = kwargs.pop("repetition_penalty", None)
    repetition_context_size = kwargs.pop("repetition_context_size", 20)
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits_processors = list(logits_processors or []) + make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
    if logits_processors is not None:
        sampling_params["logits_processors"] = logits_processors

    if "max_kv_size" in kwargs:
        sampling_params["max_kv_size"] = kwargs.pop("max_kv_size", None)

    if "prompt_cache" in kwargs:
        sampling_params["prompt_cache"] = kwargs.pop("prompt_cache", None)

    sampling_params["prefill_step_size"] = kwargs.pop("prefill_step_size", 512)

    if "kv_bits" in kwargs:
        sampling_params["kv_bits"] = kwargs.pop("kv_bits", None)

    sampling_params["kv_group_size"] = kwargs.pop("kv_group_size", 64)
    sampling_params["quantized_kv_start"] = kwargs.pop("quantized_kv_start", 0)

    if "prompt_progress_callback" in kwargs:
        sampling_params["prompt_progress_callback"] = kwargs.pop("prompt_progress_callback", None)

    return generate(model=model, tokenizer=tokenizer, prompt=prompt, verbose=verbose, **sampling_params)
