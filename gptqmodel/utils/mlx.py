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
from ..nn_modules.qlinear.torch import TorchLinear
from ..nn_modules.qlinear.torch_awq import AwqTorchLinear
from ..quantization import FORMAT
from ..quantization.config import resolve_quant_format
from .logger import setup_logger
from .mlx_packing import repack_awq_4bit, repack_gptq_4bit
from .torch import torch_empty_cache


try:
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_lm import generate
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
        group_size = module.in_features if module.group_size == -1 else module.group_size
        if (module.bits != 4 or module.pack_dtype != torch.int32
                or group_size < 32 or group_size & (group_size - 1)
                or module.in_features % group_size or module.out_features % 8
                or module.in_features % 8 or module.adapter is not None):
            return None
        if isinstance(module, TorchLinear):
            if module.qzero_format() != 2 or module.planar:
                return None
            expected_g_idx = torch.arange(module.in_features, device=module.g_idx.device) // group_size
            if not torch.equal(module.g_idx, expected_g_idx):
                return None
        layer_params[name] = {"group_size": group_size, "bits": 4, "mode": "affine"}

    weights = {}
    tied_embeddings = config.get("tie_word_embeddings", False)
    for name, module in model.named_modules():
        if name in layer_params:
            qweight = module.qweight.detach().to("cpu").numpy()
            qzeros = module.qzeros.detach().to("cpu").numpy()
            scales = module.scales.detach().to("cpu", torch.float16).numpy()
            repack = repack_gptq_4bit if isinstance(module, TorchLinear) else repack_awq_4bit
            weight, scale, biases = repack(
                qweight, qzeros, scales, module.in_features, module.out_features
            )
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
    if "sampler" in kwargs:
        sampling_params["sampler"] = kwargs.pop("sampler", None)

    if "logits_processors" in kwargs:
        sampling_params["logits_processors"] = kwargs.pop("logits_processors", None)

    if "max_kv_size" in kwargs:
        sampling_params["max_kv_size"] = kwargs.pop("max_kv_size", None)

    if "prompt_cache" in kwargs:
        sampling_params["prompt_cache"] = kwargs.pop("prompt_cache", None)

    sampling_params["prefill_step_size"] = kwargs.pop("prefill_step_size", 512)

    if "kv_bits" in kwargs:
        sampling_params["kv_bits"] = kwargs.pop("kv_bits", None)

    sampling_params["kv_group_size"] = kwargs.pop("kv_group_size", 64)
    sampling_params["quantized_kv_start"] = kwargs.pop("quantized_kv_start", 0)

    if "sampler" in kwargs:
        sampling_params["prompt_progress_callback"] = kwargs.pop("prompt_progress_callback", None)

    if kwargs.pop("temp", None) is not None:
        sampling_params["temp"] = kwargs.pop("temp")
    elif kwargs.pop("temperature", None) is not None:
        sampling_params["temp"] = kwargs.pop("temperature")

    if "repetition_penalty" in kwargs:
        sampling_params["repetition_penalty"] = kwargs.pop("repetition_penalty", None)

    if "repetition_context_size" in kwargs:
        sampling_params["repetition_context_size"] = kwargs.pop("repetition_context_size", None)

    if "top_p" in kwargs:
        sampling_params["top_p"] = kwargs.pop("top_p", None)

    if "min_p" in kwargs:
        sampling_params["min_p"] = kwargs.pop("min_p", None)

    if "min_tokens_to_keep" in kwargs:
        sampling_params["min_tokens_to_keep"] = kwargs.pop("min_tokens_to_keep", None)

    return generate(model=model, tokenizer=tokenizer, prompt=prompt, verbose=verbose, **sampling_params)
