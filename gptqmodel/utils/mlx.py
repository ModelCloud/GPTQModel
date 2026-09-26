# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# MLX-LM loader reference: Apple Inc. and MLX-LM contributors (MIT).
# Quantization format references: ParoQuant, QQQ, GGUF, bitsandbytes, and EXL3;
# format-specific credit and licenses are recorded in their converter modules.
# Native MXFP8 matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx

from copy import deepcopy
from pathlib import Path
from typing import Union

import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel

from ..models import BaseQModel
from ..nn_modules.qlinear.mlx import (AwqGemvFastMlxQuantLinear, AwqGemvMlxQuantLinear,
                                      AwqMlxQuantLinear, BitsAndBytesMlxQuantLinear,
                                      FP8MlxQuantLinear, GGUFMlxQuantLinear,
                                      LLMAwqMlxQuantLinear, MlxQuantLinear,
                                      ParoMlxQuantLinear, QQQMlxQuantLinear)
from ..nn_modules.qlinear.paroquant import ParoLinear
from ..nn_modules.qlinear.qqq import QQQTorchLinear
from ..nn_modules.qlinear.gguf import GGUFTorchLinear
from ..nn_modules.qlinear.fp8 import TorchFP8Linear
from ..nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from ..nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from ..nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from ..nn_modules.exllamav3_torch import ExllamaV3TorchLinear
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
    from mlx.utils import tree_map_with_path

    from ..nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
    from ..nn_modules.qlinear.mlx_gptq import MlxGPTQLinear
    from ..nn_modules.qlinear.mlx_awq import MlxAWQLinear
    from ..nn_modules.qlinear.mlx_fp8 import MlxFP8DenseLinear, MlxFP8Linear
    from ..nn_modules.qlinear.mlx_gguf import MlxGGUFLinear, MlxGGUFQ6KLinear
    from ..nn_modules.qlinear.mlx_bitsandbytes import MlxBitsAndBytesLinear
    from ..nn_modules.qlinear.mlx_exl3 import MlxEXL3Linear
    from ..nn_modules.qlinear.mlx_paro import MlxParoLinear
    from ..nn_modules.qlinear.mlx_qqq import MlxQQQLinear
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

log = setup_logger()


def _mlx_holder_class(module):
    if isinstance(module, ParoLinear):
        return ParoMlxQuantLinear
    if isinstance(module, QQQTorchLinear):
        return QQQMlxQuantLinear
    if isinstance(module, GGUFTorchLinear):
        return GGUFMlxQuantLinear
    if isinstance(module, TorchFP8Linear):
        return FP8MlxQuantLinear
    if isinstance(module, BitsAndBytesLinear):
        return BitsAndBytesMlxQuantLinear
    if isinstance(module, LLMAwqLinear):
        return LLMAwqMlxQuantLinear
    if isinstance(module, AwqGEMVFastLinear):
        return AwqGemvFastMlxQuantLinear
    if isinstance(module, AwqGEMVLinear):
        return AwqGemvMlxQuantLinear
    if isinstance(module, TorchLinear):
        return MlxQuantLinear
    return AwqMlxQuantLinear


def _packed_mlx_weights(model, config, lm_head_name):
    """Transfer exact packed layers or decoded weight-only layers to MLX."""
    quantized = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (TorchLinear, AwqTorchLinear, QQQTorchLinear,
                                        GGUFTorchLinear, TorchFP8Linear, BitsAndBytesLinear,
                                        ExllamaV3TorchLinear, AwqGEMVLinear,
                                        AwqGEMVFastLinear))]
    if not quantized:
        return None

    layer_params = {}
    group16 = set()
    gguf_dtype = set()
    gguf_q6_k = set()
    bitsandbytes_native = {}
    exl3_dtype = set()
    gptq_dtype = set()
    awq_dtype = set()
    paro = {}
    qqq = {}
    fp8_native = {}
    fp8_dense = set()
    dense = {}
    for name, module in quantized:
        if isinstance(module, ExllamaV3TorchLinear):
            if module.in_features <= 0 or module.out_features <= 0 or getattr(module, "trellis", None) is None:
                raise ValueError(f"EXL3 layer {name} cannot be decoded for MLX")
            dense[name] = module
            exl3_dtype.add(name)
            continue
        mlx_linear = _mlx_holder_class(module)
        if not mlx_linear.source_compatible(module):
            if isinstance(module, (ParoLinear, QQQTorchLinear, GGUFTorchLinear,
                                   TorchFP8Linear, BitsAndBytesLinear, AwqGEMVLinear,
                                   AwqGEMVFastLinear)):
                raise ValueError(f"{type(module).__name__} layer {name} cannot be transferred to MLX")
            return None
        if isinstance(module, TorchFP8Linear) and mlx_linear.native_compatible(module):
            fp8_native[name] = module
        elif isinstance(module, TorchFP8Linear):
            fp8_dense.add(name)
            dense[name] = module
            continue
        elif isinstance(module, BitsAndBytesLinear):
            bitsandbytes_native[name] = BitsAndBytesMlxQuantLinear.native_payload(module)
            continue
        layer_params[name] = mlx_linear.mlx_params(module)
        if isinstance(module, ParoLinear):
            paro[name] = module
        if isinstance(module, QQQTorchLinear):
            qqq[name] = module
        if (isinstance(module, GGUFTorchLinear) and module.gguf_tensor_qtype == "Q6_K") or (
                not isinstance(module, (QQQTorchLinear, GGUFTorchLinear, TorchFP8Linear))
                and module.group_size == 16):
            group16.add(name)
            if isinstance(module, GGUFTorchLinear) and module.gguf_tensor_qtype == "Q6_K":
                gguf_q6_k.add(name)
        elif isinstance(module, TorchLinear):
            gptq_dtype.add(name)
        elif (isinstance(module, (AwqTorchLinear, AwqGEMVLinear, AwqGEMVFastLinear, LLMAwqLinear))
              and not isinstance(module, ParoLinear)):
            awq_dtype.add(name)
        elif isinstance(module, GGUFTorchLinear):
            gguf_dtype.add(name)

    weights = {}
    tied_embeddings = config.get("tie_word_embeddings", False)
    for name, module in model.named_modules():
        if name in bitsandbytes_native:
            payload = bitsandbytes_native[name]
            affine = payload.get("affine_biases") is not None
            weights[f"{name}.weight"] = mx.array(payload["weight"]) if affine else mx.array(payload["weight"]).reshape(-1)
            weights[f"{name}.scales"] = (mx.array(payload["scales"]).astype(mx.float32)
                                               if affine else mx.array(payload["scales"]).astype(mx.float32).reshape(-1))
            weights[f"{name}.bias"] = mx.zeros((module.out_features,), dtype=mx.float32) if payload["bias"] is None else mx.array(payload["bias"]).astype(mx.float32)
            if payload["codebook"] is not None:
                weights[f"{name}.codebook"] = mx.array(payload["codebook"]).astype(mx.float32).reshape(-1)
            if payload.get("affine_biases") is not None:
                weights[f"{name}.affine_biases"] = mx.array(payload["affine_biases"]).astype(mx.float32)
        elif name in dense:
            dense_weight = (module.get_weight_tensor(dtype=torch.float16).T.contiguous()
                            if isinstance(module, ExllamaV3TorchLinear)
                            else _mlx_holder_class(module).dense_weight(module))
            weights[f"{name}.linear.weight" if name in fp8_dense or name in exl3_dtype else f"{name}.weight"] = mx.array(
                dense_weight.detach().cpu().numpy()
            )
        elif name in layer_params:
            mlx_linear = _mlx_holder_class(module)
            weight, scale, biases, _ = mlx_linear.pack_source(module)
            prefix = f"{name}.linear" if name in paro or name in qqq or name in fp8_native or name in gptq_dtype or name in awq_dtype or name in gguf_dtype else name
            weights[f"{prefix}.weight"] = mx.array(weight)
            if name in fp8_native:
                weights[f"{name}.output_scale"] = mx.array(biases).astype(mx.float32)
                weights[f"{prefix}.scales"] = mx.array(scale)
            if name in paro:
                weights[f"{name}.channel_scales"] = mx.array(
                    module.channel_scales.detach().to("cpu", torch.float16).numpy()
                )
            if name in qqq:
                _, channel_scale = module._dequantize_weight_for_torch()
                weights[f"{name}.channel_scale"] = mx.array(
                    channel_scale.detach().to("cpu", torch.float32).numpy()
                )
            if name in group16:
                weights[f"{prefix}.scales_even"] = mx.array(scale[:, ::2]).astype(mx.float32)
                weights[f"{prefix}.scales_odd"] = mx.array(scale[:, 1::2]).astype(mx.float32)
                weights[f"{prefix}.biases_even"] = mx.array(biases[:, ::2])
                weights[f"{prefix}.biases_odd"] = mx.array(biases[:, 1::2])
            elif name not in fp8_native:
                weights[f"{prefix}.scales"] = mx.array(scale)
                if biases is not None:
                    weights[f"{prefix}.biases"] = mx.array(biases)
        elif hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            # Tied embedding weights are supplied by the embedding module.
            if tied_embeddings and name == lm_head_name:
                continue
            weights[f"{name}.weight"] = mx.array(
                module.weight.detach().to("cpu", torch.float16).numpy()
            )
        if getattr(module, "bias", None) is not None and name not in bitsandbytes_native:
            prefix = f"{name}.linear" if name in paro or name in fp8_dense or name in gptq_dtype or name in awq_dtype or name in gguf_dtype or name in exl3_dtype else name
            weights[f"{prefix}.bias"] = mx.array(
                module.bias.detach().to("cpu", torch.float16).numpy()
            )
            if name in qqq:
                weights[f"{name}.linear.bias"] = mx.zeros((module.out_features,), dtype=mx.float16)

    mlx_config = deepcopy(config)
    mlx_config.pop("quantization", None)
    mlx_config.pop("quantization_config", None)
    if layer_params:
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
            return False if path in group16 or path in fp8_native else layer_params[path]
        return False

    if layer_params:
        nn.quantize(mlx_model, group_size=default["group_size"], bits=default["bits"],
                    class_predicate=predicate)
    if found != set(layer_params):
        log.warn("MLX packed layer names do not match the model; using float conversion.")
        return None
    if group16:
        def replace_group16(path, module):
            if path not in group16:
                return module
            if not isinstance(module, nn.Linear):
                raise ValueError(f"MLX group-16 layer {path} is not a linear module")
            output_dims, input_dims = module.weight.shape
            if path in gguf_q6_k:
                return MlxGGUFQ6KLinear(
                    input_dims, output_dims, bias=module.get("bias") is not None,
                )
            return MlxGroup16Linear(input_dims, output_dims, layer_params[path]["bits"],
                                    bias=module.get("bias") is not None)

        mlx_model.update_modules(tree_map_with_path(
            replace_group16, mlx_model.leaf_modules(), is_leaf=nn.Module.is_module,
        ))
    if paro or qqq or fp8_native or fp8_dense or gptq_dtype or awq_dtype or gguf_dtype or bitsandbytes_native or exl3_dtype:
        def replace_custom(path, module):
            if path in paro:
                source = paro[path]
                return MlxParoLinear(
                    module,
                    source.pairs.detach().cpu().numpy(),
                    source.theta.detach().cpu().numpy(),
                    source.channel_scales.detach().cpu().numpy(),
                    source.group_size,
                )
            if path in gptq_dtype:
                return MlxGPTQLinear(module)
            if path in awq_dtype:
                return MlxAWQLinear(module)
            if path in gguf_dtype:
                return MlxGGUFLinear(module)
            if path in bitsandbytes_native:
                return MlxBitsAndBytesLinear(**bitsandbytes_native[path])
            if path in exl3_dtype:
                return MlxEXL3Linear(module)
            if path in fp8_dense:
                return MlxFP8DenseLinear(module)
            if path in fp8_native:
                source = fp8_native[path]
                return MlxFP8Linear(
                    source.in_features, source.out_features,
                    (1.0 / source.weight_scale_inv.detach().float()).cpu().numpy(),
                    None if source.bias is None else source.bias.detach().cpu().numpy(),
                )
            if path in qqq:
                _, channel_scale = qqq[path]._dequantize_weight_for_torch()
                bias = qqq[path].bias
                return MlxQQQLinear(
                    module, channel_scale.detach().cpu().numpy(),
                    None if bias is None else bias.detach().cpu().numpy(),
                )
            return module

        mlx_model.update_modules(tree_map_with_path(
            replace_custom, mlx_model.leaf_modules(), is_leaf=nn.Module.is_module,
        ))
    mlx_model.load_weights(list(weights.items()))
    if group16 or paro or qqq or fp8_native or fp8_dense or gptq_dtype or awq_dtype or gguf_dtype or bitsandbytes_native or exl3_dtype or (dense and layer_params):
        # MLX-LM's standard loader reconstructs only native QuantizedLinear.
        # Keep this runtime model instead of round-tripping through that loader.
        mlx_config["_gptqmodel_group16_runtime" if group16 and not paro and not qqq and not dense
                   else "_gptqmodel_custom_mlx_runtime"] = True
    return mlx_model, mlx_config


def convert_gptq_to_mlx_weights(model_id_or_path: str, model: Union[PreTrainedModel, BaseQModel], gptq_config: dict, lm_head_name: str):
    if not MLX_AVAILABLE:
        raise ValueError("MLX not installed. Please install via `pip install gptqmodel[mlx] --no-build-isolation`.")

    # Keep conversion on CPU while restoring MLX's GPU default for inference.
    with mx.stream(mx.cpu):
        return _convert_gptq_to_mlx_weights(model_id_or_path, model, gptq_config, lm_head_name)


def _convert_gptq_to_mlx_weights(model_id_or_path, model, gptq_config, lm_head_name):

    quant_format = resolve_quant_format(gptq_config.get("format"), gptq_config.get("method", gptq_config.get("quant_method")))
    if quant_format != FORMAT.EXL3 and gptq_config["bits"] not in [1, 2, 3, 4, 5, 6, 7, 8]:
        raise ValueError("MLX conversion supports 1 through 8 integer bits for these formats")
    if quant_format not in [FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.GPTQ_P, FORMAT.GEMM,
                            FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ,
                            FORMAT.PAROQUANT, FORMAT.QQQ, FORMAT.GGUF, FORMAT.FP8,
                            FORMAT.BITSANDBYTES, FORMAT.EXL3]:
        raise ValueError("MLX conversion requires a supported GPT-QModel quantization format")

    if gptq_config.get("dynamic") is not None:
        print(gptq_config["dynamic"])
        for _, config in gptq_config["dynamic"].items():
            if config != {}:
                if config["bits"] not in [2, 3, 4, 5, 6, 7, 8]:
                    raise ValueError(f'MLX GPTQ conversion does not support {config["bits"]} bits in dynamic config')

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
        log.info("MLX: transferred source weights without requantization")
        return packed
    if quant_format in (FORMAT.PAROQUANT, FORMAT.QQQ, FORMAT.GGUF, FORMAT.FP8,
                        FORMAT.BITSANDBYTES, FORMAT.EXL3, FORMAT.GEMV,
                        FORMAT.GEMV_FAST, FORMAT.LLM_AWQ):
        raise ValueError(f"{quant_format} MLX inference requires transferable packed weights and runtime state")

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
