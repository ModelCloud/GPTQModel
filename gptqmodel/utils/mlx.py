# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# AWQ reference: MIT Han Lab, MIT License, https://github.com/mit-han-lab/llm-awq
# FP8 encoding layouts: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
# Marlin format: IST-DASLab contributors, MIT, https://github.com/IST-DASLab/marlin
# BitBLAS format: Microsoft Research contributors, Apache-2.0, https://github.com/microsoft/BitBLAS
# Quantization format references: Paro, QQQ, GGUF, bitsandbytes, and EXL3;
# format-specific credit and licenses are recorded in their converter modules.

from copy import copy
from pathlib import Path
from typing import Union

import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel

from ..models import BaseQModel
from ..nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from ..nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from ..nn_modules.qlinear.fp8 import TorchFP8Linear
from ..nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from ..nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from ..nn_modules.qlinear.gguf import GGUFTorchLinear
from ..nn_modules.qlinear.macaq import MacaqLinear
from ..nn_modules.qlinear.mlx import (
    AWQBitBLASMlxQuantLinear,
    AwqGemvFastMlxQuantLinear,
    AwqGemvMlxQuantLinear,
    AWQMarlinMlxQuantLinear,
    AwqMlxQuantLinear,
    BitsAndBytesMlxQuantLinear,
    FP8MlxQuantLinear,
    GGUFMlxQuantLinear,
    GPTQBitBLASMlxQuantLinear,
    GPTQMarlinMlxQuantLinear,
    LLMAwqMlxQuantLinear,
    MacaqMlxQuantLinear,
    MlxQuantLinear,
    NF12MlxQuantLinear,
    ParoMlxQuantLinear,
    QQQMlxQuantLinear,
)
from ..nn_modules.qlinear.nf12 import NF12Linear
from ..nn_modules.qlinear.paro import ParoLinear
from ..nn_modules.qlinear.qqq import QQQTorchLinear
from ..nn_modules.qlinear.torch import TorchLinear
from ..nn_modules.qlinear.torch_awq import AwqTorchLinear
from ..quantization import FORMAT
from ..quantization.config import dynamic_get, resolve_quant_format
from .logger import setup_logger
from .mlx_rope import _mlx_model_config
from .torch import torch_empty_cache


try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map_with_path
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.utils import _get_classes, load_config, quantize_model

    from ..nn_modules.qlinear.mlx_awq import MlxAWQGroup16Linear, MlxAWQLinear
    from ..nn_modules.qlinear.mlx_bitsandbytes import MlxBitsAndBytesLinear
    from ..nn_modules.qlinear.mlx_exl3 import MlxEXL3Linear
    from ..nn_modules.qlinear.mlx_fp8 import MlxFP8Linear, MlxFP8PackedLinear
    from ..nn_modules.qlinear.mlx_gguf import MlxGGUFLinear, MlxGGUFQ6KLinear
    from ..nn_modules.qlinear.mlx_gptq import MlxGPTQGroup16Linear, MlxGPTQLinear
    from ..nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
    from ..nn_modules.qlinear.mlx_macaq import MlxMACaQLinear
    from ..nn_modules.qlinear.mlx_nf12 import MlxNF12Linear
    from ..nn_modules.qlinear.mlx_paro import MlxParoLinear
    from ..nn_modules.qlinear.mlx_qqq import MlxQQQLinear
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

log = setup_logger()

# Torch storage dtypes that reach the MLX kernels through a payload.
_MLX_DTYPES = {
    torch.float16: mx.float16,
    torch.bfloat16: mx.bfloat16,
    torch.float32: mx.float32,
}


def _mlx_dtype(dtype):
    return _MLX_DTYPES[dtype]


def _exl3_signs(module, unpacked_name, packed_name, size):
    """Return EXL3 signs, including the legacy packed-bit representation."""
    unpacked = getattr(module, unpacked_name, None)
    if unpacked is not None:
        return unpacked.detach().to("cpu", torch.float16).reshape(-1)[:size].numpy()
    packed = getattr(module, packed_name, None)
    if packed is None:
        raise ValueError(f"EXL3 layer {module.name} is missing {unpacked_name}/{packed_name}")
    words = packed.detach().to("cpu").view(torch.uint16).to(torch.int32).reshape(-1)
    masks = 1 << torch.arange(16, dtype=torch.int32)
    negative = (words.unsqueeze(-1) & masks) != 0
    return torch.where(negative, -1.0, 1.0).reshape(-1)[:size].to(torch.float16).numpy()


def _mlx_holder_class(module):
    if isinstance(module, AWQBitBLASMlxQuantLinear):
        return AWQBitBLASMlxQuantLinear
    if isinstance(module, AWQMarlinMlxQuantLinear):
        return AWQMarlinMlxQuantLinear
    if isinstance(module, GPTQBitBLASMlxQuantLinear):
        return GPTQBitBLASMlxQuantLinear
    if isinstance(module, GPTQMarlinMlxQuantLinear):
        return GPTQMarlinMlxQuantLinear
    if isinstance(module, MacaqLinear):
        return MacaqMlxQuantLinear
    if isinstance(module, NF12Linear):
        return NF12MlxQuantLinear
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


def _packed_mlx_weights(model, config, lm_head_name, *, stock_compatible: bool = False):
    """Transfer exact packed layers or decoded weight-only layers to MLX.

    ``stock_compatible`` keeps only layers that map onto stock
    ``mlx.nn.QuantizedLinear`` and returns ``None`` when a layer needs
    GPT-QModel's custom MLX runtime. The export path uses it so the written
    checkpoint loads with stock ``mlx_lm.load`` on any OS (#4091 item 10).
    """
    quantized = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (TorchLinear, GPTQBitBLASMlxQuantLinear,
                                        AWQBitBLASMlxQuantLinear,
                                        AwqTorchLinear, QQQTorchLinear,
                                        GGUFTorchLinear, TorchFP8Linear, BitsAndBytesLinear,
                                        ExllamaV3TorchLinear, AwqGEMVLinear,
                                        AwqGEMVFastLinear, MacaqLinear, NF12Linear))]
    if not quantized:
        return None

    layer_params = {}
    group16 = set()
    gptq_group16 = set()
    awq_group16 = set()
    gguf_dtype = set()
    gguf_q6_k = set()
    bitsandbytes_native = {}
    exl3_native = {}
    gptq_dtype = set()
    awq_dtype = set()
    paro = {}
    qqq = {}
    fp8_native = {}
    fp8_packed = {}
    macaq_native = {}
    nf12_native = {}
    for name, module in quantized:
        if isinstance(module, ExllamaV3TorchLinear):
            if module.in_features <= 0 or module.out_features <= 0 or getattr(module, "trellis", None) is None:
                raise ValueError(f"EXL3 layer {name} cannot be transferred to MLX")
            exl3_native[name] = module
            continue
        mlx_linear = _mlx_holder_class(module)
        if not mlx_linear.source_compatible(module):
            if isinstance(module, (ParoLinear, QQQTorchLinear, GGUFTorchLinear,
                                   TorchFP8Linear, BitsAndBytesLinear, AwqGEMVLinear,
                                   AwqGEMVFastLinear, MacaqLinear, NF12Linear)):
                raise ValueError(f"{type(module).__name__} layer {name} cannot be transferred to MLX")
            return None
        if isinstance(module, TorchFP8Linear) and mlx_linear.native_compatible(module):
            fp8_native[name] = module
        elif isinstance(module, TorchFP8Linear):
            fp8_packed[name] = mlx_linear.packed_payload(module)
            continue
        elif isinstance(module, BitsAndBytesLinear):
            bitsandbytes_native[name] = BitsAndBytesMlxQuantLinear.native_payload(module)
            continue
        elif isinstance(module, MacaqLinear) and not MacaqMlxQuantLinear.stock_compatible(module):
            macaq_native[name] = MacaqMlxQuantLinear.native_payload(module)
            continue
        elif isinstance(module, NF12Linear):
            nf12_native[name] = NF12MlxQuantLinear.native_payload(module)
            continue
        layer_params[name] = mlx_linear.mlx_params(module)
        if isinstance(module, ParoLinear):
            paro[name] = module
        if isinstance(module, QQQTorchLinear):
            qqq[name] = module
        if (isinstance(module, GGUFTorchLinear) and module.gguf_tensor_qtype == "Q6_K") or (
                not isinstance(module, (QQQTorchLinear, GGUFTorchLinear, TorchFP8Linear, MacaqLinear))
                and module.group_size == 16):
            group16.add(name)
            if isinstance(module, GGUFTorchLinear) and module.gguf_tensor_qtype == "Q6_K":
                gguf_q6_k.add(name)
            elif mlx_linear in (MlxQuantLinear, GPTQMarlinMlxQuantLinear):
                gptq_group16.add(name)
            elif mlx_linear in (AwqMlxQuantLinear, AWQMarlinMlxQuantLinear):
                awq_group16.add(name)
        elif isinstance(module, (GPTQBitBLASMlxQuantLinear, TorchLinear)):
            if not stock_compatible:
                gptq_dtype.add(name)
        elif (isinstance(module, (AwqTorchLinear, AWQBitBLASMlxQuantLinear,
                                  AwqGEMVLinear, AwqGEMVFastLinear, LLMAwqLinear))
              and not isinstance(module, ParoLinear)):
            if not stock_compatible:
                awq_dtype.add(name)
        elif isinstance(module, GGUFTorchLinear):
            gguf_dtype.add(name)

    if stock_compatible:
        custom_layers = (
            set(group16) | set(paro) | set(qqq) | set(fp8_native)
            | set(fp8_packed) | set(bitsandbytes_native) | set(exl3_native)
            | set(gguf_dtype) | set(macaq_native) | set(nf12_native)
        )
        if custom_layers:
            log.info(
                "MLX: %d layer(s) need GPT-QModel's custom MLX runtime; using "
                "the stock-compatible requantization path instead.",
                len(custom_layers),
            )
            return None

    weights = {}
    tied_embeddings = config.get("tie_word_embeddings", False)
    for name, module in model.named_modules():
        if name in exl3_native:
            weights[f"{name}.trellis"] = mx.array(
                module.trellis.detach().to("cpu").contiguous().numpy()
            )
            weights[f"{name}.suh"] = mx.array(
                _exl3_signs(module, "suh", "su", module.in_features)
            )
            weights[f"{name}.svh"] = mx.array(
                _exl3_signs(module, "svh", "sv", module.out_features)
            )
        elif name in bitsandbytes_native:
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
        elif name in fp8_packed:
            payload = fp8_packed[name]
            weights[f"{name}.weight"] = mx.array(payload["weight"]).astype(mx.uint8)
            weights[f"{name}.scales"] = mx.array(payload["scales"]).astype(mx.float32).reshape(-1)
            weights[f"{name}.codebook"] = mx.array(payload["codebook"]).astype(mx.float32).reshape(256)
            weights[f"{name}.bias"] = (
                mx.zeros((module.out_features,), dtype=mx.float32)
                if payload["bias"] is None else mx.array(payload["bias"]).astype(mx.float32)
            )
        elif name in macaq_native:
            payload = macaq_native[name]
            compute_dtype = _mlx_dtype(payload["compute_dtype"])
            weights[f"{name}.weight"] = mx.array(payload["weight"])
            weights[f"{name}.scales"] = mx.array(payload["scales"])
            weights[f"{name}.pre_scale"] = mx.array(payload["pre_scale"]).astype(compute_dtype)
            weights[f"{name}.BH"] = mx.array(payload["BH"]).astype(compute_dtype)
            weights[f"{name}.bias"] = (
                mx.zeros((module.out_features,), dtype=mx.float32)
                if payload["bias"] is None else mx.array(payload["bias"]).astype(mx.float32)
            )
        elif name in nf12_native:
            payload = nf12_native[name]
            weights[f"{name}.packed"] = mx.array(payload["packed"]).astype(mx.uint8)
            weights[f"{name}.escapes"] = mx.array(payload["escapes"]).astype(mx.int16)
            weights[f"{name}.exp_base"] = mx.array([payload["exp_base"]], dtype=mx.uint8)
            weights[f"{name}.bias"] = (
                mx.zeros((module.out_features,), dtype=mx.float32)
                if payload["bias"] is None else mx.array(payload["bias"]).astype(mx.float32)
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
        if (getattr(module, "bias", None) is not None
                and name not in bitsandbytes_native and name not in fp8_packed
                and name not in macaq_native and name not in nf12_native):
            prefix = f"{name}.linear" if name in paro or name in gptq_dtype or name in awq_dtype or name in gguf_dtype else name
            weights[f"{prefix}.bias"] = mx.array(
                module.bias.detach().to("cpu", torch.float16).numpy()
            )
            if name in qqq:
                weights[f"{name}.linear.bias"] = mx.zeros((module.out_features,), dtype=mx.float16)

    mlx_config = _mlx_model_config(config)
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
            if path in gptq_group16:
                return MlxGPTQGroup16Linear(
                    input_dims, output_dims, layer_params[path]["bits"],
                    bias=module.get("bias") is not None,
                )
            if path in awq_group16:
                return MlxAWQGroup16Linear(
                    input_dims, output_dims, layer_params[path]["bits"],
                    bias=module.get("bias") is not None,
                )
            if path in gguf_q6_k:
                return MlxGGUFQ6KLinear(
                    input_dims, output_dims, bias=module.get("bias") is not None,
                )
            return MlxGroup16Linear(input_dims, output_dims, layer_params[path]["bits"],
                                    bias=module.get("bias") is not None)

        mlx_model.update_modules(tree_map_with_path(
            replace_group16, mlx_model.leaf_modules(), is_leaf=nn.Module.is_module,
        ))
    if (paro or qqq or fp8_native or fp8_packed or gptq_dtype or awq_dtype or gguf_dtype
            or bitsandbytes_native or exl3_native or macaq_native or nf12_native):
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
            if path in macaq_native:
                payload = macaq_native[path]
                compute_dtype = _mlx_dtype(payload["compute_dtype"])
                return MlxMACaQLinear(
                    in_features=payload["in_features"],
                    out_features=payload["out_features"],
                    had_block=payload["had_block"],
                    activation_precision=payload["activation_precision"],
                    compute_dtype=compute_dtype,
                    apply_rotation=payload.get("apply_rotation", True),
                    apply_pre_scale=payload.get("apply_pre_scale", True),
                )
            if path in nf12_native:
                payload = nf12_native[path]
                return MlxNF12Linear(
                    in_features=payload["in_features"],
                    out_features=payload["out_features"],
                    num_escapes=payload["escapes"].shape[0],
                )
            if path in gptq_dtype:
                return MlxGPTQLinear(module)
            if path in awq_dtype:
                return MlxAWQLinear(module)
            if path in gguf_dtype:
                return MlxGGUFLinear(module)
            if path in bitsandbytes_native:
                return MlxBitsAndBytesLinear(**bitsandbytes_native[path])
            if path in exl3_native:
                source = exl3_native[path]
                return MlxEXL3Linear(
                    source.in_features, source.out_features,
                    source._bits_per_weight(), source._codebook_name(),
                    bias=source.bias is not None,
                )
            if path in fp8_packed:
                payload = fp8_packed[path]
                return MlxFP8PackedLinear(
                    None, None, None,
                    in_features=payload["in_features"],
                    out_features=payload["out_features"],
                    scale_method=payload["scale_method"],
                    block_size=payload["block_size"],
                )
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
    if (group16 or paro or qqq or fp8_native or fp8_packed or gptq_dtype or awq_dtype
            or gguf_dtype or bitsandbytes_native or exl3_native or macaq_native or nf12_native):
        # MLX-LM's standard loader reconstructs only native QuantizedLinear.
        # Keep this runtime model instead of round-tripping through that loader.
        mlx_config["_gptqmodel_group16_runtime" if group16 and not paro and not qqq and not fp8_packed
                   else "_gptqmodel_custom_mlx_runtime"] = True
    return mlx_model, mlx_config


def convert_gptq_to_mlx_weights(
    model_id_or_path: str,
    model: Union[PreTrainedModel, BaseQModel],
    gptq_config: dict,
    lm_head_name: str,
    *,
    stock_compatible: bool = False,
):
    if not MLX_AVAILABLE:
        raise ValueError("MLX not installed. Please install via `pip install gptqmodel[mlx] --no-build-isolation`.")

    # Keep conversion on CPU while restoring MLX's GPU default for inference.
    with mx.stream(mx.cpu):
        return _convert_gptq_to_mlx_weights(
            model_id_or_path,
            model,
            gptq_config,
            lm_head_name,
            stock_compatible=stock_compatible,
        )


def _mlx_quant_predicate(dynamic, default_bits, default_group_size):
    """Map GPT-QModel `dynamic` rules onto MLX-LM's quant predicate."""

    def quant_predicate(path, module):
        # The exact packed transfer honors `dynamic` through the source
        # layer's own bits/group; this predicate keeps the requantization
        # fallback on the same contract: `-:` exclusions stay float, `+:`
        # overrides keep their bits/group.
        override = dynamic_get(dynamic, path)
        if override is False:
            return False
        if override is None:
            return True
        group_size = override.get("group_size", default_group_size)
        if group_size in (-1, 16):
            group_size = 64
        return {
            "group_size": group_size,
            "bits": override.get("bits", default_bits),
            "mode": "affine",
        }

    return quant_predicate


def _convert_gptq_to_mlx_weights(
    model_id_or_path, model, gptq_config, lm_head_name, *, stock_compatible: bool = False
):

    quant_format = resolve_quant_format(gptq_config.get("format"), gptq_config.get("method", gptq_config.get("quant_method")))
    if quant_format not in (FORMAT.EXL3, FORMAT.NF12) and gptq_config["bits"] not in [1, 2, 3, 4, 5, 6, 7, 8]:
        raise ValueError("MLX conversion supports 1 through 8 integer bits for these formats")
    if quant_format not in [FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.GPTQ_P, FORMAT.GEMM,
                            FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ,
                            FORMAT.MARLIN, FORMAT.BITBLAS,
                            FORMAT.PARO, FORMAT.QQQ, FORMAT.GGUF, FORMAT.FP8,
                            FORMAT.BITSANDBYTES, FORMAT.EXL3,
                            FORMAT.MXFP4_MACAQ, FORMAT.MXFP4, FORMAT.NF12]:
        raise ValueError("MLX conversion requires a supported GPT-QModel quantization format")

    dynamic = gptq_config.get("dynamic") or {}
    if dynamic:
        for _, config in dynamic.items():
            if config != {}:
                dynamic_bits = config.get("bits", gptq_config["bits"])
                if dynamic_bits not in [2, 3, 4, 5, 6, 7, 8]:
                    raise ValueError(f'MLX GPTQ conversion does not support {dynamic_bits} bits in dynamic config')

    model_path = Path(model_id_or_path)
    if not model_path.exists():
        model_path = Path(snapshot_download(
            model_id_or_path, allow_patterns=["config.json", "generation_config.json"]
        ))
    config = load_config(model_path)

    if isinstance(model, BaseQModel):
        model = model.model

    packed = _packed_mlx_weights(
        model, config, lm_head_name, stock_compatible=stock_compatible
    )
    if packed is not None:
        log.info("MLX: transferred source weights without requantization")
        return packed
    if quant_format in (FORMAT.PARO, FORMAT.QQQ, FORMAT.GGUF, FORMAT.FP8,
                        FORMAT.BITSANDBYTES, FORMAT.EXL3, FORMAT.GEMV,
                        FORMAT.GEMV_FAST, FORMAT.LLM_AWQ, FORMAT.MARLIN,
                        FORMAT.BITBLAS, FORMAT.MXFP4_MACAQ, FORMAT.MXFP4, FORMAT.NF12):
        raise ValueError(f"{quant_format} MLX inference requires transferable packed weights and runtime state")

    # Requantization needs an MLX-supported group size.
    if gptq_config["group_size"] in [-1, 16]:
        gptq_config["group_size"] = 64

    default_bits = gptq_config["bits"]
    default_group_size = gptq_config["group_size"]
    quant_predicate = _mlx_quant_predicate(dynamic, default_bits, default_group_size)

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
    mlx_config = _mlx_model_config(config)
    model_class, model_args_class = _get_classes(config=mlx_config)
    mlx_model = model_class(model_args_class.from_dict(mlx_config))

    # Load and quantize weights
    log.info("Starting MLX quantization...")
    mlx_model.load_weights(list(weights.items()))
    weights, mlx_config = quantize_model(mlx_model, mlx_config, group_size=gptq_config["group_size"],
                                     bits=gptq_config["bits"], quant_predicate=quant_predicate)
    log.info("MLX quantization completed")

    return weights, mlx_config


def _mlx_prompt_list(prompt):
    """Normalize ``generate()`` inputs to the prompt list mlx_lm accepts."""
    if isinstance(prompt, str):
        return [prompt]
    if torch.is_tensor(prompt):
        prompt = prompt.detach().to("cpu").tolist()
    elif hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        if not prompt:
            raise ValueError("MLX requires a non-empty prompt")
        if all(isinstance(token, int) for token in prompt):
            return [list(prompt)]
        return [list(tokens) if not isinstance(tokens, str) else tokens for tokens in prompt]
    raise TypeError(f"MLX prompt must be a string, token list, or tensor: {type(prompt).__name__}")


def _mlx_attention_mask_rows(attention_mask):
    if attention_mask is None:
        return None
    if torch.is_tensor(attention_mask):
        return attention_mask.detach().to("cpu").tolist()
    if hasattr(attention_mask, "tolist"):
        return attention_mask.tolist()
    return attention_mask


def _mlx_strip_padding(prompts, attention_mask, pad_token_id):
    """Remove padded positions from token prompts before ``mlx_lm`` sees them.

    ``mlx_lm.generate`` treats every id in a prompt as content, so a batched,
    left-padded ``input_ids`` makes the pad tokens part of each prompt (#4091
    item 8). ``attention_mask`` is exact; ``pad_token_id`` is a fallback that
    trims leading and trailing pad ids.
    """

    rows = _mlx_attention_mask_rows(attention_mask)
    if rows is not None:
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        if len(rows) != len(prompts):
            raise ValueError(
                f"MLX attention_mask has {len(rows)} rows but {len(prompts)} "
                "prompts were given"
            )
        stripped = []
        for item, mask_row in zip(prompts, rows):
            if isinstance(item, str):
                log.warn(
                    "MLX generate: attention_mask cannot be applied to string "
                    "prompts; tokenize them or pass input_ids"
                )
                return prompts
            if len(mask_row) != len(item):
                raise ValueError(
                    "MLX attention_mask row length does not match the prompt "
                    "token count"
                )
            stripped.append([token for token, keep in zip(item, mask_row) if keep])
        return stripped

    if pad_token_id is None:
        return prompts
    pad = int(pad_token_id)
    stripped = []
    for item in prompts:
        if isinstance(item, str):
            stripped.append(item)
            continue
        start, end = 0, len(item)
        while start < end and item[start] == pad:
            start += 1
        while end > start and item[end - 1] == pad:
            end -= 1
        stripped.append(item[start:end])
    return stripped


def _mlx_stop_strings(stop):
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [item for item in stop if isinstance(item, str)]


def _mlx_strip_stop(text, stop_strings):
    """Trim a generated string at the earliest stop sequence."""
    if not isinstance(text, str):
        return text
    earliest = None
    for stop in stop_strings:
        index = text.find(stop)
        if index != -1 and (earliest is None or index < earliest):
            earliest = index
    return text if earliest is None else text[:earliest]


def _mlx_tokenizer_with_eos(tokenizer, eos_token_ids):
    """Return a per-call tokenizer whose EOS set includes the caller's ids.

    The runtime binds one tokenizer into ``functools.partial`` for every call,
    so mutating it here would let an ``eos_token_id`` from one call truncate
    all later calls (#4091 item 9). Return a shallow copy instead.
    """
    ids = {int(token) for token in eos_token_ids}
    current = getattr(tokenizer, "eos_token_ids", None)
    if current is not None:
        scoped = copy(tokenizer)
        scoped.eos_token_ids = set(current) | ids
        return scoped
    if isinstance(tokenizer, TokenizerWrapper):
        # Older mlx_lm wrappers expose only the private EOS set.
        scoped = copy(tokenizer)
        scoped._eos_token_ids = set(tokenizer._eos_token_ids) | ids
        return scoped
    return TokenizerWrapper(tokenizer, eos_token_ids=ids)


@torch.inference_mode()
def mlx_generate(model, tokenizer, **kwargs,):
    if not MLX_AVAILABLE:
        raise ValueError("MLX not installed. Please install via `pip install gptqmodel[mlx] --no-build-isolation`.")

    prompt = kwargs.pop("prompt", None)
    if prompt is None:
        prompt = kwargs.pop("prompts", None)
    else:
        kwargs.pop("prompts", None)
    if prompt is None:
        prompt = kwargs.pop("input_ids", None)
    else:
        kwargs.pop("input_ids", None)
    if prompt is None:
        raise ValueError("MLX requires prompts or input_ids to be provided")
    prompts = _mlx_prompt_list(prompt)
    attention_mask = kwargs.pop("attention_mask", None)
    pad_token_id = kwargs.pop("pad_token_id", None)
    prompts = _mlx_strip_padding(prompts, attention_mask, pad_token_id)

    verbose = kwargs.pop("verbose", False)
    kwargs.pop("formatter", None)

    sampling_params = {}
    max_new_tokens = kwargs.pop("max_new_tokens", None)
    max_tokens = kwargs.pop("max_tokens", None)
    sampling_params["max_tokens"] = (
        max_new_tokens if max_new_tokens is not None else (256 if max_tokens is None else max_tokens)
    )

    seed = kwargs.pop("seed", None)
    if seed is not None:
        mx.random.seed(int(seed))

    eos_token_id = kwargs.pop("eos_token_id", None)
    if eos_token_id is not None:
        eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
        tokenizer = _mlx_tokenizer_with_eos(tokenizer, eos_ids)

    stop_strings = _mlx_stop_strings(kwargs.pop("stop", None))

    sampler = kwargs.pop("sampler", None)
    temperature = kwargs.pop("temp", None)
    if temperature is None:
        temperature = kwargs.pop("temperature", None)
    do_sample = kwargs.pop("do_sample", None)
    if temperature is None:
        # HF's `do_sample` selects greedy or temperature sampling. MLX's
        # sampler is greedy at temp 0.0, so translate the flag instead of
        # dropping it.
        if do_sample is False:
            temperature = 0.0
        elif do_sample is True:
            temperature = 1.0
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

    if kwargs:
        log.warn(
            "MLX generate: ignoring unsupported arguments: "
            + ", ".join(sorted(kwargs))
        )

    results = [
        _mlx_strip_stop(
            generate(model=model, tokenizer=tokenizer, prompt=item, verbose=verbose, **sampling_params),
            stop_strings,
        )
        for item in prompts
    ]
    return results[0] if len(results) == 1 else results
