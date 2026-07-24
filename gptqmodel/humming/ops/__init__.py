# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import json
from typing import TYPE_CHECKING

import torch

from .. import dtypes
from ..kernel.humming import HummingKernel
from .bench import tops_bench
from .hadamard import hadamard_quant_input, hadamard_transform
from .input import quant_input
from .moe import moe_fused_mul_sum
from .utils import init_humming_launcher, register_op
from .weight import (
    dequant_weight,
    pack_weight,
    process_mxfp4_w4a8_weight,
    quant_weight,
    repack_weight,
    unpack_weight,
)


def register_kernel(cubin_path: str) -> tuple[int, str]:
    init_humming_launcher()
    return torch.ops.humming.register_kernel(cubin_path)


def launch_kernel(
    *,
    configs: torch.Tensor | list[int],
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    assert weight_scale is not None, "weight_scale is required (a lone scale also rides this slot)"
    if isinstance(configs, list):
        configs = torch.tensor(configs, dtype=torch.int64, device="cpu")
    return torch.ops.humming.launch_kernel(
        configs,
        inputs,
        weight,
        weight_scale,
        weight_scale_2,
        input_scale,
        zero_point,
        bias,
        outputs,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        expert_layout,
        locks,
        top_k,
        valid_shape_m,
    )


def humming_gemm(
    *,
    layer_config: str,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    compute_config: str | None = None,
    tuning_config: str | None = None,
    input_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    assert weight_scale is not None, "weight_scale is required (a lone scale also rides this slot)"
    configs = HummingKernel.prepare_kernels(layer_config, compute_config, tuning_config)
    return torch.ops.humming.launch_kernel(
        configs,
        inputs,
        weight,
        weight_scale,
        weight_scale_2,
        input_scale,
        zero_point,
        bias,
        outputs,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        expert_layout,
        locks,
        top_k,
        valid_shape_m,
    )


def _humming_gemm_fake(
    *,
    layer_config: str,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    compute_config: str | None = None,
    tuning_config: str | None = None,
    input_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    layer_obj = json.loads(layer_config) if layer_config else {}
    compute_obj = json.loads(compute_config) if compute_config else {}

    shape_n = int(layer_obj["shape_n"]) - int(layer_obj.get("pad_shape_n", 0))
    c_dtype = dtypes.DataType.from_str(layer_obj["c_dtype"])
    output_dtype = dtypes.torch_dtype_map[c_dtype]

    shape_m = inputs.size(0)
    if compute_obj.get("gemm_type") == "indexed":
        shape_m = shape_m * top_k

    if outputs is not None:
        return outputs
    return inputs.new_empty((shape_m, shape_n), dtype=output_dtype)


register_op("humming::hadamard_transform", hadamard_transform, hadamard_transform)
register_op("humming::hadamard_quant_input", hadamard_quant_input, hadamard_quant_input)
register_op("humming::quant_input", quant_input, quant_input)
register_op("humming::quant_weight", quant_weight, quant_weight)
register_op("humming::dequant_weight", dequant_weight, dequant_weight)
register_op("humming::repack_weight", repack_weight, repack_weight)
register_op("humming::pack_weight", pack_weight, pack_weight)
register_op("humming::unpack_weight", unpack_weight, unpack_weight)
register_op("humming::humming_gemm", humming_gemm, _humming_gemm_fake)
register_op("humming::moe_fused_mul_sum", moe_fused_mul_sum, moe_fused_mul_sum)
register_op(
    "humming::process_mxfp4_w4a8_weight",
    process_mxfp4_w4a8_weight,
    process_mxfp4_w4a8_weight,
)


if not TYPE_CHECKING:
    hadamard_transform = torch.ops.humming.hadamard_transform
    hadamard_quant_input = torch.ops.humming.hadamard_quant_input
    quant_input = torch.ops.humming.quant_input
    quant_weight = torch.ops.humming.quant_weight
    dequant_weight = torch.ops.humming.dequant_weight
    repack_weight = torch.ops.humming.repack_weight
    pack_weight = torch.ops.humming.pack_weight
    process_mxfp4_w4a8_weight = torch.ops.humming.process_mxfp4_w4a8_weight
    unpack_weight = torch.ops.humming.unpack_weight
    humming_gemm = torch.ops.humming.humming_gemm
    moe_fused_mul_sum = torch.ops.humming.moe_fused_mul_sum


__all__ = [
    "hadamard_transform",
    "hadamard_quant_input",
    "quant_input",
    "quant_weight",
    "dequant_weight",
    "repack_weight",
    "pack_weight",
    "process_mxfp4_w4a8_weight",
    "unpack_weight",
    "humming_gemm",
    "tops_bench",
    "moe_fused_mul_sum",
]
