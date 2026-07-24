# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import json

import torch

from . import ops
from .config import LayerConfig


def may_quant_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    quanted_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if config.a_dtype.num_bits == 16:
        return inputs, None
    if input_scale is not None:
        return inputs, input_scale
    quanted_input, input_scale = ops.quant_input(
        inputs=inputs,
        outputs=quanted_input,
        dtype=str(config.a_dtype),
        group_size=config.input_scale_group_size or None,
    )
    return quanted_input, input_scale


def may_hadamard_quant_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    hadamard_block_size: int | None = None,
    input_scale: torch.Tensor | None = None,
    quanted_input: torch.Tensor | None = None,
    m_major_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    should_rotate = hadamard_block_size is not None and hadamard_block_size > 1
    should_quant = config.a_dtype.num_bits != 16

    if input_scale is not None:
        return inputs, input_scale
    if not should_rotate and not should_quant:
        return inputs, None
    if should_rotate and not should_quant:
        outputs = ops.hadamard_transform(
            inputs=inputs,
            block_size=hadamard_block_size,
            outputs=quanted_input,
        )
        return outputs, None
    if not should_rotate:
        return ops.quant_input(
            inputs=inputs,
            dtype=str(config.a_dtype),
            outputs=quanted_input,
            group_size=config.input_scale_group_size,
            m_major_scale=m_major_scale,
        )
    return ops.hadamard_quant_input(
        inputs=inputs,
        block_size=hadamard_block_size,
        quant_dtype=str(config.a_dtype),
        group_size=config.input_scale_group_size,
        outputs=quanted_input,
        m_major_scale=m_major_scale,
    )


def humming_forward(
    config: LayerConfig,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    weight_scale_2: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
    compute_config: dict | str | None = None,
    tuning_config: dict | list | str | None = None,
    hadamard_block_size: int | None = None,
) -> torch.Tensor:
    m_major_scale = False
    if config.input_scale_group_size > 0:
        parsed_compute_config = compute_config
        if isinstance(parsed_compute_config, str) and parsed_compute_config:
            parsed_compute_config = json.loads(parsed_compute_config)
        if isinstance(parsed_compute_config, dict):
            m_major_scale = bool(parsed_compute_config.get("use_m_major_input_scale", False))

    inputs, input_scale = may_hadamard_quant_input(
        config,
        inputs=inputs,
        hadamard_block_size=hadamard_block_size,
        input_scale=input_scale,
        m_major_scale=m_major_scale,
    )

    if isinstance(compute_config, dict):
        compute_config = json.dumps(compute_config)
    if isinstance(tuning_config, (list, dict)):
        tuning_config = json.dumps(tuning_config)

    return ops.humming_gemm(
        layer_config=config.to_str(),
        compute_config=compute_config,
        tuning_config=tuning_config,
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        zero_point=zero_point,
        bias=bias,
        weight_scale_2=weight_scale_2,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        locks=locks,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
    )
