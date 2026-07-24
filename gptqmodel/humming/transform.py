# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import math

import torch

from . import dtypes, ops
from .config import LayerConfig, MmaType, WeightScale2Type, WeightScaleType
from .schema import HummingInputSchema, HummingWeightSchema


def prepare_layer_config(
    shape_n: int,
    shape_k: int,
    weight_schema: HummingWeightSchema,
    input_schema: HummingInputSchema | None = None,
    num_experts: int | None = None,
    pad_n_to_multiple: int = 1,
    pad_k_to_multiple: int = 1,
    has_bias: bool = False,
    torch_dtype: torch.dtype | None = None,
) -> LayerConfig:
    if torch_dtype is None:
        torch_dtype = torch.get_default_dtype()
        if torch_dtype not in [torch.float16, torch.bfloat16]:
            torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    f16_dtype = dtypes.DataType.from_torch_dtype(torch_dtype)
    pad_shape_n = math.ceil(shape_n / pad_n_to_multiple) * pad_n_to_multiple - shape_n
    pad_shape_k = math.ceil(shape_k / pad_k_to_multiple) * pad_k_to_multiple - shape_k

    if input_schema is None:
        input_schema = HummingInputSchema(a_dtype=f16_dtype)

    assert isinstance(input_schema, HummingInputSchema)
    assert isinstance(weight_schema, HummingWeightSchema)

    return LayerConfig(
        a_dtype=input_schema.a_dtype or f16_dtype,
        b_dtype=weight_schema.b_dtype,
        bs_dtype=weight_schema.bs_dtype or f16_dtype,
        as_dtype=input_schema.input_scale_dtype,
        c_dtype=f16_dtype,
        shape_n=shape_n + pad_shape_n,
        shape_k=shape_k + pad_shape_k,
        pad_shape_n=pad_shape_n,
        pad_shape_k=pad_shape_k,
        num_experts=num_experts or 0,
        has_bias=has_bias,
        input_scale_group_size=input_schema.input_scale_group_size,
        weight_scale_group_size=weight_schema.weight_scale_group_size,
        weight_scale_group_size_n=weight_schema.weight_scale_group_size_n,
        weight_scale_type=weight_schema.weight_scale_type,
        weight_scale_2_type=weight_schema.weight_scale_2_type,
        has_zero_point=weight_schema.has_zero_point,
        is_fp_zero_point=weight_schema.is_fp_zero_point,
    )


def check_and_pad_tensors(config: LayerConfig, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = tensors.copy()
    schema = HummingWeightSchema(
        b_dtype=config.b_dtype,
        bs_dtype=config.bs_dtype,
        weight_scale_group_size=config.weight_scale_group_size,
        weight_scale_group_size_n=config.weight_scale_group_size_n,
        weight_scale_type=config.weight_scale_type,
        weight_scale_2_type=config.weight_scale_2_type,
        has_zero_point=config.has_zero_point,
        is_fp_zero_point=config.is_fp_zero_point,
    )

    if config.use_int_weight_scale:
        dtype = dtypes.torch_dtype_map[config.bs_dtype]
        tensors["weight_scale"] = tensors["weight_scale"].to(dtype)

    if config.use_int_weight_scale or config.use_fused_e8m0_scale:
        if "weight_scale_2" not in tensors:
            if config.weight_scale_2_type == WeightScale2Type.CHANNEL:
                shape: tuple[int, ...] = (config.shape_n - config.pad_shape_n,)
                if config.num_experts:
                    shape = (config.num_experts,) + shape
                tensors["weight_scale_2"] = torch.ones(
                    shape,
                    device=tensors["weight_scale"].device,
                    dtype=config.param_dtype,
                )
            else:
                tensors["weight_scale_2"] = torch.ones(
                    (config.num_experts or 1),
                    device=tensors["weight_scale"].device,
                    dtype=torch.float32,
                )

    schema.validate_tensors(
        tensors,
        shape_n=config.shape_n - config.pad_shape_n,
        shape_k=config.shape_k - config.pad_shape_k,
        num_experts=config.num_experts,
        param_dtype=config.param_dtype,
        has_bias=config.has_bias,
    )

    tensors_attrs = schema.get_tensors_attrs(
        shape_n=config.shape_n,
        shape_k=config.shape_k,
        num_experts=config.num_experts,
        param_dtype=config.param_dtype,
        has_bias=config.has_bias,
    )

    for key, attrs in tensors_attrs.items():
        shape = attrs["shape"]
        tensor = tensors[key]
        padding: list[int] = []
        value = 0 if tensor.dtype != torch.float8_e8m0fnu else 1
        for i in range(1, len(shape) + 1):
            padding += (0, shape[-i] - tensor.shape[-i])

        tensors[key] = torch.nn.functional.pad(tensor, pad=padding, value=value)

    return tensors


def process_int_weight_scale(
    config: LayerConfig,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.bs_dtype is not None and config.bs_dtype.num_bits == 8:
        assert weight_scale is not None
        torch_dtype = dtypes.torch_dtype_map[config.c_dtype]
        weight_scale = weight_scale.to(torch_dtype)

    assert weight_scale is not None
    dtype = weight_scale.dtype
    assert dtype in [torch.float16, torch.bfloat16]
    scale_factor = weight_scale.float().abs().max() / 1024
    weight_scale = (weight_scale.float() / scale_factor).round().to(torch.int16)
    weight_scale = weight_scale.view(dtype)

    if weight_scale_2 is not None:
        out_weight_scale_2 = weight_scale_2 * scale_factor
    else:
        out_weight_scale_2 = torch.full(
            (config.num_experts or 1,),
            fill_value=scale_factor.item(),
            device=weight_scale.device,
        )

    return weight_scale, out_weight_scale_2


def process_fused_e8m0_scale(
    config: LayerConfig,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin_dtype = weight_scale.dtype
    origin_shape = weight_scale.shape
    assert origin_dtype in [torch.uint8, torch.float8_e8m0fnu]
    num_experts = config.num_experts or 1
    is_channel = config.weight_scale_2_type == WeightScale2Type.CHANNEL
    num_rows = config.shape_n if is_channel else 1
    weight_scale = weight_scale.view(torch.uint8).view(num_experts, num_rows, -1)

    scale_max = weight_scale.amax(-1, keepdim=True)
    scale_min = weight_scale.amin(-1, keepdim=True)
    scale_range = scale_max - scale_min
    if config.a_dtype == dtypes.int8:
        max_range_val = 3
    elif config.a_dtype == dtypes.float8e4m3:
        max_range_val = 11
    else:
        raise ValueError(f"unsupported a_dtype: {config.a_dtype}")

    max_range = torch.tensor(max_range_val, dtype=torch.uint8, device=scale_range.device)
    scale_range = scale_range.minimum(max_range)
    scale_min_new = scale_max - scale_range
    delta_scale_offsets = weight_scale.maximum(scale_min_new) - weight_scale
    delta_scale_offsets = delta_scale_offsets.view(num_experts, -1)
    weight_scale = weight_scale.maximum(scale_min_new) - scale_min_new
    if config.a_dtype == dtypes.float8e4m3:
        weight_scale = weight_scale + 1
    weight_scale = weight_scale.view(origin_dtype).view(origin_shape)
    ops.process_mxfp4_w4a8_weight(weight, delta_scale_offsets, inplace=True)

    scale_factor = 2 ** (scale_min_new.view(num_experts, num_rows).float() - 127)
    scale_factor = scale_factor / 2
    if is_channel:
        if not config.num_experts:
            scale_factor = scale_factor.squeeze(0)
    else:
        scale_factor = scale_factor.view(-1)
    if weight_scale_2 is not None:
        out_weight_scale_2 = weight_scale_2.float() * scale_factor
    else:
        out_weight_scale_2 = scale_factor

    return weight, weight_scale, out_weight_scale_2


def transform_humming_weight(
    weight: torch.Tensor,
    b_dtype: dtypes.DataType,
    a_dtype: dtypes.DataType,
    zero_point: torch.Tensor | None = None,
    use_wgmma: bool = False,
    use_fused_e8m0_scale: bool = False,
    packed: bool = False,
    padded_shape_n: int | None = None,
    padded_shape_k: int | None = None,
    interleave_mode: int = 3,
    use_packed_k_layout: bool = False,
) -> torch.Tensor:
    is_moe = weight.ndim == 3
    weight = weight.unsqueeze(0) if not is_moe else weight
    if zero_point is not None:
        zero_point = zero_point.unsqueeze(0) if zero_point.ndim == 2 else zero_point
    shape_n = weight.size(-2)
    if packed:
        assert weight.size(-1) * 32 % b_dtype.num_bits == 0
        shape_k = weight.size(-1) * 32 // b_dtype.num_bits
    else:
        shape_k = weight.size(-1)

    padded_shape_n = shape_n if padded_shape_n is None else padded_shape_n
    padded_shape_k = shape_k if padded_shape_k is None else padded_shape_k
    packed_block_size_k = 256 // a_dtype.num_bits

    assert padded_shape_n % 64 == 0
    assert padded_shape_k % (2 * packed_block_size_k) == 0

    should_preprocess_for_int2fp = False
    has_zero_point = zero_point is not None and zero_point.nelement() > 0
    if b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        if a_dtype.num_bits < 16:
            should_preprocess_for_int2fp = True
        elif a_dtype == dtypes.bfloat16 and has_zero_point:
            should_preprocess_for_int2fp = b_dtype.num_bits > 6
        elif a_dtype == dtypes.bfloat16 and not has_zero_point:
            should_preprocess_for_int2fp = b_dtype.num_bits > 7

    if a_dtype == dtypes.int8 and b_dtype in [dtypes.int8, dtypes.uint8]:
        weight = (weight.view(torch.int8) - 128).view(torch.int32)

    if a_dtype == dtypes.int4 and b_dtype in [dtypes.int4, dtypes.uint4]:
        weight = weight.view(torch.uint8)
        weight1 = (weight & 0xF) - 8
        weight1 = weight1 & 0xF
        weight2 = (weight & 0xF0) - 8 * 16
        weight = (weight1 | weight2).view(torch.int32)

    if not should_preprocess_for_int2fp and has_zero_point:
        has_zero_point = False

    should_preprocess_with_zp = has_zero_point
    if zero_point is not None and zero_point.dtype.is_floating_point:
        should_preprocess_with_zp = False
        should_preprocess_for_int2fp = False

    if not has_zero_point:
        group_size_zp = 0
    else:
        assert zero_point is not None
        group_size_zp = shape_k // zero_point.size(-1)

    if use_packed_k_layout:
        assert use_wgmma, "use_packed_k_layout requires wgmma"
        assert a_dtype.num_bits == 8, "use_packed_k_layout requires 8-bit (fp8/int8) activation"
        assert not use_fused_e8m0_scale, "use_packed_k_layout is incompatible with fused-e8m0 scale"

    repacked_weight = ops.repack_weight(
        inputs=weight,
        zero_point=zero_point,
        weight_bits=b_dtype.num_bits,
        activation_bits=a_dtype.num_bits,
        is_weight_packed=packed,
        should_preprocess_for_int2fp=should_preprocess_for_int2fp,
        should_preprocess_with_zp=should_preprocess_with_zp,
        use_wgmma=use_wgmma,
        interleave_mode=interleave_mode,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
        group_size_zp=group_size_zp,
        use_packed_k_layout=use_packed_k_layout,
    )
    return repacked_weight if is_moe else repacked_weight.squeeze(0)


def transform_humming_weight_scale(
    weight_scale: torch.Tensor,
    to_apply_on_c: bool = False,
    is_blockwise: bool = False,
    is_mxmma: bool = False,
    mxmma_scale_vec: int = 4,
) -> torch.Tensor:
    if is_blockwise:
        return weight_scale.transpose(-1, -2).contiguous()

    if is_mxmma:
        if mxmma_scale_vec == 1:
            ws = weight_scale.view(torch.uint8)
            lead = ws.shape[:-2]
            n, g = ws.shape[-2:]
            ws = ws.reshape(*lead, n // 16, 2, 8, g // 2, 2)
            ndim = len(lead)
            perm = (*range(ndim), ndim + 3, ndim, ndim + 2, ndim + 1, ndim + 4)
            ws = ws.permute(*perm).contiguous()
            ws = ws.reshape(*lead, g // 2, n // 2, 4)
            return ws.view(torch.int32).squeeze(-1).contiguous()
        return weight_scale.view(torch.int32).transpose(-1, -2).contiguous()

    perm = [0, 1, 8, 9, 16, 17, 24, 25] if to_apply_on_c else [0, 8, 16, 24, 32, 40, 48, 56]
    count = sum(x < 8 for x in perm)
    perm_new = []
    for i in range(8 // count):
        perm_new += [x + count * i for x in perm]

    perm_tensor = torch.tensor(perm_new, dtype=torch.int32, device=weight_scale.device)
    weight_scale = weight_scale.transpose(-1, -2).contiguous()
    orig_shape = weight_scale.shape
    weight_scale = weight_scale.view(-1, len(perm_tensor))[:, perm_tensor]
    return weight_scale.contiguous().view(orig_shape)


def transform_humming_zero_point(
    zero_point: torch.Tensor,
    dtype: dtypes.DataType,
    packed: bool = False,
) -> torch.Tensor | None:
    num_experts = None if zero_point.ndim == 2 else zero_point.size(0)
    if zero_point.dtype.is_floating_point:
        return transform_humming_weight_scale(zero_point, False)

    if packed:
        zero_point = zero_point.transpose(-1, -2).contiguous()
        zero_point = zero_point.squeeze().view(*zero_point.shape)
        zero_point = ops.unpack_weight(zero_point, dtype.num_bits)
        zero_point = zero_point.transpose(-1, -2).contiguous()

    num_zp_bits = 4 if dtype.num_bits <= 4 else 8
    shape_n = zero_point.size(-2)
    zero_point = transform_humming_weight_scale(zero_point).to(torch.uint8).view(-1)
    if num_zp_bits == 4:
        zero_point = zero_point[..., 1::2] * 16 + zero_point[..., ::2]
    zero_point = zero_point.view(torch.int32).view(-1, shape_n * num_zp_bits // 32)
    if num_experts is not None:
        zero_point = zero_point.view(num_experts, -1, zero_point.size(-1))
    return zero_point


def transform_humming_bias(bias: torch.Tensor) -> torch.Tensor:
    return transform_humming_weight_scale(bias.unsqueeze(-1), True).squeeze(-2)


def transform_humming_tensors(
    config: LayerConfig,
    tensors: dict[str, torch.Tensor],
    already_padded: bool = False,
) -> dict[str, torch.Tensor]:
    if not already_padded:
        tensors = check_and_pad_tensors(config, tensors)
    else:
        tensors = tensors.copy()

    weight = tensors["weight"]
    zero_point = tensors["zero_point"] if config.has_zero_point else None
    bias = tensors["bias"] if config.has_bias else None
    weight_scale = None
    tensor_scale = None
    if config.weight_scale_type == WeightScaleType.TENSOR:
        tensor_scale = tensors["weight_scale"]
    else:
        weight_scale = tensors["weight_scale"]
    weight_scale_2 = None
    if config.weight_scale_2_type != WeightScale2Type.NONE:
        weight_scale_2 = tensors.get("weight_scale_2", None)

    if config.use_fused_e8m0_scale:
        assert weight_scale is not None
        weight, weight_scale, weight_scale_2 = process_fused_e8m0_scale(
            config,
            weight=weight,
            weight_scale=weight_scale,
            weight_scale_2=weight_scale_2,
        )

    interleave_mode = 3
    if config.use_fused_e8m0_scale and config.a_dtype == dtypes.float8e4m3:
        interleave_mode = 2

    weight = transform_humming_weight(
        weight=weight,
        b_dtype=config.b_dtype,
        a_dtype=config.a_dtype,
        zero_point=zero_point,
        use_wgmma=config.mma_type == MmaType.WGMMA,
        use_fused_e8m0_scale=config.use_fused_e8m0_scale,
        packed=True,
        interleave_mode=interleave_mode,
        use_packed_k_layout=config.use_packed_k_layout,
    )

    if weight_scale is not None:
        is_mxmma = config.mma_type == MmaType.MXMMA
        mxmma_scale_vec = None
        if is_mxmma:
            mxmma_scale_vec = 256 // config.a_dtype.num_bits // config.weight_scale_group_size

        weight_scale = transform_humming_weight_scale(
            weight_scale,
            to_apply_on_c=config.should_apply_bs_on_c,
            is_blockwise=config.weight_scale_type == WeightScaleType.BLOCK,
            is_mxmma=is_mxmma,
            mxmma_scale_vec=mxmma_scale_vec,
        )

    if zero_point is not None:
        zero_point = transform_humming_zero_point(zero_point, config.b_dtype, packed=True)

    if bias is not None:
        bias = transform_humming_bias(bias)

    if config.weight_scale_2_type == WeightScale2Type.CHANNEL:
        assert weight_scale_2 is not None
        weight_scale_2 = transform_humming_bias(weight_scale_2.to(config.param_dtype))

    if config.use_int_weight_scale:
        assert weight_scale is not None
        weight_scale, weight_scale_2 = process_int_weight_scale(
            config,
            weight_scale=weight_scale,
            weight_scale_2=weight_scale_2,
        )

    if weight_scale is None:
        weight_scale = tensor_scale

    outputs = {"weight": weight}
    for name, tensor in [
        ("weight_scale", weight_scale),
        ("zero_point", zero_point),
        ("weight_scale_2", weight_scale_2),
        ("bias", bias),
    ]:
        if tensor is not None:
            outputs[name] = tensor
    return outputs
