# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import warnings

import torch

from .. import dtypes, ops


def quantize_weight(
    weight: torch.Tensor,
    dtype: dtypes.DataType,
    scale_dtype: dtypes.DataType | None,
    group_size: int,
    group_size_n: int | None = None,
    has_zero_point: bool = False,
    weight_scale_2_type: str | None = None,
    is_fp_zero_point: bool = False,
    pack: bool = False,
    allow_negative_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert weight.ndim in [2, 3]
    assert not has_zero_point or scale_dtype is not None

    assert weight_scale_2_type in [None, "none", "tensor", "channel"]
    has_tensor_scale = weight_scale_2_type == "tensor"
    has_channel_scale_2 = weight_scale_2_type == "channel"
    assert not has_channel_scale_2 or scale_dtype is not None, (
        "channel weight_scale_2 requires a groupwise weight_scale"
    )
    assert not has_channel_scale_2 or group_size_n is None

    weight = weight.cuda()
    origin_ndim = weight.ndim
    weight = weight.unsqueeze(0) if weight.ndim == 2 else weight
    origin_dtype = dtypes.DataType.from_torch_dtype(weight.dtype)
    e, n, k = weight.shape
    group_size = group_size if group_size > 0 else k

    if group_size_n is not None:
        assert n % group_size_n == 0
        weight = weight.view(e, n // group_size_n, group_size_n, k // group_size, group_size)
        weight = weight.permute(0, 1, 3, 2, 4).contiguous()
        weight = weight.view(e, n * k // group_size_n // group_size, -1)
        group_size = group_size_n * group_size

    quant_group_size = 0
    if scale_dtype is not None:
        quant_group_size = group_size
    elif has_tensor_scale:
        quant_group_size = weight.nelement() // e
    flatten_weight = weight.view(e, 1, -1)
    use_flatten_weight = scale_dtype is None and has_tensor_scale
    weight_scale: torch.Tensor | None
    quanted_weight, weight_scale, zero_point = ops.quant_weight(
        flatten_weight if use_flatten_weight else weight,
        source_dtype_str=str(origin_dtype),
        target_dtype_str=str(dtype),
        group_size=quant_group_size,
        use_e8m0_scale=scale_dtype == dtypes.float8e8m0,
        has_scale=scale_dtype is not None or has_tensor_scale,
        has_zero_point=has_zero_point,
        is_fp_zero_point=is_fp_zero_point,
        allow_negative_scale=allow_negative_scale,
    )

    if zero_point.dtype == torch.float32:
        torch_dtype = torch.float16 if scale_dtype == dtypes.float16 else torch.bfloat16
        zero_point = zero_point.to(torch_dtype)

    weight_scale_2 = None
    if has_channel_scale_2:
        assert weight_scale is not None
        ws_f32 = weight_scale.float()
        if scale_dtype == dtypes.float8e8m0:
            weight_scale_2 = ws_f32.log2().mean(-1).exp2()
        elif scale_dtype in [dtypes.float8e4m3, dtypes.float8e5m2]:
            max_value = 448 if scale_dtype == dtypes.float8e4m3 else 57344
            weight_scale_2 = torch.maximum(
                ws_f32.abs().amax(-1) / max_value,
                ws_f32.abs().mean(-1),
            )
        else:
            weight_scale_2 = ws_f32.abs().mean(-1)
        weight_scale = (ws_f32 / weight_scale_2.unsqueeze(-1)).to(weight_scale.dtype)

    tensor_scale = None
    if scale_dtype is None and has_tensor_scale:
        tensor_scale = weight_scale.view(-1)
        weight_scale = None
        quanted_weight = quanted_weight.view(e, n, k)
    elif has_tensor_scale and scale_dtype == dtypes.float8e8m0:
        tensor_scale = weight_scale.float().view(e, -1).log2().mean(1).exp2()
        weight_scale = (weight_scale.float() / tensor_scale.view(e, 1, 1)).to(torch.float8_e8m0fnu)
    elif scale_dtype in [dtypes.float16, dtypes.bfloat16]:
        if has_tensor_scale:
            tensor_scale = weight_scale.view(e, -1).abs().mean(1)
            weight_scale_view = weight_scale.view(e, -1)
            weight_scale_view = weight_scale_view / tensor_scale.unsqueeze(1)
            weight_scale = weight_scale_view.view(weight_scale.shape)
        torch_dtype = torch.float16 if scale_dtype == dtypes.float16 else torch.bfloat16
        weight_scale = weight_scale.to(torch_dtype)
    elif scale_dtype in [dtypes.float8e4m3, dtypes.float8e5m2]:
        max_value = 448 if scale_dtype == dtypes.float8e4m3 else 57344
        torch_dtype = torch.float8_e4m3fn if scale_dtype == dtypes.float8e4m3 else torch.float8_e5m2
        if has_tensor_scale:
            tensor_scale1 = weight_scale.view(e, -1).max(1)[0] / max_value
            tensor_scale2 = weight_scale.view(e, -1).abs().mean(1)
            use_scale1 = (tensor_scale1 > tensor_scale2).any()
            tensor_scale = tensor_scale1 if use_scale1 else tensor_scale2
            weight_scale = weight_scale / tensor_scale.view(-1, 1, 1)
        weight_scale = weight_scale.to(torch_dtype)

    if group_size_n is not None:
        group_size = group_size // group_size_n
        quanted_weight = quanted_weight.view(
            e,
            n // group_size_n,
            k // group_size,
            group_size_n,
            group_size,
        )
        quanted_weight = quanted_weight.permute(0, 1, 3, 2, 4).contiguous()
        quanted_weight = quanted_weight.view(e, n, k)
        assert weight_scale is not None
        weight_scale = weight_scale.view(e, n // group_size_n, k // group_size)

    if origin_ndim == 2:
        quanted_weight = quanted_weight.squeeze(0)
        if weight_scale is not None and weight_scale.nelement() > 0:
            weight_scale = weight_scale.squeeze(0)
        if zero_point is not None and zero_point.nelement() > 0:
            zero_point = zero_point.squeeze(0)
        if tensor_scale is not None and tensor_scale.nelement() > 0:
            tensor_scale = tensor_scale.squeeze(0)
        if weight_scale_2 is not None:
            weight_scale_2 = weight_scale_2.squeeze(0)

    if pack:
        quanted_weight = ops.pack_weight(quanted_weight, dtype.num_bits)
        if has_zero_point and not is_fp_zero_point:
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape)
            zero_point = ops.pack_weight(zero_point, dtype.num_bits)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape)

    final_zero_point = zero_point if zero_point.nelement() > 0 else None
    final_scale_2 = weight_scale_2 if weight_scale_2 is not None else tensor_scale

    return quanted_weight, weight_scale, final_zero_point, final_scale_2


def dequantize_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    zero_point: torch.Tensor | None,
    weight_scale_2: torch.Tensor | None,
    dtype: dtypes.DataType,
    packed: bool = False,
) -> torch.Tensor:
    assert weight.dtype == torch.int32
    weight = weight.cuda()

    if packed:
        weight = ops.unpack_weight(weight, dtype.num_bits)
        if zero_point is not None and zero_point.dtype == torch.int32:
            zero_point = zero_point.transpose(-1, -2).contiguous().cuda()
            zero_point = zero_point.view(*zero_point.shape)
            zero_point = ops.unpack_weight(zero_point, dtype.num_bits)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape).float()

    if isinstance(dtype, dtypes.FloatingPointType):
        weight = ops.dequant_weight(weight, dtype.exponent_bits, dtype.mantissa_bits, True)
    else:
        assert isinstance(dtype, dtypes.IntegerType)
        assert not dtype.is_signed
        weight = weight.float()

    if zero_point is not None:
        assert weight.size(-1) % zero_point.size(-1) == 0
        group_size = weight.size(-1) // zero_point.size(-1)
        zero_point = zero_point.repeat_interleave(group_size, -1)
        weight = weight - zero_point
    elif isinstance(dtype, dtypes.IntegerType):
        assert not dtype.is_signed
        weight = weight - (1 << (dtype.num_bits - 1))

    if weight_scale is not None:
        assert weight.size(-1) % weight_scale.size(-1) == 0
        group_size = weight.size(-1) // weight_scale.size(-1)
        weight_scale = weight_scale.float()
        weight_scale = weight_scale.repeat_interleave(group_size, -1)
        if weight_scale.size(-2) != weight.size(-2):
            assert weight.size(-2) % weight_scale.size(-2) == 0
            group_size_n = weight.size(-2) // weight_scale.size(-2)
            weight_scale = weight_scale.repeat_interleave(group_size_n, -2)
        weight = weight * weight_scale

    if weight_scale_2 is not None:
        ws2 = weight_scale_2.float()
        num_experts = weight.size(0) if weight.ndim == 3 else 1
        if ws2.nelement() == num_experts:
            ws2 = ws2.view(-1, 1, 1)
            if weight.ndim == 2:
                ws2 = ws2.squeeze(0)
        else:
            ws2 = ws2.reshape(*weight.shape[:-1], 1)
        weight = weight * ws2

    return weight


def prepare_humming_weight(
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
    warnings.warn(
        "prepare_humming_weight is deprecated; use humming.transform.transform_humming_weight",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..transform import transform_humming_weight

    return transform_humming_weight(
        weight,
        b_dtype,
        a_dtype,
        zero_point,
        use_wgmma,
        use_fused_e8m0_scale,
        packed,
        padded_shape_n,
        padded_shape_k,
        interleave_mode,
        use_packed_k_layout,
    )


def prepare_humming_weight_scale(
    weight_scale: torch.Tensor,
    to_apply_on_c: bool = False,
    is_blockwise: bool = False,
    is_mxmma: bool = False,
    mxmma_scale_vec: int = 4,
) -> torch.Tensor:
    warnings.warn(
        "prepare_humming_weight_scale is deprecated; use humming.transform.transform_humming_weight_scale",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..transform import transform_humming_weight_scale

    return transform_humming_weight_scale(
        weight_scale, to_apply_on_c, is_blockwise, is_mxmma, mxmma_scale_vec
    )


def prepare_humming_zero_point(
    zero_point: torch.Tensor,
    dtype: dtypes.DataType,
    packed: bool = False,
) -> torch.Tensor | None:
    warnings.warn(
        "prepare_humming_zero_point is deprecated; use humming.transform.transform_humming_zero_point",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..transform import transform_humming_zero_point

    return transform_humming_zero_point(zero_point, dtype, packed)


def prepare_humming_bias(bias: torch.Tensor) -> torch.Tensor:
    warnings.warn(
        "prepare_humming_bias is deprecated; use humming.transform.transform_humming_bias",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..transform import transform_humming_bias

    return transform_humming_bias(bias)
