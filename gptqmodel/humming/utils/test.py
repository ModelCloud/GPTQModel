# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import enum
import importlib.metadata
import json
import math
from pathlib import Path

import torch

from .. import dtypes, ops
from ..config import GemmType
from .device import calculate_gpu_bandwidth, get_device_name
from .weight import quantize_weight


def generate_random_inputs(
    m: int,
    k: int,
    group_size: int = 0,
    dtype: dtypes.DataType = dtypes.float32,
):
    group_size = group_size if group_size > 0 else k
    input_scale: torch.Tensor | None = torch.randn(
        (m, k // group_size),
        dtype=torch.float32,
        device="cuda:0",
    )
    assert k % group_size == 0

    inputs_orig = torch.randn((m, k), dtype=torch.float32, device="cuda:0")
    init_scale = torch.rand((m, k // group_size), dtype=torch.float32, device="cuda:0")
    inputs_orig = inputs_orig * init_scale.repeat_interleave(group_size, 1)
    inputs_orig = inputs_orig / inputs_orig.std()

    if dtype in [dtypes.float16, dtypes.bfloat16]:
        torch_dtype = torch.float16 if dtype == dtypes.float16 else torch.bfloat16
        inputs = inputs_orig.to(torch_dtype)
        inputs_ref = inputs.float()
        input_scale = None
    else:
        inputs, input_scale, *_ = quantize_weight(
            inputs_orig,
            dtype=dtype,
            scale_dtype=dtypes.float32,
            group_size=group_size,
            has_zero_point=False,
        )

        inputs_ref = inputs.float()
        if isinstance(dtype, dtypes.FloatingPointType):
            inputs_ref = ops.dequant_weight(
                inputs,
                exponent_bits=dtype.exponent_bits,
                mantissa_bits=dtype.mantissa_bits,
                is_signed=True,
            )

        if dtype in [dtypes.int4, dtypes.int8]:
            inputs = inputs.to(torch.int8)
            if dtype == dtypes.int4:
                inputs = inputs.to(torch.uint8) & 0xF
                inputs = inputs[..., 1::2] * 16 + inputs[..., ::2]
                inputs = inputs.view(torch.uint8)
        elif dtype == dtypes.float4e2m1:
            inputs = inputs.to(torch.uint8)
            inputs = inputs[..., 1::2] * 16 + inputs[..., ::2]
        elif dtype == dtypes.float8e4m3:
            inputs = inputs.to(torch.uint8).view(torch.float8_e4m3fn)
        elif dtype == dtypes.float8e5m2:
            inputs = inputs.to(torch.uint8).view(torch.float8_e5m2)

        assert input_scale is not None
        inputs_ref = inputs_ref * input_scale.repeat_interleave(group_size, 1)

    return inputs_orig, inputs_ref, inputs, input_scale


def generate_random_weight(
    n,
    k,
    group_size,
    dtype,
    scale_dtype,
    group_size_n=None,
    num_experts=None,
    weight_scale_2_type=None,
    has_zero_point=False,
    is_fp_zero_point=False,
    allow_negative_scale=True,
):
    e = 1 if num_experts is None else num_experts
    dtype_orig = dtype
    group_size = group_size if group_size > 0 else k
    if has_zero_point:
        assert dtype.is_integer_type and not dtype.is_signed, (
            "dynamic zero point only supports for uint dtype"
        )

    if dtype.is_integer_type and dtype.is_signed:
        dtype = dtypes.IntegerType(is_signed=False, num_bits=dtype.num_bits)

    weight_orig = torch.randn((e, n, k), dtype=torch.float32, device="cuda:0")
    init_weight_scale = torch.rand((e, n, k // group_size), dtype=torch.float32, device="cuda:0")
    init_weight_scale = init_weight_scale + 0.01
    init_weight_bias = torch.randn((e, n, k // group_size), dtype=torch.float32, device="cuda:0")
    init_weight_bias = init_weight_bias

    weight_orig = weight_orig + init_weight_bias.repeat_interleave(group_size, -1)
    weight_orig = weight_orig * init_weight_scale.repeat_interleave(group_size, -1)
    weight_orig = weight_orig / weight_orig.std()

    quanted_weight, weight_scale, zero_point, weight_scale_2 = quantize_weight(
        weight_orig,
        dtype=dtype,
        scale_dtype=scale_dtype,
        group_size=group_size,
        group_size_n=group_size_n,
        has_zero_point=has_zero_point,
        weight_scale_2_type=weight_scale_2_type,
        is_fp_zero_point=is_fp_zero_point,
        allow_negative_scale=allow_negative_scale,
    )

    if dtype.is_integer_type and has_zero_point:
        assert zero_point is not None
        weight_ref = quanted_weight.to(zero_point.dtype)
        weight_ref = weight_ref - zero_point.repeat_interleave(group_size, -1)
        weight_ref = weight_ref.float()
    elif dtype.is_integer_type and not has_zero_point:
        weight_ref = quanted_weight.float() - 2 ** (dtype.num_bits - 1)
    else:
        weight_ref = ops.dequant_weight(
            quanted_weight,
            exponent_bits=dtype.exponent_bits,
            mantissa_bits=dtype.mantissa_bits,
            is_signed=dtype.is_signed,
        )

    if weight_scale is not None:
        weight_scale_tmp = weight_scale.float().repeat_interleave(group_size, -1)
        if group_size_n is not None:
            weight_scale_tmp = weight_scale_tmp.repeat_interleave(group_size_n, -2)
        weight_ref = weight_ref * weight_scale_tmp

    if weight_scale_2_type == "channel":
        weight_ref = weight_ref * weight_scale_2.view(e, n, 1)
    elif weight_scale_2_type == "tensor":
        weight_ref = weight_ref * weight_scale_2.view(-1, 1, 1)

    if dtype_orig.is_integer_type and dtype_orig.is_signed:
        quanted_weight = quanted_weight - 2 ** (dtype.num_bits - 1)

    if num_experts is None:
        weight_orig = weight_orig.squeeze(0)
        weight_ref = weight_ref.squeeze(0)
        quanted_weight = quanted_weight.squeeze(0)
        if weight_scale is not None:
            weight_scale = weight_scale.squeeze(0)
        if weight_scale_2 is not None:
            weight_scale_2 = weight_scale_2.squeeze(0)
        if zero_point is not None:
            zero_point = zero_point.squeeze(0)

    return weight_orig, weight_ref, quanted_weight, weight_scale, zero_point, weight_scale_2


def generate_random_bias(n, dtype):
    assert dtype in [dtypes.float16, dtypes.bfloat16]
    torch_dtype = torch.float16 if dtype == dtypes.float16 else torch.bfloat16
    bias = torch.randn((n,), dtype=torch_dtype, device="cuda:0")
    return bias


_moe_tensors_type = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


def generate_random_moe_tensors(
    shape_m: int,
    num_experts: int,
    top_k: int,
    gemm_type: GemmType | str = "indexed",
    balanced: bool = False,
    block_size_config: int | list[int] | None = None,
    expert_max_tokens: int | None = None,
) -> _moe_tensors_type:
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)

    if gemm_type == GemmType.DENSE:
        return (None,) * 5

    if balanced:
        num_tokens_per_expert = math.ceil(shape_m * top_k / num_experts)
        expert_tensor = torch.randn(
            (num_tokens_per_expert, num_experts),
            dtype=torch.float32,
            device="cuda:0",
        )

        expert_index = torch.argsort(expert_tensor, dim=1).view(-1)[: shape_m * top_k]
        expert_index = expert_index.view(shape_m, top_k)
        tensor = torch.randn((shape_m, num_experts), dtype=torch.float32, device="cuda:0")
        tensor.scatter_(1, expert_index, 1000)
    else:
        tensor = torch.randn((shape_m, num_experts), dtype=torch.float32, device="cuda:0")

    topk_ids = tensor.topk(top_k, 1)[1]
    topk_ids = topk_ids.int()

    if gemm_type in [GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED]:
        expert_num_tokens = topk_ids.view(-1).bincount(minlength=num_experts)
        if gemm_type == GemmType.GROUPED_MASKED:
            assert expert_max_tokens is not None
            assert (expert_num_tokens <= expert_max_tokens).all(), "expert_max_tokens"
            expert_layout = expert_num_tokens.int()
        else:
            expert_first_token_offset = expert_num_tokens.cumsum(0)
            expert_first_token_offset = torch.nn.functional.pad(
                expert_first_token_offset,
                pad=(1, 0),
                value=0,
            )
            expert_layout = expert_first_token_offset.long()
        return topk_ids, expert_layout, None, None, None
    else:
        assert gemm_type == GemmType.INDEXED
        if isinstance(block_size_config, int):
            block_size = block_size_config
        else:
            assert isinstance(block_size_config, list)
            for i in range(len(block_size_config) // 3):
                if shape_m > block_size_config[i * 3] and shape_m <= block_size_config[i * 3 + 1]:
                    block_size = block_size_config[i * 3 + 2]
                    break

        # TODO: moe_align_block_size cuda kernel
        part_token_ids_list = []
        expert_id_list = []
        for expert_id in range(num_experts):
            part_token_ids = torch.where(topk_ids.view(-1) == expert_id)[0]
            num_blocks = math.ceil(part_token_ids.size(0) / block_size)
            padded_size = num_blocks * block_size
            pad_size = padded_size - part_token_ids.size(0)
            part_token_ids = torch.nn.functional.pad(
                part_token_ids,
                pad=(0, pad_size),
                value=topk_ids.nelement(),
            )
            part_token_ids_list.append(part_token_ids)
            expert_id_list += [expert_id] * num_blocks

        sorted_token_ids = torch.cat(part_token_ids_list, dim=0).to(torch.int32)
        expert_ids = torch.tensor(expert_id_list, dtype=torch.int32, device=tensor.device)
        num_tokens_padded = torch.tensor(
            sorted_token_ids.size(0),
            dtype=torch.int32,
            device=tensor.device,
        )

        return topk_ids, None, sorted_token_ids, expert_ids, num_tokens_padded


def random_fill_tensor(tensor: torch.Tensor):
    if tensor.dtype == torch.int32:
        min_value = 2**31 * -1
        max_value = 2**31 - 1
        tensor.random_(min_value, max_value)
    elif tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        tensor.normal_(std=0.01)
    else:
        tensor.copy_(tensor.float().random_().to(tensor.dtype))


def save_benchmark_result(result, args, packages: list[str] | None = None):
    kwargs = dict(vars(args))
    output_file = kwargs.pop("output_file", None)
    if output_file is None:
        return
    kwargs.pop("shape_m_list", None)
    for key, value in list(kwargs.items()):
        if isinstance(value, enum.Enum):
            kwargs[key] = value.value
    if "num_experts" in kwargs and kwargs["num_experts"] is None:
        del kwargs["num_experts"]
        del kwargs["top_k"]
        del kwargs["is_moe_down"]

    versions = {}
    packages = list(packages or [])
    packages.insert(0, "torch")
    for package in packages:
        versions[package] = importlib.metadata.version(package)

    result_new = {}
    for x in result.copy():
        x = x.copy()
        shape_m = x.pop("shape_m")
        result_new[shape_m] = x

    dtype = kwargs.get("dtype", kwargs.get("a_dtype"))
    use_f16_accum = kwargs.get("use_f16_accum", False)

    data = {
        "problem": vars(args),
        "device": {
            "device_name": get_device_name(),
            "memory_gbps": calculate_gpu_bandwidth(),
            "compute_tops": ops.tops_bench(dtype, use_f16_accum=use_f16_accum),
        },
        "packages": versions,
        "result": result_new,
    }

    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


_A_DTYPE_MIN_SM = {
    dtypes.int4: 80,
    dtypes.int8: 75,
    dtypes.float4e2m1: 120,
    dtypes.float8e4m3: 89,
    dtypes.float8e5m2: 89,
    dtypes.bfloat16: 80,
    dtypes.float16: 75,
}


def _current_sm_version():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _coerce_dtype(value):
    if value is None or isinstance(value, dtypes.DataType):
        return value
    return dtypes.DataType.from_str(value)


def skip_if_unsupported(
    a_dtype=None,
    mma_type=None,
    use_cp_async=None,
    use_tma=None,
    use_warp_spec=None,
    use_mbarrier=None,
):
    """Skip a test whose hardware requirements aren't met by the current GPU.

    Capability matrix:
    - WGMMA (Hopper warp-group MMA): SM90.
    - cp.async / mbarrier: SM80+.
    - TMA / warp specialization: SM90+.
    - A-side dtype floors mirror `HummingKernel.check_dtype`.
    """
    import pytest

    sm = _current_sm_version()

    if mma_type == "wgmma" and sm != 90:
        pytest.skip(f"wgmma requires SM90, current SM is {sm}")

    a_dtype = _coerce_dtype(a_dtype)
    if a_dtype is not None and a_dtype in _A_DTYPE_MIN_SM:
        min_sm = _A_DTYPE_MIN_SM[a_dtype]
        if sm < min_sm:
            pytest.skip(f"a_dtype {a_dtype} requires SM>={min_sm}, current SM is {sm}")

    if use_cp_async and sm < 80:
        pytest.skip(f"cp.async requires SM>=80, current SM is {sm}")

    if use_mbarrier and sm < 80:
        pytest.skip(f"mbarrier requires SM>=80, current SM is {sm}")

    if use_tma and sm < 90:
        pytest.skip(f"TMA requires SM>=90, current SM is {sm}")

    if use_warp_spec and sm < 90:
        pytest.skip(f"warp specialization requires SM>=90, current SM is {sm}")
