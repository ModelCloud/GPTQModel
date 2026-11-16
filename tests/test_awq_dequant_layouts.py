# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``dequantize_model`` must decode every AWQ layout GPTQModel can write.

Each case builds a weight that is exactly representable in the target layout,
packs it with the checkpoint writer of the matching backend, decodes the file
through ``convert_awq_file``, and compares against an independent fp64 oracle
built from the codes, zero points, and scales the case chose.
"""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.utils.model_dequant import convert_awq_file, dequantize_model


_PREFIX = "model.layers.0.mlp.down_proj"
_SCALE = 2.0 ** -3


def _gemm_layout_writer(bits):
    """Pick a writer that emits the AWQ GEMM layout on this platform.

    ``AwqGEMMLinear`` and ``AwqTorchLinear`` share the AWQ GEMM packing
    (``pack_awq_columns`` with ``AWQ_ORDER``); only the runtime kernel differs.
    The native GEMM class is CUDA-only, so on ROCm (and any host without the
    extension) use the portable writer for the 4-bit file. The decoder under
    test sees the same checkpoint layout either way.
    """

    if bits == 4 and AwqGEMMLinear.validate_once()[0]:
        return AwqGEMMLinear
    return AwqTorchLinear


def _exact_weight(codes, zeros, group_size, out_features, in_features):
    """fp64 oracle: ``(code - zero) * scale`` broadcast per input group."""
    group = in_features if group_size <= 0 else group_size
    group_index = torch.arange(in_features) // group
    scale = torch.full((out_features, zeros.shape[1]), _SCALE, dtype=torch.float64)
    zero = zeros.to(torch.float64)[:, group_index]
    return (codes.to(torch.float64) - zero) * scale[:, group_index]


def _packed_tensors(prefix, module_cls, bits, group_size, out_features, in_features):
    groups = 1 if group_size <= 0 else in_features // group_size
    high = (1 << bits) - 1
    codes = torch.randint(0, high + 1, (out_features, in_features), dtype=torch.int32)
    zeros = torch.randint(0, high + 1, (out_features, groups), dtype=torch.int32)
    expected = _exact_weight(codes, zeros, group_size, out_features, in_features).to(torch.float32)

    linear = torch.nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.copy_(expected)

    module = module_cls(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    scale = torch.full((out_features, groups), _SCALE, dtype=torch.float16)
    module.pack(linear, scale, zeros.clone())

    zeros_key = getattr(module, "zeros_name", "qzeros")
    tensors = {
        f"{prefix}.qweight": module.qweight,
        f"{prefix}.{zeros_key}": getattr(module, zeros_key),
        f"{prefix}.scales": module.scales,
    }
    return {key: value.contiguous() for key, value in tensors.items()}, expected


def _decode(path, **config):
    quant_config = {"method": "awq", "bits": 4, "group_size": 128}
    quant_config.update(config)
    return convert_awq_file(path, torch.float32, "cpu", config=quant_config)


@pytest.mark.parametrize(
    "module_cls,bits,group_size,out_features,in_features,format_name",
    [
        (_gemm_layout_writer(4), 4, 128, 128, 128, "gemm"),
        (AwqTorchLinear, 8, 128, 128, 128, "gemm"),
        (AwqGEMVLinear, 4, 128, 128, 128, "gemv"),
        (AwqGEMVFastLinear, 4, 128, 128, 128, "gemv_fast"),
        (AwqGEMVFastLinear, 4, 128, 128, 128, "llm-awq"),
    ],
    ids=["gemm-4bit", "gemm-8bit", "gemv-4bit", "gemv-fast-4bit", "llm-awq-4bit"],
)
def test_awq_layouts_decode_exactly(
    tmp_path, module_cls, bits, group_size, out_features, in_features, format_name
):
    tensors, expected = _packed_tensors(
        _PREFIX, module_cls, bits, group_size, out_features, in_features
    )
    path = tmp_path / f"{format_name}.safetensors"
    save_file(tensors, str(path))

    decoded = _decode(path, format=format_name, bits=bits, group_size=group_size)

    weight = decoded[f"{_PREFIX}.weight"]
    assert weight.shape == expected.shape
    assert weight.dtype == torch.float32
    torch.testing.assert_close(weight, expected, rtol=0.0, atol=0.0)


def test_awq_dynamic_bits_are_resolved_per_module(tmp_path):
    """A 4-bit run may store one module at 8 bits through ``dynamic``."""

    other = "model.layers.0.mlp.up_proj"
    down_tensors, down_expected = _packed_tensors(
        _PREFIX, _gemm_layout_writer(4), 4, 128, 128, 128
    )
    up_tensors, up_expected = _packed_tensors(other, AwqTorchLinear, 8, 128, 128, 128)
    path = tmp_path / "dynamic.safetensors"
    save_file({**down_tensors, **up_tensors}, str(path))

    decoded = _decode(
        path,
        format="gemm",
        bits=4,
        group_size=128,
        dynamic={other: {"bits": 8}},
    )

    torch.testing.assert_close(
        decoded[f"{_PREFIX}.weight"], down_expected, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        decoded[f"{other}.weight"], up_expected, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    "module_cls,bits,format_name",
    [
        (_gemm_layout_writer(4), 4, "gemm"),
        (AwqTorchLinear, 8, "gemm"),
        (AwqGEMVLinear, 4, "gemv"),
        (AwqGEMVFastLinear, 4, "gemv_fast"),
        (AwqGEMVFastLinear, 4, "llm-awq"),
    ],
    ids=["gemm-4bit", "gemm-8bit", "gemv-4bit", "gemv-fast-4bit", "llm-awq-4bit"],
)
def test_dequantize_model_dispatches_on_the_saved_layout(
    tmp_path, module_cls, bits, format_name
):
    """``dequantize_model`` reads ``format`` from ``config.json``, not just ``bits``."""

    tensors, expected = _packed_tensors(_PREFIX, module_cls, bits, 128, 128, 128)
    source = tmp_path / "source"
    source.mkdir()
    save_file(tensors, str(source / "model.safetensors"))
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "quantization_config": {
            "method": "awq",
            "format": format_name,
            "bits": bits,
            "group_size": 128,
        },
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    dequantize_model(source, tmp_path / "out", target_dtype=torch.float32)

    with safe_open(str(tmp_path / "out" / "model.safetensors"), framework="pt") as reader:
        weight = reader.get_tensor(f"{_PREFIX}.weight")
    torch.testing.assert_close(weight, expected, rtol=0.0, atol=0.0)
