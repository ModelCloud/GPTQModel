# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import builtins

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear, _cpu_int4pack_zero_offsets
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.utils import python as python_utils


def test_free_threading_build_helper_uses_py_gil_disabled(monkeypatch):
    monkeypatch.setattr(
        python_utils.sysconfig,
        "get_config_var",
        lambda key: 1 if key == "Py_GIL_DISABLED" else None,
    )
    assert python_utils.is_free_threading_build()

    monkeypatch.setattr(
        python_utils.sysconfig,
        "get_config_var",
        lambda key: 0 if key == "Py_GIL_DISABLED" else None,
    )
    assert not python_utils.is_free_threading_build()


@pytest.mark.parametrize("kernel_cls", [TorchAtenLinear, TorchAtenAwqLinear])
def test_torch_aten_kernel_validate_once_does_not_import_external_kernels(monkeypatch, kernel_cls):
    attempted = {"value": False}
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kernels" or name.startswith("kernels."):
            attempted["value"] = True
            raise AssertionError(f"unexpected import of {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    ok, err = kernel_cls.validate_once()

    assert not attempted["value"]
    has_ops = (
        hasattr(torch.ops.aten, "_convert_weight_to_int4pack_for_cpu")
        and hasattr(torch.ops.aten, "_weight_int4pack_mm_for_cpu")
    )
    assert ok is has_ops
    if has_ops:
        assert err is None
    else:
        assert isinstance(err, ImportError)


@pytest.mark.skipif(
    not (
        hasattr(torch.ops.aten, "_convert_weight_to_int4pack_for_cpu")
        and hasattr(torch.ops.aten, "_weight_int4pack_mm_for_cpu")
    ),
    reason="CPU int4pack ATen ops are unavailable in this PyTorch build.",
)
def test_cpu_int4pack_zero_offsets_match_dense_gptq_formula():
    out_features = 16
    in_features = 32
    group_size = 32
    code = 5
    zero_code = 3
    scale = 2.0

    unpacked_weight = torch.full((out_features, in_features), code, dtype=torch.int32)
    packed_weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(unpacked_weight, 1)

    scales = torch.full((1, out_features), scale, dtype=torch.bfloat16)
    zero_codes = torch.full((1, out_features), zero_code, dtype=torch.uint8)
    zero_offsets = _cpu_int4pack_zero_offsets(zero_codes, scales, bits=4)

    scales_and_zeros = torch.zeros((1, out_features, 2), dtype=torch.bfloat16)
    scales_and_zeros[:, :, 0] = scales
    scales_and_zeros[:, :, 1] = zero_offsets

    x = torch.zeros((1, in_features), dtype=torch.bfloat16)
    x[0, 0] = 1

    out = torch.ops.aten._weight_int4pack_mm_for_cpu(x, packed_weight, group_size, scales_and_zeros)
    assert float(out[0, 0]) == pytest.approx(scale * (code - zero_code))
