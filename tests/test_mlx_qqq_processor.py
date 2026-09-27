# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QQQ processor routing checks for the native MLX quantization stage."""

import sys

import pytest
import torch

from gptqmodel.looper.qqq_processor import QQQProcessor
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.quantization import qqq as qqq_module
from gptqmodel.quantization.config import QQQConfig
from gptqmodel.quantization.qqq import QQQ, _qqq_quantize_weight_mlx_to_torch
from gptqmodel.utils.backend import BACKEND
from tests.test_mlx_qqq_quantization import _torch_oracle


if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

pytest.importorskip("mlx.core")


def test_qqq_processor_uses_portable_checkpoint_holder_on_macos():
    processor = object.__new__(QQQProcessor)
    processor.qcfg = QQQConfig(offload_to_disk=False)
    assert processor._quant_linear_kernel() == (
        QQQTorchLinear,
        BACKEND.QQQ_TORCH,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
@pytest.mark.parametrize("group_size", [-1, 128])
def test_qqq_processor_bridge_preserves_mps_values(group_size):
    torch.manual_seed(74)
    source = torch.randn((7, 256), dtype=torch.float32)
    factor = torch.eye(256, dtype=torch.float32)
    factor.diagonal(offset=1).fill_(0.05)
    expected = _torch_oracle(source.numpy(), factor.numpy(), group_size)
    actual = _qqq_quantize_weight_mlx_to_torch(
        source.to("mps"),
        factor.to("mps"),
        group_size=group_size,
    )

    for index in range(4):
        if expected[index] is not None:
            torch.testing.assert_close(
                actual[index].cpu(),
                torch.from_numpy(expected[index]),
                rtol=1e-6,
                atol=1e-6,
            )


def _quantize(source, calibration, group_size, desc_act):
    layer = torch.nn.Linear(
        source.shape[1], source.shape[0], bias=False, dtype=source.dtype
    )
    layer.weight.data.copy_(source)
    config = QQQConfig(
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=0.01,
        offload_to_disk=False,
    )
    task = QQQ(layer, config)
    task.quantizer.configure(
        4,
        perchannel=True,
        sym=True,
        mse=False,
        norm=100,
        groupsize=group_size,
    )
    task.add_batch(calibration, torch.empty(0))
    return task.quantize()


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qqq_processor_mlx_matches_torch(
    monkeypatch, group_size, desc_act, dtype
):
    torch.manual_seed(73)
    source = torch.randn((7, 256), dtype=torch.float32).to(dtype)
    calibration = torch.randn((2, 3, 256), dtype=torch.float32)

    monkeypatch.setattr(qqq_module, "_MLX_QQQ_MIN_ELEMENTS", 0)
    monkeypatch.setattr(
        qqq_module, "_mlx_qqq_quantization_available", lambda: False
    )
    expected = _quantize(source, calibration, group_size, desc_act)
    monkeypatch.setattr(
        qqq_module, "_mlx_qqq_quantization_available", lambda: True
    )
    actual = _quantize(source, calibration, group_size, desc_act)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    assert abs(actual[5] - expected[5]) <= 1e-6
    if expected[7] is None:
        assert actual[7] is None
    else:
        torch.testing.assert_close(actual[7], expected[7], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qqq_processor_dead_columns_preserve_original_extra_scale(
    monkeypatch, dtype
):
    source = torch.full((2, 256), 0.25, dtype=dtype)
    source[:, 0] = 8
    calibration = torch.randn((2, 3, 256), dtype=torch.float32)
    calibration[..., 0] = 0

    monkeypatch.setattr(qqq_module, "_MLX_QQQ_MIN_ELEMENTS", 0)
    monkeypatch.setattr(
        qqq_module, "_mlx_qqq_quantization_available", lambda: False
    )
    expected = _quantize(source, calibration, 128, False)
    monkeypatch.setattr(
        qqq_module, "_mlx_qqq_quantization_available", lambda: True
    )
    actual = _quantize(source, calibration, 128, False)

    torch.testing.assert_close(actual[7], expected[7], rtol=0, atol=0)


@pytest.mark.parametrize(
    "change",
    [
        {"blocksize": 64},
        {"group_size": 64},
        {"static_groups": True},
        {"sym": False},
        {"mse": True},
    ],
)
def test_qqq_processor_rejects_unsupported_mlx_modes(monkeypatch, change):
    monkeypatch.setattr(qqq_module, "_MLX_QQQ_MIN_ELEMENTS", 0)
    monkeypatch.setattr(
        qqq_module, "_mlx_qqq_quantization_available", lambda: True
    )
    weight = torch.zeros((2, 256), dtype=torch.float32)
    inverse_hessian = torch.eye(256, dtype=torch.float32)
    quantizer = qqq_module.Quantizer()
    group_size = change.get("group_size", 128)
    quantizer.configure(
        4,
        perchannel=True,
        sym=change.get("sym", True),
        mse=change.get("mse", False),
        groupsize=group_size,
    )
    assert not qqq_module._should_use_mlx_qqq_quantization(
        weight,
        inverse_hessian,
        quantizer,
        group_size=group_size,
        static_groups=change.get("static_groups", False),
        blocksize=change.get("blocksize", 128),
    )
