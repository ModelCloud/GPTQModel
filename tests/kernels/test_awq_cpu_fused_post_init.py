# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.nn_modules.qlinear.torch_int8_awq import TorchInt8AwqLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


@pytest.mark.parametrize(
    "kernel_cls",
    [
        pytest.param(TorchFusedAwqLinear, id="torch_fused_awq"),
        pytest.param(TorchAtenAwqLinear, id="torch_aten_awq"),
        pytest.param(TorchInt8AwqLinear, id="torch_int8_awq"),
    ],
)
def test_awq_fused_post_init_calls_adapter(monkeypatch, kernel_cls):
    if kernel_cls is TorchAtenAwqLinear:
        monkeypatch.setattr(
            TorchAtenAwqLinear,
            "cached_validate_once",
            classmethod(lambda cls: (True, None)),
        )

    calls = []

    def spy_post_init(self, weight_key, device, **kwargs):
        calls.append((weight_key, device.type, kwargs.get("lora_A"), kwargs.get("lora_B")))

    monkeypatch.setattr(Lora, "post_init", spy_post_init, raising=True)

    module = kernel_cls(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=64,
        bias=False,
        adapter=Lora(rank=1, path=None),
        register_buffers=True,
    )
    module.post_init()

    assert len(calls) == 1
    weight_key, device_type, lora_A, lora_B = calls[0]
    assert weight_key == module.name
    assert device_type == "cpu"
    assert lora_A is not None
    assert lora_B is not None
    assert getattr(module, "wf_unsqueeze_zero", None) is None
    assert getattr(module, "wf_unsqueeze_neg_one", None) is None


def test_torch_aten_awq_backend_selection(monkeypatch):
    monkeypatch.setattr(
        TorchAtenAwqLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.AWQ_TORCH_ATEN,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is TorchAtenAwqLinear
