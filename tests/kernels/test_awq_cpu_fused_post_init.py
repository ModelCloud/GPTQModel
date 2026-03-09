# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.gemm_hf_kernel_awq import HFKernelAwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqQuantLinear
from gptqmodel.nn_modules.qlinear.torch_int8_awq import TorchInt8AwqQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


@pytest.mark.parametrize(
    "kernel_cls",
    [
        pytest.param(TorchFusedAwqQuantLinear, id="torch_fused_awq"),
        pytest.param(HFKernelAwqLinear, id="hf_kernel_awq"),
        pytest.param(TorchInt8AwqQuantLinear, id="torch_int8_awq"),
    ],
)
def test_awq_fused_post_init_calls_adapter(monkeypatch, kernel_cls):
    if kernel_cls is HFKernelAwqLinear:
        monkeypatch.setattr(
            HFKernelAwqLinear,
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


def test_hf_kernel_awq_backend_selection(monkeypatch):
    monkeypatch.setattr(
        HFKernelAwqLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.HF_KERNEL_AWQ,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is HFKernelAwqLinear
