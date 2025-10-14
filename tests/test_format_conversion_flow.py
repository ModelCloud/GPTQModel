# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import threading
from unittest import mock

import torch

from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig
from gptqmodel.utils.model import pack_module


class _DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, 1))

    def to(self, *_args, **_kwargs):
        return super().to(*_args, **_kwargs)


class _DummyQuantModule:
    def __init__(self):
        self.bits = 4
        self.pack_dtype = torch.int32

    QUANT_TYPE = "gptq"

    def to(self, *_args, **_kwargs):
        return self

    def pack(self, **_kwargs):
        pass

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())

    def pack_block(self, **_kwargs):
        pass

    def pack_original(self, **_kwargs):
        pass

    def pack_gpu(self, **_kwargs):
        pass

    def qzero_format(self, format: int | None = None):
        if format is not None:
            self._fmt = format
        return getattr(self, "_fmt", 2)


def _make_quant_linear_cls(requires_v2: bool):
    return type(
        "DummyQuantLinear",
        (),
        {
            "QUANT_TYPE": "gptq",
            "REQUIRES_FORMAT_V2": requires_v2,
        },
    )


def _run_pack(quant_cfg: QuantizeConfig, requires_v2: bool) -> int:
    dummy_module = _DummyQuantModule()
    qmodules = {"layer": dummy_module}
    layers = {"layer": _DummyLayer()}
    q_scales = torch.zeros(1, 1)
    q_zeros = torch.zeros(1, 1, dtype=torch.int32)
    q_g_idx = torch.zeros(1, dtype=torch.int32)
    lock = threading.Lock()

    quant_linear_cls = _make_quant_linear_cls(requires_v2=requires_v2)
    assert getattr(quant_linear_cls, "REQUIRES_FORMAT_V2") is requires_v2

    with mock.patch("gptqmodel.utils.model.convert_gptq_v2_to_v1_format_module") as convert_mock:
        pack_module(
            name="layer",
            qModules=qmodules,
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=q_g_idx,
            layers=layers,
            quant_linear_cls=quant_linear_cls,
            lock=lock,
            quantize_config=quant_cfg,
        )

    return convert_mock.call_count


def test_pack_module_converts_for_gptq_requires_v2():
    cfg = QuantizeConfig(bits=4, quant_method=METHOD.GPTQ, format=FORMAT.GPTQ, offload_to_disk=False)
    calls = _run_pack(cfg, requires_v2=True)
    assert calls == 1


def test_pack_module_skips_for_non_gptq_method():
    cfg = QuantizeConfig(bits=4, quant_method=METHOD.AWQ, format=FORMAT.GEMM, offload_to_disk=False)
    calls = _run_pack(cfg, requires_v2=True)
    assert calls == 0


def test_pack_module_skips_when_kernel_uses_v1():
    cfg = QuantizeConfig(bits=4, quant_method=METHOD.GPTQ, format=FORMAT.GPTQ, offload_to_disk=False)
    calls = _run_pack(cfg, requires_v2=False)
    assert calls == 0
