# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MLX method registry and automatic backend selection on Apple Silicon."""

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("mlx.core")


@pytest.mark.parametrize("method,fmt,bits,group,sym,expected", [
    ("paroquant", "paroquant", 4, 128, True, "ParoMlxQuantLinear"),
    ("qqq", "qqq", 4, 128, True, "QQQMlxQuantLinear"),
    ("gguf", "gguf", 4, -1, True, "GGUFMlxQuantLinear"),
    ("fp8", "fp8", 8, -1, True, "FP8MlxQuantLinear"),
    ("bitsandbytes", "bitsandbytes", 4, -1, True, "BitsAndBytesMlxQuantLinear"),
])
def test_mlx_registry_has_method_holder(method, fmt, bits, group, sym, expected):
    from gptqmodel.models._const import DEVICE
    from gptqmodel.quantization.config import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    cls = select_quant_linear(
        bits=bits, group_size=group, desc_act=False, sym=sym,
        backend=BACKEND.MLX, format=FORMAT(fmt), quant_method=METHOD(method),
        device=DEVICE.MPS, pack_dtype=torch.int32, dtype=torch.float16,
    )
    assert cls.__name__ == expected


@pytest.mark.parametrize("method,fmt,bits,group,qtype,expected", [
    ("paroquant", "paroquant", 4, 128, None, "mlx"),
    ("qqq", "qqq", 4, 128, None, "mlx"),
    ("gguf", "gguf", 4, -1, "q4_k", "mlx"),
    ("gguf", "gguf", 4, -1, "q3_k", "auto"),
    ("fp8", "fp8", 8, -1, None, "mlx"),
    ("bitsandbytes", "bitsandbytes", 4, -1, None, "auto"),
    ("exl3", "exl3", 3.0, -1, None, "auto"),
])
def test_auto_selects_only_measured_or_exact_methods(method, fmt, bits, group, qtype, expected):
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.loader import _auto_select_mlx_backend
    from gptqmodel.quantization.config import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND

    class Config:
        def to_dict(self):
            return {"model_type": "qwen2"}

    qcfg = SimpleNamespace(
        bits=bits, group_size=group, desc_act=False, sym=True,
        pack_dtype=torch.int32, dynamic=None, rotation=None,
        runtime_bits=SimpleNamespace(name=qtype),
    )
    actual = _auto_select_mlx_backend(
        BACKEND.AUTO, DEVICE.MPS, Config(), qcfg,
        METHOD(method), FORMAT(fmt), None,
    )
    assert actual == BACKEND(expected)


@pytest.mark.parametrize("fmt,group,pack_dtype,expected", [
    ("gemv", 128, torch.int32, "AwqGemvMlxQuantLinear"),
    ("gemv", -1, torch.int32, "AwqGemvMlxQuantLinear"),
    ("gemv_fast", 16, torch.int16, "AwqGemvFastMlxQuantLinear"),
    ("llm-awq", 128, torch.int16, "LLMAwqMlxQuantLinear"),
])
def test_awq_variants_select_mlx_on_qwen38(fmt, group, pack_dtype, expected):
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.loader import _auto_select_mlx_backend
    from gptqmodel.quantization.config import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    class Config:
        def to_dict(self):
            return {"model_type": "qwen3_5"}

    selected = select_quant_linear(
        bits=4, group_size=group, desc_act=False, sym=True,
        backend=BACKEND.MLX, format=FORMAT(fmt), quant_method=METHOD.AWQ,
        device=DEVICE.MPS, pack_dtype=pack_dtype, dtype=torch.float16,
    )
    assert selected.__name__ == expected
    qcfg = SimpleNamespace(
        bits=4, group_size=group, desc_act=False, sym=True, pack_dtype=pack_dtype,
        dynamic=None, rotation=None,
    )
    assert _auto_select_mlx_backend(BACKEND.AUTO, DEVICE.MPS, Config(), qcfg,
                                    METHOD.AWQ, FORMAT(fmt), None) == BACKEND.MLX
