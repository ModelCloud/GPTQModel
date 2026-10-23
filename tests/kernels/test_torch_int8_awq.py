# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_int8_awq import TorchInt8AwqLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


def test_awq_int8_constructor_rejects_persistent_expansion():
    with pytest.raises(NotImplementedError, match="AWQ_TORCH_INT8 is retired"):
        TorchInt8AwqLinear(bits=4, group_size=32, sym=False, desc_act=False,
                           in_features=256, out_features=64)


@pytest.mark.parametrize("backend", [BACKEND.AWQ_TORCH_INT8, BACKEND.TORCH_INT8_AWQ, BACKEND.TORCH_INT8])
def test_awq_int8_aliases_explain_migration(backend):
    with pytest.raises(ValueError, match="AWQ_TORCH_INT8 is retired"):
        select_quant_linear(bits=4, group_size=128, desc_act=False, sym=True,
                            device=None, backend=backend, format=FORMAT.GEMM,
                            quant_method=METHOD.AWQ, pack_dtype=torch.int32)


def test_awq_int8_validation_is_unavailable():
    supported, error = TorchInt8AwqLinear.validate_once()
    assert not supported
    assert isinstance(error, NotImplementedError)
    assert "BACKEND.AUTO" in str(error)
