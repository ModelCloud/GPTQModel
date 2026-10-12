# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module
from gptqmodel import BACKEND
from gptqmodel.models import loader
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig


@pytest.fixture
def bitblas_validate_ready(monkeypatch):
    """Let BitBLAS kernel validation run without importing the CUDA-only runtime."""
    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    bitblas_module.BitblasLinear.cached_validate_once.cache_clear()
    yield
    bitblas_module.BitblasLinear.cached_validate_once.cache_clear()


def _gptq_config(*, sym: bool = True) -> QuantizeConfig:
    return QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=sym,
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        pack_dtype=torch.int32,
    )


def test_explicit_bitblas_gptq_coerces_unsupported_bfloat16(bitblas_validate_ready):
    dtype = loader._coerce_quantized_kernel_dtype(
        backend=BACKEND.BITBLAS,
        qcfg=_gptq_config(sym=True),
        dtype=torch.bfloat16,
    )

    assert dtype == torch.float16


def test_explicit_bitblas_gptq_keeps_supported_bfloat16_for_unsigned_weights(
    bitblas_validate_ready,
):
    dtype = loader._coerce_quantized_kernel_dtype(
        backend=BACKEND.BITBLAS,
        qcfg=_gptq_config(sym=False),
        dtype=torch.bfloat16,
    )

    assert dtype == torch.bfloat16


def test_explicit_bitblas_gptq_keeps_float16(bitblas_validate_ready):
    dtype = loader._coerce_quantized_kernel_dtype(
        backend=BACKEND.BITBLAS,
        qcfg=_gptq_config(sym=True),
        dtype=torch.float16,
    )

    assert dtype == torch.float16


def test_auto_bitblas_gptq_keeps_requested_dtype(bitblas_validate_ready):
    dtype = loader._coerce_quantized_kernel_dtype(
        backend=BACKEND.AUTO,
        qcfg=_gptq_config(sym=True),
        dtype=torch.bfloat16,
    )

    assert dtype == torch.bfloat16
