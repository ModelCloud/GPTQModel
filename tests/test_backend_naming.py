from gptqmodel.quantization.config import METHOD
from gptqmodel.utils.backend import BACKEND, normalize_backend


def test_legacy_marlin_backend_normalizes_by_quant_method():
    assert normalize_backend(BACKEND.MARLIN, quant_method=METHOD.GPTQ) == BACKEND.GPTQ_MARLIN
    assert normalize_backend(BACKEND.MARLIN, quant_method=METHOD.AWQ) == BACKEND.AWQ_MARLIN


def test_legacy_torch_backend_normalizes_by_quant_method():
    assert normalize_backend("torch", quant_method=METHOD.GPTQ) == BACKEND.GPTQ_TORCH
    assert normalize_backend("torch", quant_method=METHOD.FP8) == BACKEND.FP8_TORCH
    assert normalize_backend("torch", quant_method=METHOD.EXL3) == BACKEND.EXL3_TORCH


def test_awq_specific_legacy_backends_normalize_to_canonical_names():
    assert normalize_backend(BACKEND.TORCH_AWQ, quant_method=METHOD.AWQ) == BACKEND.AWQ_TORCH
    assert normalize_backend(BACKEND.BITBLAS_AWQ, quant_method=METHOD.AWQ) == BACKEND.AWQ_BITBLAS


def test_name_based_lookup_accepts_canonical_member_names():
    assert normalize_backend("GPTQ_MARLIN") == BACKEND.GPTQ_MARLIN
    assert normalize_backend("AWQ_GEMM_TRITON") == BACKEND.AWQ_GEMM_TRITON
