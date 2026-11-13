import pytest

from gptqmodel.quantization.config import METHOD
from gptqmodel.utils.backend import BACKEND, PROFILE, normalize_backend, normalize_profile


def test_marlin_backend_is_method_agnostic():
    assert normalize_backend(BACKEND.MARLIN, quant_method=METHOD.GPTQ) == BACKEND.MARLIN
    assert normalize_backend(BACKEND.MARLIN, quant_method=METHOD.AWQ) == BACKEND.MARLIN


def test_legacy_method_qualified_marlin_names_resolve_to_shared_backend():
    assert normalize_backend("gptq_marlin") == BACKEND.MARLIN
    assert normalize_backend("awq_marlin") == BACKEND.MARLIN
    assert normalize_backend("GPTQ_MARLIN") == BACKEND.MARLIN
    assert normalize_backend("AWQ_MARLIN") == BACKEND.MARLIN
    assert normalize_backend("gptq_marlin", quant_method=METHOD.GPTQ) == BACKEND.MARLIN
    assert normalize_backend("awq_marlin", quant_method=METHOD.AWQ) == BACKEND.MARLIN


def test_machete_backend_is_method_agnostic():
    assert normalize_backend(BACKEND.MACHETE, quant_method=METHOD.GPTQ) == BACKEND.MACHETE
    assert normalize_backend(BACKEND.MACHETE, quant_method=METHOD.AWQ) == BACKEND.MACHETE


def test_legacy_method_qualified_machete_names_resolve_to_shared_backend():
    assert normalize_backend("gptq_machete") == BACKEND.MACHETE
    assert normalize_backend("awq_machete") == BACKEND.MACHETE
    assert normalize_backend("GPTQ_MACHETE") == BACKEND.MACHETE
    assert normalize_backend("AWQ_MACHETE") == BACKEND.MACHETE
    assert normalize_backend("gptq_machete", quant_method=METHOD.GPTQ) == BACKEND.MACHETE
    assert normalize_backend("awq_machete", quant_method=METHOD.AWQ) == BACKEND.MACHETE


def test_swordfish_backend_is_method_agnostic():
    assert normalize_backend(BACKEND.SWORDFISH, quant_method=METHOD.GPTQ) == BACKEND.SWORDFISH
    assert normalize_backend(BACKEND.SWORDFISH, quant_method=METHOD.AWQ) == BACKEND.SWORDFISH


def test_legacy_method_qualified_swordfish_names_resolve_to_shared_backend():
    assert normalize_backend("gptq_swordfish") == BACKEND.SWORDFISH
    assert normalize_backend("awq_swordfish") == BACKEND.SWORDFISH
    assert normalize_backend("GPTQ_SWORDFISH") == BACKEND.SWORDFISH
    assert normalize_backend("AWQ_SWORDFISH") == BACKEND.SWORDFISH
    assert normalize_backend("gptq_swordfish", quant_method=METHOD.GPTQ) == BACKEND.SWORDFISH
    assert normalize_backend("awq_swordfish", quant_method=METHOD.AWQ) == BACKEND.SWORDFISH


def test_removed_mentaray_backend_names_are_rejected():
    with pytest.raises(ValueError):
        normalize_backend("mentaray", quant_method=METHOD.GPTQ)
    with pytest.raises(ValueError):
        normalize_backend("gptq_mentaray")
    with pytest.raises(ValueError):
        normalize_backend("awq_mentaray")


def test_legacy_torch_backend_normalizes_by_quant_method():
    assert normalize_backend("torch", quant_method=METHOD.GPTQ) == BACKEND.GPTQ_TORCH
    assert normalize_backend("torch", quant_method=METHOD.FP8) == BACKEND.FP8_TORCH
    assert normalize_backend("torch", quant_method=METHOD.EXL3) == BACKEND.EXL3_TORCH
    assert normalize_backend("torch", quant_method=METHOD.VPTQ) == BACKEND.VPTQ_TORCH


def test_awq_specific_legacy_backends_normalize_to_canonical_names():
    assert normalize_backend(BACKEND.TORCH_AWQ, quant_method=METHOD.AWQ) == BACKEND.AWQ_TORCH
    assert normalize_backend(BACKEND.BITBLAS_AWQ, quant_method=METHOD.AWQ) == BACKEND.AWQ_BITBLAS


def test_name_based_lookup_accepts_canonical_member_names():
    assert normalize_backend("MARLIN") == BACKEND.MARLIN
    assert normalize_backend("AWQ_GEMM_TRITON") == BACKEND.AWQ_GEMM_TRITON


@pytest.mark.parametrize(
    ("raw_profile", "expected"),
    [
        (None, PROFILE.AUTO),
        ("", PROFILE.AUTO),
        ("FAST", PROFILE.FAST),
        ("low-memory", PROFILE.LOW_MEMORY),
        ("low memory", PROFILE.LOW_MEMORY),
        (1, PROFILE.FAST),
        (2, PROFILE.LOW_MEMORY),
        (PROFILE.AUTO, PROFILE.AUTO),
    ],
)
def test_profile_normalization_accepts_enum_string_and_index_aliases(raw_profile, expected):
    assert normalize_profile(raw_profile) == expected


def test_profile_normalization_rejects_unknown_index():
    with pytest.raises(ValueError, match="Unknown profile index"):
        normalize_profile(99)
