import pytest

from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import METHOD
from gptqmodel.utils.backend import BACKEND, PROFILE, normalize_backend, normalize_profile, resolve_activation_backend


def test_legacy_marlin_backend_normalizes_by_quant_method():
    assert normalize_backend(BACKEND.MARLIN, quant_method=METHOD.GPTQ) == BACKEND.GPTQ_MARLIN
    assert normalize_backend(BACKEND.MARLIN, quant_method=METHOD.AWQ) == BACKEND.AWQ_MARLIN


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


def test_awq_specific_legacy_backends_normalize_to_canonical_names():
    assert normalize_backend(BACKEND.TORCH_AWQ, quant_method=METHOD.AWQ) == BACKEND.AWQ_TORCH
    assert normalize_backend(BACKEND.BITBLAS_AWQ, quant_method=METHOD.AWQ) == BACKEND.AWQ_BITBLAS


def test_name_based_lookup_accepts_canonical_member_names():
    assert normalize_backend("GPTQ_MARLIN") == BACKEND.GPTQ_MARLIN
    assert normalize_backend("AWQ_GEMM_TRITON") == BACKEND.AWQ_GEMM_TRITON
    assert normalize_backend("GPTQ_TORCH_FP8") == BACKEND.GPTQ_TORCH_FP8


def test_activation_backend_resolution_keeps_awq_auto_for_kernel_validation():
    resolved = resolve_activation_backend(
        BACKEND.AUTO,
        quant_method=METHOD.AWQ,
        checkpoint_format=FORMAT.GEMM,
        input_activations={
            "dtype": "float8_e4m3fn",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        },
    )

    assert resolved == BACKEND.AUTO


def test_activation_backend_resolution_keeps_gptq_auto_for_kernel_validation():
    resolved = resolve_activation_backend(
        BACKEND.AUTO,
        quant_method=METHOD.GPTQ,
        checkpoint_format=FORMAT.GPTQ,
        input_activations={
            "dtype": "float8_e4m3fn",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        },
    )

    assert resolved == BACKEND.AUTO


def test_activation_backend_resolution_keeps_dynamic_awq_fp8_auto_for_kernel_validation():
    resolved = resolve_activation_backend(
        BACKEND.AUTO,
        quant_method=METHOD.AWQ,
        checkpoint_format=FORMAT.GEMM,
        input_activations={
            "dtype": "float8_e4m3fn",
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
        },
    )

    assert resolved == BACKEND.AUTO


def test_activation_backend_resolution_maps_explicit_gptq_fp8_alias_to_torch_kernel():
    resolved = resolve_activation_backend(
        BACKEND.GPTQ_TORCH_FP8,
        quant_method=METHOD.GPTQ,
        checkpoint_format=FORMAT.GPTQ,
        input_activations={
            "dtype": "float8_e4m3fn",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        },
    )

    assert resolved == BACKEND.GPTQ_TORCH


def test_activation_backend_resolution_rejects_non_dedicated_awq_backend():
    with pytest.raises(ValueError, match="AWQ activation quantization currently requires"):
        resolve_activation_backend(
            BACKEND.AWQ_GEMM,
            quant_method=METHOD.AWQ,
            checkpoint_format=FORMAT.GEMM,
            input_activations={
                "dtype": "float8_e4m3fn",
                "strategy": "tensor",
                "dynamic": False,
                "symmetric": True,
            },
        )


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
