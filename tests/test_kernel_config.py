# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the explicit kernel/thread policy (issue #4180).

Every former ``GPTQMODEL_*`` / ``GPTQ_TORCH_*`` behaviour switch is now a field
on :class:`KernelConfig`. These tests pin the three properties that make that
safe: the defaults match the retired variables' defaults, the config is
process-wide and replaceable, and a bad value is rejected instead of silently
clamped into a different numerical path.
"""

import inspect

import pytest

from gptqmodel.utils.kernel import (
    KernelConfig,
    configure_kernel,
    get_kernel_config,
    reset_kernel_config,
    resolve_kernel_config,
)


def test_defaults_are_the_retired_environment_defaults():
    assert KernelConfig().to_dict() == {
        **KernelConfig().to_dict(),
        # Spot-check the fields that used to gate a numerical path.
        "gptq_fused_block_update": True,
        "floatx_cpu_native_fp8": False,
        "floatx_cpu_simd_level": 2,
        "torch_streaming": False,
        "torch_stream_tile": 512,
        "torch_triton_dequant": None,
        "torch_rocm_gemv": True,
        "torch_rocm_gemm": True,
        "torch_cache_weights": False,
        "torch_lookahead": False,
        "marlin_fp32": True,
        "fp32_accum": True,
        "awq_fused_splitk_reduce": True,
        "awq_triton_schedule": "auto",
        "blockfloat_cublas": True,
        "swordfish_deterministic": False,
        "swordfish_dense_m": 0,
        "bitsandbytes_cpu_int8_dense": True,
        "bitsandbytes_triton": False,
        "disable_bitblas": False,
        "gsq_disable_cuda_relaxation": False,
        "gsq_disable_hip_relaxation": False,
        "cpu_workers": None,
        "cuda_workers": None,
        "max_packers": None,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("gptq_fused_block_update", "1"),
        ("floatx_cpu_native_fp8", "no"),
        ("blockfloat_cublas", 1),
        ("swordfish_deterministic", "yes"),
        ("awq_fused_splitk_reduce", 0),
        ("torch_triton_dequant", "0"),
        ("torch_stream_tile", 0),
        ("npu_cholesky_block_size", 0),
        ("floatx_cpu_simd_level", -1),
        ("swordfish_dense_m", -1),
        ("paro_rotate_cta_m", 5),
        ("paro_rotate_row_pad", 1),
        ("awq_triton_schedule", "fused"),
        ("cpu_workers", 0),
        ("rocm_quant_blas", 7),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        KernelConfig(**{field: value})


def test_policy_is_process_wide_and_replaceable():
    try:
        configured = configure_kernel(KernelConfig(marlin_fp32=False))
        assert get_kernel_config() is configured
        assert get_kernel_config().marlin_fp32 is False

        replaced = configure_kernel({"marlin_fp32": True})
        assert get_kernel_config() is replaced
        assert get_kernel_config().marlin_fp32 is True
    finally:
        reset_kernel_config()


def test_owner_config_wins_over_the_process_policy():
    class _Owner:
        kernel = KernelConfig(fp32_accum=False)

    configure_kernel(KernelConfig(fp32_accum=True))
    try:
        assert resolve_kernel_config(_Owner()).fp32_accum is False
        assert resolve_kernel_config(None).fp32_accum is True
    finally:
        reset_kernel_config()


def test_rocm_quant_blas_keeps_an_explicit_empty_opt_out():
    """``""`` is a documented opt-out spelling, distinct from ``None``."""

    assert KernelConfig().rocm_quant_blas is None
    assert KernelConfig(rocm_quant_blas="").rocm_quant_blas == ""
    assert KernelConfig(rocm_quant_blas="  HIPBLASLT ").rocm_quant_blas == "hipblaslt"


def test_kernel_field_is_runtime_only():
    """The policy must not leak into the serialized checkpoint payload."""

    from gptqmodel.quantization.config import QuantizeConfig

    config = QuantizeConfig(bits=4, group_size=128, kernel=KernelConfig(disable_bitblas=True))
    payload = config.to_dict()
    assert "kernel" not in payload
    assert "kernel" not in payload.get("meta", {})


def test_loader_exposes_the_kernel_argument():
    from gptqmodel.models.auto import GPTQModel

    assert "kernel" in inspect.signature(GPTQModel.load).parameters
    assert "kernel" in inspect.signature(GPTQModel.from_quantized).parameters

    # ``ModelLoader(cls)`` decorates the model class in place, so the module
    # global is a factory rather than a class. The loader entry point must
    # accept the same policy object the public API does.
    from gptqmodel.models import loader as loader_module

    source = inspect.getsource(loader_module.ModelLoader)
    assert "kernel: Optional[Union[KernelConfig, Dict]] = None" in source
    assert "configure_kernel(kernel)" in source


def test_paro_triton_linear_reads_the_policy():
    """Every kernel that reads the policy must import the accessor.

    A missing import compiles fine and only raises when the class is built, so
    the default backend selection never reached it on a CPU-only host. Import
    the module and assert the symbol resolves before any device work.
    """

    pytest.importorskip("triton")
    from gptqmodel.nn_modules.qlinear import paro_triton

    assert paro_triton.resolve_kernel_config is resolve_kernel_config

    # The constructor path that reads the policy must resolve its name.
    source = inspect.getsource(paro_triton.ParoTritonLinear.__init__)
    assert "resolve_kernel_config(" in source
