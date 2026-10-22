# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import gptqmodel.nn_modules.exllamav3_torch as exllamav3_torch


@pytest.fixture(autouse=True)
def _clear_cpu_bf16x3_cache():
    exllamav3_torch._cpu_bf16x3_support.cache_clear()
    yield
    exllamav3_torch._cpu_bf16x3_support.cache_clear()


def _make_module(size: int = 1024, bits: int = 4, seed: int = 731) -> exllamav3_torch.ExllamaV3TorchLinear:
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        "trellis": torch.randint(
            -32768,
            32767,
            (size // 16, size // 16, bits * 16),
            dtype=torch.int16,
            generator=generator,
        ),
        "suh": (torch.rand(size, generator=generator) * 0.02 + 0.001).to(torch.float16),
        "svh": (torch.rand(size, generator=generator) * 0.02 + 0.001).to(torch.float16),
        "bias": torch.randn(size, generator=generator, dtype=torch.float16) * 0.01,
    }
    return exllamav3_torch.ExllamaV3TorchLinear.from_tensors(
        in_features=size,
        out_features=size,
        name="cpu_bf16x3",
        tensors=tensors,
    ).eval()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double()
    expected = expected.double()
    return ((actual - expected).norm() / expected.norm()).item()


def _oracle(module: exllamav3_torch.ExllamaV3TorchLinear, x: torch.Tensor) -> torch.Tensor:
    weight = module.get_weight_tensor(torch.float32).double()
    bias = module.bias.double()
    return x.double() @ weight + bias


def _matmul_precision() -> str | None:
    matmul = getattr(torch.backends.mkldnn, "matmul", None)
    return None if matmul is None else matmul.fp32_precision


def _fp32_matmul(
    module: exllamav3_torch.ExllamaV3TorchLinear,
    x: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    previous_precision = _matmul_precision()
    torch.backends.mkldnn.matmul.fp32_precision = precision
    try:
        output = x.float() @ module.get_weight_tensor(torch.float32)
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = previous_precision
    return output + module.bias.float()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 7, 25, 64])
def test_exllamav3_torch_cpu_bf16x3_matches_fp64_oracle(dtype, rows):
    module = _make_module()
    generator = torch.Generator().manual_seed(rows + (1000 if dtype == torch.float16 else 0))
    x = torch.randn(rows, module.in_features, dtype=dtype, generator=generator)
    previous_precision = _matmul_precision()

    with torch.inference_mode():
        actual = module(x)

    assert actual.dtype == dtype
    assert _matmul_precision() == previous_precision

    oracle = _oracle(module, x)
    raw_error = _relative_l2(actual, oracle)
    visible_error = _relative_l2(actual, oracle.to(dtype))

    assert raw_error <= 2e-3
    assert visible_error <= 2e-3


def test_exllamav3_torch_cpu_bf16x3_scope_restores_precision():
    if not exllamav3_torch._cpu_bf16x3_support():
        pytest.skip("oneDNN BF16x3 is unavailable on this host")

    previous_precision = _matmul_precision()
    if previous_precision is None:
        pytest.skip("torch does not expose oneDNN matmul precision")

    with exllamav3_torch._cpu_bf16x3_matmul():
        assert _matmul_precision() == "bf16"
    assert _matmul_precision() == previous_precision


def test_exllamav3_torch_cpu_bf16x3_arithmetic_matches_ieee():
    if not exllamav3_torch._cpu_bf16x3_support():
        pytest.skip("oneDNN BF16x3 is unavailable on this host")

    module = _make_module()
    x = torch.randn(25, module.in_features, dtype=torch.bfloat16)

    with torch.inference_mode():
        bf16x3 = _fp32_matmul(module, x, "bf16")
        ieee = _fp32_matmul(module, x, "ieee")

    assert _relative_l2(bf16x3, ieee) <= 1e-5


def test_exllamav3_torch_cpu_bf16x3_shape_gate():
    assert not exllamav3_torch._cpu_bf16x3_should_use(1, 512, 512)
    assert not exllamav3_torch._cpu_bf16x3_should_use(7, 512, 512)
    assert exllamav3_torch._cpu_bf16x3_should_use(32, 512, 512)
    assert exllamav3_torch._cpu_bf16x3_should_use(1, 1024, 1024)
    assert not exllamav3_torch._cpu_bf16x3_should_use(0, 1024, 1024)


def test_exllamav3_torch_cpu_bf16x3_falls_back_without_avx512_bf16(monkeypatch):
    monkeypatch.setattr(exllamav3_torch, "has_avx512_bf16", lambda: False)
    exllamav3_torch._cpu_bf16x3_support.cache_clear()

    assert not exllamav3_torch._cpu_bf16x3_support()

    module = _make_module()
    x = torch.randn(7, module.in_features, dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = module(x)

    oracle = _oracle(module, x)
    assert _relative_l2(actual, oracle.to(torch.bfloat16)) <= 2e-3


def test_exllamav3_torch_cpu_bf16x3_env_disable(monkeypatch):
    monkeypatch.setenv(exllamav3_torch._CPU_BF16X3_ENV, "0")
    exllamav3_torch._cpu_bf16x3_support.cache_clear()

    assert not exllamav3_torch._cpu_bf16x3_support()
