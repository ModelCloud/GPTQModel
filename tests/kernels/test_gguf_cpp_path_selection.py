# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GGUF_CPP must pick its matmul path deterministically (#4219).

The CPU kernel used to choose between the native ggml matmul (which quantizes
activations to Q8_1/Q8_K) and the exact torch matmul by timing both, so the
numerics of a given model could differ between machines and runs.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel
from gptqmodel.utils.kernel import KernelConfig, configure_kernel


def _kernel() -> GGUFCppKernel:
    try:
        ok, error = GGUFCppKernel.cached_validate_once()
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"llama-cpp-python GGML bridge unavailable: {exc}")
    if not ok:
        pytest.skip(f"GGML CPU bridge unavailable: {error}")
    return GGUFCppKernel(
        bits="q4_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=16,
    )


def test_cpu_path_choice_is_capability_based_not_timed(monkeypatch):
    module = _kernel()
    x = torch.zeros(2, 32)

    def _timing_forbidden(*args, **kwargs):
        raise AssertionError("the ggml/torch path choice must not time the two paths")

    monkeypatch.setattr(module, "_benchmark_forward_runner", _timing_forbidden)
    assert module._use_ggml_cpu_matmul(x) is True
    assert module._use_ggml_cpu_matmul(x) is True

    # New modules take the explicit override; existing modules keep their snapshot.
    configure_kernel(KernelConfig(gguf_cpp_native_matmul=False))
    assert module._use_ggml_cpu_matmul(x) is True
    disabled = _kernel()
    assert disabled._use_ggml_cpu_matmul(x) is False
    configure_kernel(KernelConfig(gguf_cpp_native_matmul=True))
    assert module._use_ggml_cpu_matmul(x) is True
    assert disabled._use_ggml_cpu_matmul(x) is False
    assert _kernel()._use_ggml_cpu_matmul(x) is True


def test_cpu_path_choice_is_stable_across_repeated_calls(monkeypatch):
    module = _kernel()
    configure_kernel(KernelConfig())
    x = torch.zeros(4, 32)
    choices = {module._use_ggml_cpu_matmul(x) for _ in range(5)}
    assert choices == {True}


def test_cpu_ggml_failure_falls_back_to_torch_deterministically(monkeypatch):
    module = _kernel()
    x = torch.zeros(1, 32)
    fallback = torch.ones(1, 16, dtype=x.dtype)

    def _unusable(*args, **kwargs):
        raise RuntimeError("ggml has no vec_dot for this layout")

    monkeypatch.setattr(module, "_forward_ggml_cpu", _unusable)
    monkeypatch.setattr(
        module, "_forward_torch_cpu", lambda _x, bias=None: fallback.clone()
    )

    torch.testing.assert_close(module(x), fallback)
    # The failure is remembered, so the next call does not retry ggml.
    assert module._use_ggml_cpu_matmul(x) is False
    monkeypatch.setattr(module, "_forward_ggml_cpu", lambda _x: pytest.fail("ggml retried"))
    torch.testing.assert_close(module(x), fallback)


def test_bias_epilogue_tolerance_is_documented():
    from gptqmodel.nn_modules.qlinear.gguf_cpp import _finish_ggml_output

    doc = _finish_ggml_output.__doc__ or ""
    assert "1e-2" in doc
    assert "GGUF_TORCH" in doc
