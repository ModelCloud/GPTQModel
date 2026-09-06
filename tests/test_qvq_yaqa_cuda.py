# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from gptqmodel.quantization import qvq_yaqa_cuda as tiled


@pytest.fixture
def hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper")
    if tiled.triton is None:
        pytest.skip("requires Triton")
    previous = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield torch.device("cuda", torch.cuda.current_device())
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@pytest.mark.parametrize("batch", range(2, 17))
@pytest.mark.parametrize("channels", [5120, 17408])
def test_yaqa_projection_exact_repeated_seeds(hopper, batch, channels):
    for seed in (19, 107, 809):
        generator = torch.Generator(device=hopper).manual_seed(seed)
        # Both batch strides include padding; activations also have a row stride.
        a = torch.randn(batch, 128, channels, device=hopper, generator=generator)[:, ::2]
        p = torch.randn(batch, channels + 64, 256, device=hopper, generator=generator)[:, 32:-32]
        if seed == 809:
            p = torch.randn(
                batch, 2 * channels + 64, 256, device=hopper, generator=generator,
            )[:, 32:-32:2]
        if seed == 107:
            a[:, :, ::2].mul_(1e-10)
            a[:, :, 1::2].mul_(1e10)
        elif seed == 809:
            a[:, :, :128] = 0
            p[:, ::2] = -p[:, 1::2]
        expected = torch.bmm(a, p)
        with patch.object(tiled.torch, "bmm", side_effect=AssertionError("unexpected fallback")):
            for _ in range(10):
                actual = tiled.project(a, p)
                assert torch.equal(actual, expected), (batch, channels, seed)


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 1e-38, float("inf"), float("nan")])
def test_yaqa_projection_boundaries_and_stream(hopper, value):
    stream = torch.cuda.Stream(device=hopper)
    with torch.cuda.stream(stream):
        a = torch.full((4, 64, 5120), value, device=hopper)
        p = torch.ones((4, 5120, 256), device=hopper)
        expected = torch.bmm(a, p)
        actual = tiled.project(a, p)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("case", ["batch", "tokens", "channels", "rank", "a_stride", "p_stride",
                                  "a_dtype", "p_dtype", "a_grad", "p_grad", "architecture", "triton", "precision"])
def test_yaqa_projection_cuda_fallback(hopper, case):
    a = torch.zeros((4, 64, 5120), device=hopper)
    p = torch.zeros((4, 5120, 256), device=hopper)
    if case == "batch":
        a, p = a[:1], p[:1]
    elif case == "tokens":
        a = a[:, :63]
    elif case == "channels":
        a, p = a[:, :, :-1], p[:, :-1]
    elif case == "rank":
        p = p[:, :, :-1]
    elif case == "a_stride":
        a = torch.zeros((4, 64, 10240), device=hopper)[:, :, ::2]
    elif case == "p_stride":
        p = torch.zeros((4, 5120, 512), device=hopper)[:, :, ::2]
    elif case == "a_dtype":
        a = a.double()
    elif case == "p_dtype":
        p = p.double()
    elif case == "a_grad":
        a.requires_grad_(True)
    elif case == "p_grad":
        p.requires_grad_(True)
    if case == "precision":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    with patch.object(tiled, "_is_hopper", return_value=case != "architecture"), \
         patch.object(tiled, "triton", None if case == "triton" else tiled.triton), \
         patch.object(tiled.torch, "bmm", return_value=object()) as fallback:
        assert tiled.project(a, p) is fallback.return_value
        fallback.assert_called_once_with(a, p)


def test_yaqa_projection_cpu_fallback():
    a, p = torch.randn(3, 7, 11), torch.randn(3, 11, 5)
    assert torch.equal(tiled.project(a, p), torch.bmm(a, p))


def test_yaqa_projection_without_triton():
    import builtins
    import importlib.util

    original_import = builtins.__import__

    def without_triton(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("test dependency-minimal CPU environment")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("_yaqa_without_triton", tiled.__file__)
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, "__import__", side_effect=without_triton):
        spec.loader.exec_module(module)
    a, p = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    assert module.triton is None
    assert torch.equal(module.project(a, p), torch.bmm(a, p))
