# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""C ABI integration with existing kernels; synthetic algebra, not model quality."""

import ctypes

import pytest
import torch

from gptqmodel.quantization.qvq_rank8 import P32WindowConfig, prepare_rank8
from gptqmodel.utils.qvq_window_abi import (
    WindowBuffer,
    WindowConfig,
    native_window_library,
    native_window_linear,
)


@pytest.fixture(autouse=True)
def hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("m", [1, 33, 128])
@pytest.mark.parametrize("n,bits", [(256, 2), (256, 2.5), (2048, 3), (2048, 3.5)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("bm,bn", [(0, 0), (32, 64), (32, 128), (64, 64), (64, 128), (128, 64), (128, 128)])
def test_native_abi_matches_existing_full_operator(m, n, bits, enabled, bm, bn):
    from test_qvq_grouped_runtime import _child
    from test_qvq_window_recovery import _kernel_rank8

    layer = _child("q_proj", in_features=2048, out_features=n, bits=bits, device="cuda").eval()
    if enabled:
        _kernel_rank8(layer)
    else:
        layer.rank8_A = torch.full((1,), float("nan"), device="cuda")
        layer.rank8_B = torch.full((1,), float("nan"), device="cuda")
    config = P32WindowConfig(
        algorithm="hopper_direct_decode_mma" if bm else "hopper_m16", recovery_mode="on" if enabled else "off",
        block_m=bm, block_n=bn,
    )
    x = torch.randn(m, 2048, device="cuda", dtype=torch.float16) * 0.01
    prepare_rank8(layer, config)
    with torch.no_grad():
        expected = layer(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = native_window_linear(layer, x, config)
    torch.cuda.current_stream().wait_stream(stream)
    error = (actual.double() - expected.double()).abs()
    assert torch.isfinite(actual).all()
    assert error.mean() <= 2e-3 and error.max() <= 0.046875
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_native_abi_rejects_bad_version_before_reading_buffers():
    library = native_window_library()
    error = ctypes.create_string_buffer(256)
    config = WindowConfig()
    config.abi_version = 99
    config.struct_bytes = ctypes.sizeof(config)
    status = library.qvq_p32_window_linear(
        *([WindowBuffer(1, 1)] * 10), ctypes.byref(config), None, error, len(error),
    )
    assert status != 0 and b"ABI version" in error.value


def test_native_disabled_pointers_and_external_capture_rejection(monkeypatch):
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", in_features=2048, out_features=256, device="cuda").eval()
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16) * 0.01
    library = native_window_library()
    function = library.qvq_p32_window_linear
    recorded = []

    def poison_and_record(*args):
        args = list(args)
        args[7] = WindowBuffer(1, 2**63)
        args[8] = WindowBuffer(2, 2**63)
        recorded[:] = args
        return function(*args)

    monkeypatch.setattr(library, "qvq_p32_window_linear", poison_and_record)
    output = native_window_linear(layer, x, P32WindowConfig(algorithm="hopper_m16"))
    assert torch.isfinite(output).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recorded[11] = torch.cuda.current_stream().cuda_stream
        status = function(*recorded)
        sentinel = x + 1
    assert status != 0 and b"external CUDA capture workspace" in recorded[12].value
    graph.replay()
    torch.testing.assert_close(sentinel, x + 1, atol=0, rtol=0)


@pytest.mark.parametrize("bm,bn", [(0, 0), (128, 128)])
def test_native_abi_maximum_m(bm, bn):
    test_native_abi_matches_existing_full_operator(8192, 256, 3, True, bm, bn)
