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


def test_native_window_only_uses_window_payload_without_planar_trellis():
    """The native ABI must accept the production window-only ownership mode."""
    from test_qvq_grouped_runtime import _child
    from test_qvq_window_recovery import export_window_package, load_window_package

    source = _child(
        "window_only_native", in_features=2048, out_features=2048,
        bits=2, device="cuda", input_hadamard=False, output_hadamard=False,
    ).eval()
    source.trellis.random_(-2147483648, 2147483647)
    source.post_init()
    package = export_window_package(source)
    layer = load_window_package(package, device="cuda")
    assert layer.window_only and layer.trellis is None
    x = torch.randn(1, source.in_features, device="cuda", dtype=torch.float16) * 0.01
    config = P32WindowConfig(algorithm="hopper_m16", recovery_mode="off")
    with torch.no_grad():
        expected = layer(x)
        actual = native_window_linear(layer, x, config)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_native_abi_admits_transform_free_composite_qwen_shape():
    """Composite Qwen tiles pass ABI shape validation without Hadamard."""
    library = native_window_library()
    error = ctypes.create_string_buffer(256)
    config = WindowConfig(
        3, ctypes.sizeof(WindowConfig), 1, 5120, 17408, 6, 3, 1,
        0, 0, 256, 0, 2, 1, 1, 8192, 0, 0, 0,
    )
    status = library.qvq_p32_window_linear(
        *([WindowBuffer(None, 0)] * 10), ctypes.byref(config), None, error, len(error),
    )
    assert status != 0
    assert b"input pointer is null" in error.value
    assert b"power-of-two" not in error.value


def test_native_abi_keeps_hadamard_power_of_two_guard():
    library = native_window_library()
    error = ctypes.create_string_buffer(256)
    config = WindowConfig(
        3, ctypes.sizeof(WindowConfig), 1, 5120, 17408, 6, 3, 1,
        0, 0, 256, 0, 2, 1, 1, 8192, 1, 0, 0,
    )
    status = library.qvq_p32_window_linear(
        *([WindowBuffer(None, 0)] * 10), ctypes.byref(config), None, error, len(error),
    )
    assert status != 0 and b"input Hadamard" in error.value


def test_native_composite_qwen_shape_matches_window_reference():
    """Exercise the real decoder for the production 5120 x 17408 shape."""
    from test_qvq_grouped_runtime import _child

    layer = _child(
        "qwen_composite",
        in_features=5120,
        out_features=17408,
        bits=3.0,
        device="cuda",
        input_hadamard=False,
        output_hadamard=False,
    )
    x = torch.randn(1, 5120, device="cuda", dtype=torch.float16) * 0.01
    config = P32WindowConfig(algorithm="hopper_m16", recovery_mode="off")
    prepare_rank8(layer, config)
    with torch.no_grad():
        expected = layer(x)
        actual = native_window_linear(layer, x, config)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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


def test_native_window_linear_rejects_capture_before_preparation():
    """The raw ABI entry point must fail before allocating inside capture."""
    from test_qvq_grouped_runtime import _child

    layer = _child("capture_guard", in_features=2048, out_features=256, device="cuda").eval()
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16) * 0.01
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="prepare and replay a native window graph"), torch.cuda.graph(graph):
        native_window_linear(layer, x, P32WindowConfig(algorithm="hopper_m16"))


@pytest.mark.parametrize("bm,bn", [(0, 0), (128, 128)])
def test_native_abi_maximum_m(bm, bn):
    test_native_abi_matches_existing_full_operator(8192, 256, 3, True, bm, bn)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("bm,bn", [(0, 0), (32, 64), (32, 128), (64, 64), (64, 128), (128, 64), (128, 128)])
@pytest.mark.parametrize("n,bits", [(256, 2), (256, 2.5), (2048, 3), (2048, 3.5)])
@pytest.mark.parametrize("m", [1, 33, 128])
def test_native_prepared_graph_owns_workspace_and_composes(monkeypatch, enabled, bm, bn, n, bits, m):
    from test_qvq_grouped_runtime import _child
    from test_qvq_window_recovery import _kernel_rank8

    layer = _child("q_proj", in_features=2048, out_features=n, bits=bits, device="cuda").eval()
    if enabled:
        _kernel_rank8(layer)
    config = P32WindowConfig(
        algorithm="hopper_direct_decode_mma" if bm else "hopper_m16",
        block_m=bm, block_n=bn, recovery_mode="on" if enabled else "off",
    )
    x = torch.randn(m, 2048, device="cuda", dtype=torch.float16) * 0.01
    library = native_window_library()
    original = library.qvq_p32_window_linear
    recorded = []

    def record(*args):
        recorded[:] = args
        return original(*args)

    monkeypatch.setattr(library, "qvq_p32_window_linear", record)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output = native_window_linear(layer, x, config)
    buffers = (WindowBuffer * 10)(*recorded[:10])
    if not enabled:
        buffers[7] = WindowBuffer(1, 2**63)
        buffers[8] = WindowBuffer(2, 2**63)
    handle = ctypes.c_void_p()
    error = ctypes.create_string_buffer(4096)
    status = library.qvq_p32_window_graph_create(
        buffers, recorded[10], stream.cuda_stream, ctypes.byref(handle), error, len(error),
    )
    assert status == 0, error.value
    parent = None
    try:
        # Buffer values change, addresses stay fixed. Warmup contents must not
        # be baked into the graph, and unrelated allocator pressure cannot
        # reclaim its private intermediate buffers.
        with torch.cuda.stream(stream), torch.no_grad():
            x.mul_(2)
            expected = layer(x)
            status = library.qvq_p32_window_graph_run(handle, stream.cuda_stream, error, len(error))
            assert status == 0, error.value
        stream.synchronize()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        other = torch.cuda.Stream()
        assert library.qvq_p32_window_graph_run(handle, other.cuda_stream, error, len(error)) != 0
        assert b"owning stream" in error.value
        parent = torch.cuda.CUDAGraph()
        with torch.cuda.graph(parent, stream=stream):
            if bm == 0 and m == 1:
                pending = ctypes.c_void_p(1)
                rejected = library.qvq_p32_window_graph_create(
                    buffers, recorded[10], stream.cuda_stream, ctypes.byref(pending), error, len(error),
                )
                assert rejected != 0 and pending.value is None
                assert b"outside capture" in error.value
                rejected = library.qvq_p32_window_graph_destroy(handle, error, len(error))
                assert rejected != 0 and b"during capture" in error.value
            x.mul_(0.5)
            status = library.qvq_p32_window_graph_run(handle, stream.cuda_stream, error, len(error))
            after = output + 1
        assert status == 0, error.value
        for _ in range(3):
            with torch.cuda.stream(stream), torch.no_grad():
                noise = torch.empty(1024 * 1024, device="cuda")
                noise.fill_(float("nan"))
                del noise
                torch.cuda.empty_cache()
                expected = layer(x * 0.5)
                parent.replay()
            stream.synchronize()
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            torch.testing.assert_close(after, expected + 1, atol=0, rtol=0)
    finally:
        if parent is not None:
            stream.synchronize()
            parent.reset()
        status = library.qvq_p32_window_graph_destroy(handle, error, len(error))
        assert status == 0, error.value
