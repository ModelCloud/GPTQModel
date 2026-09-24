# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact FP16 P32 input-Hadamard gate for the framework-neutral SM90 ABI."""

import ctypes
import math
import os

import pytest
import torch


class Config(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_bytes", ctypes.c_uint32),
        ("rows", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
    ]


def _library():
    path = os.environ.get("QVQ_WGMMA_RAW_LIBRARY")
    if not path:
        pytest.skip("set QVQ_WGMMA_RAW_LIBRARY to the raw device library")
    library = ctypes.CDLL(path)
    library.qvq_hadamard_input_raw_abi_version.restype = ctypes.c_uint32
    library.qvq_hadamard_input_raw_workspace_bytes.argtypes = [ctypes.POINTER(Config)]
    library.qvq_hadamard_input_raw_workspace_bytes.restype = ctypes.c_uint64
    library.qvq_hadamard_input_raw_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint64, ctypes.POINTER(Config), ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64,
    ]
    library.qvq_hadamard_input_raw_launch.restype = ctypes.c_int
    assert library.qvq_hadamard_input_raw_abi_version() == 1
    return library


def _reference(input_, scale):
    rows, width = input_.shape
    divisor = float(torch.tensor(math.sqrt(width), dtype=torch.float16))
    value = (input_ * scale).float().div(divisor).half()
    bit = 1
    while bit < width:
        pairs = value.reshape(rows, width // (2 * bit), 2, bit)
        even = pairs[:, :, 0, :].float()
        odd = pairs[:, :, 1, :].float()
        value = torch.stack(((even + odd).half(), (even - odd).half()), dim=2).reshape(rows, width)
        bit *= 2
    return value


@pytest.mark.parametrize("rows", [1, 16, 960])
def test_hadamard_input_raw_exact_and_changed_input_graph(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    library = _library()
    generator = torch.Generator().manual_seed(20260924 + rows)
    input_ = (torch.randn((rows, 8192), generator=generator) * 0.02).half().cuda()
    scale = (torch.randn((8192,), generator=generator) * 0.1 + 1.0).half().cuda()
    output = torch.empty_like(input_)
    config = Config(1, ctypes.sizeof(Config), rows, 8192)
    workspace_bytes = library.qvq_hadamard_input_raw_workspace_bytes(ctypes.byref(config))
    assert workspace_bytes == rows * 8192 * 2
    # The production FFI reuses the output as transient workspace.
    workspace = output
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch():
        error = ctypes.create_string_buffer(256)
        status = library.qvq_hadamard_input_raw_launch(
            input_.data_ptr(), scale.data_ptr(), output.data_ptr(), workspace.data_ptr(),
            workspace_bytes, ctypes.byref(config), stream.cuda_stream, error, len(error),
        )
        assert status == 0, error.value.decode()

    with torch.cuda.stream(stream):
        expected = _reference(input_, scale)
        launch()
    stream.synchronize()
    assert torch.isfinite(output).all()
    assert torch.equal(output, expected)

    if rows == 16:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        with torch.cuda.stream(stream):
            input_.normal_().mul_(0.01)
            expected_changed = _reference(input_, scale)
        stream.synchronize()
        graph.replay()
        stream.synchronize()
        assert torch.equal(output, expected_changed)
        graph.reset()


@pytest.mark.parametrize(
    "config",
    [
        Config(1, ctypes.sizeof(Config), 16, 2048),
        Config(1, ctypes.sizeof(Config), 0, 8192),
        Config(1, ctypes.sizeof(Config), 961, 8192),
        Config(2, ctypes.sizeof(Config), 16, 8192),
        Config(1, 0, 16, 8192),
    ],
)
def test_hadamard_input_raw_rejects_invalid_config(config):
    library = _library()
    assert library.qvq_hadamard_input_raw_workspace_bytes(ctypes.byref(config)) == 0
    error = ctypes.create_string_buffer(256)
    status = library.qvq_hadamard_input_raw_launch(
        None, None, None, None, 0, ctypes.byref(config), None, error, len(error),
    )
    assert status != 0
    assert error.value
