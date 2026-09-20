# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Framework-neutral WGMMA ABI gates for external compiler runtimes."""

import ctypes
import os

import pytest
import torch


class RawConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_bytes", ctypes.c_uint32),
        ("m", ctypes.c_uint32),
        ("k", ctypes.c_uint32),
        ("n", ctypes.c_uint32),
        ("transition_bits", ctypes.c_uint32),
        ("split_count", ctypes.c_uint32),
        ("algorithm", ctypes.c_uint32),
        ("block_m", ctypes.c_uint32),
        ("block_n", ctypes.c_uint32),
    ]


def _raw_library():
    path = os.environ.get("QVQ_WGMMA_RAW_LIBRARY")
    if not path:
        pytest.skip("set QVQ_WGMMA_RAW_LIBRARY to the device-only ABI library")
    library = ctypes.CDLL(path)
    library.qvq_p32_wgmma_raw_abi_version.restype = ctypes.c_uint32
    library.qvq_p32_wgmma_raw_workspace_bytes.argtypes = [ctypes.POINTER(RawConfig)]
    library.qvq_p32_wgmma_raw_workspace_bytes.restype = ctypes.c_uint64
    library.qvq_p32_wgmma_raw_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(RawConfig),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    library.qvq_p32_wgmma_raw_launch.restype = ctypes.c_int
    assert library.qvq_p32_wgmma_raw_abi_version() == 1
    return library


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.fixture(autouse=True)
def hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize(
    "m,bits,k,split_count",
    [(1, 2, 2048, 1), (16, 2.5, 2048, 1), (1, 3, 8192, 16), (16, 3.5, 2048, 1)],
)
def test_raw_abi_matches_public_wgmma_and_graph_replays_changed_input(
    m, bits, k, split_count
):
    from test_qvq_grouped_runtime import _child

    from gptqmodel.utils.qvq_cuda import _pgc16_levels
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_m16_tma_ordered_split,
    )

    library = _raw_library()
    # Match production Hopper scheduling: ordinary K2048 projections are
    # unsplit, while Llama's K8192 down projection uses 16 ordered planes.
    n = 256
    layer = _child(
        "raw_core", in_features=k, out_features=n, bits=bits,
        device="cuda", input_hadamard=False, output_hadamard=False,
    ).eval()
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.01
    window, banks, alt_id = layer._prepare_amd_p32_metadata(x.device)
    levels = _pgc16_levels(x.device, layer.codebook_version)
    alt_ids = torch.tensor([alt_id], device=x.device, dtype=torch.uint8)
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    config = RawConfig(
        1, ctypes.sizeof(RawConfig), m, k, n, round(2 * bits), split_count,
        1, 0, 0,
    )
    workspace_bytes = library.qvq_p32_wgmma_raw_workspace_bytes(ctypes.byref(config))
    workspace = torch.empty(workspace_bytes, device=x.device, dtype=torch.uint8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch():
        error = ctypes.create_string_buffer(4096)
        status = library.qvq_p32_wgmma_raw_launch(
            _ptr(x), _ptr(window), _ptr(banks), _ptr(levels), _ptr(alt_ids),
            _ptr(output), _ptr(workspace), workspace_bytes, ctypes.byref(config),
            ctypes.c_void_p(stream.cuda_stream), error, len(error),
        )
        assert status == 0, error.value.decode()

    with torch.cuda.stream(stream), torch.no_grad():
        expected = qvq_p32_window_wgmma_m16_tma_ordered_split(
            x, window, levels, banks, bits, out_features=n,
            bank_alt_id=alt_id, split_count=split_count,
        )
        launch()
    stream.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    with torch.cuda.stream(stream):
        x.normal_().mul_(0.02)
        expected_changed = qvq_p32_window_wgmma_m16_tma_ordered_split(
            x, window, levels, banks, bits, out_features=n,
            bank_alt_id=alt_id, split_count=split_count,
        )
    stream.synchronize()
    graph.replay()
    stream.synchronize()
    torch.testing.assert_close(output, expected_changed, atol=0, rtol=0)
    graph.reset()
