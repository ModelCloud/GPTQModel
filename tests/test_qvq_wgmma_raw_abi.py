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


class LaunchArg(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_void_p),
        ("size", ctypes.c_longlong),
        ("type", ctypes.c_int),
    ]


class LaunchDescriptor(ctypes.Structure):
    _fields_ = [
        ("kernel_symbol", ctypes.c_void_p),
        ("kernel_name", ctypes.c_char_p),
        ("grid_x", ctypes.c_uint),
        ("grid_y", ctypes.c_uint),
        ("grid_z", ctypes.c_uint),
        ("block_x", ctypes.c_uint),
        ("block_y", ctypes.c_uint),
        ("block_z", ctypes.c_uint),
        ("shared_memory_bytes", ctypes.c_uint),
        ("uses_pdl", ctypes.c_int),
        ("arg_count", ctypes.c_int),
        ("args", LaunchArg * 16),
        ("dependency_count", ctypes.c_int),
        ("dependencies", ctypes.c_int * 5),
    ]


class LaunchPlan(ctypes.Structure):
    _fields_ = [
        ("launch_count", ctypes.c_int),
        ("launches", LaunchDescriptor * 5),
        ("host_storage", ctypes.c_ulonglong * 128),
    ]


class RawDecodeConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "abi_version", "struct_bytes", "k", "n", "transition_bits",
        "output_layout",
    )]


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
    library.qvq_p32_wgmma_raw_launch_plan.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(RawConfig),
        ctypes.POINTER(LaunchPlan),
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    library.qvq_p32_wgmma_raw_launch_plan.restype = ctypes.c_int
    assert library.qvq_p32_wgmma_raw_abi_version() == 3
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
        3, ctypes.sizeof(RawConfig), m, k, n, round(2 * bits), split_count,
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


@pytest.mark.parametrize("k,n,block_m", [(2048, 8192, 160), (8192, 2048, 80)])
def test_w3_decoder_raw_abi_matches_wgmma_and_replays_dynamic_alt_id(k, n, block_m):
    library = _raw_library()
    library.qvq_p32_w3_decode_raw_abi_version.restype = ctypes.c_uint32
    assert library.qvq_p32_w3_decode_raw_abi_version() == 1
    decode = library.qvq_p32_w3_decode_raw_launch
    decode.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.POINTER(RawDecodeConfig), ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64]
    decode.restype = ctypes.c_int
    launch_plan = library.qvq_p32_w3_decode_raw_launch_plan
    launch_plan.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.POINTER(RawDecodeConfig), ctypes.POINTER(LaunchPlan),
        ctypes.c_void_p, ctypes.c_uint64]
    launch_plan.restype = ctypes.c_int

    device = torch.device("cuda")
    m = 960
    tiles = (k // 16) * (n // 16)
    generator = torch.Generator(device=device).manual_seed(20260924)
    window = torch.randint(-(2**31), 2**31 - 1, (tiles * 24,),
                           device=device, dtype=torch.int32, generator=generator)
    banks = torch.randint(0, 256, (tiles,), device=device,
                          dtype=torch.uint8, generator=generator)
    levels = torch.linspace(-1, 1, 256, device=device, dtype=torch.float16)
    alt = torch.tensor([2], device=device, dtype=torch.uint8)
    decoded = torch.empty((k, n), device=device, dtype=torch.float16)
    x = torch.randn((m, k), device=device, dtype=torch.float16,
                    generator=generator) * 0.02
    compressed = torch.empty((m, n), device=device, dtype=torch.float32)
    config = RawDecodeConfig(1, ctypes.sizeof(RawDecodeConfig), k, n, 6, 0)
    core = RawConfig(3, ctypes.sizeof(RawConfig), m, k, n, 6, 1, 5,
                     block_m, 64)
    error = ctypes.create_string_buffer(4096)
    plan = LaunchPlan()
    status = launch_plan(_ptr(window), _ptr(banks), _ptr(levels),
                         _ptr(alt), _ptr(decoded), ctypes.byref(config),
                         ctypes.byref(plan), error, len(error))
    assert status == 0, error.value.decode()
    assert plan.launch_count == 1
    assert plan.launches[0].kernel_name == b"qvq_p32_w3_decode_fp16"
    assert (plan.launches[0].grid_x, plan.launches[0].grid_y) == (n // 64, k // 256)
    assert plan.launches[0].arg_count == 9
    assert plan.launches[0].args[8].address == alt.data_ptr()

    def run_decode():
        status = decode(_ptr(window), _ptr(banks), _ptr(levels),
                        _ptr(alt), _ptr(decoded), ctypes.byref(config),
                        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                        error, len(error))
        assert status == 0, error.value.decode()

    status = library.qvq_p32_wgmma_raw_launch(
        _ptr(x), _ptr(window), _ptr(banks), _ptr(levels), _ptr(alt),
        _ptr(compressed), None, 0, ctypes.byref(core),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        error, len(error))
    assert status == 0, error.value.decode()
    run_decode()
    torch.cuda.synchronize()
    actual = torch.mm(x, decoded, out_dtype=torch.float32)
    torch.testing.assert_close(actual, compressed, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_decode()
    graph.replay()
    torch.cuda.synchronize()
    first_decoded = decoded.clone()
    alt.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(first_decoded, decoded)
    replay_decoded = decoded.clone()
    run_decode()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_decoded, decoded, atol=0, rtol=0)

    invalid = RawDecodeConfig(1, ctypes.sizeof(RawDecodeConfig), k, n, 5, 0)
    assert decode(_ptr(window), _ptr(banks), _ptr(levels), _ptr(alt),
                  _ptr(decoded), ctypes.byref(invalid),
                  ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                  error, len(error)) != 0


def test_raw_abi_bm64_m128_is_down_projection_only():
    library = _raw_library()
    config = RawConfig(3, ctypes.sizeof(RawConfig), 128, 2048, 2048, 6, 1, 3, 64, 64)
    plan = LaunchPlan()
    error = ctypes.create_string_buffer(4096)
    status = library.qvq_p32_wgmma_raw_launch_plan(
        None, None, None, None, None, None,
        ctypes.byref(config), ctypes.byref(plan), error, len(error),
    )
    assert status != 0
    assert b"unsupported QVQ WGMMA launch-plan geometry" in error.value


@pytest.mark.parametrize(
    "k,n,bits,block_m,block_n",
    [(2048, 512, 5, 80, 64), (2048, 2048, 5, 80, 64),
     (2048, 8192, 4, 80, 64), (2048, 8192, 6, 96, 64),
     (8192, 2048, 6, 160, 64),
     (8192, 2048, 6, 160, 128),
     (2048, 2048, 4, 64, 128),
     (2048, 8192, 4, 160, 128),
     (2048, 8192, 6, 80, 128)],
)
def test_raw_abi_m960_rejects_uncompiled_row_geometry(k, n, bits, block_m, block_n):
    library = _raw_library()
    config = RawConfig(3, ctypes.sizeof(RawConfig), 960, k, n, bits, 1, 5, block_m, block_n)
    plan = LaunchPlan()
    error = ctypes.create_string_buffer(4096)
    status = library.qvq_p32_wgmma_raw_launch_plan(
        None, None, None, None, None, None,
        ctypes.byref(config), ctypes.byref(plan), error, len(error),
    )
    assert status != 0
    assert b"unsupported QVQ WGMMA launch-plan geometry" in error.value


@pytest.mark.parametrize(
    "m,algorithm,k,n,bits,block_m,block_n,expected_grid_y",
    [
        *[(64, 2, 2048, 256, bits, 64, 64, 1) for bits in (2, 2.5, 3, 3.5)],
        *[(128, 3, 2048, 256, bits, 128, 64, 8) for bits in (2, 2.5, 3, 3.5)],
        # Production Llama 3.2 down projection. This catches row-reuse
        # schedule changes that the narrow K2048/N256 ABI gate cannot see.
        (128, 3, 8192, 2048, 3, 128, 64, 1),
        *[(128, 3, 8192, 2048, bits, 64, 64, 2) for bits in (2, 2.5, 3, 3.5)],
        # Narrow K/V retain four M16 rows/CTA; wide Q/MLP use five.
        (960, 5, 2048, 2048, 2, 80, 64, 12),
        (960, 5, 2048, 2048, 2, 64, 64, 15),
        (960, 5, 2048, 512, 2.5, 64, 64, 15),
        (960, 5, 2048, 512, 3.5, 64, 64, 15),
        (960, 5, 2048, 8192, 3, 80, 64, 12),
        (960, 5, 2048, 8192, 3, 160, 64, 6),
        (960, 5, 2048, 8192, 3, 160, 128, 6),
        (960, 5, 2048, 8192, 3, 64, 64, 15),
        (960, 5, 2048, 8192, 3, 64, 128, 15),
        (960, 5, 8192, 2048, 3, 80, 64, 12),
        (960, 5, 8192, 2048, 3, 64, 64, 15),
        (960, 5, 8192, 2048, 3, 64, 128, 15),
    ],
)
def test_raw_abi_direct_rows_matches_public_wgmma_and_needs_no_workspace(
    m, algorithm, k, n, bits, block_m, block_n, expected_grid_y
):
    from test_qvq_grouped_runtime import _child

    from gptqmodel.utils.qvq_cuda import _pgc16_levels
    from gptqmodel.utils.qvq_wgmma_cuda import qvq_p32_window_wgmma_tuned

    library = _raw_library()
    layer = _child(
        f"raw_m{m}", in_features=k, out_features=n, bits=bits,
        device="cuda", input_hadamard=False, output_hadamard=False,
    ).eval()
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.01
    window, banks, alt_id = layer._prepare_amd_p32_metadata(x.device)
    levels = _pgc16_levels(x.device, layer.codebook_version)
    alt_ids = torch.tensor([alt_id], device=x.device, dtype=torch.uint8)
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    config = RawConfig(
        3, ctypes.sizeof(RawConfig), m, k, n, round(2 * bits), 1,
        algorithm, block_m, block_n,
    )
    assert library.qvq_p32_wgmma_raw_workspace_bytes(ctypes.byref(config)) == 0
    plan = LaunchPlan()
    error = ctypes.create_string_buffer(4096)
    status = library.qvq_p32_wgmma_raw_launch_plan(
        _ptr(x), _ptr(window), _ptr(banks), _ptr(levels), _ptr(alt_ids),
        _ptr(output), ctypes.byref(config), ctypes.byref(plan), error, len(error),
    )
    assert status == 0, error.value.decode()
    assert plan.launch_count == 1
    assert plan.launches[0].kernel_symbol
    assert plan.launches[0].kernel_name == b"qvq_p32_wgmma_direct_rows"
    assert plan.launches[0].grid_x == n // block_n
    assert plan.launches[0].grid_y == expected_grid_y
    assert plan.launches[0].block_x == (288 if block_n == 128 else 160)
    assert plan.launches[0].arg_count == 11
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch():
        error = ctypes.create_string_buffer(4096)
        status = library.qvq_p32_wgmma_raw_launch(
            _ptr(x), _ptr(window), _ptr(banks), _ptr(levels), _ptr(alt_ids),
            _ptr(output), None, 0, ctypes.byref(config),
            ctypes.c_void_p(stream.cuda_stream), error, len(error),
        )
        assert status == 0, error.value.decode()

    with torch.cuda.stream(stream), torch.no_grad():
        expected = qvq_p32_window_wgmma_tuned(
            x, window, levels, banks, bits, out_features=n,
            bank_alt_id=alt_id, block_m=64 if block_m in (80, 160) else block_m, block_n=64,
        )
        if m == 128 and k == 8192 and block_m == 64:
            established = qvq_p32_window_wgmma_tuned(
                x, window, levels, banks, bits, out_features=n,
                bank_alt_id=alt_id, block_m=128, block_n=64,
            )
            torch.testing.assert_close(expected, established, atol=0, rtol=0)
        launch()
    stream.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    with torch.cuda.stream(stream):
        x.normal_().mul_(0.02)
        expected_changed = qvq_p32_window_wgmma_tuned(
            x, window, levels, banks, bits, out_features=n,
            bank_alt_id=alt_id, block_m=64 if block_m in (80, 160) else block_m, block_n=64,
        )
        if m == 128 and k == 8192 and block_m == 64:
            established_changed = qvq_p32_window_wgmma_tuned(
                x, window, levels, banks, bits, out_features=n,
                bank_alt_id=alt_id, block_m=128, block_n=64,
            )
            torch.testing.assert_close(expected_changed, established_changed, atol=0, rtol=0)
    stream.synchronize()
    graph.replay()
    stream.synchronize()
    torch.testing.assert_close(output, expected_changed, atol=0, rtol=0)
    graph.reset()


def test_raw_abi_grouped_gate_up_matches_independent_children_and_launch_plan():
    from test_qvq_grouped_runtime import _child

    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        unpack_qvq_binary_bank_ids,
    )
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.utils.qvq_cuda import _pgc16_levels
    from gptqmodel.utils.qvq_wgmma_cuda import qvq_p32_window_wgmma_tuned

    library = _raw_library()
    m, k, child_n, bits = 128, 2048, 8192, 3.0
    children = tuple(
        _child(
            name,
            in_features=k,
            out_features=child_n,
            bits=bits,
            alt_id=index + 1,
            seed=910 + index,
            device="cuda",
            input_hadamard=False,
            output_hadamard=False,
        )
        for index, name in enumerate(("gate_proj", "up_proj"))
    )
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.01
    child_windows = tuple(
        child._prepare_hopper_p32_window(x.device) for child in children
    )
    words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    k_tiles = k // 16
    child_n_tiles = child_n // 16
    window = torch.cat(
        tuple(
            value.reshape(k_tiles, child_n_tiles, words)
            for value in child_windows
        ),
        dim=1,
    ).reshape(-1, words).contiguous()
    child_selectors = tuple(
        pack_qvq_binary_bank_ids(
            unpack_qvq_binary_bank_ids(
                child.bank_ids, k_tiles * child_n_tiles * 8
            )
        ).to(device=x.device)
        for child in children
    )
    selectors = torch.cat(
        tuple(value.reshape(k_tiles, child_n_tiles) for value in child_selectors),
        dim=1,
    ).reshape(-1).contiguous()
    alt_ids = torch.tensor(
        [int(child.bank_alt_id.item()) for child in children],
        device=x.device,
        dtype=torch.uint8,
    )
    levels = _pgc16_levels(x.device, children[0].codebook_version)
    # The grouped ABI uses child-major flat storage: [gate MxN][up MxN].
    output = torch.empty(m * 2 * child_n, device=x.device, dtype=torch.float32)
    config = RawConfig(
        3, ctypes.sizeof(RawConfig), m, k, 2 * child_n, 6, 1,
        4, 128, 128,
    )
    assert library.qvq_p32_wgmma_raw_workspace_bytes(ctypes.byref(config)) == 0
    plan = LaunchPlan()
    error = ctypes.create_string_buffer(4096)
    status = library.qvq_p32_wgmma_raw_launch_plan(
        _ptr(x), _ptr(window), _ptr(selectors), _ptr(levels), _ptr(alt_ids),
        _ptr(output), ctypes.byref(config), ctypes.byref(plan), error, len(error),
    )
    assert status == 0, error.value.decode()
    assert plan.launch_count == 1
    assert plan.launches[0].kernel_name == b"qvq_p32_wgmma_direct_grouped_gate_up"
    assert plan.launches[0].grid_x == child_n // 128
    assert plan.launches[0].grid_y == 2

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        expected_children = tuple(
            qvq_p32_window_wgmma_tuned(
                x,
                child_window,
                levels,
                child_selector,
                bits,
                out_features=child_n,
                bank_alt_id=int(child.bank_alt_id.item()),
                block_m=128,
                block_n=64,
            )
            for child, child_window, child_selector in zip(
                children, child_windows, child_selectors, strict=True
            )
        )
        expected = torch.cat(
            tuple(
                child.reshape(-1) for child in expected_children
            ),
            dim=0,
        )
        status = library.qvq_p32_wgmma_raw_launch(
            _ptr(x), _ptr(window), _ptr(selectors), _ptr(levels), _ptr(alt_ids),
            _ptr(output), None, 0, ctypes.byref(config),
            ctypes.c_void_p(stream.cuda_stream), error, len(error),
        )
        assert status == 0, error.value.decode()
    stream.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
