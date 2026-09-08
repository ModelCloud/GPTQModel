"""Regression for StaticK=1 accidentally selected by the F6 shared-LUT flag.

Set QVQ_P32_TEST_LIBRARY to a freshly built public P32 ABI library. These are
synthetic kernel-contract tests, not quantization-quality measurements.
"""

import ctypes
import os
from pathlib import Path

import pytest


def test_static_k_template_guard_and_shared_level_arguments():
    source = (
        Path(__file__).parents[1] / "gptqmodel_ext/qvq/p32/qvq_p32_cuda.cu"
    ).read_text()
    assert "StaticK == 0 || StaticK % kTileRows == 0" in source
    dispatch = source[source.index("const bool use_llama_f6_shared_levels") :]
    assert dispatch.count("0, 0, 0, true><<<grid, Threads, 0, cuda_stream>>>") == 4
    assert not any(
        line.strip().startswith("0, 0, true><<<") for line in dispatch.splitlines()
    )


@pytest.fixture(scope="module", params=[7, 19, 41])
def native_case(request):
    path = os.environ.get("QVQ_P32_TEST_LIBRARY")
    if not path:
        pytest.skip("QVQ_P32_TEST_LIBRARY must name the public native ABI build")
    import torch
    from gptqmodel.quantization.qvq import (
        reconstruct_p32_window_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Public P32 CUDA ABI requires SM80")
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.qvq_last_error.restype = ctypes.c_char_p
    lib.qvq_p32_window.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_int] * 10 + [ctypes.c_void_p]
    )
    lib.qvq_p32_window.restype = ctypes.c_int
    generator = torch.Generator(device="cuda").manual_seed(request.param)
    k, n = 8192, 2048
    planar = torch.randint(
        0,
        1 << 32,
        ((k // 16) * (n // 16), 24),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    ).int()
    window = repack_p32_planar_to_window(planar, bits=3)
    banks = torch.randint(
        0,
        256,
        (planar.shape[0],),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    alternate = torch.tensor(3, dtype=torch.uint8, device="cuda")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    dense = reconstruct_p32_window_inner_weight(
        window,
        bits=3,
        in_features=k,
        out_features=n,
        bank_ids=banks,
        bank_alt_id=alternate,
    )
    inputs = (torch.randn((4, k), generator=generator, device="cuda") * 0.1).half()
    return torch, lib, window, banks, alternate, levels, dense, inputs


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("splits", [1, 8])
@pytest.mark.parametrize("reduction", [1, 2])
def test_f6_shared_levels_native_stride_eager_and_graph(
    native_case, rows, splits, reduction, request
):
    torch, lib, window, banks, alternate, levels, dense, inputs = native_case
    x = inputs[:rows].clone()
    output = torch.empty((rows, 2048), device="cuda", dtype=torch.float32)
    partials = torch.empty((splits, rows, 2048), device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    measurements = []

    def launch():
        rc = lib.qvq_p32_window(
            x.data_ptr(),
            window.data_ptr(),
            levels.data_ptr(),
            banks.data_ptr(),
            alternate.data_ptr(),
            output.data_ptr(),
            partials.data_ptr(),
            rows,
            8192,
            2048,
            6,
            splits,
            1,
            128,
            2,
            0,
            reduction,
            stream.cuda_stream,
        )
        assert rc == 0, lib.qvq_last_error().decode()

    def check():
        # S=1 writes directly to output regardless of reduction mode.
        actual = partials.sum(0) if reduction == 2 and splits > 1 else output
        reference = x.float() @ dense
        error = (actual - reference).abs()
        assert torch.isfinite(actual).all()
        mean, maximum = error.mean().item(), error.max().item()
        measurements.append((mean, maximum))
        assert mean <= 2e-3, (mean, maximum)
        assert maximum <= 0.046875, (mean, maximum)

    with torch.cuda.stream(stream):
        launch()
        check()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    with torch.cuda.stream(stream):
        for repeat in range(10):
            # Change contents while preserving all captured pointer addresses.
            x.copy_(inputs[:rows] * (1 if repeat % 2 else -1))
            graph.replay()
            check()
    stream.synchronize()
    print(
        f"F6_NATIVE_STRIDE seed={request.node.callspec.params['native_case']} "
        f"M={rows} K=8192 N=2048 splits={splits} reduction={reduction} checks={len(measurements)} "
        f"max_mae={max(m[0] for m in measurements):.9g} "
        f"max_abs={max(m[1] for m in measurements):.9g}"
    )
