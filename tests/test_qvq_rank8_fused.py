# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent output-transform composition, including range and rounding edges."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import _qvq_hadamard_fused


def test_rank8_epilogue_warp_policy_is_explicit_and_validated():
    from gptqmodel.utils.qvq_rank8_triton import _rank8_num_warps

    assert _rank8_num_warps(2048, None) == 4
    assert _rank8_num_warps(8192, 0) == 8
    assert _rank8_num_warps(8192, 4) == 4
    assert _rank8_num_warps(8192, 8) == 8
    with pytest.raises(ValueError, match="num_warps"):
        _rank8_num_warps(8192, 3)


def test_composite_hadamard_width_policy_matches_factored_bases():
    from gptqmodel.quantization.qvq_rank8 import _supported_composite_hadamard_width

    assert _supported_composite_hadamard_width(5120)
    assert _supported_composite_hadamard_width(12288)
    assert not _supported_composite_hadamard_width(17408)


def test_rank8_triton_rejects_cold_compile_during_graph_capture(monkeypatch):
    from gptqmodel.utils import qvq_rank8_triton

    monkeypatch.setattr(qvq_rank8_triton.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        qvq_rank8_triton.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    key = (
        "cuda", 0, "output_epilogue", 1, 256, False, True, True,
        "torch.float32", "torch.float32", "torch.float32", "butterfly",
    )
    with pytest.raises(RuntimeError, match="warmed before CUDA Graph capture"):
        qvq_rank8_triton._require_rank8_kernel_warm(key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [256, 2048, 8192])
@pytest.mark.parametrize("hadamard", [False, True])
def test_window_only_shared_epilogue_never_accesses_factors(n, hadamard):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_output_epilogue

    class Poison:
        def __getattribute__(self, name):
            raise AssertionError("disabled rank8 factors accessed")

    torch.manual_seed(153)
    base = torch.randn(33, n, device="cuda")
    sv = torch.randn(n, device="cuda")
    bias = torch.randn(n, device="cuda")
    expected = (
        _qvq_hadamard_fused(base, post_scale=sv, bias=bias, scale_mode=3 if n >= 2048 else 4)
        if hadamard else base * sv + bias
    ).half()
    actual = rank8_output_epilogue(
        Poison(), Poison(), base, sv, bias, hadamard=hadamard,
        output_dtype=torch.float16, rank8_enabled=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rank8_output_epilogue(
            Poison(), Poison(), base, sv, bias, hadamard=hadamard,
            output_dtype=torch.float16, rank8_enabled=False,
        )
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [2048, 8192])
@pytest.mark.parametrize("hadamard", [False, True])
def test_rank8_direct_half_store_matches_final_conversion(n, hadamard):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_output_epilogue

    torch.manual_seed(151)
    hidden = torch.randn(33, 8, device="cuda").half()
    b = torch.randn(8, n, device="cuda").half() * 0.02
    base = torch.randn(33, n, device="cuda") * 100000
    sv = torch.randn(n, device="cuda")
    bias = torch.randn(n, device="cuda")
    reference = rank8_output_epilogue(hidden, b, base, sv, bias, hadamard=hadamard).half()
    actual = rank8_output_epilogue(
        hidden, b, base, sv, bias, hadamard=hadamard, output_dtype=torch.float16
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rank8_output_epilogue(
            hidden, b, base, sv, bias, hadamard=hadamard, output_dtype=torch.float16
        )
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k", [17, 256, 2048, 8192])
@pytest.mark.parametrize("m", [1, 33, 512])
def test_rank8_tensor_core_projection_contract(k, m):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_tensor_core_projection

    torch.manual_seed(149)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(m, k, device="cuda").half()
    a = (torch.randn(k, 8, device="cuda") * 0.02).half()
    reference = (x.float() @ a.float()).half()
    actual = rank8_tensor_core_projection(x, a)
    error = (actual.float() - reference.float()).abs()
    assert torch.isfinite(actual).all()
    assert error.mean() <= 2e-3 and error.max() <= 0.046875
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rank8_tensor_core_projection(x, a)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [16, 256, 2048, 8192, 16384])
@pytest.mark.parametrize("hadamard", [False, True])
@pytest.mark.parametrize("m", [1, 17])
def test_rank8_fused_output_contract(n, hadamard, m):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_output_epilogue

    torch.manual_seed(138)
    torch.backends.cuda.matmul.allow_tf32 = False
    hidden = torch.randn(m, 8, device="cuda").half()
    b = torch.randn(8, n, device="cuda").half() * 0.02
    sv = torch.randn(n, device="cuda") * 0.1
    bias = torch.randn(n, device="cuda") * 0.01
    for scale in (1.0, 100000.0):
        # FP32 inner storage must rescue each overflowing half boundary.
        base = torch.randn(m, n, device="cuda") * scale
        added = base + hidden.float() @ b.float()
        if hadamard:
            reference = _qvq_hadamard_fused(
                added, post_scale=sv, bias=bias, scale_mode=3 if n >= 2048 else 4
            )
        else:
            reference = added * sv + bias
        actual = rank8_output_epilogue(hidden, b, base, sv, bias, hadamard=hadamard)
        delta = (actual - reference).abs()
        assert torch.isfinite(actual).all()
        assert delta.mean() <= 2e-3
        assert delta.max() <= 0.046875
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = rank8_output_epilogue(
                hidden, b, base, sv, bias, hadamard=hadamard
            )
        for _ in range(10):
            graph.replay()
            assert torch.equal(captured, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rank8_fused_strided_group_output_and_no_bias():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_output_epilogue

    torch.manual_seed(139)
    hidden = torch.randn(3, 8, device="cuda").half()
    b = torch.randn(8, 256, device="cuda").half() * 0.01
    base = torch.randn(3, 512, device="cuda")[:, 256:]
    sv = torch.ones(256, device="cuda")
    reference = _qvq_hadamard_fused(
        base + hidden.float() @ b.float(), post_scale=sv, scale_mode=4
    )
    actual = rank8_output_epilogue(hidden, b, base, sv)
    error = (actual - reference).abs()
    assert error.mean() <= 2e-3 and error.max() <= 0.046875
    with pytest.raises(ValueError, match="matrix"):
        rank8_output_epilogue(hidden.flatten(), b, base, sv)
    with pytest.raises(ValueError, match="FP32 base"):
        rank8_output_epilogue(hidden, b, base.half(), sv)
    with pytest.raises(ValueError, match="shape mismatch"):
        rank8_output_epilogue(hidden, b[:, :128], base, sv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rank8_fused_epilogue_supports_composite_folded_output_width():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_output_epilogue

    torch.manual_seed(158)
    torch.backends.cuda.matmul.allow_tf32 = False
    m, n = 17, 5120
    hidden = torch.randn(m, 8, device="cuda", dtype=torch.float16)
    b = torch.randn(8, n, device="cuda", dtype=torch.float16) * 0.02
    base = torch.randn(m, n, device="cuda", dtype=torch.float32)
    sv = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    expected = (base + hidden.float() @ b.float()) * sv + bias
    actual = rank8_output_epilogue(
        hidden, b, base, sv, bias, hadamard=False, output_dtype=torch.float16
    )
    error = (actual.float() - expected).abs()
    assert error.mean() <= 2e-3 and error.max() <= 0.046875
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rank8_output_epilogue(
            hidden, b, base, sv, bias, hadamard=False, output_dtype=torch.float16
        )
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rank8_fused_epilogue_supports_composite_hadamard_output_graph_replay():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_output_epilogue

    torch.manual_seed(159)
    torch.backends.cuda.matmul.allow_tf32 = False
    m, n = 17, 5120
    hidden = torch.randn(m, 8, device="cuda", dtype=torch.float16)
    b = torch.randn(8, n, device="cuda", dtype=torch.float16) * 0.02
    base = torch.randn(m, n, device="cuda", dtype=torch.float32)
    sv = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    added = base + hidden.float() @ b.float()
    expected = _qvq_hadamard_fused(
        added, post_scale=sv, bias=bias, scale_mode=3
    ).half()
    actual = rank8_output_epilogue(
        hidden, b, base, sv, bias, hadamard=True, output_dtype=torch.float16
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rank8_output_epilogue(
            hidden, b, base, sv, bias, hadamard=True, output_dtype=torch.float16
        )
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k,n,hadamard", [(256, 256, False), (2048, 2048, True)])
def test_rank8_fused_projection_output_matches_two_stage_and_graph(k, n, hadamard):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import (
        rank8_output_epilogue,
        rank8_project_output_epilogue,
    )

    torch.manual_seed(159)
    torch.backends.cuda.matmul.allow_tf32 = False
    m = 17
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    a = torch.randn(k, 8, device="cuda", dtype=torch.float16) * 0.02
    b = torch.randn(8, n, device="cuda", dtype=torch.float16) * 0.02
    base = torch.randn(m, n, device="cuda", dtype=torch.float32)
    sv = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    hidden = (x.float() @ a.float()).half()
    expected = rank8_output_epilogue(
        hidden, b, base, sv, bias, hadamard=hadamard, output_dtype=torch.float32
    )
    actual = rank8_project_output_epilogue(
        x, a, b, base, sv, bias, hadamard=hadamard, output_dtype=torch.float32
    )
    error = (actual - expected).abs()
    assert torch.isfinite(actual).all()
    assert error.mean() <= 2e-3 and error.max() <= 0.046875
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rank8_project_output_epilogue(
            x, a, b, base, sv, bias, hadamard=hadamard, output_dtype=torch.float32
        )
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k", [16, 256, 2048, 8192, 16384])
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("hadamard", [False, True])
def test_rank8_shared_input_producer(k, groups, hadamard):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_input_producer

    torch.manual_seed(140)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(3, k, device="cuda").half() * 0.25
    su = torch.randn(k, device="cuda").half()
    factors = tuple(
        torch.randn(k, 8, device="cuda").half() * 0.05 for _ in range(groups)
    )
    if hadamard:
        reference = _qvq_hadamard_fused(
            x, pre_scale=su, scale_mode=2 if k >= 2048 else 1
        )
    else:
        reference = x * su
    actual, hidden = rank8_input_producer(x, su, factors, hadamard=hadamard)
    assert torch.equal(actual, reference)
    for a, u in zip(factors, hidden):
        expected = (reference.float() @ a.float()).half()
        error = (u.float() - expected.float()).abs()
        assert error.mean() <= 2e-3 and error.max() <= 0.046875
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_x, captured_u = rank8_input_producer(x, su, factors, hadamard=hadamard)
    for _ in range(10):
        graph.replay()
        assert torch.equal(captured_x, actual)
        assert all(torch.equal(a, b) for a, b in zip(captured_u, hidden))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rank8_input_producer_overflow_rescue_and_stream():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_input_producer

    torch.backends.cuda.matmul.allow_tf32 = False
    # A single overflowing SU product becomes finite after normalization.
    # Strided rows also check the producer's source leading dimension.
    x = torch.zeros(3, 4096, device="cuda", dtype=torch.float16)[:, :2048]
    x[:, 0] = 200
    x[:, 1] = -64
    su = torch.full((2048,), 512, device="cuda", dtype=torch.float16)
    a = torch.full((2048, 8), 1e-5, device="cuda", dtype=torch.float16)
    a[::2, ::2] *= -1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # QVQLinear supplies contiguous input to the deployed CUDA transform.
        # The helper's noncontiguous Python fallback has a different overflow
        # boundary and is not the deployed mode2 reference.
        reference = _qvq_hadamard_fused(x.contiguous(), pre_scale=su, scale_mode=2)
        expected = (reference.float() @ a.float()).half()
        actual, (hidden,) = rank8_input_producer(x, su, a)
    stream.synchronize()
    assert torch.isfinite(actual).all() and torch.isfinite(hidden).all()
    assert torch.equal(actual, reference)
    error = (hidden.float() - expected.float()).abs()
    assert error.mean() <= 2e-3 and error.max() <= 0.046875


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rank8_input_producer_supports_composite_folded_width_graph_replay():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_input_producer

    torch.manual_seed(157)
    torch.backends.cuda.matmul.allow_tf32 = False
    k = 5120
    x = torch.randn(17, k, device="cuda", dtype=torch.float16) * 0.1
    su = torch.randn(k, device="cuda", dtype=torch.float16)
    factors = tuple(
        torch.randn(k, 8, device="cuda", dtype=torch.float16) * 0.02
        for _ in range(2)
    )
    reference = x * su
    actual, hidden = rank8_input_producer(x, su, factors, hadamard=False)
    assert torch.equal(actual, reference)
    for factor, projected in zip(factors, hidden):
        expected = (reference.float() @ factor.float()).half()
        error = (projected.float() - expected.float()).abs()
        assert error.mean() <= 2e-3 and error.max() <= 0.046875
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_x, captured_hidden = rank8_input_producer(
            x, su, factors, hadamard=False
        )
    for _ in range(3):
        graph.replay()
        assert torch.equal(captured_x, actual)
        assert all(torch.equal(a, b) for a, b in zip(captured_hidden, hidden))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rank8_input_producer_supports_composite_hadamard_width_graph_replay():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from gptqmodel.utils.qvq_rank8_triton import rank8_input_producer

    torch.manual_seed(158)
    torch.backends.cuda.matmul.allow_tf32 = False
    k = 5120
    x = torch.randn(17, k, device="cuda", dtype=torch.float16) * 0.1
    su = torch.randn(k, device="cuda", dtype=torch.float16)
    factors = tuple(
        torch.randn(k, 8, device="cuda", dtype=torch.float16) * 0.02
        for _ in range(2)
    )
    reference = _qvq_hadamard_fused(x, pre_scale=su, scale_mode=2)
    actual, hidden = rank8_input_producer(x, su, factors, hadamard=True)
    assert torch.equal(actual, reference)
    for factor, projected in zip(factors, hidden):
        expected = (reference.float() @ factor.float()).half()
        error = (projected.float() - expected.float()).abs()
        assert error.mean() <= 2e-3 and error.max() <= 0.046875
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_x, captured_hidden = rank8_input_producer(
            x, su, factors, hadamard=True
        )
    for _ in range(3):
        graph.replay()
        assert torch.equal(captured_x, actual)
        assert all(torch.equal(a, b) for a, b in zip(captured_hidden, hidden))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_input_fused_policy_admits_composite_width_without_input_hadamard():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from test_qvq_grouped_runtime import _child
    from test_qvq_window_recovery import _kernel_rank8

    from gptqmodel.quantization.qvq_rank8 import (
        P32WindowConfig,
        prepare_rank8,
        window_kernel_candidates,
    )

    layer = _child(
        "down_proj",
        in_features=5120,
        out_features=5120,
        device="cuda",
        input_hadamard=False,
    )
    _kernel_rank8(layer)
    prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
    candidates = window_kernel_candidates(layer, m=17)
    assert any(candidate.recovery_projection == "input_fused" for candidate in candidates)
    prepare_rank8(
        layer,
        next(candidate for candidate in candidates if candidate.recovery_projection == "input_fused"),
    )
    assert layer._p32_rank8_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_composite_hadamard_policy_admits_fused_epilogue_and_rejects_project_output():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from test_qvq_grouped_runtime import _child
    from test_qvq_window_recovery import _kernel_rank8

    from gptqmodel.quantization.qvq_rank8 import (
        P32WindowConfig,
        prepare_rank8,
        window_kernel_candidates,
    )

    layer = _child(
        "q_proj",
        in_features=5120,
        out_features=5120,
        device="cuda",
        input_hadamard=False,
        output_hadamard=True,
    )
    _kernel_rank8(layer)
    prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
    candidates = window_kernel_candidates(layer, m=17)
    fused = [
        candidate
        for candidate in candidates
        if candidate.recovery_kernel == "fused_epilogue"
        and candidate.recovery_projection == "separate_reference"
    ]
    assert fused
    assert not any(
        candidate.recovery_projection == "project_output_fused"
        for candidate in candidates
    )
    prepare_rank8(layer, fused[0])
    assert layer._p32_rank8_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fully_fused_policy_routes_project_output_epilogue_and_graph():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from test_qvq_grouped_runtime import _child
    from test_qvq_window_recovery import _kernel_rank8

    from gptqmodel.quantization.qvq_rank8 import P32WindowConfig, prepare_rank8

    torch.backends.cuda.matmul.allow_tf32 = False
    layer = _child(
        "q_proj",
        in_features=256,
        out_features=256,
        su=torch.ones(256, device="cuda"),
        output_hadamard=False,
        device="cuda",
    ).eval()
    _kernel_rank8(layer)
    config = P32WindowConfig(
        recovery_mode="on",
        recovery_kernel="fully_fused",
        recovery_projection="project_output_fused",
        arithmetic_signature="unverified_project_output_fused",
    )
    prepare_rank8(layer, config)
    x = torch.randn(17, 256, device="cuda", dtype=torch.float16) * 0.01
    eager = layer(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(x)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)
