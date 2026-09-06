# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent output-transform composition, including range and rounding edges."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import _qvq_hadamard_fused


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
