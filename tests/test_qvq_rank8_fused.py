# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent output-transform composition, including range and rounding edges."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import _qvq_hadamard_fused


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
