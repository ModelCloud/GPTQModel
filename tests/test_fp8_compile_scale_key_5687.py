# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""TorchFP8Linear must stay fullgraph-compilable after the payload-key cache (#5687).

``_decoded_weight_bound_ready`` used to compare a key that carries
``scale.data_ptr()``. On CUDA that comparison runs inside the traced forward
(``_native_gemm_eligible`` -> ``_weights_fit_activation_dtype``), and dynamo
rejects it as an "Unsupported: Data pointer comparison" graph break. The eager
warm-up already publishes the bound for the current payload, so a trace or
capture must read it without re-keying.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from tests.fp8_test_utils import pack_fp8_module


def _module(method, block=None, *, warm=True):
    torch.manual_seed(7)
    linear = torch.nn.Linear(256, 256, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.02)
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=256,
        out_features=256,
        bias=False,
        register_buffers=False,
        format="float8_e4m3fn",
        weight_scale_method=method,
        weight_block_size=block,
    )
    pack_fp8_module(module, linear)
    module = module.to("cuda")
    if warm:
        module.post_init()
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pointer key is only read on CUDA")
@pytest.mark.parametrize(("method", "block"), [("row", None), ("block", (128, 128))])
def test_fullgraph_compile_matches_eager(method, block):
    module = _module(method, block)
    x = torch.randn(2, 256, dtype=torch.float16, device="cuda")
    eager = module._forward_dequant_matmul(x)

    compiled = torch.compile(module._forward_dequant_matmul, fullgraph=True, backend="eager")
    actual = compiled(x)

    assert torch.isfinite(actual).all()
    # A block payload uses the Triton decode in eager but cannot trace it, so
    # the compiled graph takes the dequant path: same value, different
    # reduction order. Compare at the activation dtype's rounding limit.
    torch.testing.assert_close(actual, eager, rtol=2e-3, atol=2e-3)

    # Eager payload tracking from #5592 still works after the trace-only short-cut.
    assert module._weights_fit_activation_dtype(torch.float16)
    module.weight_scale_inv.copy_(torch.full_like(module.weight_scale_inv, 1e-3))
    assert not module._weights_fit_activation_dtype(torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pointer key is only read on CUDA")
def test_fullgraph_compile_with_a_cold_bound_cache():
    """A module compiled before ``post_init`` keeps the wide path, not a crash."""

    module = _module("block", (128, 128), warm=False)
    # ``.to("cuda")`` re-warms via ``_apply``; drop the cache again so the
    # compiled graph really starts from a cold payload (#5195's wide path).
    module._invalidate_compiled()
    assert not module._decoded_weight_bound_ready
    x = torch.randn(2, 256, dtype=torch.float16, device="cuda")

    compiled = torch.compile(module._forward_dequant_matmul, fullgraph=True, backend="eager")
    actual = compiled(x)

    assert torch.isfinite(actual).all()
